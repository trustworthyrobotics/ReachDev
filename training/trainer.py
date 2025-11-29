# trainers/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from training.losses_metrics import combined_loss, make_linear_step_weights


def _noise_aug(X: jnp.ndarray, U: jnp.ndarray, std: float, key: jax.Array):
    if std <= 0:
        return X, U
    k1, k2 = jax.random.split(key, 2)
    return (X + std * jax.random.normal(k1, X.shape, X.dtype),
            U + std * jax.random.normal(k2, U.shape, U.dtype))


def _l1_regularizer(params, lam: float):
    if lam <= 0:
        return 0.0
    leaves = jax.tree_leaves(eqx.filter(params, eqx.is_inexact_array))
    return lam * sum(jnp.sum(jnp.abs(p)) for p in leaves)


class Trainer:
    """
    Owns: model, optimizer/scheduler, train/eval steps, loop, checkpoints.

    Externally you provide:
      - model (Equinox Module)
      - loaders: objects with .epoch() yielding (X,U,_) batches
      - stats (for saving), cfg dicts, and an out_dir + save_fn
    """

    def __init__(
        self,
        *,
        model,
        train_loader,
        val_loader,
        save_fn,                # callable(path_base, model, opt_state, step, cfg, stats)
        cfg_full: Dict,
        stats: Optional[Dict] = None,
        seed: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg_full["train"]
        self.out_dir = self.cfg["out_dir"]
        self.save_fn = save_fn
        self.cfg_full = cfg_full
        self.stats = stats or {}
        self.key = jax.random.PRNGKey(seed)

        # scheduler
        steps_per_epoch = len(self.train_loader)
        if self.cfg["lr_scheduler"]["enabled"] and self.cfg["lr_scheduler"]["type"] == "CosineAnnealingLR":
            total_steps = max(1, steps_per_epoch * self.cfg["n_epoch"])
            self.lr_schedule = optax.cosine_decay_schedule(init_value=self.cfg["lr"], decay_steps=total_steps, alpha=0.0)
        else:
            self.lr_schedule = optax.constant_schedule(self.cfg["lr"])
        # optimizer
        b1 = float(self.cfg["lr_params"]["adam_beta1"])
        self.optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=self.lr_schedule, b1=b1)
        )
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        # weights & jit’d steps
        self.T_train = int(self.cfg["n_rollout"])
        self.T_valid = int(self.cfg["n_rollout_valid"])
        self.step_weights = make_linear_step_weights(self.T_train, float(self.cfg["step_weight_ub"]))
        self.noise_std = float(self.cfg["noise"])

        self._build_steps()

        self.best_val = np.inf
        self.global_step = 0

    def _build_steps(self):
        lam_l1 = float(self.cfg["lam_l1_reg"])
        T = self.T_train
        w = self.step_weights
        noise_std = self.noise_std
        optim = self.optim

        @eqx.filter_jit
        def train_step(model, opt_state, X, U, key):
            X, U = _noise_aug(X, U, std=noise_std, key=key)

            def loss_fn(m):
                loss, metrics = combined_loss(m, X, U, T=T, step_weights=w, aux_weight=0.0)
                loss = loss + _l1_regularizer(m, lam_l1)
                return loss, metrics

            (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, metrics

        @eqx.filter_jit
        def eval_step(model, X, U, T_eval):
            loss, metrics = combined_loss(model, X, U, T=T_eval, step_weights=None, aux_weight=0.0)
            return loss, metrics

        self._train_step = train_step
        self._eval_step = eval_step

    # -------------- public loop --------------

    def run(self):
        for epoch in range(1, self.cfg["n_epoch"] + 1):
            # ---- train ----
            train_losses = []
            for X, U, _ in self.train_loader.epoch():
                self.key, subk = jax.random.split(self.key)
                self.model, self.opt_state, loss, _ = self._train_step(self.model, self.opt_state, X, U, subk)
                train_losses.append(float(loss))
                self.global_step += 1
            tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            # ---- validate ----
            val_losses = []
            for Xv, Uv, _ in self.val_loader.epoch():
                vloss, _ = self._eval_step(self.model, Xv, Uv, self.T_valid)
                val_losses.append(float(vloss))
            va_loss = float(np.mean(val_losses)) if val_losses else float("nan")

            print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")

            # ---- ckpt ----
            if va_loss < self.best_val:
                self.best_val = va_loss
                self.save_fn(f"{self.out_dir}/best_model", self.model, self.opt_state, self.global_step, self.cfg_full, self.stats)
            if epoch == self.cfg["n_epoch"]:
                self.save_fn(f"{self.out_dir}/last_model", self.model, self.opt_state, self.global_step, self.cfg_full, self.stats)
