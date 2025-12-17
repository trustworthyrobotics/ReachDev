# trainers/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import sys
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DTPlanReach
from CROWN_Reach.src.utils.box_set import calculate_volume

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from training.losses_metrics import combined_loss, make_linear_step_weights

from utils.logging import Logger

def _l1_regularizer(params, lam: float):
    if lam <= 0:
        return 0.0
    leaves = jax.tree.leaves(eqx.filter(params, eqx.is_inexact_array))
    return lam * sum(jnp.sum(jnp.abs(p)) for p in leaves)

class Trainer:
    """
    Owns: model, optimizer/scheduler, train/eval steps, loop, checkpoints.
    Optional: logs via `logger` (WandB or print), without changing core logic.
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
        logger: Optional[Logger] = None,
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

        self.logger = logger
        self._wandb_cfg = self.cfg["wandb"]
        self._log_every = int(self._wandb_cfg["log_every"])
        assert self._log_every > 0
        self._save_ckpts_to_wandb = bool(self._wandb_cfg["save_checkpoints"])
        self._wandb_enabled = bool(self._wandb_cfg["enabled"]) and (self.logger is not None)

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

        # reachability analyzer
        def f_wrapper(x):
            state_next = self.model(x)
            action_next = x[self.model.Dx:]
            return jnp.concatenate([state_next, action_next], axis=-1)

        self.reach_analyzer = DTPlanReach(f_wrapper, state_dim=self.model.Dx, action_dim=self.model.Du, nn_dyn=True, n_steps_per_plan=1, step_size=1)

        self.batch_size = self.cfg["batch_size"]
        reach_cfg = self.cfg.get("reach", {})
        self.reach_mode = reach_cfg.get("mode", "none")
        assert self.reach_mode in ["none", "mid", "after"]
        self.reach_every = int(reach_cfg.get("every", 1))
        self.reach_after = float(reach_cfg.get("after", 0.5))
        self.reach_eps = float(reach_cfg.get("eps", 0.0))
        self.reach_weight = float(reach_cfg.get("weight", 0.0))

        self._build_steps()

        self.best_val = np.inf
        self.global_step = 0

    def _build_steps(self):
        lam_l1 = float(self.cfg["lam_l1_reg"])
        T = self.T_train
        w = self.step_weights
        optim = self.optim

        @eqx.filter_jit
        def train_step(model, opt_state, X, U, key):
            def loss_fn(m):
                loss, metrics = combined_loss(m, X, U, T=T, step_weights=w, aux_weight=0.0)
                loss = loss + _l1_regularizer(m, lam_l1)
                return loss, metrics

            (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, metrics

        @eqx.filter_jit
        def reach_train_step(model, opt_state, X, U, key):
            def loss_fn(m):
                loss, metrics = combined_loss(m, X, U, T=T, step_weights=w, aux_weight=0.0)
                loss = loss + _l1_regularizer(m, lam_l1)
                X_init = jnp.concatenate([X[:, 0, :], jnp.zeros_like(U[:, 0, :])], axis=-1)
                # _, reach_lo, reach_up, _, _ = self.reach_analyzer.verify(X_init-self.reach_eps, X_init+self.reach_eps, n_total_steps=self.T_train, action_seq=U[:, None])
                # reach_vol = (reach_up - reach_lo).sum()

                def f_wrapper(x):
                    state_next = m(x)
                    action_next = x[m.Dx:]
                    return jnp.concatenate([state_next, action_next], axis=-1)
                _, r_lo, r_up, _, _ = self.reach_analyzer.verify_w_model(f_wrapper, X_init-self.reach_eps, X_init+self.reach_eps, n_total_steps=self.T_train, action_seq=U[:, None])
                # reach_vol = (r_up - r_lo).sum()

                reach_vol = calculate_volume(r_lo.reshape(-1, self.T_train + 1, self.model.Dx+self.model.Du), r_up.reshape(-1, self.T_train + 1, self.model.Dx+self.model.Du))
                reach_penalty = jnp.log(1 + reach_vol / self.batch_size) * self.reach_weight
                metrics['reach_volume'] = reach_vol
                metrics['reach_penalty'] = reach_penalty
                loss = loss + reach_penalty
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
        self._reach_train_step = reach_train_step
        self._eval_step = eval_step

    def _current_lr(self) -> float:
        try:
            v = self.lr_schedule(self.global_step)
            return float(np.asarray(v))
        except Exception:
            return float(self.cfg["lr"])

    # -------------- public loop --------------

    def run(self):
        # immediately enable reachability only for "mid" mode
        reach_enabled = self.reach_mode == "mid"
        for epoch in range(1, self.cfg["n_epoch"] + 1):
            if self.reach_mode == "after" and epoch >= int(self.reach_after * self.cfg["n_epoch"]):
                reach_enabled = True
            # ---- train ----
            train_losses = []
            latest_reach_vol = 0
            latest_reach_penalty = 0
            for it, batch in enumerate(self.train_loader):
                X = batch["observations"]
                U = batch["actions"]
                W = batch["weights"] # unused currently
                self.key, subk = jax.random.split(self.key)
                if reach_enabled and self.global_step % self.reach_every == 0:
                    self.model, self.opt_state, loss, metrics = self._reach_train_step(self.model, self.opt_state, X, U, subk)
                    latest_reach_vol = metrics["reach_volume"]
                    latest_reach_penalty = metrics["reach_penalty"]
                else:
                    self.model, self.opt_state, loss, metrics = self._train_step(self.model, self.opt_state, X, U, subk)
                train_losses.append(float(loss))
                self.global_step += 1

                if (self.global_step % self._log_every == 0):
                    self.logger.log(
                        {"train/iter_loss": float(loss), "train/reach_volume": float(latest_reach_vol), "train/reach_penalty": float(latest_reach_penalty),
                         "train/mse": float(metrics["mse"]),
                         "lr": self._current_lr(), "epoch": epoch, "global_step": self.global_step},
                        step=self.global_step
                    )

            tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            # ---- validate ----
            val_losses = []
            for val_batch in self.val_loader:
                Xv = val_batch["observations"]
                Uv = val_batch["actions"]
                Wv = val_batch["weights"] # unused currently
                vloss, _ = self._eval_step(self.model, Xv, Uv, self.T_valid)
                val_losses.append(float(vloss))
            va_loss = float(np.mean(val_losses)) if val_losses else float("nan")

            self.logger.log(
                {"train/loss": tr_loss, "val/loss": va_loss,
                 "train/reach_volume": float(latest_reach_vol), "train/reach_penalty": float(latest_reach_penalty),
                 "train/mse": float(metrics["mse"]),
                 "val/mse": float(metrics["mse"]),
                    "lr": self._current_lr(), "epoch": epoch,
                    "global_step": self.global_step},
                step=self.global_step
            )

            # ---- ckpt ----
            if va_loss < self.best_val:
                self.best_val = va_loss
                path_base = f"{self.out_dir}/best_model"
                self.save_fn(path_base, self.model, self.opt_state, self.global_step, self.cfg_full, self.stats)

                if self._wandb_enabled and self._save_ckpts_to_wandb:
                    self.logger.save(path_base + ".eqx")
                    self.logger.save(path_base + ".npz")

            if epoch == self.cfg["n_epoch"]:
                path_base = f"{self.out_dir}/last_model"
                self.save_fn(path_base, self.model, self.opt_state, self.global_step, self.cfg_full, self.stats)
                if self._wandb_enabled and self._save_ckpts_to_wandb:
                    self.logger.save(path_base + ".eqx")
                    self.logger.save(path_base + ".npz")

        if self.logger is not None:
            self.logger.finish()
