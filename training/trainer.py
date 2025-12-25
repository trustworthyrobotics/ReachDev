# trainers/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import sys
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DTPlanReach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time
from training.losses_metrics import combined_loss, make_linear_step_weights

from utils.logging import Logger

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
        self.total_steps = max(1, steps_per_epoch * self.cfg["n_epoch"])
        if self.cfg["lr_scheduler"]["enabled"] and self.cfg["lr_scheduler"]["type"] == "CosineAnnealingLR": 
            self.lr_schedule = optax.cosine_decay_schedule(init_value=self.cfg["lr"], decay_steps=self.total_steps, alpha=0.0)
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
        self.T_min = 1
        if self.cfg.get("inc_horizon", False):
            self.T_schedule = lambda epoch: min(
                self.T_train, 
                self.T_min + round((self.T_train - self.T_min) * epoch / self.cfg["n_epoch"])
            )
        else:
            self.T_schedule = lambda epoch: self.T_train
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
        self.reach_eps_min = float(reach_cfg.get("eps_min", 0.0))
        self.reach_eps_max = float(reach_cfg.get("eps_max", 0.01))
        self.reach_eps_schedule = lambda step: jnp.minimum(
            self.reach_eps_max,
            self.reach_eps_min + (self.reach_eps_max - self.reach_eps_min) * (step / self.total_steps)
        )
        self.reach_weight = float(reach_cfg.get("weight", 0.0))
        self.reach_splits = reach_cfg.get("splits", {})
        self.reach_batch_size = int(reach_cfg.get("batch_size", self.batch_size))

        self._build_steps()

        self.best_val = np.inf
        self.global_step = 0
        self.reach_eps = self.reach_eps_schedule(self.global_step)

    def _build_steps(self):
        lam_jac = jnp.array(self.cfg["lam_jac_reg"])
        aux_weight = jnp.array(0.0)
        w = self.step_weights
        optim = self.optim

        @eqx.filter_jit
        def train_step(model, opt_state, X, U, key):
            def loss_fn(m):
                loss, metrics = combined_loss(m, X, U, step_weights=w, aux_weight=aux_weight, lam_jac=lam_jac)
                return loss, metrics

            (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, metrics

        reach_bs = self.reach_batch_size
        reach_splits = self.reach_splits
        reach_weight = self.reach_weight
        verify_w_model = self.reach_analyzer.verify_w_model
        D = self.model.Dx + self.model.Du

        def reach_penalty_fn(m, reach_X, reach_U, key, curr_reach_eps):
            T_reach = reach_U.shape[1]
            # pick a random subset for reachability
            if reach_X.shape[0] > reach_bs:
                perm = jax.random.permutation(key, reach_X.shape[0])
                idxs = perm[:reach_bs]
                reach_X = reach_X[idxs]
                reach_U = reach_U[idxs]
            state_init = reach_X[:, 0, :]
            state_lo = state_init - curr_reach_eps
            state_up = state_init + curr_reach_eps

            def f_wrapper(x):
                state_next = m(x)
                action_next = x[m.Dx:]
                return jnp.concatenate([state_next, action_next], axis=-1)
            X_lo = jnp.concatenate([state_lo, jnp.zeros_like(reach_U[:, 0, :])], axis=-1)
            X_up = jnp.concatenate([state_up, jnp.zeros_like(reach_U[:, 0, :])], axis=-1)
            X_lo, X_up = prepare_initial_set_v2(X_lo, X_up, splits_cfg=reach_splits)
            _, r_lo, r_up, _ = verify_w_model(f_wrapper, X_lo, X_up, n_total_steps=T_reach, action_seq=reach_U.repeat(X_up.shape[0]//reach_U.shape[0], axis=0)[:, None])

            reach_vol = calculate_volume(r_lo.reshape(-1, T_reach + 1, D), r_up.reshape(-1, T_reach + 1, D), union_init=False, mode='sum') / r_lo.shape[0]
            reach_penalty = jnp.log(1 + reach_vol) * reach_weight
            return reach_vol, reach_penalty

        @eqx.filter_jit
        def reach_train_step(model, opt_state, X, U, key, curr_reach_eps):
            def loss_fn(m):
                loss, metrics = combined_loss(m, X, U, step_weights=w, aux_weight=aux_weight, lam_jac=lam_jac)
                reach_vol, reach_penalty = reach_penalty_fn(m, X, U, key=key, curr_reach_eps=curr_reach_eps)
                metrics['reach_volume'] = reach_vol
                metrics['reach_penalty'] = reach_penalty
                loss = loss + reach_penalty
                return loss, metrics

            (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, metrics

        @eqx.filter_jit
        def eval_step(model, X, U, key):
            loss, metrics = combined_loss(model, X, U, step_weights=None, aux_weight=0.0)
            return loss, metrics

        @eqx.filter_jit
        def reach_eval_step(model, X, U, key, curr_reach_eps):
            loss, metrics = combined_loss(model, X, U, step_weights=None, aux_weight=0.0)
            reach_vol, reach_penalty = reach_penalty_fn(model, X, U, key=key, curr_reach_eps=curr_reach_eps)
            metrics['reach_volume'] = reach_vol
            metrics['reach_penalty'] = reach_penalty
            return loss, metrics

        self._train_step = train_step
        self._reach_train_step = reach_train_step
        self._eval_step = eval_step
        self._reach_eval_step = reach_eval_step

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
        latest_reach_vol = 0
        latest_reach_penalty = 0
        curr_reach_eps = 0.0
        for epoch in range(1, self.cfg["n_epoch"] + 1):
            curr_T = self.T_schedule(epoch)
            if self.reach_mode == "after" and epoch >= int(self.reach_after * self.cfg["n_epoch"]):
                reach_enabled = True
            # ---- train ----
            train_losses = []
            for it, batch in enumerate(self.train_loader):
                X = batch["observations"]
                U = batch["actions"]
                W = batch["weights"] # unused currently
                X = X[:, :curr_T + 1, :]
                U = U[:, :curr_T, :]
                
                self.key, subk = jax.random.split(self.key)
                if reach_enabled and self.global_step % self.reach_every == 0:
                    # start_time = time.time()
                    curr_reach_eps = self.reach_eps_schedule(self.global_step)
                    self.model, self.opt_state, loss, metrics = self._reach_train_step(self.model, self.opt_state, X, U, subk, curr_reach_eps)
                    # jax.block_until_ready(loss)
                    # end_time = time.time()
                    # print(f"_reach_train_step time: {end_time - start_time} seconds")
                    latest_reach_vol = metrics["reach_volume"]
                    latest_reach_penalty = metrics["reach_penalty"]
                else:
                    # start_time = time.time()
                    self.model, self.opt_state, loss, metrics = self._train_step(self.model, self.opt_state, X, U, subk)
                    # jax.block_until_ready(loss)
                    # end_time = time.time()
                    # print(f"_train_step time: {end_time - start_time} seconds")
                train_losses.append(float(loss))
                self.global_step += 1
                if (self.global_step % self._log_every == 0):
                    self.logger.log(
                        {"train/iter_loss": float(loss), "train/reach_volume": float(latest_reach_vol), "train/reach_penalty": float(latest_reach_penalty),
                         "train/mse": float(metrics["mse"]), "train/jac_reg_loss": float(metrics.get("jacobian_reg_loss", 0.0)),
                         "lr": self._current_lr(), "epoch": epoch, "global_step": self.global_step,
                         "reach_eps": float(curr_reach_eps)},
                        step=self.global_step
                    )

            tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            # ---- validate ----
            val_losses = []
            for val_batch in self.val_loader:
                Xv = val_batch["observations"]
                Uv = val_batch["actions"]
                Wv = val_batch["weights"] # unused currently
                Xv = Xv[:, :self.T_valid + 1, :]
                Uv = Uv[:, :self.T_valid, :]
                self.key, subk = jax.random.split(self.key)
                vloss, vmetrics = self._eval_step(self.model, Xv, Uv, subk)
                # if reach_enabled:
                #     vloss, vmetrics = self._reach_eval_step(self.model, Xv, Uv, subk, curr_reach_eps)
                # else:
                #     vloss, vmetrics = self._eval_step(self.model, Xv, Uv, subk)
                val_losses.append(float(vloss))
            va_loss = float(np.mean(val_losses)) if val_losses else float("nan")

            self.logger.log(
                {"train/loss": tr_loss, "val/loss": va_loss,
                 "train/reach_volume": float(latest_reach_vol), "train/reach_penalty": float(latest_reach_penalty),
                 "train/mse": float(metrics["mse"]),
                 "val/reach_volume": float(vmetrics.get("reach_volume", 0.0)), "val/reach_penalty": float(vmetrics.get("reach_penalty", 0.0)),
                 "val/mse": float(vmetrics["mse"]),
                    "lr": self._current_lr(), "epoch": epoch, "T": curr_T,
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
