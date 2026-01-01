# trainers/trainer.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import time
from training.losses_metrics import TotalLoss, make_linear_step_weights

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
        save_fn,                # callable(path_base, model, opt_state, step, cfg)
        cfg_full: Dict,
        seed: int = 0,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        train_mode = cfg_full.get("train_mode", "dt_dyn")
        assert train_mode in {"dt_dyn", "ct_dyn", "ct_ctl"}, f"Unknown train_mode: {train_mode}"
        self.cfg = cfg_full[f"train_{train_mode}"]
        self.out_dir = self.cfg["out_dir"]
        self.save_fn = save_fn
        self.cfg_full = cfg_full
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
        if self.cfg["lr_scheduler"]["type"] == "cos": 
            self.lr_schedule = optax.cosine_decay_schedule(init_value=self.cfg["lr_scheduler"]["lr_init"], decay_steps=self.total_steps, alpha=0.0)
        elif self.cfg["lr_scheduler"]["type"] == "linear":
            self.lr_schedule = optax.linear_schedule(init_value=self.cfg["lr_scheduler"]["lr_init"], end_value=self.cfg["lr_scheduler"].get("lr_final", 0.0), transition_steps=self.total_steps)
        elif self.cfg["lr_scheduler"]["type"] == "const":
            self.lr_schedule = optax.constant_schedule(self.cfg["lr_scheduler"]["lr_init"])
        # optimizer
        self.optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=self.lr_schedule)
        )
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))

        # weights & jit’d steps
        self.T_valid = int(self.cfg["n_rollout_valid"])
        self.T_init = int(self.cfg["horizon_scheduler"]["T_init"])
        self.T_final = int(self.cfg["horizon_scheduler"]["T_final"])
        self.n_epoch = int(self.cfg["n_epoch"])
        pred_mode = self.cfg.get("pred_mode", "state")
        assert pred_mode in ["state", "pose"]
        if self.cfg["horizon_scheduler"]["type"] == "linear":
            self.T_schedule = lambda epoch: self.T_init + round((self.T_final - self.T_init) * epoch / self.n_epoch)
        elif self.cfg["horizon_scheduler"]["type"] == "log":
            log_alpha = self.cfg["horizon_scheduler"].get("log_alpha", 1.0)
            self.T_schedule = lambda epoch: self.T_init + jnp.floor((self.T_final + 1 -1e-5 - self.T_init) * jnp.log1p(log_alpha * epoch / self.n_epoch) / jnp.log1p(log_alpha))
        elif self.cfg["horizon_scheduler"]["type"] == "const":
            self.T_schedule = lambda epoch: self.T_final
        self.step_weights = make_linear_step_weights(self.T_final, float(self.cfg["step_weight_ub"]))
        self.step_weights_eval = jnp.ones_like(self.step_weights)

        self.batch_size = self.cfg["batch_size"]
        reach_cfg = self.cfg.get("reach", {})
        self.reach_mode = reach_cfg.get("mode", "none")
        assert self.reach_mode in ["none", "mid", "after"]
        self.reach_every = int(reach_cfg.get("every", 1))
        self.reach_after = float(reach_cfg.get("after", 0.5))
        self.reach_eps_init = float(reach_cfg.get("eps_init", 0.0))
        self.reach_eps_final = float(reach_cfg.get("eps_final", 0.01))
        self.reach_eps_schedule = lambda step: self.reach_eps_init + (self.reach_eps_final - self.reach_eps_init) * jnp.minimum(1.0, step / self.total_steps)
        self.reach_weight = float(reach_cfg.get("weight", 0.0))
        self.reach_splits = reach_cfg.get("splits", {})
        self.reach_batch_size = int(reach_cfg.get("batch_size", self.batch_size))

        self.loss_fn = TotalLoss(
            mode=train_mode,
            state_dim=model.Dx,
            action_dim=model.Du,
            reach_cfg=reach_cfg,
            lam_jac=float(self.cfg.get("lam_jac_reg", 0.0)),
        )

        self._build_steps()

        self.best_val = np.inf
        self.global_step = 0

    def _build_steps(self):
        optim = self.optim

        @eqx.filter_jit
        def train_step(model, opt_state, X, U, key, enable_reach, reach_eps):
            def loss_fn(m):
                loss, metrics = self.loss_fn(m, X, U, enable_reach, key, self.step_weights, reach_eps, self.reach_batch_size)
                return loss, metrics

            (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, metrics

        @eqx.filter_jit
        def eval_step(model, X, U, key, enable_reach, reach_eps):
            loss, metrics = self.loss_fn(model, X, U, enable_reach, key, self.step_weights_eval, reach_eps)
            return loss, metrics

        self._train_step = train_step
        self._eval_step = eval_step

    def _current_lr(self) -> float:
        return float(self.lr_schedule(self.global_step))

    # -------------- public loop --------------

    def run(self):
        # immediately enable reachability only for "mid" mode
        reach_enabled = self.reach_mode == "mid"
        latest_reach_vol = 0
        latest_reach_penalty = 0
        curr_reach_eps = 0.0
        for epoch in range(1, self.cfg["n_epoch"] + 1):
            curr_T = int(self.T_schedule(epoch))
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

                run_reach = reach_enabled and self.global_step % self.reach_every == 0

                # start_time = time.time()
                curr_reach_eps = jnp.array(self.reach_eps_schedule(self.global_step))
                self.model, self.opt_state, loss, metrics = self._train_step(self.model, self.opt_state, X, U, subk, run_reach, curr_reach_eps)
                # jax.block_until_ready(loss)
                # end_time = time.time()
                # print(f"_train_step time: {end_time - start_time} seconds")
                if run_reach:
                    latest_reach_vol = metrics["reach_volume"]
                    latest_reach_penalty = metrics["reach_penalty"]

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
                vloss, vmetrics = self._eval_step(self.model, Xv, Uv, subk, False, curr_reach_eps)
                # if reach_enabled:
                #     vloss, vmetrics = self._reach_eval_step(self.model, Xv, Uv, subk, curr_reach_eps)
                # else:
                #     vloss, vmetrics = self._eval_step(self.model, Xv, Uv, subk)
                val_losses.append(float(vloss))
            va_loss = float(np.mean(val_losses)) if val_losses else float("nan")

            self.logger.log(
                {"train/loss": tr_loss, "val/loss": va_loss,
                 "train/mse": float(metrics["mse"]),
                 "val/reach_volume": float(vmetrics.get("reach_volume", 0.0)), "val/reach_penalty": float(vmetrics.get("reach_penalty", 0.0)),
                 "val/mse": float(vmetrics["mse"]),
                    "lr": self._current_lr(), "epoch": epoch, "T": int(curr_T),
                    "global_step": self.global_step},
                step=self.global_step
            )

            # ---- ckpt ----
            if va_loss < self.best_val:
                self.best_val = va_loss
                path_base = f"{self.out_dir}/best_model"
                self.save_fn(path_base, self.model, self.opt_state, self.global_step, self.cfg_full)

                if self._wandb_enabled and self._save_ckpts_to_wandb:
                    self.logger.save(path_base + ".eqx")
                    self.logger.save(path_base + ".npz")

            if epoch == self.cfg["n_epoch"]:
                path_base = f"{self.out_dir}/last_model"
                self.save_fn(path_base, self.model, self.opt_state, self.global_step, self.cfg_full
                )
                if self._wandb_enabled and self._save_ckpts_to_wandb:
                    self.logger.save(path_base + ".eqx")
                    self.logger.save(path_base + ".npz")

        if self.logger is not None:
            self.logger.finish()
