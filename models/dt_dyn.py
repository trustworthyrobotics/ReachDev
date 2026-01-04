# models/mlp_dynamics_eqx.py
from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

from utils.T_pushing import pose_to_kp
from models.mlp_utils import MLP

Array = jnp.ndarray
PRNGKey = jax.Array


class T_Dynamics(eqx.Module):

    # ---- static / hyper params ----
    Dx: int = eqx.field(static=True)
    Du: int = eqx.field(static=True)
    n_history: int = eqx.field(static=True)
    delta_u: bool = eqx.field(static=True)
    arch: Tuple[int, ...] = eqx.field(static=True)
    pred_mode: str = eqx.field(static=True)
    stem_size: Array = eqx.field(static=True)
    bar_size: Array = eqx.field(static=True)

    # ---- learnable parts ----
    mlp: MLP

    def __init__(
        self,
        data_cfg: dict, 
        train_cfg: dict,
        key: PRNGKey = jax.random.PRNGKey(0),
    ):
        arch_list: Sequence[int] = train_cfg["architecture"]
        assert len(arch_list) >= 1, "Architecture must have at least one hidden layer."
        self.arch = tuple(int(x) for x in arch_list)

        self.Du = int(data_cfg["action_dim"])
        self.n_history = int(train_cfg["n_history"])
        assert self.n_history == 1, "n_history must be == 1."
        self.delta_u = bool(train_cfg.get("delta_u", False))

        self.pred_mode = str(train_cfg.get("pred_mode", "state"))
        if self.pred_mode == "state":
            self.Dx = int(data_cfg["state_dim"])
        elif self.pred_mode == "pose":
            self.Dx = int(data_cfg["pose_dim"])
            scale = float(data_cfg["scale"])
            self.stem_size = jnp.array(data_cfg["stem_size"]) / scale
            self.bar_size = jnp.array(data_cfg["bar_size"]) / scale

            self.delta_u = False  # override for pose prediction

        in_dim = sum(self._input_dims())
        out_dim = self.Dx  # predict Δx or x_next

        self.mlp = MLP(
            in_size=in_dim,
            out_size=out_dim,
            hidden_size_list=self.arch,
            key=key,
        )

    # --------------------------- helpers ---------------------------
    def _input_dims(self) -> List[int]:
        # (x,u) per step → Dx+Du; with n_history steps → n_history*(Dx+Du)
        return self.n_history * self.Dx, self.n_history * self.Du

    # --------------------------- forward / rollout ---------------------------
    def forward(self, x: Array, u: Array) -> Array:
        output = x + jax.vmap(self.mlp)(jnp.concatenate([x, u], axis=-1))
        if self.delta_u:
            return output - u.repeat(self.Dx // self.Du, axis=-1)
        return output

    def forward_batchless(self, x: Array, u: Array) -> Array:
        output = x + self.mlp(jnp.concatenate([x, u], axis=-1))
        if self.delta_u:
            return output - u.repeat(self.Dx // self.Du, axis=-1)
        return output

    # --------------------------- batchless onnx for JAX ---------------------------
    def forward_batchless_single_input(self, inp):
        x = inp[:self.Dx]
        u = inp[-self.Du:]
        output = x + self.mlp(inp)
        if self.delta_u:
            return output - u.repeat(self.Dx // self.Du, axis=-1)
        return output

    # --------------------------- batch onnx for torch ---------------------------
    def forward_batch_single_input(self, inp):
        x = inp[:, :self.Dx]
        u = inp[:, -self.Du:]
        output = x + jax.vmap(self.mlp)(inp)
        if self.delta_u:
            return output - u.repeat(self.Dx // self.Du, axis=-1)
        return output

    __call__ = forward_batchless_single_input

    # it is only used for Jacobian regularization
    def forward_batchless_mlp_only(self, x: Array, u: Array) -> Array:
        return self.mlp(jnp.concatenate([x, u], axis=-1))

    def rollout(self, x0: Array, U: Array) -> Array:
        """
        Autoregressive rollout:
          x_{t+1} = f(x_t, u_t)  (with history if enabled)
        Args:
          x0: (B, Dx)              initial states
          U : (B, T, Du)           actions for T steps
          T : optional horizon (defaults to length of U or config default)
        Returns:
          X_pred: (B, T, Dx)       predictions for x_{1:T}
        """
        T = U.shape[1]

        # History buffer: last n_history states & actions.
        # For n_history=1, this is just (x_t, u_t).
        def step_fn(x_t, u_t):
            x_tp1 = self.forward(x_t, u_t)
            return x_tp1, x_tp1

        U_tm = jnp.swapaxes(U[:, :T, :], 0, 1)  # (T,B,Du)
        _, X_seq = jax.lax.scan(step_fn, x0, U_tm)  # (T,B,Dx)
        return jnp.swapaxes(X_seq, 0, 1)  # (B,T,Dx)


    # transform pose to keypoints
    def transform_fn(self, x):
        if self.pred_mode != "pose":
            return x
        B, T, D = x.shape
        x = x.reshape(-1, D)
        kp = jax.vmap(pose_to_kp, in_axes=(0, None, None))(x, self.stem_size, self.bar_size)
        return kp.reshape(B, T, -1)
