from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

from models.mlp_utils import MLP

Array = jnp.ndarray
PRNGKey = jax.Array

class Quad_Dynamics(eqx.Module):

    # ---- static / hyper params ----
    Dx: int = eqx.field(static=True)
    Dv: int = eqx.field(static=True)
    Du: int = eqx.field(static=True)
    arch: Tuple[int, ...] = eqx.field(static=True)
    dt: float = eqx.field(static=True)

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

        self.Du = int(data_cfg["dt_action_dim"])
        self.Dx = int(data_cfg["dt_state_dim"])
        assert self.Du == 3
        assert self.Dx == 6
        self.Dv = 3  # velocity commands
        self.dt = float(1 / data_cfg.get("dt_frequency", 5))

        in_dim = self.Dv + self.Du  # input: object state + action
        # even use abs_pose, no need to predict pusher position
        out_dim = self.Dv

        self.mlp = MLP(
            in_size=in_dim,
            out_size=out_dim,
            hidden_size_list=self.arch,
            key=key,
        )

    # --------------------------- helpers ---------------------------
    def _input_dims(self) -> List[int]:
        return self.Dx, self.Du

    # --------------------------- forward / rollout ---------------------------
    def forward(self, x: Array, u: Array) -> Array:
        return jax.vmap(self.forward_batchless)(x, u)

    def forward_batchless(self, x: Array, u: Array) -> Array:
        pos = x[:self.Dx - self.Dv]
        vel = x[self.Dx - self.Dv:self.Dx]
        inp = jnp.concatenate([vel, u], axis=-1)  # (3+3,)
        delta_vel = self.mlp(inp)  # (3,)
        new_vel = vel + delta_vel
        new_pos = pos + 0.5 * (vel + new_vel) * self.dt
        x_next = jnp.concatenate([new_pos, new_vel], axis=-1)  # (6,)
        return x_next

    # --------------------------- batchless onnx for JAX ---------------------------
    def forward_batchless_single_input(self, inp):
        x = inp[:self.Dx]
        u = inp[-self.Du:]
        return self.forward_batchless(x, u)

    # --------------------------- batch onnx for torch ---------------------------
    def forward_batch_single_input(self, inp):
        x = inp[:, :self.Dx]
        u = inp[:, -self.Du:]
        return self.forward(x, u)

    __call__ = forward_batchless_single_input

    # it is only used for Jacobian regularization
    def forward_batchless_for_jac(self, x: Array, u: Array) -> Array:
        return self.forward_batchless(x, u)

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

        # History buffer: last 1 states & actions.
        def step_fn(x_t, u_t):
            x_tp1 = self.forward(x_t, u_t)
            return x_tp1, x_tp1

        U_tm = jnp.swapaxes(U[:, :T, :], 0, 1)  # (T,B,Du)
        _, X_seq = jax.lax.scan(step_fn, x0, U_tm)  # (T,B,Dx)
        return jnp.swapaxes(X_seq, 0, 1)  # (B,T,Dx)
