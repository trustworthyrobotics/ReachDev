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
    frequency: float = eqx.field(static=True)
    arch: Tuple[int, ...] = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    x_mean: Array = eqx.field(static=True)
    x_std: Array = eqx.field(static=True)
    u_mean: Array = eqx.field(static=True)
    u_std: Array = eqx.field(static=True)

    # ---- learnable parts ----
    mlp: MLP

    def __init__(
        self,
        data_cfg: dict, 
        train_cfg: dict,
        key: PRNGKey = jax.random.PRNGKey(0),
        stats: Optional[dict] = None,
    ):
        arch_list: Sequence[int] = train_cfg["architecture"]
        assert len(arch_list) >= 1, "Architecture must have at least one hidden layer."
        self.arch = tuple(int(x) for x in arch_list)

        self.Du = int(data_cfg["dt_action_dim"])
        self.Dx = int(data_cfg["dt_state_dim"])
        assert self.Du == 3
        assert self.Dx == 6
        self.Dv = 3  # velocity commands
        self.frequency = float(data_cfg.get("dt_frequency", 5))
        self.dt = 1 / self.frequency

        in_dim = self.Dv + self.Du  # input: object state + action
        # even use abs_pose, no need to predict pusher position
        out_dim = self.Dv

        self.mlp = MLP(
            in_size=in_dim,
            out_size=out_dim,
            hidden_size_list=self.arch,
            key=key,
        )
        if stats is not None:
            mean = jnp.array(stats["mean"])
            std = jnp.array(stats["std"])
            assert mean.shape == (self.Dx + self.Du,)
            assert std.shape == (self.Dx + self.Du,)

            self.x_mean = mean[:self.Dx]
            self.x_std = std[:self.Dx]
            self.u_mean = mean[self.Dx:]
            self.u_std = std[self.Dx:]
        else:
            self.x_mean = jnp.zeros(self.Dx)
            self.x_std = jnp.ones(self.Dx)
            self.u_mean = jnp.zeros(self.Du)
            self.u_std = jnp.ones(self.Du)

    # --------------------------- helpers ---------------------------
    def _input_dims(self) -> List[int]:
        return self.Dx, self.Du

    # --------------------------- forward / rollout ---------------------------
    def forward(self, x: Array, u: Array) -> Array:
        return jax.vmap(self.forward_batchless)(x, u)

    def forward_batchless(self, x: Array, u: Array) -> Array:
        x = (x - self.x_mean) / self.x_std
        u = (u - self.u_mean) / self.u_std
        pos = x[:self.Dx - self.Dv]
        vel = x[self.Dx - self.Dv:self.Dx]
        inp = jnp.concatenate([vel, u], axis=-1)  # (3+3,)
        delta_vel = self.mlp(inp)  # (3,)
        new_vel = vel + delta_vel
        new_pos = pos + 0.5 * (vel + new_vel) * self.dt
        x_next = jnp.concatenate([new_pos, new_vel], axis=-1)  # (6,)
        x_next = x_next * self.x_std + self.x_mean
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

    def transform_fn(self, x: Array) -> Array:
        return x
