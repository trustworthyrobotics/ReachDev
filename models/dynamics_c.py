from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

Array = jnp.ndarray
PRNGKey = jax.Array


from models.dynamics import MLP

class Continuous_T_Dynamics(eqx.Module):
    # ---- static / hyper params ----
    Dx: int = eqx.field(static=True)
    Du: int = eqx.field(static=True)
    arch: Tuple[int, ...] = eqx.field(static=True)

    # ---- learnable parts ----
    mlp: MLP

    def __init__(
        self,
        *,
        config: dict,
        key: PRNGKey = jax.random.PRNGKey(0),
    ):
        data_cfg = config["data"]
        train_cfg = config["train_cont"]
        arch_list: Sequence[int] = train_cfg["architecture"]
        assert len(arch_list) >= 1, "Architecture must have at least one hidden layer."
        self.arch = tuple(int(x) for x in arch_list)

        self.Dx = int(data_cfg["state_dim"])
        self.Du = int(data_cfg["action_dim"])

        in_dim = sum(self._input_dims())
        out_dim = self.Dx  # predict Δx or x_next

        self.mlp = MLP(
            in_size=in_dim,
            out_size=out_dim,
            hidden_size_list=self.arch,
            key=key,
        )

    def dx(self, t, x, args):
            u = args  # pusher velocity
            # Predict the derivative of keypoints
            return self.mlp(jnp.concatenate([x, u], axis=-1))

    def forward(self, x, u, dt):
        # x: (B,Dx), u: (B,Du), dt: scalar
        # One step forward prediction
        term = diffrax.ODETerm(self.dx)
        solver = diffrax.Tsit5()
        sol = jax.vmap(diffrax.diffeqsolve)(term, solver, t0=0, t1=dt, dt0=dt/5, y0=x, args=u)
        return sol.ys[-1]

    def rollout(self, x0, U, dt):
        # x0: (B,Dx), U: (B,T,Du), dt: scalar
        # Use diffrax to integrate over the control frequency
        def step(state, u):
            term = diffrax.ODETerm(self.dx)
            solver = diffrax.Tsit5()
            sol = diffrax.diffeqsolve(term, solver, t0=0, t1=dt, dt0=dt/5, y0=state, args=u)
            return sol.ys[-1], sol.ys[-1]
            
        _, x_seq = jax.lax.scan(jax.vmap(step), x0, U)
        return x_seq

    def forward_batchless_onnx(self, inp):
        return self.mlp(inp)
    
    __call__ = forward_batchless_onnx
