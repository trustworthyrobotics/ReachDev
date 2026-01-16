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
    Ds: int = eqx.field(static=True)
    Dx: int = eqx.field(static=True)
    Du: int = eqx.field(static=True)
    frequency: float = eqx.field(static=True)
    n_history: int = eqx.field(static=True)
    delta_u: bool = eqx.field(static=True)
    arch: Tuple[int, ...] = eqx.field(static=True)
    pred_mode: str = eqx.field(static=True)
    stem_size: Array = eqx.field(static=True)
    bar_size: Array = eqx.field(static=True)
    abs_pose: bool = eqx.field(static=True)

    # ---- learnable parts ----
    mlp: MLP

    def __init__(
        self,
        data_cfg: dict, 
        train_cfg: dict,
        key: PRNGKey = jax.random.PRNGKey(0),
        *args,
        **kwargs,
    ):
        arch_list: Sequence[int] = train_cfg["architecture"]
        assert len(arch_list) >= 1, "Architecture must have at least one hidden layer."
        self.arch = tuple(int(x) for x in arch_list)

        self.Du = int(data_cfg["action_dim"])
        self.n_history = int(train_cfg["n_history"])
        assert self.n_history == 1, "n_history must be == 1."
        self.delta_u = bool(train_cfg.get("delta_u", False))
        self.frequency = float(train_cfg.get("frequency", 10))

        self.pred_mode = str(train_cfg.get("pred_mode", "state"))
        if self.pred_mode == "state":
            self.Ds = int(data_cfg["state_dim"])
        elif self.pred_mode == "pose":
            self.Ds = int(data_cfg["pose_dim"])
            scale = float(data_cfg["scale"])
            self.stem_size = jnp.array(data_cfg["stem_size"]) / scale
            self.bar_size = jnp.array(data_cfg["bar_size"]) / scale

            self.delta_u = False  # override for pose prediction

        self.abs_pose = bool(train_cfg.get("abs_pose", False))
        if self.abs_pose:
            self.Dx = self.Ds + self.Du  # include pusher position in state
        else:
            self.Dx = self.Ds  # only object state

        in_dim = self.Ds + self.Du  # input: object state + action
        # even use abs_pose, no need to predict pusher position
        out_dim = self.Ds

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
        if self.abs_pose:
            x_obj = x[:self.Ds]
            x_pusher = x[-self.Du:]
            if self.pred_mode == "state":
                x_rel = x_obj - x_pusher.repeat(self.Ds // self.Du, axis=-1)
            elif self.pred_mode == "pose":
                x_rel = jnp.concatenate([x_obj[:self.Du] - x_pusher, x_obj[self.Du:]], axis=-1)
            dx_obj = x_rel + self.mlp(jnp.concatenate([x_rel, u], axis=-1))
            x_pusher_next = x_pusher + u

            if self.pred_mode == "state":
                x_obj_next = dx_obj + x_pusher_next.repeat(self.Ds // self.Du, axis=-1)
            elif self.pred_mode == "pose":
                x_obj_next = jnp.concatenate([dx_obj[:self.Du] + x_pusher_next, dx_obj[self.Du:]], axis=-1)
            return jnp.concatenate([x_obj_next, x_pusher_next], axis=-1)
        else:
            output = x + self.mlp(jnp.concatenate([x, u], axis=-1))
            if self.delta_u:
                if self.pred_mode == "state":
                    return output - u.repeat(self.Ds // self.Du, axis=-1)
                elif self.pred_mode == "pose":
                    return jnp.concatenate([output[:self.Du] - u, output[self.Du:]], axis=-1)
            else:
                return output

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


    # transform pose to keypoints
    def transform_fn(self, x):
        return x  # default no transform
        if self.pred_mode != "pose":
            return x
        B, T, D = x.shape
        x = x.reshape(-1, D)
        kp = jax.vmap(pose_to_kp, in_axes=(0, None, None))(x, self.stem_size, self.bar_size)
        return kp.reshape(B, T, -1)
