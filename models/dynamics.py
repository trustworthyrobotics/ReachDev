# models/mlp_dynamics_eqx.py
from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


Array = jnp.ndarray
PRNGKey = jax.Array


class MLPDynamics(eqx.Module):
    """
    Equinox MLP for discrete dynamics modeling.
    Predicts either Δx (default) or x_{t+1} directly, with optional history window.

    Inputs:
      - x: (..., Dx)                     current state
      - u: (..., Du)                     current action

    Config keys used (typical):
      train:
        architecture: [128, 256, 256, 128]   # hidden sizes
        n_history: 1                          # num past steps of (x,u) to include
        n_rollout: 6                          # default rollout horizon (fallback only)

    Public methods:
      - forward(x, u) -> Δx or x_next
      - rollout_model(x0, U, T=None) -> X_pred[0:T] (batch-first)

    Shapes:
      x:  (B, Dx)
      u:  (B, Du)
      x0: (B, Dx)
      U:  (B, T, Du)
      returns: (B, T, Dx)
    """

    # ---- static / hyper params ----
    Dx: int = eqx.field(static=True)
    Du: int = eqx.field(static=True)
    n_history: int = eqx.field(static=True)
    activation_name: str = eqx.field(static=True)
    arch: Tuple[int, ...] = eqx.field(static=True)
    default_rollout_T: int = eqx.field(static=True)

    # ---- learnable parts ----
    mlp: eqx.nn.MLP

    # ---- (optional) normalization stats, kept non-trainable ----
    x_mean: Optional[Array] = None
    x_std: Optional[Array] = None
    u_mean: Optional[Array] = None
    u_std: Optional[Array] = None

    # --------------------------- construction ---------------------------

    def __init__(
        self,
        *,
        config: dict,
        key: PRNGKey = jax.random.PRNGKey(0),
        stats: Optional[dict] = None,
    ):
        data_cfg = config["data"]
        train_cfg = config["train"]
        arch_list: Sequence[int] = train_cfg["architecture"]
        assert len(arch_list) >= 1, "Architecture must have at least one hidden layer."
        self.arch = tuple(int(x) for x in arch_list)

        self.Dx = int(data_cfg["state_dim"])
        self.Du = int(data_cfg["action_dim"])
        self.n_history = int(train_cfg["n_history"])
        assert self.n_history == 1, "n_history must be == 1."
        self.default_rollout_T = int(train_cfg["n_rollout"])
        self.activation_name = train_cfg["activation"]

        if self.activation_name == "relu":
            act = jax.nn.relu
        elif self.activation_name == "tanh":
            act = jax.nn.tanh
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        in_dim = sum(self._input_dims())
        out_dim = self.Dx  # predict Δx or x_next

        self.mlp = eqx.nn.MLP(
            in_size=in_dim,
            out_size=out_dim,
            width_size=self.arch[0],
            depth=len(self.arch),
            activation=act,
            # final_activation=lambda y: y,  # identity
            key=key,
        )

        # normalization stats
        self.x_mean = None if stats is None else jnp.asarray(stats["x_mean"])
        self.x_std = None if stats is None else jnp.asarray(stats["x_std"])
        self.u_mean = None if stats is None else jnp.asarray(stats["u_mean"])
        self.u_std = None if stats is None else jnp.asarray(stats["u_std"])

    # --------------------------- helpers ---------------------------
    def _input_dims(self) -> List[int]:
        # (x,u) per step → Dx+Du; with n_history steps → n_history*(Dx+Du)
        return self.n_history * self.Dx, self.n_history * self.Du

    def _maybe_norm_x(self, x: Array) -> Array:
        if (self.x_mean is not None) and (self.x_std is not None):
            return (x - self.x_mean) / (self.x_std + 1e-8)
        return x

    def _maybe_denorm_x(self, x: Array) -> Array:
        if (self.x_mean is not None) and (self.x_std is not None):
            return x * (self.x_std + 1e-8) + self.x_mean
        return x

    def _maybe_norm_u(self, u: Array) -> Array:
        if (self.u_mean is not None) and (self.u_std is not None):
            return (u - self.u_mean) / (self.u_std + 1e-8)
        return u
    # --------------------------- core funcs ---------------------------

    def forward(self, x: Array, u: Array) -> Array:
        """
        One-step model inference.
        If n_history==1, x=(B,Dx), u=(B,Du).
        """
        x_n = self._maybe_norm_x(x)
        u_n = self._maybe_norm_u(u)
        h = jnp.concatenate([x_n, u_n], axis=-1)  # (B, in_dim)
        y = jax.vmap(self.mlp)(h)                # (B, Dx)

        return self._maybe_denorm_x(x_n + y)

    def rollout_model(self, x0: Array, U: Array, T: Optional[int] = None) -> Array:
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
        T = int(T or U.shape[1] if U.ndim == 3 else self.default_rollout_T)
        assert U.ndim == 3 and U.shape[1] >= T, "U must be (B,T,Du)."

        # History buffer: last n_history states & actions.
        # For n_history=1, this is just (x_t, u_t).
        def step_fn(x_t, u_t):
            x_tp1 = self.forward(x_t, u_t)
            return x_tp1, x_tp1

        U_tm = jnp.swapaxes(U[:, :T, :], 0, 1)  # (T,B,Du)
        _, X_seq = jax.lax.scan(step_fn, x0, U_tm)  # (T,B,Dx)
        return jnp.swapaxes(X_seq, 0, 1)  # (B,T,Dx)


class T_Dynamics(MLPDynamics):
    def forward(self, x: Array, u: Array) -> Array:
        """
        One-step model inference.
        If n_history==1, x=(B,Dx), u=(B,Du).
        """
        tc,tl,tr,bt = x[..., 0:2],x[..., 2:4],x[..., 4:6],x[..., 6:8]
        block_angle = [
            jnp.arctan2((tl - tc)[...,1],(tl - tc)[...,0]),
            jnp.arctan2((tc - tr)[...,1],(tc - tr)[...,0]),
            jnp.arctan2((bt - tc)[...,1],(bt - tc)[...,0])+jnp.pi*3/2,
            jnp.arctan2((tl - tr)[...,1],(tl - tr)[...,0]),
        ][-2]
        transition_matrix = jnp.zeros(x.shape[:-1] + (2, 2))
        transition_matrix = transition_matrix.at[..., 0, 0].set( jnp.cos(block_angle))
        transition_matrix = transition_matrix.at[..., 0, 1].set(-jnp.sin(block_angle))
        transition_matrix = transition_matrix.at[..., 1, 0].set( jnp.sin(block_angle))
        transition_matrix = transition_matrix.at[..., 1, 1].set( jnp.cos(block_angle))
        local_state = ((x.reshape(x.shape[0],-1,2) - tc[:,None,:]) @ transition_matrix).reshape(x.shape[0],-1)
        local_action = ((u - tc)[:,None,:] @ transition_matrix).reshape(u.shape[0],-1)
        x_n = self._maybe_norm_x(local_state)
        u_n = self._maybe_norm_u(local_action)
        h = jnp.concatenate([x_n, u_n], axis=-1)  # (B, in_dim)
        y = jax.vmap(self.mlp)(h)                # (B, Dx)
        output = local_state + y
        output = (output.reshape(x.shape[0],-1,2) @ transition_matrix.transpose((0,2,1)) + tc[:,None,:]).reshape(x.shape[0],-1)
        return self._maybe_denorm_x(output)

    def forward(self, x: Array, u: Array) -> Array:
        return x + jax.vmap(self.mlp)(jnp.concatenate([x, u], axis=-1))


    def forward(self, x: Array, u: Array) -> Array:
        return x + jax.vmap(self.mlp)(jnp.concatenate([x, u], axis=-1)) - u.repeat(4,axis=-1)

    __call__ = forward

def load_t_dynamics_model(config: dict, model_path: str) -> T_Dynamics:
    model_def = T_Dynamics(config=config)
    with open(model_path, "rb") as f:
        model: T_Dynamics = eqx.tree_deserialise_leaves(f, model_def)
    return model