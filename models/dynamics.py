# models/mlp_dynamics_eqx.py
from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


Array = jnp.ndarray
PRNGKey = jax.Array


class MLP(eqx.Module):
    layers: Tuple[Union[eqx.nn.Linear, Callable], ...]
    in_size: Union[int, str]
    out_size: Union[int, str]
    hidden_size_list: Iterable[int]
    depth: int
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""
    def __init__(
        self,
        in_size: Union[int, str],
        out_size: Union[int, str],
        hidden_size_list: Iterable[int],
        *,
        key: PRNGKey,
    ):
        """**Arguments**:

        - `in_size`: The input size. The input to the module should be a vector of
            shape `(in_features,)`
        - `out_size`: The output size. The output from the module will be a vector
            of shape `(out_features,)`.
        """
        depth = len(hidden_size_list)
        keys = jax.random.split(key, depth + 1)
        layers = []

        # Input layer
        layers.append(
            eqx.nn.Linear(
                in_size,
                hidden_size_list[0],
                key=keys[0],
                dtype=jnp.float32
            )
        )
        # Hidden layers
        for i in range(1, depth):
            layers.append(
                eqx.nn.Linear(
                    hidden_size_list[i - 1],
                    hidden_size_list[i],
                    key=keys[i],
                    dtype=jnp.float32
                )
            )
        # Output layer
        layers.append(
            eqx.nn.Linear(
                hidden_size_list[-1],
                out_size,
                key=keys[-1],
                dtype=jnp.float32
            )
        )

        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size_list = hidden_size_list
        self.depth = depth

    def forward(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x
    __call__ = forward

class T_Dynamics(eqx.Module):

    # ---- static / hyper params ----
    Dx: int = eqx.field(static=True)
    Du: int = eqx.field(static=True)
    n_history: int = eqx.field(static=True)
    activation_name: str = eqx.field(static=True)
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
        train_cfg = config["train"]
        arch_list: Sequence[int] = train_cfg["architecture"]
        assert len(arch_list) >= 1, "Architecture must have at least one hidden layer."
        self.arch = tuple(int(x) for x in arch_list)

        self.Dx = int(data_cfg["state_dim"])
        self.Du = int(data_cfg["action_dim"])
        self.n_history = int(train_cfg["n_history"])
        assert self.n_history == 1, "n_history must be == 1."
        self.activation_name = train_cfg["activation"]

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
        return x + jax.vmap(self.mlp)(jnp.concatenate([x, u], axis=-1)) - u.repeat(4,axis=-1)

    def forward_single(self, x: Array, u: Array) -> Array:
        return x + self.mlp(jnp.concatenate([x, u], axis=-1)) - u.repeat(4,axis=-1)

    # --------------------------- batchless onnx for JAX ---------------------------
    def forward_batchless_onnx(self, inp):
        x = inp[:self.Dx]
        u = inp[-self.Du:]
        output = x + self.mlp(jnp.concatenate([x, u], axis=0)) - u.repeat(4,axis=-1)
        return output

    # --------------------------- batch onnx for torch ---------------------------
    def forward_batch_onnx(self, inp):
        x = inp[:, :self.Dx]
        u = inp[:, -self.Du:]
        output = x + jax.vmap(self.mlp)(jnp.concatenate([x, u], axis=1)) - u.repeat(4,axis=-1)
        return output

    __call__ = forward_batchless_onnx
    # __call__ = forward_batch_onnx

    def rollout_model(self, x0: Array, U: Array) -> Array:
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


def load_t_dynamics_model(config: dict, model_path: str) -> T_Dynamics:
    model_def = T_Dynamics(config=config)
    with open(model_path, "rb") as f:
        model: T_Dynamics = eqx.tree_deserialise_leaves(f, model_def)
    return model