from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

Array = jnp.ndarray
PRNGKey = jax.Array

class MLP(eqx.Module):
    layers: Tuple[Union[eqx.nn.Linear, Callable], ...]
    in_size: Union[int, str] = eqx.field(static=True)
    out_size: Union[int, str] = eqx.field(static=True)
    hidden_size_list: Iterable[int] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""
    def __init__(
        self,
        in_size: Union[int, str],
        out_size: Union[int, str],
        hidden_size_list: Iterable[int],
        activation: str = "relu",
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
        if activation == "relu":
            self.activation = jax.nn.relu
        elif activation == "tanh":
            self.activation = jax.nn.tanh
        elif activation == "sigmoid":
            self.activation = jax.nn.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    __call__ = forward

