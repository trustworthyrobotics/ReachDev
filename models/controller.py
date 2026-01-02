from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

Array = jnp.ndarray
PRNGKey = jax.Array

from models.dynamics import MLP

class T_controller(eqx.Module):
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
        train_cfg = config["train"]
        arch_list: Sequence[int] = train_cfg["controller_architecture"]
        assert len(arch_list) >= 1, "Architecture must have at least one hidden layer."
        self.arch = tuple(int(x) for x in arch_list)

        self.ref_act = train_cfg.get("ref_action", True)

        pred_mode = str(train_cfg.get("pred_mode", "state"))
        if pred_mode == "state":
            self.Dx = int(data_cfg["state_dim"])
        elif pred_mode == "pose":
            self.Dx = int(data_cfg["pose_dim"])
        self.Du = int(data_cfg["action_dim"])

        in_dim = self.Dx * 2  # current state + target state
        out_dim = self.Du  # predict action

        self.mlp = MLP(
            in_size=in_dim,
            out_size=out_dim,
            hidden_size_list=self.arch,
            key=key,
        )

    def forward(self, x, x_target, ref_action=None):
        # x: (B,Dx), x_target: (B,Dx), ref_action: (B,Du)
        if self.ref_act:
            inp = jnp.concatenate([x, x_target, ref_action], axis=-1)
            return ref_action + self.mlp(inp)  # (B,Du) predicted action
        else:
            inp = jnp.concatenate([x, x_target], axis=-1)
            return self.mlp(inp)  # (B,Du) predicted action