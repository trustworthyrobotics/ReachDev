from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

Array = jnp.ndarray
PRNGKey = jax.Array

from models.mlp_utils import MLP

class T_controller(eqx.Module):
    # ---- static / hyper params ----
    Dx: int = eqx.field(static=True)
    Du: int = eqx.field(static=True)
    Dr: int = eqx.field(static=True)
    arch: Tuple[int, ...] = eqx.field(static=True)
    ref_act: bool = eqx.field(static=True)
    use_delta: bool = eqx.field(static=True)

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

        self.ref_act = train_cfg.get("ref_action", False)
        self.use_delta = train_cfg.get("use_delta", False)

        pred_mode = str(train_cfg.get("pred_mode", "state"))
        if pred_mode == "state":
            self.Dx = int(data_cfg["state_dim"])
        elif pred_mode == "pose":
            self.Dx = int(data_cfg["pose_dim"])
        self.Du = int(data_cfg["action_dim"])

        if self.use_delta:
            in_dim = self.Dx  # input: state difference
        else:
            in_dim = self.Dx * 2  # current state + target state
        if self.ref_act:
            in_dim += self.Du
        self.Dr = in_dim - self.Dx
        out_dim = self.Du  # predict action

        self.mlp = MLP(
            in_size=in_dim,
            out_size=out_dim,
            hidden_size_list=self.arch,
            key=key,
        )

    def _input_dims(self) -> List[int]:
        dims = [self.Dx, self.Dx]
        if self.ref_act:
            dims.append(self.Du)
        return dims

    def forward(self, x, x_target, ref_action=None):
        # x: (B,Dx), x_target: (B,Dx), ref_action: (B,Du)
        return jax.vmap(self.forward_batchless)(x, x_target, ref_action)

    def forward_batchless(self, x, x_target, ref_action=None):
        # x: (Dx,), x_target: (Dx,), ref_action: (Du,)
        if self.ref_act:
            if self.use_delta:
                inp = jnp.concatenate([x_target - x, ref_action], axis=-1)
            else:
                inp = jnp.concatenate([x, x_target, ref_action], axis=-1)
            return ref_action + self.mlp(inp)  # (Du,) predicted action
        else:
            if self.use_delta:
                inp = x_target - x
            else:
                inp = jnp.concatenate([x, x_target], axis=-1)
            return self.mlp(inp)  # (Du,) predicted action

    def forward_batchless_single_input(self, inp):
        x = inp[:self.Dx]
        x_target = inp[self.Dx:2*self.Dx]
        if self.ref_act:
            ref_action = inp[2*self.Dx:2*self.Dx + self.Du]
            return self.forward_batchless(x, x_target, ref_action)
        else:
            return self.forward_batchless(x, x_target)
        
    __call__ = forward_batchless_single_input

    # it is only used for Jacobian regularization
    def forward_batchless_for_jac(self, x, x_target, ref_action=None) -> Array:
        return self.mlp(jnp.concatenate([x, x_target, ref_action] if self.ref_act else [x, x_target], axis=-1))

def load_t_controller_model(data_config: dict, train_config: dict, model_path: str) -> T_controller:
    model_def = T_controller(data_config, train_config)
    with open(model_path, "rb") as f:
        model: T_controller = eqx.tree_deserialise_leaves(f, model_def)
    return model
