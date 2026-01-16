import jax
import jax.numpy as jnp
import onnx
from jaxonnxruntime.backend import Backend as ONNXJaxBackend
from jaxonnxruntime.core import config_class
config = config_class.config
config.update("jaxort_only_allow_initializers_as_static_args", False)
from jax_verify import IntervalBound, backward_crown_bound_propagation

import equinox as eqx
import yaml

from models.load import load_model
from models.T_pushing.dt_dyn import T_Dynamics

state_dim = 8
action_dim = 2
horizon = 3

model=load_model(model_dir="output/runs/T_pushing_test", model_type="dt_dyn", mode="best")

def f(x):
    return model(x)

def plan(x):
    state_init = x[:state_dim]
    action_seq = x[state_dim:].reshape((horizon, action_dim))
    x_curr = state_init
    for t in range(horizon):
        action_curr = action_seq[t]
        input_dyn = jnp.concatenate([x_curr, action_curr], axis=0)
        x_next = f(input_dyn)
        x_curr = x_next
    return x_curr

model_jax = plan

# -0.10673869, -0.48860836,  0.40756166, -0.17958558,  0.92186177, 0.12943733,  0.94835174, -1.0796111
state_init = jnp.array([-0.10673869, -0.48860836,  0.40756166, -0.17958558,  0.92186177, 0.12943733,  0.94835174, -1.0796111], dtype=jnp.float32)
eps = 0.05
state_lo = state_init - eps
state_hi = state_init + eps
action_init = jnp.ones((action_dim * horizon,), dtype=jnp.float32) * 0
action_eps = 0.0
action_lo = action_init - action_eps
action_hi = action_init + action_eps
x_lo = jnp.concatenate([state_lo, action_lo], axis=0)
x_hi = jnp.concatenate([state_hi, action_hi], axis=0)
output_bounds = backward_crown_bound_propagation(model_jax, IntervalBound(x_lo, x_hi))

print("Output lower bounds:", output_bounds.lower)
print("Output upper bounds:", output_bounds.upper)


def objective_fn(m):
    def plan(x):
        state_init = x[:state_dim]
        action_seq = x[state_dim:].reshape((horizon, action_dim))
        x_curr = state_init
        for t in range(horizon):
            action_curr = action_seq[t]
            input_dyn = jnp.concatenate([x_curr, action_curr], axis=0)
            x_next = m(input_dyn)
            x_curr = x_next
        return x_curr

    output_bounds = backward_crown_bound_propagation(plan, IntervalBound(x_lo, x_hi))
    # We want to minimize the final position x and y (first two dimensions)
    return jnp.sum(output_bounds.upper - output_bounds.lower)

model_grads = eqx.filter_value_and_grad(objective_fn, has_aux=False)(model)
print("model grads:", [g for g in jax.tree.leaves(model_grads)])