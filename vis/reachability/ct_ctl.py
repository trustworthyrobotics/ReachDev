import numpy as np
import pickle
import jax
# jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax import random as jrandom
import equinox as eqx
import time
import sys


from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import CT_Ctl_Reach
from CROWN_Reach.src.reachability_baseline import CT_Ctl_Reach_immrax, DT_Plan_Reach_CROWN
from CROWN_Reach.src.utils.vis import visualize_flowpipe_time
from CROWN_Reach.src.settings import CONFIG
from models.mlp_utils import MLP
import numpy as np

from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics


state_dim = 12
action_dim = 3
ct_dyn = Continuous_Quad_Dynamics({})


# state_dim = 4
# action_dim = 1

# def dynamics(x):
#     x1,x2,x3,x4,u1 = x
#     dx1 = x2
#     dx2 = 2 * u1
#     dx3 = x4
#     dx4 = (0.08*0.41*(9.8 * jnp.sin(x3) - 2*u1 * jnp.cos(x3)) - 0.0021 * x4) / 0.0105
#     return jnp.stack([dx1, dx2, dx3, dx4], axis=0)
# ct_dyn = dynamics

hidden_size_list = [64, 64]

key = jrandom.PRNGKey(0)
model = MLP(
    in_size=state_dim,
    out_size=action_dim,
    hidden_size_list=hidden_size_list,
    key=key,
)

reach_eps = 0.05
n_reach_batch = 128
print(f"reach_eps: {reach_eps}, n_reach_batch: {n_reach_batch}")

# random input
key, subkey = jrandom.split(key)
z = jrandom.normal(subkey, (n_reach_batch, state_dim+action_dim))

z_init_lo = z - reach_eps
z_init_up = z + reach_eps

def f_wrapper(x):
    dx = ct_dyn(x)
    du = jnp.zeros_like(x[state_dim:])
    return jnp.concatenate([dx, du], axis=-1)

init_remainder = reach_eps
frr_rounds = 5
frr_stop_ratio = 0.95
sr_window_size = 100
# CONFIG["TRUNCATE_TO_AFFINE"] = True
n_dyn_steps_per_ctl = 1
dyn_frequency = 100

methods = ['immrax']
T_reach_list = [10]

max_action_seq = jnp.zeros((n_reach_batch, max(T_reach_list), 0))

n_runs = 10

result_dict = {}

for method in methods:
    if method == 'ours':
        reach_analyzer = CT_Ctl_Reach(f_wrapper, state_dim=state_dim, action_dim=action_dim, nn_dyn=False, controller=model,
                                  n_steps_per_control=n_dyn_steps_per_ctl, step_size=1/dyn_frequency,
                                  init_remainder=init_remainder, frr_rounds=frr_rounds, frr_stop_ratio=frr_stop_ratio, sr_window_size=sr_window_size)
    else:
        reach_analyzer = CT_Ctl_Reach_immrax(f_wrapper, state_dim=state_dim, action_dim=action_dim, nn_dyn=False, controller=model,
                                  n_steps_per_control=n_dyn_steps_per_ctl, step_size=1/dyn_frequency,
                                  init_remainder=init_remainder, frr_rounds=frr_rounds, frr_stop_ratio=frr_stop_ratio, sr_window_size=sr_window_size)
    jit_verify = eqx.filter_jit(reach_analyzer.verify)

    result_dict[method] = {}
    for T_reach in T_reach_list:
        print(f"Method: {method}, T_reach: {T_reach}")

        reference_seq = max_action_seq[:, :T_reach, :]

        start_time = time.time()
        ts, r_lo, r_up, x_nexts_all, _ = jit_verify(z_init_lo, z_init_up, n_total_steps=T_reach, reference_seq=reference_seq)
        end_time = time.time()
        compile_time = end_time - start_time
        print(f"Compile time for reachability analysis over {T_reach} steps: {compile_time:.4f} seconds")

        start_time = time.time()
        for _ in range(n_runs):
            ts, r_lo, r_up, x_nexts_all, _ = jit_verify(z_init_lo, z_init_up, n_total_steps=T_reach, reference_seq=reference_seq)
        end_time = time.time()
        run_time = (end_time - start_time) / n_runs
        print(f"Avg run time for reachability analysis over {T_reach} steps: {run_time:.4f} seconds")

        actual_r_dim = state_dim+action_dim if method == 'ours' else state_dim
        r_lo = r_lo.reshape(n_reach_batch, -1, T_reach + 1, actual_r_dim)  # [N, splits, T+1, Dx+Du]
        r_up = r_up.reshape(n_reach_batch, -1, T_reach + 1, actual_r_dim)  # [N, splits, T+1, Dx+Du]

        r_lo_agg = jnp.min(r_lo, axis=1, keepdims=False)  # [N, T+1, Dx+Du]
        r_up_agg = jnp.max(r_up, axis=1, keepdims=False)  # [N, T+1, Dx+Du]

        reach_vols = calculate_volume(r_lo_agg[..., :state_dim], r_up_agg[..., :state_dim], union_init=False, mode="sum", keep_time=True, keep_batch=True)

        print("Reach vols:", reach_vols.mean(axis=0))  # [T+1,]

        result_dict[method][T_reach] = {
            'compile_time': compile_time,
            'run_time': run_time,
            'reach_vols': np.array(reach_vols),
            'r_lo': np.array(r_lo_agg),
            'r_up': np.array(r_up_agg),
        }

out_file = f"experiments/results/reach_ct_ctl_eps{reach_eps}_batch{n_reach_batch}.pkl"
with open(out_file, "wb") as f:
    pickle.dump(result_dict, f)
print(f"Saved results to {out_file}")