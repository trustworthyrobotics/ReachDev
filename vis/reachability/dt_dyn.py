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
from CROWN_Reach.src.reachability import DT_Plan_Reach
from CROWN_Reach.src.reachability_baseline import DT_Plan_Reach_CROWN
from CROWN_Reach.src.utils.vis import visualize_flowpipe_time
from models.mlp_utils import MLP
import numpy as np


state_dim = 5
action_dim = 2
hidden_size_list = [96, 96, 96]

key = jrandom.PRNGKey(0)
model = MLP(
    in_size=state_dim + action_dim,
    out_size=state_dim,
    hidden_size_list=hidden_size_list,
    key=key,
)

reach_eps = 0.05
n_reach_batch = 32
print(f"reach_eps: {reach_eps}, n_reach_batch: {n_reach_batch}")

# random input
key, subkey = jrandom.split(key)
z = jrandom.normal(subkey, (n_reach_batch, state_dim+action_dim))

z_init_lo = z - reach_eps
z_init_up = z + reach_eps

def f_wrapper(x):
    state_next = model(x) + x[:state_dim]
    action_next = x[state_dim:]
    return jnp.concatenate([state_next, action_next], axis=-1)


methods = ['ours', 'crown']
T_reach_list = [2, 4, 6, 8, 10]

# random action sequence
key, subkey = jrandom.split(key)
max_action_seq = jrandom.normal(subkey, (1, max(T_reach_list), action_dim))

n_runs = 10

result_dict = {}

for method in methods:
    if method == 'ours':
        reach_analyzer = DT_Plan_Reach(f_wrapper, state_dim=state_dim, action_dim=action_dim, nn_dyn=True, n_steps_per_plan=1, step_size=1)
    else:
        reach_analyzer = DT_Plan_Reach_CROWN(f_wrapper, state_dim=state_dim, action_dim=action_dim, nn_dyn=True, n_steps_per_plan=1, step_size=1)
    jit_verify = eqx.filter_jit(reach_analyzer.verify)

    result_dict[method] = {}
    for T_reach in T_reach_list:
        print(f"Method: {method}, T_reach: {T_reach}")

        action_seq = max_action_seq[:, :T_reach, :]

        start_time = time.time()
        ts, r_lo, r_up, x_nexts_all, _ = jit_verify(z_init_lo, z_init_up, n_total_steps=T_reach, action_seq=action_seq.repeat(z_init_up.shape[0]//action_seq.shape[0], axis=0)[:, None])
        end_time = time.time()
        compile_time = end_time - start_time
        print(f"Compile time for reachability analysis over {T_reach} steps: {compile_time:.4f} seconds")

        start_time = time.time()
        for _ in range(n_runs):
            ts, r_lo, r_up, x_nexts_all, _ = jit_verify(z_init_lo, z_init_up, n_total_steps=T_reach, action_seq=action_seq.repeat(z_init_up.shape[0]//action_seq.shape[0], axis=0)[:, None])
        end_time = time.time()
        run_time = (end_time - start_time) / n_runs
        print(f"Avg run time for reachability analysis over {T_reach} steps: {run_time:.4f} seconds")

        r_lo = r_lo.reshape(n_reach_batch, -1, T_reach + 1, state_dim+action_dim)  # [N, splits, T+1, Dx+Du]
        r_up = r_up.reshape(n_reach_batch, -1, T_reach + 1, state_dim+action_dim)  # [N, splits, T+1, Dx+Du]

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

out_file = f"experiments/results/reach_dt_dyn_eps{reach_eps}_batch{n_reach_batch}.pkl"
with open(out_file, "wb") as f:
    pickle.dump(result_dict, f)
print(f"Saved results to {out_file}")