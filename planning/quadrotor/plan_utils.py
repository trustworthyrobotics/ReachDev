import random
import math
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrandom
import sys
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DT_Plan_Reach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2

def _gen_pose_list(num_test, seed, lo, hi):
    # lo, hi: [state_dim]
    shift = 0
    num_test+=shift
    key = jrandom.PRNGKey(seed)
    random_poses = jrandom.uniform(key, (num_test, len(lo)), minval=0.0, maxval=1.0)
    scaled_poses = random_poses * jnp.array(hi - lo)[None, :] + jnp.array(lo)[None, :]
    return scaled_poses[shift:, :]


def generate_test_cases(seed, num_test, test_id=0):
    if test_id == 0:
        mid = jnp.zeros((12,))
        pos_eps = 1
        other_eps = 0.05
        lo = jnp.concatenate([mid[:3] - pos_eps, mid[3:] - other_eps])
        hi = jnp.concatenate([mid[:3] + pos_eps, mid[3:] + other_eps])
        init_pose_list = _gen_pose_list(num_test, seed, lo, hi)
        target_mid = mid.at[:3].set(10)
        target_lo = jnp.concatenate([target_mid[:3] - pos_eps, target_mid[3:] - other_eps])
        target_hi = jnp.concatenate([target_mid[:3] + pos_eps, target_mid[3:] + other_eps])
        target_pose_list = _gen_pose_list(num_test, seed, target_lo, target_hi)
    else:
        raise ValueError(f"Unknown test_id: {test_id}")
    return init_pose_list, target_pose_list

def plot_cost_stat(cost_stat, out_path):
    # cost_stat: (num_test, max_steps)
    plt.figure()
    for i in range(cost_stat.shape[0]):
        plt.plot(cost_stat[i], alpha=0.5, color="black")

    quantiles = np.percentile(cost_stat, [25, 50, 75], axis=0)
    plt.plot(quantiles[0], color="blue", label="25th percentile")
    plt.plot(quantiles[1], color="orange", label="50th percentile")
    plt.plot(quantiles[2], color="green", label="75th percentile")
    plt.legend()

    plt.xlabel("Time step")
    plt.ylabel("Step cost")
    plt.title("Step Cost over Time for Each Test Case")
    plt.grid()
    out_name = os.path.join(out_path, "step_costs.png")
    plt.savefig(out_name)
    print(f"Step cost plot saved to {out_name}")
    plt.close()
    return

def make_rollout_and_reward_fns(
    dt_dyn,
    planning_config: Dict,
    reach_config: Dict = {},
):
    horizon = planning_config["horizon"]
    cost_norm = planning_config["cost_norm"]
    only_final_cost = planning_config["only_final_cost"]
    enable_reach = planning_config.get("reach_in_obj", False) or planning_config.get("refinement", {}).get("reach_in_obj", False)

    if enable_reach:
        # reachability part
        def f_wrapper(x):
            state_next = dt_dyn(x)
            action_next = x[dt_dyn.Dx:]
            return jnp.concatenate([state_next, action_next], axis=-1)


        reach_analyzer = DT_Plan_Reach(f_wrapper, state_dim=dt_dyn.Dx, action_dim=dt_dyn.Du, nn_dyn=True, n_steps_per_plan=1, step_size=1)
        eps = reach_config.get("eps", 0.05)
        splits_cfg = reach_config.get("refine_splits", {})
        reach_weight = reach_config.get("reach_weight", 0.1)
        reach_loss_max = reach_config.get("reach_loss_max", 10.0)

        _calculate_volume = lambda lo, up: calculate_volume(lo, up, union_init=False, mode='sum')

    obstacle_config = planning_config.get("obstacle", {})
    enable_obstacle = obstacle_config.get("enable", False)
    if enable_obstacle:
        obstacle_pos_list = jnp.array(obstacle_config["pos_list"])  # [n_obstacle, 3]
        obstacle_size_list = jnp.array(obstacle_config["size_list"])  # [n_obstacle]
        norm = obstacle_config.get("norm", 2)
        penalty = obstacle_config.get("penalty", 100.0)
        inflate = obstacle_config.get("inflate", 0.0)

        def obstacle_cost_fn(state_seqs: jnp.ndarray) -> jnp.ndarray:
            # state_seqs: [n_sample, horizon, state_dim]
            pos_seqs = state_seqs[..., :3]  # [n_sample, horizon, 3]
            n_sample = pos_seqs.shape[0]
            pos_seqs_exp = pos_seqs[:, :, None, :]  # [n_sample, horizon, 1, 3]
            obstacle_pos_exp = obstacle_pos_list[None, None, :, :]  # [1, 1, n_obstacle, 3]
            dists = jnp.linalg.norm(pos_seqs_exp - obstacle_pos_exp, axis=-1, ord=norm)  # [n_sample, horizon, n_obstacle]
            size_exp = obstacle_size_list[None, None, :] * (1 + inflate)  # [1, 1, n_obstacle]
            penalties = jnp.maximum(0.0, size_exp - dists)  # [n_sample, horizon, n_obstacle]
            total_penalty = jnp.sum(penalties, axis=(1, 2)) * penalty # [n_sample]
            return total_penalty

    # [state_dim], [n_sample, horizon, action_dim] -> [n_sample, horizon, state_dim]
    def rollout_fn(state_cur: jnp.ndarray, act_seqs: jnp.ndarray, use_reach: bool) -> jnp.ndarray:
        state_cur = state_cur[None].repeat(act_seqs.shape[0], axis=0) # [B, Dx]
        state_seqs = dt_dyn.rollout(state_cur, act_seqs)
        if use_reach:
            reach_info = cal_reach(state_cur, act_seqs)
        return state_seqs, reach_info if use_reach else {}

    def cal_reach(state_cur: jnp.ndarray, act_seqs: jnp.ndarray) -> jnp.ndarray:
        B = act_seqs.shape[0]
        Da = dt_dyn.Du
        T = act_seqs.shape[1]
        state_lo = state_cur[:, :dt_dyn.Ds] - eps
        state_up = state_cur[:, :dt_dyn.Ds] + eps

        X_lo = jnp.concatenate([state_lo, jnp.zeros((B, Da))], axis=-1)
        X_up = jnp.concatenate([state_up, jnp.zeros((B, Da))], axis=-1)
        X_lo, X_up = prepare_initial_set_v2(X_lo, X_up, splits_cfg=splits_cfg)
        B_reach = X_lo.shape[0]

        _, r_lo, r_up, _, _ = reach_analyzer.verify(X_lo, X_up, n_total_steps=T, action_seq=act_seqs.repeat(B_reach//B, axis=0)[:, None])
        r_lo = r_lo.reshape(B, B_reach//B, T + 1, -1)
        r_up = r_up.reshape(B, B_reach//B, T + 1, -1)

        reach_vol = jax.vmap(_calculate_volume)(r_lo, r_up)  # [B]
        r_lo_agg = jnp.min(r_lo, axis=1)  # [B, T+1, Ds]
        r_up_agg = jnp.max(r_up, axis=1)  # [B, T+1, Ds]
        return {"reach_vol": reach_vol, "r_lo": r_lo_agg, "r_up": r_up_agg}

    # assume all are scaled
    def reward_fn(state_seqs: jnp.ndarray, act_seqs: jnp.ndarray, use_reach: bool, reach_aux: Dict, target_state: jnp.ndarray) -> Dict:
        cost_seqs = jnp.linalg.norm(state_seqs - target_state[None, None, :], axis=-1, ord=cost_norm) ** cost_norm
        if only_final_cost:
            costs = cost_seqs[:, -1]
        else:
            step_weight = jnp.linspace(1, horizon + 1, horizon) / horizon
            costs = jnp.sum(cost_seqs * step_weight[None, :], axis=-1)
        reach_loss = 0.0
        if use_reach:
            reach_loss = jnp.clip(reach_weight * jnp.log1p(reach_aux["reach_vol"]), a_max=reach_loss_max)
            costs = costs + reach_loss  # penalize large reachable set
        if enable_obstacle:
            obs_costs = obstacle_cost_fn(state_seqs)
            costs = costs + obs_costs
        return {"rewards": -costs, "reward_seqs": -cost_seqs, "reach_aux": reach_aux, "reach_loss": reach_loss}

    def step_cost_fn_np(state, target_state):
        return (np.linalg.norm(target_state - state, cost_norm)) ** cost_norm

    def step_cost_fn(state, target_state):
        return (jnp.linalg.norm(target_state - state, cost_norm)) ** cost_norm

    return rollout_fn, reward_fn, step_cost_fn, step_cost_fn_np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_planning_animation(polys_seqs, window_size=(500, 500), fps=10, save_path="plan.gif"):
    """
    polys_seqs: List of arrays, each [steps, horizon, vertices, 2]
    """
    fig, ax = plt.subplots()
    # figsize=(window_size[0]/100, window_size[1]/100)
    n_samples = polys_seqs.shape[0]
    n_sim_steps = polys_seqs.shape[1] - 1
    horizon = polys_seqs.shape[2] - 1
    
    # Setup Colormap for the horizon
    colors = plt.cm.rainbow(np.linspace(0, 1, horizon))

    def update(frame):
        ax.clear()
        ax.set_xlim(0, window_size[0])
        ax.set_ylim(0, window_size[1])
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f"Sim Step: {frame}")

        # Iterate through different polygon types (stem, bar, etc.)
        # 1. Plot Planned Horizon (Increasing transparency)
        for t in range(1, horizon + 1):
            for j in range(n_samples):
                vertices = polys_seqs[j, frame, t]
                alpha = 1.0 - (t / (horizon + 1)) # Fade out into the future
                
                poly = plt.Polygon(vertices, facecolor=colors[-t], 
                                edgecolor='none', alpha=alpha * 0.8, zorder=horizon + 1-t)
                ax.add_patch(poly)
        for j in range(n_samples):
            # 2. Plot Current State (Highlighted)
            curr_vertices = polys_seqs[j, frame, 0]
            curr_poly = plt.Polygon(curr_vertices, 
                                    facecolor='darkred', 
                                    edgecolor='none', linewidth=2, zorder=horizon + 1)
            ax.add_patch(curr_poly)

    ani = FuncAnimation(fig, update, frames=n_sim_steps + 1, repeat=False)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    print(f"Animation saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    import pickle
    pkl_file = "output/planning/T_pushing/0_uniform_0.05_0.01_mppi_1000_True/001407/planning_res_0000.pkl"
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # parameters
    stem_size = [30, 91]
    bar_size = [120, 30]
    window_size = [500, 500]
    scale = 100

    act_seqs = np.array([d["act_seq"] for d in data])
    state_seqs = np.array([d["state_seq"] for d in data])[..., :3]
    pusher_pos_seqs = np.array([d["pusher_pos_seq"] for d in data])

    # r_lo_seqs, r_up_seqs: (n_sim_steps+1, horizon+1, 3)
    r_lo_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_lo'] for d in data]).reshape((*state_seqs.shape[:2], -1))[..., :3]
    r_up_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_up'] for d in data]).reshape((*state_seqs.shape[:2], -1))[..., :3]

    r_lo_seqs[..., :2] = r_lo_seqs[..., :2] * scale
    r_up_seqs[..., :2] = r_up_seqs[..., :2] * scale

    # sample from r_lo_seqs and r_up_seqs
    n_samples = 8
    sample_states = np.random.uniform(size=(n_samples, *state_seqs.shape))
    sample_state_seqs = r_lo_seqs[None] + sample_states * (r_up_seqs - r_lo_seqs)[None]

    pass
