import random
import math
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys

from utils.T_pushing import hole_to_walls_aabbs, detect_T_hole_interaction, detect_T_hole_interaction_set
from utils.misc import box_corners_nd, box_corners_nd_jax
from envs.T_pushing.t_sim import get_t_comp_centers_w_com, gen_vertices_from_poses
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DT_Plan_Reach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2

def _gen_pose_list(num_test, seed, x_bound, y_bound, theta_bound=None, theta_factor=1):
    shift = 0
    num_test+=shift
    random.seed(seed)
    if theta_bound is None:
        return [np.array([random.randint(*x_bound), random.randint(*y_bound)]) for i in range(num_test)][shift:]
    return [
        np.array(
            [
                random.randint(*x_bound),
                random.randint(*y_bound),
                math.radians(random.randint(*theta_bound) * theta_factor),
            ]
        )
        for i in range(num_test)
    ][shift:]


def generate_test_cases(seed, num_test, test_id=0):
    shift = 0
    num_test += shift
    if test_id == 0:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (180, 200), (170, 190), None)
        init_pose_list = _gen_pose_list(num_test, seed, (240, 250), (130, 150), (30, 60))
        target_pose_list = _gen_pose_list(num_test, seed, (230, 250), (280, 300), (90, 120))
    elif test_id == 1:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (140, 160), (170, 190), None)
        init_pose_list = _gen_pose_list(num_test, seed, (180, 200), (130, 150), (90, 135))
        target_pose_list = _gen_pose_list(num_test, seed, (250, 250), (420, 420), (180, 180))
    elif test_id == 2:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (125, 125), (140, 140), None)
        init_pose_list = _gen_pose_list(num_test, seed, (180, 180), (140, 140), (90, 90))
        target_pose_list = _gen_pose_list(num_test, seed, (250, 250), (420, 420), (180, 180))
    elif test_id == 3:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (330, 340), (160 + 80, 170 + 80), None)
        init_pose_list = _gen_pose_list(num_test, seed, (340, 360), (130 + 80, 150 + 80), (180+45, 180+75))
        target_pose_list = _gen_pose_list(num_test, seed, (250, 250), (420, 420), (180, 180))
    elif test_id == 4:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (180, 200), (220, 240), None)
        init_pose_list = _gen_pose_list(num_test, seed, (180, 200), (180, 200), (60, 120))
        target_pose_list = _gen_pose_list(num_test, seed, (300, 320), (330, 350), (150, 210))
    elif test_id == 5:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (200, 220), (220, 240), None)
        init_pose_list = _gen_pose_list(num_test, seed, (180, 200), (240, 260), (100, 120))
        target_pose_list = _gen_pose_list(num_test, seed, (250, 250), (80, 80), (0, 0))
    elif test_id == 6:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (180, 200), (170, 190), None)
        init_pose_list = _gen_pose_list(num_test, seed, (240, 250), (130, 150), (30, 60))
        target_pose_list = _gen_pose_list(num_test, seed, (210, 230), (280, 300), (90, 120))
    elif test_id == -1:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (0, 0), (400, 400), None)
        init_pose_list = _gen_pose_list(num_test, seed, (130, 130), (400, 400), (90, 90))
        target_pose_list = _gen_pose_list(num_test, seed, (280, 280), (400, 400), (90, 90))
    # elif test_id == -2:
    #     init_pusher_pos_list = _gen_pose_list(num_test, seed, (125, 125), (140, 140), None)
    #     init_pose_list = _gen_pose_list(num_test, seed, (180, 180), (140, 140), (90, 90))
    #     target_pose_list = _gen_pose_list(num_test, seed, (250, 250), (420, 420), (180, 180))
    else:
        raise ValueError(f"Unknown test_id: {test_id}")
    init_pusher_pos_list = init_pusher_pos_list[shift:]
    init_pose_list = init_pose_list[shift:]
    target_pose_list = target_pose_list[shift:]
    return init_pusher_pos_list, init_pose_list, target_pose_list

def get_pusher_pos_seq(pusher_start_pos, act_seqs):
    n_sample, horizon, action_dim = act_seqs.shape
    pusher_pos_seqs = jnp.zeros((n_sample, horizon + 1, action_dim))

    # initialize first step
    pusher_pos_seqs = pusher_pos_seqs.at[:, 0, 0].add(pusher_start_pos[0])
    pusher_pos_seqs = pusher_pos_seqs.at[:, 0, 1].add(pusher_start_pos[1])
    def body_fn(carry, i):
        pos = carry
        next_pos = pos.at[:, i + 1, :].set(pos[:, i, :] + act_seqs[:, i, :])
        return next_pos, None

    pusher_pos_seqs, _ = jax.lax.scan(body_fn, pusher_pos_seqs, jnp.arange(horizon))
    return pusher_pos_seqs

def get_abs_states(state_seqs, pusher_start_pos, act_seqs, pred_mode="state"):
    pusher_pos_seqs = get_pusher_pos_seq(pusher_start_pos, act_seqs)
    if pred_mode == "pose":
        abs_state_seqs = state_seqs.at[:, :, 0:2].add(pusher_pos_seqs[:, 1:, 0:2])
    else:
        abs_state_seqs = state_seqs.at[:, :, ::2].add(pusher_pos_seqs[:, 1:, 0:1])
        abs_state_seqs = abs_state_seqs.at[:, :, 1::2].add(pusher_pos_seqs[:, 1:, 1:2])
    return abs_state_seqs, pusher_pos_seqs

def plot_cost_stat(cost_stat, out_name):
    cost_stat = np.asarray(cost_stat)
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
    plt.savefig(out_name)
    print(f"Step cost plot saved to {out_name}")
    plt.close()

    num_test, max_steps = cost_stat.shape
    data = [cost_stat[:, t] for t in range(max_steps)]
    xticks = np.arange(max_steps)
    xlabel = "Time step"
    plt.figure(figsize=(max(10, len(data) * 0.18), 5))
    plt.boxplot(
        data,
        positions=np.arange(len(data)),
        widths=0.6,
        showfliers=False,   # cleaner; set True if you want outliers
    )
    plt.xticks(np.arange(len(data)), xticks)
    plt.xlabel(xlabel)
    plt.ylabel("Step cost")
    plt.title("Step Cost Distribution over Time (Boxplot across test cases)")
    plt.grid(True, axis="y")
    plt.legend(loc="best")

    out_name = out_name.replace(".png", "_boxplot.png")
    plt.savefig(out_name, bbox_inches="tight", dpi=200)
    print(f"Boxplot saved to {out_name}")

    return

def make_rollout_and_reward_fns(
    dt_dyn,
    config: Dict,
    abs_pose: bool = True,
    pred_mode: str = "pose",
):
    planning_config = config["planning"]
    data_config = config["data"]
    horizon = planning_config["horizon"]
    cost_norm = planning_config["cost_norm"]
    only_final_cost = planning_config["only_final_cost"]
    reach_config = planning_config.get("reach_in_obj", {})
    refine_config = planning_config.get("refinement", {})
    reach_refine_config = refine_config.get("reach_in_obj", {})
    enable_reach = reach_config.get("enable", False) or (refine_config.get('enable', False) and reach_refine_config.get("enable", False))

    scale = data_config.get("scale", 1.0)

    if enable_reach:
        # reachability part
        def f_wrapper(x):
            state_next = dt_dyn(x)
            action_next = x[dt_dyn.Dx:]
            return jnp.concatenate([state_next, action_next], axis=-1)

        reach_analyzer = DT_Plan_Reach(f_wrapper, state_dim=dt_dyn.Dx, action_dim=dt_dyn.Du, nn_dyn=True, n_steps_per_plan=1, step_size=1)
        eps = reach_config.get("eps", 0.05)

        _calculate_volume = lambda lo, up: calculate_volume(lo, up, union_init=False, mode='sum')

    hole_config = planning_config.get("hole", {})
    enable_hole = hole_config.get("enable", False)
    if enable_hole:
        assert pred_mode == "pose", "Hole interaction only implemented for pose prediction mode."
        stem_size = data_config["stem_size"]
        bar_size = data_config["bar_size"]

        h_T = jnp.array([[stem_size[0] / 2, stem_size[1] / 2],
                         [bar_size[0] / 2, bar_size[1] / 2]]) / scale  # [2,2]
        c_s, c_b = get_t_comp_centers_w_com(stem_size, bar_size)
        c_T_ori = jnp.array([c_s, c_b])[None, :] / scale  # [1,2,2]

        hole_center = jnp.array(hole_config["center"])  # [2,]
        hole_size = jnp.array(hole_config["size"])  # [2,]
        c_wall, h_wall = hole_to_walls_aabbs(hole_center, hole_size, window_size=data_config["window_size"])
        c_wall = jnp.array(c_wall) / scale
        h_wall = jnp.array(h_wall) / scale

        enforce_reach_steps = hole_config.get("enforce_reach_steps", horizon) + 1
        # def _detect_interaction_set(r_lo, r_up):
        #     r_lo = r_lo[:enforce_reach_steps, :]  # [horizon, 3]
        #     r_up = r_up[:enforce_reach_steps, :]  # [horizon, 3]
        #     # r_lo, r_up: [horizon, 3]
        #     c_T_lo = c_T_ori + r_lo[..., None, 0:2] # [horizon, 2, 2]
        #     c_T_up = c_T_ori + r_up[..., None, 0:2] # [horizon, 2, 2]
        #     angle_T_lo = jnp.concatenate([r_lo[..., 2:3], r_lo[..., 2:3]+jnp.pi/2], axis=-1) # [horizon, 2]
        #     angle_T_up = jnp.concatenate([r_up[..., 2:3], r_up[..., 2:3]+jnp.pi/2], axis=-1) # [horizon, 2]
        #     interact, _ = detect_T_hole_interaction_set(c_wall, h_wall, c_T_lo, c_T_up, h_T, angle_T_lo, angle_T_up) # [horizon]
        #     return interact

        def _detect_interaction_set(r_lo, r_up):
            # r_lo, r_up: [horizon, 3]
            r_lo = r_lo[:enforce_reach_steps, :3]  # [horizon, 3]
            r_up = r_up[:enforce_reach_steps, :3]  # [horizon, 3]
            sample_states = box_corners_nd_jax(r_lo, r_up).reshape(-1, *r_lo.shape[1:])  # [n * horizon, 3]

            c_T = c_T_ori + sample_states[..., None, 0:2] # [n * horizon, 2, 2]
            angle_T = jnp.concatenate([sample_states[..., 2:3], sample_states[..., 2:3]+jnp.pi/2], axis=-1) # [n * horizon, 2]
            interact, margin = detect_T_hole_interaction(c_wall, h_wall, c_T, h_T, angle_T) # [n * horizon]
            interact = jnp.any(interact.reshape(-1, enforce_reach_steps), axis=0)  # [horizon]
            margin = margin.reshape(-1, enforce_reach_steps).max(axis=0)  # [horizon]
            return margin

        def _detect_interaction(state_seq):
            # state_seq: [horizon, 3]
            c_T = c_T_ori + state_seq[..., None, 0:2] # [horizon, 2, 2]
            angle_T = jnp.concatenate([state_seq[..., 2:3], state_seq[..., 2:3]+jnp.pi/2], axis=-1) # [horizon, 2]
            interact, margin = detect_T_hole_interaction(c_wall, h_wall, c_T, h_T, angle_T) # [horizon]
            return margin
        penalty_factor = hole_config.get("penalty", 2.0)
        pass

    # [state_dim], [n_sample, horizon, action_dim] -> [n_sample, horizon, state_dim]
    def rollout_fn(state_cur: jnp.ndarray, act_seqs: jnp.ndarray, reach_config: dict) -> jnp.ndarray:
        state_cur = state_cur[None].repeat(act_seqs.shape[0], axis=0) # [B, Dx]
        state_seqs = dt_dyn.rollout(state_cur, act_seqs)
        enable_reach = reach_config.get("enable", False)
        if enable_reach:
            reach_info = cal_reach(state_cur, act_seqs, splits_cfg = reach_config.get("splits", {}))
        return state_seqs, reach_info if enable_reach else {}

    def cal_reach(state_cur: jnp.ndarray, act_seqs: jnp.ndarray, splits_cfg: dict) -> jnp.ndarray:
        B = act_seqs.shape[0]
        Da = dt_dyn.Du
        T = act_seqs.shape[1]
        state_lo = state_cur[:, :dt_dyn.Ds] - eps
        state_up = state_cur[:, :dt_dyn.Ds] + eps

        if abs_pose:
            state_lo = jnp.concatenate([state_lo, state_cur[:, dt_dyn.Ds:dt_dyn.Dx]], axis=-1)
            state_up = jnp.concatenate([state_up, state_cur[:, dt_dyn.Ds:dt_dyn.Dx]], axis=-1)

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
    def reward_fn(state_seqs: jnp.ndarray, act_seqs: jnp.ndarray, reach_config: dict, reach_aux: Dict, target_state: jnp.ndarray, pusher_pos: jnp.ndarray) -> Dict:
        # state_seqs: [n_sample, horizon, state_dim]
        if abs_pose:
            abs_state_seqs = state_seqs[..., :-act_seqs.shape[-1]]
        else:
            abs_state_seqs, _ = get_abs_states(state_seqs, pusher_pos, act_seqs, pred_mode=pred_mode)

        cost_seqs = jnp.linalg.norm(abs_state_seqs - target_state[None, None, :], axis=-1, ord=cost_norm) ** cost_norm
        if only_final_cost:
            costs = cost_seqs[:, -1]
        else:
            step_weight = jnp.linspace(1, horizon + 1, horizon) / horizon
            costs = jnp.sum(cost_seqs * step_weight[None, :], axis=-1)
        reach_loss = 0.0
        enable_reach = reach_config.get("enable", False)
        if enable_reach:
            reach_loss = reach_config.get("weight", 1.0) * jnp.log1p(reach_aux["reach_vol"])
            if reach_config.get("clip", True):
                reach_loss = jnp.clip(reach_loss, a_max=reach_config.get("loss_max", 10.0))

            costs = costs + reach_loss  # penalize large reachable set
        collision_loss = 0.0
        if enable_hole:
            if enable_reach:
                margin = jax.vmap(_detect_interaction_set)(reach_aux["r_lo"], reach_aux["r_up"])  # [n_sample, horizon]
                collision_loss = jnp.sum(margin, axis=-1) * reach_config.get("collision_penalty", 1.0)
                costs = costs + collision_loss
            else:
                margin = jax.vmap(_detect_interaction)(abs_state_seqs)  # [n_sample, horizon]
                collision_loss = jnp.sum(margin, axis=-1) * penalty_factor
                costs = costs + collision_loss
        return {"rewards": -costs, "reward_seqs": -cost_seqs, "reach_aux": reach_aux, "reach_loss": reach_loss, "collision_loss": collision_loss}

    def step_cost_fn_np(state, target_state):
        # assume input are not scaled
        diff = target_state - state
        if pred_mode == "pose":
            diff[:2] = diff[:2] / scale
        else:
            diff[::2] = diff[::2] / scale
            diff[1::2] = diff[1::2] / scale
        return (np.linalg.norm(diff, cost_norm)) ** cost_norm

    def step_cost_fn(state, target_state):
        # assume input are not scaled
        diff = target_state - state
        if pred_mode == "pose":
            diff = diff.at[..., :2].divide(scale)
        else:
            diff = diff.at[..., ::2].divide(scale)
            diff = diff.at[..., 1::2].divide(scale)
        return (jnp.linalg.norm(diff, cost_norm)) ** cost_norm

    return rollout_fn, reward_fn, step_cost_fn, step_cost_fn_np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_planning_animation(polys_seqs, pusher_pos_seqs, target_polys=None, gt_polys_seqs=None, window_size=(500, 500), pusher_size=5, obs_dict={}, fps=10, add_edge=False, save_path="plan.gif"):
    """
    polys_seqs: List of arrays, each [samples, steps, horizon, vertices, 2]
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

        # Plot Obstacles
        if obs_dict and "obs_pos_list" in obs_dict and "obs_size_list" in obs_dict:
            for obs_pos, obs_size in zip(obs_dict["obs_pos_list"], obs_dict["obs_size_list"]):
                obs_pos = np.array(obs_pos, dtype=np.int32)
                obs_size = np.array(obs_size, dtype=np.int32)
                if obs_dict.get("obs_norm", 1) == 2:  # circle
                    ax.add_patch(plt.Circle((obs_pos[0], obs_pos[1]), obs_size, color='gray', alpha=0.7, zorder=0))
                else:  # rectangle
                    ax.add_patch(plt.Rectangle((obs_pos[0] - obs_size[0], obs_pos[1] - obs_size[1]), 2*obs_size[0], 2*obs_size[1], color='gray', alpha=0.7, zorder=0))

        # Plot Target Polygon
        # target_polys: [vertices, 2]
        if target_polys is not None:
            target_poly = plt.Polygon(target_polys, facecolor='black', edgecolor='none', alpha=0.5, zorder=0)
            ax.add_patch(target_poly)

        # Iterate through different polygon types (stem, bar, etc.)
        # 1. Plot Planned Horizon (Increasing transparency)
        for t in range(1, horizon + 1):
            for j in range(n_samples):
                vertices = polys_seqs[j, frame, t]
                alpha = 1.0 - (t / (horizon + 1)) # Fade out into the future
                
                poly = plt.Polygon(vertices, facecolor=colors[-t], 
                                edgecolor=colors[-t] if add_edge else 'none', alpha=alpha * 0.8, zorder=horizon + 1-t)
                ax.add_patch(poly)
            if gt_polys_seqs is not None:
                for j in range(gt_polys_seqs.shape[0]):
                    # 2. Plot GT State
                    gt_vertices = gt_polys_seqs[j, frame, t]
                    gt_poly = plt.Polygon(gt_vertices, 
                                        facecolor=colors[-t], 
                                edgecolor='black', alpha=alpha * 0.8, zorder=horizon + 1-t)
                    ax.add_patch(gt_poly)
            # 3. Plot Pusher Position
            pusher_pos = pusher_pos_seqs[frame, t]
            ax.add_patch(plt.Circle((pusher_pos[0], pusher_pos[1]), pusher_size, color='black', fill=True, alpha=alpha * 0.8, zorder=horizon + 1-t))

        for j in range(n_samples):
            # 2. Plot Current State (Highlighted)
            curr_vertices = polys_seqs[j, frame, 0]
            curr_poly = plt.Polygon(curr_vertices, 
                                    facecolor='darkred', alpha=0.9,
                                    edgecolor='darkred' if add_edge else 'none', linewidth=2, zorder=horizon + 1)
        
            ax.add_patch(curr_poly)

        if gt_polys_seqs is not None:
            for j in range(gt_polys_seqs.shape[0]):
                curr_gt_vertices = gt_polys_seqs[j, frame, 0]
                curr_gt_poly = plt.Polygon(curr_gt_vertices, 
                                        facecolor='darkred', alpha=0.9,
                                        edgecolor='black', linewidth=2, zorder=horizon + 1)
                ax.add_patch(curr_gt_poly)
        # 3. Plot Current Pusher Position
        curr_pusher_pos = pusher_pos_seqs[frame, 0]
        ax.add_patch(plt.Circle((curr_pusher_pos[0], curr_pusher_pos[1]), pusher_size, color='black', fill=True, zorder=horizon + 1))

    if n_sim_steps == 0:
        update(0)
        save_path = save_path.replace(".gif", ".png")
        plt.savefig(save_path)
    else:
        ani = FuncAnimation(fig, update, frames=n_sim_steps + 1, repeat=False)
        ani.save(save_path, writer=PillowWriter(fps=fps))
    print(f"Animation saved to {save_path}")
    plt.close()

def merge_t_shape(stem_vertices, bar_vertices):
    """
    Assumes vertices are ordered [BL, BR, TR, TL] 
    (Bottom-Left, Bottom-Right, Top-Right, Top-Left)
    """
    # Extract specific corners to create a single 8-point boundary
    # This logic depends on your specific vertex ordering!
    
    merged_poly = np.concatenate([
        stem_vertices[..., 0:1, :], # Stem Bottom Left
        stem_vertices[..., 1:2, :], # Stem Bottom Right
        stem_vertices[..., 2:3, :], # Stem Top Right
        bar_vertices[..., 1:2, :],  # Bar Bottom Right
        bar_vertices[..., 2:3, :],  # Bar Top Right
        bar_vertices[..., 3:4, :],  # Bar Top Left
        bar_vertices[..., 0:1, :],  # Bar Bottom Left
        stem_vertices[..., 3:4, :], # Stem Top Left
    ], axis=-2)
    return merged_poly

def plot_plan_from_poses(state_seqs, pusher_pos_seqs, target_pose=None, gt_state_seqs=None, stem_size=(30, 90), bar_size=(120, 30), window_size=(500, 500), obs_dict={}, fps=10, add_edge=False, save_path="plan.gif"):
    # state_seqs: (N, n_sim_steps+1, horizon+1, 3), pusher_pos_seqs: (n_sim_steps+1, horizon+1, 2), target_pose: (3,)
    # a list of arrays (N, n_sim_steps+1, horizon+1, n_vertices, 2). there are stem and bar polys with 4 vertices each
    polys_seqs = merge_t_shape(*np.array(gen_vertices_from_poses(stem_size, bar_size, state_seqs)))
    target_polys = merge_t_shape(*np.array(gen_vertices_from_poses(stem_size, bar_size, target_pose))) if target_pose is not None else None
    gt_polys_seqs = merge_t_shape(*np.array(gen_vertices_from_poses(stem_size, bar_size, gt_state_seqs))) if gt_state_seqs is not None else None
    plot_planning_animation(polys_seqs, pusher_pos_seqs, target_polys=target_polys, gt_polys_seqs=gt_polys_seqs, window_size=window_size, obs_dict=obs_dict, fps=fps, add_edge=add_edge, save_path=save_path)

if __name__ == "__main__":
    import pickle
    pkl_file = "output/planning/T_pushing/1_uniform_0.05_0.015_mppi_1000_True_False_True_True/221448_034342/planning_res_0000.pkl"
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # parameters
    stem_size = [30, 90]
    bar_size = [120, 30]
    window_size = [500, 500]
    scale = 100

    # obs_dict = {}
    obs_dict = {
        "obs_norm": 1,
        "obs_pos_list": [[110., 460.], [390., 460.]],
        "obs_size_list": [[110.,  40.], [110.,  40.]],
    }

    act_seqs = np.array([d["act_seq"] for d in data])
    state_seqs = np.array([d["state_seq"] for d in data])[..., :3]
    pusher_pos_seqs = np.array([d["pusher_pos_seq"] for d in data])

    # r_lo_seqs, r_up_seqs: (n_sim_steps+1, horizon+1, 3)
    r_lo_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_lo'] for d in data]).reshape((*state_seqs.shape[:2], -1))[..., :3]
    r_up_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_up'] for d in data]).reshape((*state_seqs.shape[:2], -1))[..., :3]

    r_lo_seqs[..., :2] = r_lo_seqs[..., :2] * scale
    r_up_seqs[..., :2] = r_up_seqs[..., :2] * scale

    # # sample from r_lo_seqs and r_up_seqs
    # # n_samples = 8
    # # sample_states = np.random.uniform(size=(n_samples, *state_seqs.shape))
    # # sample_state_seqs = r_lo_seqs[None] + sample_states * (r_up_seqs - r_lo_seqs)[None]

    # sample_state_seqs = box_corners_nd(r_lo_seqs, r_up_seqs)  # (8, n_sim_steps+1, horizon+1, 3)
    # n_samples = sample_state_seqs.shape[0]

    # target_pose = [250, 420, math.radians(180)]

    # fps = 2
    # save_path = pkl_file.replace(".pkl", ".gif")
    # plot_plan_from_poses(state_seqs[None], pusher_pos_seqs, target_pose=target_pose, stem_size=stem_size, bar_size=bar_size, window_size=window_size, obs_dict=obs_dict, fps=fps, save_path=save_path)
    # save_path = pkl_file.replace(".pkl", "_sample.gif")
    # plot_plan_from_poses(sample_state_seqs, pusher_pos_seqs, target_pose=target_pose, stem_size=stem_size, bar_size=bar_size, window_size=window_size, obs_dict=obs_dict, fps=fps, save_path=save_path)
    # pass


    config_path = "configs/T_pushing.yaml"
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    data_config = config["data"]
    planning_config = config["planning"]
    hole_config = planning_config.get("hole", {})

    hole_center = jnp.array(hole_config["center"])  # [2,]
    hole_size = jnp.array(hole_config["size"])  # [2,]
    c_wall, h_wall = hole_to_walls_aabbs(hole_center, hole_size, window_size=data_config["window_size"])
    c_wall = jnp.array(c_wall)
    h_wall = jnp.array(h_wall)
    stem_size = data_config["stem_size"]
    bar_size = data_config["bar_size"]
    scale = data_config.get("scale", 1.0)
    h_T = jnp.array([[stem_size[0] / 2, stem_size[1] / 2],
                        [bar_size[0] / 2, bar_size[1] / 2]])  # [2,2]
    c_s, c_b = get_t_comp_centers_w_com(stem_size, bar_size)
    c_T_ori = jnp.array([c_s, c_b])[None, :]  # [1,2,2]

    def _detect_interaction_set(r_lo, r_up):
        # state_seq: [horizon, 3]
        c_T_lo = c_T_ori + r_lo[..., None, 0:2] # [horizon, 2, 2]
        c_T_up = c_T_ori + r_up[..., None, 0:2] # [horizon, 2, 2]
        angle_T_lo = jnp.concatenate([r_lo[..., 2:3], r_lo[..., 2:3]+jnp.pi/2], axis=-1) # [horizon, 2]
        angle_T_up = jnp.concatenate([r_up[..., 2:3], r_up[..., 2:3]+jnp.pi/2], axis=-1) # [horizon, 2]
        interact, _ = detect_T_hole_interaction_set(c_wall, h_wall, c_T_lo, c_T_up, h_T, angle_T_lo, angle_T_up) # [horizon]
        return interact

    test_state = jnp.array([250, 410, jnp.pi])[None, None]
    eps = jnp.array([scale * 0.05, scale * 0.05, 0.2])[None, None]
    test_r_lo = test_state - eps
    test_r_up = test_state + eps

    interact_seqs = jax.vmap(_detect_interaction_set)(test_r_lo, test_r_up)  # (n_sim_steps+1, horizon+1, horizon)
    if interact_seqs.sum() > 0:
        print(interact_seqs)
        print("Interaction detected!")

    target_pose = [250, 420, math.radians(180)]
    save_path = pkl_file.replace(".pkl", "_test.gif")
    plot_plan_from_poses(jnp.stack([test_r_lo, test_r_up]), pusher_pos_seqs[0:1, 0:1], target_pose=target_pose, stem_size=stem_size, bar_size=bar_size, window_size=window_size, obs_dict=obs_dict, fps=1, save_path=save_path)

    # def _detect_interaction(state_seq):
    #     # state_seq: [horizon, 3]
    #     c_T = c_T_ori + state_seq[..., None, 0:2] # [horizon, 2, 2]
    #     angle_T = jnp.concatenate([state_seq[..., 2:3], state_seq[..., 2:3]+jnp.pi/2], axis=-1) # [horizon, 2]
    #     interact, _ = detect_T_hole_interaction(c_wall, h_wall, c_T, h_T, angle_T) # [horizon]
    #     return interact
    
    # # interact_seqs = jax.vmap(_detect_interaction)(state_seqs[-2:-1])  # (n_sim_steps+1, horizon+1, horizon)

    # test_state = jnp.array([236, 420, jnp.pi*1.001])[None, None]
    # interact_seqs = jax.vmap(_detect_interaction)(test_state)
    # save_path = pkl_file.replace(".pkl", "_test.gif")
    # plot_plan_from_poses(test_state[None], pusher_pos_seqs[0:1, 0:1], target_pose=target_pose, stem_size=stem_size, bar_size=bar_size, window_size=window_size, obs_dict=obs_dict, fps=fps, save_path=save_path)
    # if interact_seqs.sum() > 0:
    #     print(interact_seqs)
    #     print("Interaction detected!")