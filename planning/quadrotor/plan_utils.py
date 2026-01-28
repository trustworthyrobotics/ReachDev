import random
import math
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update('jax_platforms', 'cpu')
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
    elif test_id == 2:
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

def plot_cost_stat(cost_stat, out_name):
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
    return

def make_rollout_and_reward_fns(
    dt_dyn,
    planning_config: Dict,
    reach_config: Dict = {},
):
    horizon = planning_config["horizon"]
    cost_norm = planning_config["cost_norm"]
    only_final_cost = planning_config["only_final_cost"]
    reach_config = planning_config.get("reach_in_obj", {})
    refine_config = planning_config.get("refinement", {})
    reach_refine_config = refine_config.get("reach_in_obj", {})
    enable_reach = reach_config.get("enable", False) or (refine_config.get('enable', False) and reach_refine_config.get("enable", False))

    if enable_reach:
        # reachability part
        def f_wrapper(x):
            state_next = dt_dyn(x)
            action_next = x[dt_dyn.Dx:]
            return jnp.concatenate([state_next, action_next], axis=-1)


        reach_analyzer = DT_Plan_Reach(f_wrapper, state_dim=dt_dyn.Dx, action_dim=dt_dyn.Du, nn_dyn=True, n_steps_per_plan=1, step_size=1)
        eps = reach_config.get("eps", 0.05)
        splits_cfg = reach_config.get("refine_splits", {})

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
    def rollout_fn(state_cur: jnp.ndarray, act_seqs: jnp.ndarray, reach_config: dict) -> jnp.ndarray:
        state_cur = state_cur[None].repeat(act_seqs.shape[0], axis=0) # [B, Dx]
        state_seqs = dt_dyn.rollout(state_cur, act_seqs)
        enable_reach = reach_config.get("enable", False)
        if enable_reach:
            reach_info = cal_reach(state_cur, act_seqs)
        return state_seqs, reach_info if enable_reach else {}

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
    def reward_fn(state_seqs: jnp.ndarray, act_seqs: jnp.ndarray, reach_config: dict, reach_aux: Dict, target_state: jnp.ndarray) -> Dict:
        cost_seqs = jnp.linalg.norm(state_seqs - target_state[None, None, :], axis=-1, ord=cost_norm) ** cost_norm
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

def plot_planning_animation(pose_seqs, dt, out_path, targets=None, gt_state_seqs=None, r_lo_seqs=None, r_up_seqs=None, obs_config=None, fps=10):
    """
    Save a GIF visualizing re-planned 3D trajectories over simulation steps.

    Args:
        pose_seqs: (num_quads, n_steps, horizon, 3)
            pose_seqs[q, k, t] is the planned position for quad q,
            at simulation step k, at horizon index t (t=0 is current pose).
        num_quads: int
        dt: float (seconds per sim step, used for labeling; GIF fps is chosen automatically)
        out_path: str, e.g. "plan.gif"
        targets: optional (num_quads, 3)
        obs_config: optional dict:
            {
              "enable": bool,
              "pos_list": [ (3,), ... ],
              "size_list": [ float, ... ],
              "norm": 1 or 2,   # 1=cube (L1 ball), else sphere (L2 ball)
            }
    """
    pose_seqs = np.asarray(pose_seqs)

    num_quads, n_steps, horizon, _ = pose_seqs.shape
    assert horizon >= 2, "horizon should be >= 2 to visualize future plan"

    # # Choose a reasonable GIF fps from dt (clamped).
    # fps = int(np.clip(round(1.0 / max(dt, 1e-6)), 6, 20))

    # Precompute global axis limits to avoid flicker.
    pts = pose_seqs.reshape(-1, 3)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    extra_pts = []
    if targets is not None:
        targets = np.asarray(targets)
        extra_pts.append(targets.reshape(-1, 3))
    if obs_config is not None and obs_config.get("enable", False):
        pos_list = obs_config.get("pos_list", [])
        size_list = obs_config.get("size_list", [])
        norm = obs_config.get("norm", 2)
        for p, s in zip(pos_list, size_list):
            p = np.asarray(p).reshape(3)
            s = float(s)
            if norm == 1:
                # cube of side length s centered at p
                corners = np.array([[dx, dy, dz] for dx in (-s/2, s/2)
                                              for dy in (-s/2, s/2)
                                              for dz in (-s/2, s/2)], dtype=float) + p[None, :]
                extra_pts.append(corners)
            else:
                # sphere radius s centered at p -> bounding box is enough for limits
                bb = np.array([
                    p + np.array([ s, 0, 0]), p + np.array([-s, 0, 0]),
                    p + np.array([0,  s, 0]), p + np.array([0, -s, 0]),
                    p + np.array([0, 0,  s]), p + np.array([0, 0, -s]),
                ], dtype=float)
                extra_pts.append(bb)

    if extra_pts:
        extra_pts = np.concatenate(extra_pts, axis=0)
        xs = np.concatenate([xs, extra_pts[:, 0]])
        ys = np.concatenate([ys, extra_pts[:, 1]])
        zs = np.concatenate([zs, extra_pts[:, 2]])

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    # Padding
    def pad(lo, hi):
        span = max(hi - lo, 1e-3)
        p = 0.08 * span
        return lo - p, hi + p

    x_min, x_max = pad(x_min, x_max)
    y_min, y_max = pad(y_min, y_max)
    z_min, z_max = pad(z_min, z_max)

    # Colormap along horizon (t=1..horizon-1)
    horizon_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, horizon))[::-1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def _draw_obstacles(ax_):
        if obs_config is None or not obs_config.get("enable", False):
            return
        pos_list = obs_config.get("pos_list", [])
        size_list = obs_config.get("size_list", [])
        norm = obs_config.get("norm", 2)

        for obs_pos, obs_size in zip(pos_list, size_list):
            obs_pos = np.asarray(obs_pos).reshape(3)
            obs_size = float(obs_size)
            if norm == 1:
                # cube centered at obs_pos with side length obs_size
                r = [-obs_size / 2.0, obs_size / 2.0]
                X, Y = np.meshgrid(r, r)
                # 6 faces
                ax_.plot_surface(X + obs_pos[0], Y + obs_pos[1], np.full_like(X, obs_pos[2] - obs_size/2),
                                 alpha=0.25, color="gray")
                ax_.plot_surface(X + obs_pos[0], Y + obs_pos[1], np.full_like(X, obs_pos[2] + obs_size/2),
                                 alpha=0.25, color="gray")
                ax_.plot_surface(X + obs_pos[0], np.full_like(X, obs_pos[1] - obs_size/2), Y + obs_pos[2],
                                 alpha=0.25, color="gray")
                ax_.plot_surface(X + obs_pos[0], np.full_like(X, obs_pos[1] + obs_size/2), Y + obs_pos[2],
                                 alpha=0.25, color="gray")
                ax_.plot_surface(np.full_like(X, obs_pos[0] - obs_size/2), X + obs_pos[1], Y + obs_pos[2],
                                 alpha=0.25, color="gray")
                ax_.plot_surface(np.full_like(X, obs_pos[0] + obs_size/2), X + obs_pos[1], Y + obs_pos[2],
                                 alpha=0.25, color="gray")
            else:
                # sphere radius obs_size
                u, v = np.mgrid[0:2*np.pi:22j, 0:np.pi:12j]
                x = obs_size * np.cos(u) * np.sin(v) + obs_pos[0]
                y = obs_size * np.sin(u) * np.sin(v) + obs_pos[1]
                z = obs_size * np.cos(v) + obs_pos[2]
                ax_.plot_surface(x, y, z, alpha=0.25, color="gray")

    def update(k):
        ax.clear()

        # fixed bounds/view
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=25, azim=-60)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Planning @ step {k}/{n_steps-1}   (t = {k*dt:.2f}s)")

        _draw_obstacles(ax)

        if targets is not None:
            for q_id in range(targets.shape[0]):
                tx, ty, tz = targets[q_id]
                ax.scatter(tx, ty, tz, marker="X", s=90, label=f"Target {q_id}", depthshade=True)

        # Plot each quad: history (executed), current, and planned horizon
        for q_id in range(num_quads):
            # executed/history uses t=0 from each step as the realized state snapshot
            # hist = pose_seqs[q_id, :k+1, 0, :]  # (k+1, 3)
            # ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], linewidth=2, alpha=0.5)

            cur = pose_seqs[q_id, k, 0, :]
            ax.scatter(cur[0], cur[1], cur[2], s=60, depthshade=True, color='darkred')
            if r_lo_seqs is not None and r_up_seqs is not None:
                # draw reachable set at current step as AABB
                lo = r_lo_seqs[q_id, k, 0]   # (3,)
                up = r_up_seqs[q_id, k, 0]   # (3,)
                draw_aabb_surfaces(ax, lo, up, face_color='darkred', face_alpha=0.5)

            plan = pose_seqs[q_id, k, :, :]  # (horizon, 3)

            # Colored planned segments along horizon (like your rainbow time plot)
            points = plan.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (horizon-1, 2, 3)
            lc = Line3DCollection(segments, cmap="rainbow_r", linewidth=4, alpha=1.0)
            lc.set_array(np.linspace(0.0, 1.0, horizon - 1))
            ax.add_collection3d(lc)

            # Fading planned points into the future
            for t in range(1, horizon):
                p = plan[t]
                alpha = 1.0 - (t / horizon)

                if r_lo_seqs is not None and r_up_seqs is not None:
                    reach_alpha = 0.4 * alpha
                    reach_color = horizon_colors[t - 1]

                    # r_lo_seqs, r_up_seqs: (num_quads, n_steps, horizon, 3)
                    lo = r_lo_seqs[q_id, k, t]   # (3,)
                    up = r_up_seqs[q_id, k, t]   # (3,)

                    # draw AABB reachable box as 12 edges
                    draw_aabb_surfaces(ax, lo, up, face_color=reach_color, face_alpha=reach_alpha)
                # else:
                #     ax.scatter(p[0], p[1], p[2], s=20, alpha=0.8 * alpha, depthshade=False, color=horizon_colors[t - 1])

        if gt_state_seqs is not None:
            for q_id in range(gt_state_seqs.shape[0]):
                # executed/history uses t=0 from each step as the realized state snapshot
                # hist = pose_seqs[q_id, :k+1, 0, :]  # (k+1, 3)
                # ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], linewidth=2, alpha=0.5)

                cur = gt_state_seqs[q_id, k, 0, :]
                ax.scatter(cur[0], cur[1], cur[2], s=60, depthshade=True, color='darkred')

                plan = gt_state_seqs[q_id, k, :, :]  # (horizon, 3)

                # Colored planned segments along horizon (like your rainbow time plot)
                points = plan.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (horizon-1, 2, 3)
                lc = Line3DCollection(segments, cmap="rainbow_r", linewidth=4, alpha=1.0,)
                lc.set_array(np.linspace(0.0, 1.0, horizon - 1))
                ax.add_collection3d(lc)

        # # Avoid an overcrowded legend if many quads
        # if targets is not None and num_quads <= 6:
        #     ax.legend(loc="upper left")

        return []
    if n_steps == 1:
        update(0)
        out_path = out_path.replace(".gif", "_single_frame.png")
        plt.savefig(out_path)
    else:
        ani = FuncAnimation(fig, update, frames=n_steps, repeat=False)
        ani.save(out_path, writer=PillowWriter(fps=fps))
    print(f"Planning animation saved to {out_path}")
    plt.close(fig)

def draw_aabb_surfaces(ax, lo, hi, face_color, face_alpha):
    x0, y0, z0 = lo
    x1, y1, z1 = hi

    # 8 corners
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])

    # 6 faces (each is 4 corners)
    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom (z0)
        [v[4], v[5], v[6], v[7]],  # top    (z1)
        [v[0], v[1], v[5], v[4]],  # y=y0
        [v[2], v[3], v[7], v[6]],  # y=y1
        [v[1], v[2], v[6], v[5]],  # x=x1
        [v[3], v[0], v[4], v[7]],  # x=x0
    ]

    poly = Poly3DCollection(
        faces,
        facecolors=[face_color] * 6,
        edgecolors="none",
        alpha=face_alpha,
    )
    ax.add_collection3d(poly)

if __name__ == "__main__":
    import pickle
    # pkl_file = "output/planning/quadrotor/0_uniform_0.2_0.01_mppi_1000/mlp_214411_152209/planning_res_0000.pkl"
    pkl_file = "output/planning/quadrotor/0_uniform_0.2_0.01_mppi_1000/mlp_235110_152209/planning_res_0000.pkl"
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # parameters
    scale = 1
    dt = 0.2
    obs_config = {
        "enable": True,
        "pos_list": [ [5.0, 5.0, 5.0] ],
        "size_list": [2.0],
        "norm": 2,
    }

    state_dim = 3
    act_seqs = np.array([d["act_seq"] for d in data])
    state_seqs = np.array([d["state_seq"] for d in data])[..., :state_dim]

    # r_lo_seqs, r_up_seqs: (n_sim_steps+1, horizon+1, 3)
    r_lo_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_lo'] for d in data]).reshape((*state_seqs.shape[:2], -1))[..., :state_dim]
    r_up_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_up'] for d in data]).reshape((*state_seqs.shape[:2], -1))[..., :state_dim]

    # sample from r_lo_seqs and r_up_seqs
    n_samples = 8
    sample_states = np.random.uniform(size=(n_samples, *state_seqs.shape))
    sample_state_seqs = r_lo_seqs[None] + sample_states * (r_up_seqs - r_lo_seqs)[None]

    state_seqs = state_seqs[None]
    # state_seqs = state_seqs[:, :11]
    # sample_state_seqs = sample_state_seqs[:, :11]
    targets = np.array([11.0, 11.0, 9.0])[None]
    # plot_planning_animation(state_seqs, dt, "output/plan.gif", targets=targets, obs_config=obs_config)
    # plot_planning_animation(sample_state_seqs, dt, "output/plan_sample.gif", targets=targets, obs_config=obs_config)
    plot_planning_animation(state_seqs, dt, "output/plan_reach.gif", targets=targets, r_lo_seqs=r_lo_seqs[None], r_up_seqs=r_up_seqs[None], obs_config=obs_config)

    pass
