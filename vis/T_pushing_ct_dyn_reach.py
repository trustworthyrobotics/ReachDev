import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from io import BytesIO
import optax
import yaml
import os
import pickle
import jax
import jax.numpy as jnp
from jax import random as jrandom
import equinox as eqx
import time
import hydra
from omegaconf import DictConfig
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2

sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import CT_Plan_Reach
from CROWN_Reach.src.utils.vis import visualize_flowpipe_time
from models.mlp_utils import load_model
from models.ct_dyn import Continuous_T_Dynamics
from utils.T_pushing import pose_to_kp
from envs.T_pushing.t_sim import T_Sim
import numpy as np
from scipy.spatial import ConvexHull
import itertools

def generate_zonotopes(c, L, R_lo, R_up):
    """
    Generates vertices for N/2 Zonotopes.
    
    Args:
        c: [Batch, N] - Center/Constant offset
        L: [Batch, N, M] - Linear coefficients
        R_lo: [Batch, N] - Lower bound of remainder interval
        R_up: [Batch, N] - Upper bound of remainder interval
        M: int - Dimension of the input x (where x \in [-1, 1]^M)
        
    Returns:
        List of lists: [Batch][Keypoint_Idx] -> ndarray of vertices defining the hull
    """
    batch_size, N = c.shape
    num_keypoints = N // 2
    
    # 1. Generate all 2^M vertices of the unit hypercube
    # For M=8, this is 256 vertices.
    M = L.shape[-1]
    hypercube_vertices = np.array(list(itertools.product([-1, 1], repeat=M))) # [2^M, M]
    
    # 2. Calculate the Remainder "Box" center and radius
    # We treat the remainder R as an extra interval added to the zonotope
    r_center = (R_up + R_lo) / 2
    r_radius = (R_up - R_lo) / 2
    
    batch_zonotopes = []

    for b in range(batch_size):
        keypoint_hulls = []
        for k in range(num_keypoints):
            idx = k * 2
            # Extract 2xM linear mapping for this keypoint
            L_k = L[b, idx:idx+2, :] # [2, M]
            c_k = c[b, idx:idx+2]    # [2]
            
            # Map hypercube vertices to 2D: (L * x)
            # hypercube_vertices is [2^M, M], L_k.T is [M, 2]
            z_vertices = hypercube_vertices @ L_k.T # [2^M, 2]
            
            # Add the constant offset and remainder center
            offset = c_k + r_center[b, idx:idx+2]
            z_vertices += offset
            
            # To account for the remainder interval box (R), we effectively 
            # expand the zonotope by the Minkowski sum of the remainder box.
            # A simple way: add the 4 corners of the remainder box to the segments
            # or just expand the existing vertices by the radius.
            rk_rad = r_radius[b, idx:idx+2]
            expansion = np.array(list(itertools.product([-1, 1], repeat=2))) * rk_rad
            
            # Combine all possible points (Zonotope vertices + Remainder expansion)
            # This is technically the Minkowski sum of two zonotopes
            final_points = []
            for exp_v in expansion:
                final_points.append(z_vertices + exp_v)
            final_points = np.vstack(final_points)
            
            # Compute Convex Hull to get the tightest 2D polygon
            hull = ConvexHull(final_points)
            keypoint_hulls.append(final_points[hull.vertices])
            
        batch_zonotopes.append(keypoint_hulls)
        
    return batch_zonotopes

def plot(r_lo, r_up, trajs, pxy, scale, window_size, file_name, zonotopes=None):
    """
    Plots trajectories and reachability bounds for a T-shape object (4 keypoints).
    """
    # 1. Preprocessing (Denormalization and coordinate shifting)
    Dx = r_lo.shape[-1]
    # Repeat pxy [1, T+1, 2] to match [..., 8] for 4 keypoints
    pxy_rep = np.tile(pxy, (1, 1, Dx // pxy.shape[-1])) 
    
    r_lo = (r_lo + pxy_rep) * scale
    r_up = (r_up + pxy_rep) * scale
    trajs = (trajs + pxy_rep) * scale
    pxy_scaled = pxy[0] * scale # [T+1, 2]

    T = trajs.shape[1] - 1
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use rainbow colormap for time steps
    cmap = plt.get_cmap("gist_rainbow")
    order = np.array([0, 2, 1, 3])  # TL, TR, TC, B to draw top bar then stem
    # 2. Plot Trajectories and Bounds
    for t in range(T + 1):
        color = cmap(t / T)
        
        # Plot Reachable Set Bounds (Rectangles for each of the 4 keypoints)
        # r_lo/up shape: [N_boxes, T+1, 8]
        for b in range(r_lo.shape[0]):
            for i in range(4):  # 4 keypoints
                idx = i * 2
                width = r_up[b, t, idx] - r_lo[b, t, idx]
                height = r_up[b, t, idx+1] - r_lo[b, t, idx+1]
                
                rect = patches.Rectangle(
                    (r_lo[b, t, idx], r_lo[b, t, idx+1]),
                    width, height,
                    linewidth=0.5,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.1  # Semitransparent
                )
                ax.add_patch(rect)

        # Plot Sample Trajectories
        # trajs shape: [n_samples, T+1, 8]
        for s in range(trajs.shape[0]):
            tee_kp = trajs[s, t, :8]
            plt.plot(tee_kp[::2][order], tee_kp[1::2][order], c=color, alpha=0.5)

    # Plot Zonotopes if provided
    if zonotopes is not None:
        zonotopes = np.array(zonotopes) * scale + pxy_scaled[None, None, -1:, :]  # [B, 4, num_vertices, 2]
        for b in range(len(zonotopes)):
            for k in range(len(zonotopes[b])):
                zono_pts = zonotopes[b][k]
                ax.fill(zono_pts[:, 0], zono_pts[:, 1], color='blue', alpha=0.2, label='Zonotope' if (b==0 and k==0) else "")
                ax.plot(np.append(zono_pts[:, 0], zono_pts[0, 0]), 
                        np.append(zono_pts[:, 1], zono_pts[0, 1]), color='blue', linewidth=2)

    # 3. Plot Pusher Path
    ax.plot(pxy_scaled[:, 0], pxy_scaled[:, 1], color='black', 
            marker='o', markersize=4, label='Pusher', linewidth=2, zorder=10)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Reachability Analysis: {T} steps")
    
    ax.set_xlim(0, window_size)
    ax.set_ylim(0, window_size)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Plot saved to {file_name}")
    plt.close()

def plot_v2(trajs, pxy, scale, window_size, file_name):
    """
    Method 1: Footprint Sweep Visualization.
    Uses transform_fn([B, T, 3]) -> [B, T, 8] to draw T-shapes.
    """
    # 1. Preprocessing (Denormalization and coordinate shifting)
    Dx = trajs.shape[-1]
    # Repeat pxy [1, T+1, 2] to match [..., 8] for 4 keypoints
    pxy_rep = np.tile(pxy, (1, 1, Dx // pxy.shape[-1])) 
    
    trajs = (trajs + pxy_rep) * scale
    pxy_scaled = pxy[0] * scale # [T+1, 2]

    # 1. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("gist_rainbow")
    
    # r_lo shape is [1, T+1, 3] (assuming batch size 1 for the reach tube)
    B_reach, T_plus_1, _ = trajs.shape
    T = T_plus_1 - 1
    
    # Drawing order provided: TL, TC, TR, B (0, 2, 1, 3) to draw top bar then stem
    order = np.array([0, 2, 1, 3]) 

    # 5. Plotting
    for t in range(T + 1):
        color = cmap(t / T)
        
        # Plot 64 semitransparent T-shapes at this time step
        for s in range(trajs.shape[0]):
            tee_kp = trajs[s, t, :8]
            ax.plot(tee_kp[::2][order], tee_kp[1::2][order], 
                    color=color, alpha=0.5, linewidth=2)

    # Plot Pusher Path in black
    ax.plot(pxy_scaled[:, 0], pxy_scaled[:, 1], color='black', 
            marker='o', markersize=4, label='Pusher', linewidth=2, zorder=10)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim(0, window_size)
    ax.set_ylim(0, window_size)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"T-Shape Footprint Sweep ({T} steps)")
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Plot saved to {file_name}")
    plt.close()

# @hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
# def main(config: DictConfig):
def main():
    model_dir = "output/runs/T_pushing_ct_dyn/"
    model_dir = model_dir + "lr0.0025_mid_0.08_0.05_0.002_20260104_210809"
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    train_config = config["train_ct_dyn"] if "train_ct_dyn" in config else config["train"]
    data_dir = train_config.get("data_dirr", "output/data/T_pushing_freq10")
    # data_dir = "output/data/T_pushing_freq1_1_1"
    
    scale = float(data_config["scale"])  # data was normalized by /scale
    pred_mode = train_config.get("pred_mode", "state")
    stem_size = jnp.array(data_config["stem_size"])
    bar_size = jnp.array(data_config["bar_size"])
    window_size = data_config["window_size"] * data_config.get("enlarge_factor_for_gen", 1)
    state_dim = data_config["state_dim"]
    pose_dim = data_config.get("pose_dim", 3)
    action_dim = data_config["action_dim"]
    key = jrandom.PRNGKey(config["settings"]["seed"])

    use_eval = True
    if use_eval:
        eval_p_path = os.path.join(data_dir, "data_eval.p")
    else:
        eval_p_path = os.path.join(data_dir, "data.p")
    model = load_model(data_config, train_config, model_class=Continuous_T_Dynamics, model_dir=model_dir, mode="best")

    def f_wrapper(x):
        dx = model(x)
        du = jnp.zeros_like(x[model.Dx:])
        return jnp.concatenate([dx, du], axis=-1)
    frequency = float(data_config["ct_dyn"]["frequency"])
    reach_cfg = train_config["reach"]
    reach_analyzer = CT_Plan_Reach(f_wrapper, state_dim=model.Dx, action_dim=model.Du, nn_dyn=True, n_steps_per_plan=1, step_size=1/frequency, init_remainder=reach_cfg.get("init_remainder", 1e-1), frr_rounds=reach_cfg.get("frr_rounds", 5), frr_stop_ratio=reach_cfg.get("frr_stop_ratio", 0.95), sr_window_size=reach_cfg.get("sr_window_size", 100))

    # -----------------------------
    # 2) Load eval data
    #    Expect dict with keys: "obs": [B, seq_len, obs_dim],
    #                           "action": [B, seq_len, act_dim]
    # -----------------------------
    with open(eval_p_path, "rb") as f:
        eval_data = pickle.load(f)
    eps_arr = np.array(eval_data)  # [B, T, 15]

    T_reach = train_config["n_rollout_valid"]
    T_reach = 10

    # Everything inside file is normalized by /scale → denormalize for visualization
    eps_denorm = eps_arr.astype(np.float32)               # [B,T,15], unnormalized
    eps_norm = eps_denorm / scale                    # [B,T,15], normalized

    selected_eps_idx = 20
    if pred_mode == "state":
        state_init = jnp.array(eps_norm[selected_eps_idx, 0, :state_dim])[None]      # [1, Dx]
        act_state_dim = state_dim
    if pred_mode == "pose":
        state_init = jnp.array(eps_norm[selected_eps_idx, 0, state_dim:state_dim+pose_dim])[None]      # [1, Dx]
        state_init = state_init.at[0, -1].set(state_init[0, -1] * scale)  # denormalize angle
        act_state_dim = pose_dim
        def transform_fn(pose):
            B, T, D = pose.shape
            pose = pose.reshape(-1, D)
            kp = jax.vmap(pose_to_kp, in_axes=(0, None, None))(pose, stem_size/scale, bar_size/scale)
            return kp.reshape(B, T, -1)
    action_seq = jnp.array(eps_norm[selected_eps_idx, :T_reach, -action_dim:])[None]      # [1, T, Du]
    pusher_pos_seq = jnp.array(eps_norm[selected_eps_idx, :T_reach+1, state_dim+pose_dim:-action_dim])[None]  # [1, T+1, 2]

    reach_eps = float(train_config["reach"]["eps_final"])
    # reach_eps = 0.02
    state_init_lo = state_init - reach_eps
    state_init_up = state_init + reach_eps
    z_init_lo = jnp.concatenate([state_init_lo, jnp.zeros((1, action_dim))], axis=-1) # [1, Dx+Du]
    z_init_up = jnp.concatenate([state_init_up, jnp.zeros((1, action_dim))], axis=-1) # [1, Dx+Du]
    reach_splits = train_config["reach"].get("splits", None)
    n_split = 2 if pred_mode == "state" else 4
    reach_splits = {i: n_split for i in range(act_state_dim)}
    z_init_lo, z_init_up = prepare_initial_set_v2(z_init_lo, z_init_up, splits_cfg=reach_splits)

    print(f"action seq: {action_seq.tolist()}")
    # print(f"pusher pos seq: {pusher_pos_seq.tolist()}")
    enable_action_opt = False
    n_opt_steps = 100
    if enable_action_opt:
        # optimize the action sequence for tighter reachability
        lr_schedule = optax.constant_schedule(0.001)
        optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_schedule)
        )
        opt_state = optim.init(eqx.filter(action_seq, eqx.is_inexact_array))
        @eqx.filter_jit
        def reach_opt_step(act_seq, opt_state, key):
            def loss_fn(_act_seq):
                ts, r_lo, r_up, x_nexts_all, _ = reach_analyzer.verify_w_model(f_wrapper, z_init_lo, z_init_up, n_total_steps=T_reach, action_seq=_act_seq.repeat(z_init_up.shape[0]//_act_seq.shape[0], axis=0)[:, None])
                r_lo = r_lo.reshape(-1, T_reach + 1, act_state_dim+action_dim)[..., :act_state_dim]  # [B, T+1, Dx]
                r_up = r_up.reshape(-1, T_reach + 1, act_state_dim+action_dim)[..., :act_state_dim]  # [B, T+1, Dx]

                _vol = calculate_volume(r_lo, r_up, union_init=False, mode="sum")
                _loss = jnp.log(1 + _vol)
                return _loss, _vol
            (loss, vol), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(act_seq)
            updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(act_seq, eqx.is_inexact_array))
            act_seq = eqx.apply_updates(act_seq, updates)
            return act_seq, opt_state, loss, vol

        start_time = time.time()
        key, subkey = jrandom.split(key)
        _, opt_state, loss, vol = reach_opt_step(action_seq, opt_state, subkey)
        jax.block_until_ready(loss)
        end_time = time.time()
        print(f"compile time: {end_time - start_time:.4f} sec")

        start_time = time.time()
        for opt_i in range(n_opt_steps):
            key, subkey = jrandom.split(key)
            action_seq, opt_state, loss, vol = reach_opt_step(action_seq, opt_state, subkey)
            if (opt_i + 1) % 10 == 0:
                print(f"Action opt step {opt_i+1}/{n_opt_steps}, loss: {loss:.4f}, vol: {vol:.4f}")

        jax.block_until_ready(loss)
        end_time = time.time()
        print(f"Optimization time for {n_opt_steps} steps: {end_time - start_time:.4f} sec")

        init_pusher_pos = pusher_pos_seq[:, 0:1, :]
        pusher_pos_seq = init_pusher_pos + jnp.cumsum(action_seq, axis=1)
        pusher_pos_seq = jnp.concatenate([init_pusher_pos, pusher_pos_seq], axis=1)   # [1, T+1, 2]
        print(f"action seq: {action_seq.tolist()}")
        # print(f"pusher pos seq: {pusher_pos_seq.tolist()}")
    # perform reachability analysis
    ts, r_lo, r_up, x_nexts_all, _ = reach_analyzer.verify_w_model(f_wrapper, z_init_lo, z_init_up, n_total_steps=T_reach, action_seq=action_seq.repeat(z_init_up.shape[0]//action_seq.shape[0], axis=0)[:, None])
    r_lo = r_lo.reshape(-1, T_reach + 1, act_state_dim+action_dim)[..., :act_state_dim]  # [B, T+1, Dx]
    r_up = r_up.reshape(-1, T_reach + 1, act_state_dim+action_dim)[..., :act_state_dim]  # [B, T+1, Dx]

    # aggregate volume over all partitions
    r_lo_agg = jnp.min(r_lo, axis=0, keepdims=True)  # [1, T+1, Dx]
    r_up_agg = jnp.max(r_up, axis=0, keepdims=True)

    vol = float(calculate_volume(r_lo, r_up, union_init=False, mode="sum"))
    print(f"Reachable set volume over {T_reach} steps: {vol}")

    n_samples = 64 if z_init_lo.shape[0] == 1 else z_init_lo.shape[0]

    # sample_state_init = state_init_lo + (state_init_up - state_init_lo) * jrandom.uniform(key, shape=(n_samples, state_init_lo.shape[1])) # [n_samples, Dx]
    raw_samples = jrandom.uniform(key, shape=(z_init_lo.shape[0], max(n_samples // z_init_lo.shape[0], 1), act_state_dim)) # [n_partitions, n_per_partition, Dx]
    sample_state_init = z_init_lo[:, jnp.newaxis, :act_state_dim] + (z_init_up - z_init_lo)[:, jnp.newaxis, :act_state_dim] * raw_samples
    sample_state_init = sample_state_init.reshape(-1, act_state_dim)  # [n_samples, Dx]

    sample_rollout = model.rollout(sample_state_init, action_seq.repeat(n_samples, axis=0))
    sample_rollout = jnp.concatenate([sample_state_init[:, None, :], sample_rollout], axis=1)  # [n_samples, T+1, Dx]

    out_dir = os.path.join(model_dir, f"{selected_eps_idx}_reach_eps{reach_eps}_steps{T_reach}_{n_split}_{pred_mode}_{enable_action_opt}_{n_opt_steps}")
    os.makedirs(out_dir, exist_ok=True)

    # save r_lo, r_up, sample_rollout, pusher_pos_seq, scale, window_size
    arch_file_name = os.path.join(out_dir, "reach_data.npz")
    np.savez(arch_file_name, r_lo=np.array(r_lo_agg), r_up=np.array(r_up_agg), sample_rollout=np.array(sample_rollout), pusher_pos_seq=np.array(pusher_pos_seq), scale=scale, window_size=window_size)
    # exit()
    r_lo_agg, r_up_agg, sample_rollout, pusher_pos_seq, scale, window_size = np.load(arch_file_name).values()

    if pred_mode == "state":
        outfile = os.path.join(out_dir, f"reach_pushing.png")
        # zonotopes = generate_zonotopes(
        #     c=xF.P.c[:,:state_dim],          # [B, Dx]
        #     L=xF.P.L[:, :state_dim, 1:state_dim+1],          # [B, Dx, Dx]
        #     R_lo=xF.R.lo[:, :state_dim],    # [B, Dx]
        #     R_up=xF.R.hi[:, :state_dim]     # [B, Dx]
        # )

        plot(r_lo_agg, r_up_agg, sample_rollout, pusher_pos_seq, scale, window_size, outfile)
    elif pred_mode == "pose":
        sample_r = r_lo_agg + (r_up_agg - r_lo_agg) * np.random.uniform(size=(n_samples, *r_lo_agg.shape[1:]))
        outfile = os.path.join(out_dir, f"reach.png")
        plot_v2(np.array(transform_fn(jnp.array(sample_r))), pusher_pos_seq, scale, window_size, outfile)
        outfile = os.path.join(out_dir, f"sample.png")
        plot_v2(np.array(transform_fn(jnp.array(sample_rollout))), pusher_pos_seq, scale, window_size, outfile)

        param_dict = {"stem_size": data_config["stem_size"], 
                    "bar_size": data_config["bar_size"], 
                    "pusher_size": data_config["pusher_size"],
                    "save_img": True,
                    "enable_vis": False,
                    "window_size": data_config["window_size"],}

        sample_env = []
        sample_state_init = np.array(sample_state_init) * scale
        sample_state_init[:, -1] = sample_state_init[:, -1] / scale  # angle back to normalized
        pusher_pos_seq_denorm = np.array(pusher_pos_seq) * scale
        for i in range(n_samples):
            init_pose = sample_state_init[i, :pose_dim]
            pusher_pos = pusher_pos_seq_denorm[0, 0, :]
            init_pose[:2] = init_pose[:2] + pusher_pos[:2]
            env = T_Sim(param_dict=param_dict, init_poses=[init_pose], pusher_pos=pusher_pos)
            env_output = []
            
            for j in range(2):
                env_dict = env.update((pusher_pos[0], pusher_pos[1]), rel=True)
            env_output.append(np.concatenate([env_dict["com_pos"] / scale, env_dict["angle"]], axis=0))
            for j in range(T_reach):
                pusher_pos = pusher_pos_seq_denorm[0, j+1, :]
                env_dict = env.update((pusher_pos[0], pusher_pos[1]), rel=True)
                env_output.append(np.concatenate([env_dict["com_pos"] / scale, env_dict["angle"]], axis=0))
            sample_env.append(np.array(env_output))

        outfile = os.path.join(out_dir, f"env.png")
        plot_v2(np.array(transform_fn(jnp.array(sample_env))), pusher_pos_seq, scale, window_size, outfile)

        # norm_samples = (raw_samples * 2 - 1).reshape(-1, act_state_dim)  # [n_partitions * n_per_partition (1), Dx], in [-1, 1]
        # norm_lo = norm_up = jnp.concatenate([jnp.zeros((norm_samples.shape[0], 1)), norm_samples, jnp.zeros((norm_samples.shape[0], action_dim))], axis=-1).repeat(T_reach+1, axis=0)  # [n_samples*(T_reach+1), 1+Dx+Du]

        # taylor_range = x_nexts_all.eval_interval(norm_lo, norm_up)
        # taylor_lo, taylor_up = taylor_range.lo[:, :act_state_dim], taylor_range.hi[:, :act_state_dim]
        # taylor_lo = taylor_lo.reshape(-1, T_reach+1, act_state_dim)
        # taylor_up = taylor_up.reshape(-1, T_reach+1, act_state_dim)
        # print(f"max diff:{np.max(taylor_up - taylor_lo, axis=(0))}")

        # outfile = os.path.join(out_dir, f"reach_sample.png")
        # plot_v2(np.array(transform_fn((taylor_lo+taylor_up)/2)), pusher_pos_seq, scale, window_size, outfile)

    for idx in range(act_state_dim):
        outfile = os.path.join(out_dir, f"reach_{idx}.png")
        visualize_flowpipe_time(
            times=ts,
            lowers=r_lo,
            uppers=r_up,
            trajs=sample_rollout,
            state_idx=idx,
            file_name=outfile,
            print_boxes=False,
            draw_boxes=True,
            aggregate_partitions=True,
            stride=1,
            draw_traj=True,
        )


if __name__ == "__main__":
    main()