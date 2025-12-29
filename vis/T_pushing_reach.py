import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from io import BytesIO
import yaml
import os
import pickle
import jax.numpy as jnp
from jax import random as jrandom
import equinox as eqx
import hydra
from omegaconf import DictConfig
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DTPlanReach
from CROWN_Reach.src.utils.vis import visualize_flowpipe_time
from models.dynamics import load_t_dynamics_model


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


# @hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
# def main(config: DictConfig):
def main():
    model_dir = "output/runs/T_pushing/"
    model_dir = model_dir + "log_cos_128_mid_1_0.6_eps0.08_0.05_w0.002_j0.0_True_20251228_191734"
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_dir = config["data"]["out_path"]
    scale = float(config["data"].get("scale", 1.0))  # data was normalized by /scale

    eval_p_path = os.path.join(data_dir, "data.p")
    # model_path = os.path.join( model_dir, "last_model.eqx")
    model_path = os.path.join( model_dir, "best_model.eqx")
    model = load_t_dynamics_model(config=config, model_path=model_path)

    def f_wrapper(x):
        state_next = model(x)
        action_next = x[model.Dx:]
        return jnp.concatenate([state_next, action_next], axis=-1)

    reach_analyzer = DTPlanReach(f_wrapper, state_dim=model.Dx, action_dim=model.Du, nn_dyn=True, n_steps_per_plan=1, step_size=1)

    # -----------------------------
    # 2) Load eval data
    #    Expect dict with keys: "obs": [B, seq_len, obs_dim],
    #                           "action": [B, seq_len, act_dim]
    # -----------------------------
    with open(eval_p_path, "rb") as f:
        eval_data = pickle.load(f)
    eps_arr = np.array(eval_data)  # [B, T, 12]

    B, T, _ = eps_arr.shape
    state_dim = model.Dx
    action_dim = model.Du
    scale = float(config["data"]["scale"])
    T_reach = config["train"]["n_rollout_valid"]
    T_reach = 10
    window_size = config["data"]["window_size"]

    # Everything inside file is normalized by /scale → denormalize for visualization
    eps_denorm = eps_arr.astype(np.float32)               # [B,T,12], unnormalized
    eps_norm = eps_denorm / scale                    # [B,T,12], normalized

    selected_eps_idx = 0
    state_init = jnp.array(eps_norm[selected_eps_idx, 0, :state_dim])[None]      # [1, Dx]
    action_seq = jnp.array(eps_norm[selected_eps_idx, :T_reach, -action_dim:])[None]      # [1, T, Du]

    reach_eps = float(config["train"]["reach"]["eps_final"])
    reach_eps = 0.02
    state_init_lo = state_init - reach_eps
    state_init_up = state_init + reach_eps
    z_init_lo = jnp.concatenate([state_init_lo, jnp.zeros((1, action_dim))], axis=-1) # [1, Dx+Du]
    z_init_up = jnp.concatenate([state_init_up, jnp.zeros((1, action_dim))], axis=-1) # [1, Dx+Du]
    reach_splits = config["train"]["reach"].get("splits", None)
    # reach_splits = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
    z_init_lo, z_init_up = prepare_initial_set_v2(z_init_lo, z_init_up, splits_cfg=reach_splits)

    ts, r_lo, r_up, xF, _ = reach_analyzer.verify(z_init_lo, z_init_up, n_total_steps=T_reach, action_seq=action_seq.repeat(z_init_up.shape[0]//action_seq.shape[0], axis=0)[:, None])
    r_lo = r_lo.reshape(-1, T_reach + 1, state_dim+action_dim)[..., :state_dim]  # [B, T+1, Dx]
    r_up = r_up.reshape(-1, T_reach + 1, state_dim+action_dim)[..., :state_dim]  # [B, T+1, Dx]

    # aggregate volume over all partitions
    r_lo = jnp.min(r_lo, axis=0, keepdims=True)  # [1, T+1, Dx]
    r_up = jnp.max(r_up, axis=0, keepdims=True)

    vol = float(calculate_volume(r_lo, r_up, union_init=False, mode="sum"))
    print(f"Reachable set volume over {T_reach} steps: {vol}")

    key = jrandom.PRNGKey(config["settings"]["seed"])
    n_samples = 64
    sample_state_init = state_init_lo + (state_init_up - state_init_lo) * jrandom.uniform(key, shape=(n_samples, state_init_lo.shape[1]))
    sample_rollout = model.rollout_model(sample_state_init, action_seq.repeat(n_samples, axis=0))
    sample_rollout = jnp.concatenate([sample_state_init[:, None, :], sample_rollout], axis=1)  # [n_samples, T+1, Dx]

    pxy = eps_norm[selected_eps_idx, :T_reach+1, state_dim:state_dim+2][None]  # [1, T+1, 2]

    out_dir = os.path.join("output", "vis", "T_pushing")
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"reach_pushing.pdf")

    # save r_lo, r_up, sample_rollout, pxy, scale, window_size
    arch_file_name = os.path.join(out_dir, "reach_data.npz")
    np.savez(arch_file_name, r_lo=np.array(r_lo), r_up=np.array(r_up), sample_rollout=np.array(sample_rollout), pxy=np.array(pxy), scale=scale, window_size=window_size)
    # exit()
    r_lo, r_up, sample_rollout, pxy, scale, window_size = np.load(arch_file_name).values()

    # zonotopes = generate_zonotopes(
    #     c=xF.P.c[:,:state_dim],          # [B, Dx]
    #     L=xF.P.L[:, :state_dim, 1:state_dim+1],          # [B, Dx, Dx]
    #     R_lo=xF.R.lo[:, :state_dim],    # [B, Dx]
    #     R_up=xF.R.hi[:, :state_dim]     # [B, Dx]
    # )

    plot(r_lo, r_up, sample_rollout, pxy, scale, window_size, outfile)

    for idx in range(state_dim):
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