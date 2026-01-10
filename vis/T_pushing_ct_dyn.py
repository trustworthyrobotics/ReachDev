import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from io import BytesIO
import os
import pickle
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import hydra
from omegaconf import DictConfig
import yaml

from models.mlp_utils import load_model
from models.ct_dyn import Continuous_T_Dynamics
from utils.T_pushing import pose_to_kp

def plot_Tee(tee_kp, c="orange", label=""):
    """tee_kp: np.array([TL.x,TL.y,TC.x,TC.y,TR.x,TR.y,B.x,B.y])"""
    order = np.array([0, 2, 1, 3])  # TL, TR, TC, B to draw top bar then stem
    plt.plot(tee_kp[::2][order], tee_kp[1::2][order], c=c, alpha=1, label=label, linewidth=4)

def plot_agent(agent_kp, c="blue", label=""):
    """agent_kp: np.array([agent.x,agent.y])"""
    plt.scatter(*agent_kp, s=30, c=c, label=label)

def plot_frame(video_path, gt, pred, B, fps=10, xlim=(-1, 1), ylim=(-1, 1)):
    """
    Save a GIF comparing GT vs Pred for one episode index B.

    Args:
        video_path: path ending with .gif
        gt, pred: arrays shaped [Batch, T, 10? or >=10]
            expecting: first 8 entries are T keypoints (flattened), last 2 are pusher xy
            i.e., [TC.x,TC.y,TL.x,TL.y,TR.x,TR.y,B.x,B.y, px, py]
        action: [Batch, T, 2] (optional trajectory to plot; pass a dummy if unused)
        B: batch index to visualize
        fps: frames per second for the GIF
        xlim, ylim: axis limits
    """
    T = gt.shape[1]
    frames = []

    for i in range(T):
        plt.clf()
        # Tee shapes
        plot_Tee(gt[B, i, :8],  c="green",  label="GT")
        plot_Tee(pred[B, i, :8], c="orange", label="Pred")

        # Pusher positions
        plot_agent(gt[B, i, 8:],       c="black",  label="GT Pusher")

        plt.legend()
        plt.gca().set_aspect('equal')
        plt.xlim(*xlim)
        plt.ylim(*ylim)

        # Render current figure into an image array
        buf = BytesIO()
        plt.gcf().canvas.print_figure(buf, format='png', dpi=120)
        buf.seek(0)
        frame = iio.imread(buf)
        frames.append(frame)
        buf.close()

    # Write GIF in one shot
    iio.mimsave(video_path, frames, fps=fps, loop=0)
    print(f"Saved GIF to {video_path}")

def rel_to_abs_kp_plus_pusher(eps_denorm: np.ndarray) -> np.ndarray:
    """
    eps_denorm: [B, T, 12] denormalized by *scale
      [:, :, 0:8]  -> relative keypoints (4*(x,y))
      [:, :, 8:10] -> pusher (x_p, y_p)
      [:, :, 10:12]-> pusher velocity (unused for abs conversion)
    Returns:
      arr_vis: [B, T, 10] = [abs_kp(8), pusher_xy(2)]
    """
    B, T, D = eps_denorm.shape
    assert D == 12
    rel = eps_denorm[:, :, 0:8]              # [B,T,8]
    pxy = eps_denorm[:, :, 8:10]             # [B,T,2]
    rel_xy = rel.reshape(B, T, 4, 2)            # [B,T,4,2]
    abs_xy = rel_xy + pxy[:, :, None, :]        # broadcast add
    abs_flat = abs_xy.reshape(B, T, 8)          # back to [B,T,8]
    vis = np.concatenate([abs_flat, pxy], axis=-1)  # [B,T,10]
    return vis


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
def main(config: DictConfig):
    model_dir = config["test_models"]["ct_dyn_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    # override config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    train_cfg = config["train_ct_dyn"] if "train_ct_dyn" in config else config["train"]
    data_dir = train_cfg.get("data_dir", "output/data/T_pushing_freq10")
    model_dir = train_cfg["out_dir"]
    scale = float(data_cfg.get("scale", 1.0))  # data was normalized by /scale
    pred_mode = train_cfg.get("pred_mode", "state")
    stem_size = jnp.array(data_cfg["stem_size"])
    bar_size = jnp.array(data_cfg["bar_size"])
    state_dim = data_cfg["state_dim"]
    pose_dim = data_cfg.get("pose_dim", 3)
    action_dim = data_cfg["action_dim"]
    abs_pose = train_cfg.get("abs_pose", False)

    use_eval = True
    if use_eval:
        eval_p_path = os.path.join(data_dir, "data_eval.p")
    else:
        eval_p_path = os.path.join(data_dir, "data.p")
    model = load_model(data_config=data_cfg, train_config=train_cfg, model_class=Continuous_T_Dynamics, model_dir=model_dir, mode="best")

    # -----------------------------
    # 2) Load eval data
    #    Expect dict with keys: "obs": [B, seq_len, obs_dim],
    #                           "action": [B, seq_len, act_dim]
    # -----------------------------
    with open(eval_p_path, "rb") as f:
        eval_data = pickle.load(f)
    eps_arr = np.array(eval_data)  # [B, T, 15]

    B, T, _ = eps_arr.shape
    horizon = min(100, T-1)
    T = horizon + 1
    # Everything inside file is normalized by /scale → denormalize for visualization
    eps_denorm = eps_arr.astype(np.float32)[:, :T, :]               # [B,T,15], unnormalized
    eps_norm = eps_denorm / scale                    # [B,T,15], normalized

    # ----------------- actions (U) and initial states -----------------
    # U_t = (v_x, v_y) at time t; use first T-1 actions to predict next T-1 states.
    U_norm = eps_norm[:, :-1, -action_dim:]                    # [B,T-1,2] normalized velocities
    if pred_mode == "state":
        T_dim = state_dim
        x_gt_norm = jnp.array(eps_norm[:, :, :state_dim])                   # [B,T,8]    normalized state
    elif pred_mode == "pose":
        T_dim = pose_dim
        x_gt_norm = jnp.array(eps_norm[:, :, state_dim:state_dim+pose_dim])                         # [B,T,3]     normalized state
        x_gt_norm = x_gt_norm.at[:, :, -1].set(x_gt_norm[:, :, -1] * scale)  # denormalize angle
        def transform_fn(pose):
            B, T, D = pose.shape
            pose = pose.reshape(-1, D)
            kp = jax.vmap(pose_to_kp, in_axes=(0, None, None))(pose, stem_size, bar_size)
            return kp.reshape(B, T, -1)
    else:
        raise ValueError(f"Unknown pred_mode: {pred_mode}")

    if abs_pose:
        x_gt_norm = jnp.concatenate([x_gt_norm, eps_norm[:, :, state_dim+pose_dim:-action_dim]], axis=-1)  # [B,T_dim+2]

    x0_norm = x_gt_norm[:, 0, :]                         # [B,T_dim/T_dim+2]     normalized initial state

    # ----------------- load model & rollout (batch) -----------------
    # rollout(x0: [B,Dx], U: [B,T,Du]) -> X_pred: [B,T,Dx]
    X_pred_norm = model.rollout(jnp.asarray(x0_norm), jnp.asarray(U_norm))
    # prepend x0 to get [B,T,Dx]
    X_pred_full_norm = jnp.concatenate([x0_norm[:, None, :], X_pred_norm], axis=1)
    X_pred_full_denorm = X_pred_full_norm * scale
    
    pred_diff = jnp.abs(X_pred_full_norm - x_gt_norm)[:, :, :T_dim]
    mean_diff = jnp.mean(pred_diff, axis=(0)) # mean over B
    print(f"Mean absolute error per dim over time: {mean_diff[::horizon//10]}")

    if pred_mode == "pose":
        # renormalize angle in predicted poses
        X_pred_full_denorm = X_pred_full_denorm.at[:, :, pose_dim-1].set(X_pred_full_denorm[:, :, pose_dim-1] / scale)
        X_pred_full_denorm = transform_fn(X_pred_full_denorm)

    # ----------------- build GT/PRED arrays for plot_frame -----------------
    if abs_pose:
        gt_vis = np.concatenate([eps_denorm[..., :state_dim], eps_denorm[..., state_dim+pose_dim:-action_dim]], axis=-1)         # [B,T,10]
        pred_vis = np.concatenate([X_pred_full_denorm, eps_denorm[:, :, state_dim+pose_dim:-action_dim]], axis=-1)  # [B,T,10]
    else:
        gt_vis = rel_to_abs_kp_plus_pusher(np.concatenate([eps_denorm[..., :state_dim], eps_denorm[..., state_dim+pose_dim:]], axis=-1))         # [B,T,10]
        pred_vis = rel_to_abs_kp_plus_pusher(np.concatenate([X_pred_full_denorm, eps_denorm[:, :, state_dim+pose_dim:]], axis=-1))  # [B,T,10]

    # ----------------- write one GIF per episode -----------------
    out_dir = model_dir
    window_size = data_cfg["window_size"] * data_cfg.get("enlarge_factor_for_gen", 1)
    os.makedirs(out_dir, exist_ok=True)
    B = min(B, 10)
    for b in range(B):
        out_path = os.path.join(out_dir, f"ep{'_eval' if use_eval else ''}_{b:04d}.gif")
        plot_frame(
            out_path,
            gt=gt_vis,
            pred=pred_vis,
            B=b,
            fps=30,
            xlim=(0, window_size),
            ylim=(0, window_size),
        )
    print(f"Saved {B} GIFs to {out_dir}")

if __name__ == "__main__":
    main()
