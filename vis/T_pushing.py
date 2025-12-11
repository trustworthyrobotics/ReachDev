import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from io import BytesIO
import os
import pickle
import jax.numpy as jnp
import equinox as eqx
import hydra
from omegaconf import DictConfig

from models.dynamics import load_t_dynamics_model

def plot_Tee(tee_kp, c="orange", label=""):
    """tee_kp: np.array([TL.x,TL.y,TC.x,TC.y,TR.x,TR.y,B.x,B.y])"""
    order = np.array([0, 2, 1, 3])  # TL, TR, TC, B to draw top bar then stem
    plt.plot(tee_kp[::2][order], tee_kp[1::2][order], c=c, alpha=0.5, label=label)

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
        plot_agent(gt[B, i, 8:],       c="green",  label="GT Agent")
        plot_agent(pred[B, i, 8:],     c="orange", label="Pred Agent")

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
    data_dir = config["data"]["out_path"]
    model_dir = config["train"]["out_dir"]
    scale = float(config["data"].get("scale", 1.0))  # data was normalized by /scale

    eval_p_path = os.path.join(data_dir, "data_eval.p")
    model_path = os.path.join( config["train"]["out_dir"], "last_model.eqx")
    model = load_t_dynamics_model(config=config, model_path=model_path)

    # -----------------------------
    # 2) Load eval data
    #    Expect dict with keys: "obs": [B, seq_len, obs_dim],
    #                           "action": [B, seq_len, act_dim]
    # -----------------------------
    with open(eval_p_path, "rb") as f:
        eval_data = pickle.load(f)
    eps_arr = np.array(eval_data)  # [B, T, 12]

    B, T, _ = eps_arr.shape
    scale = float(config["data"]["scale"])

    # Everything inside file is normalized by /scale → denormalize for visualization
    eps_denorm = eps_arr.astype(np.float32)               # [B,T,12], unnormalized
    eps_norm = eps_denorm / scale                    # [B,T,12], normalized

    # ----------------- actions (U) and initial states -----------------
    # U_t = (v_x, v_y) at time t; use first T-1 actions to predict next T-1 states.
    U_norm = eps_norm[:, :-1, 10:12]                    # [B,T-1,2] normalized velocities
    x0_norm = eps_norm[:, 0, :8]                         # [B,8]     normalized initial state

    # ----------------- load model & rollout (batch) -----------------
    # rollout_model(x0: [B,Dx], U: [B,T-1,Du]) -> X_pred: [B,T-1,Dx]
    X_pred_norm = model.rollout_model(jnp.asarray(x0_norm), jnp.asarray(U_norm))
    # prepend x0 to get [B,T,Dx]
    X_pred_full_norm = jnp.concatenate([x0_norm[:, None, :], X_pred_norm], axis=1)
    X_pred_full_denorm = np.asarray(X_pred_full_norm) * scale
    eps_pred_denorm = np.concatenate([X_pred_full_denorm, eps_denorm[:, :, 8:12]], axis=-1)  # [B,T,12]

    # ----------------- build GT/PRED arrays for plot_frame -----------------
    gt_vis = rel_to_abs_kp_plus_pusher(eps_denorm)         # [B,T,10]
    pred_vis = rel_to_abs_kp_plus_pusher(eps_pred_denorm)  # [B,T,10]
    print(f"vis error: {np.abs(pred_vis - gt_vis).mean()}")

    # ----------------- write one GIF per episode -----------------
    out_dir = os.path.join("output", "vis", "T_pushing")
    window_size = config["data"]["window_size"]
    os.makedirs(out_dir, exist_ok=True)
    B = min(B, 10)
    for b in range(B):
        out_path = os.path.join(out_dir, f"ep_{b:04d}.gif")
        plot_frame(
            out_path,
            gt=gt_vis,
            pred=pred_vis,
            B=b,
            fps=10,
            xlim=(0, window_size),
            ylim=(0, window_size),
        )
    print(f"Saved {B} GIFs to {out_dir}")

if __name__ == "__main__":
    main()
