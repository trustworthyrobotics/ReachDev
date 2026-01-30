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
from omegaconf import DictConfig, open_dict
import yaml

from models.load import load_model
from models.T_pushing.ct_dyn import Continuous_T_Dynamics
from models.T_pushing.ct_ctl import T_controller
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
        plot_agent(pred[B, i, 8:],    c="red",    label="Pred Pusher")

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
    if "testing" in config:
        testing_config = config["testing"]
        mode = testing_config.get("mode", "certified")
        assert mode in {"certified", "regular"}, f"Unknown testing mode: {mode}"
        model_config = testing_config[mode]
        with open_dict(config):
            config["test_models"] = model_config
    model_dir = config["test_models"]["ct_ctl_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    # override config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    train_cfg = config["train_ct_ctl"] if "train_ct_ctl" in config else config["train"]
    data_dir = train_cfg.get("data_dir", "output/data/T_pushing_freq10_ctl")

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
    model: T_controller = load_model(model_dir=model_dir, model_type="ct_ctl", mode="best")

    ct_dyn_dir = data_cfg["ct_ctl"]["model_dir"]
    ct_dyn: Continuous_T_Dynamics = load_model(model_dir=ct_dyn_dir, model_type="ct_dyn", mode="best")

    # -----------------------------
    # 2) Load eval data
    #    Expect dict with keys: "obs": [B, seq_len, obs_dim],
    #                           "action": [B, seq_len, act_dim]
    # -----------------------------
    with open(eval_p_path, "rb") as f:
        eval_data = pickle.load(f)
    eps_arr = np.array(eval_data)  # [B, T, 15]

    B, T, _ = eps_arr.shape
    horizon = min(10, T-1)
    n_track = 2
    T = horizon * n_track + 1
    start_time_step = 50
    # Everything inside file is normalized by /scale → denormalize for visualization
    eps_denorm = eps_arr.astype(np.float32)[:, start_time_step:start_time_step+T, :]               # [B,T,15], unnormalized
    eps_norm = eps_denorm / scale                    # [B,T,15], normalized

    U_norm = eps_norm[:, :-1, -action_dim:]                    # [B,T,2] normalized velocities
    if pred_mode == "state":
        x_norm = eps_norm[:, :, :state_dim]                   # [B,T+1,8]    normalized initial state
    elif pred_mode == "pose":
        x_norm = eps_norm[:, :, state_dim:state_dim+pose_dim]                         # [B,3]     normalized initial state
        x_norm[:, :, -1] = x_norm[:, :, -1] * scale  # denormalize angle
        def transform_fn(pose):
            B, T, D = pose.shape
            pose = pose.reshape(-1, D)
            kp = jax.vmap(pose_to_kp, in_axes=(0, None, None))(pose, stem_size, bar_size)
            return kp.reshape(B, T, -1)
    else:
        raise ValueError(f"Unknown pred_mode: {pred_mode}")

    x_norm = jnp.array(x_norm)
    U_norm = jnp.array(U_norm)
    if abs_pose:
        x_norm = jnp.concatenate([x_norm, eps_norm[:, :, state_dim+pose_dim:-action_dim]], axis=-1)  # [B,10]

    X_curr = x_norm[:, 0, :] # [B, Dx]
    def one_step_ctl_dyn(carry, _):
        X_curr, X_tgt, U_ref = carry  # [B, Dx]
        U_pred = model.forward(X_curr, X_tgt, U_ref)  # [B, Du]
        X_next = ct_dyn.forward(X_curr, U_pred)  # [B, Dx]
        return (X_next, X_tgt, U_ref), (X_next, U_pred)

    def track(carry, xs):
        X_curr = carry  # [B, Dx]
        X_tgt, U_ref = xs[:, :-action_dim], xs[:, -action_dim:]  # [B, Dx], [B, Du]
        _, (X_preds, U_preds) = jax.lax.scan(one_step_ctl_dyn, (X_curr, X_tgt, U_ref), None, length=horizon)
        return X_preds[-1], (X_preds, U_preds)

    X_tgts = x_norm[:, horizon::horizon, :state_dim if pred_mode == "state" else pose_dim]  # [B, n_track, Dx]
    U_refs = U_norm.reshape(B, n_track, horizon, action_dim).mean(axis=2)  # [B, n_track, Du]
    _, (X_preds, U_preds) = jax.lax.scan(track, X_curr, jnp.concatenate([X_tgts, U_refs], axis=-1).transpose(1,0,2), length=n_track)

    X_preds = X_preds.reshape(-1, B, x_norm.shape[2]).transpose(1,0,2)  # [B, T, Dx]
    U_preds = U_preds.reshape(-1, B, action_dim).transpose(1,0,2)  # [B, T, Du]

    # U_gts_j = jnp.array(eps_norm[:, :-1, -action_dim:])
    # X_gts_j = jnp.array(eps_norm[:, 1:, state_dim:state_dim+pose_dim])
    # X_preds_gt = ct_dyn.rollout(X_curr, U_gts_j)
    # U_preds = U_gts_j
    # X_preds = X_preds_gt

    X_preds_norm = jnp.concatenate([X_curr[:, None, :], X_preds], axis=1)
    if pred_mode == "pose":
        X_gts_norm = jnp.array(eps_norm[:, :, state_dim:state_dim+pose_dim])
    else:
        X_gts_norm = jnp.array(eps_norm[:, :, :state_dim])
    if abs_pose:
        X_gts_norm = jnp.concatenate([X_gts_norm, eps_norm[:, :, state_dim+pose_dim:-action_dim]], axis=-1)  # [B,T,10]
    pred_diff = X_preds_norm - X_gts_norm
    mean_diff = jnp.mean(pred_diff ** 2, axis=(0, 2)) # mean over B
    print(f"MSE: {mean_diff}")

    # save to npz for further analysis
    npz_path = os.path.join(model_dir, f"pred_eval.npz")
    np.savez_compressed(npz_path,
        X_preds=np.array(X_preds_norm),
        X_gts=np.array(X_gts_norm),
    )
    print(f"Saved prediction npz to {npz_path}")

    exit()

    X_preds = X_preds_norm * scale  # [B,T, Dx]
    if pred_mode == "pose":
        # renormalize angle in predicted poses
        X_preds = X_preds.at[:, :, pose_dim-1].set(X_preds[:, :, pose_dim-1] / scale)
        X_preds = transform_fn(X_preds)
    X_preds = np.array(X_preds)
    U_preds = np.array(U_preds)
    U_preds = np.concatenate([U_preds, eps_norm[:, -1:, -action_dim:]], axis=1) * scale  # [B,T, Du]
    pusher_pos_curr = eps_denorm[:, 0, state_dim+pose_dim:-action_dim]  # [B,2]
    pusher_pos_preds = np.cumsum(np.concatenate([pusher_pos_curr[:, None, :], U_preds[:, :-1, :] * float(ct_dyn.dt)], axis=1), axis=1)  # [B,T,2]
    
    X_gts = eps_denorm[..., :state_dim]
    U_gts = eps_denorm[..., -action_dim:]
    X_diff = np.abs(X_preds - X_gts)
    U_diff = np.abs(U_preds - U_gts)
    X_tgt_diff = X_diff[:, horizon::horizon]
    # print(f"kp state error mean: {X_diff.mean(axis=(0,2))}, max: {X_diff.max(axis=(0,2))}, action error mean: {U_diff.mean(axis=(0,2))}, max: {U_diff.max(axis=(0,2))}")
    # print(f"tagret kp state error mean: {X_tgt_diff.mean(axis=(0,2))}, max: {X_tgt_diff.max(axis=(0,2))}")
    print(f"tagret kp state error q25: {np.percentile(X_tgt_diff, 25, axis=(0,2))}, q50: {np.percentile(X_tgt_diff, 50, axis=(0,2))}, q75: {np.percentile(X_tgt_diff, 75, axis=(0,2))}")
    print(f"action error q25: {np.percentile(U_diff, 25, axis=(0,2))}, q50: {np.percentile(U_diff, 50, axis=(0,2))}, q75: {np.percentile(U_diff, 75, axis=(0,2))}")
    # exit()
    if abs_pose:
        gt_vis = np.concatenate([eps_denorm[..., :state_dim], eps_denorm[..., state_dim+pose_dim:-action_dim]], axis=-1)         # [B,T,10]
        pred_vis = np.concatenate([X_preds, pusher_pos_preds], axis=-1)  # [B,T,10]
    else:
        gt_vis = rel_to_abs_kp_plus_pusher(np.concatenate([eps_denorm[..., :state_dim], eps_denorm[..., state_dim+pose_dim:]], axis=-1))         # [B,T,10]
        pred_vis = rel_to_abs_kp_plus_pusher(np.concatenate([X_preds, pusher_pos_preds, U_preds], axis=-1))  # [B,T,10]

    # ----------------- write one GIF per episode -----------------
    out_dir = model_dir
    window_size = data_cfg["window_size"] * data_cfg.get("enlarge_factor_for_gen", 1)
    os.makedirs(out_dir, exist_ok=True)
    max_vis = 10
    # # select 5 samples with largest X_tgt_diff and 5 samples with smallest X_tgt_diff
    # sample_indices = np.concatenate([
    #     np.argsort(-X_tgt_diff.max(axis=(1,2)))[:max_vis//2],
    #     np.argsort(X_tgt_diff.max(axis=(1,2)))[:max_vis//2]
    # ])
    sample_indices = np.random.choice(B, size=min(B, max_vis), replace=False)
    for b in sample_indices:
        # if b != 3:
        #     continue
        out_path = os.path.join(out_dir, f"ep{'_eval' if use_eval else ''}_{b:04d}.gif")
        plot_frame(
            out_path,
            gt=gt_vis,
            pred=pred_vis,
            B=b,
            fps=20,
            xlim=(0, window_size),
            ylim=(0, window_size),
        )
    print(f"Saved {B} GIFs to {out_dir}")

if __name__ == "__main__":
    main()