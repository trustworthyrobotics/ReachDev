import sys
import numpy as np
import os
import pickle
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom
import hydra
from omegaconf import DictConfig, open_dict
import yaml
import time

sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DT_Plan_Reach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2
from CROWN_Reach.src.utils.vis import visualize_flowpipe_time
from CROWN_Reach.src.settings import CONFIG

from models.load import load_model
from models.quadrotor.dt_dyn import Quad_Dynamics
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml")
def main(config: DictConfig):
    if "testing" in config:
        testing_config = config["testing"]
        mode = testing_config.get("mode", "certified")
        assert mode in {"certified", "regular"}, f"Unknown testing mode: {mode}"
        model_config = testing_config[mode]
        with open_dict(config):
            config["test_models"] = model_config
    task_name = config["settings"]["task_name"]
    mode = "dt_dyn"
    model_dir = config["test_models"][f"{mode}_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    # override config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    train_cfg = config[f"train_{mode}"]
    data_dir = train_cfg["data_dir"]

    use_eval = True
    if use_eval:
        eval_p_path = os.path.join(data_dir, "data_eval.p")
    else:
        eval_p_path = os.path.join(data_dir, "data.p")

    model: Quad_Dynamics = load_model(model_dir=model_dir, model_type=mode, mode="best", task_name=task_name)

    with open(eval_p_path, "rb") as f:
        episodes = jnp.array(pickle.load(f))

    B = episodes.shape[0]
    if episodes.shape[2] == 18:
        raise NotImplementedError("Data with 18 dimensions is for CT case.")
        state_dim = 12  # pos, vel, rpy, rates
        v_cmd_dim = 3  # vel_cmd
        action_dim = 3
    elif episodes.shape[2] == 19:
        raise NotImplementedError("Data with 19 dimensions is for CT case.")
        state_dim = 12  # pos, vel, rpy, rates
        v_cmd_dim = 3  # vel_cmd
        action_dim = 4
    elif episodes.shape[2] == 9:
        state_dim = 6  # pos, vel
        action_dim = 3
    else:
        raise ValueError(f"Unknown data dimension: {episodes.shape[2]}")
    
    # select_samples = [0, 1]
    # n_reach_batch = len(select_samples)

    n_reach_batch = episodes.shape[0]
    select_samples = np.arange(n_reach_batch).tolist()

    start_time_step = 0
    horizon = 10
    episodes = episodes[select_samples, start_time_step:start_time_step + horizon + 1, :]  # [B, T+1, Dx+Du]

    X_gt = episodes[..., :state_dim]
    U_gt = episodes[:, :-1, -action_dim:]

    # 
    def f_wrapper(x):
        state_next = model(x)
        action_next = x[-model.Du:]
        return jnp.concatenate([state_next, action_next], axis=-1)
    dyn_frequency = float(model.frequency)

    reach_cfg = train_cfg["reach"]
    eps = float(train_cfg["reach"]["eps_final"])
    eps = 0.3

    CONFIG["TRUNCATE_TO_AFFINE"] = reach_cfg.get("linear", False)
    reach_analyzer = DT_Plan_Reach(f_wrapper, state_dim=state_dim, action_dim=action_dim, nn_dyn=True, n_steps_per_plan=1, step_size=1, config=CONFIG)

    start_time = time.time()

    state_init = X_gt[:, 0, :] # [B, Dx]
    state_lo = state_init - eps
    state_up = state_init + eps
    splits_cfg = {}
    X_lo = jnp.concatenate([state_lo, jnp.zeros_like(U_gt[:, 0, :])], axis=-1)
    X_up = jnp.concatenate([state_up, jnp.zeros_like(U_gt[:, 0, :])], axis=-1)
    X_lo, X_up = prepare_initial_set_v2(X_lo, X_up, splits_cfg=splits_cfg)

    T_reach = horizon

    ts, r_lo, r_up, _, _ = reach_analyzer.verify(X_lo, X_up, n_total_steps=T_reach, action_seq=U_gt.repeat(X_up.shape[0]//U_gt.shape[0], axis=0)[:, None])
    D = model.Dx + model.Du

    r_lo = r_lo.reshape(n_reach_batch, -1, T_reach + 1, D)
    r_up = r_up.reshape(n_reach_batch, -1, T_reach + 1, D)
    r_lo_agg = jnp.min(r_lo, axis=1, keepdims=False)
    r_up_agg = jnp.max(r_up, axis=1, keepdims=False)

    reach_vols = calculate_volume(r_lo_agg[..., :state_dim], r_up_agg[..., :state_dim], union_init=False, mode="sum", keep_time=True, keep_batch=True)
    
    n_samples = 32
    n_samples = n_samples * n_reach_batch
    key = jrandom.PRNGKey(42)

    # sample_state_init = state_init_lo + (state_init_up - state_init_lo) * jrandom.uniform(key, shape=(n_samples, state_init_lo.shape[1])) # [n_samples, Dx]
    raw_samples = jrandom.uniform(key, shape=(X_lo.shape[0], max(n_samples // X_lo.shape[0], 1), state_dim)) # [n_partitions, n_per_partition, Dx]
    sample_state_init = X_lo[:, jnp.newaxis, :state_dim] + (X_up - X_lo)[:, jnp.newaxis, :state_dim] * raw_samples
    sample_state_init = sample_state_init.reshape(-1, state_dim)  # [n_samples, Dx]

    X_curr = sample_state_init # [B, Dx]
    U_gt = U_gt.repeat(X_curr.shape[0]//U_gt.shape[0], axis=0)  # [B, T, Du]
    X_preds = model.rollout(X_curr, U_gt)  # [B, T, Dx]
    X_preds = jnp.concatenate([X_curr[:, None, :], X_preds], axis=1)  # [B, T+1, Dx]
    U_gt = jnp.concatenate([jnp.zeros_like(U_gt[:, :1, :]), U_gt], axis=1)  # [B, T+1, Du] for plotting

    sample_rollout = X_preds.reshape(n_reach_batch, -1, T_reach + 1, state_dim)  # [N, n_per_partition, T+1, Dx]
    sample_r_lo = sample_rollout.min(axis=1, keepdims=False)  # [N, T+1, Dx]
    sample_r_up = sample_rollout.max(axis=1, keepdims=False)  # [N, T+1, Dx]

    sample_vols = calculate_volume(sample_r_lo[..., :state_dim], sample_r_up[..., :state_dim], union_init=False, mode='sum', keep_time=True, keep_batch=True)  # [N, T+1]

    end_time = time.time()
    print(f"Reachability analysis for {n_reach_batch} eps took {end_time - start_time:.2f} seconds.")

    # save to npz for further analysis
    npz_path = os.path.join(model_dir, f"reach_eval.npz")
    np.savez_compressed(npz_path,
        reach_vols=np.array(reach_vols),
        sample_vols=np.array(sample_vols),
        reach_r_lo=np.array(r_lo_agg),
        reach_r_up=np.array(r_up_agg),
        sample_r_lo=np.array(sample_r_lo),
        sample_r_up=np.array(sample_r_up),
    )
    print(f"Saved reachability npz to {npz_path}")
    print(f"Reachable set volume over {T_reach} steps: {reach_vols.mean(axis=0)}")
    print(f"Sampled rollout volume over {T_reach} steps: {sample_vols.mean(axis=0)}")

    # print(f"X error: {jnp.abs(X_gt - X_preds).mean()}")
    # exit()

    if n_reach_batch > 1:
        exit()

    out_dir = os.path.join(model_dir, f"vis_reach")
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(model.Dx + action_dim):
        outfile = os.path.join(out_dir, f"reach_{idx}.png")
        visualize_flowpipe_time(
            times=ts,
            lowers=r_lo_agg,
            uppers=r_up_agg,
            trajs=np.concatenate([X_preds, U_gt], axis=-1),
            state_idx=idx,
            file_name=outfile,
            print_boxes=False,
            draw_boxes=True,
            aggregate_partitions=True,
            stride=1,
            draw_traj=True,
        )

    n_vis = min(1, len(select_samples))
    for i in range(n_vis):
        plot_3d_trajectories(X_gt[i, :, :3][:, None], num_quads=1, dt=model.dt, out_path=os.path.join(out_dir, f"gt_trajectories_{i}.png"))
        plot_3d_trajectories(X_preds[i, :, :3][:, None], num_quads=1, dt=model.dt, out_path=os.path.join(out_dir, f"pred_trajectories_{i}.png"))
        plot_quad_states_actions(X_gt[i, :, :6], U_gt[i], dt=model.dt, out_path=os.path.join(out_dir, f"gt_states_vcmd_{i}.png"))
        plot_quad_states_actions(X_preds[i, :, :6], U_gt[i], dt=model.dt, out_path=os.path.join(out_dir, f"pred_states_vcmd_{i}.png"))

if __name__ == "__main__":
    main()