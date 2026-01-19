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
from omegaconf import DictConfig
import yaml

sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import CT_Ctl_Reach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2
from CROWN_Reach.src.utils.vis import visualize_flowpipe_time
from CROWN_Reach.src.settings import CONFIG

from models.load import load_model
from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics
from models.quadrotor.ct_ctl import MLP_Controller, PID_Controller
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml")
def main(config: DictConfig):
    task_name = config["settings"]["task_name"]
    mode = "ct_ctl"
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

    use_pid = False
    if use_pid:
        model: PID_Controller = PID_Controller(data_cfg)  # use PID for testing CT dynamics only
    else:
        model: MLP_Controller = load_model(model_dir=model_dir, model_type="ct_ctl", mode="best", task_name=task_name)

    ct_dyn = Continuous_Quad_Dynamics(data_cfg)

    with open(eval_p_path, "rb") as f:
        episodes = jnp.array(pickle.load(f))

    B = episodes.shape[0]
    if episodes.shape[2] == 18:
        state_dim = 12  # pos, vel, rpy, rates
        v_cmd_dim = 3  # vel_cmd
        action_dim = 3
    elif episodes.shape[2] == 19:
        state_dim = 12  # pos, vel, rpy, rates
        v_cmd_dim = 3  # vel_cmd
        action_dim = 4
    elif episodes.shape[2] == 9:
        raise NotImplementedError("Data with 9 dimensions is for DT case.")
        state_dim = 6  # pos, vel
        action_dim = 3
    else:
        raise ValueError(f"Unknown data dimension: {episodes.shape[2]}")
    
    select_samples = [0]
    start_time_step = 20
    horizon = 10
    episodes = episodes[select_samples, start_time_step:start_time_step + horizon + 1, :]  # [B, T+1, Dx+Du]

    X_gt = episodes[..., :state_dim]
    v_cmds = episodes[:, :-1, state_dim:state_dim+v_cmd_dim]
    U_gt = episodes[..., -action_dim:]

    # 
    def f_wrapper(x):
        dx = ct_dyn(x)
        du = jnp.zeros_like(x[ct_dyn.Dx:])
        return jnp.concatenate([dx, du], axis=-1)
    dyn_frequency = float(ct_dyn.frequency)
    n_dyn_steps_per_ctl = round(ct_dyn.frequency / model.frequency)

    reach_cfg = train_cfg["reach"]
    eps = float(train_cfg["reach"]["eps_final"])
    eps = 0.02
    # init_remainder = reach_cfg.get("init_remainder", 1e-1)
    # frr_rounds = reach_cfg.get("frr_rounds", 5)
    # frr_stop_ratio = reach_cfg.get("frr_stop_ratio", 0.95)
    # sr_window_size = reach_cfg.get("sr_window_size", 100)
    init_remainder = eps * 5
    frr_rounds = 5
    frr_stop_ratio = 0.95
    sr_window_size = 100
    CONFIG["TRUNCATE_TO_AFFINE"] = reach_cfg.get("linear", False)
    reach_analyzer = CT_Ctl_Reach(f_wrapper, state_dim=model.Dx, action_dim=model.Du, nn_dyn=False, controller=model,
                                  n_steps_per_control=n_dyn_steps_per_ctl, step_size=1/dyn_frequency,
                                  init_remainder=init_remainder, frr_rounds=frr_rounds, frr_stop_ratio=frr_stop_ratio, sr_window_size=sr_window_size)

    state_init = X_gt[:, 0, :] # [B, Dx]
    state_lo = state_init - eps
    state_up = state_init + eps
    splits_cfg = {}
    # splits_cfg = {i: 2 for i in range(model.Dx//2)}
    X_lo = jnp.concatenate([state_lo, jnp.zeros_like(U_gt[:, 0, :])], axis=-1)
    X_up = jnp.concatenate([state_up, jnp.zeros_like(U_gt[:, 0, :])], axis=-1)
    X_lo, X_up = prepare_initial_set_v2(X_lo, X_up, splits_cfg=splits_cfg)
    
    reference_seq = v_cmds.repeat(max(X_lo.shape[0] // v_cmds.shape[0], 1), axis=0)  # [B, T, Dv]
    T_reach = horizon * reach_analyzer.n_steps_per_control
    enable_reach = True
    if use_pid:
        enable_reach = False
    if enable_reach:
        ts, r_lo, r_up, _, init_shrinked_all = reach_analyzer.verify_w_model(f_wrapper, model, X_lo, X_up, n_total_steps=T_reach, reference_seq=reference_seq)
        D = model.Dx + model.Du
        r_lo = r_lo.reshape(-1, T_reach + 1, D)
        r_up = r_up.reshape(-1, T_reach + 1, D)
        # roll actions forward a step to match state dims
        r_lo = r_lo.at[:, :-1, model.Dx:].set(r_lo[:, 1:, model.Dx:])
        r_up = r_up.at[:, :-1, model.Dx:].set(r_up[:, 1:, model.Dx:])
        reach_vol = calculate_volume(r_lo, r_up, union_init=False, mode='sum') / r_lo.shape[0]
        print(f"Reachable set volume at time step {T_reach} over {r_lo.shape[0]} partitions: {reach_vol}")
        print(f"init_shrinked_all: {init_shrinked_all.all()}")
    n_samples = 64
    key = jrandom.PRNGKey(42)

    # sample_state_init = state_init_lo + (state_init_up - state_init_lo) * jrandom.uniform(key, shape=(n_samples, state_init_lo.shape[1])) # [n_samples, Dx]
    raw_samples = jrandom.uniform(key, shape=(X_lo.shape[0], max(n_samples // X_lo.shape[0], 1), model.Dx)) # [n_partitions, n_per_partition, Dx]
    sample_state_init = X_lo[:, jnp.newaxis, :model.Dx] + (X_up - X_lo)[:, jnp.newaxis, :model.Dx] * raw_samples
    sample_state_init = sample_state_init.reshape(-1, model.Dx)  # [n_samples, Dx]

    X_curr = sample_state_init # [B, Dx]
    def one_step_ctl_dyn(carry, xs):
        X_curr = carry  # [B, Dx]
        v_cmd = xs  # [B, Dv]
        U_pred = model.forward(X_curr, v_cmd)  # [B, Du]
        X_next = ct_dyn.rollout(X_curr, U_pred[:, None].repeat(n_dyn_steps_per_ctl, axis=1))  # [B, n_dyn_steps_per_ctl, Dx]
        # X_next = X_next[:, -1, :]  # take the last state after n_dyn_steps_per_ctl

        # X_next = ct_dyn.forward(X_curr, U_pred, dt=model.dt)  # [B, Dx]

        return X_next[:, -1, :], (X_next, U_pred)

    v_cmds = v_cmds.repeat(max(n_samples // v_cmds.shape[0], 1), axis=0)  # [B, T, Dv]
    _, (X_preds_full, U_preds) = jax.lax.scan(one_step_ctl_dyn, X_curr, v_cmds.transpose(1, 0, 2), length=horizon)
    X_preds_full = X_preds_full.transpose(1,0,2,3).reshape(n_samples, T_reach, -1)  # [B, T, Dx]
    X_preds_full = jnp.concatenate([X_curr[:, None, :], X_preds_full], axis=1)  # [B, T+1, Dx]
    X_preds = X_preds_full[:, ::n_dyn_steps_per_ctl, :]  # [B, T+1, Dx]
    U_preds_full = U_preds.transpose(1,0,2).repeat(n_dyn_steps_per_ctl, axis=1)  # [B, T, Du]
    U_preds_full = jnp.concatenate([U_preds_full, U_preds_full[:, -1:, :]], axis=1)  # [B, T+1, Du]
    U_preds = U_preds_full[:, ::n_dyn_steps_per_ctl, :]  # [B, T+1, Du]
    v_cmds_preds = X_preds[:, 1:, v_cmd_dim: 2*v_cmd_dim]  # [B, T+1, Dv]
    v_cmds_preds = jnp.concatenate([v_cmds_preds, v_cmds_preds[:, -1:, :]], axis=1)  # [B, T+1, Dv] for plotting
    U_gt = U_gt.at[:, -1, :].set(U_gt[:, -2, :])  # for plotting
    v_cmds = jnp.concatenate([v_cmds, v_cmds[:, -1:, :]], axis=1)  # [B, T+1, Dv] for plotting

    print(f"X error: {jnp.abs(X_gt - X_preds).mean()}")
    print(f"U error: {jnp.abs(U_gt - U_preds).mean()}")
    print(f"v_cmd error: {jnp.abs(v_cmds - v_cmds_preds).mean()}")
    # exit()

    out_dir = os.path.join(model_dir, f"vis_reach_T{horizon}_eps{eps}_split{X_lo.shape[0]}")
    os.makedirs(out_dir, exist_ok=True)

    if enable_reach:
        for idx in range(model.Dx + action_dim):
            outfile = os.path.join(out_dir, f"reach_{idx}.png")
            visualize_flowpipe_time(
                times=ts,
                lowers=r_lo,
                uppers=r_up,
                trajs=np.concatenate([X_preds_full, U_preds_full], axis=-1),
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
        plot_quad_states_actions(X_gt[i, :, :6], v_cmds[i], dt=model.dt, out_path=os.path.join(out_dir, f"gt_states_vcmd_{i}.png"))
        plot_quad_states_actions(X_preds[i, :, :6], v_cmds[i], dt=model.dt, out_path=os.path.join(out_dir, f"pred_states_vcmd_{i}.png"))
        plot_quad_states_actions(X_gt[i], U_gt[i], dt=model.dt, out_path=os.path.join(out_dir, f"gt_states_actions_{i}.png"))
        plot_quad_states_actions(X_preds[i], U_preds[i], dt=model.dt, out_path=os.path.join(out_dir, f"pred_states_actions_{i}.png"))


if __name__ == "__main__":
    main()