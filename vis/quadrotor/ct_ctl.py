import numpy as np
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

from models.load import load_model
from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics
from models.quadrotor.ct_ctl import MLP_Controller, PID_Controller
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml")
def main(config: DictConfig):
    task_name = config["settings"]["task_name"]
    model_dir = config["test_models"]["ct_ctl_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    # override config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    train_cfg = config["train_ct_ctl"] if "train_ct_ctl" in config else config["train"]
    data_dir = train_cfg.get("data_dir", "output/data/T_pushing_freq10_ctl")

    use_eval = True
    if use_eval:
        eval_p_path = os.path.join(data_dir, "data_eval.p")
    else:
        eval_p_path = os.path.join(data_dir, "data.p")

    use_pid = False
    if use_pid:
        # model: MLP_Controller = load_model(model_dir=model_dir, model_type="ct_ctl", mode="best", task_name=task_name)
        model = PID_Controller(data_cfg)  # use PID for testing CT dynamics only
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
    
    start_time_step = 0
    horizon = 20
    episodes = episodes[:, start_time_step:start_time_step + horizon + 1, :]  # [B, T+1, Dx+Du]

    X_gt = episodes[..., :state_dim]
    v_cmds = episodes[:, :-1, state_dim:state_dim+v_cmd_dim]
    U_gt = episodes[..., -action_dim:]

    X_curr = X_gt[:, 0, :] # [B, Dx]

    n_dyn_steps_per_ctl = round(ct_dyn.frequency / model.frequency)

    def one_step_ctl_dyn(carry, xs):
        X_curr = carry  # [B, Dx]
        v_cmd = xs  # [B, Dv]
        U_pred = model.forward(X_curr, v_cmd)  # [B, Du]
        # X_next = ct_dyn.rollout(X_curr, U_pred[:, None].repeat(n_dyn_steps_per_ctl, axis=1))  # [B, n_dyn_steps_per_ctl, Dx]
        # X_next = X_next[:, -1, :]  # take the last state after n_dyn_steps_per_ctl

        X_next = ct_dyn.forward(X_curr, U_pred, dt=model.dt)  # [B, Dx]

        return X_next, (X_next, U_pred)


    _, (X_preds, U_preds) = jax.lax.scan(one_step_ctl_dyn, X_curr, v_cmds.transpose(1, 0, 2), length=horizon)
    X_preds = X_preds.transpose(1,0,2)  # [B, T, Dx]
    X_preds = jnp.concatenate([X_curr[:, None, :], X_preds], axis=1)  # [B, T+1, Dx]
    U_preds = U_preds.transpose(1,0,2)  # [B, T, Du]
    U_preds = jnp.concatenate([U_preds, U_preds[:, -1:, :]], axis=1)  # [B, T+1, Du]
    v_cmds_preds = X_preds[:, 1:, v_cmd_dim: 2*v_cmd_dim]  # [B, T+1, Dv]
    v_cmds_preds = jnp.concatenate([v_cmds_preds, v_cmds_preds[:, -1:, :]], axis=1)  # [B, T+1, Dv] for plotting
    U_gt = U_gt.at[:, -1, :].set(U_gt[:, -2, :])  # for plotting
    v_cmds = jnp.concatenate([v_cmds, v_cmds[:, -1:, :]], axis=1)  # [B, T+1, Dv] for plotting

    print(f"X error: {jnp.abs(X_gt - X_preds).mean()}")
    print(f"U error: {jnp.abs(U_gt - U_preds).mean()}")
    print(f"v_cmd error: {jnp.abs(v_cmds - v_cmds_preds).mean()}")
    exit()

    out_dir = os.path.join(model_dir, f"vis{'_pid' if use_pid else ''}")
    os.makedirs(out_dir, exist_ok=True)
    n_samples = 5
    for i in range(n_samples):
        plot_3d_trajectories(X_gt[i, :, :3][:, None], num_quads=1, dt=model.dt, out_path=os.path.join(out_dir, f"gt_trajectories_{i}.png"))
        plot_3d_trajectories(X_preds[i, :, :3][:, None], num_quads=1, dt=model.dt, out_path=os.path.join(out_dir, f"pred_trajectories_{i}.png"))
        plot_quad_states_actions(X_gt[i, :, :6], v_cmds[i], dt=model.dt, out_path=os.path.join(out_dir, f"gt_states_vcmd_{i}.png"))
        plot_quad_states_actions(X_preds[i, :, :6], v_cmds[i], dt=model.dt, out_path=os.path.join(out_dir, f"pred_states_vcmd_{i}.png"))
        plot_quad_states_actions(X_gt[i], U_gt[i], dt=model.dt, out_path=os.path.join(out_dir, f"gt_states_actions_{i}.png"))
        plot_quad_states_actions(X_preds[i], U_preds[i], dt=model.dt, out_path=os.path.join(out_dir, f"pred_states_actions_{i}.png"))


if __name__ == "__main__":
    main()