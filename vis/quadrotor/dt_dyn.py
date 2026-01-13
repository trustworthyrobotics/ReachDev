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
from models.quadrotor.dt_dyn import Quad_Dynamics
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml")
def main(config: DictConfig):
    task_name = config["settings"]["task_name"]
    model_dir = config["test_models"]["dt_dyn_dir"]
    config_path = os.path.join(model_dir, "config.yaml")
    # override config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    train_cfg = config["train_dt_dyn"]
    data_dir = train_cfg.get("data_dir", "output/data/T_pushing_freq10_ctl")

    use_eval = True
    if use_eval:
        eval_p_path = os.path.join(data_dir, "data_eval.p")
    else:
        eval_p_path = os.path.join(data_dir, "data.p")
    model: Quad_Dynamics = load_model(model_dir=model_dir, model_type="dt_dyn", mode="best", task_name=task_name)

    with open(eval_p_path, "rb") as f:
        episodes = np.array(pickle.load(f))

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
    
    start_time_step = 0
    horizon = 10
    episodes = episodes[:, start_time_step:start_time_step + horizon + 1, :]  # [B, T+1, Dx+Du]

    X_gt = episodes[..., :state_dim]
    U_gt = episodes[..., -action_dim:]

    X_curr = X_gt[:, 0, :] # [B, Dx]

    X_preds = model.rollout(X_curr, U_gt)  # [B, T, Dx]

    out_dir = os.path.join(model_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    n_samples = 5
    for i in range(n_samples):
        plot_3d_trajectories(X_gt[i, :, :3][:, None], num_quads=1, dt=model.dt, out_path=os.path.join(out_dir, f"gt_trajectories_{i}.png"))
        plot_quad_states_actions(X_gt[i, :, :6], U_gt[i], dt=model.dt, out_path=os.path.join(out_dir, f"gt_states_vcmd_{i}.png"))
        plot_3d_trajectories(X_preds[i, :, :3][:, None], num_quads=1, dt=model.dt, out_path=os.path.join(out_dir, f"pred_trajectories_{i}.png"))
        plot_quad_states_actions(X_preds[i, :, :6], U_gt[i], dt=model.dt, out_path=os.path.join(out_dir, f"pred_states_vcmd_{i}.png"))


if __name__ == "__main__":
    main()