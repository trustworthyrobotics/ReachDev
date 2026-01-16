import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import os
import pickle
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from envs.quadrotor.quad_sim import Quad_Sim_Ctl, Quad_Sim_DT
from envs.quadrotor.helper import sample_vel_cmd_sequence

@hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml", version_base=None)
def main(config: DictConfig) -> None:
    data_config = config["data"]
    data_mode = config["settings"].get("data_mode", "ct_ctl")
    # data_config["frequency"] is the ode frequency limit
    frequency = min(data_config[data_mode]["frequency"], data_config["ct_frequency"])

    data_config[data_mode]["episode_length"] = data_config[data_mode]["episode_length"] * frequency
    data_config[data_mode]["num_episodes"] = data_config[data_mode]["num_episodes"] // frequency

    batch_size = data_config[data_mode].get("batch_size", 256)
    SAVE_IMG = data_config.get("gif", False)
    if SAVE_IMG:
        batch_size = 1
    episode_length = data_config[data_mode]["episode_length"] + 1 # include initial state
    num_episodes = data_config[data_mode]["num_episodes"]
    num_batches = int(np.ceil(num_episodes / batch_size))

    training = config["data"].get("training", True) # generate training or testing data
    key = jax.random.PRNGKey(config["settings"].get("seed", 0))
    out_path = hydra.utils.to_absolute_path(data_config[data_mode]["out_path"])
    os.makedirs(out_path, exist_ok=True)
    if data_mode == "dt_dyn":
        env_class = Quad_Sim_DT    
    elif data_mode == "ct_ctl":
        env_class = Quad_Sim_Ctl
    env = env_class(data_config=data_config, num_quads=batch_size)

    Dv = 3  # velocity command dimension
    acc_limits = jnp.full((env.num_quads, Dv), data_config.get("acc_limits", 1.0))
    vel_limits = jnp.full((env.num_quads, Dv), data_config.get("vel_limits", 2.0))
    vel_limits = jnp.stack([-vel_limits, vel_limits], axis=1)  # (num_quads, 2, Dv)

    batch_list = []

    sample_fn_jit = jax.jit(sample_vel_cmd_sequence, static_argnames=("num_quads", "dt", "n_steps"))

    for batch_idx in tqdm(range(num_batches)):
        key, subkey = jax.random.split(key)
        env.reset()
        vel_cmd_seq = sample_fn_jit(subkey, env.num_quads, dt=env.dt, n_steps=episode_length, amax=acc_limits, v_bounds=vel_limits)
        eps_list = []
        for step in range(episode_length):
            v_cmds = vel_cmd_seq[step] # (num_quads, 3)
            env_dict = env.update(v_cmds, n_sim_time=env.dt)
            if data_mode == "dt_dyn":
                eps_list.append(np.concatenate([env_dict["state"], env_dict["action"]], axis=-1))  # (num_quads, Dx+Du)
            elif data_mode == "ct_ctl":
                eps_list.append(np.concatenate([env_dict["state"], v_cmds, env_dict["action"]], axis=-1))  # (num_quads, Dx+Dv+Du)
        batch_data = np.stack(eps_list, axis=0)  # (episode_length, num_quads, Dx+Du)
        batch_data[:-1, :, env.Dx:] = batch_data[1:, :, env.Dx:]  # shift actions
        batch_list.append(batch_data)
        if SAVE_IMG:
            env.visualize(out_path)

    batch_list = np.concatenate(batch_list, axis=1).transpose(1,0,2)  # (num_episodes, episode_length, Dx+Du)

    if data_config["saving"]:
        out_name = os.path.join(out_path, f"data{'_eval' if not training else ''}.p")
        with open(out_name, "wb") as f:
            pickle.dump(batch_list, f)
        print(f"Saved {data_mode} data to {out_name}: shape {batch_list.shape}")


if __name__ == "__main__":
    main()