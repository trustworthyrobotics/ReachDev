import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from envs.quadrotor.quad_sim import Quad_Sim

@hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml", version_base=None)
def main(config: DictConfig) -> None:
    data_config = config["data"]
    data_mode = config["settings"].get("data_mode", "dt_dyn")
    # data_config["frequency"] is the ode frequency limit
    frequency = min(data_config[data_mode]["frequency"], data_config["frequency"])

    data_config[data_mode]["episode_length"] = data_config[data_mode]["episode_length"] * frequency
    data_config[data_mode]["num_episodes"] = data_config[data_mode]["num_episodes"] // frequency


    env = Quad_Sim(data_config=data_config)