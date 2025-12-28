import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from io import BytesIO
import os
import pickle
import jax.numpy as jnp
from jax import random as jrandom
import equinox as eqx
import hydra
from omegaconf import DictConfig
import sys
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DTPlanReach
from CROWN_Reach.src.utils.vis import visualize_flowpipe_time
from models.dynamics import load_t_dynamics_model

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
def main(config: DictConfig):
    data_dir = config["data"]["out_path"]
    model_dir = "output/runs/T_pushing/linear_cos_128_mid_1_0.6_eps0.05_0.02_w0.001_j0.0_20251228_144642"
    scale = float(config["data"].get("scale", 1.0))  # data was normalized by /scale

    eval_p_path = os.path.join(data_dir, "data_eval.p")
    model_path = os.path.join( model_dir, "last_model.eqx")
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

    # Everything inside file is normalized by /scale → denormalize for visualization
    eps_denorm = eps_arr.astype(np.float32)               # [B,T,12], unnormalized
    eps_norm = eps_denorm / scale                    # [B,T,12], normalized

    selected_eps_idx = 0
    state_init = jnp.array(eps_norm[selected_eps_idx, 0, :state_dim])[None]      # [1, Dx]
    action_seq = jnp.array(eps_norm[selected_eps_idx, :T_reach, -action_dim:])[None]      # [1, T, Du]

    reach_eps = float(config["train"]["reach"]["eps_final"])
    state_init_lo = state_init - reach_eps
    state_init_up = state_init + reach_eps
    z_init_lo = jnp.concatenate([state_init_lo, jnp.zeros((1, action_dim))], axis=-1) # [1, Dx+Du]
    z_init_up = jnp.concatenate([state_init_up, jnp.zeros((1, action_dim))], axis=-1) # [1, Dx+Du]

    ts, r_lo, r_up, _, _ = reach_analyzer.verify(z_init_lo, z_init_up, n_total_steps=T_reach, action_seq=action_seq.repeat(z_init_up.shape[0]//action_seq.shape[0], axis=0)[:, None])
    r_lo = r_lo.reshape(-1, T_reach + 1, state_dim+action_dim)  # [B, T+1, Dx+Du]
    r_up = r_up.reshape(-1, T_reach + 1, state_dim+action_dim)  # [B, T+1, Dx+Du]

    key = jrandom.PRNGKey(config["settings"]["seed"])
    n_samples = 64
    sample_state_init = state_init_lo + (state_init_up - state_init_lo) * jrandom.uniform(key, shape=(n_samples, state_init_lo.shape[1]))
    sample_rollout = model.rollout_model(sample_state_init, action_seq.repeat(n_samples, axis=0))
    sample_rollout = jnp.concatenate([sample_state_init[:, None, :], sample_rollout], axis=1)  # [n_samples, T+1, Dx]

    out_dir = os.path.join("output", "vis", "T_pushing")
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