# data/generate_single_pendulum.py
import argparse
import yaml
import numpy as np
import os
import jax
import jax.numpy as jnp
from jax import lax
from envs.single_pendulum import SinglePendulumEnv
from vis.pendulum import save_pendulum_gif


def _assert(dcfg):
    # Validate state/action dims for this env
    if not (dcfg["state_dim"] == 2 and dcfg["action_dim"] == 1):
        raise ValueError("SinglePendulum expects state_dim=2 and action_dim=1.")

    lo = jnp.array([dcfg["initial_set"][0][0], dcfg["initial_set"][1][0]])
    hi = jnp.array([dcfg["initial_set"][0][1], dcfg["initial_set"][1][1]])
    cond = (lo[0] <= 1.0) & (hi[0] >= 1.175) & (lo[1] <= 0.0) & (hi[1] >= 0.2)
    if not bool(cond):
        raise ValueError(
            "initial_set must strictly cover [1.0,1.175] × [0.0,0.2]. "
            f"Got θ∈[{float(lo[0])},{float(hi[0])}], θ̇∈[{float(lo[1])},{float(hi[1])}]."
        )


def batched_rollout(env: SinglePendulumEnv, dcfg: dict, key: jax.Array):
    """
    Returns:
      X_traj: (E, H+1, 2)
      U_traj: (E, H, 1)
    """
    E = int(dcfg["num_episodes"])
    H = int(dcfg["episode_length"])
    # ctrl_dt = float(dcfg["control_step_size"])
    # ode_dt = float(dcfg["ode_step_size"])
    # n_steps_per_ctrl = int(jnp.ceil(ctrl_dt / ode_dt))
    act_lo = jnp.array([interval[0] for interval in dcfg["action_range"]])
    act_hi = jnp.array([interval[1] for interval in dcfg["action_range"]])

    # Sample all initial states & actions once (batched)
    key, k_init, k_act = jax.random.split(key, 3)
    X0 = env.sample_initial_states(k_init, E)                      # (E, 2)
    U_traj = jax.random.uniform(k_act, (E, H, 1), minval=act_lo, maxval=act_hi)
    U_traj = env.clip_action(U_traj)                               # (E, H, 1)

    # Time-major actions for lax.scan: (H, E, 1)
    U_tm = jnp.swapaxes(U_traj, 0, 1)

    def scan_body(x_t, u_t):  # x_t: (E,2), u_t: (E,1)
        x_tp1 = env.step(x_t, u_t)
        return x_tp1, x_tp1    # carry, y

    # Run a single scan over horizon
    X_seq_tm, X_seq_tm_out = lax.scan(scan_body, X0, U_tm)         # both (H, E, 2)
    # Prepend initial state to get H+1 states, time-major → episode-major
    X_all_tm = jnp.concatenate([X0[None, ...], X_seq_tm_out], axis=0)  # (H+1, E, 2)
    X_traj = jnp.swapaxes(X_all_tm, 0, 1)                               # (E, H+1, 2)

    return X_traj, U_traj


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]


    _assert(dcfg)
    env = SinglePendulumEnv.from_config(cfg)

    # Lightweight dict of scalars for JIT (avoids tracing large dicts)
    dcfg_flat = {
        "num_episodes": int(dcfg["num_episodes"]),
        "episode_length": int(dcfg["episode_length"]),
        "control_step_size": float(dcfg["control_step_size"]),
        "action_range": [[float(a) for a in act_range] for act_range in dcfg["action_range"]],
    }

    key = jax.random.PRNGKey(0)
    # Use the non-jitted path (already vectorized). If you prefer, uncomment the jitted one.
    X_traj, U_traj = batched_rollout(env, dcfg_flat, key)
    # Or: X_traj, U_traj = jax.jit(batched_rollout)(env, dcfg_flat, key)

    out_path = dcfg["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        X_traj=np.array(X_traj, dtype=np.float32),
        U_traj=np.array(U_traj, dtype=np.float32),
        ode_dt=np.array(dcfg["ode_step_size"], dtype=np.float32),
        ctrl_dt=np.array(dcfg["control_step_size"], dtype=np.float32),
    )
    print(f"Saved trajectories to {out_path}")
    print(f"X_traj shape: {tuple(X_traj.shape)}, U_traj shape: {tuple(U_traj.shape)}")

    gif_path = dcfg.get("gif_path", "output/single_pendulum.gif")
    save_pendulum_gif(
        X_traj=np.array(X_traj),
        U_traj=np.array(U_traj),
        out_path=gif_path,
        L=env.L,
        ctrl_dt=env.ctrl_dt,
        episode_idx=0,
    )
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/single_pendulum.yaml")
    args = parser.parse_args()
    main(args.config)
