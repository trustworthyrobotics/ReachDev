import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


import jax
import jax.numpy as jnp

def sample_vel_cmd_sequence(
    key: jax.Array,
    amax: float = 1.0,          # max acceleration (m/s^2)
    dt: float = 0.2,            # command update period (s), e.g. 5 Hz
    n_steps: int = 50,         # number of command steps
    v0: jnp.ndarray | None = None,   # (3,) initial velocity command
    v_bounds: jnp.ndarray | None = None,  # (2,3): [[vx_min,vy_min,vz_min],[vx_max,...]]
) -> jnp.ndarray:
    """
    Generate a random piecewise-constant velocity command sequence with bounded acceleration.
    Returns v_cmd_seq of shape (T, 3), where T = int(horizon/dt)+1, including v0 at t=0.
    """
    T = n_steps + 1
    if v0 is None:
        v0 = jnp.zeros((3,), dtype=jnp.float32)
    else:
        v0 = jnp.asarray(v0, dtype=jnp.float32)

    dv_max = amax * dt
    dv = jax.random.uniform(key, (T - 1, 3), minval=-dv_max, maxval=dv_max)  # (T-1,3)

    v_seq = v0[None, :] + jnp.cumsum(dv, axis=0)             # (T-1,3), starts at v0+dv0
    v_seq = jnp.concatenate([v0[None, :], v_seq], axis=0)    # (T,3)
    if v_bounds is not None:
        v_bounds = jnp.asarray(v_bounds, dtype=jnp.float32)
        assert v_bounds.shape == (2, 3)
        v_seq = jnp.clip(v_seq, v_bounds[0], v_bounds[1])
    return v_seq

    # dv_max = amax * dt  # max change in velocity per command tick

    # if v_bounds is not None:
    #     v_bounds = jnp.asarray(v_bounds, dtype=jnp.float32)
    #     assert v_bounds.shape == (2, 3)

    # def step(carry, k):
    #     v = carry
    #     # random delta-v with ||dv||_inf <= dv_max
    #     k, sub = jax.random.split(k)
    #     dv = jax.random.uniform(sub, (3,), minval=-dv_max, maxval=dv_max)
    #     v_next = v + dv
    #     if v_bounds is not None:
    #         v_next = jnp.clip(v_next, v_bounds[0], v_bounds[1])
    #     return v_next, v_next

    # keys = jax.random.split(key, T - 1)
    # _, v_hist = jax.lax.scan(step, v0, keys)  # (T-1,3)
    # v_seq = jnp.concatenate([v0[None, :], v_hist], axis=0)  # (T,3)
    # return v_seq

def plot_quad_states_actions(state_seq, action_seq, dt, out_path):
    """Plots states and actions in a grid."""
    # state_seq: (T, 12 / 6)
    # action_seq: (T, 3 / 4)
    state_dim = state_seq.shape[1]
    assert state_dim == 12 or state_dim == 6, "State dimension must be 12 or 6."
    action_dim = action_seq.shape[1]
    time = np.arange(len(state_seq)) * dt
    if state_dim == 6:
        assert action_dim == 3, "Action dimension must be 3."
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        state_names = [
            "Pos X", "Pos Y", "Pos Z",
            "Vel X", "Vel Y", "Vel Z"
        ]
        action_names = ["Vel Cmd X", "Vel Cmd Y", "Vel Cmd Z"]
    else:
        assert action_dim == 3 or action_dim == 4, "Action dimension must be 3 or 4."
        if action_dim == 3:
            fig, axes = plt.subplots(5, 3, figsize=(15, 20))
        else:
            fig, axes = plt.subplots(5, 4, figsize=(20, 20))
        # state_names = [
        #     "Pos N", "Pos E", "Alt", "V Lon", "V Lat", "V Ver",
        #     "Roll", "Pitch", "Yaw", "Rate R", "Rate P", "Rate Y"
        # ]
        state_names = [
            "Pos X", "Pos Y", "Pos Z", "Vel X", "Vel Y", "Vel Z",
            "Roll", "Pitch", "Yaw", "Rate Roll", "Rate Pitch", "Rate Yaw"
        ]
        action_names = ["Thrust (u1)", "Roll (u2)", "Pitch (u3)", "Yaw (u4)"]

    fig.suptitle(f"Quadrotor Telemetry", fontsize=16)
    

    # Plot 12 States
    for i in range(state_dim):
        ax = axes[i // 3, i % 3]
        ax.plot(time, state_seq[:, i], label='Actual')
        ax.set_title(state_names[i])
        ax.grid(True)

    # Plot 3 Actions in the remaining slots
    for i in range(action_dim):
        ax = axes[-1, i]
        ax.step(time, action_seq[:, i], where='post', color='r')
        ax.set_title(action_names[i])
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path)
    print(f"Telemetry plot saved to {out_path}")
    plt.close()


def plot_3d_trajectories(pose_seqs, num_quads, dt, out_path): 
    # pose_seqs: (T, num_quads, 3)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    num_steps = pose_seqs.shape[0]
    time_indices = np.linspace(0, num_steps * dt, num_steps)

    for q_id in range(num_quads):
        # Extract [x, y, z] trajectory
        traj = pose_seqs[:, q_id, :]  # (T, 3)
        
        # Create segments: [[p0, p1], [p1, p2], ..., [pn-1, pn]]
        points = traj.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create the collection with the rainbow colormap
        lc = Line3DCollection(segments, cmap='rainbow', linewidth=2)
        lc.set_array(time_indices) # Map colors to time
        
        line = ax.add_collection3d(lc)

    # Auto-scale limits (Collections don't trigger auto-scale)
    all_traj = pose_seqs.reshape(-1, 3)
    ax.set_xlim(all_traj[..., 0].min(), all_traj[..., 0].max())
    ax.set_ylim(all_traj[..., 1].min(), all_traj[..., 1].max())
    ax.set_zlim(all_traj[..., 2].min(), all_traj[..., 2].max())

    ax.set_xlabel('North (x1)')
    ax.set_ylabel('East (x2)')
    ax.set_zlabel('Altitude (x3)')
    
    # Add a colorbar to show time progression
    cbar = fig.colorbar(line, ax=ax, fraction=0.02, pad=0.1)
    cbar.set_label('Normalized Time')

    plt.savefig(out_path)
    print(f"3D trajectory plot saved to {out_path}")
    plt.close()