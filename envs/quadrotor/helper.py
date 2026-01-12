import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_quad_states_actions(state_seq, action_seq, dt, out_path):
    """Plots states and actions in a grid."""
    # state_seq: (T, 12 / 6)
    # action_seq: (T, 3)
    state_dim = state_seq.shape[1]
    assert state_dim == 12 or state_dim == 6, "State dimension must be 12 or 6."
    assert action_seq.shape[1] == 3, "Action dimension must be 3."
    time = np.arange(len(state_seq)) * dt
    if state_dim == 6:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    else:
        fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(f"Quadrotor Telemetry", fontsize=16)
    
    state_names = [
        "Pos N", "Pos E", "Alt", "V Lon", "V Lat", "V Ver",
        "Roll", "Pitch", "Yaw", "Rate R", "Rate P", "Rate Y"
    ]
    action_names = ["Thrust (u1)", "Roll (u2)", "Pitch (u3)"]

    # Plot 12 States
    for i in range(state_dim):
        ax = axes[i // 3, i % 3]
        ax.plot(time, state_seq[:, i], label='Actual')
        ax.set_title(state_names[i])
        ax.grid(True)

    # Plot 3 Actions in the remaining slots
    for i in range(3):
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