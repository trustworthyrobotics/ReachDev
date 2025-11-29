# vis/pendulum.py
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def save_pendulum_gif(
    X_traj: np.ndarray,
    U_traj: np.ndarray,
    out_path: str,
    *,
    L: float = 0.5,
    ctrl_dt: float = 0.05,
    episode_idx: int = 0,
    dpi: int = 120,
    fps: Optional[int] = None,
    figsize=(4, 4),
):
    """
    Save a single-episode trajectory as an animated GIF.

    Args:
        X_traj: (E, H+1, 2) array, states [theta, theta_dot]
        U_traj: (E, H, 1) array, actions [T]
        out_path: path to write the GIF (e.g., "out/pendulum.gif")
        L: pendulum length (for drawing only)
        ctrl_dt: timestep between actions (seconds)
        episode_idx: which episode to render
        dpi: figure DPI for frames
        fps: frames per second in GIF (defaults to round(1/ctrl_dt))
        figsize: matplotlib figure size
    """
    assert X_traj.ndim == 3 and X_traj.shape[-1] == 2, "X_traj must be (E,H+1,2)"
    assert U_traj.ndim == 3 and U_traj.shape[-1] == 1, "U_traj must be (E,H,1)"
    E, H1, _ = X_traj.shape
    E2, H, _ = U_traj.shape
    assert E == E2 and H1 == H + 1, "Shapes must satisfy X:(E,H+1,2), U:(E,H,1)"

    ep = int(episode_idx)
    if not (0 <= ep < E):
        raise IndexError(f"episode_idx out of range: 0 <= {ep} < {E}")

    thetas = X_traj[ep, :, 0]         # (H+1,)
    theta_dots = X_traj[ep, :, 1]     # (H+1,)
    torques = U_traj[ep, :, 0]        # (H,)

    if fps is None:
        fps = max(1, int(round(1.0 / float(ctrl_dt))))
    duration = 1.0 / fps  # seconds per frame (imageio uses duration per frame)

    # World extents (slightly larger than pendulum sweep)
    r = 1.2 * L
    xlim = (-r, r)
    ylim = (-r, r)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames = []

    # Pre-create figure; we’ll redraw each frame for simplicity and clarity
    for k in range(H + 1):
        theta = float(thetas[k])
        theta_dot = float(theta_dots[k])
        # position of bob; θ measured from UPWARD vertical (θ=0 is upright)
        x = L * np.sin(theta)
        y = L * np.cos(theta)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Single Pendulum", pad=8)

        # pivot
        ax.plot(0, 0, "ko", ms=4)
        # rod
        ax.plot([0, x], [0, y], lw=3)
        # bob
        ax.plot(x, y, "o", ms=8)

        # annotation panel
        t_sec = k * ctrl_dt
        T_k = float(torques[k]) if k < H else float("nan")
        text = (
            f"step: {k:3d}\n"
            f"time: {t_sec:6.3f}s\n"
            f"θ: {theta:+8.4f} rad\n"
            f"θ̇: {theta_dot:+8.4f} rad/s\n"
            f"T: {T_k:+8.4f} Nm"
        )
        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, lw=0.5),
            va="bottom", ha="left",
        )

        # tiny ground line for reference
        ax.plot([xlim[0], xlim[1]], [0, 0], "k:", lw=0.5, alpha=0.3)

        fig.canvas.draw()
        # Convert to numpy array (RGB)
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, [1, 2, 3]]  # ARGB → RGB
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_path, frames, duration=duration, loop=0)
    print(f"[GIF] saved {out_path}  (frames={len(frames)}, fps={fps})")
