# data/dataloader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Dict

import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class TrajectoryDataset:
    """
    Holds trajectories and (optionally) normalization stats.

    X_traj: (E, H+1, Dx) states
    U_traj: (E, H, Du)   actions
    """
    X_traj: np.ndarray
    U_traj: np.ndarray
    stats: Optional[Dict[str, np.ndarray]] = None  # {"x_mean","x_std","u_mean","u_std"}

    @property
    def E(self) -> int: return int(self.X_traj.shape[0])
    @property
    def H(self) -> int: return int(self.U_traj.shape[1])
    @property
    def Dx(self) -> int: return int(self.X_traj.shape[2])
    @property
    def Du(self) -> int: return int(self.U_traj.shape[2])

    def split_episodes(self, train_ratio: float = 0.9, *, seed: int = 0
                       ) -> Tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Randomly split by episodes (not by transitions)."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.E)
        n_tr = int(round(train_ratio * self.E))
        tr_idx, va_idx = perm[:n_tr], perm[n_tr:]
        tr = TrajectoryDataset(self.X_traj[tr_idx].copy(), self.U_traj[tr_idx].copy(), None)
        va = TrajectoryDataset(self.X_traj[va_idx].copy(), self.U_traj[va_idx].copy(), None)
        return tr, va

    def fit_standardizer(self) -> Dict[str, np.ndarray]:
        """
        Compute per-dimension mean/std over *all* time steps and episodes.
        Returns numpy dict; also stored to self.stats.
        """
        x = self.X_traj.reshape(-1, self.Dx)         # (E*(H+1), Dx)
        u = self.U_traj.reshape(-1, self.Du)         # (E*H, Du)
        x_mean, x_std = x.mean(0), x.std(0) + 1e-8
        u_mean, u_std = u.mean(0), u.std(0) + 1e-8
        self.stats = {
            "x_mean": x_mean.astype(np.float32),
            "x_std":  x_std.astype(np.float32),
            "u_mean": u_mean.astype(np.float32),
            "u_std":  u_std.astype(np.float32),
        }
        return self.stats

    def apply_standardizer(self, stats: Dict[str, np.ndarray]) -> "TrajectoryDataset":
        """Return a *new* dataset normalized with provided stats."""
        Xm = (self.X_traj - stats["x_mean"]) / stats["x_std"]
        Um = (self.U_traj - stats["u_mean"]) / stats["u_std"]
        return TrajectoryDataset(Xm.astype(np.float32), Um.astype(np.float32), stats)


class TrajectoryDataLoader:
    """
    Windowed, episode-aware, fully vectorized batch iterator.

    Each batch returns:
      X: (B, T+1, Dx)   states
      U: (B, T,   Du)   actions
      Y: (B, T,   Dx)   next states (shifted: X[:,1:])

    Windows are drawn uniformly across all episodes and valid start indices.
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        seq_len: int,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        assert seq_len >= 1, "seq_len must be >= 1"
        self.ds = dataset
        self.T = int(seq_len)
        self.B = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.key = jax.random.PRNGKey(seed)

        E, H = self.ds.E, self.ds.H
        # valid start indices per episode: s in [0, H - T]
        W = max(0, H - self.T + 1)
        if W == 0:
            raise ValueError(f"seq_len={self.T} is longer than episode length H={H}.")
        # all (episode, start) pairs
        ep = np.repeat(np.arange(E, dtype=np.int32), W)
        st = np.tile(np.arange(W, dtype=np.int32), E)
        self.pairs = np.stack([ep, st], axis=1)  # (N, 2)
        self.N = self.pairs.shape[0]

    def __len__(self) -> int:
        if self.drop_last:
            return self.N // self.B
        return (self.N + self.B - 1) // self.B

    def epoch(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Iterate once over all windows.

        Yields:
          X: (B, 1, Dx), U: (B, T, Du), Y: (B, T, Dx).
        """
        key = self.key
        if self.shuffle:
            key, sk = jax.random.split(key)
            perm = np.array(jax.random.permutation(sk, self.N))
        else:
            perm = np.arange(self.N)

        for i in range(0, self.N, self.B):
            idx = perm[i:i + self.B]
            if len(idx) < self.B and self.drop_last:
                break
            batch = self._gather(idx)
            X, U = batch
            Y = X[:, 1:, :]  # next states
            # X = X[:, 0:1, :]  # initial states

            yield X, U, Y

        self.key = key  # persist PRNG

    # -------- internals --------
    def _gather(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized gather using advanced indexing + take_along_axis."""
        ep_idx = self.pairs[idx, 0]                 # (B,)
        st_idx = self.pairs[idx, 1]                 # (B,)

        # Shapes
        B = ep_idx.shape[0]
        T = self.T
        Dx, Du = self.ds.Dx, self.ds.Du

        # Select episodes
        X_ep = self.ds.X_traj[ep_idx]               # (B, H+1, Dx)
        U_ep = self.ds.U_traj[ep_idx]               # (B, H,   Du)

        # Time indices per sample
        tX = st_idx[:, None] + np.arange(T + 1)[None, :]  # (B, T+1)
        tU = st_idx[:, None] + np.arange(T)[None, :]      # (B, T)

        # Gather with take_along_axis on time dimension
        X = np.take_along_axis(X_ep, tX[..., None], axis=1).reshape(B, T + 1, Dx)
        U = np.take_along_axis(U_ep, tU[..., None], axis=1).reshape(B, T, Du)
        return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(U, dtype=jnp.float32)


# -------- convenience functions --------

def load_npz_as_dataset(npz_path: str) -> TrajectoryDataset:
    """
    Expect npz with:
      X_traj: (E,H+1,Dx)
      U_traj: (E,H,Du)
      (optional) ctrl_dt, ode_dt
    """
    Z = np.load(npz_path)
    X_traj = Z["X_traj"]
    U_traj = Z["U_traj"]
    return TrajectoryDataset(X_traj=X_traj, U_traj=U_traj, stats=None)


def build_loaders_from_npz(
    npz_path: str,
    seq_len: int,
    batch_size: int,
    train_ratio: float = 0.9,
    *,
    normalize: bool = True,
    seed: int = 0,
) -> Tuple[TrajectoryDataLoader, TrajectoryDataLoader, Dict[str, np.ndarray]]:
    """
    High-level helper: load -> split -> (optional) normalize -> build loaders.
    Returns (train_loader, val_loader, stats_dict).
    """
    full = load_npz_as_dataset(npz_path)
    train_ds, val_ds = full.split_episodes(train_ratio=train_ratio, seed=seed)

    stats = None
    if normalize:
        stats = train_ds.fit_standardizer()
        train_ds = train_ds.apply_standardizer(stats)
        val_ds = val_ds.apply_standardizer(stats)

    train_loader = TrajectoryDataLoader(train_ds, seq_len=seq_len, batch_size=batch_size,
                                        shuffle=True, drop_last=True, seed=seed)
    val_loader = TrajectoryDataLoader(val_ds, seq_len=seq_len, batch_size=batch_size,
                                      shuffle=False, drop_last=False, seed=seed+1)
    return train_loader, val_loader, (stats or {})


# -------- example usage --------
if __name__ == "__main__":
    path = "output/data/single_pendulum.npz"
    tr_loader, va_loader, stats = build_loaders_from_npz(
        path, seq_len=10, batch_size=128, train_ratio=0.9, normalize=True, seed=0
    )
    print("Train iters per epoch:", len(tr_loader))
    for X, U, Y in tr_loader.epoch():
        print("Batch shapes:", X.shape, U.shape, Y.shape)
        break
