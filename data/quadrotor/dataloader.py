from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Dict, Optional, Union
from typing import Iterator
import pickle, numpy as np
import jax
import jax.numpy as jnp

@dataclass
class DynamicsDataset:
    obs: jnp.ndarray         # [M, n_sample, 12/6]
    act: jnp.ndarray         # [M, n_sample, 3/4]
    weights: jnp.ndarray     # [M, n_roll]

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def get(self, idx: Union[int, jnp.ndarray, np.ndarray]) -> Dict[str, jnp.ndarray]:
        return {
            "observations": self.obs[idx],
            "actions":      self.act[idx][:, :-1, :],
            "weights":      self.weights[idx],
        }

# ---------- main loader ----------
def load_dynamics_dataset(data_cfg: dict, train_cfg: dict,
                              phase: str,
                              seed: int = 0) -> DynamicsDataset:
    # ---- Config & dims ----
    n_his  = int(train_cfg["n_history"])
    n_roll = int(train_cfg["horizon_scheduler"]["T_final"] if phase == "train" else train_cfg["n_rollout_valid"])
    train_ratio = float(train_cfg["train_valid_ratio"])
    noise_std   = float(train_cfg["noise"]) if phase == "train" else 0.0
    augment_en  = bool(train_cfg["data_augment"])

    # ---- Load and normalize ----
    data_file_name = os.path.join(train_cfg["data_dir"], "data.p")
    with open(data_file_name, "rb") as fp:
        episodes = pickle.load(fp)  # [B, T, state_dim + action_dim]

    scale = float(data_cfg["scale"])
    episodes = np.array(episodes, dtype=np.float32) / scale

    T_ep = episodes.shape[1]
    if episodes.shape[2] == 18:
        state_dim = 12 + 3  # pos, vel, rpy, rates, vel_cmd
        action_dim = 3
    elif episodes.shape[2] == 19:
        state_dim = 12 + 3  # pos, vel, rpy, rates, vel_cmd
        action_dim = 4
    elif episodes.shape[2] == 9:
        state_dim = 6  # pos, vel
        action_dim = 3
    else:
        raise ValueError(f"Unknown data dimension: {episodes.shape[2]}")

    num_train = int(len(episodes) * train_ratio)
    if phase == "train":
        episodes = episodes[:num_train]
    elif phase == "valid":
        episodes = episodes[num_train:]
    else:
        raise AssertionError(f"Unknown phase {phase}")

    # Adjust n_roll and n_sample
    n_roll   = min(T_ep - n_his, n_roll)
    n_sample = n_his + n_roll

    # ---- Windowing ----
    obs_list, act_list = [], []
    for ep in episodes:
        for i in range(T_ep - n_sample + 1):
            win = ep[i:i + n_sample]                    # [n_sample, D]
            obs_list.append(win[:, :state_dim])           # [n_sample, 2K]
            act_list.append(win[:, -action_dim:])          # [n_sample, 2]
    
    obs = np.asarray(obs_list, dtype=np.float32)        # [M, n_sample, 2K]
    act = np.asarray(act_list, dtype=np.float32)        # [M, n_sample, 2]

    # ---- Train-time noise on observations ----
    if phase == "train" and noise_std > 0.0:
        np.random.seed(seed)
        obs = obs + np.random.normal(0.0, noise_std, size=obs.shape).astype(obs.dtype)

    # ---- Train-time augmentation (single shared rotation) ----
    if phase == "train" and augment_en:
        pass
    # ---- Weights (ones) ----
    M = obs.shape[0]
    weights = np.ones((M, n_roll), dtype=np.float32)

    # ---- Final shuffle (train & valid) ----
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    obs, act, weights = obs[perm], act[perm], weights[perm]

    # ---- Cast to JAX ----
    return DynamicsDataset(
        obs=jnp.asarray(obs),
        act=jnp.asarray(act),
        weights=jnp.asarray(weights),
    )

class Dataloader:
    """
    Minimal JAX-friendly dataloader for single-device training.

    - Iterating over the dataloader yields exactly ONE epoch.
    - Shuffles indices each epoch with an internal PRNGKey (unless shuffle=False).
    - Returns batches as dicts with keys:
        {"observations", "actions", "weights", "pusher_pos", "indices"}
    - Works with the provided DynamicsDataset (dataset.get(...) slices JAX arrays).

    Example:
        ds = load_dynamics_dataset(config, phase="train")
        dl = Dataloader(ds, batch_size=128, seed=0, shuffle=True, drop_last=True)
        for batch in dl:  # one epoch
            train_step(batch)
        # next epoch:
        for batch in dl:
            train_step(batch)
    """

    def __init__(self,
                 dataset,
                 batch_size: int,
                 *,
                 seed: int = 0,
                 shuffle: bool = True,
                 drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._key = jax.random.PRNGKey(int(seed))

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n // self.batch_size) if self.drop_last else ( (n + self.batch_size - 1) // self.batch_size )

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        # advance RNG for this epoch
        self._key, sub = jax.random.split(self._key)
        N = len(self.dataset)

        if self.shuffle:
            perm = jax.random.permutation(sub, N)
        else:
            perm = jnp.arange(N)

        if self.drop_last:
            limit = (N // self.batch_size) * self.batch_size
            perm = perm[:limit]

        # yield contiguous slices of the permutation
        for start in range(0, perm.shape[0], self.batch_size):
            idx = perm[start:start + self.batch_size]  # [B]
            batch = self.dataset.get(idx)              # dict of jnp arrays
            batch["indices"] = idx
            yield batch

def build_loaders(
    data_cfg: dict, train_cfg: dict, train_mode: str, seed: int = 0
) -> tuple[ Dataloader, Dataloader]:
    """
    Builds train and valid dataloaders from config.
    Also returns the full dataset stats (DynamicsDataset) for reference.
    """
    ds_train = load_dynamics_dataset(data_cfg, train_cfg, seed=seed, phase="train")
    ds_valid = load_dynamics_dataset(data_cfg, train_cfg, seed=seed, phase="valid")

    batch_size = int(train_cfg["batch_size"])

    dl_train = Dataloader(
        dataset=ds_train,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        drop_last=True,
    )

    dl_valid = Dataloader(
        dataset=ds_valid,
        batch_size=batch_size,
        seed=seed + 1,
        shuffle=False,
        drop_last=False,
    )

    return dl_train, dl_valid

if __name__ == "__main__":
    # simple test
    import yaml
    from omegaconf import DictConfig, OmegaConf

    config_path = "configs/quadrotor.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = OmegaConf.create(config)
    data_cfg = config["data"]
    train_mode = config["settings"]["train_mode"]
    tr_cfg = config[f"train_{train_mode}"]
    dl_train, dl_valid = build_loaders(data_cfg, tr_cfg, train_mode, seed=0)