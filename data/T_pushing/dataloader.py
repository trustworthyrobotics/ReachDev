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
    obs: jnp.ndarray         # [M, n_sample, 2K]         (relative keypoints)
    act: jnp.ndarray         # [M, n_sample, 2]          (pusher velocity vx, vy)
    pusher_pos: jnp.ndarray  # [M, n_sample, 2]          (xp, yp)
    weights: jnp.ndarray     # [M, n_roll]

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def get(self, idx: Union[int, jnp.ndarray, np.ndarray]) -> Dict[str, jnp.ndarray]:
        return {
            "observations": self.obs[idx], # [B, n_sample, 2K]
            "actions":      self.act[idx][:, :-1, :], # [B, n_sample-1, 2]
            "weights":      self.weights[idx],
            "pusher_pos":   self.pusher_pos[idx],
        }

    def update_weights(self,
                       indices: Union[np.ndarray, jnp.ndarray],
                       new_weight: Union[np.ndarray, jnp.ndarray],
                       weight_ub: float) -> "DynamicsDataset":
        W = np.array(self.weights)
        if indices.ndim == 1:
            W[indices] *= np.reshape(np.array(new_weight), (-1, 1))
            W[indices] = np.clip(W[indices], 1.0, weight_ub)
        elif indices.ndim == 2:
            i = indices[:, 0]; j = indices[:, 1]
            W[i, j] *= np.array(new_weight)
            W[i, j] = np.clip(W[i, j], 1.0, weight_ub)
        else:
            raise AssertionError("Unknown indices shape")
        return DynamicsDataset(
            obs=self.obs, act=self.act, pusher_pos=self.pusher_pos,
            weights=jnp.asarray(W, dtype=self.weights.dtype),
        )

# @dataclass
# class ControlDataset:
#     obs: jnp.ndarray         # [M, Dx]         (relative keypoints)
#     act: jnp.ndarray         # [M, Du]          (pusher velocity vx, vy)
#     pusher_pos: jnp.ndarray  # [M, 2]          (xp, yp)
#     weights: jnp.ndarray     # [M, 1]
#     targets: jnp.ndarray      # [M, Dx] Target state at t + K
#     ref_actions: jnp.ndarray  # [M, Du] Mean action over [t, t + K]

#     def get(self, idx: Union[int, jnp.ndarray, np.ndarray]) -> Dict[str, jnp.ndarray]:
#         return {
#             "observations": self.obs[idx],
#             "actions":      self.act[idx],
#             "weights":      self.weights[idx],
#             "pusher_pos":   self.pusher_pos[idx],
#             "targets": self.targets[idx],
#             "ref_actions": self.ref_actions[idx],
#         }

# ---------- helpers: 2D rotation for stacked (x,y) pairs ----------
def _rotate_xy_pairs(arr: np.ndarray, theta: float) -> np.ndarray:
    """
    arr: [..., 2*K] where consecutive pairs are (x,y).
    Rotates each (x,y) pair by theta.
    """
    if arr.shape[-1] % 2 != 0:
        raise AssertionError("Last dim must be even: pairs of (x,y)")
    c, s = np.cos(theta), np.sin(theta)
    x = arr[..., 0::2]
    y = arr[..., 1::2]
    xr = c * x - s * y
    yr = s * x + c * y
    out = arr.copy()
    out[..., 0::2] = xr
    out[..., 1::2] = yr
    return out

def _rotate_2d(arr: np.ndarray, theta: float) -> np.ndarray:
    """
    arr: [..., 2] (xp,yp) or (vx,vy)
    """
    c, s = np.cos(theta), np.sin(theta)
    x = arr[..., 0]
    y = arr[..., 1]
    out = arr.copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out

def _maybe_augment_train(obs: np.ndarray,
                         pusher_pos: np.ndarray,
                         act: np.ndarray,
                         *,
                         enabled: bool,
                         seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply ONE shared random rotation theta to obs (2K pairs), pusher_pos (2), act (2).
    """
    if not enabled:
        return obs, pusher_pos, act
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    obs_r = _rotate_xy_pairs(obs, theta)          # [M, T, 2K]
    pos_r = _rotate_2d(pusher_pos, theta)         # [M, T, 2]
    act_r = _rotate_2d(act, theta)                # [M, T, 2]
    return obs_r, pos_r, act_r

# ---------- main loader ----------
def load_dynamics_dataset(data_cfg: dict, train_cfg: dict,
                              phase: str,
                              seed: int = 0) -> DynamicsDataset:
    """
    Per-timestep vector (pre-normalized here):
      [x1-xp, y1-yp, ..., xK-xp, yK-yp, xp, yp, vx, vy]  ==  [rel(2K), pos(2), vel(2)]
    """

    # ---- Config & dims ----
    n_his  = int(train_cfg["n_history"])
    n_roll = int(train_cfg["horizon_scheduler"]["T_final"] if phase == "train" else train_cfg["n_rollout_valid"])
    train_ratio = float(train_cfg["train_valid_ratio"])
    noise_std   = float(train_cfg["noise"]) if phase == "train" else 0.0
    augment_en  = bool(train_cfg["data_augment"])
    pred_mode   = str(train_cfg["pred_mode"])
    assert pred_mode in ["state", "pose"], f"Unknown pred_mode {pred_mode}"

    # ---- Load and normalize ----
    data_file_name = os.path.join(train_cfg["data_dir"], "data.p")
    with open(data_file_name, "rb") as fp:
        episodes = pickle.load(fp)  # list of [T, 2K+4]

    scale = float(data_cfg["scale"])
    episodes = [ep.astype(np.float32) / scale for ep in episodes]

    T_ep = int(episodes[0].shape[0])

    num_train = int(len(episodes) * train_ratio)
    if phase == "train":
        episodes = episodes[:num_train]
    elif phase == "valid":
        episodes = episodes[num_train:]
    else:
        raise AssertionError(f"Unknown phase {phase}")
    state_dim = int(data_cfg["state_dim"])
    pose_dim = int(data_cfg["pose_dim"])
    action_dim = int(data_cfg["action_dim"])

    # Adjust n_roll and n_sample
    n_roll   = min(T_ep - n_his, n_roll)
    n_sample = n_his + n_roll

    # ---- Windowing ----
    obs_list, pusher_pos_list, act_list = [], [], []
    for ep in episodes:
        for i in range(T_ep - n_sample + 1):
            win = ep[i:i + n_sample]                    # [n_sample, D]
            if pred_mode == "state":
                obs_list.append(win[:, :state_dim])           # [n_sample, 2K]
            elif pred_mode == "pose":
                obs_list.append(win[:, state_dim:state_dim+pose_dim])     # [n_sample, pose_dim]
            pusher_pos_list.append(win[:, state_dim+pose_dim:-action_dim])  # [n_sample, 2]
            act_list.append(win[:, -action_dim:])          # [n_sample, 2]
    
    obs = np.asarray(obs_list, dtype=np.float32)        # [M, n_sample, 2K]
    if pred_mode == "pose":
        obs[:, :, -1] *= scale  # scale back the theta dimension
    pusher_pos = np.asarray(pusher_pos_list, dtype=np.float32) # [M, n_sample, 2]
    act = np.asarray(act_list, dtype=np.float32)        # [M, n_sample, 2]

    # ---- Train-time noise on observations ----
    if phase == "train" and noise_std > 0.0:
        np.random.seed(seed)
        obs = obs + np.random.normal(0.0, noise_std, size=obs.shape).astype(obs.dtype)

    # ---- Train-time augmentation (single shared rotation) ----
    if phase == "train" and augment_en:
        # Use a deterministic seed derived from config['settings']['seed'] but offset so it doesn't
        # collide with the final shuffle’s RNG usage.
        seed_aug = seed * 1664525 + 1013904223  # LCG-style mix
        obs, pusher_pos, act = _maybe_augment_train(obs, pusher_pos, act,
                                                    enabled=True, seed=seed_aug)

    # ---- Weights (ones) ----
    M = obs.shape[0]
    weights = np.ones((M, n_roll), dtype=np.float32)

    # ---- Final shuffle (train & valid) ----
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    obs, act, pusher_pos, weights = obs[perm], act[perm], pusher_pos[perm], weights[perm]

    # ---- Cast to JAX ----
    return DynamicsDataset(
        obs=jnp.asarray(obs),
        act=jnp.asarray(act),
        pusher_pos=jnp.asarray(pusher_pos),
        weights=jnp.asarray(weights),
    )

# def load_controller_dataset(data_cfg: dict, train_cfg: dict,
#                  phase: str,
#                  seed: int = 0) -> Union[DynamicsDataset, ControlDataset]:
    
#     # 1. Reuse existing loading/normalization logic
#     scale = float(data_cfg["scale"])
#     data_file_name = os.path.join(train_cfg["data_dir"], "data.p")
#     with open(data_file_name, "rb") as fp:
#         episodes = [ep.astype(np.float32) / scale for ep in pickle.load(fp)]

#     # 2. Train/Valid split
#     train_ratio = float(train_cfg["train_valid_ratio"])
#     num_train = int(len(episodes) * train_ratio)
#     episodes = episodes[:num_train] if phase == "train" else episodes[num_train:]

#     state_dim = int(data_cfg["state_dim"])
#     pose_dim = int(data_cfg["pose_dim"])
#     action_dim = int(data_cfg["action_dim"])
#     pred_mode = str(train_cfg["pred_mode"])

#     # 3. Windowing Logic
#     obs_list, pusher_pos_list, act_list = [], [], []
#     target_list, ref_act_list = [], []

#     # Calculate target window K (e.g., 10 steps)
#     # ctl_frequency=10, target_frequency=1 => K=10
#     K = int(train_cfg["ctl_frequency"] // train_cfg["target_frequency"])

#     for ep in episodes:
#         T_ep = ep.shape[0]
        
#         for i in range(1, T_ep):
#             if pred_mode == "state":
#                 target_obs = ep[i, :state_dim]
#             else:
#                 target_obs = ep[i, state_dim : state_dim + pose_dim]
            
#             min_curr_step = max(0, i - K)
#             for j in range(min_curr_step, i):
#                 if pred_mode == "state":
#                     curr_obs = ep[j, :state_dim]
#                 else:
#                     curr_obs = ep[j, state_dim : state_dim + pose_dim]

#                 obs_list.append(curr_obs)
#                 pusher_pos_list.append(ep[j, state_dim + pose_dim : -action_dim])
#                 act_list.append(ep[j, -action_dim:])
#                 target_list.append(target_obs)
#                 # select the average action over a window with size up to K.
#                 # if i - K >= 0 and j is a intermediate step, we still take the average from i-K to i
#                 avg_act = np.mean(ep[min(j, min_curr_step) : i, -action_dim:], axis=0)
#                 ref_act_list.append(avg_act)

#     # 4. Conversion and Shuffling
#     obs = np.asarray(obs_list)
#     if pred_mode == "pose": 
#         obs[..., -1] *= scale # Scale back theta
    
#     act = np.asarray(act_list)
#     pusher_pos = np.asarray(pusher_pos_list)
#     M = obs.shape[0]
    
#     rng = np.random.default_rng(seed)
#     perm = rng.permutation(M)

#     targets = np.asarray(target_list)
#     if pred_mode == "pose": targets[..., -1] *= scale
#     ref_actions = np.asarray(ref_act_list)
    
#     return ControlDataset(
#         obs=jnp.asarray(obs[perm]),
#         act=jnp.asarray(act[perm]),
#         pusher_pos=jnp.asarray(pusher_pos[perm]),
#         weights=jnp.ones((M)), # Controller weights are always 1
#         targets=jnp.asarray(targets[perm]),
#         ref_actions=jnp.asarray(ref_actions[perm])
#     )


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
    # if train_mode in {"dt_dyn", "ct_dyn"}:
    #     ds_train = load_dynamics_dataset(data_cfg, train_cfg, seed=seed, phase="train")
    #     ds_valid = load_dynamics_dataset(data_cfg, train_cfg, seed=seed, phase="valid")
    # elif train_mode == "ct_ctl":
    #     ds_train = load_controller_dataset(data_cfg, train_cfg, seed=seed, phase="train")
    #     ds_valid = load_controller_dataset(data_cfg, train_cfg, seed=seed, phase="valid")
    # else:
    #     raise ValueError(f"Unknown train_mode: {train_mode}")
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

    config_path = "configs/T_pushing.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = OmegaConf.create(config)
    data_cfg = config["data"]
    train_mode = config["settings"]["train_mode"]
    tr_cfg = config[f"train_{train_mode}"]
    dl_train, dl_valid = build_loaders(data_cfg, tr_cfg, train_mode, seed=0)