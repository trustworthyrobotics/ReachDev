# train.py
from __future__ import annotations
import os
import yaml
import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from models.dynamics import MLPDynamics
from data.dataloader import build_loaders_from_npz
from training.trainer import Trainer


def _save_ckpt(path_base: str, model, opt_state, step: int, cfg: dict, stats: dict):
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    eqx.tree_serialise_leaves(path_base + ".eqx", model)
    np.savez(path_base + ".npz", step=np.array(step), config=np.array([yaml.dump(cfg)]), **(stats or {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/single_pendulum.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    tr_cfg = cfg["train"]

    os.makedirs(tr_cfg["out_dir"], exist_ok=True)
    # loaders
    train_loader, val_loader, stats = build_loaders_from_npz(
        npz_path=data_cfg["out_path"],
        seq_len=tr_cfg["n_rollout"],
        batch_size=tr_cfg["batch_size"],
        train_ratio=tr_cfg["train_valid_ratio"],
        normalize=data_cfg["normalize"],
        seed=cfg["settings"]["seed"],
    )

    # model
    key = jax.random.PRNGKey(cfg["settings"]["seed"])
    model = MLPDynamics(key=key, config=cfg)
    if stats:
        model = eqx.tree_at(
            lambda m: (m.x_mean, m.x_std, m.u_mean, m.u_std),
            model,
            (jnp.asarray(stats["x_mean"]), jnp.asarray(stats["x_std"]),
             jnp.asarray(stats["u_mean"]), jnp.asarray(stats["u_std"]))
        )

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_fn=_save_ckpt,
        cfg_full=cfg,
        stats=stats,
    )

    trainer.run()
