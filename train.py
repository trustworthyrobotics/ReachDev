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
from training.trainer import Trainer

from utils.logging import PrintLogger, WandbLogger


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
    if "single_pendulum" in args.config:
        from data.single_pendulum.dataloader import build_loaders
    elif "T_pushing" in args.config:
        from data.T_pushing.dataloader import build_loaders
    else:
        raise ValueError(f"Unknown task in config path: {args.config}")
    train_loader, val_loader, stats = build_loaders(cfg)

    # model
    key = jax.random.PRNGKey(cfg["settings"]["seed"])
    model = MLPDynamics(key=key, config=cfg, stats=stats)

    if bool(cfg["train"]["wandb"]["enabled"]):
        logger = WandbLogger(
            config=cfg,
        )
    else:
        logger = PrintLogger()

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_fn=_save_ckpt,
        cfg_full=cfg,
        stats=stats,
        logger=logger,
    )

    trainer.run()
