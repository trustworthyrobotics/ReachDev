# train.py
from __future__ import annotations
import os
import yaml
import jax
import equinox as eqx
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from jax2onnx import to_onnx

from models.dynamics import MLPDynamics, T_Dynamics
from training.trainer import Trainer

from utils.logging import PrintLogger, WandbLogger


def _save_ckpt(path_base: str, model, opt_state, step: int, cfg: dict, stats: dict):
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    eqx.tree_serialise_leaves(path_base + ".eqx", model)
    to_onnx(model, [(sum(model._input_dims()),)], return_mode="file", output_path = path_base + ".onnx", opset=19)
    np.savez(path_base + ".npz", step=np.array(step), config=np.array([yaml.dump(OmegaConf.to_yaml(cfg))]), **(stats or {}))


@hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml", version_base=None)
def main(config: DictConfig) -> None:
    tr_cfg = config["train"]

    os.makedirs(tr_cfg["out_dir"], exist_ok=True)
    # loaders
    task_name = config["settings"]["task_name"]
    if "single_pendulum" in task_name:
        from data.single_pendulum.dataloader import build_loaders
    elif "T_pushing" in task_name:
        from data.T_pushing.dataloader import build_loaders
    else:
        raise ValueError(f"Unknown task in config path: {task_name}")
    train_loader, val_loader, stats = build_loaders(config)

    # model
    key = jax.random.PRNGKey(config["settings"]["seed"])
    if "single_pendulum" in task_name:
        model = MLPDynamics(config=config, stats=stats, key=key)
    elif "T_pushing" in task_name:
        model = T_Dynamics(config=config, stats=stats, key=key)

    if bool(config["train"]["wandb"]["enabled"]):
        logger = WandbLogger(
            config= OmegaConf.to_container(config, resolve=True)
        )
    else:
        logger = PrintLogger()

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_fn=_save_ckpt,
        cfg_full=config,
        stats=stats,
        logger=logger,
    )

    # with jax.disable_jit():
    #     trainer.run()
    trainer.run()

if __name__ == "__main__":
    main()