# train.py
from __future__ import annotations
import os
import yaml
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import equinox as eqx
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from jax2onnx import to_onnx
from datetime import datetime
from training.trainer import Trainer

from utils.logging import PrintLogger, WandbLogger


def _save_ckpt(path_base: str, model, opt_state, step: int, cfg: dict):
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    eqx.tree_serialise_leaves(path_base + ".eqx", model)
    to_onnx(model, [(sum(model._input_dims()),)], return_mode="file", output_path = path_base + ".onnx", opset=19)
    np.savez(path_base + ".npz", step=np.array(step), config=np.array([yaml.dump(OmegaConf.to_yaml(cfg))]))


@hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml", version_base=None)
def main(config: DictConfig) -> None:
    train_mode = config["settings"].get("train_mode", "dt_dyn")
    assert train_mode in {"dt_dyn", "ct_dyn", "ct_ctl"}, f"Unknown train_mode: {train_mode}"
    tr_cfg = config[f"train_{train_mode}"]
    data_cfg = config["data"]

    tr_cfg["wandb"]["run_name"] = f"{tr_cfg['wandb']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tr_cfg["out_dir"] = os.path.join(tr_cfg["out_dir"], tr_cfg["wandb"]["run_name"])
    os.makedirs(tr_cfg["out_dir"], exist_ok=True)   
    # copy the config file to the output directory
    with open(os.path.join(tr_cfg["out_dir"], "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))
    # loaders
    task_name = config["settings"]["task_name"]
    if "T_pushing" in task_name:
        from data.T_pushing.dataloader import build_loaders
    else:
        raise ValueError(f"Unknown task in config path: {task_name}")
    train_loader, val_loader = build_loaders(data_cfg, tr_cfg, train_mode, seed=config["settings"]["seed"])

    # model
    key = jax.random.PRNGKey(config["settings"]["seed"])
    if "T_pushing" in task_name:
        if train_mode == "dt_dyn":
            from models.dt_dyn import T_Dynamics
            model = T_Dynamics(data_cfg, tr_cfg, key=key)
        elif train_mode == "ct_dyn":
            from models.ct_dyn import Continuous_T_Dynamics
            model = Continuous_T_Dynamics(data_cfg, tr_cfg, key=key)
        elif train_mode == "ct_ctl":
            from models.ct_ctl import T_controller
            model = T_controller(data_cfg, tr_cfg, key=key)
            from models.mlp_utils import load_model
            from models.ct_dyn import Continuous_T_Dynamics
            model_dir = data_cfg["ct_ctl"]["model_dir"]
            ct_dyn = load_model(data_cfg, config[f"train_ct_dyn"], model_class=Continuous_T_Dynamics, model_dir=model_dir, mode="best")
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    if bool(tr_cfg["wandb"]["enabled"]):
        logger = WandbLogger(
            config= OmegaConf.to_container(config, resolve=True),
            train_mode=train_mode,
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
        logger=logger,
        ct_dyn=ct_dyn if train_mode == "ct_ctl" else None,
    )

    # with jax.disable_jit():
    #     trainer.run()
    trainer.run()

if __name__ == "__main__":
    main()