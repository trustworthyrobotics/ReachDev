# train.py
from __future__ import annotations
import os
import yaml
import jax
# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
# jax.profiler.save_device_memory_profile("memory.prof")
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
# from jax2onnx import to_onnx
from datetime import datetime
from training.trainer import Trainer
from training.losses_metrics import TotalLoss, TotalLossCtl, TotalLoss_quad, TotalLossCtl_quad

from utils.logging import PrintLogger, WandbLogger


def _save_ckpt(path_base: str, model, opt_state, step: int, cfg: dict):
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    eqx.tree_serialise_leaves(path_base + ".eqx", model)
    # to_onnx(model, [(sum(model._input_dims()),)], return_mode="file", output_path = path_base + ".onnx", opset=19)
    np.savez(path_base + ".npz", step=np.array(step), config=np.array([yaml.dump(OmegaConf.to_yaml(cfg))]))

@hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml", version_base=None)
# @hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml", version_base=None)
def main(config: DictConfig) -> None:
    if config["settings"].get("device", "gpu") == "cpu":
        jax.config.update('jax_platforms', 'cpu')
        print("Using CPU for training.")

    train_mode = config["settings"].get("train_mode", "dt_dyn")
    assert train_mode in {"dt_dyn", "ct_dyn", "ct_ctl"}, f"Unknown train_mode: {train_mode}"
    tr_cfg = config[f"train_{train_mode}"]
    data_cfg = config["data"]

    tr_cfg["wandb"]["run_name"] = f"{tr_cfg['wandb']['run_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tr_cfg["out_dir"] = os.path.join(tr_cfg["out_dir"], tr_cfg["wandb"]["run_name"])
    os.makedirs(tr_cfg["out_dir"], exist_ok=True)   
    # copy the config file to the output directory
    with open(os.path.join(tr_cfg["out_dir"], "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))
    # loaders
    task_name = config["settings"]["task_name"]
    if "T_pushing" in task_name:
        from data.T_pushing.dataloader import build_loaders
    elif "quadrotor" in task_name:
        from data.quadrotor.dataloader import build_loaders  
    else:
        raise ValueError(f"Unknown task in config path: {task_name}")
    train_loader, val_loader = build_loaders(data_cfg, tr_cfg, train_mode, seed=config["settings"]["seed"])

    # model
    key = jax.random.PRNGKey(config["settings"]["seed"])
    if "T_pushing" in task_name:
        if train_mode == "dt_dyn":
            from models.T_pushing.dt_dyn import T_Dynamics
            model = T_Dynamics(data_cfg, tr_cfg, key=key)
            loss_class = TotalLoss
        elif train_mode == "ct_dyn":
            from models.T_pushing.ct_dyn import Continuous_T_Dynamics
            model = Continuous_T_Dynamics(data_cfg, tr_cfg, key=key)
            loss_class = TotalLoss
        elif train_mode == "ct_ctl":
            from models.T_pushing.ct_ctl import T_controller
            model = T_controller(data_cfg, tr_cfg, key=key)
            from models.load import load_model
            model_dir = data_cfg["ct_ctl"]["model_dir"]
            ct_dyn = load_model(model_dir=model_dir, model_type="ct_dyn", mode="best")
            loss_class = TotalLossCtl
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")
    elif "quadrotor" in task_name:
        standardize = tr_cfg.get("standardize", True)
        stats = None
        if standardize:
            stats_path = os.path.join(tr_cfg["data_dir"], "norm_stats.npz")
            with open(stats_path, "rb") as f:
                stats = np.load(f)
                stats = {k: jnp.array(stats[k]) for k in stats.files}
        if train_mode == "dt_dyn":
            from models.quadrotor.dt_dyn import Quad_Dynamics
            model = Quad_Dynamics(data_cfg, tr_cfg, key=key, stats=stats)
            loss_class = TotalLoss_quad
        elif train_mode == "ct_dyn":
            print("CT dynamics for quadrotor uses analytical model, no training needed.")
            exit(0)
        elif train_mode == "ct_ctl":
            from models.quadrotor.ct_ctl import MLP_Controller
            model = MLP_Controller(data_cfg, tr_cfg, key=key, stats=stats)
            from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics
            ct_dyn = Continuous_Quad_Dynamics(data_cfg)
            loss_class = TotalLossCtl_quad
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    loss_fn = loss_class(
        mode=train_mode,
        state_dim=model.Dx,
        action_dim=model.Du,
        reach_cfg=tr_cfg.get("reach", {}),
        dyn_frequency=model.frequency if train_mode != "ct_ctl" else ct_dyn.frequency,
        lam_jac=float(tr_cfg.get("lam_jac_reg", 0.0)),
        ct_dyn=ct_dyn if train_mode == "ct_ctl" else None,
        reference_dim=model.Dr if train_mode == "ct_ctl" else None,
        ctl_frequency=float(tr_cfg["ctl_frequency"]) if train_mode == "ct_ctl" else None,
        loss_mode=tr_cfg.get("loss_mode", "s") if train_mode == "ct_ctl" else None,
    )

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
        loss_fn=loss_fn,
    )

    # with jax.disable_jit():
    #     trainer.run()
    trainer.run()

if __name__ == "__main__":
    main()