# utils/logging.py
from __future__ import annotations
from typing import Optional, Dict, Any
import os

import wandb


class Logger:
    """Duck-typed logger API used by Trainer."""
    def log(self, kv: Dict[str, Any], step: Optional[int] = None): ...
    def save(self, path: str): ...
    def finish(self): ...


class PrintLogger(Logger):
    def log(self, kv, step=None):
        msg = " | ".join(f"{k}={float(v):.6f}" if isinstance(v, (int, float)) else f"{k}={v}"
                         for k, v in kv.items())
        if step is not None:
            print(f"[step {step}] {msg}")
        else:
            print(msg)

    def save(self, path: str):
        # No-op for console logging
        return

    def finish(self):
        return


class WandbLogger(Logger):
    def __init__(self, *, config: dict):
        self.wandb = wandb
        wandb_cfg = config["train"]["wandb"]
        project = wandb_cfg["project"]
        name = wandb_cfg["run_name"]
        mode = wandb_cfg["mode"]
        self.run = wandb.init(project=project, name=name,
                              mode=mode, config=config)

    def log(self, kv, step=None):
        # Ensure plain Python scalars
        safe = {}
        for k, v in kv.items():
            try:
                safe[k] = float(v)
            except Exception:
                safe[k] = v
        self.run.log(safe, step=step)

    def save(self, path: str):
        # Upload a file as an artifact-like attachment
        if os.path.exists(path):
            self.wandb.save(path)

    def finish(self):
        self.run.finish()
