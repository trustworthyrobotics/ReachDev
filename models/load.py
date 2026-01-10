
import yaml
import equinox as eqx

from models.dt_dyn import T_Dynamics
from models.ct_dyn import Continuous_T_Dynamics
from models.ct_ctl import T_controller

def load_model(model_dir: str, model_type: str, mode: str="best") -> eqx.Module:
    assert model_type in ["dt_dyn", "ct_dyn", "ct_ctl"], f"Unknown model type {model_type} for loading model."
    config_path = f"{model_dir}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    train_config = config[f"train_{model_type}"]
    if model_type == "dt_dyn":
        model_class = T_Dynamics
    elif model_type == "ct_dyn":
        model_class = Continuous_T_Dynamics
    elif model_type == "ct_ctl":
        model_class = T_controller
    model_def = model_class(data_config, train_config)
    assert mode in ["best", "last"], f"Unknown mode {mode} for loading model."
    model_path = f"{model_dir}/{mode}_model.eqx"
    with open(model_path, "rb") as f:
        model = eqx.tree_deserialise_leaves(f, model_def)
    return model
