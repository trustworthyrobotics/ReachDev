
import yaml
import equinox as eqx

from models.dt_dyn import T_Dynamics
from models.ct_dyn import Continuous_T_Dynamics
from models.ct_ctl import T_controller

from models.quadrotor.dt_dyn import Quad_Dynamics
from models.quadrotor.ct_ctl import MLP_Controller
from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics

def load_model(model_dir: str, model_type: str, mode: str="best", task_name: str="T_pushing") -> eqx.Module:
    assert model_type in ["dt_dyn", "ct_dyn", "ct_ctl"], f"Unknown model type {model_type} for loading model."
    config_path = f"{model_dir}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    train_config = config[f"train_{model_type}"]
    if task_name == "T_pushing":
        if model_type == "dt_dyn":
            model_class = T_Dynamics
        elif model_type == "ct_dyn":
            model_class = Continuous_T_Dynamics
        elif model_type == "ct_ctl":
            model_class = T_controller
    else:
        assert task_name == "quadrotor", f"Unknown task name {task_name} for loading model."
        if model_type == "dt_dyn":
            model_class = Quad_Dynamics
        elif model_type == "ct_dyn":
            raise NotImplementedError("Continuous-time dynamics model for quadrotor uses analytical model, no loading needed.")
            model_class = Continuous_Quad_Dynamics
        elif model_type == "ct_ctl":
            model_class = MLP_Controller

    model_def = model_class(data_config, train_config)
    assert mode in ["best", "last"], f"Unknown mode {mode} for loading model."
    model_path = f"{model_dir}/{mode}_model.eqx"
    with open(model_path, "rb") as f:
        model = eqx.tree_deserialise_leaves(f, model_def)
    return model
