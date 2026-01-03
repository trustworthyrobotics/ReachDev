import math
import numpy as np
import random
from pymunk import Vec2d
import jax
import jax.numpy as jnp
import equinox as eqx

from envs.T_pushing.t_sim import T_Sim, get_keypoints_from_pose, get_pose_from_keypoints
from models.dynamics_c import Continuous_T_Dynamics, load_t_dynamics_model

class ShadowBody:
    """Minimalist body to mimic pymunk.Body."""
    def __init__(self, position=None, angle=0.0, color=(0, 0, 0, 255), label=""):
        self.position = Vec2d(*(position if position is not None else (0, 0)))
        self.angle = float(angle)
        self.velocity = [0, 0]
        self.color = color  # Default color
        self.label = label  # Optional label for identification

class ShadowSpace:
    """Minimalist space to mimic pymunk.Space and handle NN inference."""
    def __init__(self, model: Continuous_T_Dynamics, param_dict: dict):
        self.model = model
        self.param_dict = param_dict
        self.scale = param_dict.get("scale", 1.0)
        self.bodies = [] # [pusher_body, object_body]
        self.forward = eqx.filter_jit(self.model.forward_batchless)
        # self.forward = self.model.forward_batchless

    def add(self, *items):
        for item in items:
            if isinstance(item, ShadowBody):
                self.bodies.append(item)
            # Shapes (Poly/Circle) are ignored as NN handles geometry implicitly

    def remove(self, *items):
        for item in items:
            if isinstance(item, ShadowBody):
                self.bodies.remove(item)
            # Shapes are ignored

    def step(self, dt):
        # 1. Identify roles
        for body in self.bodies:
            body: ShadowBody
            if body.label == "object_0":
                obj_body = body
            elif body.label == "pusher":
                pusher_body = body

        # 3. Prepare NN Inputs
        # Get keypoints from current object pose (requires helper from T_Sim context)
        # Note: We use jnp.array for the model
        obj_x = obj_body.position.x
        obj_y = obj_body.position.y
        obj_theta = obj_body.angle
        pusher_x = pusher_body.position.x
        pusher_y = pusher_body.position.y
        if self.model.pred_mode == "state":
            x = get_keypoints_from_pose([obj_x, obj_y, obj_theta], self.param_dict).flatten()
            x = (x - np.array([pusher_x, pusher_y] * 4)) / self.scale  # relative kp
        elif self.model.pred_mode == "pose":
            x = jnp.array([(obj_x - pusher_x) / self.scale, (obj_y - pusher_y) / self.scale, obj_theta])
        else:
            raise ValueError(f"Unknown pred_mode: {self.model.pred_mode}")
        u = jnp.array([pusher_body.velocity[0], pusher_body.velocity[1]]) / self.scale

        # 4. Neural Inference (x_next = x_curr + integral of mlp output)
        # Using the batchless forward call
        new_x = np.array(self.forward(x, u))  # Convert back to numpy

        # 2. Update Pusher Kinematics (Euler integration)
        pusher_body.position = Vec2d(pusher_x + pusher_body.velocity[0] * dt, pusher_y + pusher_body.velocity[1] * dt)

        # 5. Update Object Body
        # We need to recover pose (x, y, theta) from predicted keypoints
        pusher_x = pusher_body.position.x
        pusher_y = pusher_body.position.y
        if self.model.pred_mode == "state":
            new_x = new_x * self.scale + np.array([pusher_x, pusher_y] * 4)
            new_x = get_pose_from_keypoints(
                new_x.reshape(-1, 2), 
                self.param_dict
            )
        elif self.model.pred_mode == "pose":
            new_x[:2] = new_x[:2] * self.scale + np.array([pusher_x, pusher_y])
        else:
            raise ValueError(f"Unknown pred_mode: {self.model.pred_mode}")

        obj_body.position = Vec2d(new_x[0], new_x[1])
        obj_body.angle = new_x[2]
        obj_body.velocity = [(new_x[0] - obj_x) / dt, (new_x[1] - obj_y) / dt]

class NN_T_Sim(T_Sim):
    def __init__(self, param_dict, model: Continuous_T_Dynamics, init_poses=None, target_poses=None, pusher_pos=None):
        # We skip the standard T_Sim create_world and do it manually to use Shadows
        self.model = model

        # Initialize Base_Sim attributes
        super().__init__(param_dict, init_poses, target_poses, pusher_pos, step_dt=float(model.dt)) # Force simulation step to match NN training dt

    def create_world(self, init_poses, pusher_pos):
        self.space = ShadowSpace(self.model, self.param_dict)
        self.add_objects(self.obj_num, init_poses)
        self.add_pusher(pusher_pos)
        self.wait(1.0)


    def create_object(self, id, pose=None):
        color = self.object_colors[id % len(self.object_colors)]
        if pose is None:
            angle = random.random() * math.pi * 2
            position = Vec2d(random.randint(int(0.4 * self.width), int(0.6 * self.width)), random.randint(int(0.4 * self.height), int(0.6 * self.height)))
        else:
            angle = pose[2]
            position = Vec2d(pose[0], pose[1])

        body = ShadowBody(position=position, angle=angle, color=color, label=f"object_{id}")
        shape = [None, None]  # Shapes are ignored in NN_T_Sim
        return body, shape

    def create_pusher(self, position):
        # Overridden to use ShadowBody
        pusher_body = ShadowBody(position=position, color=self.pusher_color, label="pusher")
        shape = None  # Shapes are ignored in NN_T_Sim
        return pusher_body, shape

if __name__ == "__main__":
    import os
    import yaml
    param_dict = {
        "stem_size": (10, 60),
        "bar_size": (60, 10),
        "pusher_size": 5,
        "scale": 100,
        "save_img": True,
        "enable_vis": False,
        "window_size": 500
    }

    model_dir = "output/runs/T_pushing_ct_dyn/"
    model_dir = model_dir + "log_20_lr0.0025_20260102_230034"
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    train_cfg = config["train_ct_dyn"] if "train_ct_dyn" in config else config["train"]
    model_path = os.path.join(model_dir, "best_model.eqx")
    model = load_t_dynamics_model(data_config=data_cfg, train_config=train_cfg, model_path=model_path)
    # init_poses = [[[250,250,math.radians(45)], [150,150,math.radians(-45)]]]
    init_poses = [[250, 250, math.radians(0)]]
    target_poses = [[250, 250, math.radians(45)]]
    pusher_pos = [200, 200]
    sim = NN_T_Sim(
        param_dict=param_dict,
        model=model,
        init_poses=init_poses,
        target_poses=target_poses,
        pusher_pos=pusher_pos,
    )
    # [[250,250,math.radians(45)], [150,150,math.radians(-45)]]
    sim.render()
    print(sim.get_all_object_positions())
    print(sim.get_all_object_keypoints())
    print(sim.get_current_state())
    print(sim.get_all_object_keypoints(target=True))
    for i in range(5):
        env_dict = sim.update(action=np.array([200.0 + i * 10, 200.0 + i * 10]), n_sim_time=0.1)

    sim.save_gif("t_sim_nn_test.gif", fps=1)