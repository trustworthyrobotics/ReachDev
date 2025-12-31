import random
import math
import os
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from pyparsing import Dict
import json

from envs.T_pushing.t_sim import generate_init_target_states, T_Sim
from models.dynamics import load_t_dynamics_model
from planning.planner import MPPIPlanner, CEMPlanner


def _gen_pose_list(num_test, seed, x_bound, y_bound, theta_bound=None, theta_factor=1):
    shift = 0
    num_test+=shift
    random.seed(seed)
    if theta_bound is None:
        return [np.array([random.randint(*x_bound), random.randint(*y_bound)]) for i in range(num_test)][shift:]
    return [
        np.array(
            [
                random.randint(*x_bound),
                random.randint(*y_bound),
                math.radians(random.randint(*theta_bound) * theta_factor),
            ]
        )
        for i in range(num_test)
    ][shift:]


def generate_test_cases(seed, num_test):
    init_pusher_pos_list = _gen_pose_list(num_test, seed, (180, 200), (170, 190), None)
    init_pose_list = _gen_pose_list(num_test, seed, (240, 250), (130, 150), (30, 60))
    target_pusher_pos_list = _gen_pose_list(num_test, seed, (200, 210), (250, 260), None)
    target_pose_list = _gen_pose_list(num_test, seed, (230, 250), (280, 300), (90, 120))
    return init_pusher_pos_list, init_pose_list, target_pusher_pos_list, target_pose_list

def get_pusher_pos_seq(pusher_start_pos, act_seqs):
    n_sample, horizon, action_dim = act_seqs.shape
    pusher_pos_seqs = jnp.zeros((n_sample, horizon + 1, action_dim))

    # initialize first step
    pusher_pos_seqs = pusher_pos_seqs.at[:, 0, 0].add(pusher_start_pos[0])
    pusher_pos_seqs = pusher_pos_seqs.at[:, 0, 1].add(pusher_start_pos[1])
    def body_fn(carry, i):
        pos = carry
        next_pos = pos.at[:, i + 1, :].set(pos[:, i, :] + act_seqs[:, i, :])
        return next_pos, None

    pusher_pos_seqs, _ = jax.lax.scan(body_fn, pusher_pos_seqs, jnp.arange(horizon))
    return pusher_pos_seqs

def get_abs_states(state_seqs, pusher_start_pos, act_seqs):
    pusher_pos_seqs = get_pusher_pos_seq(pusher_start_pos, act_seqs)
    abs_state_seqs = state_seqs.at[:, :, ::2].add(pusher_pos_seqs[:, 1:, 0:1])
    abs_state_seqs = abs_state_seqs.at[:, :, 1::2].add(pusher_pos_seqs[:, 1:, 1:2])
    return abs_state_seqs, pusher_pos_seqs

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
def main(config: DictConfig):
    data_config = config["data"]
    planning_config = config["planning"]
    seed = config["settings"]["seed"]
    num_test = planning_config["num_test"]
    init_pusher_pos_list, init_pose_list, target_pusher_pos_list, target_pose_list = generate_test_cases(seed, num_test)

    model_path = os.path.join( config["train"]["out_dir"], "last_model.eqx")
    model = load_t_dynamics_model(config=config, model_path=model_path)
    param_dict = {"stem_size": data_config["stem_size"], 
                "bar_size": data_config["bar_size"], 
                "pusher_size": data_config["pusher_size"],
                "save_img": True,
                "enable_vis": False,
                "window_size": data_config["window_size"],}

    action_bound = planning_config["action_bound"]
    scale = float(data_config["scale"])
    state_dim, action_dim = data_config["state_dim"], data_config["action_dim"]
    action_lower_lim = -action_bound * jnp.ones((action_dim,)) / scale
    action_upper_lim = action_bound * jnp.ones((action_dim,)) / scale
    
    cost_norm, only_final_cost = planning_config["cost_norm"], planning_config["only_final_cost"]
    max_steps = planning_config["max_steps"]
    horizon = planning_config["horizon"]
    n_act_step = planning_config["n_act_step"]

    # [n_his=1, state_dim], [n_sample, horizon, action_dim] -> [n_sample, horizon, state_dim]
    def rollout_fn(state_cur: jnp.ndarray, act_seqs: jnp.ndarray) -> jnp.ndarray:
        state_cur = state_cur[None].repeat(act_seqs.shape[0], axis=0)
        state_seqs = model.rollout_model(state_cur, act_seqs)
        return state_seqs

    # assume all are scaled
    def reward_fn(state_seqs: jnp.ndarray, act_seqs: jnp.ndarray, target_state: jnp.ndarray, pusher_pos: jnp.ndarray) -> Dict:
        abs_state_seqs, _ = get_abs_states(state_seqs, pusher_pos, act_seqs)

        cost_seqs = jnp.linalg.norm(abs_state_seqs - target_state[None, None, :], axis=-1, ord=cost_norm) ** cost_norm
        if only_final_cost:
            costs = cost_seqs[:, -1]
        else:
            step_weight = jnp.linspace(1, horizon + 1, horizon) / horizon
            costs = jnp.sum(cost_seqs * step_weight[None, :], axis=-1)
        return {"rewards": -costs, "reward_seqs": -cost_seqs}

    def step_cost_fn(state, target_state):
        return (np.linalg.norm(target_state - state, cost_norm)) ** cost_norm

    for i in range(num_test):
        init_pusher_pos = init_pusher_pos_list[i]
        init_pose = init_pose_list[i]
        target_pusher_pos = target_pusher_pos_list[i]
        target_pose = target_pose_list[i]
        print(f"Test case {i}:")
        print(f"  Init pusher pos: {init_pusher_pos}")
        print(f"  Init pose: {init_pose}")
        print(f"  Target pusher pos: {target_pusher_pos}")
        print(f"  Target pose: {target_pose}")

        init_state, target_state = generate_init_target_states(
            init_pose, target_pose, param_dict={"stem_size": data_config["stem_size"], "bar_size": data_config["bar_size"]}
        )
        scaled_target_state = target_state / scale

        env = T_Sim(param_dict=param_dict, init_poses=[init_pose], target_poses=[target_pose], pusher_pos=init_pusher_pos)
        for i in range(2):
            env_dict = env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)
            env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)
        planner = CEMPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)

        # executtion loop
        planning_res_list = []
        gt_states = [env_state.tolist()]
        t = 0
        while t < max_steps:
            env_state, env_dict = env.get_env_state(False)
            env_state = jnp.array(env_state) / scale
            pusher_pos = env_state[state_dim : state_dim + 2]
            state_cur = env_state[:state_dim]
            # relative to pusher
            state_cur = state_cur.at[::2].add(-pusher_pos[0])
            state_cur = state_cur.at[1::2].add(-pusher_pos[1])
            key = jax.random.PRNGKey(seed + t)
            init_act_seq = jax.random.uniform(key,(horizon, action_dim),minval=action_lower_lim,maxval=action_upper_lim,)
            planning_res = planner.trajectory_optimization(key, state_cur, init_act_seq, skip=False, target_state=scaled_target_state, pusher_pos=pusher_pos)
            act_seq = planning_res["act_seq"] * scale  # (horizon, action_dim)
            state_seq = planning_res["state_seq"] * scale  # (horizon, state_dim)
            abs_state_seq, pusher_pos_seq = get_abs_states(state_seq[None, :, :], pusher_pos * scale, act_seq[None, :, :])
            abs_state_seq = abs_state_seq[0]
            pusher_pos_seq = pusher_pos_seq[0]
            res = {
                "time_step": t,
                "act_seq": act_seq.tolist(),
                "state_seq": abs_state_seq.tolist(),
                "pusher_pos_seq": pusher_pos_seq.tolist(),
                "reward": planning_res["reward"].tolist()
            }
            planning_res_list.append(res)
            for step in range(n_act_step):
                next_pusher_pos = pusher_pos_seq[step + 1, :]
                env_dict = env.update((next_pusher_pos[0], next_pusher_pos[1]), rel=False)
                env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)
                t += 1
                gt_states.append(env_state.tolist())
                step_cost = step_cost_fn(env_state[:state_dim], target_state)
                print(f"   step {t} cost: {step_cost}, action: {env_dict['action']}")
                if t >= max_steps:
                    break

        # save results
        out_dir = os.path.join(planning_config["out_path"], "planning_results", f"T_pushing", f"test_case_{i:04d}")
        os.makedirs(out_dir, exist_ok=True)
        # planning_res_path = os.path.join(out_dir, "planning_res.npy")
        # np.save(planning_res_path, planning_res_list)
        # gt_states_path = os.path.join(out_dir, "gt_states.npy")
        # np.save(gt_states_path, np.array(gt_states))
        planning_res_path = os.path.join(out_dir, "planning_res.json")
        with open(planning_res_path, "w") as f:
            json.dump(planning_res_list, f, indent=4, separators=(",", ": "))
        gt_states_path = os.path.join(out_dir, "gt_states.json")
        with open(gt_states_path, "w") as f:
            json.dump(gt_states, f, indent=4, separators=(",", ": "))
        env.save_gif(os.path.join(out_dir, "planning_vis.gif"))
        env.close()

if __name__ == "__main__":
    main()