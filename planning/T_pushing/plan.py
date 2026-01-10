import random
import math
import os
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
from pyparsing import Dict
import json
import matplotlib.pyplot as plt

from envs.T_pushing.t_sim import generate_init_target_states, T_Sim
from models.load import load_model
from models.dt_dyn import T_Dynamics
from models.ct_dyn import Continuous_T_Dynamics
from models.ct_ctl import T_controller
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


def generate_test_cases(seed, num_test, test_id=0):
    if test_id == 0:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (180, 200), (170, 190), None)
        init_pose_list = _gen_pose_list(num_test, seed, (240, 250), (130, 150), (30, 60))
        target_pose_list = _gen_pose_list(num_test, seed, (230, 250), (280, 300), (90, 120))
    elif test_id == 1:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (100, 100), (100, 100), None)
        init_pose_list = _gen_pose_list(num_test, seed, (120, 120), (120, 120), (30, 60))
        target_pose_list = _gen_pose_list(num_test, seed, (200, 200), (200, 200), (0, 0))
    elif test_id == 2:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (100, 100), (100, 100), None)
        init_pose_list = _gen_pose_list(num_test, seed, (150, 175), (90, 110), (90, 90))
        target_pose_list = _gen_pose_list(num_test, seed, (215, 235), (365, 385), (160, 200))
    elif test_id == 3:
        init_pusher_pos_list = _gen_pose_list(num_test, seed, (100, 100), (100, 100), None)
        init_pose_list = _gen_pose_list(num_test, seed, (150, 200), (80, 140), (-90, -90))
        target_pose_list = _gen_pose_list(num_test, seed, (225, 225), (340, 380), (315, 315))
    else:
        raise ValueError(f"Unknown test_id: {test_id}")
    return init_pusher_pos_list, init_pose_list, target_pose_list

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

def get_abs_states(state_seqs, pusher_start_pos, act_seqs, pred_mode="state"):
    pusher_pos_seqs = get_pusher_pos_seq(pusher_start_pos, act_seqs)
    if pred_mode == "pose":
        abs_state_seqs = state_seqs.at[:, :, 0:2].add(pusher_pos_seqs[:, 1:, 0:2])
    else:
        abs_state_seqs = state_seqs.at[:, :, ::2].add(pusher_pos_seqs[:, 1:, 0:1])
        abs_state_seqs = abs_state_seqs.at[:, :, 1::2].add(pusher_pos_seqs[:, 1:, 1:2])
    return abs_state_seqs, pusher_pos_seqs

def plot_cost_stat(cost_stat, out_path):
    # cost_stat: (num_test, max_steps)
    plt.figure()
    for i in range(cost_stat.shape[0]):
        plt.plot(cost_stat[i], alpha=0.5, color="black")

    quantiles = np.percentile(cost_stat, [25, 50, 75], axis=0)
    plt.plot(quantiles[0], color="blue", label="25th percentile")
    plt.plot(quantiles[1], color="orange", label="50th percentile")
    plt.plot(quantiles[2], color="green", label="75th percentile")
    plt.legend()

    plt.xlabel("Time step")
    plt.ylabel("Step cost")
    plt.title("Step Cost over Time for Each Test Case")
    plt.grid()
    out_name = os.path.join(out_path, "step_costs.png")
    plt.savefig(out_name)
    print(f"Step cost plot saved to {out_name}")
    plt.close()
    return

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
def main(config: DictConfig):
    data_config = config["data"]
    train_config = config["train_dt_dyn"]
    planning_config = config["planning"]
    seed = config["settings"]["seed"]
    num_test = planning_config["num_test"]
    test_id = planning_config.get("test_id", 0)
    init_pusher_pos_list, init_pose_list, target_pose_list = generate_test_cases(seed, num_test, test_id=test_id)

    dt_dyn_dir = config["test_models"]["dt_dyn_dir"]
    dt_dyn: T_Dynamics = load_model(model_dir=dt_dyn_dir, model_type="dt_dyn", mode="best")
    abs_pose = dt_dyn.abs_pose
    pred_mode = dt_dyn.pred_mode
    param_dict = {"stem_size": data_config["stem_size"], 
                "bar_size": data_config["bar_size"], 
                "pusher_size": data_config["pusher_size"],
                "save_img": True,
                "enable_vis": False,
                "window_size": data_config["window_size"],}

    out_dir = os.path.join(planning_config["out_path"], dt_dyn_dir[-6:])
    os.makedirs(out_dir, exist_ok=True)

    action_bound = planning_config["action_bound"]
    scale = float(data_config["scale"])
    state_dim, pose_dim, action_dim = data_config["state_dim"], data_config["pose_dim"], data_config["action_dim"]
    T_dim = state_dim if pred_mode == "state" else pose_dim
    action_lower_lim = -action_bound * jnp.ones((action_dim,)) / scale
    action_upper_lim = action_bound * jnp.ones((action_dim,)) / scale
    
    cost_norm, only_final_cost = planning_config["cost_norm"], planning_config["only_final_cost"]
    max_steps = planning_config["max_steps"] + 1  # +1 to account for initial step
    horizon = planning_config["horizon"]
    n_act_step = planning_config["n_act_step"]

    enable_ctl = planning_config.get("enable_ctl", False)
    if enable_ctl:
        assert abs_pose, "Controller can only be enabled when using absolute pose prediction."
    ct_ctl_dir = config["test_models"]["ct_ctl_dir"]
    ct_ctl: T_controller = load_model(model_dir=ct_ctl_dir, model_type="ct_ctl", mode="best")
    ctl_frequency = ct_ctl.ctl_frequency

    noise_type = planning_config.get("disturbance", {}).get("type", "none")
    noise_init = planning_config.get("disturbance", {}).get("init", 0.0)
    noise_inter = planning_config.get("disturbance", {}).get("inter", 0.0)

    # [n_his=1, state_dim], [n_sample, horizon, action_dim] -> [n_sample, horizon, state_dim]
    def rollout_fn(state_cur: jnp.ndarray, act_seqs: jnp.ndarray) -> jnp.ndarray:
        state_cur = state_cur[None].repeat(act_seqs.shape[0], axis=0)
        state_seqs = dt_dyn.rollout(state_cur, act_seqs)
        # noise_param = noise_inter
        # if noise_type == "normal":
        #     noise = jax.random.normal(key, shape=(*state_seqs.shape[:2], T_dim)) * noise_param
        # elif noise_type == "uniform":
        #     noise = jax.random.uniform(key, shape=(*state_seqs.shape[:2], T_dim), minval=-1.0, maxval=1.0) * noise_param
        # else:
        #     noise = jnp.zeros((*state_seqs.shape[:2], T_dim))
        # state_seqs = state_seqs.at[:, :, 0:T_dim].add(noise)

        return state_seqs

    # assume all are scaled
    def reward_fn(state_seqs: jnp.ndarray, act_seqs: jnp.ndarray, target_state: jnp.ndarray, pusher_pos: jnp.ndarray) -> Dict:
        if abs_pose:
            abs_state_seqs = state_seqs[..., :-act_seqs.shape[-1]]
        else:
            abs_state_seqs, _ = get_abs_states(state_seqs, pusher_pos, act_seqs, pred_mode=pred_mode)

        cost_seqs = jnp.linalg.norm(abs_state_seqs - target_state[None, None, :], axis=-1, ord=cost_norm) ** cost_norm
        if only_final_cost:
            costs = cost_seqs[:, -1]
        else:
            step_weight = jnp.linspace(1, horizon + 1, horizon) / horizon
            costs = jnp.sum(cost_seqs * step_weight[None, :], axis=-1)
        return {"rewards": -costs, "reward_seqs": -cost_seqs}

    def step_cost_fn_np(state, target_state):
        return (np.linalg.norm(target_state - state, cost_norm)) ** cost_norm

    def step_cost_fn(state, target_state):
        return (jnp.linalg.norm(target_state - state, cost_norm)) ** cost_norm

    planner_type = planning_config.get("planner", "mppi").lower()
    if planner_type == "mppi":
        planner = MPPIPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    elif planner_type == "cem":
        planner = CEMPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    cost_stat = []
    for i in range(num_test):
        init_pusher_pos = init_pusher_pos_list[i]
        init_pose = init_pose_list[i]
        target_pose = target_pose_list[i]
        print(f"Test case {i}:")
        # print(f"  Init pusher pos: {init_pusher_pos}")
        # print(f"  Init pose: {init_pose}")
        # print(f"  Target pusher pos: {target_pusher_pos}")
        # print(f"  Target pose: {target_pose}")

        if pred_mode == "pose":
            target_state = target_pose
            scaled_target_state = target_state / scale
            scaled_target_state[2] = target_state[2]  # do not scale angle
        else:
            init_state, target_state = generate_init_target_states(
                init_pose, target_pose, param_dict={"stem_size": data_config["stem_size"], "bar_size": data_config["bar_size"]}
            )
            scaled_target_state = target_state / scale
        scaled_target_state = jnp.array(scaled_target_state)
        env = T_Sim(param_dict=param_dict, init_poses=[init_pose], target_poses=[target_pose], pusher_pos=init_pusher_pos)
        for _ in range(2):
            env_dict = env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)
            env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)

        # executtion loop
        planning_res_list = []
        step_cost_list = []
        gt_states = [env_state.tolist()]
        t = 0
        succeed = False
        while t < max_steps:
            env_dict = env.get_env_state(not abs_pose)
            pusher_pos = jnp.array(env_dict["pusher_pos"]) / scale
            if pred_mode == "pose":
                state_cur = jnp.array(np.concatenate([env_dict["com_pos"] / scale, env_dict["angle"]], axis=0))
            else:
                state_cur = jnp.array(env_dict["state"][:state_dim]) / scale
            if abs_pose:
                state_cur = jnp.concatenate([state_cur, pusher_pos], axis=0)
            key = jax.random.PRNGKey(seed + t)

            noise_param = noise_init
            if noise_type == "normal":
                noise = jax.random.normal(key, shape=(T_dim,)) * noise_param
            elif noise_type == "uniform":
                noise = jax.random.uniform(key, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param
            else:
                noise = jnp.zeros((T_dim,))
            state_cur = state_cur.at[0:T_dim].add(noise)

            init_act_seq = jax.random.uniform(key,(horizon, action_dim),minval=action_lower_lim,maxval=action_upper_lim,)
            # with jax.disable_jit():
            #     planning_res = eqx.filter_jit(planner.trajectory_optimization)(key, state_cur, init_act_seq, skip=False, target_state=scaled_target_state, pusher_pos=pusher_pos)
            planning_res = eqx.filter_jit(planner.trajectory_optimization)(key, state_cur, init_act_seq, skip=False, target_state=scaled_target_state, pusher_pos=pusher_pos)
            scaled_act_seq = planning_res["act_seq"]
            scaled_state_seq = planning_res["state_seq"]
            act_seq = scaled_act_seq * scale  # (horizon, action_dim)
            state_seq = scaled_state_seq * scale  # (horizon, state_dim)
            if pred_mode == "pose":
                state_seq = state_seq.at[:, pose_dim - 1].set(state_seq[:, pose_dim - 1]/scale)  # do not scale angle
            if abs_pose:
                abs_scaled_state_seq = jnp.concatenate([state_cur[None, :], scaled_state_seq], axis=0)
                abs_state_seq = abs_scaled_state_seq * scale
                pusher_pos_seq = abs_scaled_state_seq[:, -action_dim:] * scale
            else:
                abs_state_seq, pusher_pos_seq = get_abs_states(state_seq[None, :, :], pusher_pos * scale, act_seq[None, :, :], pred_mode=pred_mode)
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
                if enable_ctl:
                    sub_target = abs_scaled_state_seq[step + 1, :-action_dim]
                    ref_action = scaled_act_seq[step, :]
                    sub_env_states = []
                    for ctl_step in range(ctl_frequency):
                        if (step_cost_fn(sub_target, state_cur[:-action_dim]) < 2e-1) and (not succeed):
                            next_action = ref_action
                        else:
                            next_action = eqx.filter_jit(ct_ctl.forward_batchless)(state_cur, sub_target, ref_action)
                        next_pusher_pos = (pusher_pos + next_action) * scale
                        env_dict = env.update((next_pusher_pos[0], next_pusher_pos[1]), rel=False, n_sim_time=1/ctl_frequency)

                        pusher_pos = jnp.array(env_dict["pusher_pos"]) / scale
                        if pred_mode == "pose":
                            state_cur = jnp.array(np.concatenate([env_dict["com_pos"] / scale, env_dict["angle"]], axis=0))
                            env_state = np.concatenate([np.array(env_dict["com_pos"]), np.array(env_dict["angle"]), env_dict["pusher_pos"]], axis=0)
                        else:
                            state_cur = jnp.array(env_dict["state"][:state_dim]) / scale
                            env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)
                        state_cur = jnp.concatenate([state_cur, pusher_pos], axis=0)
                        
                        noise_param = noise_init
                        if noise_type == "normal":
                            noise = jax.random.normal(key, shape=(T_dim,)) * noise_param
                        elif noise_type == "uniform":
                            noise = jax.random.uniform(key, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param
                        else:
                            noise = jnp.zeros((T_dim,))
                        state_cur = state_cur.at[0:T_dim].add(noise)

                        sub_env_states.append(env_state)
                    env_state = np.array(sub_env_states)
                    step_cost = step_cost_fn_np(env_state[-1][:-action_dim], target_state)
                else:
                    sub_env_states = []
                    for ctl_step in range(ctl_frequency):
                        next_action = scaled_act_seq[step, :]
                        next_pusher_pos = (pusher_pos + next_action) * scale
                        env_dict = env.update((next_pusher_pos[0], next_pusher_pos[1]), rel=False, n_sim_time=1/ctl_frequency)
                        if pred_mode == "pose":
                            env_state = np.concatenate([np.array(env_dict["com_pos"]), np.array(env_dict["angle"]), env_dict["pusher_pos"]], axis=0)
                        else:
                            env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)
                        sub_env_states.append(env_state)
                    env_state = np.array(sub_env_states)
                    step_cost = step_cost_fn_np(env_state[-1][:-action_dim], target_state)

                    # next_pusher_pos = pusher_pos_seq[step + 1, :]
                    # env_dict = env.update((next_pusher_pos[0], next_pusher_pos[1]), rel=False)
                    # if pred_mode == "pose":
                    #     env_state = np.concatenate([np.array(env_dict["com_pos"]), np.array(env_dict["angle"]), env_dict["pusher_pos"]], axis=0)
                    # else:
                    #     env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)
                    # step_cost = step_cost_fn_np(env_state[:-action_dim], target_state)

                t += 1
                gt_states.append(env_state.tolist())
                print(f"   step {t} cost: {step_cost}, action: {env_dict['action']}")
                step_cost_list.append(step_cost)
                if step_cost < 30:
                    succeed = True
                if t >= max_steps:
                    break

        cost_stat.append(step_cost_list)
        print(f"final cost: {step_cost_list[-1]}, total cost: {sum(step_cost_list)}")

        # save results
        # planning_res_path = os.path.join(out_dir, "planning_res.npy")
        # np.save(planning_res_path, planning_res_list)
        # gt_states_path = os.path.join(out_dir, "gt_states.npy")
        # np.save(gt_states_path, np.array(gt_states))
        planning_res_path = os.path.join(out_dir, f"planning_res_{i:04d}.json")
        with open(planning_res_path, "w") as f:
            json.dump(planning_res_list, f, indent=4, separators=(",", ": "))
        gt_states_path = os.path.join(out_dir, f"gt_states_{i:04d}.json")
        with open(gt_states_path, "w") as f:
            json.dump(gt_states, f, indent=4, separators=(",", ": "))
        env.save_gif(os.path.join(out_dir, f"planning_vis_{i:04d}.gif"))
        env.close()

    cost_stat = np.array(cost_stat)  # (num_test, max_steps)
    avg_step_cost = np.mean(cost_stat, axis=0)
    print(f"Average step cost over time over {num_test} test cases: {avg_step_cost}")
    plot_cost_stat(cost_stat, out_dir)
    return

if __name__ == "__main__":
    main()