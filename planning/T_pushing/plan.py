import os
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import pickle

from envs.T_pushing.t_sim import generate_init_target_states, T_Sim
from models.load import load_model
from models.T_pushing.dt_dyn import T_Dynamics
from models.T_pushing.ct_dyn import Continuous_T_Dynamics
from models.T_pushing.ct_ctl import T_controller
from planning.planner import MPPIPlanner, CEMPlanner
from planning.T_pushing.plan_utils import generate_test_cases, get_abs_states, make_rollout_and_reward_fns, plot_cost_stat, plot_plan_from_poses
from utils.T_pushing import hole_to_walls_aabbs
from utils.misc import box_corners_nd

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml")
def main(config: DictConfig):
    data_config = config["data"]
    train_config = config["train_dt_dyn"]
    planning_config = config["planning"]
    verbose = planning_config.get("verbose", False)
    seed = config["settings"]["seed"]

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

    hole_config = planning_config.get("hole", {})
    hole_enable = hole_config.get("enable", False)
    obs_dict = {}
    if hole_enable:
        hole_center = hole_config["center"]
        hole_size = hole_config["size"]
        c_wall, h_wall = hole_to_walls_aabbs(hole_center, hole_size, window_size=data_config["window_size"])
        obs_dict["obs_pos_list"] = c_wall
        obs_dict["obs_size_list"] = h_wall
        obs_dict["obs_norm"] = 1
        param_dict.update(obs_dict)

    enable_ctl = planning_config.get("enable_ctl", False)
    if enable_ctl:
        assert abs_pose, "Controller can only be enabled when using absolute pose prediction."
    ct_ctl_dir = config["test_models"]["ct_ctl_dir"]
    ct_ctl: T_controller = load_model(model_dir=ct_ctl_dir, model_type="ct_ctl", mode="best")
    ctl_frequency = ct_ctl.ctl_frequency

    out_dir = os.path.join(planning_config["out_path"], f"{dt_dyn_dir[-6:]}_{ct_ctl_dir[-6:]}")
    os.makedirs(out_dir, exist_ok=True)

    noise_type = planning_config.get("disturbance", {}).get("type", "none")
    assert noise_type in {"none", "normal", "uniform"}, f"Unknown disturbance type: {noise_type}"
    if noise_type != "none":
        assert pred_mode == "pose"
    noise_init = planning_config.get("disturbance", {}).get("init", 0.0)
    noise_inter = planning_config.get("disturbance", {}).get("inter", 0.0)

    # reward and planning part
    rollout_fn, reward_fn, step_cost_fn, step_cost_fn_np = make_rollout_and_reward_fns(
        dt_dyn,
        config,
        abs_pose,
        pred_mode,
        reach_config=config.get("reachability", {}),
    )
    planner_type = planning_config.get("planner", "mppi").lower()
    if planner_type == "mppi":
        planner = MPPIPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    elif planner_type == "cem":
        planner = CEMPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    def trans_fn(env_dict):
        pusher_pos = jnp.array(env_dict["pusher_pos"]) / scale
        if pred_mode == "pose":
            state_cur = jnp.array(np.concatenate([env_dict["com_pos"] / scale, env_dict["angle"]], axis=0))
            env_state = np.concatenate([np.array(env_dict["com_pos"]), np.array(env_dict["angle"]), env_dict["pusher_pos"]], axis=0)
        else:
            state_cur = jnp.array(env_dict["state"][:state_dim]) / scale
            env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"]], axis=0)
        state_cur = jnp.concatenate([state_cur, pusher_pos], axis=0)
        return state_cur, env_state, pusher_pos

    num_test = planning_config["num_test"]
    test_id = planning_config.get("test_id", 0)
    init_pusher_pos_list, init_pose_list, target_pose_list = generate_test_cases(seed, num_test, test_id=test_id)

    key = jax.random.PRNGKey(seed)
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
            _, env_state, _ = trans_fn(env_dict)

        # executtion loop
        planning_res_list = []
        step_cost_list = []
        gt_states = [env_state]
        t = 0
        succeed = False
        init_follow = True
        while t < max_steps:
            env_dict = env.get_env_state(not abs_pose)
            state_cur, _, pusher_pos = trans_fn(env_dict)
            key = jax.random.PRNGKey(seed + t)

            noise_param = noise_init
            noise = jnp.zeros((T_dim,))
            if not succeed:
                key, subkey = jax.random.split(key)
                if noise_type == "normal":
                    noise = jax.random.normal(subkey, shape=(T_dim,)) * noise_param
                elif noise_type == "uniform":
                    noise = jax.random.uniform(subkey, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param
            state_cur = state_cur.at[0:T_dim].add(noise)
            # env.force_update([[noise[0] * scale, noise[1] * scale, noise[2]]])  # apply disturbance

            key, subkey = jax.random.split(key)
            init_act_seq = jax.random.uniform(subkey,(horizon, action_dim),minval=action_lower_lim,maxval=action_upper_lim,)
            key, subkey = jax.random.split(key)
            # with jax.disable_jit():
            #     planning_res = eqx.filter_jit(planner.trajectory_optimization)(key, state_cur, init_act_seq, skip=False, target_state=scaled_target_state, pusher_pos=pusher_pos)
            planning_res = eqx.filter_jit(planner.trajectory_optimization)(subkey, state_cur, init_act_seq, skip=False, target_state=scaled_target_state, pusher_pos=pusher_pos)
            scaled_act_seq = planning_res["act_seq"]
            scaled_state_seq = planning_res["state_seq"]
            act_seq = scaled_act_seq * scale  # (horizon, action_dim)
            if abs_pose:
                abs_scaled_state_seq = jnp.concatenate([state_cur[None, :], scaled_state_seq], axis=0)
                abs_state_seq = abs_scaled_state_seq * scale
                if pred_mode == "pose":
                    abs_state_seq = abs_state_seq.at[:, pose_dim - 1].set(abs_state_seq[:, pose_dim - 1]/scale)  # do not scale angle
                pusher_pos_seq = abs_scaled_state_seq[:, -action_dim:] * scale
            else:
                abs_state_seq, pusher_pos_seq = get_abs_states(scaled_state_seq * scale[None, :, :], pusher_pos * scale, act_seq[None, :, :], pred_mode=pred_mode)
                abs_state_seq = abs_state_seq[0]
                pusher_pos_seq = pusher_pos_seq[0]
            res = {
                "time_step": t,
                "act_seq": act_seq,
                "state_seq": abs_state_seq,
                "pusher_pos_seq": pusher_pos_seq,
                "planning_res": planning_res,
            }
            if verbose:
                print(f"reach vol: {planning_res['aux']['eval_out']['reach_aux'].get('reach_vol', None)}")
            planning_res_list.append(res)
            for step in range(n_act_step):
                sub_target = abs_scaled_state_seq[step + 1, :-action_dim]
                ref_action = scaled_act_seq[step, :]
                sub_env_states = []
                for ctl_step in range(ctl_frequency):
                    if enable_ctl:
                        if (step_cost_fn(sub_target, state_cur[:-action_dim]) < 4e-1) and (init_follow):
                            next_action = ref_action
                            if verbose:
                                print("   skip controller")
                        else:
                            next_action = eqx.filter_jit(ct_ctl.forward_batchless)(state_cur, sub_target, ref_action)
                            init_follow = False
                            if verbose:
                                print(f"   controller action: {next_action}")
                    else:
                        next_action = ref_action
                    next_pusher_pos = (pusher_pos + next_action) * scale
                    env_dict = env.update((next_pusher_pos[0], next_pusher_pos[1]), rel=False, n_sim_time=1/ctl_frequency)
                    state_cur, env_state, pusher_pos = trans_fn(env_dict)

                    noise_param = noise_inter
                    noise = jnp.zeros((T_dim,))
                    if not succeed:
                        key, subkey = jax.random.split(key)
                        if noise_type == "normal":
                            noise = jax.random.normal(subkey, shape=(T_dim,)) * noise_param
                        elif noise_type == "uniform":
                            noise = jax.random.uniform(subkey, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param

                    # state_cur = state_cur.at[0:T_dim].add(noise)
                    env.force_update([[noise[0] * scale, noise[1] * scale, noise[2]]])  # apply disturbance

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
                gt_states.append(env_state)
                if verbose:
                    print(f"   step {t} cost: {step_cost}, action: {env_dict['action']}")
                step_cost_list.append(step_cost)
                if step_cost < 20:
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
        planning_res_path = os.path.join(out_dir, f"planning_res_{i:04d}.pkl")
        with open(planning_res_path, "wb") as f:
            pickle.dump(planning_res_list, f)
        gt_states_path = os.path.join(out_dir, f"gt_states_{i:04d}.pkl")
        with open(gt_states_path, "wb") as f:
            pickle.dump(gt_states, f)
        env.save_gif(os.path.join(out_dir, f"sim_vis_{i:04d}.gif"))
        if pred_mode == "pose":
            act_seqs = np.array([d["act_seq"] for d in planning_res_list])
            state_seqs = np.array([d["state_seq"] for d in planning_res_list])[..., :pose_dim]
            pusher_pos_seqs = np.array([d["pusher_pos_seq"] for d in planning_res_list])

            plot_plan_from_poses(
                state_seqs=state_seqs[None],
                pusher_pos_seqs=pusher_pos_seqs,
                target_pose=target_pose,
                stem_size=data_config["stem_size"],
                bar_size=data_config["bar_size"],
                window_size=(data_config["window_size"], data_config["window_size"]),
                obs_dict=obs_dict,
                fps=5,
                save_path=os.path.join(out_dir, f"plan_vis_{i:04d}.gif"),
            )
            enable_reach = planning_config.get("reach_in_obj", {}).get("enable", False) or (planning_config.get("refinement", {}).get('enable', False) and planning_config.get("refinement", {}).get("reach_in_obj", False))
            if enable_reach:
                # r_lo_seqs, r_up_seqs: (n_sim_steps+1, horizon+1, 3)
                r_lo_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_lo'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :pose_dim]
                r_up_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_up'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :pose_dim]

                r_lo_seqs[..., :2] = r_lo_seqs[..., :2] * scale
                r_up_seqs[..., :2] = r_up_seqs[..., :2] * scale

                # sample from r_lo_seqs and r_up_seqs: choose corners for each dimension, 2^3 = 8 samples
                sample_states = box_corners_nd(r_lo_seqs, r_up_seqs)  # (8, n_sim_steps+1, horizon+1, 3)
                n_samples = sample_states.shape[0]
                sample_states = np.random.uniform(size=(n_samples, *state_seqs.shape))
                sample_state_seqs = r_lo_seqs[None] + sample_states * (r_up_seqs - r_lo_seqs)[None]
                plot_plan_from_poses(
                    state_seqs=sample_state_seqs,
                    pusher_pos_seqs=pusher_pos_seqs,
                    target_pose=target_pose,
                    stem_size=data_config["stem_size"],
                    bar_size=data_config["bar_size"],
                    window_size=(data_config["window_size"], data_config["window_size"]),
                    obs_dict=obs_dict,
                    fps=5,
                    save_path=os.path.join(out_dir, f"plan_reach_vis_{i:04d}.gif"),
                )
        env.close()

    cost_stat = np.array(cost_stat)  # (num_test, max_steps)
    avg_step_cost = np.mean(cost_stat, axis=0)
    print(f"Average step cost over time over {num_test} test cases: {avg_step_cost}")
    plot_cost_stat(cost_stat, os.path.join(out_dir, "step_costs.png"))
    return

if __name__ == "__main__":
    main()