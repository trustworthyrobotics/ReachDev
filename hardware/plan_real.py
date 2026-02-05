import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import jax

from hardware.iiwa import IiwaHardwareEnv
from hardware.realsense import PlanarPoseDetectorAPI
# jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import pickle
import time
import datetime

from envs.T_pushing.t_sim import generate_init_target_states, T_Sim
from models.load import load_model
from models.T_pushing.dt_dyn import T_Dynamics
from models.T_pushing.ct_dyn import Continuous_T_Dynamics
from models.T_pushing.ct_ctl import T_controller
from planning.planner import MPPIPlanner, CEMPlanner
from planning.T_pushing.plan_utils import generate_test_cases, get_abs_states, make_rollout_and_reward_fns, plot_cost_stat, plot_plan_from_poses
from utils.T_pushing import hole_to_walls_aabbs
from utils.misc import box_corners_nd

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing_real.yaml")
def main(config: DictConfig):
    if "testing" in config:
        testing_config = config["testing"]
        mode = testing_config.get("mode", "certified")
        assert mode in {"certified", "regular"}, f"Unknown testing mode: {mode}"
        model_config = testing_config[mode]
        with open_dict(config):
            config["test_models"] = model_config
    data_config = config["data"]
    train_config = config["train_dt_dyn"]
    planning_config = config["planning"]
    verbose = planning_config.get("verbose", False)
    seed = config["settings"]["seed"]
    key = jax.random.PRNGKey(seed)

    dt_dyn_dir = config["test_models"]["dt_dyn_dir"]
    dt_dyn: T_Dynamics = load_model(model_dir=dt_dyn_dir, model_type="dt_dyn", mode="best")
    abs_pose = dt_dyn.abs_pose
    pred_mode = dt_dyn.pred_mode

    # -------------- real --------------
    # real_to_sim_scale = planning_config.get("real_to_sim_scale", 0.6)
    real_to_sim_scale = 1.0 
    assert real_to_sim_scale == 1.0
    sim_to_real_scale = 1.0 / real_to_sim_scale

    stem_size = np.array(data_config["stem_size"]) * sim_to_real_scale
    bar_size = np.array(data_config["bar_size"]) * sim_to_real_scale
    pusher_size = np.array(data_config["pusher_size"]) * sim_to_real_scale
    window_size = int(data_config["window_size"] * sim_to_real_scale)
    scale = float(data_config["scale"]) * sim_to_real_scale
    action_bound = float(planning_config["action_bound"]) * sim_to_real_scale
    # -------------- real --------------

    param_dict = {"stem_size": stem_size,
                "bar_size": bar_size, 
                "pusher_size": pusher_size,
                "save_img": True,
                "enable_vis": False,
                "window_size": window_size,}

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
        hole_center = np.array(hole_config["center"]) * sim_to_real_scale
        hole_size = np.array(hole_config["size"]) * sim_to_real_scale
        c_wall, h_wall = hole_to_walls_aabbs(hole_center, hole_size, window_size=window_size)
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
    if planning_config.get("add_timestamp", False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_dir, timestamp)
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
    )
    planner_type = planning_config.get("planner", "mppi").lower()
    if planner_type == "mppi":
        planner = MPPIPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    elif planner_type == "cem":
        planner = CEMPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    # JIT compile
    compile_start = time.time()
    jit_trajopt = eqx.filter_jit(planner.trajectory_optimization)
    jit_trajopt(key, jnp.zeros((T_dim + action_dim,)), jnp.zeros((horizon, action_dim)), skip=True, target_state=jnp.zeros((T_dim,)), pusher_pos=jnp.zeros((action_dim,)))
    jit_trajopt(key, jnp.zeros((T_dim + action_dim,)), jnp.zeros((horizon, action_dim)), skip=False, target_state=jnp.zeros((T_dim,)), pusher_pos=jnp.zeros((action_dim,)))

    jit_ctl = eqx.filter_jit(ct_ctl.forward_batchless)
    jit_ctl(jnp.zeros((T_dim + action_dim,)), jnp.zeros((T_dim,)), jnp.zeros((action_dim,)))
    compile_time = time.time() - compile_start
    print(f"JIT compilation time: {compile_time} seconds")

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

    cost_stat = []
    plan_time_stat = []
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
                init_pose, target_pose, param_dict={"stem_size": stem_size, "bar_size": bar_size}
            )
            scaled_target_state = target_state / scale
        scaled_target_state = jnp.array(scaled_target_state)

        # -------------- real --------------
        env = IiwaHardwareEnv(
            use_ik=True,
            period_sec=0.01,
            max_delta_per_step=0.001,
            realtime_rate=1.0,
            ik_update_period_sec=0.01
        )
        # One-call safe startup (holds current pose, then enables).
        env.start(timeout_sec=1.0)

        z_hi = 0.38
        z_lo = 0.28
        # move to init position (transformation from workspace to robot frame is inside the function)
        env.move_to_target_xyz_in_world_mm(np.array([init_pusher_pos[0], init_pusher_pos[1], z_hi], dtype=float), timeout_sec=10.0, verbose=False, hold_count_required=5)

        env.move_to_target_xyz_in_world_mm(np.array([init_pusher_pos[0], init_pusher_pos[1], z_lo], dtype=float), timeout_sec=10.0, verbose=False, hold_count_required=5)

        # detector = PlanarPoseDetectorAPI(w = 1280, h = 720, fps = 30, max_stored_images = 1000)
        detector = PlanarPoseDetectorAPI(w = 640, h = 360, fps = 30, max_stored_images = 1000)

        # mimic sim api
        def get_state():
            # real obj pose (x, y, theta) in workspace frame in mm.
            state_cur = detector.get_planar_pose_in_world_mm(blocking=True, store_image=False)

            # current pusher pos in base frame in mm
            pusher_pos = env.get_tool_metrics_in_world_mm()[:2]
            
            env_state = np.concatenate([state_cur, pusher_pos], axis=0)

            # in NN scale
            state_cur[:2] = state_cur[:2] / scale
            pusher_pos = pusher_pos / scale
            state_cur = jnp.array(state_cur)
            pusher_pos = jnp.array(pusher_pos)
            return state_cur, env_state, pusher_pos

        state_cur, env_state, pusher_pos = get_state()
        # TODO: visualize initial setup

        # -------------- real --------------

        # executtion loop
        planning_res_list = []
        step_cost_list = []
        gt_states = [env_state]
        t = 0
        succeed = False
        init_follow = True
        init_follow_steps = planning_config.get("init_follow_steps", 1)
        while t < max_steps:
            # -------------- real --------------
            state_cur, env_state, pusher_pos = get_state()
            # -------------- real --------------

            key = jax.random.PRNGKey(seed + t)

            # noise_param = noise_init
            # noise = jnp.zeros((T_dim,))
            # if not succeed:
            #     key, subkey = jax.random.split(key)
            #     if noise_type == "normal":
            #         noise = jax.random.normal(subkey, shape=(T_dim,)) * noise_param
            #     elif noise_type == "uniform":
            #         noise = jax.random.uniform(subkey, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param
            # state_cur = state_cur.at[0:T_dim].add(noise)

            # key, subkey = jax.random.split(key)
            # init_act_seq = jax.random.uniform(subkey,(horizon, action_dim),minval=action_lower_lim,maxval=action_upper_lim,)

            init_act_seq = jnp.zeros((horizon, action_dim))
            key, subkey = jax.random.split(key)
            # with jax.disable_jit():
            #     planning_res = eqx.filter_jit(planner.trajectory_optimization)(key, state_cur, init_act_seq, skip=False, target_state=scaled_target_state, pusher_pos=pusher_pos)
            start_plan_time = time.time()
            planning_res = jit_trajopt(subkey, state_cur, init_act_seq, skip=succeed, target_state=scaled_target_state, pusher_pos=pusher_pos)
            plan_time_stat.append(time.time() - start_plan_time)
            if verbose and 'collision_loss' in planning_res['aux']['eval_out'] and planning_res['aux']['eval_out']['collision_loss'] > 0:
                print(f"Step {t} planning result:")
                print(f"   collision loss: {planning_res['aux']['eval_out']['collision_loss']}")

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

                if (step_cost_fn(sub_target, state_cur[:-action_dim]) < 0.15) or (init_follow) or succeed:
                    use_ctl = False
                else:
                    use_ctl = enable_ctl

                # for ctl_step in range(ctl_frequency):
                #     if use_ctl:
                #         # if verbose:
                #         #     print(f"   step_cost: {step_cost_fn(sub_target, state_cur[:-action_dim])}, init_follow: {init_follow}")
                #         next_action = jit_ctl(state_cur, sub_target, ref_action)
                #         if verbose:
                #             print(f"   controller action: {next_action}")
                #     else:
                #         if verbose:
                #             print("   skip controller")
                #         next_action = ref_action
                #     next_pusher_pos = (pusher_pos + next_action) * scale
                #     env.move_to_target_xyz_in_world_mm(np.array([next_pusher_pos[0], next_pusher_pos[1], z_lo], dtype=float), timeout_sec=5.0, verbose=False, hold_count_required=1)
                #     state_cur, env_state, pusher_pos = get_state()

                #     # noise_param = noise_inter
                #     # noise = jnp.zeros((T_dim,))
                #     # if not succeed:
                #     #     key, subkey = jax.random.split(key)
                #     #     if noise_type == "normal":
                #     #         noise = jax.random.normal(subkey, shape=(T_dim,)) * noise_param
                #     #     elif noise_type == "uniform":
                #     #         noise = jax.random.uniform(subkey, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param

                #     # # state_cur = state_cur.at[0:T_dim].add(noise)
                #     # env.force_update([[noise[0] * scale, noise[1] * scale, noise[2]]])  # apply disturbance

                #     sub_env_states.append(env_state)
                # env_state = np.array(sub_env_states)
                # step_cost = step_cost_fn_np(env_state[-1][:-action_dim], target_state)

                next_pusher_pos = pusher_pos_seq[step + 1, :]
                env.move_to_target_xyz_in_world_mm(np.array([next_pusher_pos[0], next_pusher_pos[1], z_lo], dtype=float), timeout_sec=5.0, verbose=False, hold_count_required=1)
                state_cur, env_state, pusher_pos = get_state()
                step_cost = step_cost_fn_np(env_state[:-action_dim], target_state)

                t += 1
                if t > init_follow_steps:
                    init_follow = False
                gt_states.append(env_state)
                if verbose:
                    print(f"   step {t} cost: {step_cost}")
                step_cost_list.append(step_cost)
                if (not succeed) and step_cost < 0.25:
                    print(f"Task succeeded at step {t} with step cost {step_cost}")
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

        # -------------- real --------------
        # imgs_dir = os.path.join(out_dir, f"vis_{i:04d}")
        # os.makedirs(imgs_dir, exist_ok=True)
        # detector.save_images(os.path.join(out_dir, f"vis_{i:04d}", "img"))

        # -------------- real --------------


        if pred_mode == "pose":
            act_seqs = np.array([d["act_seq"] for d in planning_res_list])
            state_seqs = np.array([d["state_seq"] for d in planning_res_list])[..., :pose_dim]
            pusher_pos_seqs = np.array([d["pusher_pos_seq"] for d in planning_res_list])

            plot_plan_from_poses(
                state_seqs=state_seqs[None],
                pusher_pos_seqs=pusher_pos_seqs,
                target_pose=target_pose,
                stem_size=stem_size,
                bar_size=bar_size,
                window_size=(window_size, window_size),
                obs_dict=obs_dict,
                fps=5,
                save_path=os.path.join(out_dir, f"plan_vis_{i:04d}.gif"),
            )
            reach_config = planning_config.get("reach_in_obj", {})
            refine_config = planning_config.get("refinement", {})
            reach_refine_config = refine_config.get("reach_in_obj", {})
            enable_reach = reach_config.get("enable", False) or (refine_config.get('enable', False) and reach_refine_config.get("enable", False))
            if enable_reach:
                # r_lo_seqs, r_up_seqs: (n_sim_steps+1, horizon+1, 3)
                r_lo_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_lo'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :pose_dim]
                r_up_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_up'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :pose_dim]
                reach_vols = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['reach_vol'] for d in planning_res_list]).reshape((state_seqs.shape[0], -1))
                print(f"Average reach volume over time: {np.mean(reach_vols, axis=0)}")

                vis_reach_steps = planning_config.get("vis_reach_steps", horizon)
                r_lo_seqs = r_lo_seqs[:, :vis_reach_steps + 1, :]
                r_up_seqs = r_up_seqs[:, :vis_reach_steps + 1, :]
                state_seqs = state_seqs[:, :vis_reach_steps + 1, :]
                pusher_pos_seqs = pusher_pos_seqs[:, :vis_reach_steps + 1, :]

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
                    stem_size=stem_size,
                    bar_size=bar_size,
                    window_size=(window_size, window_size),
                    obs_dict=obs_dict,
                    fps=5,
                    save_path=os.path.join(out_dir, f"plan_reach_vis_{i:04d}.gif"),
                )
        

        # -------------- real --------------
        env.move_to_target_xyz_in_world_mm(np.array([init_pusher_pos[0], init_pusher_pos[1], z_hi], dtype=float), timeout_sec=10.0, verbose=False, hold_count_required=5)
        detector.close()
        # -------------- real --------------

    cost_stat = np.array(cost_stat)  # (num_test, max_steps)
    avg_step_cost = np.mean(cost_stat, axis=0)
    print(f"Average step cost over time over {num_test} test cases: {avg_step_cost}")
    plot_cost_stat(cost_stat, os.path.join(out_dir, "step_costs.png"))

    plan_time_stat = np.array(plan_time_stat)
    avg_plan_time = np.mean(plan_time_stat)
    print(f"Average planning time per step over {num_test} test cases: {avg_plan_time} seconds")

    # save overall stats
    stats = {
        "cost_stat": cost_stat,
        "plan_time_stat": plan_time_stat,
        "jit_compile_time": compile_time,
    }

    stats_path = os.path.join(out_dir, "planning_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # copy config file to out_dir
    with open(os.path.join(out_dir, "planning_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))


    return

if __name__ == "__main__":
    main()