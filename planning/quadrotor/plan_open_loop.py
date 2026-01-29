import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import jax

from utils.misc import box_corners_nd, random_sample_nd
jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import pickle

from envs.quadrotor.quad_sim import Quad_Sim_Ctl
from models.load import load_model
from models.quadrotor.dt_dyn import Quad_Dynamics
from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics
from models.quadrotor.ct_ctl import MLP_Controller, PID_Controller
from planning.planner import MPPIPlanner, CEMPlanner
from planning.quadrotor.plan_utils import generate_test_cases, make_rollout_and_reward_fns, plot_cost_stat, plot_planning_animation
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories, sample_vel_cmd_sequence


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml")
def main(config: DictConfig):
    if "testing" in config:
        testing_config = config["testing"]
        mode = testing_config.get("mode", "certified")
        assert mode in {"certified", "regular"}, f"Unknown testing mode: {mode}"
        model_config = testing_config[mode]
        with open_dict(config):
            config["test_models"] = model_config
    task_name = config["settings"]["task_name"]
    data_config = config["data"]
    train_config = config["train_dt_dyn"]
    planning_config = config["planning"]
    verbose = planning_config.get("verbose", False)
    seed = config["settings"]["seed"]

    dt_dyn_dir = config["test_models"]["dt_dyn_dir"]
    dt_dyn: Quad_Dynamics = load_model(model_dir=dt_dyn_dir, model_type="dt_dyn", mode="best", task_name=task_name)

    action_bound = planning_config["action_bound"]
    scale = float(data_config["scale"])
    assert scale == 1
    ct_state_dim, ct_action_dim = data_config["ct_state_dim"], data_config["ct_action_dim"]
    dt_state_dim, dt_action_dim = data_config["dt_state_dim"], data_config["dt_action_dim"]

    action_lower_lim = -action_bound * jnp.ones((dt_action_dim,)) / scale
    action_upper_lim = action_bound * jnp.ones((dt_action_dim,)) / scale
    action_bounds = jnp.stack([action_lower_lim, action_upper_lim], axis=0)[None]
    acc_limits = jnp.full((1, 3), data_config.get("acc_limits", 1.0))
    
    cost_norm, only_final_cost = planning_config["cost_norm"], planning_config["only_final_cost"]
    max_steps = planning_config["max_steps"] + 1  # +1 to account for initial step
    horizon = planning_config["horizon"]
    n_act_step = planning_config["n_act_step"]

    enable_ctl = planning_config.get("enable_ctl", True)
    assert enable_ctl, "Only support closed-loop control with CT controller."
    ct_ctl_dir = config["test_models"]["ct_ctl_dir"]


    use_pid = planning_config.get("use_pid", False)
    out_dir = os.path.join(planning_config["out_path"], f"{'pid' if use_pid else 'mlp'}_{dt_dyn_dir[-6:]}_{ct_ctl_dir[-6:]}")
    os.makedirs(out_dir, exist_ok=True)

    if use_pid:
        ct_ctl = PID_Controller(data_config)
    else:
        ct_ctl: MLP_Controller = load_model(model_dir=ct_ctl_dir, model_type="ct_ctl", mode="best", task_name=task_name)

    ctl_frequency = ct_ctl.frequency
    dyn_frequency = dt_dyn.frequency
    assert ctl_frequency % dyn_frequency == 0
    n_ctl_per_dyn = round(ctl_frequency / dyn_frequency)

    dist_config = planning_config.get("disturbance", {})
    noise_type = dist_config.get("type", "none")
    assert noise_type in {"none", "normal", "uniform"}, f"Unknown disturbance type: {noise_type}"
    noise_init = dist_config.get("init", 0.0)
    noise_inter = dist_config.get("inter", 0.0)
    per_act = dist_config.get("per_act", False)

    # reward and planning part
    rollout_fn, reward_fn, step_cost_fn, step_cost_fn_np = make_rollout_and_reward_fns(
        dt_dyn,
        planning_config,
        reach_config=config.get("reachability", {}),
    )
    planner_type = planning_config.get("planner", "mppi").lower()
    if planner_type == "mppi":
        planner = MPPIPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    elif planner_type == "cem":
        planner = CEMPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    num_test = planning_config["num_test"]
    test_id = planning_config.get("test_id", 0)
    init_pose_list, target_pose_list = generate_test_cases(seed, num_test, test_id=test_id)

    open_loop = True
    if open_loop:
        max_steps = horizon
        n_act_step = horizon
        noise_init = 0.3
        noise_inter = 0.0

        test_id = 2
        init_pose_list, target_pose_list = generate_test_cases(seed, num_test, test_id=test_id)

        sample_init_pose_list = []
        num_sample_per_case = 16
        for i in range(num_test):
            init_pose = init_pose_list[i]
            init_bound = np.ones_like(init_pose) * scale * noise_init
            init_pose_lo = init_pose - init_bound
            init_pose_up = init_pose + init_bound
            sample_init_pose = random_sample_nd(init_pose_lo, init_pose_up, num_samples=num_sample_per_case, seed=seed)
            sample_init_pose_list.append(sample_init_pose)

        sample_init_pose_list = np.concatenate(sample_init_pose_list, axis=0)
        
        num_sample_per_case = sample_init_pose_list.shape[0] // num_test
        num_test = sample_init_pose_list.shape[0]
        open_loop_cost_stat = []
        summary_dict = {}

    key = jax.random.PRNGKey(seed)
    cost_stat = []
    for i in range(num_test):
        if open_loop:
            sample_init_pose = sample_init_pose_list[i]
            target_pose = target_pose_list[i // num_sample_per_case]
        else:
            init_pose = init_pose_list[i]
            target_pose = target_pose_list[i]
        print(f"Test case {i}:")

        target_state = target_pose[:dt_state_dim]
        scaled_target_state = target_state / scale
        scaled_target_state = jnp.array(scaled_target_state)
        env = Quad_Sim_Ctl(data_config, num_quads=1, init_poses=init_pose[None], target_poses=target_pose[None], controller=ct_ctl)
        env_dict = env.get_env_states()
        env_state = np.concatenate([env_dict["state"], jnp.zeros((1, dt_action_dim,)), env_dict["action"]], axis=-1).squeeze()

        if open_loop:
            # sample env with disturbance, evaluate execution performance
            sample_env = Quad_Sim_Ctl(data_config, num_quads=1, init_poses=sample_init_pose[None], target_poses=target_pose[None], controller=ct_ctl)
            env_dict = sample_env.get_env_states()
            env_state = np.concatenate([env_dict["state"], jnp.zeros((1, dt_action_dim,)), env_dict["action"]], axis=-1).squeeze()            

        # executtion loop
        planning_res_list = []
        step_cost_list = []
        gt_states = [env_state[None]]
        t = 0
        succeed = False
        while t < max_steps:
            env_dict = env.get_env_states()
            state_cur = jnp.array(env_dict["state"]).squeeze()[:dt_state_dim] / scale
            if open_loop:
                sample_env_dict = sample_env.get_env_states()
                state_cur_sample = jnp.array(sample_env_dict["state"]).squeeze()[:dt_state_dim] / scale
            
            key = jax.random.PRNGKey(seed + t)

            # key, subkey = jax.random.split(key)
            # noise_param = noise_init
            # if noise_type == "normal":
            #     noise = jax.random.normal(subkey, shape=(dt_state_dim,)) * noise_param
            # elif noise_type == "uniform":
            #     noise = jax.random.uniform(subkey, shape=(dt_state_dim,), minval=-1.0, maxval=1.0) * noise_param
            # else:
            #     noise = jnp.zeros((dt_state_dim,))
            # state_cur = state_cur.at[0:dt_state_dim].add(noise)
            # # env.force_update([[noise[0] * scale, noise[1] * scale, noise[2]]])  # apply disturbance

            if open_loop and i % num_sample_per_case == 0:
                key, subkey = jax.random.split(key)
                init_act_seq = 0 * jax.random.uniform(subkey,(horizon, dt_action_dim),minval=action_lower_lim,maxval=action_upper_lim,)
                # init_act_seq = eqx.filter_jit(sample_vel_cmd_sequence)(subkey, amax=acc_limits, num_quads=1, dt=1/dyn_frequency, n_steps=horizon, v0=state_cur[-dt_action_dim:][None], v_bounds=action_bounds).squeeze(axis=1)[1:]
                key, subkey = jax.random.split(key)
                # with jax.disable_jit():
                #     planning_res = eqx.filter_jit(planner.trajectory_optimization)(key, state_cur, init_act_seq, skip=succeed, target_state=scaled_target_state)
                planning_res = eqx.filter_jit(planner.trajectory_optimization)(subkey, state_cur, init_act_seq, skip=succeed, target_state=scaled_target_state)
                scaled_act_seq = planning_res["act_seq"]
            if open_loop:
                sample_planning_res = eqx.filter_jit(planner.trajectory_optimization)(subkey, state_cur_sample, scaled_act_seq, skip=True, target_state=scaled_target_state)

            scaled_state_seq = planning_res["state_seq"]
            act_seq = scaled_act_seq * scale  # (horizon, action_dim)
            scaled_state_seq = jnp.concatenate([state_cur[None, :], scaled_state_seq], axis=0)
            state_seq = scaled_state_seq * scale

            res = {
                "time_step": t,
                "act_seq": act_seq,
                "state_seq": state_seq,
                "planning_res": planning_res,
            }
            if verbose:
                print(f"reach vol: {planning_res['aux']['eval_out']['reach_aux'].get('reach_vol', None)}")
            planning_res_list.append(res)
            for step in range(n_act_step):
                v_cmd = scaled_act_seq[step, :]
                sub_env_states = []

                if per_act:
                    key, subkey = jax.random.split(key)
                    noise_param = noise_inter
                    if noise_type == "normal":
                        noise = jax.random.normal(subkey, shape=(ct_state_dim,)) * noise_param
                    elif noise_type == "uniform":
                        noise = jax.random.uniform(subkey, shape=(ct_state_dim,), minval=-1.0, maxval=1.0) * noise_param
                    else:
                        noise = jnp.zeros((ct_state_dim,))

                    env.force_update(noise[:ct_state_dim][None])  # apply disturbance
                    if open_loop:
                        sample_env.force_update(noise[:ct_state_dim][None])  # apply disturbance

                for ctl_step in range(n_ctl_per_dyn):
                    if not per_act:
                        key, subkey = jax.random.split(key)
                        noise_param = noise_inter
                        if noise_type == "normal":
                            noise = jax.random.normal(subkey, shape=(ct_state_dim,)) * noise_param
                        elif noise_type == "uniform":
                            noise = jax.random.uniform(subkey, shape=(ct_state_dim,), minval=-1.0, maxval=1.0) * noise_param
                        else:
                            noise = jnp.zeros((ct_state_dim,))

                        env.force_update(noise[:dt_state_dim][None] * scale)  # apply disturbance
                        if open_loop:
                            sample_env.force_update(noise[:dt_state_dim][None] * scale)  # apply disturbance

                    next_action = v_cmd[None]
                    if verbose:
                        print(f"   controller action: {next_action}")

                    # with jax.disable_jit():
                    #     env_dict = env.update(next_action, n_sim_time=1/ctl_frequency)
                    env_dict = env.update(next_action, n_sim_time=1/ctl_frequency)
                    if open_loop:
                        env_dict = sample_env.update(next_action, n_sim_time=1/ctl_frequency)

                    # state_cur = jnp.array(env_dict["state"]).squeeze()[:dt_state_dim] / scale
                    env_state = np.concatenate([env_dict["state"], next_action, env_dict["action"]], axis=-1).squeeze()

                    sub_env_states.append(env_state)
                env_state = np.array(sub_env_states)
                step_cost = step_cost_fn_np(env_state[-1][:dt_state_dim], target_state)

                t += 1
                gt_states.append(env_state)
                if verbose:
                    print(env_state[-1][:dt_state_dim])
                    print(f"   step {t} cost: {step_cost}, action: {env_dict['action']}")
                step_cost_list.append(step_cost)
                if step_cost < 1:
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
        # env.save_gif(os.path.join(out_dir, f"planning_vis_{i:04d}.gif"))

        X_gt = jnp.concatenate(gt_states, axis=0)[:, None]
        v_cmds = X_gt[:, :, ct_state_dim:ct_state_dim + dt_action_dim]
        U_gt = X_gt[:, :, -ct_action_dim:]
        plot_3d_trajectories(X_gt[:, :, :3], num_quads=1, dt=ct_ctl.dt, out_path=os.path.join(out_dir, f"gt_trajectories_{i}.png"), targets=target_pose[None, :3], obs_config=planning_config.get("obstacle", None))
        plot_quad_states_actions(X_gt[:, 0, :dt_state_dim], v_cmds[:, 0], dt=ct_ctl.dt, out_path=os.path.join(out_dir, f"gt_states_vcmd_{i}.png"))
        plot_quad_states_actions(X_gt[:, 0, :ct_state_dim], U_gt[:, 0], dt=ct_ctl.dt, out_path=os.path.join(out_dir, f"gt_states_actions_{i}.png"))
        env.close()

        act_seqs = np.array([d["act_seq"] for d in planning_res_list])
        state_seqs = np.array([d["state_seq"] for d in planning_res_list])[..., :dt_state_dim]

        if open_loop:
            # state_seqs: predicted state seqs starting from unnoisy initial state
            # GT state of noisy execution
            gt_states = np.concatenate([gt[-1:, :dt_state_dim] for gt in gt_states], axis=0)
            # predicted state seqs starting from noisy initial state
            pred_state_seqs = np.concatenate([state_cur_sample[None], sample_planning_res["state_seq"]], axis=0)[..., :dt_state_dim]
            open_loop_cost_stat.append([step_cost_fn_np(s, target_pose[:dt_state_dim]) for s in pred_state_seqs])

            summary_dict[i] = {"state_seqs": state_seqs, "pred_state_seqs": pred_state_seqs, "gt_states": gt_states, "act_seqs": act_seqs, "target_pose": target_pose}
        
        plot_planning_animation(state_seqs[None, :, :, :3], dt_dyn.dt, os.path.join(out_dir, f"plan_vis_{i:04d}.gif"), targets=target_pose[None, :3], obs_config=planning_config.get("obstacle", None))
        reach_config = planning_config.get("reach_in_obj", {})
        refine_config = planning_config.get("refinement", {})
        reach_refine_config = refine_config.get("reach_in_obj", {})
        enable_reach = reach_config.get("enable", False) or (refine_config.get('enable', False) and reach_refine_config.get("enable", False))
        if enable_reach:
            # r_lo_seqs, r_up_seqs: (n_sim_steps+1, horizon+1, 3)
            r_lo_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_lo'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :dt_state_dim]
            r_up_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_up'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :dt_state_dim]
            reach_vols = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['reach_vol'] for d in planning_res_list])

            # sample from r_lo_seqs and r_up_seqs: choose corners for each dimension, 2^6 = 64 samples
            sample_state_seqs = box_corners_nd(r_lo_seqs, r_up_seqs)  # (64, n_sim_steps+1, horizon+1, 3)

            plot_planning_animation(state_seqs[None, :, :, :3], dt_dyn.dt, os.path.join(out_dir, f"plan_reach_vis_{i:04d}.gif"), targets=target_pose[None, :3], r_lo_seqs=r_lo_seqs[None, :, :, :3], r_up_seqs=r_up_seqs[None, :, :, :3], obs_config=planning_config.get("obstacle", None))

            if open_loop:
                summary_dict[i]["r_lo_seqs"] = r_lo_seqs
                summary_dict[i]["r_up_seqs"] = r_up_seqs
                summary_dict[i]["sample_state_seqs"] = sample_state_seqs
                summary_dict[i]["reach_vols"] = reach_vols

    cost_stat = np.array(cost_stat)  # (num_test, max_steps)
    cost_stat_file = os.path.join(out_dir, "step_costs.npy")
    np.save(cost_stat_file, cost_stat)
    avg_step_cost = np.mean(cost_stat, axis=0)
    std_step_cost = np.std(cost_stat, axis=0)
    print(f"Average step cost over time over {num_test} test cases: {avg_step_cost}")
    print(f"Std of step cost over time over {num_test} test cases: {std_step_cost}")
    plot_cost_stat(cost_stat, os.path.join(out_dir, "step_costs.png"))

    if open_loop:
        open_loop_cost_stat = np.array(open_loop_cost_stat)
        cost_stat_file = os.path.join(out_dir, "open_loop_step_costs.npy")
        np.save(cost_stat_file, open_loop_cost_stat)
        avg_step_cost = np.mean(open_loop_cost_stat, axis=0)
        std_step_cost = np.std(open_loop_cost_stat, axis=0)
        print(f"Open-loop Average step cost over time over {num_test} test cases: {avg_step_cost}")
        print(f"Open-loop Std of step cost over time over {num_test} test cases: {std_step_cost}")
        plot_cost_stat(open_loop_cost_stat, os.path.join(out_dir, "open_loop_step_costs.png"))

        summary_path = os.path.join(out_dir, "open_loop_summary.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(summary_dict, f)

        for i in range(num_test // num_sample_per_case):
            act_seqs = summary_dict[i * num_sample_per_case]["act_seqs"]
            print(act_seqs)
            state_seqs = summary_dict[i * num_sample_per_case]["state_seqs"][..., :3]
            agg_gt_states = np.array([summary_dict[i * num_sample_per_case + j]["gt_states"] for j in range(num_sample_per_case)])[..., :3]
            agg_pred_state_seqs = np.array([summary_dict[i * num_sample_per_case + j]["pred_state_seqs"] for j in range(num_sample_per_case)])[..., :3]
            target_pose = summary_dict[i * num_sample_per_case]["target_pose"]

            plot_planning_animation(state_seqs[None], dt_dyn.dt, os.path.join(out_dir, f"open_loop_agg_plan_vis_{i:04d}.gif"), targets=target_pose[None, :3], gt_state_seqs=agg_gt_states[:, None], obs_config=planning_config.get("obstacle", None))
            plot_planning_animation(state_seqs[None], dt_dyn.dt, os.path.join(out_dir, f"open_loop_agg_plan_vis_pred_{i:04d}.gif"), targets=target_pose[None, :3], gt_state_seqs=agg_pred_state_seqs[:, None], obs_config=planning_config.get("obstacle", None))
            if enable_reach:
                # r_lo_seqs = summary_dict[i * num_sample_per_case]["r_lo_seqs"]
                # r_up_seqs = summary_dict[i * num_sample_per_case]["r_up_seqs"]
                print(f"reach vols: {summary_dict[i * num_sample_per_case]['reach_vols']}")
                sample_state_seqs = summary_dict[i * num_sample_per_case]["sample_state_seqs"]

                plot_planning_animation(state_seqs[None], dt_dyn.dt, os.path.join(out_dir, f"open_loop_agg_plan_reach_vis_{i:04d}.gif"), targets=target_pose[None, :3], gt_state_seqs=agg_gt_states[:, None], r_lo_seqs=r_lo_seqs[None, :, :, :3], r_up_seqs=r_up_seqs[None, :, :, :3], obs_config=planning_config.get("obstacle", None))


    # copy config file to out_dir
    with open(os.path.join(out_dir, "planning_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))

    return

if __name__ == "__main__":
    main()