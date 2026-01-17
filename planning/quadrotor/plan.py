import os
import numpy as np
import hydra
from omegaconf import DictConfig
import jax
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
from planning.quadrotor.plan_utils import generate_test_cases, make_rollout_and_reward_fns, plot_cost_stat
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories, sample_vel_cmd_sequence


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml")
def main(config: DictConfig):
    task_name = config["settings"]["task_name"]
    data_config = config["data"]
    train_config = config["train_dt_dyn"]
    planning_config = config["planning"]
    verbose = planning_config.get("verbose", False)
    seed = config["settings"]["seed"]
    num_test = planning_config["num_test"]
    test_id = planning_config.get("test_id", 0)
    init_pose_list, target_pose_list = generate_test_cases(seed, num_test, test_id=test_id)

    dt_dyn_dir = config["test_models"]["dt_dyn_dir"]
    dt_dyn: Quad_Dynamics = load_model(model_dir=dt_dyn_dir, model_type="dt_dyn", mode="best", task_name=task_name)

    action_bound = planning_config["action_bound"]
    scale = float(data_config["scale"])
    assert scale == 1
    ct_state_dim, ct_action_dim = data_config["ct_state_dim"], data_config["ct_action_dim"]
    dt_state_dim, dt_action_dim = data_config["dt_state_dim"], data_config["dt_action_dim"]

    action_lower_lim = -action_bound * jnp.ones((dt_action_dim,)) / scale
    action_upper_lim = action_bound * jnp.ones((dt_action_dim,)) / scale
    
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

    noise_type = planning_config.get("disturbance", {}).get("type", "none")
    assert noise_type in {"none", "normal", "uniform"}, f"Unknown disturbance type: {noise_type}"
    noise_init = planning_config.get("disturbance", {}).get("init", 0.0)
    noise_inter = planning_config.get("disturbance", {}).get("inter", 0.0)

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

    key = jax.random.PRNGKey(seed)
    cost_stat = []
    for i in range(num_test):
        init_pose = init_pose_list[i]
        target_pose = target_pose_list[i]
        print(f"Test case {i}:")
        # print(f"  Init pusher pos: {init_pusher_pos}")
        # print(f"  Init pose: {init_pose}")
        # print(f"  Target pusher pos: {target_pusher_pos}")
        # print(f"  Target pose: {target_pose}")

        target_state = target_pose[:dt_state_dim]
        scaled_target_state = target_state / scale
        scaled_target_state = jnp.array(scaled_target_state)
        env = Quad_Sim_Ctl(data_config, num_quads=1, init_poses=init_pose[None], target_poses=target_pose[None], controller=ct_ctl)
        env_dict = env.get_env_states()
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
            key = jax.random.PRNGKey(seed + t)

            key, subkey = jax.random.split(key)
            noise_param = noise_init
            if noise_type == "normal":
                noise = jax.random.normal(subkey, shape=(ct_state_dim,)) * noise_param
            elif noise_type == "uniform":
                noise = jax.random.uniform(subkey, shape=(ct_state_dim,), minval=-1.0, maxval=1.0) * noise_param
            else:
                noise = jnp.zeros((ct_state_dim,))
            # state_cur = state_cur.at[0:ct_state_dim].add(noise)
            # env.force_update([[noise[0] * scale, noise[1] * scale, noise[2]]])  # apply disturbance

            key, subkey = jax.random.split(key)
            init_act_seq = 0 * jax.random.uniform(subkey,(horizon, dt_action_dim),minval=action_lower_lim,maxval=action_upper_lim,)
            key, subkey = jax.random.split(key)
            # with jax.disable_jit():
            #     planning_res = eqx.filter_jit(planner.trajectory_optimization)(key, state_cur, init_act_seq, skip=False, target_state=scaled_target_state)
            planning_res = eqx.filter_jit(planner.trajectory_optimization)(subkey, state_cur, init_act_seq, skip=False, target_state=scaled_target_state)
            scaled_act_seq = planning_res["act_seq"]
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
                for ctl_step in range(n_ctl_per_dyn):
                    next_action = v_cmd[None]
                    if verbose:
                        print(f"   controller action: {next_action}")

                    # with jax.disable_jit():
                    #     env_dict = env.update(next_action, n_sim_time=1/ctl_frequency)
                    env_dict = env.update(next_action, n_sim_time=1/ctl_frequency)

                    # state_cur = jnp.array(env_dict["state"]).squeeze()[:dt_state_dim] / scale
                    env_state = np.concatenate([env_dict["state"], next_action, env_dict["action"]], axis=-1).squeeze()
                    
                    key, subkey = jax.random.split(key)
                    noise_param = noise_inter
                    if noise_type == "normal":
                        noise = jax.random.normal(subkey, shape=(ct_state_dim,)) * noise_param
                    elif noise_type == "uniform":
                        noise = jax.random.uniform(subkey, shape=(ct_state_dim,), minval=-1.0, maxval=1.0) * noise_param
                    else:
                        noise = jnp.zeros((ct_state_dim,))
                    # state_cur = state_cur + noise

                    env.force_update([noise * scale])  # apply disturbance

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
        plot_3d_trajectories(X_gt[:, :, :3], num_quads=1, dt=ct_ctl.dt, out_path=os.path.join(out_dir, f"gt_trajectories_{i}.png"))
        plot_quad_states_actions(X_gt[:, 0, :dt_state_dim], v_cmds[:, 0], dt=ct_ctl.dt, out_path=os.path.join(out_dir, f"pred_states_vcmd_{i}.png"))
        plot_quad_states_actions(X_gt[:, 0, :ct_state_dim], U_gt[:, 0], dt=ct_ctl.dt, out_path=os.path.join(out_dir, f"gt_states_actions_{i}.png"))
        env.close()

    cost_stat = np.array(cost_stat)  # (num_test, max_steps)
    avg_step_cost = np.mean(cost_stat, axis=0)
    print(f"Average step cost over time over {num_test} test cases: {avg_step_cost}")
    plot_cost_stat(cost_stat, out_dir)
    return

if __name__ == "__main__":
    main()