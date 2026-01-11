import random
import math
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import sys
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DT_Plan_Reach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2

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

def make_rollout_and_reward_fns(
    dt_dyn,
    planning_config: Dict,
    abs_pose: bool = True,
    pred_mode: str = "pose",
    reach_config: Dict = {},
):
    horizon = planning_config["horizon"]
    cost_norm = planning_config["cost_norm"]
    only_final_cost = planning_config["only_final_cost"]
    enable_reach = planning_config.get("reach_in_obj", False) or planning_config.get("refinement", {}).get("reach_in_obj", False)

    if enable_reach:
        # reachability part
        def f_wrapper(x):
            state_next = dt_dyn(x)
            action_next = x[dt_dyn.Dx:]
            return jnp.concatenate([state_next, action_next], axis=-1)


        reach_analyzer = DT_Plan_Reach(f_wrapper, state_dim=dt_dyn.Dx, action_dim=dt_dyn.Du, nn_dyn=True, n_steps_per_plan=1, step_size=1)
        eps = reach_config.get("eps", 0.05)
        splits_cfg = reach_config.get("init_splits", {})

        _calculate_volume = lambda lo, up: calculate_volume(lo, up, union_init=False, mode='sum')


    # [state_dim], [n_sample, horizon, action_dim] -> [n_sample, horizon, state_dim]
    def rollout_fn(state_cur: jnp.ndarray, act_seqs: jnp.ndarray, use_reach: bool) -> jnp.ndarray:
        state_cur = state_cur[None].repeat(act_seqs.shape[0], axis=0) # [B, Dx]
        state_seqs = dt_dyn.rollout(state_cur, act_seqs)
        if use_reach:
            reach_info = cal_reach(state_cur, act_seqs)
        return state_seqs, reach_info if use_reach else {}

    def cal_reach(state_cur: jnp.ndarray, act_seqs: jnp.ndarray) -> jnp.ndarray:
        B = act_seqs.shape[0]
        Da = dt_dyn.Du
        T = act_seqs.shape[1]
        state_lo = state_cur[:, :dt_dyn.Ds] - eps
        state_up = state_cur[:, :dt_dyn.Ds] + eps

        if abs_pose:
            state_lo = jnp.concatenate([state_lo, state_cur[:, dt_dyn.Ds:dt_dyn.Dx]], axis=-1)
            state_up = jnp.concatenate([state_up, state_cur[:, dt_dyn.Ds:dt_dyn.Dx]], axis=-1)

        X_lo = jnp.concatenate([state_lo, jnp.zeros((B, Da))], axis=-1)
        X_up = jnp.concatenate([state_up, jnp.zeros((B, Da))], axis=-1)
        X_lo, X_up = prepare_initial_set_v2(X_lo, X_up, splits_cfg=splits_cfg)
        B_reach = X_lo.shape[0]

        _, r_lo, r_up, _, _ = reach_analyzer.verify(X_lo, X_up, n_total_steps=T, action_seq=act_seqs[:, None])
        r_lo = r_lo.reshape(B, B_reach//B, T + 1, -1)
        r_up = r_up.reshape(B, B_reach//B, T + 1, -1)

        reach_vol = jax.vmap(_calculate_volume)(r_lo, r_up)  # [B]
        return {"reach_vol": reach_vol, "r_lo": r_lo, "r_up": r_up}

    # assume all are scaled
    def reward_fn(state_seqs: jnp.ndarray, act_seqs: jnp.ndarray, use_reach: bool, aux: Dict, target_state: jnp.ndarray, pusher_pos: jnp.ndarray) -> Dict:
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
        if use_reach:
            reach_loss = jnp.log1p(aux["reach_vol"])
            costs = costs + reach_loss  # penalize large reachable set
        return {"rewards": -costs, "reward_seqs": -cost_seqs}

    def step_cost_fn_np(state, target_state):
        return (np.linalg.norm(target_state - state, cost_norm)) ** cost_norm

    def step_cost_fn(state, target_state):
        return (jnp.linalg.norm(target_state - state, cost_norm)) ** cost_norm

    return rollout_fn, reward_fn, step_cost_fn, step_cost_fn_np