import time
import numpy as np
import random
import pickle
import os
import math
from multiprocessing import Pool
import hydra
from omegaconf import DictConfig

from envs.T_pushing.helper import rand_float, get_truncated_normal, gen_act
import envs.T_pushing.t_sim as t_sim
from tqdm import tqdm
"""
output format: (state, pusher position, pusher velocity)
state: relative position of keypoints of object to pusher: x1-xp, y1-yp, x2-xp, y2-yp, ...
pusher position: xp, yp
pusher velocity: vx, vy
data: (x1-xp, y1-yp, x2-xp, y2-yp, x3-xp, y3-yp, x4-xp, y4-yp, xp, yp, vx, vy)
"""


def gen_data(config, process_id, seed, num_episode):
    random.seed(seed)
    np.random.seed(seed)
    data_config = config["data"]
    scale, window_size, episode_length, state_dim, action_dim = (
        data_config["scale"],
        data_config["window_size"],
        data_config["episode_length"],
        data_config["state_dim"],
        data_config["action_dim"],
    )
    episode_shift = data_config.get("episode_shift", 0)
    episode_length += episode_shift
    training, visualizing, saving, gif = (
        data_config["training"],
        data_config["visualizing"],
        data_config["saving"],
        data_config["gif"],
    )
    param_dict = {"stem_size": data_config["stem_size"], 
                  "bar_size": data_config["bar_size"], 
                  "pusher_size": data_config["pusher_size"],
                  "save_img": gif,
                  "enable_vis": visualizing,
                  "window_size": window_size,}

    frequency = data_config["frequency"]
    n_sim_step = round(60 / frequency)

    num_transitions = random.randint(0, 6)
    transit_preiod = episode_length // (num_transitions + 1)
    lo, hi = window_size * 0.1, window_size * 0.9
    pose_lo, pose_hi = window_size * 0.4, window_size * 0.6
    dx, dy = 0, 0
    inverse_factor = 15
    inverse = random.randint(0, inverse_factor) == 1
    scale_lb = 0.8
    scale_ub = 7
    scale1 = rand_float(scale_lb, scale_ub)
    scale2 = rand_float(scale_lb, scale_ub)
    # print(f"scale1: {scale1}, scale2: {scale2}")
    noise1 = get_truncated_normal(mean=0, sd=10, low=-20, upp=20)
    noise2 = get_truncated_normal(mean=0, sd=4, low=-10, upp=10)
    action_bound = data_config["action_bound"] * 1.05

    # box_range = action_bound * 4
    box_range = 100
    save_dir = os.path.join(data_config["out_path"], "vis")

    sim = t_sim.T_Sim(param_dict=param_dict)
    dataset = []
    for j in tqdm(range(num_episode)):
        # init
        episode = []
        dx = dy = 0
        # generate initial state
        init_poses = [rand_float(pose_lo, pose_hi), rand_float(pose_lo, pose_hi), rand_float(0, 2 * math.pi)]
        init_poses = [init_poses]
        sim.refresh(init_poses)
        x_obj, y_obj = sim.get_all_object_positions()[random.randint(0, sim.obj_num - 1)]

        x_pusher = rand_float(max(x_obj - box_range, lo), min(x_obj + box_range, hi))
        y_pusher = rand_float(max(y_obj - box_range, lo), min(y_obj + box_range, hi))
        # allow the simulator to a resting position
        for i in range(2):
            env_dict = sim.update((x_pusher, y_pusher), n_sim_step=n_sim_step)
        # add the initial state to the episode
        env_state = np.concatenate([env_dict["state"], env_dict["pusher_pos"], env_dict["action"]], axis=0)
        episode.append(env_state)

        # simulate an episode
        for i in range(episode_length):
            centers = sim.get_all_object_positions()
            keypoints = sim.get_all_object_keypoints()
            keypoints = [keypoint for object in keypoints for keypoint in object]

            keypoints.extend(centers)
            if i % transit_preiod == 0:
                index = random.randint(0, len(keypoints) - 1)
                mode = random.randint(0, 2) % 2
                # mode = 0
                if not training:
                    mode = 1
                # mode = random.randint(0, 1)
                # mode = 1
                inverse = random.randint(0, inverse_factor) == 1
                scale1 = rand_float(scale_lb, scale_ub)
                scale2 = rand_float(scale_lb, scale_ub)
            x_obj, y_obj = keypoints[index]
            x_pusher, y_pusher = sim.get_pusher_position()
            beta1 = rand_float(0.4, 0.8)
            beta2 = rand_float(0.4, 0.8)
            # mode = 2
            if mode == 0:
                ddx = gen_act(x_obj - x_pusher + noise1.rvs(), scale1, action_bound)
                ddy = gen_act(y_obj - y_pusher + noise1.rvs(), scale2, action_bound)
            elif mode == 1:
                ddx = (x_obj - x_pusher) * rand_float(1.5, 2.5) + rand_float(-1, 1) * (y_obj - y_pusher)
                ddy = (y_obj - y_pusher) * rand_float(1.5, 2.5) + rand_float(-1, 1) * (x_obj - x_pusher)
            else:
                ddx = random.randint(0, action_bound) * np.sign(x_obj - x_pusher)
                ddy = random.randint(0, action_bound) * np.sign(y_obj - y_pusher)
            ddx = np.clip(ddx, -1.5 * action_bound, 1.5 * action_bound)
            ddy = np.clip(ddy, -1.5 * action_bound, 1.5 * action_bound)
            dx = beta1 * dx + (1 - beta1) * ddx + noise2.rvs()
            dy = beta2 * dy + (1 - beta2) * ddy + noise2.rvs()
            if inverse:
                dx = -dx
                dy = -dy
            # dx *= action_rescale 
            # dy *= action_rescale
            dx = np.clip(dx, -action_bound, action_bound)
            dy = np.clip(dy, -action_bound, action_bound)
            x_pusher = np.clip(x_pusher + dx, lo, hi)
            y_pusher = np.clip(y_pusher + dy, lo, hi)
            env_dict = sim.update((x_pusher, y_pusher), n_sim_step=n_sim_step)
            # TODO: unify the API
            env_state = (
                np.concatenate([env_dict["state"], env_dict["pusher_pos"], env_dict["action"]], axis=0)
            )
            episode[-1][-action_dim:] = env_state[-action_dim:]
            episode.append(env_state)
            if visualizing and sim.SAVE_IMG == False:
                time.sleep(0.03)

        episode = np.array(episode[episode_shift:-1])
        dataset.append(episode)
        # import pdb; pdb.set_trace()
        if gif:  # and j == 0
            sim.save_gif(os.path.join(save_dir, f"demo_{process_id}{j}.gif"))
            # sim.SAVE_IMG = False
            # sim.save_mp4(os.path.join(save_dir, f"demo_{round(scale1,2)}_{round(scale2,2)}_{j}.mp4"))

    dataset = np.array(dataset)
    print([round(np.percentile(dataset[:, :, -2], 10 * i), 5) for i in range(11)])
    print([round(np.percentile(dataset[:, :, -1], 10 * i), 5) for i in range(11)])
    # Update file naming to include process_id
    data_dir = data_config["out_path"]
    filename = os.path.join(
        data_dir, f"data{'' if training else '_eval'}_tmp_{process_id}.p"
    )
    if saving:
        with open(filename, "wb") as fp:
            pickle.dump(dataset, fp)
        print(f"save {filename}")
    if visualizing:
        sim.close()


def parallel_gen_data(args):
    # Unpack arguments if you passed a tuple or dictionary
    config, process_id, seed, num_episode = args
    # Call the original gen_data function
    start_time = time.time()
    gen_data(config, process_id, seed, num_episode)
    print(f"Process {process_id} finished in {time.time() - start_time} seconds")

@hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing.yaml", version_base=None)
def main(config: DictConfig) -> None:
    data_config = config["data"]
    frequency = data_config["frequency"]
    if frequency != 60:
        data_config["episode_length"] = data_config["episode_length"] * frequency
        data_config["num_episodes"] = data_config["num_episodes"] // frequency

    num_episodes, training, visualizing, saving, gif = (
        data_config["num_episodes"],
        data_config["training"],
        data_config["visualizing"],
        data_config["saving"],
        data_config["gif"],
    )

    print(
        f"num_episodes: {num_episodes}, training: {training}, visualizing: {visualizing}, saving: {saving}, gif: {gif}"
    )
    base_seed = config["settings"]["seed"]
    if not training:
        base_seed = config["settings"]["seed"] + 100
    os.makedirs(data_config["out_path"], exist_ok=True)
    data_dir = data_config["out_path"]
    os.makedirs(data_dir, exist_ok=True)
    if gif:
        # import pdb; pdb.set_trace()
        os.makedirs(os.path.join(data_dir, "vis"), exist_ok=True)
    final_filename = os.path.join(data_dir, f"data{'' if training else '_eval'}.p")

    # Setup multiprocessing pool
    num_processes = min(data_config["num_workers"], os.cpu_count() - 2)

    print(f"num_processes: {num_processes}")
    if num_processes == 1:
        # If only one process, just call the function directly
        gen_data(config, 0, base_seed, num_episodes)
    else:
        pool = Pool(processes=num_processes)
        process_seeds = [base_seed + i for i in range(num_processes)]
        # Prepare arguments for each process, including the new process_id
        process_args = [(config, i, process_seeds[i], num_episodes // num_processes) for i in range(num_processes)]

        # Start parallel data generation
        pool.map(parallel_gen_data, process_args)

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()
    print("Finished generating data")
    # Combine data from separate files
    if saving:
        combined_data = []
        for i in range(num_processes):
            filename = os.path.join(data_dir, f"data{'' if training else '_eval'}_tmp_{i}.p")
            with open(filename, "rb") as fp:
                combined_data.extend(pickle.load(fp))
                os.remove(filename)
        if os.path.exists(final_filename):
            os.remove(final_filename)
            print(f"remove existing {final_filename}")
        # Save combined data
        with open(final_filename, "wb") as fp:
            pickle.dump(combined_data, fp)
        print(f"save {final_filename}")


if __name__ == "__main__":
    main()
