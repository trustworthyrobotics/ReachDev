import pickle
import os, sys
import numpy as np
from torch.utils.data import Dataset
sys.path.append(os.getcwd())
from envs.T_pushing.helper import rotate_state, reset_seed

class DynamicsDataset(Dataset):
    def __init__(self, config, phase):
        self.config = config
        # dynamics data expected to be a single pickle file
        # should be a list of numpy matrices
        # each matrix represents an episode in step order
        # and is of shape (# steps, state_dim + action_dim) representing (s, a)
        data_file_name = config["train"]["data_path"]

        with open(data_file_name, "rb") as fp:
            data_load = pickle.load(fp)

        data_config = config["data"]
        train_config = config["train"]
        state_dim = data_config["state_dim"]
        action_dim = data_config["action_dim"]
        self.phase = phase
        n_his = train_config["n_history"]
        n_roll = train_config["n_rollout"]
        if phase == "valid":
            n_roll = (
                train_config["n_rollout_valid"]
            )
        n_sample = n_his + n_roll

        num_train = int(len(data_load) * train_config["train_valid_ratio"])
        if phase == "train":
            data_load = data_load[:num_train]
        elif phase == "valid":
            data_load = data_load[num_train:]
        else:
            raise AssertionError("Unknown phase %s" % phase)
        n_roll = min(data_load[0].shape[0]-n_his, n_roll)
        n_sample = n_his + n_roll
        self.n_roll = n_roll
        self.n_sample = n_sample
        self.n_his = n_his
        self.episode_length = len(data_load[0]) #  - 1
        self.obs = []
        self.act = []
        self.weights = []
        # import pdb; pdb.set_trace()
        for ep in data_load:
            for i in range(len(ep) - n_sample + 1):
                self.obs.append(ep[i : i + n_sample, :-action_dim])
                self.act.append(ep[i : i + n_sample, -action_dim:])
        self.obs = np.array(self.obs)
        self.act = np.array(self.act)

        self.weights = np.ones((self.obs.shape[0], n_roll))
        print('weights percentile', [round(np.percentile(self.weights, 5*i),5) for i in range(21)])

        print(f"state shape {self.obs.shape}")
        # print percentile for every dim of state
        # for j in range(state_dim):
        #     print(f"state dim {j} percentile", [round(np.percentile(self.obs[:, :, j], 5 * i), 5) for i in range(21)])

        print("x vel percentile", [round(np.percentile(self.act[:, :, -2], 5 * i), 5) for i in range(21)])
        print("y vel percentile", [round(np.percentile(self.act[:, :, -1], 5 * i), 5) for i in range(21)])

        pusher_pos_idx = state_dim
        self.pusher_pos = self.obs[:, :, pusher_pos_idx : pusher_pos_idx + 2]
        self.obs = self.obs[:, :, :state_dim]
        # add noise
        if phase == "train":
            eps = train_config["noise"]
            np.random.seed(config["seed"])
            noise = np.random.normal(0, eps, size=self.obs.shape)
            self.obs = self.obs + noise

        # shuffle together
        if phase == "train" and data_config["augment"]:
            self.augment()
        else:
            self.shuffle()  

    def augment(self):
        seed = np.random.randint(0, 100000)
        self.obs = rotate_state(self.obs, seed)
        self.act = rotate_state(self.act, seed)
        self.pusher_pos = rotate_state(self.pusher_pos, seed)
        reset_seed(self.config["seed"])
        self.shuffle()

    def shuffle(self):
        # np.random.seed(self.config["seed"])
        idx = np.random.permutation(range(len(self.obs)))
        # np.random.seed(self.config["seed"])
        self.obs = self.obs[idx]
        self.act = self.act[idx]
        self.weights = self.weights[idx]
        self.pusher_pos = self.pusher_pos[idx]

    # only called when data.enable_hnm is True
    def update_weights(self, indices, new_weight):
        if len(indices.shape) == 1:
            self.weights[indices] *= new_weight.reshape(len(new_weight), 1)
            self.weights[indices] = np.clip(self.weights[indices], 1, self.config["data"]["weight_ub"])
        elif len(indices.shape) == 2:
            self.weights[indices[:, 0], indices[:, 1]] *= new_weight
            self.weights[indices[:, 0], indices[:, 1]] = np.clip(
                self.weights[indices[:, 0], indices[:, 1], 0], 1, self.config["data"]["weight_ub"]
            )
        else:
            raise AssertionError("Unknown indices shape")
        print("weights percentile", [round(np.percentile(self.weights, 5 * i), 5) for i in range(21)])
        self.shuffle()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            "observations": self.obs[idx],
            "actions": self.act[idx],
            "weights": self.weights[idx],
            "pusher_pos": self.pusher_pos[idx],
        }


if __name__ == "__main__":
    pass
