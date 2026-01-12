import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import List, Dict
import hydra
from omegaconf import DictConfig
import os
from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics
from models.quadrotor.ct_ctl import PID_Controller, MLP_Controller
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories


class Quad_Sim:
    def __init__(self, data_config: dict, init_poses=None, target_poses=None):
        self.model = Continuous_Quad_Dynamics(data_config)
        self.Dx = self.model.Dx
        self.Du = self.model.Du
        self.frequency = data_config.get("ct_frequency", 200)
        
        # JIT the batch forward for the whole fleet
        self.forward_batch = eqx.filter_jit(jax.vmap(self.model.forward_batchless))
        
        self.num_quads = data_config.get("num_quads", 1)
        self.curr_states = init_poses  # (num_quads, Dx)
        self.SAVE_IMG = data_config.get("gif", False)
        self.reset(init_poses, target_poses)

    def reset(self, init_poses=None, target_poses=None):
        if init_poses is None:
            init_poses = jnp.zeros((self.num_quads, self.Dx))
        self.curr_states = init_poses
        if target_poses is not None:
            target_poses = jnp.array(target_poses)
        self.target_poses = target_poses
        self.history = []

    def update(self, actions: jnp.ndarray, n_sim_time=1/60) -> Dict[str, jnp.ndarray]:
        """
        updates states of all quads. 
        actions: (num_quads, 3)
        returns: dict with 'action' and 'state'
        """
        n_sim_steps = int(self.frequency * n_sim_time)

        # Step dynamics forward
        # forward_batch integrates from t=0 to t=self.dt (typically 0.1s for this benchmark)
        for _ in range(n_sim_steps):
            next_states = self.forward_batch(self.curr_states, actions)
            self.curr_states = next_states

            # Log to history
            step_data = {
                'state': next_states,
                'action': actions
            }
            self.history.append(step_data)
            
        return step_data

    def visualize(self, out_dir, fps=30):
        if not self.SAVE_IMG or not self.history:
            return
        
        import os
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 1. plot 3D Trajectory Overview
        pose_seqs = np.array([h['state'] for h in self.history])[:, :, :3]  # (T, num_quads, 3)
        out_path = os.path.join(out_dir, "trajectories_3d.png")
        plot_3d_trajectories(pose_seqs, self.num_quads, dt=1/self.frequency, out_path=out_path)
        
        # 2.plot individual quad telemetry
        for i in range(self.num_quads):
            state_seq = np.array([h['state'][i] for h in self.history])  # (T, 12)
            action_seq = np.array([h['action'][i] for h in self.history])  # (T, 3)
            out_path = os.path.join(out_dir, f"quad_{i}_telemetry.png")
            plot_quad_states_actions(state_seq, action_seq, dt=1/self.frequency, out_path=out_path)
            
        print(f"Visualization saved to {out_dir}")

    def close(self):
        pass

class Quad_Sim_DT:
    def __init__(self, data_config: dict, init_poses=None, target_poses=None):
        self.Dx_dt = data_config.get("dt_state_dim", 6)
        self.Du_dt = data_config.get("dt_action_dim", 3)
        assert self.Dx_dt == 6
        assert self.Du_dt == 3
        self.ct_sim = Quad_Sim(data_config, init_poses, target_poses)
        self.num_quads = self.ct_sim.num_quads
        self.frequency = data_config.get("dt_frequency", 5)
        self.ct_ctl = PID_Controller(data_config)
        self.ctl_frequency = data_config.get("ctl_frequency", 20)
        self.reset(init_poses, target_poses)
        self.control_step_fn = eqx.filter_jit(self.ct_ctl)

    def reset(self, init_poses=None, target_poses=None):
        if init_poses is None:
            init_poses = jnp.zeros((self.num_quads, self.Dx_dt))
        self.curr_states = init_poses
        if target_poses is not None:
            target_poses = jnp.array(target_poses)
        self.target_poses = target_poses
        ct_init_poses = jnp.concatenate([init_poses, jnp.zeros((self.num_quads, self.ct_sim.Dx - self.Dx_dt))], axis=-1)
        ct_target_poses = None if target_poses is None else jnp.concatenate([target_poses, jnp.zeros((self.num_quads, self.ct_sim.Dx - self.Dx_dt))], axis=-1)
        self.history = []
        self.ct_sim.reset(ct_init_poses, ct_target_poses)

    def update(self, v_cmds: jnp.ndarray, n_sim_time=1/5) -> Dict[str, jnp.ndarray]:
        """
        updates states of all quads. 
        v_cmds: (num_quads, 3) velocity commands
        returns: dict with 'action' and 'state'
        """
        n_sim_steps = int(self.frequency * n_sim_time)
        n_ctl_per_sim = int(self.ctl_frequency // self.frequency)
        ctl_time_per_step = 1 / self.ctl_frequency

        for step in range(n_sim_steps):
            for i in range(n_ctl_per_sim):
                actions = jax.vmap(self.control_step_fn)(self.ct_sim.curr_states, v_cmds)
                step_data = self.ct_sim.update(actions, n_sim_time=ctl_time_per_step)

        # Log to history
        step_data = {
            'state': step_data['state'][:, :self.Dx_dt],
            'action': v_cmds
        }
        self.history.append(step_data)

        return step_data

    def visualize(self, out_dir, fps=30):
        if not self.ct_sim.SAVE_IMG or not self.history:
            return

        # 1. plot 3D Trajectory Overview
        pose_seqs = np.array([h['state'] for h in self.history])[:, :, :3]  # (T, num_quads, 3)
        out_path = os.path.join(out_dir, "dt_trajectories_3d.png")
        plot_3d_trajectories(pose_seqs, self.num_quads, dt=1/self.frequency, out_path=out_path)
        
        # 2.plot individual quad telemetry
        for i in range(self.num_quads):
            state_seq = np.array([h['state'][i] for h in self.history])  # (T, 6)
            action_seq = np.array([h['action'][i] for h in self.history])  # (T, 3)
            out_path = os.path.join(out_dir, f"dt_quad_{i}_telemetry.png")
            plot_quad_states_actions(state_seq, action_seq, dt=1/self.frequency, out_path=out_path)
            
        print(f"Visualization saved to {out_dir}")

    def close(self):
        self.ct_sim.close()
        pass

@hydra.main(config_path=os.path.join(os.getcwd(), "configs"), config_name="quadrotor.yaml", version_base=None)
def main(config: DictConfig) -> None:
    data_config = config["data"]
    data_mode = config["settings"].get("data_mode", "dt_dyn")
    # data_config["frequency"] is the ode frequency limit
    frequency = min(data_config[data_mode]["frequency"], data_config["ct_frequency"])

    episode_length = data_config[data_mode]["episode_length"] * frequency

    # env = Quad_Sim(data_config=data_config)

    # const_action = jnp.array([9.81, 0.1, 0.1])[None].repeat(env.num_quads, axis=0)

    # for step in range(episode_length):
    #     env.update(const_action, n_sim_time=1/frequency)

    # out_dir = "output/quad_sim_test"
    # os.makedirs(out_dir, exist_ok=True)
    # env.visualize(out_dir=out_dir)

    # env.close()

    env_dt = Quad_Sim_DT(data_config=data_config)

    const_action = jnp.array([1.0, 1.0, 1.0])[None].repeat(env_dt.num_quads, axis=0)

    for step in range(episode_length):
        env_dt.update(const_action, n_sim_time=1/env_dt.frequency)
    out_dir = "output/quad_sim_dt_test"
    os.makedirs(out_dir, exist_ok=True)
    env_dt.visualize(out_dir=out_dir)

    env_dt.close()

if __name__ == "__main__":
    main()