import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import List, Dict, Optional
import hydra
from omegaconf import DictConfig
import os
from models.quadrotor.ct_dyn import Continuous_Quad_Dynamics
from models.quadrotor.ct_ctl import PID_Controller, MLP_Controller
from envs.quadrotor.helper import plot_quad_states_actions, plot_3d_trajectories, sample_vel_cmd_sequence


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

    def add_history(self, step_data: Dict[str, jnp.ndarray]):
        if self.SAVE_IMG:
            self.history.append(step_data)
        return

    def update(self, actions: jnp.ndarray, n_sim_time=None) -> Dict[str, jnp.ndarray]:
        """
        updates states of all quads. 
        actions: (num_quads, 3)
        returns: dict with 'action' and 'state'
        """
        if n_sim_time is None:
            n_sim_time = 1 / self.frequency
        n_sim_steps = max(1, int(self.frequency * n_sim_time))

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
            self.add_history(step_data)
            
        return step_data

    def visualize(self, out_dir, fps=30):
        if not self.SAVE_IMG or not self.history:
            return
        
        import os
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 1. plot 3D Trajectory Overview
        pose_seqs = np.array([h['state'] for h in self.history])[:, :, :3]  # (T, num_quads, 3)
        out_path = os.path.join(out_dir, "ct_trajectories_3d.png")
        plot_3d_trajectories(pose_seqs, self.num_quads, dt=1/self.frequency, out_path=out_path)
        
        # 2.plot individual quad telemetry
        for i in range(self.num_quads):
            state_seq = np.array([h['state'][i] for h in self.history])  # (T, 12)
            action_seq = np.array([h['action'][i] for h in self.history])  # (T, 3)
            out_path = os.path.join(out_dir, f"ct_quad_{i}_telemetry.png")
            plot_quad_states_actions(state_seq, action_seq, dt=1/self.frequency, out_path=out_path)
            
        print(f"Visualization saved to {out_dir}")

    def close(self):
        self.history = []

class Quad_Sim_Ctl:
    """CT sim + controller wrapper.
    - State: 12D CT state (same as Quad_Sim)
    - Action: v_cmd (3D) at ctl_frequency
    Internally: v_cmd -> low-level u (thrust/torques) -> step CT sim for ctl_dt.
    """
    def __init__(self, data_config: dict, init_poses=None, target_poses=None, controller=None):
        self.ct_sim = Quad_Sim(data_config, init_poses, target_poses)
        self.num_quads = self.ct_sim.num_quads

        self.ctl_frequency = data_config.get("ctl_frequency", 20)
        self.ctl_dt = 1.0 / self.ctl_frequency

        # controller maps: (x12, v_cmd3) -> u_low (Du=3 or 4)
        self.controller = PID_Controller(data_config) if controller is None else controller

        # jit + vmap for fleet
        self._ctl_fn = eqx.filter_jit(self.controller)
        self._ctl_batch = eqx.filter_jit(jax.vmap(self._ctl_fn))

        self.reset(init_poses, target_poses)

    @property
    def curr_states(self):
        return self.ct_sim.curr_states

    @property
    def SAVE_IMG(self):
        return self.ct_sim.SAVE_IMG


    def reset(self, init_poses=None, target_poses=None):
        self.ct_sim.reset(init_poses, target_poses)
        self.history = []

    def add_history(self, step_data: Dict[str, jnp.ndarray]):
        if self.SAVE_IMG:
            self.history.append(step_data)
        return

    def update(self, v_cmds: jnp.ndarray, n_sim_time: Optional[float] = None) -> Dict[str, jnp.ndarray]:
        """
        v_cmds: (num_quads, 3)
        Advances the CT sim for n_sim_time seconds using control ticks at ctl_frequency.
        Returns last step_data with CT state, v_cmd, and low-level action.
        """
        if n_sim_time is None:
            n_sim_time = self.ctl_dt

        n_ctl_steps = max(1, int(round(self.ctl_frequency * n_sim_time)))

        step_data = None
        for _ in range(n_ctl_steps):
            # 1) compute low-level action from current CT state and v_cmd
            u_low = self._ctl_batch(self.ct_sim.curr_states, v_cmds)  # (num_quads, Du)

            # 2) advance CT sim for one controller tick
            ct_step = self.ct_sim.update(u_low, n_sim_time=self.ctl_dt)

            # 3) log
            step_data = {
                "state": ct_step["state"],     # (num_quads, 12)
                "action": v_cmds,               # (num_quads, 3)
            }
            self.add_history(step_data)

        return step_data

    def visualize(self, out_dir, fps=30):
        if not self.SAVE_IMG or not self.history:
            return

        # 1. plot 3D Trajectory Overview
        pose_seqs = np.array([h['state'] for h in self.history])[:, :, :3]  # (T, num_quads, 3)
        out_path = os.path.join(out_dir, "ctl_trajectories_3d.png")
        plot_3d_trajectories(pose_seqs, self.num_quads, dt=self.ctl_dt, out_path=out_path)
        
        # 2.plot individual quad telemetry
        for i in range(self.num_quads):
            state_seq = np.array([h['state'][i] for h in self.history])  # (T, 12)
            action_seq = np.array([h['action'][i] for h in self.history])  # (T, 3)
            out_path = os.path.join(out_dir, f"ctl_quad_{i}_telemetry.png")
            plot_quad_states_actions(state_seq, action_seq, dt=self.ctl_dt, out_path=out_path)
            
        self.ct_sim.visualize(out_dir=out_dir, fps=fps)

    def close(self):
        self.ct_sim.close()
        self.history = []

class Quad_Sim_DT:
    """DT wrapper around Quad_Sim_Ctl.
    - Exposed state: 6D [pos(3), vel(3)] (slice of CT state)
    - Action: v_cmd (3D) at dt_frequency
    """
    def __init__(self, data_config: dict, init_poses=None, target_poses=None, controller=None):
        self.Dx_dt = int(data_config.get("dt_state_dim", 6))
        self.Du_dt = int(data_config.get("dt_action_dim", 3))
        assert self.Dx_dt == 6 and self.Du_dt == 3

        self.frequency = data_config.get("dt_frequency", 5)
        self.dt = 1.0 / self.frequency

        # use the CT-with-controller sim
        self.ct_ctl_sim = Quad_Sim_Ctl(data_config, init_poses, target_poses, controller=controller)
        self.num_quads = self.ct_ctl_sim.num_quads

        self.reset(init_poses, target_poses)

    @property
    def SAVE_IMG(self):
        return self.ct_ctl_sim.SAVE_IMG

    def reset(self, init_poses=None, target_poses=None):
        if init_poses is None:
            init_poses = jnp.zeros((self.num_quads, self.Dx_dt))
        self.curr_states = init_poses

        if target_poses is not None:
            target_poses = jnp.asarray(target_poses)

        # lift DT init -> CT init by padding remaining dims with zeros
        ct_Dx = self.ct_ctl_sim.ct_sim.Dx  # 12
        ct_init = jnp.concatenate([init_poses, jnp.zeros((self.num_quads, ct_Dx - self.Dx_dt))], axis=-1)
        ct_target = None if target_poses is None else jnp.concatenate(
            [target_poses, jnp.zeros((self.num_quads, ct_Dx - self.Dx_dt))], axis=-1
        )

        self.ct_ctl_sim.reset(ct_init, ct_target)
        self.history = []

    def add_history(self, step_data: Dict[str, jnp.ndarray]):
        if self.SAVE_IMG:
            self.history.append(step_data)
        return

    def update(self, v_cmds: jnp.ndarray, n_sim_time: Optional[float] = None) -> Dict[str, jnp.ndarray]:
        """
        v_cmds: (num_quads, 3)
        Advances by n_sim_time seconds at DT rate (default: 1/dt_frequency).
        """
        if n_sim_time is None:
            n_sim_time = self.dt

        n_dt_steps = max(1, int(round(self.frequency * n_sim_time)))

        step_data = None
        for _ in range(n_dt_steps):
            # advance CT-with-controller for one DT tick
            ct_step = self.ct_ctl_sim.update(v_cmds, n_sim_time=self.dt)

            # expose only DT state = first 6 dims of CT state
            self.curr_states = ct_step["state"][:, :self.Dx_dt]

            step_data = {
                "state": self.curr_states,  # (num_quads, 6)
                "action": v_cmds,           # (num_quads, 3)
            }
            self.add_history(step_data)

        return step_data

    def visualize(self, out_dir, fps=30):
        if not self.SAVE_IMG or not self.history:
            return

        # 1. plot 3D Trajectory Overview
        pose_seqs = np.array([h['state'] for h in self.history])[:, :, :3]  # (T, num_quads, 3)
        out_path = os.path.join(out_dir, "dt_trajectories_3d.png")
        plot_3d_trajectories(pose_seqs, self.num_quads, dt=self.dt, out_path=out_path)
        
        # 2.plot individual quad telemetry
        for i in range(self.num_quads):
            state_seq = np.array([h['state'][i] for h in self.history])  # (T, 6)
            action_seq = np.array([h['action'][i] for h in self.history])  # (T, 3)
            out_path = os.path.join(out_dir, f"dt_quad_{i}_telemetry.png")
            plot_quad_states_actions(state_seq, action_seq, dt=self.dt, out_path=out_path)
            
        self.ct_ctl_sim.visualize(out_dir=out_dir, fps=fps)

    def close(self):
        self.ct_ctl_sim.close()
        self.history = []

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

    # 
    acc = 1
    # random velocity commands
    key = jax.random.PRNGKey(0)
    vel_cmd_seq = sample_vel_cmd_sequence(
        key=key,
        amax=acc,
        dt=1/env_dt.frequency,
        n_steps=episode_length,
        v0=jnp.array([0.0, 0.0, 0.0]),
        v_bounds=jnp.array([[-5.0, -5.0, -2.0], [5.0, 5.0, 2.0]]),
    )  # (episode_length, 3)

    for step in range(episode_length):
        v_cmds = vel_cmd_seq[step][None, :].repeat(env_dt.num_quads, axis=0)
        env_dt.update(v_cmds, n_sim_time=1/env_dt.frequency)
    out_dir = "output/quad_sim_dt_test"
    os.makedirs(out_dir, exist_ok=True)
    env_dt.visualize(out_dir=out_dir)

    env_dt.close()

if __name__ == "__main__":
    main()