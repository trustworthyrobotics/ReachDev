import jax.numpy as jnp
import equinox as eqx

from models.mlp_utils import MLP

class Base_Controller(eqx.Module):
    Dx: int = eqx.field(static=True, default=12)
    Du: int = eqx.field(static=True, default=3)
    Dv: int = eqx.field(static=True, default=3)  # velocity commands
    action_bounds: jnp.ndarray = eqx.field(static=True)

    def __init__(self, data_cfg: dict):
        self.Dx = data_cfg.get("ct_state_dim", 12)
        self.Du = data_cfg.get("ct_action_dim", 3)
        self.Dv = data_cfg.get("dt_action_dim", 3)  # velocity commands
        if self.Du == 3:
            self.action_bounds = jnp.array(data_cfg.get("action_bounds", [[-15.0, -1.0, -1.0], [15.0, 1.0, 1.0]]))
        else:
            self.action_bounds = jnp.array(data_cfg.get("action_bounds", [[-15.0, -1.0, -1.0, -0.5], [15.0, 1.0, 1.0, 0.5]]))
        assert self.Dx == 12
        assert self.Du == 3 or self.Du == 4
        assert self.action_bounds.shape == (2, self.Du)
        assert self.Dv == 3
    
    def __call__(self, x, v_cmd):
        raise NotImplementedError

class PID_Controller(Base_Controller):
    # Gains for the three nested loops
    # Outer: Velocity -> Target Angles
    # Mid:   Angles   -> Target Rates
    # Inner: Rates    -> Torques
    kp_vel: jnp.ndarray = eqx.field(static=True)
    kp_att: jnp.ndarray = eqx.field(static=True)
    kp_rate: jnp.ndarray = eqx.field(static=True)
    m: float = eqx.field(static=True)
    g: float = eqx.field(static=True)

    def __init__(self, data_cfg: dict):
        super().__init__(data_cfg)
        self.g = data_cfg.get("g", 9.81)
        self.m = data_cfg.get("m", 1.4)
        
        pid_config = data_cfg.get("pid_gains", {})
        self.kp_vel = jnp.array(pid_config.get("kp", [1.5, 1.5, 10.0])) # [Lon, Lat, Ver]
        self.kp_att = jnp.array(pid_config.get("kp_att", [5.0, 5.0, 2.0]))
        self.kp_rate = jnp.array(pid_config.get("kp_rate", [10.0, 10.0, 5.0])) # [Roll, Pitch, Yaw]

    def __call__(self, x, v_cmd):
        # x: Full 12D state [pos_3, vel_3, att_3, rates_3]
        # v_cmd: Target [v_lon, v_lat, v_ver]
        
        # 1. OUTER LOOP: Velocity -> Target Angles/Thrust
        # Flip the sign of the vertical command because x6 is positive-down
        v_cmd_adjusted = jnp.array([v_cmd[0], v_cmd[1], -v_cmd[2]])
        v_err = v_cmd_adjusted - x[3:6]
        
        # Vertical velocity controls thrust (u1)
        # u1 is INCREMENTAL thrust. Also, the ODE has -u1/m, so positive u1 creates negative acceleration (Up).
        u1 = self.m * (self.kp_vel[2] * v_err[2])
        
        # Horizontal velocities set target Roll/Pitch
        # Lon velocity (x4) is increased by increasing Pitch (x8)
        # Lat velocity (x5) is increased by increasing Roll (x7)
        pitch_des = -self.kp_vel[0] * v_err[0] 
        roll_des = self.kp_vel[1] * v_err[1]
        yaw_des = 0.0 # Maintain north or current heading
        
        # 2. MID LOOP: Attitude -> Target Rates
        att_err = jnp.array([roll_des, pitch_des, yaw_des]) - x[6:9]
        rate_des = self.kp_att * att_err
        
        # 3. INNER LOOP: Rates -> Torques
        rate_err = rate_des - x[9:12]
        torques = self.kp_rate * rate_err
        
        if self.Du == 4:
            # u2: Roll torque, u3: Pitch torque, u4: Yaw torque
            u = jnp.array([u1, torques[0], torques[1], torques[2]])
        else:
            u = jnp.array([u1, torques[0], torques[1]])
        return jnp.clip(u, self.action_bounds[0], self.action_bounds[1])


class MLP_Controller(Base_Controller):
    model: MLP

    def __init__(self, data_cfg: dict, train_cfg: dict = {}):
        super().__init__(data_cfg)
        arch_list = train_cfg["architecture"]
        activation = train_cfg.get("activation", "relu")
        self.model = MLP(
            in_size=self.Dx + self.Dv,
            out_size=self.Du,
            hidden_sizes=arch_list,
            activation=activation,
        )

    def forward_batchless(self, x, v_cmd):
        u = self.model(jnp.concatenate([x, v_cmd]))
        return jnp.clip(u, self.action_bounds[0], self.action_bounds[1])
    
    __call__ = forward_batchless