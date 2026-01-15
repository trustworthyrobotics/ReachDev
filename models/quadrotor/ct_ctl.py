from typing import List
import jax
import jax.numpy as jnp
import equinox as eqx

from models.mlp_utils import MLP

Array = jnp.ndarray
PRNGKey = jax.Array

class Base_Controller(eqx.Module):
    Dx: int = eqx.field(static=True, default=12)
    Ds: int = eqx.field(static=True, default=12)
    Du: int = eqx.field(static=True, default=3)
    Dv: int = eqx.field(static=True, default=3)  # velocity commands
    Dr: int = eqx.field(static=True, default=3)
    frequency: float = eqx.field(static=True, default=50.0)  # control frequency
    dt: float = eqx.field(static=True, default=0.02)  # control timestep
    action_bounds: Array = eqx.field(static=True)

    def __init__(self, data_cfg: dict):
        self.Dx = data_cfg.get("ct_state_dim", 12)
        self.Ds = self.Dx
        self.Du = data_cfg.get("ct_action_dim", 3)
        self.Dv = data_cfg.get("dt_action_dim", 3)  # velocity commands
        self.Dr = self.Dv # reference velocity commands
        self.frequency = data_cfg.get("ctl_frequency", 50.0)
        self.dt = 1.0 / self.frequency
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
    # Outer: Velocity -> Target Angles + Thrust
    # Mid:   Angles   -> Target Rates
    # Inner: Rates    -> Torques
    kp_vel: Array = eqx.field(static=True)
    kp_att: Array = eqx.field(static=True)
    kp_rate: Array = eqx.field(static=True)
    m: float = eqx.field(static=True)
    g: float = eqx.field(static=True)

    def __init__(self, data_cfg: dict):
        super().__init__(data_cfg)
        self.g = data_cfg.get("g", 9.81)
        self.m = data_cfg.get("m", 1.4)

        pid_config = data_cfg.get("pid_gains", {})
        # For world-velocity model: x[3:6] is world velocity [vx, vy, vz] (z-up if you follow ENU)
        self.kp_vel  = jnp.array(pid_config.get("kp_vel",      [1.5, 1.5, 10.0]))
        self.kp_att  = jnp.array(pid_config.get("kp_att",  [5.0, 5.0, 2.0]))
        self.kp_rate = jnp.array(pid_config.get("kp_rate", [10.0, 10.0, 5.0]))

    def forward_batchless(self, x, v_cmd):
        # x: 12D state [pos(3), velW(3), rpy(3), pqr(3)]
        # v_cmd: desired world velocity [vx, vy, vz] (ENU z-up)

        # unpack
        vx, vy, vz = x[3], x[4], x[5]
        roll, pitch, yaw = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]

        # 1) OUTER LOOP: velocity -> desired accel (world)
        v_err = v_cmd - x[3:6]
        ax_cmd = self.kp_vel[0] * v_err[0]
        ay_cmd = self.kp_vel[1] * v_err[1]
        az_cmd = self.kp_vel[2] * v_err[2]

        # 1a) Thrust with gravity compensation (world-velocity dynamics)
        # dvz = (u1/m)*b3z - g, b3z = cos(roll)*cos(pitch)
        c7, c8 = jnp.cos(roll), jnp.cos(pitch)
        b3z = c7 * c8
        # avoid divide-by-zero if you ever get near 90deg (optional)
        b3z = jnp.where(jnp.abs(b3z) < 1e-3, 1e-3 * jnp.sign(b3z + 1e-6), b3z)
        u1 = self.m * (self.g + az_cmd) / b3z

        # 1b) Desired roll/pitch from desired horizontal accel, yaw-aware (small-angle hover approx)
        c9, s9 = jnp.cos(yaw), jnp.sin(yaw)
        pitch_des = (ax_cmd * c9 + ay_cmd * s9) / self.g
        roll_des  = (ax_cmd * s9 - ay_cmd * c9) / self.g
        yaw_des = 0.0

        # 2) MID LOOP: attitude -> desired body rates
        att_err = jnp.array([roll_des, pitch_des, yaw_des]) - jnp.array([roll, pitch, yaw])
        rate_des = self.kp_att * att_err  # [p_des, q_des, r_des]

        # 3) INNER LOOP: body rates -> torques
        rate_err = rate_des - jnp.array([p, q, r])
        torques = self.kp_rate * rate_err  # [tau_x, tau_y, tau_z]

        if self.Du == 4:
            u = jnp.array([u1, torques[0], torques[1], torques[2]])
        else:
            u = jnp.array([u1, torques[0], torques[1]])

        return jnp.clip(u, self.action_bounds[0], self.action_bounds[1])

    def forward(self, x: Array, v_cmd: Array) -> Array:
        return jax.vmap(self.forward_batchless)(x, v_cmd)

    __call__ = forward_batchless

class MLP_Controller(Base_Controller):
    model: MLP
    enforce_action_bounds: bool = eqx.field(static=True, default=True)
    enable_standardization: bool = eqx.field(static=True, default=False)
    x_mean: Array = eqx.field(static=True)
    x_std: Array = eqx.field(static=True)
    v_mean: Array = eqx.field(static=True)
    v_std: Array = eqx.field(static=True)
    u_mean: Array = eqx.field(static=True)
    u_std: Array = eqx.field(static=True)

    def __init__(self, data_cfg: dict, train_cfg: dict, key: PRNGKey = jax.random.PRNGKey(0), stats: dict = None):
        super().__init__(data_cfg)
        arch_list = train_cfg["architecture"]
        activation = train_cfg.get("activation", "relu")
        self.enforce_action_bounds = train_cfg.get("enforce_action_bounds", True)
        self.model = MLP(
            in_size=self.Dx + self.Dv,
            out_size=self.Du,
            hidden_size_list=arch_list,
            activation=activation,
            key=key,
        )
        if stats is not None:
            self.enable_standardization = True
            mean = jnp.array(stats["mean"])
            std = jnp.array(stats["std"])
            assert mean.shape == (self.Dx + self.Dv + self.Du,)
            assert std.shape == (self.Dx + self.Dv + self.Du,)

            self.x_mean = mean[:self.Dx]
            self.x_std = std[:self.Dx]
            self.v_mean = mean[self.Dx:self.Dx + self.Dv]
            self.v_std = std[self.Dx:self.Dx + self.Dv]
            self.u_mean = mean[self.Dx + self.Dv:]
            self.u_std = std[self.Dx + self.Dv:]
        else:
            self.enable_standardization = False
            self.x_mean = jnp.zeros(self.Dx)
            self.x_std = jnp.ones(self.Dx)
            self.v_mean = jnp.zeros(self.Dv)
            self.v_std = jnp.ones(self.Dv)
            self.u_mean = jnp.zeros(self.Du)
            self.u_std = jnp.ones(self.Du)

    def _input_dims(self) -> List[int]:
        return self.Dx, self.Dr

    def forward(self, x: Array, v_cmd: Array) -> Array:
        return jax.vmap(self.forward_batchless)(x, v_cmd)

    def forward_batchless(self, x, v_cmd):
        if self.enable_standardization:
            # standardize
            x = (x - self.x_mean) / self.x_std
            v_cmd = (v_cmd - self.v_mean) / self.v_std
        raw = self.model(jnp.concatenate([x, v_cmd], axis=-1))
        if self.enforce_action_bounds:
            lo, hi = self.action_bounds[0], self.action_bounds[1]
            # we use sigmoid to replace tanh because jax_verify does not handle tanh well
            u = 0.5 * (hi + lo) + (hi - lo) * jax.nn.sigmoid(2 * raw) - 1
            if self.enable_standardization:
                # unstandardize
                u = u * self.u_std + self.u_mean
            return u
        else:
            if self.enable_standardization:
                # unstandardize
                raw = raw * self.u_std + self.u_mean
            return raw

    def forward_batchless_single_input(self, inp):
        x = inp[:self.Dx]
        v_cmd = inp[-self.Dv:]
        return self.forward_batchless(x, v_cmd)
        
    __call__ = forward_batchless_single_input

