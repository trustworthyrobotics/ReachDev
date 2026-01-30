import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

Array = jnp.ndarray

# analytical continuous-time quadrotor dynamics
class Continuous_Quad_Dynamics(eqx.Module):
    Dx: int = eqx.field(static=True, default=12)
    Du: int = eqx.field(static=True, default=3)
    g: float = eqx.field(static=True)
    m: float = eqx.field(static=True)
    J_x: float = eqx.field(static=True)
    J_y: float = eqx.field(static=True)
    J_z: float = eqx.field(static=True)
    tau_phi: float = eqx.field(static=True)
    frequency: float = eqx.field(static=True)
    dt: Array = eqx.field(static=True)
    dt0: Array = eqx.field(static=True)
    stepsize_controller: diffrax.PIDController = eqx.field(static=True)

    def __init__(
        self,
        data_cfg: dict, 
        *args, **kwargs,
    ):
        self.Dx = data_cfg.get("ct_state_dim", 12)
        self.Du = data_cfg.get("ct_action_dim", 3)
        assert self.Dx == 12
        assert self.Du == 3 or self.Du == 4
        self.frequency = float(data_cfg.get("ct_frequency", 100))
        self.dt = 1 / self.frequency
        self.dt0 = self.dt / 5
        self.g = data_cfg.get("g", 9.81)
        self.m = data_cfg.get("m", 1.4)
        self.J_x = data_cfg.get("J_x", 0.054)
        self.J_y = data_cfg.get("J_y", 0.054)
        self.J_z = data_cfg.get("J_z", 0.104)
        self.tau_phi = data_cfg.get("tau_phi", 0.0)
        if data_cfg.get("ode_step_size_controller", "pid") == "const":
            self.stepsize_controller = diffrax.ConstantStepSize()
        else:
            self.stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-6)

    # def dx(self, t, x, args):
    #     x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12 = x
    #     if self.Du == 4:
    #         u1,u2,u3,u4 = args
    #     else:
    #         u1,u2,u3 = args
    #         u4 = self.tau_phi  # no yaw control
    #     c7, s7 = jnp.cos(x7), jnp.sin(x7)   # roll
    #     c8, s8 = jnp.cos(x8), jnp.sin(x8)   # pitch
    #     c9, s9 = jnp.cos(x9), jnp.sin(x9)   # yaw
    #     sec8 = 1.0 / c8
    #     g = self.g
    #     m = self.m
    #     J_x = self.J_x
    #     J_y = self.J_y
    #     J_z = self.J_z

    #     dx1 = c8*c9*x4 + (s7*s8*c9 - c7*s9)*x5 + (c7*s8*c9 + s7*s9)*x6
    #     dx2 = c8*s9*x4 + (s7*s8*s9 + c7*c9)*x5 + (c7*s8*s9 - s7*c9)*x6
    #     dx3 = s8*x4 - s7*c8*x5 - c7*c8*x6
    #     dx4 = x12*x5 - x11*x6 - g*s8
    #     dx5 = x10*x6 - x12*x4 + g*c8*s7
    #     dx6 = x11*x4 - x10*x5 + g*c8*c7 - g - u1/m
    #     dx7 = x10 + (s7*(s8*sec8))*x11 + (c7*(s8*sec8))*x12
    #     dx8 = c7*x11 - s7*x12
    #     dx9 = (s7*sec8)*x11 - (c7*sec8)*x12
    #     dx10 = (J_y - J_z)/J_x * x11 * x12 + u2/J_x
    #     dx11 = (J_z - J_x)/J_y * x10 * x12 + u3/J_y
    #     dx12 = (J_x - J_y)/J_z * x10 * x11 + u4/J_z
    #     return jnp.array([dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9,dx10,dx11,dx12])

    def dx(self, t, x, args):
        # State:
        # x1,x2,x3: world position
        # x4,x5,x6: world velocity   <-- DIFFERENT from the benchmark model
        x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12 = x

        # Input:
        # u1: thrust magnitude (hover near u1 = m*g)
        # u2,u3,u4: body torques
        if self.Du == 4:
            u1,u2,u3,u4 = args
        else:
            u1,u2,u3 = args
            u4 = 0.0  # no yaw control

        # roll/pitch/yaw
        c7, s7 = jnp.cos(x7), jnp.sin(x7)   # roll  (phi)
        c8, s8 = jnp.cos(x8), jnp.sin(x8)   # pitch (theta)
        c9, s9 = jnp.cos(x9), jnp.sin(x9)   # yaw   (psi)

        sec8 = 1.0 / c8  # same singularity as your current model

        g  = self.g
        m  = self.m
        Jx = self.J_x
        Jy = self.J_y
        Jz = self.J_z

        # 1) Position kinematics (world)
        dx1 = x4
        dx2 = x5
        dx3 = x6

        # 2) Translational dynamics (world)
        # b3 = R_WB * e3  (3rd column of body->world rotation for ZYX Euler)
        b3x = c7*s8*c9 + s7*s9
        b3y = c7*s8*s9 - s7*c9
        b3z = c7*c8

        # World acceleration: vdot = (u1/m) * b3 + [0,0,-g]
        dx4 = (u1 / m) * b3x
        dx5 = (u1 / m) * b3y
        dx6 = (u1 / m) * b3z - g

        # 3) Euler kinematics driven by body rates (keep same as your benchmark)
        # (x10,x11,x12) are p,q,r
        dx7 = x10 + (s7*(s8*sec8))*x11 + (c7*(s8*sec8))*x12
        dx8 = c7*x11 - s7*x12
        dx9 = (s7*sec8)*x11 + (c7*sec8)*x12

        # 4) Rigid-body rotational dynamics (same as your benchmark)
        dx10 = (Jy - Jz)/Jx * x11 * x12 + u2/Jx
        dx11 = (Jz - Jx)/Jy * x10 * x12 + u3/Jy
        dx12 = (Jx - Jy)/Jz * x10 * x11 + u4/Jz

        return jnp.array([dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9,dx10,dx11,dx12])

    def forward(self, x: Array, u: Array, dt: Array=None) -> Array:
        # x: (B,Dx), u: (B,Du)
        # One step forward prediction
        _step = jax.vmap(lambda x,u: self.forward_batchless(x, u, dt))
        return _step(x, u)

    def forward_batchless(self, x: Array, u: Array, dt: Array=None) -> Array:
        if dt is None:
            dt = self.dt
        # x: (Dx,), u: (Du,)
        term = diffrax.ODETerm(self.dx)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(term, solver, t0=0, t1=dt, dt0=self.dt0, y0=x, args=u, stepsize_controller=self.stepsize_controller)
        return sol.ys[-1]

    def rollout(self, x0: Array, U: Array) -> Array:
        # x0: (B,Dx), U: (B,T,Du)
        # Use diffrax to integrate over the control frequency
        def step(state, u):
            x_next = self.forward_batchless(state, u)
            return x_next, x_next
            
        _, x_seq = jax.lax.scan(jax.vmap(step), x0, U.transpose(1,0,2))
        return x_seq.transpose(1,0,2)

    def forward_batchless_single_input(self, inp):
        x = inp[:self.Dx]
        u = inp[-self.Du:]
        return self.dx(0.0, x, u)
    
    __call__ = forward_batchless_single_input

    # it is only used for Jacobian regularization
    def forward_batchless_for_jac(self, x: Array, u: Array) -> Array:
        return self.dx(0.0, x, u)
