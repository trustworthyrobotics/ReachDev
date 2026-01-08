from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

Array = jnp.ndarray
PRNGKey = jax.Array


from models.dt_dyn import T_Dynamics

class Continuous_T_Dynamics(T_Dynamics):
    dt: Array = eqx.field(static=True)
    dt0: Array = eqx.field(static=True)
    stepsize_controller: diffrax.Controller = eqx.field(static=True)

    def __init__(
        self,
        data_cfg: dict, 
        train_cfg: dict,
        key: PRNGKey = jax.random.PRNGKey(0),
    ):
        super().__init__(data_cfg, train_cfg, key)
        frequency = train_cfg.get("frequency", 10)
        self.dt = jnp.array(1.0 / frequency, dtype=jnp.float32)
        self.dt0 = jnp.array(self.dt / 5, dtype=jnp.float32)
        if train_cfg.get("ode_step_size_controller", "pid") == "const":
            self.stepsize_controller = diffrax.ConstantStepSize()
        else:
            self.stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-6)

    def dx(self, t, x, args):
        u = args  # pusher velocity
        if self.abs_pose:
            x_obj = x[:self.Ds]  # object state
            x_pusher = x[self.Ds:]  # pusher position
            if self.pred_mode == "state":
                x_rel = x_obj - x_pusher.repeat(self.Ds // self.Du, axis=-1)
            elif self.pred_mode == "pose":
                x_rel = jnp.concatenate([x_obj[:self.Du] - x_pusher, x_obj[self.Du:]], axis=-1)
            dx_obj = x_rel + self.mlp(jnp.concatenate([x_rel, u], axis=-1))
            return jnp.concatenate([dx_obj, u], axis=-1)
        else:
            # Predict the derivative of keypoints
            return self.mlp(jnp.concatenate([x, u], axis=-1))

    def forward(self, x: Array, u: Array) -> Array:
        # x: (B,Dx), u: (B,Du)
        # One step forward prediction
        return jax.vmap(self.forward_batchless)(x, u)

    def forward_batchless(self, x: Array, u: Array) -> Array:
        # x: (Dx,), u: (Du,)
        term = diffrax.ODETerm(self.dx)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(term, solver, t0=0, t1=self.dt, dt0=self.dt0, y0=x, args=u, stepsize_controller=self.stepsize_controller)
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
