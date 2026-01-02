from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

Array = jnp.ndarray
PRNGKey = jax.Array


from models.dynamics import T_Dynamics

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
            # Predict the derivative of keypoints
            return self.mlp(jnp.concatenate([x, u], axis=-1))

    def forward(self, x, u):
        # x: (B,Dx), u: (B,Du)
        # One step forward prediction
        term = diffrax.ODETerm(self.dx)
        solver = diffrax.Tsit5()
        sol = jax.vmap(diffrax.diffeqsolve)(term, solver, t0=0, t1=self.dt, dt0=self.dt0, y0=x, args=u, stepsize_controller=self.stepsize_controller)
        return sol.ys[-1]

    def rollout(self, x0, U):
        # x0: (B,Dx), U: (B,T,Du)
        # Use diffrax to integrate over the control frequency
        def step(state, u):
            term = diffrax.ODETerm(self.dx)
            solver = diffrax.Tsit5()
            sol = diffrax.diffeqsolve(term, solver, t0=0, t1=self.dt, dt0=self.dt0, y0=state, args=u, stepsize_controller=self.stepsize_controller)
            return sol.ys[-1], sol.ys[-1]
            
        _, x_seq = jax.lax.scan(jax.vmap(step), x0, U.transpose(1,0,2))
        return x_seq.transpose(1,0,2)

    def forward_batchless_single_input(self, inp):
        return self.mlp(inp)
    
    __call__ = forward_batchless_single_input


def load_t_dynamics_model(data_config: dict, train_config: dict, model_path: str) -> Continuous_T_Dynamics:
    model_def = Continuous_T_Dynamics(data_config, train_config)
    with open(model_path, "rb") as f:
        model: Continuous_T_Dynamics = eqx.tree_deserialise_leaves(f, model_def)
    return model