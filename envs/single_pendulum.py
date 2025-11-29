from typing import Tuple, Dict, Any
from functools import partial
import jax.numpy as jnp
import jax
from jax import jit, lax


class SinglePendulumEnv:
    """
    Single pendulum:
      State x = [θ, θ̇]
      Control u = [T]
      Continuous-time dynamics:
        ẋ1 = x2
        ẋ2 = (g/L) * sin(x1) + (1/(m*L^2)) * (u - c*x2)
    Discrete step uses sub-stepping Euler with robust dt partition.
    """

    def __init__(
        self,
        m: float = 0.5,
        L: float = 0.5,
        c: float = 0.0,
        g: float = 1.0,
        ode_dt: float = 0.01,
        ctrl_dt: float = 0.05,
        init_lo: jnp.ndarray = jnp.array([0.3, -0.8]),
        init_hi: jnp.ndarray = jnp.array([1.7,  0.8]),
        act_lo: jnp.ndarray = jnp.array([-2.0]),
        act_hi: jnp.ndarray = jnp.array([2.0]),
    ):
        self.m, self.L, self.c, self.g = m, L, c, g
        self.ode_dt = float(ode_dt)
        self.ctrl_dt = float(ctrl_dt)
        self.n_substeps = int(jnp.ceil(self.ctrl_dt / self.ode_dt))
        self.init_lo = init_lo
        self.init_hi = init_hi
        self.act_lo = act_lo
        self.act_hi = act_hi

    # -------- continuous dynamics --------
    @partial(jit, static_argnums=0)
    def f(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x[..., 0], x[..., 1]
        T = u[..., 0]
        dx1 = x2
        dx2 = (self.g / self.L) * jnp.sin(x1) + (1.0 / (self.m * self.L**2)) * (T - self.c * x2)
        return jnp.stack([dx1, dx2], axis=-1)

    def step(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        def euler_step(x_curr, _):
            return x_curr + self.ode_dt * self.f(x_curr, u), None
        x_next, _ = lax.scan(euler_step, x, None, length=self.n_substeps)
        return x_next

    # -------- sampling helpers --------
    def sample_initial_states(self, key, n: int) -> jnp.ndarray:
        lo, hi = self.init_lo[None, :], self.init_hi[None, :]
        return jax.random.uniform(key, shape=(n, 2), minval=lo, maxval=hi)

    def clip_action(self, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(u, self.act_lo, self.act_hi)

    # -------- construction from config --------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SinglePendulumEnv":
        d = cfg["data"]
        # sanity checks
        assert d["state_dim"] == 2 and d["action_dim"] == 1, "SinglePendulum expects state_dim=2, action_dim=1."
        init_lo = jnp.array([interval[0] for interval in d["initial_set"]])
        init_hi = jnp.array([interval[1] for interval in d["initial_set"]])
        act_lo = jnp.array([interval[0] for interval in d["action_range"]])
        act_hi = jnp.array([interval[1] for interval in d["action_range"]])
        return cls(
            m=d["m"], L=d["L"], c=d["c"], g=d["g"],
            ode_dt=d["ode_step_size"], ctrl_dt=d["control_step_size"],
            init_lo=init_lo, init_hi=init_hi,
            act_lo=act_lo, act_hi=act_hi,
        )
