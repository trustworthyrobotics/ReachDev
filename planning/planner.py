# planners_base_jax.py
from __future__ import annotations
from typing import Callable, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import lax
from jax.nn import softmax

Array=jnp.ndarray
PRNGKey=jax.Array


class SamplingPlannerBase(eqx.Module):
    """JAX/Equinox base for sampling-based planners.
    """

    # core dims/limits
    action_dim:int
    horizon:int
    n_sample:int
    n_update_iter:int
    action_lower_lim:Array
    action_upper_lim:Array
    use_last:bool
    reject_bad:bool

    # external functions
    rollout_fn:Callable
    eval_fn:Callable

    # ---- init from config ----
    def __init__(self, config:dict, model_rollout_fn: Callable, evaluate_traj_fn: Callable, action_lower_lim: Array, action_upper_lim: Array):
        planning_config: dict = config["planning"]
        # required
        self.rollout_fn      = model_rollout_fn
        self.eval_fn         = evaluate_traj_fn
        self.action_dim      = int(config["data"]["action_dim"])
        self.n_sample        = int(planning_config["n_sample"])
        self.horizon    = int(planning_config["horizon"])
        self.n_update_iter   = int(planning_config["n_update_iter"])
        self.action_lower_lim= jnp.asarray(action_lower_lim)
        self.action_upper_lim= jnp.asarray(action_upper_lim)

        self.use_last      = bool(planning_config.get("use_last", True))
        self.reject_bad    = bool(planning_config.get("reject_bad", False))

    # ---- utilities shared by subclasses ----
    def _clip_actions(self, act:Array)->Array:
        """Clip actions into box limits."""
        return jnp.clip(act, self.action_lower_lim, self.action_upper_lim)

    def _evaluate(self, state_cur:Array, act_seqs:Array, *args, **kwargs)->Tuple[Array,Dict]:
        """Rollout + evaluate; returns (rewards[B], aux_dict)."""
        # rollout_fn expects: state_cur, action_seqs (B,H,Du)                # :contentReference[oaicite:16]{index=16}
        state_seqs=self.rollout_fn(state_cur, act_seqs)
        # eval_fn expects: state_seqs, action_seqs                           # :contentReference[oaicite:17]{index=17}
        eval_out=self.eval_fn(state_seqs, act_seqs, *args, **kwargs)
        rewards=eval_out["rewards"]
        return rewards, {"model_out":state_seqs, "eval_out":eval_out}

    # entry point for planners; subclasses must implement the algorithmic loop
    def plan(self, key:PRNGKey, state_cur:Array, init_action_seq:Array, *args, **kwargs)->Tuple[PRNGKey,Dict]:
        """Return (key, result_dict) with at least {'act_seq':(H,Du), 'best_reward':...}."""
        raise NotImplementedError

    # optional thin wrapper if you want parity with torch’s trajectory_optimization
    def trajectory_optimization(self, key:PRNGKey, state_cur:Array, init_action_seq:Array, skip:bool=False, *args, **kwargs)->Dict:
        """Keeps your Torch API shape, but delegates to `plan`."""
        if skip:
            # evaluate the given init_action_seq (1,H,Du)
            rewards, aux = self._evaluate(state_cur, init_action_seq[None,...], *args, **kwargs)
            state_seq = aux["model_out"][0]  # (H,Ds)
            rewards = rewards[0]
            return {
                "act_seq": init_action_seq,
                "state_seq": state_seq,
                "reward": rewards,
            }
        key, out = self.plan(key, state_cur, init_action_seq, *args, **kwargs)
        return out

class MPPIPlanner(SamplingPlannerBase):
    """Model Predictive Path Integral (MPPI) planner.

    Config keys (in addition to SamplingPlannerBase):
      OPTIONAL:
        - reward_weight: float (inverse temperature λ; default 10.0)
          # aliases for convenience:
          - mppi_lambda, temperature (uses 1/temperature if provided)
          If True, return the softmax-weighted mean trajectory as the final candidate.
          The best sampled trajectory is always returned in 'act_seq_best'.
    """
    reward_weight: float
    noise_level: float
    noise_decay: float
    beta_filter: float

    def __init__(self, config:dict, model_rollout_fn: Callable, evaluate_traj_fn: Callable, action_lower_lim: Array, action_upper_lim: Array):
        super().__init__(config, model_rollout_fn, evaluate_traj_fn, action_lower_lim, action_upper_lim)
        mppi_config = config["planning"]["mppi"]
        self.reward_weight = float(mppi_config["reward_weight"])
        self.noise_level   = float(mppi_config["noise_level"])
        self.noise_decay   = float(mppi_config.get("noise_decay", 1.0))
        self.beta_filter   = float(mppi_config.get("beta_filter", 0.7))

    # default sampler: temporal-filtered Gaussian around a base sequence
    def _sample_action_sequences_default(
        self, key:PRNGKey, base_seq:Array, noise_level:Array
    )->Tuple[PRNGKey,Array]:
        """base_seq:(H,Du) → (B,H,Du), using beta-filtered noise and decay."""
        B=self.n_sample; H=self.horizon; Du=self.action_dim
        key_eps, key= jax.random.split(key)

        eps = jax.random.normal(key_eps, (B,H,Du)) * noise_level
        # temporal filtering
        def step(prev,t):
            cur=self.beta_filter*eps[:,t,:] + (1.0-self.beta_filter)*prev
            return cur, cur
        init=jnp.zeros((B,Du), dtype=base_seq.dtype)
        _, resid = lax.scan(step, init, jnp.arange(H))        # (H,B,Du)
        resid = jnp.swapaxes(resid,0,1)                        # (B,H,Du)

        act = jnp.broadcast_to(base_seq,(B,H,Du)) + resid
        act = self._clip_actions(act)
        return key, act

    def plan(
        self, key:PRNGKey, state_cur:Array, init_action_seq:Array, *args, **kwargs
    )->Tuple[PRNGKey,Dict]:
        """Run MPPI for n_update_iter iterations.

        Args:
          key: PRNGKey
          state_cur: (...,) current state (passed through to rollout_fn)
          init_action_seq: (H,Du) initial mean sequence

        Returns:
          key, {
            'act_seq': (H,Du)  # chosen sequence (last iter seq if use_last else best sample)
            'state_seq': (H,Ds)  # resulting state sequence from rollout
            'best_reward': float
            'iter_reward_max': (T,), 'iter_reward_mean': (T,)
          }
        """
        H, Du = self.horizon, self.action_dim

        # carry: key, mean_seq, best_seq, best_rew, noise_level, prev_rewards, prev_act
        best_seq0 = init_action_seq
        best_rew0 = -jnp.inf
        nlvl0     = self.noise_level
        prev_rewards0 = -jnp.inf*jnp.ones((self.n_sample,), dtype=init_action_seq.dtype)
        prev_act0     = jnp.broadcast_to(init_action_seq,(self.n_sample,H,Du))

        def one_iter(carry, it):
            key, mean_seq, best_seq, best_rew, nlvl, prev_rewards, prev_act = carry

            # sample around mean_seq using the (possibly overridden) sampler
            key, act_seqs = self._sample_action_sequences_default(key, mean_seq, nlvl)  # (B,H,Du)

            # rollout + evaluate
            rewards, _aux = self._evaluate(state_cur, act_seqs, *args, **kwargs)  # (B,)

            # optional: reject_bad — keep previous better samples elementwise
            if self.reject_bad:
                use_prev = rewards < prev_rewards
                act_seqs = jnp.where(use_prev[:,None,None], prev_act, act_seqs)
                rewards  = jnp.where(use_prev, prev_rewards, rewards)

            # softmax weights (stable)
            w = softmax(rewards * self.reward_weight, axis=0)        # (B,)
            mean_new = jnp.einsum("b,bhd->hd", w, act_seqs)           # (H,Du)
            mean_new = self._clip_actions(mean_new)

            # track best
            i_best = jnp.argmax(rewards)
            best_seq = jnp.where(rewards[i_best] > best_rew, act_seqs[i_best], best_seq)
            best_rew = jnp.maximum(best_rew, rewards.max())

            # update noise level
            nlvl = nlvl * self.noise_decay

            logs = {
                "reward_max": rewards.max(),
                "reward_mean": rewards.mean(),
            }
            new_carry = (key, mean_new, best_seq, best_rew, nlvl, rewards, act_seqs)
            return new_carry, logs

        carry0 = (key, init_action_seq, best_seq0, best_rew0, nlvl0, prev_rewards0, prev_act0)
        (key, final_mean, best_seq, best_rew, _nlvlT, _prevR, _prevA), logs = lax.scan(
            one_iter, carry0, jnp.arange(self.n_update_iter)
        )

        # choose output action sequence
        act_seq_choice = jax.lax.cond(
            self.use_last,
            lambda _: final_mean,
            lambda _: best_seq,
            operand=None,
        )
        rewards, _aux = self._evaluate(state_cur, act_seq_choice[None,...], *args, **kwargs)  # (B,)
        state_seq = _aux["model_out"][0]  # (H,Ds)
        rewards = rewards[0]

        out = {
            "act_seq": act_seq_choice,      # main action sequence to execute
            "state_seq": state_seq,         # resulting state sequence from rollout
            "reward": rewards,
            "iter_reward_max": logs["reward_max"],
            "iter_reward_mean": logs["reward_mean"],
        }
        return key, out


class CEMPlanner(SamplingPlannerBase):
    """Cross-Entropy Method (CEM) planner.

    Config keys (in addition to SamplingPlannerBase):
      OPTIONAL:
        - elite_ratio: float in (0,1] (default 0.05)
        - min_n_elites: int >=1 (default 10)
        - use_full_cov: bool (default False)  # True=(P,P) covariance, False=diag length P
        - cov_jitter: float (default 1e-6)    # numerical stability
        - init_mean: Optional[Array (H,Du)]   # if absent, use 'init_action_seq' at call
        - init_cov_scale: float (default 0.25)# scale^2 * (box_range^2) as initial variance
        - mean_momentum: float in [0,1] (default 0.0)  # EMA on mean updates
        - cov_momentum: float in [0,1] (default 0.0)   # EMA on covariance updates
    """
    elite_ratio: float
    min_n_elites: int
    use_full_cov: bool
    cov_jitter: float
    init_cov_scale: float
    mean_momentum: float
    cov_momentum: float

    def __init__(self, config:dict, model_rollout_fn: Callable, evaluate_traj_fn: Callable, action_lower_lim: Array, action_upper_lim: Array):
        super().__init__(config, model_rollout_fn, evaluate_traj_fn, action_lower_lim, action_upper_lim)
        cem_config = config["planning"]["cem"]
        self.elite_ratio       = float(cem_config["elite_ratio"])
        self.min_n_elites      = int(cem_config["min_n_elites"])
        self.use_full_cov      = bool(cem_config.get("use_full_cov", False))
        self.cov_jitter        = float(cem_config.get("cov_jitter", 1e-6))
        self.init_cov_scale    = float(cem_config.get("init_cov_scale", 0.25))
        self.mean_momentum     = float(cem_config.get("mean_momentum", 0.0))
        self.cov_momentum      = float(cem_config.get("cov_momentum", 0.0))

    def _flatten(self, x:Array)->Array:
        # (..., H, Du) -> (..., P)
        H, Du = self.horizon, self.action_dim
        return x.reshape(x.shape[:-2]+(H*Du,))

    def _unflatten(self, z:Array)->Array:
        # (..., P) -> (..., H, Du)
        H, Du = self.horizon, self.action_dim
        return z.reshape(z.shape[:-1]+(H,Du))

    def _init_mean_cov(self, init_action_seq:Array)->Tuple[Array,Array]:
        """Return (mean[P], cov[(P,) or (P,P)])."""
        H, Du = self.horizon, self.action_dim
        # mean
        mean = self._flatten(init_action_seq)

        # variance from box range
        box_rng = (self.action_upper_lim - self.action_lower_lim)  # (Du,)
        diag_du = (self.init_cov_scale * (box_rng**2)).repeat(H)   # (P,)
        if self.use_full_cov:
            cov = jnp.diag(diag_du)
        else:
            cov = diag_du
        return mean, cov

    def _sample_gaussian(
        self, key:PRNGKey, mean:Array, cov:Array, n:int
    )->Tuple[PRNGKey,Array]:
        """Return (key, samples[B,P]) from N(mean,cov)."""
        P = mean.shape[0]
        if self.use_full_cov:
            key, k = jax.random.split(key)
            # ensure PSD
            cov_psd = cov + self.cov_jitter*jnp.eye(P, dtype=mean.dtype)
            L = jnp.linalg.cholesky(cov_psd)
            z = jax.random.normal(k, (n, P), dtype=mean.dtype)
            x = z @ L.T + mean[None, :]
            return key, x
        else:
            key, k = jax.random.split(key)
            std = jnp.sqrt(cov + self.cov_jitter)
            z = jax.random.normal(k, (n, P), dtype=mean.dtype)
            x = z * std[None, :] + mean[None, :]
            return key, x

    def _fit_elites(self, elites:Array)->Tuple[Array,Array]:
        """elites: (E,P) -> (mean[P], cov[(P,) or (P,P)])."""
        if self.use_full_cov:
            # unbiased covariance across rows (samples)
            mean = elites.mean(axis=0)
            xc = elites - mean
            E = elites.shape[0]
            # cov = (xc^T xc)/(E-1)
            cov = (xc.T @ xc) / jnp.maximum(E-1, 1)
            cov = cov + self.cov_jitter*jnp.eye(mean.shape[0], dtype=elites.dtype)
            return mean, cov
        else:
            mean = elites.mean(axis=0)
            var  = elites.var(axis=0) + self.cov_jitter
            return mean, var

    def plan(
        self, key:PRNGKey, state_cur:Array, init_action_seq:Array, *args, **kwargs
    )->Tuple[PRNGKey,Dict]:
        """Run CEM for n_update_iter iterations."""
        H, Du = self.horizon, self.action_dim
        P = H*Du
        B = self.n_sample
        n_elite = jnp.maximum(int(self.elite_ratio*B), self.min_n_elites)

        mean0, cov0 = self._init_mean_cov(init_action_seq)

        # For tracking the single best sample (optional)
        best_sample0 = self._flatten(init_action_seq)
        best_reward0 = -jnp.inf

        def one_iter(carry, it):
            key, mean, cov, best_sample, best_reward = carry

            # sample
            key, X = self._sample_gaussian(key, mean, cov, B)   # (B,P)
            A = self._unflatten(X)                               # (B,H,Du)
            # clamp samples to box
            A = jnp.clip(
                A,
                self.action_lower_lim[None,None,:],
                self.action_upper_lim[None,None,:],
            )

            # evaluate
            rewards, _ = self._evaluate(state_cur, A, *args, **kwargs)           # (B,)

            # update best sample
            argmax = jnp.argmax(rewards)
            sample_star = X[argmax]
            reward_star = rewards[argmax]
            take_new = reward_star > best_reward
            best_sample = jnp.where(take_new, sample_star, best_sample)
            best_reward = jnp.where(take_new, reward_star, best_reward)

            # select elites
            idx = jnp.argsort(rewards)[::-1][:n_elite]
            elites = X[idx]                                      # (E,P)

            # fit distribution
            mean_fit, cov_fit = self._fit_elites(elites)

            # EMA smoothing
            mean_new = (1.0 - self.mean_momentum)*mean_fit + self.mean_momentum*mean
            if self.use_full_cov:
                cov_new = (1.0 - self.cov_momentum)*cov_fit + self.cov_momentum*cov
                cov_new = cov_new + self.cov_jitter*jnp.eye(P, dtype=cov_new.dtype)
            else:
                cov_new = (1.0 - self.cov_momentum)*cov_fit + self.cov_momentum*cov
                cov_new = cov_new + self.cov_jitter

            # clamp mean in box (reshape then flatten)
            mean_new = self._flatten(jnp.clip(self._unflatten(mean_new), self.action_lower_lim[None,None,:], self.action_upper_lim[None,None,:],)).reshape(-1)

            logs = {
                "reward_max": rewards.max(),
                "reward_mean": rewards.mean(),
                "elite_reward_mean": rewards[idx].mean(),
            }
            return (key, mean_new, cov_new, best_sample, best_reward), logs

        carry0 = (key, mean0, cov0, best_sample0, best_reward0)
        (key, meanT, covT, best_sampleT, best_rewardT), logs = lax.scan(
            one_iter, carry0, jnp.arange(self.n_update_iter)
        )

        # final candidate = mean of the last iteration (standard CEM practice)
        final_seq = jnp.clip(self._unflatten(meanT), self.action_lower_lim[None, :], self.action_upper_lim[None, :],)
        # evaluate the final candidate (optional)
        best_rewards_final, aux_final = self._evaluate(state_cur, final_seq[None, ...], *args, **kwargs)  # (1,)

        final_vs_tracked_better = best_rewards_final[0] >= best_rewardT
        act_seq_best = jax.lax.select(final_vs_tracked_better, final_seq, self._unflatten(best_sampleT))

        act_seq_choice = jax.lax.cond(
            self.use_last,
            lambda _: final_seq,
            lambda _: act_seq_best,
            operand=None,
        )
        rewards, _aux = self._evaluate(state_cur, act_seq_choice[None,...], *args, **kwargs)  # (B,)
        state_seq = _aux["model_out"][0]  # (H,Ds)
        rewards = rewards[0]

        out = {
            "act_seq": act_seq_choice,      # main action sequence to execute
            "state_seq": state_seq,         # resulting state sequence from rollout
            "reward": rewards,
            "iter_reward_max": logs["reward_max"],
            "iter_reward_mean": logs["reward_mean"],
            "iter_elite_reward_mean": logs["elite_reward_mean"],
            # You can also return covT/meanT if you want to inspect the final distribution:
            # "final_mean_flat": meanT, "final_cov": covT,
        }
        return key, out
