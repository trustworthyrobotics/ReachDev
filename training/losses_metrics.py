# trainers/losses_metrics.py
from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable

import jax
import jax.numpy as jnp
import equinox as eqx

import sys
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DTPlanReach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2

Array = jnp.ndarray


# ----------------------------- utilities -----------------------------

def make_linear_step_weights(T: int, ub: float) -> Array:
    """
    Linearly increase per-step weights from 1.0 up to <= ub across T steps.
    Returns shape (T,).
    """
    if T == 1:
        return jnp.array([1.0], dtype=jnp.float32)
    w = 1.0 + (jnp.arange(T, dtype=jnp.float32) * (ub - 1.0) / (T - 1))
    return jnp.minimum(w, jnp.array(ub, dtype=jnp.float32))


# ------------------------------ losses -------------------------------

def multistep_free_rollout_loss(
    model,
    X: Array,
    U: Array,
    *,
    step_weights: Optional[Array] = None,
    transform_fn: Optional[Callable[[Array], Array]] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Multi-step free-rollout MSE with optional per-step weights.

    Args:
      X: (B, T+1, Dx)
      U: (B, T,   Du)
      step_weights: (T,) or None. If provided, normalized by sum(weights).
    Returns:
      loss: scalar
      metrics: dict with mse, rmse and per-step versions
    """
    X_tgt = X[:, 1:, :]
    X_pred = model.rollout(X[:, 0, :], U)           # (B,T,Dx)
    if transform_fn:
        X_pred = transform_fn(X_pred)
        X_tgt = transform_fn(X_tgt)

    err = X_pred - X_tgt                                     # (B,T,Dx)

    # Per-step MSE (averaged over batch and state dims): (T,)
    mse_t = jnp.mean(err**2, axis=(0, 2))

    if step_weights is not None:
        w = step_weights[:X_pred.shape[1]]  # (T,)
        w = w / jnp.sum(w)
        loss = jnp.sum(mse_t * w)
    else:
        loss = jnp.mean(err**2)

    metrics = {
        "mse": jnp.mean(err**2),
        "rmse": jnp.sqrt(jnp.mean(err**2)),
        "mse_per_step": mse_t,
        "rmse_per_step": jnp.sqrt(mse_t + 1e-12),
    }
    return loss, metrics


def combined_loss(
    model,
    X: Array,
    U: Array,
    *,
    step_weights: Optional[Array] = None,
    lam_jac: float = 0.0,
    transform_fn: Optional[Callable[[Array], Array]] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Free-rollout loss + λ * 1-step auxiliary (optional).
    """
    loss, metrics = multistep_free_rollout_loss(model, X, U, step_weights=step_weights, transform_fn=transform_fn)
    metrics.update({"loss": loss})

    jac_loss = jacobian_reg_loss(model, X, U)
    loss = loss + lam_jac * jac_loss
    metrics['jacobian_reg_loss'] = jac_loss
    return loss, metrics

def jacobian_reg_loss(model, X: jnp.ndarray, U: jnp.ndarray):
    """
    Computes the Frobenius norm of the Jacobian df/dx along a trajectory.
    
    Args:
        model: The Equinox model
        X: (B, T+1, Dx) states
        U: (B, T, Du) actions
    """
    # 2. Use vmap to compute Jacobians over the batch and time
    # This gives us (B, T, Dx, Dx) tensor
    batch_jac_fn = jax.vmap(jax.vmap(jax.jacobian(model.forward_batchless, argnums=0)))
    
    # We only need the first T states to predict the next T states
    jacobians = batch_jac_fn(X[:, :-1, :], U)
    
    # 3. Calculate Frobenius Norm: sqrt(sum of squares of all elements)
    # We often use the squared Frobenius norm for easier optimization
    jac_loss = jnp.mean(jnp.square(jacobians))
    
    return jac_loss

def l1_reg_loss(model):
    params = eqx.filter(model, eqx.is_inexact_array)
    def leaf_l1(acc, leaf):
            return acc + jnp.sum(jnp.abs(leaf))
    total_l1 = eqx.tree_reduce(leaf_l1, params, 0.0)
    return total_l1


class MSELoss(eqx.Module):
    """Handles multi-step rollout prediction error."""
    def __call__(self, model, X, U, step_weights):
        # Initial state x0
        x_tgt = X[:, 1:, :]
        # model.rollout handles DT vs CT (ODE) internally
        x_pred = model.rollout(X[:, 0, :], U) 
        
        x_pred = model.transform_fn(x_pred)
        x_tgt = model.transform_fn(x_tgt)

        err_sq = (x_pred - x_tgt)**2
        mse_t = jnp.mean(err_sq, axis=(0, 2)) # Average over Batch and Dims

        w = step_weights[:x_pred.shape[1]]
        w = w / jnp.sum(w)
        loss = jnp.sum(mse_t * w)

        return loss, {"mse": loss, "mse_per_step": mse_t}

class JacobianReg(eqx.Module):
    """Penalizes high sensitivity to stabilize reachability sets."""
    def __call__(self, model, X, U):
        # vmap over batch and time to compute df/dx
        # Note: argnums=0 assumes model.forward(x, u)
        batch_jac_fn = jax.vmap(jax.vmap(jax.jacobian(model.forward_batchless, argnums=0)))
        jacobians = batch_jac_fn(X[:, :-1, :], U)
        loss = jnp.mean(jnp.square(jacobians))
        return loss, {"jac_reg": loss}

class ReachabilityPenalty(eqx.Module):
    """Calculates the volume of the reachable set."""
    mode: str = eqx.field(static=True)
    reach_analyzer: any # Your DTPlanReach or a CT equivalent
    
    def __init__(self, mode, state_dim, action_dim):
        self.mode = mode
        if mode == 'dt_dyn':
            self.reach_analyzer = DTPlanReach(None, state_dim=state_dim, action_dim=action_dim, nn_dyn=True, n_steps_per_plan=1, step_size=1)
        elif mode == 'ct_dyn':
            raise NotImplementedError("CT dynamics reachability not implemented yet.")
        elif mode == 'ct_ctl':
            raise NotImplementedError("CT control reachability not implemented yet.")
        else:
            raise ValueError(f"Unknown mode for ReachabilityPenalty: {mode}")

    def __call__(self, model, X, U, eps, reach_bs, key, splits_cfg):
        T_reach = U.shape[1]
        D = model.Dx + model.Du
        # pick a random subset for reachability
        if X.shape[0] > reach_bs:
            perm = jax.random.permutation(key, X.shape[0])
            idxs = perm[:reach_bs]
            X = X[idxs]
            U = U[idxs]
        state_init = X[:, 0, :]
        state_lo = state_init - eps
        state_up = state_init + eps

        if self.mode == 'dt_dyn':
            def f_wrapper(x):
                state_next = model(x)
                action_next = x[model.Dx:]
                return jnp.concatenate([state_next, action_next], axis=-1)
        elif self.mode == 'ct_dyn':
            raise NotImplementedError("CT dynamics reachability not implemented yet.")
        elif self.mode == 'ct_ctl':
            raise NotImplementedError("CT control reachability not implemented yet.")
        else:
            raise ValueError(f"Unknown mode for ReachabilityPenalty: {self.mode}")
        X_lo = jnp.concatenate([state_lo, jnp.zeros_like(U[:, 0, :])], axis=-1)
        X_up = jnp.concatenate([state_up, jnp.zeros_like(U[:, 0, :])], axis=-1)
        X_lo, X_up = prepare_initial_set_v2(X_lo, X_up, splits_cfg=splits_cfg)
        _, r_lo, r_up, _ = self.reach_analyzer.verify_w_model(f_wrapper, X_lo, X_up, n_total_steps=T_reach, action_seq=U.repeat(X_up.shape[0]//U.shape[0], axis=0)[:, None])

        reach_vol = calculate_volume(r_lo.reshape(-1, T_reach + 1, D), r_up.reshape(-1, T_reach + 1, D), union_init=False, mode='sum') / r_lo.shape[0]
        reach_penalty = jnp.log(1 + reach_vol)
        return reach_penalty, {"reach_volume": reach_vol, "reach_penalty": reach_penalty}
    

class TotalLoss(eqx.Module):
    mse_layer: MSELoss
    jac_layer: JacobianReg
    reach_layer: Optional[ReachabilityPenalty]
    
    # Weights and Configs
    lam_jac: float
    lam_reach: float
    reach_splits: Dict

    def __init__(self, mode: str, state_dim: int, action_dim: int, reach_cfg: dict, lam_jac: float = 0.0,):
        
        self.mse_layer = MSELoss()
        self.jac_layer = JacobianReg()
        if reach_cfg.get("mode", "none") != "none":
            self.reach_layer = ReachabilityPenalty(mode, state_dim, action_dim)
        else:
            self.reach_layer = None
        self.lam_jac = lam_jac
        self.lam_reach = reach_cfg.get("weight", 0.0)
        self.reach_splits = reach_cfg.get("splits", {})

    def __call__(self, model, X, U, enable_reach, key, step_weights, reach_eps=0.0, reach_bs=32):
        
        # 1. Always compute MSE and Jacobian (Standard JAX code)
        l_mse, m_mse = self.mse_layer(model, X, U, step_weights=step_weights)
        l_jac, m_jac = self.jac_layer(model, X, U)
        
        total_loss = l_mse + (self.lam_jac * l_jac)
        metrics = {**m_mse, **m_jac}

        if enable_reach:
            l_reach, m_reach = self.reach_layer(
                model, X, U, reach_eps, reach_bs,
                key, self.reach_splits
            )
            total_loss = total_loss + (self.lam_reach * l_reach)
            metrics.update(m_reach)
        return total_loss, metrics
