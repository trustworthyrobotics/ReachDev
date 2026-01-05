# trainers/losses_metrics.py
from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable, Union

import jax
import jax.numpy as jnp
import equinox as eqx

import sys
sys.path.append('CROWN_Reach')
from CROWN_Reach.src.reachability import DT_Plan_Reach, CT_Plan_Reach, CT_Track_Ctl_Reach
from CROWN_Reach.src.utils.box_set import calculate_volume, prepare_initial_set_v2

Array = jnp.ndarray

from models.dt_dyn import T_Dynamics
from models.ct_dyn import Continuous_T_Dynamics
from models.ct_ctl import T_controller

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
class MSELoss(eqx.Module):
    """Handles multi-step rollout prediction error."""
    def __call__(self, model: Union[T_Dynamics, Continuous_T_Dynamics], X, U, step_weights):
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
    def __call__(self, model: Union[T_Dynamics, Continuous_T_Dynamics], X, U):
        # vmap over batch and time to compute df/dx
        # Note: argnums=0 assumes model.forward(x, u)
        batch_jac_fn = jax.vmap(jax.vmap(jax.jacobian(model.forward_batchless_mlp_only, argnums=0)))
        jacobians = batch_jac_fn(X[:, :-1, :], U)
        loss = jnp.mean(jnp.square(jacobians))
        return loss, {"jac_reg": loss}

class ReachabilityPenalty(eqx.Module):
    """Calculates the volume of the reachable set."""
    mode: str = eqx.field(static=True)
    reach_analyzer: Union[DT_Plan_Reach, CT_Plan_Reach, CT_Track_Ctl_Reach] = eqx.field(static=True)
    
    def __init__(self, mode, state_dim, action_dim, reach_cfg, frequency):
        self.mode = mode
        if mode == 'dt_dyn':
            self.reach_analyzer = DT_Plan_Reach(None, state_dim=state_dim, action_dim=action_dim, nn_dyn=True, n_steps_per_plan=1, step_size=int(1/frequency))
        elif mode == 'ct_dyn':
            self.reach_analyzer = CT_Plan_Reach(None, state_dim=state_dim, action_dim=action_dim, nn_dyn=True, n_steps_per_plan=1, step_size=1/frequency, init_remainder=reach_cfg.get("init_remainder", 1e-1), frr_rounds=reach_cfg.get("frr_rounds", 5), frr_stop_ratio=reach_cfg.get("frr_stop_ratio", 0.95), sr_window_size=reach_cfg.get("sr_window_size", 100))
        elif mode == 'ct_ctl':
            raise NotImplementedError("CT control reachability not implemented yet.")
        else:
            raise ValueError(f"Unknown mode for ReachabilityPenalty: {mode}")

    def __call__(self, model: Union[T_Dynamics, Continuous_T_Dynamics], X, U, eps, reach_bs, key, splits_cfg):
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
            def f_wrapper(x):
                dx = model(x)
                du = jnp.zeros_like(x[model.Dx:])
                return jnp.concatenate([dx, du], axis=-1)
        elif self.mode == 'ct_ctl':
            raise NotImplementedError("CT control reachability not implemented yet.")
        else:
            raise ValueError(f"Unknown mode for ReachabilityPenalty: {self.mode}")
        X_lo = jnp.concatenate([state_lo, jnp.zeros_like(U[:, 0, :])], axis=-1)
        X_up = jnp.concatenate([state_up, jnp.zeros_like(U[:, 0, :])], axis=-1)
        X_lo, X_up = prepare_initial_set_v2(X_lo, X_up, splits_cfg=splits_cfg)
        _, r_lo, r_up, _, _ = self.reach_analyzer.verify_w_model(f_wrapper, X_lo, X_up, n_total_steps=T_reach, action_seq=U.repeat(X_up.shape[0]//U.shape[0], axis=0)[:, None])

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

    def __init__(self, mode: str, state_dim: int, action_dim: int, reach_cfg: dict, frequency: float, lam_jac: float, *args, **kwargs):
        self.mse_layer = MSELoss()
        self.jac_layer = JacobianReg()
        if reach_cfg.get("mode", "none") != "none":
            self.reach_layer = ReachabilityPenalty(mode, state_dim, action_dim, reach_cfg, frequency)
        else:
            self.reach_layer = None
        self.lam_jac = lam_jac
        self.lam_reach = reach_cfg.get("weight", 0.0)
        self.reach_splits = reach_cfg.get("splits", {})

    def __call__(self, model: Union[T_Dynamics, Continuous_T_Dynamics], X, U, enable_reach, key, step_weights, reach_eps=0.0, reach_bs=32):
        
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

class MSELossCtl(eqx.Module):
    ct_dyn: Continuous_T_Dynamics = eqx.field(static=True)

    """Handles multi-step rollout control error."""
    def __call__(self, model: T_controller, X, U, step_weights):
        # X: (B,T+1,Dx), U: (B,T,Du)
        X_curr = X[:, 0, :] # [B, Dx]
        X_tgt = X[:, -1, :] # [B, Dx]
        U_ref = U.mean(axis=1)  # [B, Du]

        def one_step_ctl_dyn(carry, _):
            X_curr = carry  # [B, Dx]
            U_pred = model.forward(X_curr, X_tgt, U_ref)  # [B, Du]
            X_next = self.ct_dyn.forward(X_curr, U_pred)  # [B, Dx]
            return X_next, (X_next, U_pred)

        T = U.shape[1]
        _, (X_preds, U_preds) = jax.lax.scan(one_step_ctl_dyn, X_curr, None, length=T)
        X_preds = X_preds.transpose(1,0,2)  # [B, T, Dx]
        U_preds = U_preds.transpose(1,0,2)  # [B, T, Du]

        # Average over Batch and Dims
        # mse_t = jnp.mean((U_preds - U)**2, axis=(0, 2)) + jnp.mean((X_preds - X[:, 1:, :])**2, axis=(0, 2))
        mse_t = jnp.mean((X_preds - X[:, 1:, :])**2, axis=(0, 2))

        w = step_weights[:T]
        w = w / jnp.sum(w)
        loss = jnp.sum(mse_t * w)

        return loss, {"mse": loss, "mse_per_step": mse_t}


class TotalLossCtl(eqx.Module):
    mse_layer: MSELossCtl
    jac_layer: JacobianReg
    reach_layer: Optional[ReachabilityPenalty]
    
    # Weights and Configs
    lam_jac: float
    lam_reach: float
    reach_splits: Dict

    def __init__(self, mode: str, state_dim: int, action_dim: int, reach_cfg: dict, lam_jac: float, ct_dyn: Continuous_T_Dynamics):
        self.mse_layer = MSELossCtl(ct_dyn)
        self.jac_layer = JacobianReg()
        if reach_cfg.get("mode", "none") != "none":
            self.reach_layer = ReachabilityPenalty(mode, state_dim, action_dim)
        else:
            self.reach_layer = None
        self.lam_jac = lam_jac
        self.lam_reach = reach_cfg.get("weight", 0.0)
        self.reach_splits = reach_cfg.get("splits", {})

    def __call__(self, model: T_controller, X, U, enable_reach, key, step_weights, reach_eps=0.0, reach_bs=32):
        
        # 1. Always compute MSE and Jacobian (Standard JAX code)
        l_mse, m_mse = self.mse_layer(model, X, U, step_weights=step_weights)
        # l_jac, m_jac = self.jac_layer(model, X, U)
        
        # total_loss = l_mse + (self.lam_jac * l_jac)
        # metrics = {**m_mse, **m_jac}

        total_loss = l_mse
        metrics = {**m_mse}

        if enable_reach:
            l_reach, m_reach = self.reach_layer(
                model, X, U, reach_eps, reach_bs,
                key, self.reach_splits
            )
            total_loss = total_loss + (self.lam_reach * l_reach)
            metrics.update(m_reach)
        return total_loss, metrics
