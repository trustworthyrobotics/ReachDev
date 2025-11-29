# trainers/losses_metrics.py
from __future__ import annotations
from typing import Optional, Dict, Tuple

import jax
import jax.numpy as jnp


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


def mse(x: Array, y: Array, axis=None) -> Array:
    return jnp.mean((x - y) ** 2, axis=axis)


def rmse(x: Array, y: Array, axis=None) -> Array:
    return jnp.sqrt(mse(x, y, axis=axis))


# ---------------------------- core rollouts ---------------------------

def free_rollout(model, X: Array, U: Array, T: Optional[int] = None) -> Array:
    """
    Autoregressive rollout using model.rollout_model.

    Args:
      X: (B, T+1, Dx)  ground-truth states
      U: (B, T,   Du)  actions
      T: optional override horizon (<= U.shape[1])
    Returns:
      X_pred: (B, T, Dx) predicted states x_{1:T}
      X_tgt : (B, T, Dx) target states   x_{1:T}
    """
    T = int(T or (U.shape[1]))
    X_tgt = X[:, 1:T + 1, :]
    X_pred = model.rollout_model(X[:, 0, :], U[:, :T, :], T=T)
    return X_pred, X_tgt


# ------------------------------ losses -------------------------------

def multistep_free_rollout_loss(
    model,
    X: Array,
    U: Array,
    *,
    T: Optional[int] = None,
    step_weights: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Multi-step free-rollout MSE with optional per-step weights.

    Args:
      X: (B, T+1, Dx)
      U: (B, T,   Du)
      T: horizon used for loss (<= U.shape[1])
      step_weights: (T,) or None. If provided, normalized by sum(weights).
    Returns:
      loss: scalar
      metrics: dict with mse, rmse and per-step versions
    """
    X_pred, X_tgt = free_rollout(model, X, U, T=T)           # (B,T,Dx)
    err = X_pred - X_tgt                                     # (B,T,Dx)

    # Per-step MSE (averaged over batch and state dims): (T,)
    mse_t = jnp.mean(err**2, axis=(0, 2))

    if step_weights is not None:
        w = step_weights.astype(jnp.float32)
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


def onestep_teacher_forcing_loss(
    model,
    X: Array,
    U: Array,
    T: Optional[int] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Teacher-forcing 1-step loss over a window:
      predict x_{t+1} from x_t, u_t for t in [0..T-1].

    Args:
      X: (B, T+1, Dx)
      U: (B, T,   Du)
      T: optional horizon (<= U.shape[1])
    """
    T = int(T or U.shape[1])
    x_t = X[:, :T, :]             # (B,T,Dx)
    u_t = U[:, :T, :]             # (B,T,Du)
    x_tp1 = X[:, 1:T + 1, :]      # (B,T,Dx)

    # Flatten time into batch, run in one call, then unflatten.
    B, TT, Dx = x_t.shape
    x_pred = model.forward(x_t.reshape(B * TT, Dx), u_t.reshape(B * TT, -1))
    x_pred = x_pred.reshape(B, TT, Dx)

    err = x_pred - x_tp1
    loss = jnp.mean(err**2)
    metrics = {
        "mse_1step": jnp.mean(err**2),
        "rmse_1step": jnp.sqrt(jnp.mean(err**2)),
    }
    return loss, metrics


def combined_loss(
    model,
    X: Array,
    U: Array,
    *,
    T: Optional[int] = None,
    step_weights: Optional[Array] = None,
    aux_weight: float = 0.0,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Free-rollout loss + λ * 1-step auxiliary (optional).
    """
    loss_free, m_free = multistep_free_rollout_loss(model, X, U, T=T, step_weights=step_weights)
    if aux_weight > 0.0:
        loss_1s, m_1s = onestep_teacher_forcing_loss(model, X, U, T=T)
        loss = loss_free + aux_weight * loss_1s
        metrics = {**m_free, **m_1s, "loss": loss, "loss_free": loss_free, "loss_1step": loss_1s}
    else:
        loss = loss_free
        metrics = {**m_free, "loss": loss}
    return loss, metrics


# ------------------------------ metrics ------------------------------

def compute_rollout_metrics(
    X_pred: Array,
    X_tgt: Array,
    *,
    x_std: Optional[Array] = None,
) -> Dict[str, Array]:
    """
    Extra metrics on a finished rollout.
    Args:
      X_pred, X_tgt: (B,T,Dx)
      x_std: (Dx,) optional; if provided, report NRMSE using it.
    """
    err = X_pred - X_tgt
    mse_all = jnp.mean(err**2)
    rmse_all = jnp.sqrt(mse_all)

    mse_t = jnp.mean(err**2, axis=(0, 2))          # (T,)
    rmse_t = jnp.sqrt(mse_t + 1e-12)

    out = {
        "mse": mse_all,
        "rmse": rmse_all,
        "mse_per_step": mse_t,
        "rmse_per_step": rmse_t,
    }

    if x_std is not None:
        # Normalize per-state, then aggregate.
        norm_err = err / (x_std[None, None, :] + 1e-8)
        nrmse_all = jnp.sqrt(jnp.mean(norm_err**2))
        out["nrmse"] = nrmse_all

    return out
