import numpy as np
import jax
import jax.numpy as jnp

def box_corners_nd_jax(lo, hi):
    """
    lo, hi: jnp.ndarray of shape [..., N]
    return: jnp.ndarray of shape [2**N, ..., N]
    """
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)
    assert lo.shape == hi.shape
    N = lo.shape[-1]
    K = lo.ndim - 1              # number of batch dims in "..."
    M = 1 << N                   # 2**N
    # All binary combinations (2^N, N)
    bits = ((jnp.arange(2**N)[:, None] & (1 << jnp.arange(N)[::-1])) > 0).astype(int)

    # Broadcast
    lo = lo[None, ...]        # (1, ..., N)
    hi = hi[None, ...]        # (1, ..., N)
    # reshape bits to (M, 1, 1, ..., 1, N) so it broadcasts over lo/hi batch dims
    bits = bits.reshape((M,) + (1,) * K + (N,))

    corners = lo * (1 - bits) + hi * bits
    return corners

def box_corners_nd(lo, hi):
    """
    lo, hi: np.ndarray of shape [..., N]
    return: np.ndarray of shape [2**N, ..., N]
    """
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    assert lo.shape == hi.shape
    N = lo.shape[-1]
    K = lo.ndim - 1              # number of batch dims in "..."
    M = 1 << N                   # 2**N
    # All binary combinations (2^N, N)
    bits = ((np.arange(2**N)[:, None] & (1 << np.arange(N)[::-1])) > 0).astype(int)

    # Broadcast
    lo = lo[None, ...]        # (1, ..., N)
    hi = hi[None, ...]        # (1, ..., N)
    # reshape bits to (M, 1, 1, ..., 1, N) so it broadcasts over lo/hi batch dims
    bits = bits.reshape((M,) + (1,) * K + (N,))

    corners = lo * (1 - bits) + hi * bits
    return corners

def random_sample_nd(lo, hi, num_samples, seed=0):
    """
    lo, hi: np.ndarray of shape [..., N]
    return: np.ndarray of shape [num_samples, ..., N]
    """
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    assert lo.shape == hi.shape
    N = lo.shape[-1]
    K = lo.ndim - 1              # number of batch dims in "..."
    
    # Broadcast
    lo = lo[None, ...]        # (1, ..., N)
    hi = hi[None, ...]        # (1, ..., N)

    rng = np.random.default_rng(seed)
    samples = rng.uniform(size=(num_samples,) + (1,) * K + (N,))  # (num_samples, 1, 1, ..., 1, N)
    samples = lo + (hi - lo) * samples
    return samples