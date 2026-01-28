import jax
import jax.numpy as jnp
import numpy as np

def pose_to_kp(pose, stem_size, bar_size):
    """
    Computes keypoints for a T-shape object in JAX.
    pose: [3] (x, y, theta)
    stem_size: [width, height]
    bar_size: [width, height]
    Returns: [4, 2]
    """
    w_s, h_s = stem_size
    w_b, h_b = bar_size
    
    # 1. Calculate Center of Mass (CoM) relative to stem bottom center (0,0)
    y_s = h_s / 2.0
    y_b = h_s + h_b / 2.0
    m_s, m_b = w_s * h_s, w_b * h_b
    y_m = (m_s * y_s + m_b * y_b) / (m_s + m_b)
    
    # 2. Define keypoint offsets relative to CoM
    # Order: Left Bar, Top Center Bar, Right Bar, Bottom Stem
    offsets = jnp.array([
        [-w_b / 2.0, y_b - y_m],
        [0.0,        y_b - y_m],
        [w_b / 2.0,  y_b - y_m],
        [0.0,        0.0 - y_m]
    ])  # Shape: [4, 2]

    # 3. Extract pose components
    pos = pose[:2]      # [2]
    angle = pose[2]     # [] (scalar)
    # 4. Batched Rotation Matrix
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    
    # Construct rotation matrices: [2, 2]
    rot_mats = jnp.stack([
        jnp.stack([cos_a, -sin_a], axis=-1),
        jnp.stack([sin_a, cos_a], axis=-1)
    ], axis=-2)

    # 5. Rotate and Translate
    # We use jnp.einsum for a clean batched matrix multiplication: [2, 2] x [4, 2]^T
    rotated_offsets = jnp.einsum('ij,kj->ki', rot_mats, offsets)
    
    # Add the position (pos is [2], adds to [4, 2] via broadcasting)
    keypoints = rotated_offsets + pos[jnp.newaxis, :]
    
    return keypoints.reshape(-1)  # Return as [8,]

from typing import Tuple
def _obb_axes(angle: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return OBB unit axes (u, v) given angle (radians)."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    u = jnp.stack([c, s], axis=-1)      # (...,2)
    v = jnp.stack([-s, c], axis=-1)     # (...,2)  (u rotated by +90deg)
    return u, v

def aabb_vs_obb_sat_2d(
    c_aabb: jnp.ndarray,  # (...,2)
    h_aabb: jnp.ndarray,  # (...,2)
    c_obb: jnp.ndarray,   # (...,2) broadcastable with c_aabb
    h_obb: jnp.ndarray,   # (...,2) broadcastable with c_aabb
    angle: jnp.ndarray,   # (...)   broadcastable with c_aabb
    eps: float = 0.0,     # optional tolerance
) -> jnp.ndarray:
    """
    2D SAT test: AABB vs OBB. Returns bool array (...) for intersection.
    Touching counts as intersection unless you set eps<0 or change comparisons.
    """
    u, v = _obb_axes(angle)             # (...,2), (...,2)
    d = c_obb - c_aabb                  # (...,2)
    dx, dy = d[..., 0], d[..., 1]

    hAx, hAy = h_aabb[..., 0], h_aabb[..., 1]
    hBx, hBy = h_obb[..., 0], h_obb[..., 1]

    # Axis L = ex = (1,0)
    # d = |dx|
    # rA = hAx
    # rB = hBx*|u_x| + hBy*|v_x|
    sep_ex = jnp.abs(dx) > (hAx + hBx * jnp.abs(u[..., 0]) + hBy * jnp.abs(v[..., 0]) + eps)

    # Axis L = ey = (0,1)
    sep_ey = jnp.abs(dy) > (hAy + hBx * jnp.abs(u[..., 1]) + hBy * jnp.abs(v[..., 1]) + eps)

    # Axis L = u
    du = jnp.abs(dx * u[..., 0] + dy * u[..., 1])
    rA_u = hAx * jnp.abs(u[..., 0]) + hAy * jnp.abs(u[..., 1])
    # since u,v are orthonormal from trig:
    rB_u = hBx
    sep_u = du > (rA_u + rB_u + eps)

    # Axis L = v
    dv = jnp.abs(dx * v[..., 0] + dy * v[..., 1])
    rA_v = hAx * jnp.abs(v[..., 0]) + hAy * jnp.abs(v[..., 1])
    rB_v = hBy
    sep_v = dv > (rA_v + rB_v + eps)

    separated = sep_ex | sep_ey | sep_u | sep_v
    return ~separated

def detect_T_hole_interaction(
    c_wall: jnp.ndarray,   # (2,2)
    h_wall: jnp.ndarray,   # (2,2)
    c_T: jnp.ndarray,      # (n,2,2)  parts: 0=stem, 1=bar (your convention)
    h_T: jnp.ndarray,      # (2,2)    per-part half extents (shared across all n)
    angle_T: jnp.ndarray,  # (n,2)    per-part angles
    eps: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      interact: (n,) bool
        True if ANY (part in {stem,bar}) intersects ANY wall AABB.

      interact_part_wall: (n,2,2) bool
        interact_part_wall[n, part, wall] is True if that part intersects that wall.
    """
    # Broadcast to (wall, n, part, ...)
    cA = c_wall[:, None, None, :]         # (2,1,1,2)
    hA = h_wall[:, None, None, :]         # (2,1,1,2)
    cB = c_T[None, :, :, :]               # (1,n,2,2)
    hB = h_T[None, None, :, :]            # (1,1,2,2)
    ang = angle_T[None, :, :]             # (1,n,2)

    inter_w_n_p = aabb_vs_obb_sat_2d(cA, hA, cB, hB, ang, eps=eps)  # (2,n,2)

    # (n,2,2): part-wall
    interact_part_wall = jnp.transpose(inter_w_n_p, (1, 2, 0))

    # (n,): any wall & any part
    interact = jnp.any(inter_w_n_p, axis=(0, 2))
    return interact, interact_part_wall

def hole_to_walls_aabbs(c_hole, s_hole, window_size, tol=1e-6):
    """
    Convert a boundary hole spec into two wall AABBs (centers + half-extents).

    Inputs:
      c_hole: (2,) array-like, hole center ON the boundary.
      s_hole: (2,) array-like, [width_along_boundary, height_into_workspace].
      window_size: scalar, workspace is [0,window_size]x[0,window_size].

    Returns:
      c_wall: (2,2) float, centers of two AABBs
      h_wall: (2,2) float, half extents of two AABBs

    Conventions:
      - If hole is on top/bottom edge (y=0 or y=window_size):
          width is along x, height is along y inward.
      - If hole is on left/right edge (x=0 or x=window_size):
          width is along y, height is along x inward.
    """
    c_hole = np.asarray(c_hole, dtype=float).reshape(2,)
    s_hole = np.asarray(s_hole, dtype=float).reshape(2,)
    W = float(window_size)

    assert W > 0.0
    assert s_hole[0] > 0.0 and s_hole[1] > 0.0, "Hole width/height must be positive."

    x, y = c_hole
    w, h = s_hole  # width along boundary, height into workspace

    def near(a, b):
        return abs(a - b) <= tol

    on_left  = near(x, 0.0)
    on_right = near(x, W)
    on_bot   = near(y, 0.0)
    on_top   = near(y, W)

    # Must be on exactly one boundary line (not interior).
    assert (on_left or on_right or on_bot or on_top), "c_hole must lie on a workspace boundary."
    # Disallow ambiguous corner holes (both x boundary and y boundary).
    assert not ((on_left or on_right) and (on_bot or on_top)), "c_hole at a corner is ambiguous."

    walls = []  # each wall as (xmin, xmax, ymin, ymax)

    if on_top or on_bot:
        # Hole on horizontal edge: width along x, height along y inward.
        x_left  = x - 0.5 * w
        x_right = x + 0.5 * w

        assert 0.0 < x_left < x_right < W, "Hole must be strictly inside boundary span so two walls exist."
        assert h <= W + tol, "Hole height cannot exceed window size."

        if on_top:
            y0, y1 = W - h, W
        else:  # on_bot
            y0, y1 = 0.0, h

        assert -tol <= y0 <= y1 <= W + tol

        # Left wall: x in [0, x_left], y in [y0, y1]
        walls.append((0.0, x_left, y0, y1))
        # Right wall: x in [x_right, W], y in [y0, y1]
        walls.append((x_right, W, y0, y1))

    else:
        # Hole on vertical edge: width along y, height along x inward.
        y_bot = y - 0.5 * w
        y_top = y + 0.5 * w

        assert 0.0 < y_bot < y_top < W, "Hole must be strictly inside boundary span so two walls exist."
        assert h <= W + tol, "Hole height cannot exceed window size."

        if on_right:
            x0, x1 = W - h, W
        else:  # on_left
            x0, x1 = 0.0, h

        assert -tol <= x0 <= x1 <= W + tol

        # Bottom wall segment: y in [0, y_bot], x in [x0, x1]
        walls.append((x0, x1, 0.0, y_bot))
        # Top wall segment: y in [y_top, W], x in [x0, x1]
        walls.append((x0, x1, y_top, W))

    # Convert (xmin,xmax,ymin,ymax) -> center + half extents
    c_wall = np.zeros((2, 2), dtype=float)
    h_wall = np.zeros((2, 2), dtype=float)
    for i, (xmin, xmax, ymin, ymax) in enumerate(walls):
        assert xmax >= xmin and ymax >= ymin
        c_wall[i] = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=float)
        h_wall[i] = np.array([(xmax - xmin) * 0.5, (ymax - ymin) * 0.5], dtype=float)

    return c_wall, h_wall


import jax.numpy as jnp
from typing import Tuple

_PI = jnp.pi
_TWOPI = 2.0 * jnp.pi

def _interval_contains_k(a: jnp.ndarray, b: jnp.ndarray, shift: float, period: float) -> jnp.ndarray:
    """
    Returns True if there exists integer k such that (shift + k*period) in [a,b].
    Works elementwise for arrays.
    Assumes b >= a.
    """
    k_min = jnp.ceil((a - shift) / period)
    k_max = jnp.floor((b - shift) / period)
    return k_min <= k_max

def _max_abs_sin_cos(theta_lo: jnp.ndarray, theta_hi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute exact maxima of |sin(theta)| and |cos(theta)| over theta in [lo, hi],
    for elementwise lo/hi. Assumes theta_hi >= theta_lo.

    If interval length >= 2pi, both maxima are 1.
    Otherwise:
      max |sin| = 1 if interval contains (pi/2 + k*pi), else max(|sin(lo)|,|sin(hi)|)
      max |cos| = 1 if interval contains (0 + k*pi),    else max(|cos(lo)|,|cos(hi)|)
    """
    lo = theta_lo
    hi = theta_hi
    length = hi - lo

    # If spans full period, maxima are 1
    full = length >= _TWOPI

    # sin hits ±1 at pi/2 + k*pi
    sin_hits_1 = _interval_contains_k(lo, hi, shift=float(_PI / 2.0), period=float(_PI))
    max_abs_sin_end = jnp.maximum(jnp.abs(jnp.sin(lo)), jnp.abs(jnp.sin(hi)))
    max_abs_sin = jnp.where(full | sin_hits_1, 1.0, max_abs_sin_end)

    # cos hits ±1 at k*pi
    cos_hits_1 = _interval_contains_k(lo, hi, shift=0.0, period=float(_PI))
    max_abs_cos_end = jnp.maximum(jnp.abs(jnp.cos(lo)), jnp.abs(jnp.cos(hi)))
    max_abs_cos = jnp.where(full | cos_hits_1, 1.0, max_abs_cos_end)

    return max_abs_sin, max_abs_cos

def _aabb_intersect(
    cA: jnp.ndarray, hA: jnp.ndarray,
    cB: jnp.ndarray, hB: jnp.ndarray,
    eps: float = 0.0
) -> jnp.ndarray:
    """
    AABB-AABB intersection test in 2D.
    Touching counts as intersection; eps>0 makes it more conservative.
    Shapes broadcast.
    """
    d = jnp.abs(cB - cA)  # (...,2)
    return (d[..., 0] <= (hA[..., 0] + hB[..., 0] + eps)) & (d[..., 1] <= (hA[..., 1] + hB[..., 1] + eps))

def detect_T_hole_interaction_set(
    c_wall: jnp.ndarray,      # (2,2)
    h_wall: jnp.ndarray,      # (2,2)
    c_T_lo: jnp.ndarray,      # (n,2,2)
    c_T_hi: jnp.ndarray,      # (n,2,2)
    h_T: jnp.ndarray,         # (2,2)  per-part half extents (shared)
    angle_T_lo: jnp.ndarray,  # (n,2)
    angle_T_hi: jnp.ndarray,  # (n,2)
    eps: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Guaranteed-safe set-based check.

    Returns:
      safe: (n,) bool
        True iff BOTH parts (stem+bar) are guaranteed NOT to intersect EITHER wall
        for ANY pose in the provided intervals.

      safe_part_wall: (n,2,2) bool
        safe_part_wall[n, part, wall] is True iff that part's pose-set is
        guaranteed NOT to intersect that wall.
    """
    # Basic sanity (JAX-friendly: these are runtime asserts only if you enable them)
    # Expect lo <= hi elementwise for the interval meaning used here.
    # If you have wrap-around angle intervals, you need to pre-normalize/split them.
    # (Keeping it simple + sound for the typical small-angle-uncertainty case.)
    # You can remove these in production.
    # Note: jnp.all(...) is traced under jit; keep outside jit or use chex if needed.
    # assert bool(jnp.all(c_T_hi >= c_T_lo)), "Expect c_T_lo <= c_T_hi"
    # assert bool(jnp.all(angle_T_hi >= angle_T_lo)), "Expect angle_T_lo <= angle_T_hi"

    n = c_T_lo.shape[0]

    # Over-approx each uncertain OBB part (n,2) -> an AABB (center+half)
    c_mid = 0.5 * (c_T_lo + c_T_hi)          # (n,2,2)
    c_rad = 0.5 * (c_T_hi - c_T_lo)          # (n,2,2) center uncertainty radius

    # Angle-dependent max axis-aligned half extents for rotated rectangle:
    # ex(theta)=|cos|*hBx + |sin|*hBy, ey(theta)=|sin|*hBx + |cos|*hBy
    max_abs_sin, max_abs_cos = _max_abs_sin_cos(angle_T_lo, angle_T_hi)  # (n,2), (n,2)

    hBx = h_T[None, :, 0]  # (1,2)
    hBy = h_T[None, :, 1]  # (1,2)

    ex_max = hBx * max_abs_cos + hBy * max_abs_sin   # (n,2)
    ey_max = hBx * max_abs_sin + hBy * max_abs_cos   # (n,2)

    # Total AABB half extents = center-uncertainty radius + rotation envelope
    h_set = jnp.stack([c_rad[..., 0] + ex_max, c_rad[..., 1] + ey_max], axis=-1)  # (n,2,2)

    # Now test AABB(set) vs wall AABB for each wall, n, part.
    cA = c_wall[:, None, None, :]   # (2,1,1,2)
    hA = h_wall[:, None, None, :]   # (2,1,1,2)
    cB = c_mid[None, :, :, :]       # (1,n,2,2)
    hB = h_set[None, :, :, :]       # (1,n,2,2)

    inter_w_n_p = _aabb_intersect(cA, hA, cB, hB, eps=eps)  # (2,n,2)

    # interact_part_wall: (n,2,2) with axis order (n, part, wall), possibly interacted
    interact_part_wall = jnp.transpose(inter_w_n_p, (1, 2, 0))

    # interact: (n,) True if ANY wall & ANY part possibly intersects
    interact = jnp.any(inter_w_n_p, axis=(0, 2))  # (n,)
    return interact, interact_part_wall
