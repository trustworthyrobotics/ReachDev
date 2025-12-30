import jax
import jax.numpy as jnp


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