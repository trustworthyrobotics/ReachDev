import numpy as np
import random
from scipy.stats import truncnorm

"""
helper functions for parsing object geometry
"""

def transform_polys_wrt_pose_2d(poly_list, pose):
    # poly_list: list of 2D polygons, each represented by a list of vertices
    x, y, angle = pose
    translation_vector = np.array([x, y])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed_poly_list = []
    for vertices in poly_list:
        transformed_vertices = np.dot(vertices, rotation_matrix.T) + translation_vector
        transformed_poly_list.append(transformed_vertices)

    return transformed_poly_list


def get_rect_vertices(w, h):
    w /= 2
    h /= 2
    return np.array([[-w, -h], [w, -h], [w, h], [-w, h]])


def calculate_com(x_i, y_i, m_i):
    """
    Calculate the center of mass (CoM) for a composite object based on
    the masses (or areas) and coordinates of individual parts.

    Parameters:
    - x_i: List or array of x-coordinates of the centers of mass of the components.
    - y_i: List or array of y-coordinates of the centers of mass of the components.
    - m_i: List or array of masses (or areas) of the components.

    Returns:
    - (C_x, C_y): A tuple representing the x and y coordinates of the composite CoG.
    """
    total_mass = sum(m_i)
    C_x = sum(m * x for m, x in zip(m_i, x_i)) / total_mass
    C_y = sum(m * y for m, y in zip(m_i, y_i)) / total_mass
    return (C_x, C_y)

def keypoints_to_pose_2d_SVD(kp1, kp2):
    """
    Estimate 2D rigid transform (x, y, theta) aligning kp1 -> kp2 via batched Kabsch (SVD).

    Parameters
    ----------
    kp1 : array-like, shape (n, 2)
        Source keypoints (shared for the whole batch).
    kp2 : array-like, shape (n, 2) or (batch_size, n, 2)
        Target keypoints. If (n,2), treated as a single batch.

    Returns
    -------
    pose : ndarray, shape (3,) or (batch_size, 3)
        (tx, ty, theta) per batch, where theta is in radians.
    """
    kp1 = np.asarray(kp1, dtype=np.float64)
    kp2 = np.asarray(kp2, dtype=np.float64)

    single = (kp2.ndim == 2)
    if single:
        kp2 = kp2[None, ...]        # (1, n, 2)
    kp1 = kp1[None, ...]            # (1, n, 2); broadcasts against kp2

    # Centers
    kp1_center = kp1.mean(axis=1, keepdims=True)          # (1, 1, 2)
    kp2_center = kp2.mean(axis=1, keepdims=True)          # (B, 1, 2)

    # Zero-mean
    kp1c = kp1 - kp1_center                               # (1, n, 2) -> broadcast
    kp2c = kp2 - kp2_center                               # (B, n, 2)

    # Batched covariance H = kp1c^T @ kp2c  -> (B, 2, 2)
    H = np.einsum('bni,bnj->bij', kp1c, kp2c)

    # Batched SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=True)       # each (B, 2, 2), (B,2), (B,2,2)

    # Rotation with reflection handling: R = V * D * U^T, where D fixes det(R)=+1
    R_raw = Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)  # (B, 2, 2)
    det_R = np.linalg.det(R_raw)                          # (B,)

    D = np.tile(np.eye(2)[None, ...], (R_raw.shape[0], 1, 1))  # (B,2,2)
    D[:, 1, 1] = np.sign(det_R)                                # diag=[1, sign(det)]

    R = Vt.transpose(0, 2, 1) @ D @ U.transpose(0, 2, 1)  # (B, 2, 2)

    # Translation: t = mu2 - R * mu1
    mu1 = kp1_center.transpose(0, 2, 1)                   # (1, 2, 1)
    mu2 = kp2_center.transpose(0, 2, 1)                   # (B, 2, 1)
    t = (mu2 - R @ mu1).transpose(0, 2, 1).squeeze(1)     # (B, 2)

    # Angle
    theta = np.arctan2(R[:, 1, 0], R[:, 0, 0])            # (B,)

    pose = np.concatenate([t, theta[:, None]], axis=1)    # (B, 3)
    if single:
        pose = pose[0]
    return pose

def get_truncated_normal(mean=0, sd=1, low=-10, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def gen_act(delta, scale=1, bound=30):
    if delta > 0:
        act = random.random() * bound * (scale + 1) - bound
        return act if act <= 0 or delta > bound else act / scale
    else:
        act = -random.random() * bound * (scale + 1) + bound
        return act if act >= 0 or delta < -bound else act / scale
