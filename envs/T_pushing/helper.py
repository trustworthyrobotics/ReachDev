import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import random
import math
import re
import cv2
from PIL import Image
from io import TextIOWrapper
import os, psutil
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
    Calculate the 2D pose transformation from keypoints using SVD supporting batch processing with torch tensors.

    Parameters:
    - kp1: 2D keypoints in the original frame. Shape: (n, 2) array
    - kp2: 2D keypoints in the target frame. Shape: (batch_size, n, 2) tensor or (n, 2) array

    Returns:
    - pose: 2D pose transformations for each batch, Shape: (batch_size, 3) or (3,)
            each containing 2D position (x, y) and rotation angle in radians.
    """
    # Convert inputs to torch.Tensor if they are numpy.ndarray
    original_is_ndarray = False
    if isinstance(kp2, np.ndarray):
        original_is_ndarray = True
        kp1 = torch.tensor(kp1, dtype=torch.float32)
        kp2 = torch.tensor(kp2, dtype=torch.float32)
    elif isinstance(kp2, torch.Tensor):
        kp1 = torch.tensor(kp1, dtype=torch.float32, device=kp2.device)
    kp1 = kp1.unsqueeze(0)
    original_in_batch = True
    if kp2.dim() == 2:
        original_in_batch = False
        kp2 = kp2.unsqueeze(0)

    # Center the keypoints
    kp1_center = kp1.mean(dim=1, keepdim=True)
    kp2_center = kp2.mean(dim=1, keepdim=True)
    kp1_centered = kp1 - kp1_center
    kp2_centered = kp2 - kp2_center

    # Compute the covariance matrix for each batch
    H = torch.matmul(kp1_centered.transpose(-2, -1), kp2_centered)

    # Perform SVD
    U, S, Vt = torch.linalg.svd(H, full_matrices=True)

    # Compute the rotation matrix
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    # Ensure the determinant of the rotation matrix is 1 (correcting for reflection if necessary)
    det_R = torch.linalg.det(R)
    reflection_correction = torch.diag_embed(torch.ones(R.shape[:-1], device=R.device))
    reflection_correction[:, -1, -1] = torch.sign(det_R)
    R = torch.matmul(Vt.transpose(-2, -1), reflection_correction.matmul(U.transpose(-2, -1)))

    # Compute the translation vector
    t = (kp2_center - torch.matmul(R, kp1_center.transpose(-2, -1)).transpose(-2, -1)).squeeze(1)

    # Compute the rotation angle in radians
    theta = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    # Concatenate translation and rotation to form the pose
    pose = torch.cat((t, theta.unsqueeze(-1)), dim=1)

    # Convert back to numpy.ndarray if the original input was ndarray
    if not original_in_batch:
        pose = pose.squeeze(0)
    if original_is_ndarray:
        pose = pose.numpy()

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
