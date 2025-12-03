import sys, os
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import time

from envs.T_pushing.base_sim import Base_Sim
from envs.T_pushing.helper import transform_polys_wrt_pose_2d, get_rect_vertices, calculate_com, keypoints_to_pose_2d_SVD

""" 
main class for the T-shaped pushing task
"""

real_factor = 1


class T_Sim(Base_Sim):
    def __init__(self, param_dict, init_poses=None, target_poses=None, pusher_pos=None):
        super().__init__(param_dict)
        if target_poses is None:
            self.target_positions = None
            self.target_angles = None
        else:
            self.target_positions = [target_pose[:2] for target_pose in target_poses]
            self.target_angles = [target_pose[2] for target_pose in target_poses]  # in radians
        if init_poses is not None:
            self.obj_num = len(init_poses)
        elif "obj_num" in param_dict:
            self.obj_num = param_dict["obj_num"]
        else:
            self.obj_num = 1
        self.param_dict = param_dict
        stem_size = param_dict["stem_size"]
        bar_size = param_dict["bar_size"]
        if not isinstance(stem_size, tuple):
            stem_size = (stem_size[0], stem_size[1])
        if not isinstance(bar_size, tuple):
            bar_size = (bar_size[0], bar_size[1])
        self.stem_size = stem_size
        self.bar_size = bar_size
        self.create_world(init_poses, pusher_pos)

    def create_object(self, id, pose=None):
        """
        Create a single T-shaped piece by defining its shapes, mass, etc.
        """
        color = self.object_colors[id % len(self.object_colors)]
        stem_size, bar_size = self.stem_size, self.bar_size
        mass = self.obj_mass  # Total mass of the T shape

        # Get vertices for stem and bar
        stem_vertices, bar_vertices = self.get_t_comp_vertices()

        # Calculate moments for each part with equal mass distribution
        stem_square = stem_size[0] * stem_size[1]
        bar_square = bar_size[0] * bar_size[1]
        total_square = stem_square + bar_square
        stem_moment = pymunk.moment_for_poly(mass * stem_square / total_square, stem_vertices)
        bar_moment = pymunk.moment_for_poly(mass * bar_square / total_square, bar_vertices)
        moment = stem_moment + bar_moment
        body = pymunk.Body(mass, moment)
        if pose is None:
            body.angle = random.random() * math.pi * 2
            body.position = Vec2d(random.randint(200, 300), random.randint(200, 300))
        else:
            body.angle = pose[2]
            body.position = Vec2d(pose[0], pose[1])

        # Create the stem shape
        stem = pymunk.Poly(body, stem_vertices)
        stem.color = color
        stem.elasticity = self.elasticity
        stem.friction = self.friction
        # Create the bar shape
        bar = pymunk.Poly(body, bar_vertices)
        bar.color = color
        bar.elasticity = self.elasticity
        bar.friction = self.friction

        return body, [stem, bar]

    def get_object_keypoints(self, index, target=False):
        return get_keypoints_from_pose(self.get_object_pose(index, target), self.param_dict, self.include_com)

    def get_object_vertices(self, index, target=False):
        return transform_polys_wrt_pose_2d(self.get_t_comp_vertices(), self.get_object_pose(index, target))

    def gen_vertices_from_pose(self, pose):
        return transform_polys_wrt_pose_2d(self.get_t_comp_vertices(), pose)

    def get_current_state(self):
        obj_keypoints = self.get_all_object_keypoints()[0]
        state = np.concatenate(
            (obj_keypoints, np.array([self.pusher_body.position]), np.array([self.velocity])), axis=0
        )
        return state.flatten()

    def get_t_comp_vertices(self, flatten=False):
        """
        Get the vertices of the stem and bar of the T shape.
        """
        stem_size, bar_size = self.stem_size, self.bar_size
        center_stem, center_bar, com = get_t_comp_centers(stem_size, bar_size)
        center_stem = np.array(center_stem) - np.array(com)
        center_bar = np.array(center_bar) - np.array(com)
        stem_vertices = get_rect_vertices(*stem_size)
        stem_vertices = stem_vertices + center_stem
        if flatten:
            stem_vertices = stem_vertices.flatten()
        bar_vertices = get_rect_vertices(*bar_size)
        bar_vertices = bar_vertices + center_bar
        if flatten:
            bar_vertices = bar_vertices.flatten()
        return [stem_vertices.tolist(), bar_vertices.tolist()]


""" 
some helper functions
"""


def generate_init_target_states(init_poses, target_poses, param_dict, include_com=False):
    init_states = get_keypoints_from_pose(init_poses[0], param_dict, include_com)
    target_states = get_keypoints_from_pose(target_poses[0], param_dict, include_com)
    return init_states.flatten(), target_states.flatten()


def get_t_comp_centers(stem_size, bar_size):
    """
    Get the center of the stem, bar and the T shape.
    """
    w_s, h_s = stem_size
    w_b, h_b = bar_size
    # consider the bottom center of the stem as the origin
    # we calulate the center of mass of the stem
    x_s, y_s = 0, h_s / 2
    x_b, y_b = 0, h_s + h_b / 2
    m_s, m_b = w_s * h_s, w_b * h_b
    x_m, y_m = calculate_com([x_s, x_b], [y_s, y_b], [m_s, m_b])
    return [x_s, y_s], [x_b, y_b], [x_m, y_m]


def get_keypoints_from_pose(pose, param_dict, include_com=False):
    stem_size, bar_size = param_dict["stem_size"], param_dict["bar_size"]
    pos = Vec2d(pose[0], pose[1])
    angle = pose[2]
    w_s, h_s = stem_size
    w_b, h_b = bar_size
    _, [x_b, y_b], [x_m, y_m] = get_t_comp_centers(stem_size, bar_size)
    com = Vec2d(x_m, y_m)
    # Left Center, Middle Top Center, Right Center, Bottom Center
    # # consider the bottom center of the stem as the origin
    offsets = [
        Vec2d(-w_b / 2, y_b),
        Vec2d(0, y_b),
        Vec2d(w_b / 2, y_b),
        Vec2d(0, 0),
    ]
    if include_com:
        offsets.append(Vec2d(0, 0))
    # Calculate the global position of each keypoint
    keypoints = []
    for offset in offsets:
        offset = offset - com
        rotated_offset = offset.rotated(angle)
        keypoints.append(pos + rotated_offset)

    return np.array(keypoints)


def get_pose_from_keypoints(keypoints, param_dict):
    """
    Get the pose of the T shape from its keypoints.
    """
    stem_size, bar_size = param_dict["stem_size"], param_dict["bar_size"]
    w_s, h_s = stem_size
    w_b, h_b = bar_size
    _, [x_b, y_b], [x_m, y_m] = get_t_comp_centers(stem_size, bar_size)
    model_points = np.array(
        [
            [-w_b / 2, y_b],
            [0, y_b],
            [w_b / 2, y_b],
            [0, 0],
        ]
    ) - np.array([x_m, y_m])

    return keypoints_to_pose_2d_SVD(model_points, keypoints)


def get_offests_w_origin(param_dict):
    stem_size, bar_size = param_dict["stem_size"], param_dict["bar_size"]
    _, _, [x_g, y_g] = get_t_comp_centers(stem_size, bar_size)
    # hardcode as the top left corner of the bar, deponds on the .obj file
    com_w_origin = np.array([bar_size[0] / 2, y_g - (bar_size[1] + stem_size[1])])
    offest_w_com = get_keypoints_from_pose([0, 0, 0], param_dict)
    return offest_w_com + com_w_origin


if __name__ == "__main__":
    param_dict = {
        "stem_size": (10, 60),
        "bar_size": (60, 10),
        "pusher_size": 5,
        "save_img": False,
        "enable_vis": True,
    }

    # init_poses = [[[250,250,math.radians(45)], [150,150,math.radians(-45)]]]
    init_poses = [[250, 250, math.radians(0)]]
    target_poses = [[250, 250, math.radians(45)]]
    sim = T_Sim(
        param_dict=param_dict,
        init_poses=init_poses,
        target_poses=target_poses,
    )
    # [[250,250,math.radians(45)], [150,150,math.radians(-45)]]
    sim.render()
    print(sim.get_all_object_positions())
    print(sim.get_all_object_keypoints())
    print(sim.get_current_state())
    print(sim.get_all_object_keypoints(target=True))
    for i in range(5):
        sim.render()
        time.sleep(0.5)
