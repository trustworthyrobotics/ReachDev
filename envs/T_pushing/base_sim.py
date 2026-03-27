import sys, os
import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import time

import pymunk
from pymunk import Vec2d
import math
import numpy as np
import random
import cv2
import imageio.v2 as iio


class Base_Sim(object):
    def __init__(self, param_dict, step_dt=1.0 / 60.0):
        self.param_dict = param_dict
        self.SAVE_IMG, self.ENABLE_VIS = (
            param_dict["save_img"],
            param_dict["enable_vis"],
        )
        self.include_com = False
        self.width = self.height = param_dict["window_size"]
        self.elasticity = 0.1
        self.friction = 0.1
        self.obj_mass = 0.5
        self.velocity = np.array([0, 0])
        self.target_positions = None
        self.target_angles = None
        self.pusher_body = None
        self.pusher_shape = None
        self.pusher_size = param_dict["pusher_size"]
        self.global_time = 0.0
        self.step_dt = step_dt
        self.obj_num = 0
        self.obj_list = []
        self.image_list = []
        self.current_image = None

        self.object_colors = [(255, 165, 0, 255), (54, 85, 146, 255)]
        self.target_colors = [(255, 99, 71, 255), (27, 42, 73, 255)]
        self.pusher_color = (0, 0, 0, 255)
        self.background_color = (255, 255, 255, 255)
        self.obs_color = (128, 128, 128, 255)

        # Draw the actual pose center / (x, y) location by default
        self.show_pose_center = param_dict.get("show_pose_center", True)
        self.pose_center_radius = param_dict.get("pose_center_radius", 4)
        self.pose_center_color = param_dict.get("pose_center_color", (0, 0, 255, 255))
        self.pose_center_target_color = param_dict.get(
            "pose_center_target_color", (255, 0, 255, 255)
        )

        # Optional horizontal constraint visualization:
        # world-space threshold and world->pixel scale
        self.x_constraint = param_dict.get("x_constraint", None)
        self.y_constraint = param_dict.get("y_constraint", None)
        self.constraint_scale = param_dict.get("constraint_scale", 1.0)
        self.constraint_color = param_dict.get("constraint_color", (0, 0, 255, 255))

        # Existing obstacle API
        self.obs_pos_list = param_dict.get("obs_pos_list", None)
        self.obs_size_list = param_dict.get("obs_size_list", None)
        self.obs_norm = param_dict.get("obs_norm", None)

        # New API: obstacles = [[x, y, r], ...] for circular obstacles
        self.obstacles = param_dict.get("obstacles", None)
        if self.obstacles is not None:
            self.obstacles = np.asarray(self.obstacles, dtype=np.float32)
            assert self.obstacles.ndim == 2 and self.obstacles.shape[1] == 3, \
                "param_dict['obstacles'] must have shape (num_obstacles, 3) with rows [x, y, r]"

            # Reuse existing renderer path
            self.obs_pos_list = [obs[:2] for obs in self.obstacles]
            self.obs_size_list = [int(round(obs[2])) for obs in self.obstacles]
            self.obs_norm = 2

    def create_world(self, init_poses, pusher_pos):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, 0)  # planar setting
        self.space.damping = 0.0001  # quasi-static. low value is higher damping.
        self.space.iterations = 5
        self.add_objects(self.obj_num, init_poses)
        self.add_pusher(pusher_pos)
        self.wait(1.0)

    def add_objects(self, obj_num, poses=None):
        if poses is None:
            for i in range(obj_num):
                self.add_object(i)
        else:
            for i in range(obj_num):
                self.add_object(i, poses[i])

    def add_object(self, id, pose=None):
        body, shape_components = self.create_object(id, pose)
        self.space.add(body, *shape_components)
        self.obj_list.append([body, shape_components])

    def create_object(self, id, poses=None):
        raise NotImplementedError

    def remove_all_objects(self):
        for i in range(len(self.obj_list)):
            body = self.obj_list[i][0]
            shapes = self.obj_list[i][1]
            self.space.remove(body, *shapes)
        self.obj_list = []

    def get_object_pose(self, index, target=False):
        if target:
            pos = self.target_positions[index]
            angle = self.target_angles[index]
            pose = [pos[0], pos[1], angle]
        else:
            body: pymunk.Body = self.obj_list[index][0]
            pos = body.position
            angle = body.angle
            pose = [pos.x, pos.y, angle]
        return pose

    def get_all_object_poses(self, target=False):
        if target and self.target_positions is None:
            return None
        all_poses = []
        for i in range(len(self.obj_list)):
            all_poses.append(self.get_object_pose(i, target))
        return all_poses

    def update_object_pose(self, index, new_pose):
        body = self.obj_list[index][0]
        body.angle = new_pose[2]
        body.position = pymunk.Vec2d(new_pose[0], new_pose[1])
        self.wait(1.0)
        return

    def get_all_object_positions(self):
        return [body.position for body, _ in self.obj_list]

    def get_all_object_angles(self):
        return [body.angle for body, _ in self.obj_list]

    def get_object_keypoints(self, index, target=False, **kwargs):
        raise NotImplementedError

    def get_all_object_keypoints(self, target=False, **kwargs):
        if target and self.target_positions is None:
            return None
        all_keypoints = []
        for i in range(len(self.obj_list)):
            all_keypoints.append(self.get_object_keypoints(i, target, **kwargs))
        return all_keypoints

    def get_object_vertices(self, index, target=False, **kwargs):
        raise NotImplementedError

    def get_all_object_vertices(self, target=False, **kwargs):
        if target and self.target_positions is None:
            return None
        all_vertices = []
        for i in range(len(self.obj_list)):
            all_vertices.append(self.get_object_vertices(i, target, **kwargs))
        return all_vertices

    def get_kp_state(self):
        return np.array(self.get_all_object_keypoints()).flatten()

    def get_current_state(self):
        raise NotImplementedError

    def create_pusher(self, position):
        body = pymunk.Body(1e7, float("inf"))
        if position is None:
            body.position = Vec2d(
                random.randint(int(self.width * 0.25), int(self.width * 0.75)),
                random.randint(int(self.height * 0.25), int(self.height * 0.75)),
            )
        else:
            body.position = Vec2d(position[0], position[1])
        shape = pymunk.Circle(body, radius=self.pusher_size)
        shape.elasticity = 0.1
        shape.friction = 0.6
        shape.color = self.pusher_color
        return body, shape

    def add_pusher(self, position):
        self.pusher_body, self.pusher_shape = self.create_pusher(position)
        self.space.add(self.pusher_body, self.pusher_shape)

    def remove_pusher(self):
        self.space.remove(self.pusher_body, self.pusher_shape)

    def get_pusher_position(self):
        if self.pusher_body is None:
            return None
        return np.array(self.pusher_body.position)

    def update(self, action, rel=True, n_sim_time=1):
        uxf, uyf = action

        if self.pusher_body is None:
            self.add_pusher((uxf, uyf))
            return None

        uxi, uyi = self.pusher_body.position

        theta = np.arctan2(uyf - uyi, uxf - uxi)
        length = np.linalg.norm(np.array([uxf - uxi, uyf - uyi]), ord=2)

        self.velocity = np.array([np.cos(theta), np.sin(theta)]) * length
        self.pusher_body.velocity = self.velocity.tolist()

        n_sim_step = round(n_sim_time / self.step_dt)
        for i in range(n_sim_step):
            self.pusher_body.velocity = self.velocity.tolist()
            self.space.step(self.step_dt)
            self.global_time += self.step_dt
        self.render()

        return self.get_env_state(rel)

    def force_update(self, deltas):
        for i in range(len(deltas)):
            obj_body = self.obj_list[i][0]
            delta = deltas[i]
            obj_body.position += Vec2d(delta[0], delta[1])
            obj_body.angle += delta[2]
        return

    def get_env_state(self, rel=True):
        env_dict = {
            "state": self.get_kp_state(),
            "pusher_pos": self.get_pusher_position(),
            "action": self.velocity,
            "com_pos": np.array(self.get_all_object_positions()).flatten(),
            "angle": np.array(self.get_all_object_angles()).flatten(),
        }
        if rel:
            env_dict["state"][0::2] -= env_dict["pusher_pos"][0]
            env_dict["state"][1::2] -= env_dict["pusher_pos"][1]
            env_dict["com_pos"][0::2] -= env_dict["pusher_pos"][0]
            env_dict["com_pos"][1::2] -= env_dict["pusher_pos"][1]

        return env_dict

    def wait(self, time):
        t = 0
        while t < time:
            self.space.step(self.step_dt)
            t += self.step_dt

    def _draw_pose_center(self, img, pos, color, radius):
        pos = np.array(pos, dtype=np.int32)
        x, y = int(pos[0]), int(pos[1])

        # Filled dot
        cv2.circle(img, (x, y), radius, color[:3], -1)

        # Small crosshair for visibility
        cross = max(3, radius + 2)
        cv2.line(img, (x - cross, y), (x + cross, y), color[:3], 1)
        cv2.line(img, (x, y - cross), (x, y + cross), color[:3], 1)

    def render(self):
        if not (self.ENABLE_VIS or self.SAVE_IMG):
            return

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.background_color[:3]

        if self.y_constraint is not None:
            y_pix = int(round(float(self.y_constraint) * float(self.constraint_scale)))
            y_pix = max(0, min(self.height - 1, y_pix))

            dash_length = 12
            gap_length = 8
            thickness = 2
            line_color = self.constraint_color[:3]

            x = 0
            while x < self.width:
                x_end = min(x + dash_length, self.width - 1)
                cv2.line(
                    img,
                    (x, y_pix),
                    (x_end, y_pix),
                    line_color,
                    thickness,
                )
                x += dash_length + gap_length

        if self.x_constraint is not None:
            x_pix = int(round(float(self.x_constraint) * float(self.constraint_scale)))
            x_pix = max(0, min(self.width - 1, x_pix))

            dash_length = 12
            gap_length = 8
            thickness = 2
            line_color = self.constraint_color[:3]

            y = 0
            while y < self.height:
                y_end = min(y + dash_length, self.height - 1)
                cv2.line(
                    img,
                    (x_pix, y),
                    (x_pix, y_end),
                    line_color,
                    thickness,
                )
                y += dash_length + gap_length

        if self.obs_pos_list is not None:
            for obs_pos, obs_size in zip(self.obs_pos_list, self.obs_size_list):
                obs_pos = np.array(obs_pos, dtype=np.int32)

                if self.obs_norm == 2:
                    radius = int(round(obs_size))
                    cv2.circle(img, tuple(obs_pos), radius, self.obs_color[:3], -1)

                elif self.obs_norm == 1:
                    obs_size = np.array(obs_size, dtype=np.int32)
                    cv2.rectangle(
                        img,
                        tuple(obs_pos - obs_size),
                        tuple(obs_pos + obs_size),
                        self.obs_color[:3],
                        -1,
                    )

        for draw_target in [True, False]:
            obj_list = self.get_all_object_vertices(target=draw_target)
            if obj_list is None:
                continue
            for i, obj in enumerate(obj_list):
                polys = np.array(obj, np.int32)
                color = self.object_colors[i % len(self.object_colors)][:3]
                if draw_target:
                    color = self.target_colors[i % len(self.target_colors)][:3]
                cv2.fillPoly(img, polys, color)

        # Draw the actual pose center / (x, y) that body.position represents
        if self.show_pose_center:
            current_positions = self.get_all_object_positions()
            for pos in current_positions:
                self._draw_pose_center(
                    img,
                    pos,
                    self.pose_center_color,
                    self.pose_center_radius,
                )

            if self.target_positions is not None:
                for pos in self.target_positions:
                    self._draw_pose_center(
                        img,
                        pos,
                        self.pose_center_target_color,
                        self.pose_center_radius,
                    )

        pusher_pos = self.get_pusher_position()
        assert pusher_pos is not None, "Pusher position is not initialized!"
        pusher_pos = np.array(pusher_pos, dtype=np.int32)
        cv2.circle(img, tuple(pusher_pos), self.pusher_size, self.pusher_color[:3], -1)

        img = cv2.flip(img, 0)
        if self.ENABLE_VIS:
            cv2.imshow("Simulator", img)
            cv2.waitKey(1)
        if self.SAVE_IMG:
            self.image_list.append(img)
            self.current_image = img

        return

    def close(self):
        """
        Close the simulation.
        """

    def get_img_state(self):
        img = self.current_image
        assert img is not None, "Image is not initialized!"
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        return img

    def save_gif(self, filename="output_video.gif", fps=30):
        if not self.SAVE_IMG:
            print("no save")
            return
        images = []
        for frame in self.image_list:
            images.append(np.array(frame))
        iio.mimsave(filename, images, fps=fps)
        print(f"-----Gif saved as {filename} ----")

    def refresh(self, new_poses=None):
        self.remove_all_objects()
        self.remove_pusher()
        self.pusher_body = None
        self.pusher_shape = None
        self.add_objects(self.obj_num, new_poses)

        self.wait(1.0)
        self.image_list = []