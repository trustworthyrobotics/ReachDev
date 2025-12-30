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
from PIL import Image, ImageSequence


class Base_Sim(object):
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.SAVE_IMG, self.ENABLE_VIS = (
            param_dict["save_img"],
            param_dict["enable_vis"],
        )
        self.include_com = False
        # Sim window parameters. These also define the resolution of the image
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
        self.obj_num = 0
        self.obj_list = []
        self.static_obj_list = []
        self.image_list = []
        self.current_image = None
        # self.object_colors = [(54, 85, 146, 255), (247, 193, 67, 255)]
        # self.pusher_color = (255, 0, 0, 255)
        # self.background_color = (220,220,220, 255)
        # self.obs_color = (255,215,0, 255)
        self.object_colors = [(255, 165, 0, 255), (54, 85, 146, 255)]
        self.target_colors = [(255, 99, 71, 255), (27, 42, 73, 255)]
        self.pusher_color = (0, 0, 0, 255)
        self.background_color = (255,255,255, 255)
        self.obs_color = (128, 128, 128, 255)
        self.space = pymunk.Space()
        self.obs_pos_list = param_dict.get("obs_pos_list", None)
        self.obs_size_list = param_dict.get("obs_size_list", None)
        self.obs_type = param_dict.get("obs_type", None)

    def create_world(self, init_poses, pusher_pos):
        self.space.gravity = Vec2d(0, 0)  # planar setting
        self.space.damping = 0.0001  # quasi-static. low value is higher damping.
        self.space.iterations = 5  # TODO(terry-suh): re-check. what does this do?
        self.add_objects(self.obj_num, init_poses)
        self.add_pusher(pusher_pos)
        self.wait(1.0)
        # self.image_list = []
        # self.current_image = None

    def add_objects(self, obj_num, poses=None):
        """
        Create and add multiple object to sim.
        """
        if poses is None:
            for i in range(obj_num):
                self.add_object(i)
        else:
            for i in range(obj_num):
                self.add_object(i, poses[i])
        if hasattr(self, 'create_static_object'):
            s_obj1, shape_1 = self.create_static_object([125,125,0])
            self.space.add(s_obj1, *shape_1)
            self.static_obj_list.append([s_obj1, shape_1])  # Adjust storage to handle multiple shapes

            s_obj2, shape_2 = self.create_static_object([275,275,0])
            self.space.add(s_obj2, *shape_2)
            self.static_obj_list.append([s_obj2, shape_2])  # Adjust storage to handle multiple shapes
            

    def add_object(self, id, pose=None):
        """
        Create and add a single object to sim.
        """
        body, shape_components = self.create_object(id, pose)
        self.space.add(body, *shape_components)
        self.obj_list.append([body, shape_components])  # Adjust storage to handle multiple shapes

    def create_object(self, id, poses=None):
        """
        Create a single object by defining its shape, mass, etc.
        """
        raise NotImplementedError

    def remove_all_objects(self):
        """
        Remove all objects from sim.
        """
        for i in range(len(self.obj_list)):
            body = self.obj_list[i][0]
            shapes = self.obj_list[i][1]
            self.space.remove(body, *shapes)
        for i in range(len(self.static_obj_list)):
            body = self.static_obj_list[i][0]
            shapes = self.static_obj_list[i][1]
            self.space.remove(body, *shapes)
        self.obj_list = []
        self.static_obj_list = []

    def get_object_pose(self, index, target=False):
        """
        Return the pose of an object in sim.
        """
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
        """
        Return the poses of all objects in sim.
        """
        if target and self.target_positions is None:
            return None
        all_poses = []
        for i in range(len(self.obj_list)):
            all_poses.append(self.get_object_pose(i, target))
        return all_poses

    def update_object_pose(self, index, new_pose):
        """
        Update the pose of an object in sim.
        """
        body = self.obj_list[index][0]
        body.angle = new_pose[2]
        body.position = pymunk.Vec2d(new_pose[0], new_pose[1])
        self.wait(1.0)  # Give some time for collision pieces to stabilize.
        return

    def get_all_object_positions(self):
        """
        Return the positions of all objects in sim.
        """
        return [body.position for body, _ in self.obj_list]

    def get_all_object_angles(self):
        """
        Return the angles of all objects in sim.
        """
        return [body.angle for body, _ in self.obj_list]

    def get_object_keypoints(self, index, target=False, **kwargs):
        """
        Return the keypoints of an object in sim.
        """
        raise NotImplementedError

    def get_all_object_keypoints(self, target=False, **kwargs):
        """
        Return the keypoints of all objects in sim.
        """
        if target and self.target_positions is None:
            return None
        all_keypoints = []
        
        for i in range(len(self.obj_list)):
            all_keypoints.append(self.get_object_keypoints(i, target, **kwargs))
        # for i in range(len(self.static_obj_list)):
        #     all_keypoints.append(np.array([self.static_obj_list[i][0].position]))
        # reorg_keypoints = []
        # for keyp in all_keypoints:
        #     reorg_keypoints.extend(keyp)
        
        return all_keypoints

    def get_object_vertices(self, index, target=False, **kwargs):
        """
        Return the vertices of an object in sim.
        """
        raise NotImplementedError

    def get_all_object_vertices(self, target=False, **kwargs):
        """
        Return the vertices of all objects in sim.
        """
        if target and self.target_positions is None:
            return None
        all_vertices = []
        for i in range(len(self.obj_list)):
            all_vertices.append(self.get_object_vertices(i, target, **kwargs))

        return all_vertices

    def gen_vertices_from_pose(self, pose, **kwargs):
        """
        Generate vertices from a pose.
        """
        raise NotImplementedError

    def get_static_state(self,):
        all_keypoints = []
        for i in range(len(self.static_obj_list)):
            all_keypoints.append(np.array(self.static_obj_list[i][0].position))
        return np.array(all_keypoints).flatten()
    
    def get_kp_state(self):
        """
        Return the keypoints of all objects in sim.
        """
        return np.array(self.get_all_object_keypoints()).flatten()

    def get_current_state(self):
        raise NotImplementedError

    def create_pusher(self, position):
        """
        Create a single pusher by defining its shape, mass, etc.
        """
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
        """
        Create and add a single pusher to the sim.
        """
        self.pusher_body, self.pusher_shape = self.create_pusher(position)
        self.space.add(self.pusher_body, self.pusher_shape)

    def remove_pusher(self):
        """
        Remove pusher from simulation.
        """
        self.space.remove(self.pusher_body, self.pusher_shape)

    def get_pusher_position(self):
        """
        Return the position of the pusher.
        """
        if self.pusher_body is None:
            return None
        return np.array(self.pusher_body.position)

    def update(self, action, rel=True, n_sim_step=60):
        """
        Once given a control action, run the simulation forward and return.
        """
        # Parse into integer coordinates
        uxf, uyf = action

        # add the pusher if not added
        if self.pusher_body is None:
            self.add_pusher((uxf, uyf))
            return None

        uxi, uyi = self.pusher_body.position

        # transform into angular coordinates
        theta = np.arctan2(uyf - uyi, uxf - uxi)
        length = np.linalg.norm(np.array([uxf - uxi, uyf - uyi]), ord=2)
        # length /= 1.5
        step_dt = 1.0 / 60.0
        self.velocity = np.array([np.cos(theta), np.sin(theta)]) * length
        self.pusher_body.velocity = self.velocity.tolist()

        for i in range(n_sim_step):
            # make sure that pos_next = pos_curr + vel * dt (in pixel space)
            self.pusher_body.velocity = self.velocity.tolist()
            self.space.step(step_dt)
            self.global_time += step_dt
            # print(self.pusher_body.velocity )
        # Wait 1 second in sim time to slow down moving pieces, and render.
        # self.wait(1.0)
        self.render()
        
        env_dict = {
            "state": np.concatenate([self.get_kp_state(), self.get_static_state()]),
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

    def get_env_state(self, rel=True):
        """
        Return the environment state.
        """
        env_dict = {
            "state": self.get_kp_state(),
            "pusher_pos": self.get_pusher_position(),
            "action": self.velocity,
        }
        if rel:
            env_dict["state"][0::2] -= env_dict["pusher_pos"][0]
            env_dict["state"][1::2] -= env_dict["pusher_pos"][1]

        env_state = np.concatenate([env_dict["state"], env_dict["pusher_pos"], env_dict["action"]], axis=0)
        return env_state, env_dict

    def wait(self, time):
        """
        Wait for some time in the simulation. Gives some time to stabilize bodies in collision.
        """
        t = 0
        step_dt = 1 / 60.0
        while t < time:
            self.space.step(step_dt)
            t += step_dt

    """
    2.2 Methods related to rendering and image publishing
    """

    def render(self):
        if not (self.ENABLE_VIS or self.SAVE_IMG):
            return
        # start_time = time.time()

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.background_color[:3]

        if self.obs_pos_list is not None:
            for obs_pos, obs_size in zip(self.obs_pos_list, self.obs_size_list):
                obs_pos = np.array(obs_pos, dtype=np.int32)
                obs_size = int(obs_size)
                if self.obs_type == "circle":
                    cv2.circle(img, obs_pos, obs_size, self.obs_color[:3], -1)
                elif self.obs_type == "square":
                    cv2.rectangle(img, tuple(obs_pos - obs_size), tuple(obs_pos + obs_size), self.obs_color[:3], -1)
        poses = self.get_all_object_poses()
        for draw_target in [True, False]:
            obj_list = self.get_all_object_vertices(target=draw_target)
            if obj_list is None:
                continue
            if len(self.static_obj_list) > 0:
                cv2.circle(img, np.array(self.static_obj_list[0][0].position, dtype=np.int32), self.obstacle_size, self.obstacle_color, -1)
                cv2.circle(img, np.array(self.static_obj_list[1][0].position, dtype=np.int32), self.obstacle_size, self.obstacle_color, -1)
            for i, obj in enumerate(obj_list):
                polys = np.array(obj, np.int32)
                color = self.object_colors[i % len(self.object_colors)][:3]
                if draw_target:
                    color = self.target_colors[i % len(self.target_colors)][:3]
                cv2.fillPoly(img, polys, color)

        pusher_pos = self.get_pusher_position()
        assert pusher_pos is not None, "Pusher position is not initialized!"
        pusher_pos = np.array(pusher_pos, dtype=np.int32)
        cv2.circle(img, pusher_pos, self.pusher_size, self.pusher_color[:3], -1)

        # cv2 has BGR format, and flipped y-axis
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 0)
        if self.ENABLE_VIS:
            cv2.imshow("Simulator", img)
            cv2.waitKey(1)
        # print("Update Image Time: ", time.time() - start_time)
        if self.SAVE_IMG:
            self.image_list.append(img)
            self.current_image = img

        return

    def gen_img_from_poses(self, poses, pusher_pos, img_file=None, size=None):
        """
        Generate an image from a list of object poses and pusher position.
        """
        # assert len(poses) == self.obj_num, "Number of poses does not match the number of objects!"
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.background_color[:3]
        for i in range(self.obj_num):
            pose = poses[i]
            obj = self.gen_vertices_from_pose(pose)
            polys = np.array(obj, np.int32)
            cv2.fillPoly(img, polys, self.object_colors[i % len(self.object_colors)][:3])

        # static object pose
        if len(self.static_obj_list) > 0:
            cv2.circle(img, np.array(self.static_obj_list[0][0].position, dtype=np.int32), self.obstacle_size, self.obstacle_color, -1)
            cv2.circle(img, np.array(self.static_obj_list[1][0].position, dtype=np.int32), self.obstacle_size, self.obstacle_color, -1)
            
        pusher_pos = np.array(pusher_pos, dtype=np.int32)
        cv2.circle(img, pusher_pos, self.pusher_size, self.pusher_color[:3], -1)

        # cv2 has BGR format, and flipped y-axis
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 0)
        if size is not None:
            img = cv2.resize(img, size)
        if img_file is not None:
            cv2.imwrite(img_file, img)
            print(f"img saved to {img_file}")
        # cv2.imwrite("test.png", img)
        # print("img saved to test.png")
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = img / 255.0
        return img

    def close(self):
        """
        Close the simulation.
        """
        # if self.window is not None:
        #     self.window.close()
        # cv2.destroyAllWindows()

    def get_img_state(self):
        img = self.current_image
        assert img is not None, "Image is not initialized!"
        img = cv2.resize(img, (self.img_size, self.img_size))
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = img / 255.0
        return img

    def save_mp4(self, filename="output_video.mp4", fps=10):
        """
        Save the list of images as a video.

        Parameters:
            filename (str): Video filename
            fps (int): Frames per second
        """
        if not self.SAVE_IMG:
            print("no save")
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))

        try:
            for frame in self.image_list:
                out.write(frame)
            out.release()
            print(f"Video saved as {filename}")
        except Exception as e:
            out.release()  # Make sure to release the video writer object
            print(f"Failed to save video. Error: {e}")
        # self.image_list = []

    def play_mp4(self, filename="output_video.mp4"):
        """
        Play the video using OpenCV (or do other processing)

        Parameters:
            filename (str): Video filename
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return
        cap = cv2.VideoCapture(filename)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(filename, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_gif(self, filename="output_video.gif", fps=30):
        """
        Save the list of images as a gif.

        Parameters:
            filename (str): Gif filename
            fps (int): Frames per second
        """
        if not self.SAVE_IMG:
            print("no save")
            return
        images = []
        for frame in self.image_list:
            images.append(np.array(frame))
        iio.mimsave(filename, images, fps=fps)
        print(f"-----Gif saved as {filename} ----")
        # self.image_list = []

    def play_gif(self, filename="output_video.gif"):
        """
        Play the gif using OpenCV (or do other processing)

        Parameters:
            filename (str): Gif filename
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return
        # Read the gif from the file
        img = Image.open(filename)
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]

        for frame in frames:
            # Convert the PIL image to an OpenCV frame
            # opencv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            opencv_image = np.array(frame)

            # Display the frame
            cv2.imshow(filename, opencv_image)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    """
    3. Methods for External Commands
    """

    def refresh(self, new_poses=None):
        self.remove_all_objects()
        self.remove_pusher()
        self.pusher_body = None
        self.pusher_shape = None
        self.add_objects(self.obj_num, new_poses)
        
        self.wait(1.0)  # Give some time for collision pieces to stabilize.
        self.image_list = []