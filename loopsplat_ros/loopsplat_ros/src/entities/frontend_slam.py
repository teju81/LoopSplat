""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import time
import math
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from loopsplat_interfaces.msg import F2B, B2F, F2G, CameraMsg, GaussiansMsg

from loopsplat_ros.src.utils.io_utils import load_config

import numpy as np
import torch
import roma

from loopsplat_ros.src.entities.arguments import OptimizationParams
from loopsplat_ros.src.entities.datasets import get_dataset
from loopsplat_ros.src.gsr.camera import Camera
from loopsplat_ros.src.entities.gaussian_model import GaussianModel
from loopsplat_ros.src.entities.mapper import Mapper
from loopsplat_ros.src.entities.tracker import Tracker
from loopsplat_ros.src.entities.lc import Loop_closure
from loopsplat_ros.src.entities.logger import Logger
from loopsplat_ros.src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from loopsplat_ros.src.utils.mapper_utils import exceeds_motion_thresholds 
from loopsplat_ros.src.utils.utils import np2torch, setup_seed, torch2np
from loopsplat_ros.src.utils.graphics_utils import getProjectionMatrix2 
from loopsplat_ros.src.utils.vis_utils import *  # noqa - needed for debugging
from loopsplat_ros.src.utils.ros_utils import (
    convert_ros_array_message_to_tensor, 
    convert_ros_multi_array_message_to_tensor, 
    convert_tensor_to_ros_message, 
    convert_numpy_array_to_ros_message, 
    convert_ros_multi_array_message_to_numpy, 
)
from loopsplat_ros.src.gui.gui_utils import (
    ParamsGUI,
    GaussianPacket,
    Packet_vis2main,
    create_frustum,
    cv_gl,
    get_latest_queue,
)
from diff_gaussian_rasterization import GaussianRasterizationSettings
from loopsplat_ros.src.utils.utils import render_gaussian_model
from munch import munchify
from loopsplat_ros.src.utils.multiprocessing_utils import clone_obj


class FrontEnd(Node):

    def __init__(self, config: dict) -> None:
        super().__init__('frontend_slam_node')

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config
        self.pipeline_params = munchify(config["pipeline_params"])
        self.gaussian_model = GaussianModel(0)
        self.submap_id = 0
        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])
        self.exposures_ab = torch.zeros(len(self.dataset), 2)

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)
        self.enable_exposure = self.tracker.enable_exposure
        self.loop_closer = Loop_closure(config, self.dataset, self.logger)
        self.loop_closer.submap_path = self.output_path / "submaps"

        self.queue_size = 100
        self.msg_counter = 0
        self.f2g_publisher = self.create_publisher(F2G,'Front2GUI',self.queue_size)

        self.f2b_publisher = self.create_publisher(F2B,'Front2Back',self.queue_size)

        self.b2f_subscriber = self.create_subscription(B2F, '/Back2Front', self.b2f_listener_callback, self.queue_size)
        self.b2f_subscriber  # prevent unused variable warning
        
        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])
        print('Loop closure config')
        pprint.PrettyPrinter().pprint(config["lc"])

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                print(f"\nNew submap at {frame_id}")
                return True
        elif frame_id in self.new_submap_frame_ids:
            return True
        return False

    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        self.heard_from_backend = False
        # Allow for other nodes to come online by sleeping for a short while - otherwise the initial frontend ros messages dont reach them
        time.sleep(2)

        for frame_id in range(len(self.dataset)):

            if frame_id in [0, 1]:
                estimated_c2w = self.dataset[frame_id][-1]
                exposure_ab = torch.nn.Parameter(torch.tensor(
                    0.0, device="cuda")), torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
            else:

                # If backend doesnt comeup on time then wait for it
                while not self.heard_from_backend:
                    continue

                estimated_c2w, exposure_ab = self.tracker.track(
                    frame_id, self.gaussian_model,
                    torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
            exposure_ab = exposure_ab if self.enable_exposure else None
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            self.publish_message_to_backend(frame_id, estimated_c2w, exposure_ab)

    def b2f_listener_callback(self, b2f_msg):
        self.get_logger().info('I heard from backend: %s' % b2f_msg.msg)


        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device="cuda")

        # Gaussian model update part
        frame_id, submap_id, min_id, max_id, submap_kf_poses, submap_gaussians = self.convert_from_b2f_ros_msg(b2f_msg)

        if b2f_msg.msg == "loopclosure":
            self.estimated_c2ws[min_id:max_id + 1] = submap_kf_poses
            # To DO : Update the submap Gaussians
            # To DO : Send Updated submaps to GUI
        elif b2f_msg.msg == "sync":
            self.gaussian_model = submap_gaussians
            # Send updated info to GUI Node
            current_frame = Camera.init_from_dataset(self.dataset, frame_id, projection_matrix)
            gaussian_packet = GaussianPacket(
                submap_id=self.submap_id,
                gaussians=clone_obj(self.gaussian_model),
                current_frame=current_frame,
                gtcolor=current_frame.original_image,
                gtdepth=current_frame.depth
            )
            self.publish_message_to_gui(gaussian_packet)
        else:
            print("Unsupported message received from backend")

        self.heard_from_backend = True

    def convert_from_b2f_ros_msg(self, b2f_msg):
        frame_id = b2f_msg.frame_id
        submap_gaussians = GaussianModel(0)
        submap_gaussians.training_setup(self.opt)
        submap_id = None
        submap_kf_poses = None
        min_id = None
        max_id = None

        submap_gaussians.active_sh_degree = b2f_msg.gaussian.active_sh_degree
        submap_gaussians.max_sh_degree = b2f_msg.gaussian.max_sh_degree

        submap_gaussians._xyz = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.xyz, self.device)

        submap_gaussians._features_dc = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.features_dc, self.device)
        submap_gaussians._features_rest = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.features_rest, self.device)

        # # Number of features in features_dc
        # num_features_dc = b2f_msg.gaussian.num_features_dc

        # # Recover dc and rest of the features from concatenated tensor
        # self.gaussians._features_dc = features_cat[:, :num_features_dc]
        # self.gaussians._features_rest = features_cat[:, num_features_dc:]

        submap_gaussians._scaling = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.scaling, self.device)
        submap_gaussians._rotation = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.rotation, self.device)
        submap_gaussians._opacity = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.opacity, self.device)
        submap_gaussians.max_radii2D = convert_ros_array_message_to_tensor(b2f_msg.gaussian.max_radii2d, self.device)
        submap_gaussians.xyz_gradient_accum = convert_ros_multi_array_message_to_tensor(b2f_msg.gaussian.xyz_gradient_accum, self.device)

        # If loopclosure tag then update the poses of keyframes in the submap
        if b2f_msg.msg == "loopclosure":
            submap_id = b2f_msg.submap_id
            #b2f_msg.num_kfs = max_id - min_id + 1
            min_id = b2f_msg.min_id
            max_id = b2f_msg.max_id
            submap_kf_poses = convert_ros_multi_array_message_to_tensor(b2f_msg.submap_kf_poses)

        return frame_id, submap_id, min_id, max_id, submap_kf_poses, submap_gaussians

    def convert_to_f2b_ros_msg(self, frame_id, estimated_c2w, exposure_ab):
        f2b_msg = F2B()

        f2b_msg.msg = f'Hello world {self.msg_counter}'
        f2b_msg.frame_id = frame_id
        f2b_msg.estimated_c2w = convert_numpy_array_to_ros_message(estimated_c2w)
        if self.enable_exposure:
            f2b_msg.exposure_a = exposure_ab[0]
            f2b_msg.exposure_b = exposure_ab[1]
        f2b_msg.should_start_new_submap = self.should_start_new_submap(frame_id)

        return f2b_msg


    def publish_message_to_backend(self, frame_id, estimated_c2w, exposure_ab):
        f2b_msg = self.convert_to_f2b_ros_msg(frame_id, estimated_c2w, exposure_ab)
        self.get_logger().info(f'Publishing to Backend Node: {self.msg_counter}')
        self.f2b_publisher.publish(f2b_msg)
        self.msg_counter += 1

    def publish_message_to_gui(self, gaussian_packet):
        f2g_msg = self.convert_to_f2g_ros_msg(gaussian_packet)
        f2g_msg.msg = f'Hello world {self.msg_counter}'
        self.get_logger().info(f'Publishing to GUI Node: {self.msg_counter}')


        self.f2g_publisher.publish(f2g_msg)
        self.msg_counter += 1

    def convert_to_f2g_ros_msg(self, gaussian_packet):
        
        f2g_msg = F2G()

        f2g_msg.msg = "Sending 3D Gaussians"
        f2g_msg.has_gaussians = gaussian_packet.has_gaussians
        f2g_msg.submap_id = gaussian_packet.submap_id

        if gaussian_packet.has_gaussians:
            f2g_msg.active_sh_degree = gaussian_packet.active_sh_degree 

            f2g_msg.max_sh_degree = gaussian_packet.max_sh_degree
            f2g_msg.get_xyz = convert_tensor_to_ros_message(gaussian_packet.get_xyz)
            f2g_msg.get_features = convert_tensor_to_ros_message(gaussian_packet.get_features)
            f2g_msg.get_scaling = convert_tensor_to_ros_message(gaussian_packet.get_scaling)
            f2g_msg.get_rotation = convert_tensor_to_ros_message(gaussian_packet.get_rotation)
            f2g_msg.get_opacity = convert_tensor_to_ros_message(gaussian_packet.get_opacity)

            f2g_msg.n_obs = gaussian_packet.n_obs

            if gaussian_packet.gtcolor is not None:
                print(type(gaussian_packet.gtcolor))
                f2g_msg.gtcolor = convert_numpy_array_to_ros_message(gaussian_packet.gtcolor)
            
            if gaussian_packet.gtdepth is not None:
                f2g_msg.gtdepth = convert_numpy_array_to_ros_message(gaussian_packet.gtdepth)
        

            f2g_msg.current_frame = self.get_camera_msg_from_viewpoint(gaussian_packet.current_frame)

            f2g_msg.finish = gaussian_packet.finish


        return f2g_msg

    def get_camera_msg_from_viewpoint(self, viewpoint):

        if viewpoint is None:
            return None

        viewpoint_msg = CameraMsg()
        viewpoint_msg.uid = viewpoint.uid
        viewpoint_msg.device = viewpoint.device
        viewpoint_msg.rot = convert_tensor_to_ros_message(viewpoint.R)
        viewpoint_msg.trans = viewpoint.T.tolist()
        viewpoint_msg.rot_gt = convert_numpy_array_to_ros_message(viewpoint.R_gt)
        viewpoint_msg.trans_gt = viewpoint.T_gt.tolist()
        viewpoint_msg.original_image = convert_numpy_array_to_ros_message(viewpoint.original_image)
        viewpoint_msg.depth = convert_numpy_array_to_ros_message(viewpoint.depth)
        viewpoint_msg.fx = viewpoint.fx
        viewpoint_msg.fy = viewpoint.fy
        viewpoint_msg.cx = viewpoint.cx
        viewpoint_msg.cy = viewpoint.cy
        viewpoint_msg.fovx = viewpoint.FoVx
        viewpoint_msg.fovy = viewpoint.FoVy
        viewpoint_msg.image_width = viewpoint.image_width
        viewpoint_msg.image_height = viewpoint.image_height
        viewpoint_msg.cam_rot_delta = viewpoint.cam_rot_delta.tolist()
        viewpoint_msg.cam_trans_delta = viewpoint.cam_trans_delta.tolist()
        viewpoint_msg.exposure_a = viewpoint.exposure_a.item()
        viewpoint_msg.exposure_b = viewpoint.exposure_b.item()
        viewpoint_msg.projection_matrix = convert_tensor_to_ros_message(viewpoint.projection_matrix)

        return viewpoint_msg

def spin_thread(node):
    # Spin the node continuously in a separate thread
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

def main():
    rclpy.init()
    config_path = '/root/code/loopsplat_ros_ws/src/loopsplat_ros/loopsplat_ros/configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml'
    config = load_config(config_path)
    setup_seed(config["seed"])
    frontend = FrontEnd(config)
    try:
        # Start the spin thread for continuously handling callbacks
        spin_thread_instance = threading.Thread(target=spin_thread, args=(frontend,))
        spin_thread_instance.start()

        # Run the main logic (this will execute in parallel with message handling)
        frontend.run()
        
    finally:
        frontend.destroy_node()
        rclpy.shutdown()
        spin_thread_instance.join()  # Wait for the spin thread to finish
