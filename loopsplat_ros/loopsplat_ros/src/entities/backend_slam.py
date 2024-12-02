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


class BackEnd(Node):

    def __init__(self, config: dict) -> None:
        super().__init__('backend_slam_node')

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config
        self.pipeline_params = munchify(config["pipeline_params"])
        self.gaussian_map = {}

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
        self.b2f_publisher = self.create_publisher(B2F,'Back2Front',self.queue_size)

        self.f2b_subscriber = self.create_subscription(F2B, '/Front2Back', self.f2b_listener_callback, self.queue_size)
        self.f2b_subscriber  # prevent unused variable warning

        
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

    def save_current_submap(self):
        """Saving the current submap's checkpoint and resetting the Gaussian model
        """
        
        gaussian_params = self.gaussian_model.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
    
    def start_new_submap(self, frame_id: int):
        """ Initializes a new submap.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
        """
        
        self.gaussian_model = GaussianModel(0)
        self.gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.keyframes_info = {}
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
        self.mapping_frame_ids.append(frame_id) if frame_id not in self.mapping_frame_ids else self.mapping_frame_ids
        self.submap_id += 1
        self.loop_closer.submap_id += 1
        return
    
    def rigid_transform_gaussians(self, gaussian_params, tsfm_matrix):
        '''
        Apply a rigid transformation to the Gaussian parameters.
        
        Args:
            gaussian_params (dict): Dictionary containing Gaussian parameters.
            tsfm_matrix (torch.Tensor): 4x4 rigid transformation matrix.
            
        Returns:
            dict: Updated Gaussian parameters after applying the transformation.
        '''
        # Transform Gaussian centers (xyz)
        tsfm_matrix = torch.from_numpy(tsfm_matrix).float()
        xyz = gaussian_params['xyz']
        pts_ones = torch.ones((xyz.shape[0], 1))
        pts_homo = torch.cat([xyz, pts_ones], dim=1)
        transformed_xyz = (tsfm_matrix @ pts_homo.T).T[:, :3]
        gaussian_params['xyz'] = transformed_xyz

        # Rotate covariance matrix (rotation)
        rotation = gaussian_params['rotation']
        cur_rot = roma.unitquat_to_rotmat(rotation)
        rot_mat = tsfm_matrix[:3, :3].unsqueeze(0)  # Adding batch dimension
        new_rot = rot_mat @ cur_rot
        new_quat = roma.rotmat_to_unitquat(new_rot)
        gaussian_params['rotation'] = new_quat.squeeze()

        return gaussian_params
        
    def update_keyframe_poses(self, lc_output, submaps_kf_ids, cur_frame_id):
        '''
        Update the keyframe poses using the correction from pgo, currently update the frame range that covered by the keyframes.
        
        '''
        for correction in lc_output:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']
            submap_kf_ids = submaps_kf_ids[submap_id]
            min_id, max_id = min(submap_kf_ids), max(submap_kf_ids)
            self.estimated_c2ws[min_id:max_id + 1] = torch.from_numpy(correct_tsfm).float() @ self.estimated_c2ws[min_id:max_id + 1]
        
        # last tracked frame is based on last submap, update it as well
        self.estimated_c2ws[cur_frame_id] = torch.from_numpy(lc_output[-1]['correct_tsfm']).float() @ self.estimated_c2ws[cur_frame_id]

    def apply_correction_to_submaps(self, correction_list):
        submaps_kf_ids= {}
        for correction in correction_list:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']

            submap_ckpt_name = str(submap_id).zfill(6) + ".ckpt"
            submap_ckpt = torch.load(self.output_path / "submaps" / submap_ckpt_name)
            submaps_kf_ids[submap_id] = submap_ckpt["submap_keyframes"]

            gaussian_params = submap_ckpt["gaussian_params"]
            updated_gaussian_params = self.rigid_transform_gaussians(
                gaussian_params, correct_tsfm)

            submap_ckpt["gaussian_params"] = updated_gaussian_params
            torch.save(submap_ckpt, self.output_path / "submaps" / submap_ckpt_name)
        return submaps_kf_ids

    def run(self) -> None:

        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        self.gaussian_model = GaussianModel(0)
        self.gaussian_model.training_setup(self.opt)
        self.submap_id = 0

        while True:
            time.sleep(0.1)
            continue

    def update_map(self, frame_id, estimated_c2w, exposure_ab, should_start_new_submap) -> None:

        self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

        if should_start_new_submap:
            # first save current submap and its keyframe info
            self.save_current_submap()
            
            # update submap infomation for loop closer
            self.loop_closer.update_submaps_info(self.keyframes_info)
            
            # apply loop closure
            lc_output = self.loop_closer.loop_closure(self.estimated_c2ws)
            
            if len(lc_output) > 0:
                submaps_kf_ids = self.apply_correction_to_submaps(lc_output)
                self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)
                # Update keyframes (their poses) and previously computed gaussians
                # Need to send updated keyframes and gaussians to GUI
            
            save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
            self.gaussian_map[self.submap_id] = self.gaussian_model
            
            self.start_new_submap(frame_id)

        if frame_id in self.mapping_frame_ids:
            print("\nMapping frame", frame_id)
            self.gaussian_model.training_setup(self.opt, exposure_ab) 
            estimated_c2w = torch2np(self.estimated_c2ws[frame_id])
            new_submap = not bool(self.keyframes_info)
            opt_dict = self.mapper.map(
                frame_id, estimated_c2w, self.gaussian_model, new_submap, exposure_ab)

            # Keyframes info update
            self.keyframes_info[frame_id] = {
                "keyframe_id": frame_id, 
                "opt_dict": opt_dict,
            }
            if self.enable_exposure:
                self.keyframes_info[frame_id]["exposure_a"] = exposure_ab[0].item()
                self.keyframes_info[frame_id]["exposure_b"] = exposure_ab[1].item()
        
        if frame_id == len(self.dataset) - 1 and self.config['lc']['final']:
            print("\n Final loop closure ...")
            self.loop_closer.update_submaps_info(self.keyframes_info)
            lc_output = self.loop_closer.loop_closure(self.estimated_c2ws, final=True)
            if len(lc_output) > 0:
                submaps_kf_ids = self.apply_correction_to_submaps(lc_output)
                self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)

        self.publish_message_to_frontend(frame_id, "sync")


    def f2b_listener_callback(self, f2b_msg):
        self.get_logger().info('I heard from frontend: %s' % f2b_msg.msg)
        frame_id = f2b_msg.frame_id
        estimated_c2w = convert_ros_multi_array_message_to_numpy(f2b_msg.estimated_c2w)
        if self.enable_exposure:
            exposure_a = f2b_msg.exposure_a
            exposure_b = f2b_msg.exposure_b
        else:
            exposure_a = 0.0
            exposure_b = 0.0
        exposure_ab = torch.nn.Parameter(torch.tensor(exposure_a, device="cuda")), torch.nn.Parameter(torch.tensor(exposure_b, device="cuda"))
        should_start_new_submap = f2b_msg.should_start_new_submap
        self.update_map(frame_id, estimated_c2w, exposure_ab, should_start_new_submap)

    def publish_message_to_frontend(self, frame_id, tag="sync", submap_id=0, min_id=0, max_id=0):
        b2f_msg = self.convert_to_b2f_ros_msg(frame_id, tag, submap_id, min_id, max_id)
        self.get_logger().info(f'Publishing to Frontend Node: {self.msg_counter}')
        self.b2f_publisher.publish(b2f_msg)
        self.msg_counter += 1

    def convert_to_b2f_ros_msg(self, frame_id, tag, submap_id, min_id, max_id):

        b2f_msg = B2F()
        b2f_msg.msg = tag
        b2f_msg.frame_id = frame_id

        #Gaussian part of the message
        b2f_msg.gaussian.active_sh_degree = 0

        b2f_msg.gaussian.max_sh_degree = self.gaussian_model.max_sh_degree
        # np_arr = np.random.rand(2,3)
        # tensor_msg = torch.from_numpy(np_arr)
        b2f_msg.gaussian.xyz = convert_tensor_to_ros_message(self.gaussian_model._xyz)
        b2f_msg.gaussian.features_dc = convert_tensor_to_ros_message(self.gaussian_model._features_dc)
        b2f_msg.gaussian.features_rest = convert_tensor_to_ros_message(self.gaussian_model._features_rest)
        b2f_msg.gaussian.scaling = convert_tensor_to_ros_message(self.gaussian_model._scaling)
        b2f_msg.gaussian.rotation = convert_tensor_to_ros_message(self.gaussian_model._rotation)
        b2f_msg.gaussian.opacity = convert_tensor_to_ros_message(self.gaussian_model._opacity)
        b2f_msg.gaussian.max_radii2d = self.gaussian_model.max_radii2D.tolist()
        b2f_msg.gaussian.xyz_gradient_accum = convert_tensor_to_ros_message(self.gaussian_model.xyz_gradient_accum)

        # If loopclosure tag then update the poses of keyframes in the submap
        if tag == "loopclosure":
            b2f_msg.submap_id = submap_id
            b2f_msg.num_kfs = max_id - min_id + 1
            b2f_msg.min_id = min_id
            b2f_msg.max_id = max_id
            b2f_msg.submap_kf_poses = convert_tensor_to_ros_message(self.estimated_c2ws[min_id:max_id + 1])

        return b2f_msg

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
    backend = BackEnd(config)
    try:
        # Start the spin thread for continuously handling callbacks
        spin_thread_instance = threading.Thread(target=spin_thread, args=(backend,))
        spin_thread_instance.start()

        # Run the main logic (this will execute in parallel with message handling)
        backend.run()
        
    finally:
        backend.destroy_node()
        rclpy.shutdown()
        spin_thread_instance.join()  # Wait for the spin thread to finish
