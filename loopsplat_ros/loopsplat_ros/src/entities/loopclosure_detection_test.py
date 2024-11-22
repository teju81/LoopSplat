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
from loopsplat_interfaces.msg import F2G, CameraMsg, GaussiansMsg

from loopsplat_ros.src.utils.io_utils import load_config

import numpy as np
import torch
import roma
import cv2

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


class GaussianSLAM(Node):

    def __init__(self, config: dict) -> None:
        super().__init__('loopclosure_detection_test_node')

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
        self.f2g_publisher = self.create_publisher(F2G,'Front2GUI',self.queue_size)
        
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

    def start_new_submap(self, frame_id: int) -> None:
        """ Initializes a new submap.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
        """

        self.mapper.keyframes = []
        self.keyframes_info = {}
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
        self.mapping_frame_ids.append(frame_id) if frame_id not in self.mapping_frame_ids else self.mapping_frame_ids
        self.submap_id += 1
        self.loop_closer.submap_id += 1
        return

    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        self.submap_id = 0

        for frame_id in range(len(self.dataset)):
            print(f"Processing frame id: {frame_id}...")
            estimated_c2w = self.dataset[frame_id][-1]
            curr_frame = self.dataset[frame_id][1]
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            # Reinitialize gaussian model for new segment
            if self.should_start_new_submap(frame_id) or (frame_id == len(self.dataset) - 1 and self.config['lc']['final']):
                
                # update submap infomation for loop closer
                self.loop_closer.update_submaps_info(self.keyframes_info)
                
                # Detect loop closure
                if self.submap_id<3:
                        print(f"\nNo loop closure detected at submap no.{self.submap_id}")
                else:
                    matched_loop_closure_submap_ids = self.loop_closer.detect_closure(self.submap_id)
                    if len(matched_loop_closure_submap_ids) == 0:
                        print(f"\nNo loop closure detected at submap no.{self.submap_id}")
                    else:
                        print("Loop closure detected")
                        print(f"Matching submap ids: {matched_loop_closure_submap_ids}")

                        # # Visualize one keyframe from each submap id
                        # curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                        # cv2.imshow(f'Query KF in submap ID {self.submap_id}', curr_frame_rgb)
                        # for mid_pt in matched_loop_closure_submap_ids:
                        #     mid = mid_pt.item()
                        #     print(self.loop_closer.submap_lc_info[mid]['kf_ids'])
                        #     fid = self.loop_closer.submap_lc_info[mid]["kf_ids"][0]
                        #     matched_color_img = self.dataset[fid][1]
                        #     matched_color_img_rgb = cv2.cvtColor(matched_color_img, cv2.COLOR_BGR2RGB)
                        #     cv2.imshow(f'Matched Submap ID {mid}', matched_color_img_rgb)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                self.start_new_submap(frame_id)

            if frame_id in self.mapping_frame_ids:
                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": frame_id
                }

def spin_thread(node):
    # Spin the node continuously in a separate thread
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

def main():
    rclpy.init()
    config_path = '/root/code/loopsplat_ros_ws/src/loopsplat_ros/loopsplat_ros/configs/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household.yaml'
    config = load_config(config_path)
    setup_seed(config["seed"])
    gslam = GaussianSLAM(config)
    try:
        # Start the spin thread for continuously handling callbacks
        spin_thread_instance = threading.Thread(target=spin_thread, args=(gslam,))
        spin_thread_instance.start()

        # Run the main logic (this will execute in parallel with message handling)
        gslam.run()
        
    finally:
        gslam.destroy_node()
        rclpy.shutdown()
        spin_thread_instance.join()  # Wait for the spin thread to finish
