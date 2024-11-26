from argparse import ArgumentParser
import os, glob
import copy
import time
import numpy as np
import torch, roma
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

from loopsplat_ros.src.entities.logger import Logger
from loopsplat_ros.src.entities.datasets import BaseDataset
from loopsplat_ros.src.entities.gaussian_model import GaussianModel
from loopsplat_ros.src.entities.arguments import OptimizationParams

from loopsplat_ros.src.gsr.descriptor import GlobalDesc, LocalDesc
from loopsplat_ros.src.gsr.camera import Camera
from loopsplat_ros.src.gsr.solver import gaussian_registration as gs_reg
from loopsplat_ros.src.gsr.pcr import (preprocess_point_cloud, execute_global_registration)

from loopsplat_ros.src.utils.utils import np2torch, torch2np
from loopsplat_ros.src.utils.graphics_utils import getProjectionMatrix2, focal2fov
from loopsplat_ros.src.utils.eval_utils import eval_ate

class PGO_Edge:
    def __init__(self, src_id, tgt_id, overlap=0.):
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.overlap_ratio = overlap
        self.success = False
        self.transformation = np.identity(4)
        self.information = np.identity(6)
        self.transformation_gt = np.identity(4)
        
    def __str__(self) -> str:
        return f"source_id : {self.s}, target_id : {self.t}, success : {self.success}, \
            transformation : {self.transformation}, information : {self.information}, \
            overlap_ratio : {self.overlap_ratio}, transformation_gt : {self.transformation_gt}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
class Loop_closure(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Initializes the LC with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the loop closure process.
        """
        self.device = "cuda"
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.netvlad = GlobalDesc()
        self.superpoint = LocalDesc()
        self.submap_lc_info = dict()
        self.submap_id = 0
        self.submap_path = None
        self.pgo_count = 0
        self.n_loop_edges = 0
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.proj_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx = self.config["cam"]["fx"],
                fy = self.config["cam"]["fx"],
                cx = self.config["cam"]["cx"],
                cy = self.config["cam"]["cy"],
                W = self.config["cam"]["W"],
                H = self.config["cam"]["H"],
            ).T
        self.fovx = focal2fov(self.config["cam"]["fx"], self.config["cam"]["W"])
        self.fovy = focal2fov(self.config["cam"]["fy"], self.config["cam"]["H"])
        self.min_interval = self.config['lc']['min_interval']
        
        # TODO: rename below
        self.config["Training"] = {"edge_threshold": 4.0}
        self.config["Dataset"] = {"type": "replica"}
        
        self.max_correspondence_distance_coarse = self.config['lc']['voxel_size'] * 15
        self.max_correspondence_distance_fine = self.config['lc']['voxel_size'] * 1.5
        
    def update_submaps_info(self, keyframes_info):
        """Update the submaps_info with current submap 

        Args:
            keyframes_info (dict): a dictionary of all submap information for loop closures
        """

        #Extract global and local features for each keyframe using hierarchical localization package
        with torch.no_grad():
            kf_ids, submap_desc, local_descriptors, keypoints, scores = [], [], [], [], []
            for key in keyframes_info.keys():
                np_color_img = self.dataset[key][1]
                np_grayscale_image = np.dot(np_color_img[..., :3], [0.2989, 0.5870, 0.1140])
                kps, local_desc, kp_scores = self.superpoint(np2torch(np_grayscale_image, self.device).unsqueeze(0)[None]/255.0)
                keypoints.append(kps[0])
                scores.append(kp_scores[0])
                local_descriptors.append(local_desc[0])
                global_desc = self.netvlad(np2torch(np_color_img, self.device).permute(2, 0, 1)[None]/255.0)
                submap_desc.append(global_desc)
            submap_desc = torch.cat(submap_desc)
            self_sim = torch.einsum("id,jd->ij", submap_desc, submap_desc)
            score_min, _ = self_sim.topk(max(int(len(submap_desc) * self.config["lc"]["min_similarity"]), 1))
            print("Updating Submap info...")
            print(f"self sim shape: {self_sim.shape}")
            print(f"score min shape: {score_min.shape}")
            
        self.submap_lc_info[self.submap_id] = {
                "submap_id": self.submap_id,
                "kf_ids": np.array(sorted(list(keyframes_info.keys()))),
                "keypoints": keypoints,
                "local_descriptors": local_descriptors,
                "keypoint_scores": scores,
                "global_desc": submap_desc,
                "self_sim": score_min, # per image self similarity within the submap
            }
    
    def submap_loader(self, id: int):
        """load submap data for loop closure

        Args:
            id (int): submap id to load
        """
        submap_dict = torch.load(self.submap_paths[id], map_location=torch.device(self.device))
        gaussians = GaussianModel(sh_degree=0)
        gaussians.restore_from_params(submap_dict['gaussian_params'], self.opt)
        submap_cams = []
        
        for kf_id in submap_dict['submap_keyframes']:
            _, rgb, depth, c2w_gt = self.dataset[kf_id]
            c2w_est = self.c2ws_est[kf_id]
            T_gt = torch.from_numpy(c2w_gt).to(self.device).inverse()
            T_est = torch.linalg.inv(c2w_est).to(self.device)
            cam_i = Camera(kf_id, None, None,
                   T_gt, 
                   self.proj_matrix, 
                   self.config["cam"]["fx"],
                   self.config["cam"]["fx"],
                   self.config["cam"]["cx"],
                   self.config["cam"]["cy"],
                   self.fovx, 
                   self.fovy, 
                   self.config["cam"]["H"], 
                   self.config["cam"]["W"])
            cam_i.R = T_est[:3, :3]
            cam_i.T = T_est[:3, 3]
            rgb_path = self.dataset.color_paths[kf_id]
            depth_path = self.dataset.depth_paths[kf_id]
            depth = np.array(Image.open(depth_path)) / self.config['cam']['depth_scale']
            cam_i.depth = depth
            cam_i.rgb_path = rgb_path
            cam_i.depth_path = depth_path
            cam_i.config = self.config
            submap_cams.append(cam_i)
        
        data_dict = {
            "submap_id": id,
            "gaussians": gaussians,
            "kf_ids": submap_dict['submap_keyframes'],
            "cameras": submap_cams,
            "global_desc": self.submap_lc_info[id]['global_desc']
            }
        
        return data_dict
    
    def detect_closure(self, query_id: int, final=False):
        """detect closure given a submap_id, we only match it to the submaps before it

        Args:
            query_id (int): the submap id used to detect closure

        Returns:
            torch.Tensor: 1d vector of matched submap_id
        """
        n_submaps = self.submap_id + 1
        query_info = self.submap_lc_info[query_id]
        iterator = range(query_id+1, n_submaps) if final else range(query_id)
        db_info_list = [self.submap_lc_info[i] for i in iterator]
        db_desc_map_id = []
        for db_info in db_info_list:
            db_desc_map_id += [db_info['submap_id'] for _ in db_info['global_desc']]
        db_desc_map_id = torch.Tensor(db_desc_map_id).to(self.device)
        
        query_desc = query_info['global_desc']
        db_desc = torch.cat([db_info['global_desc'] for db_info in db_info_list])
        
        with torch.no_grad():
            cross_sim = torch.einsum("id,jd->ij", query_desc, db_desc)
            self_sim = query_info['self_sim']
            print(f"Number of submaps {n_submaps}")
            print(f"query desc shape: {query_desc.shape}")
            print(f"DB desc shape: {db_desc.shape}")
            print("Cross sim and self sim are given below")
            print(f"cross sim shape: {cross_sim.shape}")
            print(f"score min shape: {self_sim.shape}")
            print("retrieving matches.....")
            
            
            # For each frame in the query submap, retrieve the frames where 
            # cross similarity with the query frames is greater than the median self similarity score of the query frame

            query_sim_median = self_sim[:,[-1]]
            
            #Row indicates query frame, column indicates frame in the database
            matches = torch.argwhere(cross_sim > query_sim_median)[:,-1] #Ignore row and return only column IDs
            print(f"Matched frame IDs: {matches}")
            matched_map_ids = db_desc_map_id[matches].long().unique()
            print(f"Matched submap IDs: {matched_map_ids}")
        
        # filter out invalid matches
        filtered_mask = abs(matched_map_ids - query_id) > self.min_interval
        matched_map_ids = matched_map_ids[filtered_mask]
        print(f"Candidate submap ids need to be at a minimum interval of {self.min_interval} from query id: {query_id}")
        print(f"Filtered Matched submap IDs: {matched_map_ids}")


        # Perform geometric verification if there is a submap match

        matcher_type = 'BF'
        if matcher_type == 'BF':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif matcher_type == 'FLANN':
            index_params = dict(algorithm=1, trees=5)  # FLANN parameters for KD-Tree
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Invalid matcher_type. Use 'BF' or 'FLANN'.")

        query_kf_ids = self.submap_lc_info[query_id]["kf_ids"]
        # Choose the first frame
        ind = 0
        query_kf_id = query_kf_ids[ind]
        query_details = (query_id, ind, query_kf_id)
        query_image = self.dataset[query_kf_id][1]
        curr_frame_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        curr_frame_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)


        kp_scores_query = self.submap_lc_info[query_id]["keypoint_scores"][ind]
        kp_query = self.submap_lc_info[query_id]["keypoints"][ind]
        kp_query_np = kp_query.cpu().numpy()
        kp_query_cv = [cv2.KeyPoint(x=kp[0], y=kp[1], size=int(kp_scores_query[i].item() * 50)) for i, kp in enumerate(kp_query_np)]
        ld_query = self.submap_lc_info[query_id]["local_descriptors"][ind]
        ld_query = ld_query.T.cpu().numpy().astype(np.float32)

        for mid_pt in matched_map_ids:
            
            mid = mid_pt.item()
            best_avg_distance = float('inf')
            
            for ind, fid in enumerate(self.submap_lc_info[mid]["kf_ids"]):
                matched_color_img = self.dataset[fid][1]
                matched_color_img_rgb = cv2.cvtColor(matched_color_img, cv2.COLOR_BGR2RGB)
                matched_color_img_gray = cv2.cvtColor(matched_color_img, cv2.COLOR_BGR2GRAY)
                
                kp_scores_2 = self.submap_lc_info[mid]["keypoint_scores"][ind]
                kp_2 = self.submap_lc_info[mid]["keypoints"][ind]
                kp_2_np = kp_2.cpu().numpy()
                kp_2_cv = [cv2.KeyPoint(x=kp[0], y=kp[1], size=int(kp_scores_2[i].item() * 50)) for i, kp in enumerate(kp_2_np)]
                ld_2 = self.submap_lc_info[mid]["local_descriptors"][ind]
                ld_2 = ld_2.T.cpu().numpy().astype(np.float32)

                # Match descriptors using BFMatcher
                matches = matcher.match(ld_query, ld_2)
                matching_details = (mid, ind, fid)

                # match_annotated_img = cv2.drawMatches(curr_frame_gray, kp_query_cv, matched_color_img_gray, kp_2_cv, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # side_by_side = cv2.hconcat([curr_frame_rgb, matched_color_img_rgb])
                # stacked_image = np.vstack((side_by_side, match_annotated_img))
                # cv2.imshow(f'Query Details: {query_details} and current matching frame details {matching_details}', stacked_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Find the key frame with the closest distance
                avg_distance = sum(match.distance for match in matches) / len(matches)

                # Update the best match
                if avg_distance < best_avg_distance:
                    best_avg_distance = avg_distance
                    best_ind = ind
                    best_kfid = fid
                    best_matched_color_img_rgb = matched_color_img_rgb
                    best_matched_color_img_gray = matched_color_img_gray
                    best_kp_2_cv = kp_2_cv
                    best_matches = matches
                    best_matching_details = matching_details

            match_annotated_img = cv2.drawMatches(curr_frame_gray, kp_query_cv, best_matched_color_img_gray, best_kp_2_cv, best_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            side_by_side = cv2.hconcat([curr_frame_rgb, best_matched_color_img_rgb])
            stacked_image = np.vstack((side_by_side, match_annotated_img))
            cv2.imshow(f'Query Details: {query_details} has best match with {best_matching_details}', stacked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                #cv2.imshow(f'Query KF ID: 0 in submap ID {query_id} has matched with keyframe id: 0 in Submap ID {mid}', side_by_side)
                #cv2.imshow(f'Keypoint Matches', match_annotated_img)
                #break


        return matched_map_ids

    def detect_closurev2(self, query_id: int, final=False):
        """detect closure given a submap_id, we only match it to the submaps before it

        Args:
            query_id (int): the submap id used to detect closure

        Returns:
            torch.Tensor: 1d vector of matched submap_id
        """
        n_submaps = self.submap_id + 1
        query_info = self.submap_lc_info[query_id]
        iterator = range(query_id+1, n_submaps) if final else range(query_id)
        db_info_list = [self.submap_lc_info[i] for i in iterator]
        db_desc_map_id = []
        for db_info in db_info_list:
            db_desc_map_id += [db_info['submap_id'] for _ in db_info['global_desc']]
        db_desc_map_id = torch.Tensor(db_desc_map_id).to(self.device)
        
        query_desc = query_info['global_desc']
        db_desc = torch.cat([db_info['global_desc'] for db_info in db_info_list])
        
        with torch.no_grad():
            cross_sim = torch.einsum("id,jd->ij", query_desc, db_desc)
            self_sim = query_info['self_sim']
            print(f"Number of submaps {n_submaps}")
            print(f"query desc shape: {query_desc.shape}")
            print(f"DB desc shape: {db_desc.shape}")
            print("Cross sim and self sim are given below")
            print(f"cross sim shape: {cross_sim.shape}")
            print(f"score min shape: {self_sim.shape}")
            print("retrieving matches.....")
            
            # For each frame in the query submap, retrieve the frames where 
            # cross similarity with the query frames is greater than the median self similarity score of the query frame

            query_sim_median = self_sim[:,[-1]]
            
            #Row indicates query frame, column indicates frame in the database
            scores = cross_sim[cross_sim > query_sim_median]
            matches = torch.argwhere(cross_sim > query_sim_median)[:,-1] #Ignore row and return only column IDs
            print(f"Matched frame IDs: {matches}")
            matched_map_ids = db_desc_map_id[matches].long()
            print(f"Matched submap IDs: {matched_map_ids.unique()}")
        # filter out invalid matches
        filtered_mask = abs(matched_map_ids - query_id) > self.min_interval
        matched_map_ids = matched_map_ids[filtered_mask]
        print(f"Candidate submap ids need to be at a minimum interval of {self.min_interval} from query id: {query_id}")
        print(f"Filtered Matched submap IDs: {matched_map_ids.unique()}")

        # Aggregate scores for the submap
        submap_scores = torch.zeros(n_submaps, device=self.device)
        if matched_map_ids.numel() > 0:
            for idx, submap_id in enumerate(matched_map_ids):
                submap_scores[submap_id] += scores[idx]
            best_submap_id = torch.argmax(submap_scores)
            best_submap_score = torch.max(submap_scores)
            print(f"Submap scoring is done... Submap scores: {submap_scores}")
            print(f"Best submap is identified as {best_submap_id} with a score pf {best_submap_score}")

            # Perform geometric verification if there is a submap match

            matcher_type = 'BF'
            if matcher_type == 'BF':
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            elif matcher_type == 'FLANN':
                index_params = dict(algorithm=1, trees=5)  # FLANN parameters for KD-Tree
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise ValueError("Invalid matcher_type. Use 'BF' or 'FLANN'.")

            # Find the best query and candidate keyframe pairs
            best_submap_id = best_submap_id.item()
            best_avg_distance = float('inf')
            query_kf_ids = self.submap_lc_info[query_id]["kf_ids"]

            # Iterate through the key frames in the query submap
            for qidx, query_kf_id in enumerate(query_kf_ids):
                query_details = (query_id, qidx, query_kf_id)
                query_image = self.dataset[query_kf_id][1]
                curr_frame_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
                curr_frame_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

                kp_scores_query = self.submap_lc_info[query_id]["keypoint_scores"][qidx]
                kp_query = self.submap_lc_info[query_id]["keypoints"][qidx]
                kp_query_np = kp_query.cpu().numpy()
                kp_query_cv = [cv2.KeyPoint(x=kp[0], y=kp[1], size=int(kp_scores_query[i].item() * 50)) for i, kp in enumerate(kp_query_np)]
                ld_query = self.submap_lc_info[query_id]["local_descriptors"][qidx]
                ld_query = ld_query.T.cpu().numpy().astype(np.float32)

                # Iterate through the key frames in the best candidate submap
                for idx, fid in enumerate(self.submap_lc_info[best_submap_id]["kf_ids"]):
                    matched_color_img = self.dataset[fid][1]
                    matched_color_img_rgb = cv2.cvtColor(matched_color_img, cv2.COLOR_BGR2RGB)
                    matched_color_img_gray = cv2.cvtColor(matched_color_img, cv2.COLOR_BGR2GRAY)

                    kp_scores_2 = self.submap_lc_info[best_submap_id]["keypoint_scores"][idx]
                    kp_2 = self.submap_lc_info[best_submap_id]["keypoints"][idx]
                    kp_2_np = kp_2.cpu().numpy()
                    kp_2_cv = [cv2.KeyPoint(x=kp[0], y=kp[1], size=int(kp_scores_2[i].item() * 50)) for i, kp in enumerate(kp_2_np)]
                    ld_2 = self.submap_lc_info[best_submap_id]["local_descriptors"][idx]
                    ld_2 = ld_2.T.cpu().numpy().astype(np.float32)

                    # Match descriptors using Brute Force Matcher
                    # Alternative approaches include using knn matcher or superglue
                    matches = matcher.match(ld_query, ld_2)
                    matching_details = (best_submap_id, idx, fid)

                    # Remove outliers and work only with inliers
                    points_query = np.float32([kp_query_cv[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    points_train = np.float32([kp_2_cv[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    H, mask = cv2.findHomography(points_query, points_train, cv2.RANSAC, 5.0) # Alternate approaches - Retrieve fundamental matrix
                    inliers = [matches[i] for i in range(len(mask)) if mask[i]]
                    outliers_by_geometry = [matches[i] for i in range(len(mask)) if not mask[i]]

                    # Find the key frame with the closest distance
                    avg_distance = sum(inlier.distance for inlier in inliers) / len(inliers)

                    # match_annotated_img = cv2.drawMatches(curr_frame_gray, kp_query_cv, matched_color_img_gray, kp_2_cv, inliers[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    # side_by_side = cv2.hconcat([curr_frame_rgb, matched_color_img_rgb])
                    # stacked_image = np.vstack((side_by_side, match_annotated_img))
                    # cv2.imshow(f'Query Details: {query_details} and current matching frame details {matching_details}', stacked_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #print(f"After comparing with query frame number: {qidx}, with candidate keyframe {idx} found {len(inliers)} key point matches and average distance is {avg_distance}")

                    # Update the best match
                    if avg_distance < best_avg_distance:
                        best_avg_distance = avg_distance
                        best_ind = idx
                        best_kfid = fid
                        best_matched_color_img_rgb = matched_color_img_rgb
                        best_matched_color_img_gray = matched_color_img_gray
                        best_kp_2_cv = kp_2_cv
                        best_matches = inliers
                        best_matching_details = matching_details
                        best_qidx = qidx
                        #print(f"This is the best match till now... Updating the details of the best match!!")
            print(f"Best match obtained: Query KFID: {best_qidx} and Candidate KFID: {best_kfid} have {len(best_matches)} with avg. distance {best_avg_distance}")

            query_kf_id = query_kf_ids[best_qidx]
            query_details = (query_id, best_qidx, query_kf_id)
            query_image = self.dataset[query_kf_id][1]
            curr_frame_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            curr_frame_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

            kp_scores_query = self.submap_lc_info[query_id]["keypoint_scores"][best_qidx]
            kp_query = self.submap_lc_info[query_id]["keypoints"][best_qidx]
            kp_query_np = kp_query.cpu().numpy()
            kp_query_cv = [cv2.KeyPoint(x=kp[0], y=kp[1], size=int(kp_scores_query[i].item() * 50)) for i, kp in enumerate(kp_query_np)]
            ld_query = self.submap_lc_info[query_id]["local_descriptors"][best_qidx]
            ld_query = ld_query.T.cpu().numpy().astype(np.float32)

            match_annotated_img = cv2.drawMatches(curr_frame_gray, kp_query_cv, best_matched_color_img_gray, best_kp_2_cv, best_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            side_by_side = cv2.hconcat([curr_frame_rgb, best_matched_color_img_rgb])
            stacked_image = np.vstack((side_by_side, match_annotated_img))
            cv2.imshow(f'Query Details: {query_details} has best match with {best_matching_details}', stacked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                #cv2.imshow(f'Query KF ID: 0 in submap ID {query_id} has matched with keyframe id: 0 in Submap ID {mid}', side_by_side)
                #cv2.imshow(f'Keypoint Matches', match_annotated_img)
                #break

        # To DO: For Map merging - Alternative is to use ICP with Gaussians (implemented in this file, take a look at graph construction for details)
        # Step 1: Find object points from Query key frame (Robot A) - converting pixel locations into 3D world coordinates using the camera intrinsics
        # Step 2: Find image points from candidate key frame (Robot B) - matching 2D keypoints present in the keyframe
        # Step 3: Apply PNP to recover pose - R and T
        # Step 4: Apply transformation on all submaps of the 

        return matched_map_ids.unique()

    def construct_pose_graph(self, final=False):
        """Build the pose graph from detected loops

        Returns:
            _type_: _description_
        """
        n_submaps = self.submap_id + 1
        pose_graph = o3d.pipelines.registration.PoseGraph()
        submap_list = []
        
        # initialize pose graph node from odometry with identity matrix
        for i in range(n_submaps):
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.identity(4)))
            submap_list.append(self.submap_loader(i))
        
        # log info for edge analysis
        self.cam_dict = dict()
        self.kf_ids, self.kf_submap_ids = [], []
        for submap in submap_list:
            for cam in submap['cameras']:
                self.kf_submap_ids.append(submap["submap_id"])
                self.kf_ids.append(cam.uid)
                self.cam_dict[cam.uid] = copy.deepcopy(cam)
        self.kf_submap_ids = np.array(self.kf_submap_ids)
        
        odometry_edges, loop_edges = [], []
        new_submap_valid_loop = False
        for source_id in tqdm(reversed(range(1, n_submaps))):
            matches = self.detect_closure(source_id, final)
            iterator = range(source_id+1, n_submaps) if final else range(source_id)
            for target_id in iterator:
                if abs(target_id - source_id)== 1: # odometry edge
                    reg_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "identity")
                    transformation = reg_dict['transformation']
                    information = reg_dict['information']
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation,
                                                                information,
                                                                uncertain=False))
                    # analyse 
                    gt_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "gt")
                    ae = roma.rotmat_geodesic_distance(torch.from_numpy(gt_dict['transformation'][:3,:3]), torch.from_numpy(reg_dict['transformation'][:3, :3])) * 180 /torch.pi
                    te = np.linalg.norm(gt_dict['transformation'][:3,3] - reg_dict["transformation"][:3,3])
                    odometry_edges.append((source_id, target_id, ae.item(), te.item()))
                    # TODO: update odometry edge with the PGO_edge class

                elif target_id in matches: # loop closure edge
                    reg_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "gs_reg")
                    if not reg_dict['successful']: continue
                    
                    if np.isnan(reg_dict["transformation"][:3,3]).any() or reg_dict["transformation"][3,3]!=1.0: continue
                    
                    # analyse 
                    gt_dict = self.pairwise_registration(submap_list[source_id], submap_list[target_id], "gt")
                    ae = roma.rotmat_geodesic_distance(torch.from_numpy(reg_dict['transformation'][:3,:3]), torch.from_numpy(gt_dict['transformation'][:3, :3])) * 180 /torch.pi
                    te = np.linalg.norm(gt_dict['transformation'][:3,3] - reg_dict["transformation"][:3,3])
                    loop_edges.append((source_id, target_id, ae.item(), te.item()))
                    
                    transformation = reg_dict['transformation']
                    information = reg_dict['information']
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation,
                                                                information,
                                                                uncertain=True))
                    new_submap_valid_loop = True
            
            if source_id == self.submap_id and not new_submap_valid_loop:
                break    
                
        return pose_graph, odometry_edges, loop_edges
    
    def loop_closure(self, estimated_c2ws, final=False):
        '''
        Compute loop closure correction
        
        returns: 
            None or the pose correction for each submap
        '''
        
        print("\nDetecting loop closures ...")
        # first see if current submap generates any new edge to the pose graph
        correction_list = []
        self.c2ws_est = estimated_c2ws.detach()        
        self.submap_paths = sorted(glob.glob(str(self.submap_path/"*.ckpt")), key=lambda x: int(x.split('/')[-1][:-5]))
        
        
        if self.submap_id<3 or len(self.detect_closure(self.submap_id)) == 0:
            print(f"\nNo loop closure detected at submap no.{self.submap_id}")
            return correction_list

        print(f"\nLoop closure detected at submap no.{self.submap_id}...")
        pose_graph, odometry_edges, loop_edges = self.construct_pose_graph(final)
        
        # save pgo edge analysis result
        
        if len(loop_edges)>0 and len(loop_edges) > self.n_loop_edges: 
            
            print("Optimizing PoseGraph ...")
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=self.max_correspondence_distance_fine,
                edge_prune_threshold=self.config['lc']['pgo_edge_prune_thres'],
                reference_node=0)
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
            
            self.pgo_count += 1
            self.n_loop_edges = len(loop_edges)
            
            for id in range(self.submap_id+1):
                submap_correction = {
                    'submap_id': id,
                    "correct_tsfm": pose_graph.nodes[id].pose}
                correction_list.append(submap_correction)
                
            self.analyse_pgo(odometry_edges, loop_edges, pose_graph)
        
        else:
            print("No valid loop edges or new loop edges. Skipping ...")
            
        return correction_list
    
    def analyse_pgo(self, odometry_edges, loop_edges, pose_graph):
        """analyse the results from pose graph optimization

        Args:
            odometry_edges (list): list of error in odometry edges
            loop_edges (list): list of error in loop edges
        """
        pgo_save_path = Path(self.config["data"]["output_path"])/"pgo"/str(self.pgo_count)
        
        print("Evaluating ATE before pgo ...")
        eval_ate(self.cam_dict, self.kf_ids, str(pgo_save_path/"before"), 100, final=True)
        
        corrected_cams = {}
        corrected_ids = []
        for i, node in enumerate(pose_graph.nodes):
            for kf_id in self.submap_lc_info[i]['kf_id']:
                cam = self.cam_dict[kf_id]
                updated_RT = (cam.get_T @ torch.from_numpy(node.pose).cuda().float().inverse())
                cam.update_RT(updated_RT[:3, :3], updated_RT[:3, 3])
                corrected_cams[cam.uid] = cam
                corrected_ids.append(cam.uid)
        print("Evaluating ATE after pgo ...")
        eval_ate(corrected_cams, self.kf_ids, str(pgo_save_path/"after_gs"), None, True)
        
        submap_cams = {}
        for i in range(len(self.submap_lc_info)):
            global_kf = self.cam_dict[self.submap_lc_info[i]['kf_id'][0]]
            corr_delta = global_kf.get_T.inverse() @ global_kf.get_T_gt
            for kf_id in self.submap_lc_info[i]['kf_id']:
                cam = self.cam_dict[kf_id]
                updated_RT = (cam.get_T @ corr_delta)
                cam.update_RT(updated_RT[:3, :3], updated_RT[:3, 3])
                submap_cams[cam.uid] = cam
        print("Evaluating ATE within each submap ...")
        eval_ate(corrected_cams, self.kf_ids, str(pgo_save_path/"submap"), None, True)
        
        # save edge errors
        odometry_re_errors = [edge[2] for edge in odometry_edges]
        loop_re_errors = [edge[2] for edge in loop_edges]
        odometry_te_errors = [edge[3]*100 for edge in odometry_edges]
        loop_te_errors = [edge[3]*100 for edge in loop_edges]

        # create plot for rotation errors  
        # Combine errors and create labels
        all_errors = odometry_re_errors + loop_re_errors
        all_edges = odometry_edges + loop_edges
        colors = ['blue'] * len(odometry_re_errors) + ['orange'] * len(loop_re_errors)
        labels = ['Odometry'] * len(odometry_re_errors) + ['Loop Closure'] * len(loop_re_errors)

        # Calculate the medians
        median_odometry_error = np.median(odometry_re_errors)
        median_loop_error = np.median(loop_re_errors)

        # Create bar plot
        plt.figure(figsize=(12, 6))

        # Plot each error as a separate bar
        for i in range(len(all_errors)):
            plt.bar(i, all_errors[i], color=colors[i], label=labels[i] if i == 0 or labels[i] != labels[i-1] else "")

        # Plot the median lines
        plt.axhline(y=median_odometry_error, color='blue', linestyle='--', label=f'Median Odometry Error: {median_odometry_error:.2f} degrees')
        plt.axhline(y=median_loop_error, color='orange', linestyle='--', label=f'Median Loop Error: {median_loop_error:.2f} degrees')

        # Add labels, title, and legend
        plt.xlabel('Edges')
        plt.ylabel('Error (degrees)')
        plt.title('Odometry and Loop Closure Edge Errors with Medians')
        plt.legend()

        # Set x-ticks to show edge labels
        plt.xticks(range(len(all_errors)), labels, rotation=90)

        plt.tight_layout()
        plot_filename = pgo_save_path/"submap_all_edge_ae.png"
        plt.savefig(plot_filename)
        
        # create plot for translational errors  
        # Combine errors and create labels
        all_errors = odometry_te_errors + loop_te_errors
        all_edges = odometry_edges + loop_edges
        colors = ['blue'] * len(odometry_re_errors) + ['orange'] * len(loop_re_errors)
        labels = ['Odometry'] * len(odometry_re_errors) + ['Loop Closure'] * len(loop_re_errors)

        # Calculate the medians
        median_odometry_error = np.median(odometry_te_errors)
        median_loop_error = np.median(loop_te_errors)

        # Create bar plot
        plt.figure(figsize=(12, 6))

        # Plot each error as a separate bar
        for i in range(len(all_errors)):
            plt.bar(i, all_errors[i], color=colors[i], label=labels[i] if i == 0 or labels[i] != labels[i-1] else "")

        # Plot the median lines
        plt.axhline(y=median_odometry_error, color='blue', linestyle='--', label=f'Median Odometry Error: {median_odometry_error:.2f} cm')
        plt.axhline(y=median_loop_error, color='orange', linestyle='--', label=f'Median Loop Error: {median_loop_error:.2f} cm')

        # Add labels, title, and legend
        plt.xlabel('Edges')
        plt.ylabel('Error (cm)')
        plt.title('Odometry and Loop Closure Edge Errors with Medians')
        plt.legend()

        # Set x-ticks to show edge labels
        plt.xticks(range(len(all_errors)), labels, rotation=90)

        plt.tight_layout()
        plot_filename = pgo_save_path/"submap_all_edge_te.png"
        plt.savefig(plot_filename)
        return
    
    def submap_to_segment(self, submap):
        segment = {
            "points": submap['gaussians'].get_xyz().detach().cpu(),
            "keyframe": submap['cameras'][0].get_T.detach().cpu(),
            "gt_camera": submap['cameras'][0].get_T_gt.detach().cpu(),
        }
        return segment

    def pairwise_registration(self, submap_source, submap_target, method="gs_reg"):

        segment_source = self.submap_to_segment(submap_source)
        segment_target = self.submap_to_segment(submap_target)
        max_correspondence_distance_coarse = 0.3
        max_correspondence_distance_fine = 0.03

        source_points = segment_source["points"]
        target_points = segment_target["points"]

        # source_colors = segment_source["points_color"]
        # target_colors = segment_target["points_color"]

        cloud_source = o3d.geometry.PointCloud()
        cloud_source.points = o3d.utility.Vector3dVector(np.array(source_points))
        # cloud_source.colors = o3d.utility.Vector3dVector(np.array(source_colors))
        cloud_source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
        keyframe_source = segment_source["keyframe"]
        camera_location_source = keyframe_source[:3, 3].cpu().numpy()
        cloud_source.orient_normals_towards_camera_location(
            camera_location=camera_location_source)

        cloud_target = o3d.geometry.PointCloud()
        cloud_target.points = o3d.utility.Vector3dVector(np.array(target_points))
        # cloud_target.colors = o3d.utility.Vector3dVector(np.array(target_colors))
        cloud_target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
        keyframe_target = segment_target["keyframe"]
        camera_location_target = keyframe_target[:3, 3].cpu().numpy()
        cloud_target.orient_normals_towards_camera_location(
            camera_location=camera_location_target)
        
        output = dict()
        if method == "gt":
            gt_source = segment_source["gt_camera"]
            gt_target = segment_target["gt_camera"]
            delta_src = gt_source.inverse() @ keyframe_source
            delta_tgt = gt_target.inverse() @ keyframe_target
            delta = delta_tgt.inverse() @ delta_src
            output["transformation"] = np.array(delta)
        elif method == "icp":
            icp_coarse = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_coarse, np.identity(
                    4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            icp_fine = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_fine,
                icp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            delta = icp_fine.transformation
            output["transformation"] = np.array(delta)
        elif method == "robust_icp":
            voxel_size = 0.04
            sigma = 0.01

            source_down, source_fpfh = preprocess_point_cloud(
                cloud_source, voxel_size, camera_location_source)
            target_down, target_fpfh = preprocess_point_cloud(
                cloud_target, voxel_size, camera_location_target)
            
            tic = time.perf_counter()

            result_ransac = execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size)

            loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
            icp_fine = o3d.pipelines.registration.registration_icp(
                cloud_source, cloud_target, max_correspondence_distance_fine,
                result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss))
            
            toc = time.perf_counter()
            delta = icp_fine.transformation

            # compute success to gt delta
            gt_source = segment_source["gt_camera"]
            gt_target = segment_target["gt_camera"]
            rel_gt = gt_source@gt_target.inverse()
            delta_gt = rel_gt@keyframe_target@keyframe_source.inverse()
            output["transformation_gt_mag"] = torch.abs(delta_gt).mean().item()
            output["transformation_mag"] = torch.abs(
                torch.tensor(delta)).mean().item()
            output["transformation"] = np.array(delta)
            output["fitness"] = icp_fine.fitness
            output["inlier_rmse"] = icp_fine.inlier_rmse
            output["registration_time"] = toc-tic

        elif method == "identity":
            delta = np.identity(4)
            output["transformation"] = delta

        elif method == "gs_reg":
            res = gs_reg(submap_source, submap_target, self.config['lc']['registration'])
            delta = res['pred_tsfm'].cpu().numpy()
            output["transformation"] = delta
            output["successful"] = res["successful"]
                
        else:
            raise NotImplementedError("Unknown registration method!")

        output["information"] = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            cloud_source,
            cloud_target,
            max_correspondence_distance_fine,
            np.array(delta)
        )

        output["n_points"] = min(len(cloud_source.points),
                                len(cloud_target.points))
        output['pc_src'] = cloud_source
        output['pc_tgt'] = cloud_target
        return output
    