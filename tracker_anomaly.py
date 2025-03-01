from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lap
import gc
import contextlib
import io
import tempfile
import yaml
from loguru import logger
from fast_reid.fast_reid_interfece import FastReIDInterface
import argparse
import os
import shutil
from PIL import Image as Img
import cv2
import torch
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from numpy import random
import numpy as np
from PIL import Image, ImageTk
import time
import math
from subprocess import *
import copy
import os.path as osp
import json
from itertools import permutations
#################################################### detector related import start ######################################################################
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
##### yolox #####
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.ssp_tracker import SSPTracker
from yolox.tracker import matching
from tracker.tracking_utils.timer import Timer
from tracker.gmc import GMC
#################################################### detector related import end ########################################################################

current_video_segment_predicted_tracks = {}
current_video_segment_predicted_tracks_bboxes = {}
current_video_segment_representative_frames = {}
current_video_segment_all_traj_all_object_features = {}
previous_video_segment_all_traj_all_object_features = {}
previous_video_segment_predicted_tracks = {}
previous_video_segment_predicted_tracks_bboxes = {}
previous_video_segment_representative_frames = {}
current_video_segment_predicted_tracks_backup = {}
current_video_segment_predicted_tracks_bboxes_backup = {}
current_video_segment_representative_frames_backup = {}
current_video_segment_all_traj_all_object_features_backup = {}
stitching_tracklets_dict = {}
tracklet_pose_collection_large_temporal_stride_buffer = []
tracklet_len = 10 # 滑动窗口长度
median_filter_radius = 4
num_samples_around_each_joint = 3
maximum_possible_number = math.exp(10)
average_sampling_density_hori_vert = 7
bbox_confidence_threshold = 0.6 #5 # 0.45
gap = 60  # gap设置太大会把最后一段轨迹给分开来
head_bbox_confidence_threshold = 0.55 # 0.6 # 0.45
temporal_length_thresh_inside_tracklet = 5
tracklet_confidence_threshold = 0.6
update_representative_frames_confidence_threshold = 0.9
spatialconsistency_reidsimilarity_debate_thresh = 0.7
large_temporal_stride_thresh = 100
stitching_tracklets_sure_or_not_last_time = [0]
stepback_adjustment_stride_thresh = 100
maximum_number_people = 3
reid_thresh = 0.98
trajectory_intersection_thresh = 0.9
split_single_trajectory_thresh = 0.01
max_movement_between_adjacent_frames = 27.865749586185547 * 2
image_height = 480
image_width = 856
body_head_width_ratio_factor = 3
num_frames_for_traj_prediction = 10 # Under severe motion, the movements are unpredictable, so use shorter historical trajs
batch_id = 0
frame_height = 0
frame_width = 0
batch_stride = tracklet_len - 1
batch_stride_write = tracklet_len - 1
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# Global
trackerTimer = Timer()
timer = Timer()
import sys
sys.path.insert(0,"/home/allenyljiang/Documents/SSP_EM/yolox/")
def write_vis_results(img_path,vis_folder, results):
    '''
    每张图片进行写入
    '''
    dst_dir = os.path.join(vis_folder, img_path.split('/')[-2],img_path.split('/')[-1])
    if not os.path.exists(os.path.dirname(dst_dir)):
        os.makedirs(os.path.dirname(dst_dir),exist_ok=True)
    for frame_id, tlwhs, track_ids, scores in [results[-1]]: # write each frame information, only write the last frame
        curr_img = cv2.imread(img_path)
        for xyxy, track_id, score in zip(tlwhs, track_ids, scores):
            if track_id < 0:
                continue
            x1, y1, x2, y2 = xyxy
            left, top = int(x1), int(y1)
            right, bottom = int(x2), int(y2)
            if score > 0.6:
                cv2.putText(curr_img, str(track_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0 , 255), 2) # (字体大小i，颜色，字体粗细)
            else:
                cv2.putText(curr_img, str(track_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imwrite(dst_dir,curr_img)
def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for xyxy, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = xyxy
                w,h = abs(x2-x1),abs(y2-y1)
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))
def multi_gmc(dets, H=np.eye(2, 3)):
    if len(dets) > 0:

        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        t = H[:2, 2]  # Translation part

        for i,det in enumerate(dets):
            det[1] -= det[0]
            det = det.flatten()
            det = R8x8[:4,:4].dot(det)
            det[:2] += t
            det[1] += det[0]
            dets[i,:] = det[:].reshape(2,2)
        return dets
def warp_pos_torch(pos, warp_matrix=np.eye(2, 3, dtype=np.float32)):
    '''
    Args:
        pos: (x1y1x2y2),tensor
        warp_matrix: (2*3)matrix,tensor

    Returns:

    '''
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1) # (top,left)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1) # (right,bottom)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2) # torch.mm 矩阵相乘
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()
def warp_pos(pos, warp_matrix=np.eye(2, 3, dtype=np.float32)):
    '''
    Args:
        pos: (x1y1x2y2),tensor
        warp_matrix: (2*3)matrix,tensor

    Returns:

    '''
    p1 = np.array([pos[0, 0], pos[0, 1], 1]).reshape((3, 1)) # (top,left)
    p2 = np.array([pos[1, 0], pos[1, 1], 1]).reshape((3, 1)) # (right,bottom)
    p1_n = np.matmul(warp_matrix, p1).reshape((1, 2)) # torch.mm 矩阵相乘
    p2_n = np.matmul(warp_matrix, p2).reshape((1, 2))
    warp_box = [(p1_n[0][0],p1_n[0][1]),(p2_n[0][0],p2_n[0][1])]
    return warp_box
def track_processing(split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features,split_each_track_valid_mask):
    #iou_thresh,iou_thresh_step = statistic[0],statistic[1]
    curr_predicted_tracks = {}
    curr_predicted_tracks_bboxes = {}
    curr_predicted_tracks_bboxes_test = {} # 测试使用
    curr_predicted_tracks_confidence_score = {}
    curr_representative_frames = {}
    mapping_frameid_to_human_centers = {}  # 暂存curr_predicted_tracks的value
    mapping_frameid_to_bbox = {}
    mapping_frameid_to_confidence_score = {}
    trajectory_node_dict = {}
    trajectory_idswitch_dict = {}
    trajectory_idswitch_reliability_dict = {} # 保存每条轨迹切断的每一段置信度  key:trajectory id value:list eg[1,3,5]  the sum of node_valid_mask
    trajectory_segment_nodes_dict = {}
    for track_id in split_each_track:
        # curr_predicted_tracks[track_id] = mapping_frameid_to_human_centers
        confidence_score_max = 0
        node_id_max = 0
        # print(track_id,' started!' )
        mapping_track_time_to_bbox = {}
        trajectory_node_list = []
        trajectory_idswitch_list = []
        trajectory_idswitch_reliability_list = []
        # trajectory_similarity_list = []
        trajectory_idswitch_reliability = 0
        trajectory_segment_list = []
        trajectory_segment = []
        for idx,node_pair in enumerate(split_each_track[track_id]):  # node，edge
            # if int(node_pair[1]) % 2 == 0:
            if idx % 2 == 0: # 偶数位置表示人的node
                node_id = int(node_pair[1] )/ 2
                trajectory_node_list.append(int(node_id))
                # print(node_id)
            else:
                continue
            # mapping_node_id_to_bbox[mapping_node_id_to_bbox.index(int(node_pair[0]))][2] # str:img
            frame_id = mapping_node_id_to_bbox[node_id][2]  # 转化为int进行加减
            bbox = mapping_node_id_to_bbox[node_id][0]
            if split_each_track_valid_mask[track_id][idx] != 1: # 该节点无效
                trajectory_idswitch_list.append(int(idx/2)) # id从0开始
                trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
                trajectory_segment_list.append(trajectory_segment[:])
                trajectory_idswitch_reliability = 0
                trajectory_segment = []
            elif split_each_track_valid_mask[track_id][idx] == 1:
                trajectory_idswitch_reliability += 1
                trajectory_segment.append(int(node_id))
            confidence_score = mapping_node_id_to_bbox[node_id][1]
            if confidence_score > confidence_score_max:
                confidence_score_max = confidence_score
                node_id_max = node_id
            mapping_frameid_to_human_centers[int(frame_id.split('.')[0])] = [(bbox[0][0] + bbox[1][0]) / 2,
                                                                             (bbox[0][1] + bbox[1][1]) / 2]  # 同一帧中图片相连?
            mapping_frameid_to_bbox[frame_id] = bbox
            # mapping_frameid_to_bbox[frame_id] = [bbox,confidence_score]
            mapping_frameid_to_confidence_score[frame_id] = confidence_score
            mapping_track_time_to_bbox[int(node_id)] = [frame_id,bbox,confidence_score]
            # current_video_segment_all_traj_all_object_features[track_id] = [[node_id], mapping_node_id_to_features[node_id]]  # ???
        trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
        trajectory_segment_list.append(trajectory_segment)
        trajectory_node_dict[track_id] = copy.deepcopy(trajectory_node_list)
        trajectory_idswitch_dict[track_id] = copy.deepcopy(trajectory_idswitch_list)
        trajectory_idswitch_reliability_dict[track_id] = copy.deepcopy(trajectory_idswitch_reliability_list)
        trajectory_segment_nodes_dict[track_id] = copy.deepcopy(trajectory_segment_list)
        curr_predicted_tracks[track_id] = copy.deepcopy(mapping_frameid_to_human_centers) # 直接等于之后操作会影响到curr_predicted_tracks
        curr_predicted_tracks_bboxes[track_id] = copy.deepcopy(mapping_frameid_to_bbox)
        curr_predicted_tracks_bboxes_test[track_id] = copy.deepcopy(mapping_track_time_to_bbox)
        curr_predicted_tracks_confidence_score[track_id] = copy.deepcopy(mapping_frameid_to_confidence_score)
        # 可能刚好在第一个
        if node_id_max == 0:
            node_id_max =  list(mapping_track_time_to_bbox.keys())[0]
        curr_representative_frames[track_id] = [node_id_max,(bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0]),mapping_node_id_to_features[node_id_max]]  # 高度,宽度
        mapping_frameid_to_human_centers.clear()
        mapping_frameid_to_bbox.clear()
        mapping_frameid_to_confidence_score.clear()
    return curr_predicted_tracks_bboxes_test,trajectory_node_dict,trajectory_idswitch_dict,trajectory_idswitch_reliability_dict,trajectory_segment_nodes_dict


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes # 1
        self.confthre = exp.test_conf # 0.09
        self.nmsthre = exp.nmsthre # 0.7
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        # ImageNet均值和方差
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img) # '000003.jpg'
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std) # img:(1080,1920,3)--(3,896,1600)
        # imagenet 数据集均值和方差  mean = [0.485,0.456,0.406]
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img) # (1,29400,6)
            outputs = postprocess(outputs, self.num_classes,self.confthre, self.nmsthre) # (40,7),self.confthre = 0.09,

        return outputs, img_info

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int
    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def compute_iou_single_box(curr_img_boxes, next_img_boxes):# Order: top, bottom, left, right
    intersect_vert = min([curr_img_boxes[1], next_img_boxes[1]]) - max([curr_img_boxes[0], next_img_boxes[0]])
    intersect_hori = min([curr_img_boxes[3], next_img_boxes[3]]) - max([curr_img_boxes[2], next_img_boxes[2]])
    union_vert = max([curr_img_boxes[1], next_img_boxes[1]]) - min([curr_img_boxes[0], next_img_boxes[0]])
    union_hori = max([curr_img_boxes[3], next_img_boxes[3]]) - min([curr_img_boxes[2], next_img_boxes[2]])
    if intersect_vert > 0 and intersect_hori > 0 and union_vert > 0 and union_hori > 0:
        corresponding_coefficient = float(intersect_vert) * float(intersect_hori) / (float(curr_img_boxes[1] - curr_img_boxes[0]) * float(curr_img_boxes[3] - curr_img_boxes[2]) + float(next_img_boxes[1] - next_img_boxes[0]) * float(next_img_boxes[3] - next_img_boxes[2]) - float(intersect_vert) * float(intersect_hori))
    else:
        corresponding_coefficient = 0.0
    return corresponding_coefficient

def convert_node_ids(mapping_node_id_to_bbox, mapping_edge_id_to_cost):
    src_id = 1
    dst_id = 2 * max(mapping_node_id_to_bbox) + 2 # 其余node变成了2×node，2×node+1
    result_mapping_node_id_to_bbox = {}
    result_mapping_node_id_to_bbox_str = ''
    # Allen: make sure that even a traj with only 2 nodes and one edge can be involved
    most_unreliable_edge_cost = max([mapping_edge_id_to_cost[x] for x in mapping_edge_id_to_cost])
    even_node_cost = math.log(1.0 / 2 / 1.0)
    src_dst_node_cost = math.log(maximum_possible_number)
    # even_node_cost_add = -abs(2 * src_dst_node_cost + even_node_cost) - 0.1 # -abs(2 * src_dst_node_cost + most_unreliable_edge_cost + 2 * even_node_cost) / 2 - 0.1
    even_node_cost_add = -abs(2 * src_dst_node_cost + even_node_cost) - 0.1  # -abs(2 * src_dst_node_cost + most_unreliable_edge_cost + 2 * even_node_cost) / 2 - 0.1

    for mapping_node_id_to_bbox_key in mapping_node_id_to_bbox:
        result_mapping_node_id_to_bbox[str(src_id)+'_'+str(int(mapping_node_id_to_bbox_key) * 2)] = math.log(maximum_possible_number) # 源到该节点
        result_mapping_node_id_to_bbox[str(int(mapping_node_id_to_bbox_key) * 2)+'_'+str(int(mapping_node_id_to_bbox_key) * 2 + 1)] = math.log(1.0 / 2 / 1.0)+even_node_cost_add # 该节点自身cost         math.log(1.0 / 2 / mapping_node_id_to_bbox[mapping_node_id_to_bbox_key][1])
        result_mapping_node_id_to_bbox[str(int(mapping_node_id_to_bbox_key) * 2 + 1)+'_'+str(dst_id)] = math.log(maximum_possible_number) # 该节点到汇节点的cost
        result_mapping_node_id_to_bbox_str += ('a*' + str(src_id)+'*'+str(int(mapping_node_id_to_bbox_key) * 2) + '*' + str(math.log(maximum_possible_number)) + '~')
        result_mapping_node_id_to_bbox_str += ('a*' + str(int(mapping_node_id_to_bbox_key) * 2)+'*'+str(int(mapping_node_id_to_bbox_key) * 2 + 1) + '*' + str(math.log(1.0 / 2 / 1.0)+even_node_cost_add) + '~') # str(math.log(1.0 / 2 / mapping_node_id_to_bbox[mapping_node_id_to_bbox_key][1])) + '~')
        result_mapping_node_id_to_bbox_str += ('a*' + str(int(mapping_node_id_to_bbox_key) * 2 + 1)+'*'+str(dst_id) + '*' + str(math.log(maximum_possible_number)) + '~')
    return result_mapping_node_id_to_bbox, result_mapping_node_id_to_bbox_str

def convert_edge_ids(mapping_edge_id_to_cost):
    result_mapping_edge_id_to_cost = {}
    result_mapping_edge_id_to_cost_str = ''
    for mapping_edge_id_to_cost_key in mapping_edge_id_to_cost:
        curr_edge_former_id = int(mapping_edge_id_to_cost_key.split('_')[0]) * 2 + 1
        curr_edge_latter_id = int(mapping_edge_id_to_cost_key.split('_')[1]) * 2
        result_mapping_edge_id_to_cost[str(curr_edge_former_id)+'_'+str(curr_edge_latter_id)] = mapping_edge_id_to_cost[mapping_edge_id_to_cost_key]
        result_mapping_edge_id_to_cost_str += ('a*' + str(curr_edge_former_id)+'*'+str(curr_edge_latter_id)+'*'+str(mapping_edge_id_to_cost[mapping_edge_id_to_cost_key])+'~')
    return result_mapping_edge_id_to_cost, result_mapping_edge_id_to_cost_str
# input
# idx_stride_between_frame_pair: temporal stride between current frame pair
# curr_frame_dict: a dict storing 'bbox_list', 'head_bbox_list', 'box_confidence_scores', 'target_body_box_coord' and 'img_dir' of former frame in current frame pair
# tracklet_pose_collection: dicts of all frames
# next_frame_dict: a dict storing 'bbox_list', 'head_bbox_list', 'box_confidence_scores', 'target_body_box_coord' and 'img_dir' of latter frame in current frame pair
# person_to_person_matching_matrix_normalized: output of the last function
# node_id_cnt: starting node id of nodes in former frame in current frame pair
# tracklet_inner_idx: frame id of the former frame
# mapping_node_id_to_bbox: an empty dict
# mapping_node_id_to_features: an empty dict
# mapping_edge_id_to_cost: an empty dict
# mapping_frameid_bbox_to_features: a dict mapping string 'frameid_bbox coordinates' to a 512-D feature vector, the string 'frameid_bbox coordinates' has length 36, an example is '0930[(467.0, 313.0), (508.0, 424.0)]' where 0930 is frame id, the coordinates are in the format [(left, top), (right, bottom)]
# output
# mapping_node_id_to_bbox: a dict, each key is an interger node id, each node is the index of one bbox in a batch of frames, each value is a list with three elements:
# bbox coordinates [(left coordinate, top coordinate), (right coordinate, bottom coordinate)]
# bbox confidence a floating number
# a string indicating frame name, example '0000.jpg'
# mapping_node_id_to_features: a dict, each key is same as the keys in mapping_node_id_to_bbox, the value is a 512-D floating feature vector of the person in the bbox
# mapping_edge_id_to_cost: a dict, each key is a string, example '12_13' describes the matching error between node '12' and node '13', each value is a floating number
def prepare_costs_for_tracking_alg(idx_stride_between_frame_pair, curr_frame_dict, tracklet_pose_collection, next_frame_dict, person_to_person_matching_matrix_normalized, node_id_cnt, tracklet_inner_idx, mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost, mapping_frameid_bbox_to_features):
    if idx_stride_between_frame_pair == 1:
        for idx_bbox in range(len(curr_frame_dict['bbox_list'])):
            if node_id_cnt + idx_bbox not in mapping_node_id_to_bbox:
                mapping_node_id_to_bbox[node_id_cnt + idx_bbox] = [curr_frame_dict['bbox_list'][idx_bbox], curr_frame_dict['box_confidence_scores'][idx_bbox], tracklet_pose_collection[tracklet_inner_idx]['img_dir'].split('/')[-1]]
                mapping_node_id_to_features[node_id_cnt + idx_bbox] = mapping_frameid_bbox_to_features[curr_frame_dict['img_dir'].split('/')[-1][:-4] + str(curr_frame_dict['bbox_list'][idx_bbox])]
        for idx_bbox in range(len(next_frame_dict['bbox_list'])):
            if node_id_cnt + len(curr_frame_dict['bbox_list']) + idx_bbox not in mapping_node_id_to_bbox:
                mapping_node_id_to_bbox[node_id_cnt + len(curr_frame_dict['bbox_list']) + idx_bbox] = [next_frame_dict['bbox_list'][idx_bbox], next_frame_dict['box_confidence_scores'][idx_bbox],
                    tracklet_pose_collection[tracklet_inner_idx + idx_stride_between_frame_pair]['img_dir'].split('/')[-1]]
                mapping_node_id_to_features[node_id_cnt + len(curr_frame_dict['bbox_list']) + idx_bbox] = \
                    mapping_frameid_bbox_to_features[next_frame_dict['img_dir'].split('/')[-1][:-4] + str(next_frame_dict['bbox_list'][idx_bbox])]
        for idx_bbox_row in range(len(curr_frame_dict['bbox_list'])):
            for idx_bbox_col in range(len(next_frame_dict['bbox_list'])):
                if person_to_person_matching_matrix_normalized[idx_bbox_row, idx_bbox_col] != 0.0:
                    mapping_edge_id_to_cost[str(node_id_cnt + idx_bbox_row) + '_' + str(node_id_cnt + len(curr_frame_dict['bbox_list']) + idx_bbox_col)] = math.log(
                        1.0 / person_to_person_matching_matrix_normalized[idx_bbox_row, idx_bbox_col] / 2.0)
    elif idx_stride_between_frame_pair == 2:
        middle_frame_dict = tracklet_pose_collection[tracklet_inner_idx + idx_stride_between_frame_pair - 1]
        for idx_bbox in range(len(next_frame_dict['bbox_list'])):
            if node_id_cnt + len(curr_frame_dict['bbox_list']) + len(middle_frame_dict['bbox_list']) + idx_bbox not in mapping_node_id_to_bbox:
                mapping_node_id_to_bbox[node_id_cnt + len(curr_frame_dict['bbox_list']) + len(middle_frame_dict['bbox_list']) + idx_bbox] = [next_frame_dict['bbox_list'][idx_bbox], next_frame_dict['box_confidence_scores'][idx_bbox],
                                                                   tracklet_pose_collection[tracklet_inner_idx + idx_stride_between_frame_pair]['img_dir'].split('/')[-1]]
                mapping_node_id_to_features[node_id_cnt + len(curr_frame_dict['bbox_list']) + len(middle_frame_dict['bbox_list']) + idx_bbox] = \
                    mapping_frameid_bbox_to_features[next_frame_dict['img_dir'].split('/')[-1][:-4] + str(next_frame_dict['bbox_list'][idx_bbox])]
        for idx_bbox_row in range(len(curr_frame_dict['bbox_list'])):
            for idx_bbox_col in range(len(next_frame_dict['bbox_list'])):
                if person_to_person_matching_matrix_normalized[idx_bbox_row, idx_bbox_col] != 0.0:
                    mapping_edge_id_to_cost[str(node_id_cnt + idx_bbox_row) + '_' + str(node_id_cnt + len(curr_frame_dict['bbox_list']) + len(middle_frame_dict['bbox_list']) + idx_bbox_col)] = math.log(
                        1.0 / person_to_person_matching_matrix_normalized[idx_bbox_row, idx_bbox_col] / 2.0)
    elif idx_stride_between_frame_pair > 2:
        raise Exception('Not implemented yet!')
    return mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost

def tracking(mapping_node_id_to_bbox, mapping_edge_id_to_cost, tracklet_inner_cnt):
    # result_mapping_node_id_to_bbox:每个节点与source\sink\初始化
    result_mapping_node_id_to_bbox, result_mapping_node_id_to_bbox_str = convert_node_ids(mapping_node_id_to_bbox, mapping_edge_id_to_cost)
    result_mapping_edge_id_to_cost, result_mapping_edge_id_to_cost_str = convert_edge_ids(mapping_edge_id_to_cost)
    transfer_data_to_tracker = str(2 * len(mapping_node_id_to_bbox) + 2) + '*' + str(
        len((result_mapping_node_id_to_bbox_str + result_mapping_edge_id_to_cost_str).split('~')) - 1) + '~' \
                             + result_mapping_node_id_to_bbox_str + result_mapping_edge_id_to_cost_str
    transfer_data_to_tracker = transfer_data_to_tracker[:-1] # 转化为ssp算法需要的格式


    tracking_start_time = time.time()
    p = Popen('call/call', stdin=PIPE, stdout=PIPE, encoding='gbk')
    result = p.communicate(input=transfer_data_to_tracker)
    p.terminate()
    p.wait()
    result = [result[0], result[1]]


    tracking_end_time = time.time()
    print(str(tracking_end_time - tracking_start_time))
    return result

def compute_iou_between_body_and_head(head_box_detected, box_detected):#[(left, top), (right, bottom)]
    corresponding_coefficient_matrix = np.zeros((len(head_box_detected), len(box_detected)))
    for idx_row in range(len(head_box_detected)):
        for idx_col in range(len(box_detected)):
            intersect_vert = min([head_box_detected[idx_row][1][1], box_detected[idx_col][1][1]]) - max([head_box_detected[idx_row][0][1], box_detected[idx_col][0][1]])
            intersect_hori = min([head_box_detected[idx_row][1][0], box_detected[idx_col][1][0]]) - max([head_box_detected[idx_row][0][0], box_detected[idx_col][0][0]])
            union_vert = head_box_detected[idx_row][1][1] - head_box_detected[idx_row][0][1]
            union_hori = head_box_detected[idx_row][1][0] - head_box_detected[idx_row][0][0]
            if intersect_vert > 0 and intersect_hori > 0 and union_vert > 0 and union_hori > 0:
                corresponding_coefficient_matrix[idx_row, idx_col] = float(intersect_vert) * float(intersect_hori) / float(union_vert) / float(union_hori)
            else:
                corresponding_coefficient_matrix[idx_row, idx_col] = 0.0
    return corresponding_coefficient_matrix

def compute_iou_between_bbox_list(head_box_detected, box_detected):#[(left, top), (right, bottom)]
    corresponding_coefficient_matrix = np.zeros((len(head_box_detected), len(box_detected)))
    for idx_row in range(len(head_box_detected)):
        for idx_col in range(len(box_detected)):
            corresponding_coefficient_matrix[idx_row, idx_col] = compute_iou_single_box([head_box_detected[idx_row][0][1], head_box_detected[idx_row][1][1], head_box_detected[idx_row][0][0], head_box_detected[idx_row][1][0]], \
                                                                                        [box_detected[idx_col][0][1], box_detected[idx_col][1][1], box_detected[idx_col][0][0], box_detected[idx_col][1][0]])
    return corresponding_coefficient_matrix

def evaluate_prediction(dataloader, data_dict):
    if not is_main_process():
        return 0, 0, None

    logger.info("Evaluate in main process...")

    annType = ["segm", "bbox", "keypoints"]

    # inference_time = statistics[0].item()
    # track_time = statistics[1].item()
    # n_samples = statistics[2].item()

    # a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
    # a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

    # time_info = ", ".join(
    #     [
    #         "Average {} time: {:.2f} ms".format(k, v)
    #         for k, v in zip(
    #             ["forward", "track", "inference"],
    #             [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
    #         )
    #     ]
    # )

    # info = time_info + "\n"

    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_dict) > 0:
        cocoGt = dataloader.dataset.coco

        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)
        '''
        try:
            from yolox.layers import COCOeval_opt as COCOeval
        except ImportError:
            from pycocotools import cocoeval as COCOeval
            logger.warning("Use standard COCOeval.")
        '''
        #from pycocotools.cocoeval import COCOeval
        from yolox.layers import COCOeval_opt as COCOeval
        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        # info += redirect_string.getvalue()
        return cocoEval.stats[0], cocoEval.stats[1]
    else:
        return 0, 0

def convert_to_coco_format(dataloader, outputs, info_imgs, ids):
    data_list = []
    for (output, img_h, img_w, img_id) in zip(
        outputs, info_imgs[0], info_imgs[1], ids
    ):
        if output is None:
            continue
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            dataloader.dataset.img_size[0] / float(img_h), dataloader.dataset.img_size[1] / float(img_w)
        )
        bboxes /= scale
        bboxes = xyxy2xywh(bboxes)

        cls = output[:, 6] # 0
        scores = output[:, 4] * output[:, 5]
        for ind in range(bboxes.shape[0]):
            label = dataloader.dataset.class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
    return data_list

def tracklet_collection(img_path,img_size,outputs, img_info, box_detected, box_confidence_scores, tracklet_pose_collection,tracklet_pose_collection_second,bbox_confidence_threshold = (0.6,0.1)):

    # img_info["ratio"] = img
    img_h,img_w = img_info["height"], img_info["width"]
    output = outputs[0]
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    scale = min(
        img_size[0] / float(img_h), img_size[1] / float(img_w)
    )
    bboxes /= scale # xyxy
    #bboxes_wh = xyxy2xywh(bboxes)
    cls = output[:, 6] # 0
    scores = output[:, 4] * output[:, 5]
    for ind in range(bboxes.shape[0]):
        width = abs(float(bboxes[ind][2].data.cpu().numpy()) - float(bboxes[ind][0].data.cpu().numpy()))
        heigth = abs(float(bboxes[ind][1].data.cpu().numpy()) - float(bboxes[ind][3].data.cpu().numpy()))
        aspect_ratio = width / heigth
        if aspect_ratio > 1.6 or width*heigth < 100:
            continue
        box_detected.append([(float(bboxes[ind][0].data.cpu().numpy()), float(bboxes[ind][1].data.cpu().numpy())), (float(bboxes[ind][2].data.cpu().numpy()), float(bboxes[ind][3].data.cpu().numpy()))])
        box_confidence_scores.append(float(scores[ind].data.cpu().numpy()) + 1e-4*random.random())
    box_detected_high = [box_detected[box_confidence_scores.index(x)] for x in box_confidence_scores if x >= bbox_confidence_threshold[0]] # 0.4
    box_confidence_scores_high = [box_confidence_scores[box_confidence_scores.index(x)] for x in box_confidence_scores if x >= bbox_confidence_threshold[0]]
    box_detected_second = [box_detected[box_confidence_scores.index(x)] for x in box_confidence_scores if x < bbox_confidence_threshold[0] and x > bbox_confidence_threshold[1]]
    box_confidence_scores_second = [box_confidence_scores[box_confidence_scores.index(x)] for x in box_confidence_scores if x < bbox_confidence_threshold[0] and x > bbox_confidence_threshold[1]]
    if len(box_detected) == 0:
        tracklet_pose_collection.append([])
        return tracklet_pose_collection
    tracklet_pose_collection_tmp = {}
    tracklet_pose_collection_tmp['bbox_list'] = box_detected_high
    tracklet_pose_collection_tmp['box_confidence_scores'] = box_confidence_scores_high
    tracklet_pose_collection_tmp['img_dir'] = img_path
    tracklet_pose_collection_tmp['foreignmatter_bbox_list'] = []
    tracklet_pose_collection_tmp['foreignmatter_box_confidence_scores'] = []
    tracklet_pose_collection.append(tracklet_pose_collection_tmp)
    ## second thresh ##
    # if len(box_detected_second) > 0: # 只有大于0的时候才加入
    tracklet_pose_collection_second_tmp = {}
    tracklet_pose_collection_second_tmp['bbox_list'] = box_detected_second
    tracklet_pose_collection_second_tmp['box_confidence_scores'] = box_confidence_scores_second
    tracklet_pose_collection_second_tmp['img_dir'] = img_path
    tracklet_pose_collection_second.append(tracklet_pose_collection_second_tmp)

    return tracklet_pose_collection,tracklet_pose_collection_second

def compute_inter_person_similarity_worker(files,input_list, whether_use_iou_similarity_or_not,whether_use_reid_similarity_or_not):
    # tracklet_inner_idx: 帧索引 tracklet_inner_base_idx:当前batch开始的帧    node_id_cnt
    tracklet_inner_idx, tracklet_inner_base_idx, node_id_cnt, tracklet_pose_collection, idx_stride_between_frame_pair, maximum_possible_number, max_row_num_of_person_to_person_matching_matrix_normalized, \
        max_col_num_of_person_to_person_matching_matrix_normalized, num_of_person_to_person_matching_matrix_normalized_copies, node_id_cnt_list, all_people_features = \
        input_list[0], input_list[1], input_list[2], input_list[3], input_list[4], input_list[5], input_list[6], input_list[7], input_list[8], input_list[9], input_list[10]

    # collect all bounding boxes in frame pairs
    curr_frame_dict = tracklet_pose_collection[tracklet_inner_idx]  # 当前帧tracklet信息
    next_frame_dict = tracklet_pose_collection[tracklet_inner_idx + idx_stride_between_frame_pair] # 下一个桢tracklet信息

    # matrix storing the iou similarity between each box from previous frame and each box from next frame, each row corresponds to one box in prev, each col - one box in next
    person_to_person_matching_matrix = np.ones((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list']))) * maximum_possible_number #  乘以max_number的含义?

    # matrix storing the appearance similarity between ...
    person_to_person_matching_matrix_iou = np.zeros((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list'])))
    #person_to_person_matching_matrix_iou_before = np.zeros((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list'])))
    # ????
    person_to_person_depth_matching_matrix_iou = np.ones((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list']))) * 0.5

    person_to_person_matching_matrix_confidence = np.zeros((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list'])))
    person_to_person_matching_matrix_offset = np.zeros((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list'])))
    # person_to_person_matching_matrix_confidence: 每个元素表示两个人的置信度之和
    # evaluate_time_start = time.time()
    for curr_person_bbox_coord in curr_frame_dict['bbox_list']:
        for next_person_bbox_coord in next_frame_dict['bbox_list']:
            idx1 = files.index(curr_frame_dict['img_dir'])
            idx2 = files.index(next_frame_dict['img_dir'])
            # warp1 = gmc.apply(idx1+1)

            vector1 = all_people_features[int(np.sum([len(x['bbox_list']) for x in tracklet_pose_collection[0:tracklet_inner_idx]]) + curr_frame_dict['bbox_list'].index(curr_person_bbox_coord))] # 当前帧bbox的特征向量
            vector2 = all_people_features[int(np.sum([len(x['bbox_list']) for x in tracklet_pose_collection[0:(tracklet_inner_idx + idx_stride_between_frame_pair)]]) + next_frame_dict['bbox_list'].index(next_person_bbox_coord))] # 下一帧bbox的特征向量
            person_to_person_matching_matrix[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                1.0 - min([np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)), 1.0])

            # person_to_person_matching_matrix_iou[
            #     curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
            #     min([max([compute_iou_single_box([curr_person_bbox_coord[0][1], curr_person_bbox_coord[1][1], curr_person_bbox_coord[0][0], curr_person_bbox_coord[1][0]], \
            #         [next_person_bbox_coord[0][1], next_person_bbox_coord[1][1], next_person_bbox_coord[0][0], next_person_bbox_coord[1][0]]), 0.0]), 1.0])
            # if idx2 - idx1 == 1:
            #     curr_person_bbox_coord_gmc = warp_pos(np.array(curr_person_bbox_coord),warp1)
            # else:
            #     warp2 = gmc.apply(idx1 + 2)
            #     curr_person_bbox_coord_gmc = warp_pos(np.array(warp_pos(np.array(curr_person_bbox_coord),warp1)),warp2) # 返回的为list:(tuple)

            # iou:top,bottom,left,right
            person_to_person_matching_matrix_iou[
                curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                min([max([compute_iou_single_box([curr_person_bbox_coord[0][1], curr_person_bbox_coord[1][1], curr_person_bbox_coord[0][0], curr_person_bbox_coord[1][0]], \
                    [next_person_bbox_coord[0][1], next_person_bbox_coord[1][1], next_person_bbox_coord[0][0], next_person_bbox_coord[1][0]]), 0.0]), 1.0])
            # person_to_person_matching_matrix_iou_before[
            # curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
            # min([max([compute_iou_single_box([curr_person_bbox_coord[0][1], curr_person_bbox_coord[1][1], curr_person_bbox_coord[0][0], curr_person_bbox_coord[1][0]], \
            #     [next_person_bbox_coord[0][1], next_person_bbox_coord[1][1], next_person_bbox_coord[0][0], next_person_bbox_coord[1][0]]), 0.0]), 1.0])
            person_to_person_depth_matching_matrix_iou[
                curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                1.0 / max([abs(curr_person_bbox_coord[1][1] - next_person_bbox_coord[1][1]), \
                           person_to_person_depth_matching_matrix_iou[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)]])
            person_to_person_matching_matrix_confidence[
                curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                (curr_frame_dict['box_confidence_scores'][curr_frame_dict['bbox_list'].index(curr_person_bbox_coord)] + next_frame_dict['box_confidence_scores'][next_frame_dict['bbox_list'].index(next_person_bbox_coord)])/2
            cons_curr = ((curr_person_bbox_coord[1][0] - curr_person_bbox_coord[0][0]) + (curr_person_bbox_coord[1][1]-curr_person_bbox_coord[0][1])) / 2.
            cons_next = ((next_person_bbox_coord[1][0] - next_person_bbox_coord[0][0]) + (next_person_bbox_coord[1][1]-next_person_bbox_coord[0][1])) / 2.
            person_to_person_matching_matrix_offset[
                curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                (abs((curr_person_bbox_coord[0][0]+curr_person_bbox_coord[1][0])/2.-(next_person_bbox_coord[0][0]+next_person_bbox_coord[1][0])/2.) + \
            abs((curr_person_bbox_coord[0][1]+curr_person_bbox_coord[1][1])/2.-(next_person_bbox_coord[0][1]+next_person_bbox_coord[1][1])/2.)) / ((cons_curr+cons_next) /2.)
    #person_to_person_matching_matrix_offset[person_to_person_matching_matrix_offset > 0.05] = 0.05
    # person_to_person_matching_matrix_offset[person_to_person_matching_matrix_offset > 0.5] = 1
    #person_to_person_matching_matrix_offset[np.where(person_to_person_matching_matrix_offset > 1.)] = 1.0
    #denominator[np.where(denominator == 0)] = 1.0
    #evaluate_time_end = time.time()
    # corner case: only one person
    if person_to_person_matching_matrix.shape[0] == 1 and person_to_person_matching_matrix.shape[1] == 1 and person_to_person_matching_matrix[0][0] == 0.0:
        person_to_person_matching_matrix[0][0] = 1.0
    else:
        # replace zero entries in the matrix "person_to_person_matching_matrix" with half minimum value to facilitate division
        person_to_person_matching_matrix[np.where(person_to_person_matching_matrix==0)] = np.min(person_to_person_matching_matrix[np.where(person_to_person_matching_matrix>0)]) / 2.0
        # similarity is inversely proportional to matching error
        person_to_person_matching_matrix = 1.0 / person_to_person_matching_matrix / np.max(
            1.0 / person_to_person_matching_matrix)  # similarity
        #person_to_person_matching_matrix_offset = 1.0 /person_to_person_matching_matrix_offset / np.max(1.0 / person_to_person_matching_matrix_offset)
    # similarity is the summation of appearance and iou similarity
    if whether_use_iou_similarity_or_not and whether_use_reid_similarity_or_not:
        person_to_person_matching_matrix = person_to_person_matching_matrix * person_to_person_matching_matrix_iou #*person_to_person_matching_matrix_offset# +person_to_person_matching_matrix_offset # *person_to_person_matching_matrix_confidence # * person_to_person_depth_matching_matrix_iou
    elif whether_use_iou_similarity_or_not and not whether_use_reid_similarity_or_not:# 只使用iou信息
        person_to_person_matching_matrix = person_to_person_matching_matrix_iou #* person_to_person_matching_matrix_offset # +person_to_person_matching_matrix_offset
    # 默认情况下为只是用reid信息
    person_to_person_matching_matrix_copy = copy.deepcopy(person_to_person_matching_matrix)
    denominator = person_to_person_matching_matrix_copy.max(axis=1).reshape(person_to_person_matching_matrix_copy.max(axis=1).shape[0], 1)
    denominator[np.where(denominator==0)] = 1.0 #  每一列最大值
    person_to_person_matching_matrix_copy_normalized = person_to_person_matching_matrix_copy / denominator
    for idx_col in range(person_to_person_matching_matrix_copy_normalized.shape[1]):
        if len(np.where(person_to_person_matching_matrix_copy_normalized[:, idx_col]==1.0)[0].tolist()) > 1:
            list_idx_compete = np.where(person_to_person_matching_matrix_copy_normalized[:, idx_col] == 1.0)[0].tolist()
            list_idx_compete_ori = np.argsort(person_to_person_matching_matrix_copy[:, idx_col][list_idx_compete]).tolist()
            list_idx_compete_ordered = np.array(list_idx_compete)[list_idx_compete_ori].tolist()
            for list_idx_compete_ordered_ele in list_idx_compete_ordered:
                person_to_person_matching_matrix_copy_normalized[list_idx_compete_ordered_ele, :] *= 0.99**(len(list_idx_compete_ordered)-1-list_idx_compete_ordered.index(list_idx_compete_ordered_ele))
    person_to_person_matching_matrix_copy_normalized *= 200

    return person_to_person_matching_matrix_copy_normalized, idx_stride_between_frame_pair, node_id_cnt
# input:
# current_video_segment_representative_frames_current_tracklet_id: a floating vector describing the reid features of an identity in current batch of frames
# previous_video_segment_all_traj_all_object_features: a dict, each key is an ID in previous batch of frames, each value is corresponding feature vector in representative frame
def information_gain(current_video_segment_representative_frames_current_tracklet_id, previous_video_segment_all_traj_all_object_features):
    # time_steps_shared_by_previous_traj = [y for y in previous_video_segment_all_traj_all_object_features[[x for x in previous_video_segment_all_traj_all_object_features][0]]]
    # for previous_video_segment_ID in [x for x in previous_video_segment_all_traj_all_object_features][1:]:
    #     time_steps_shared_by_previous_traj = list(set(time_steps_shared_by_previous_traj).intersection(set([y for y in previous_video_segment_all_traj_all_object_features[previous_video_segment_ID]])))
    # consider the case one person disappears for a long time, his appearance time may not overlap with other people's time steps, so this function exits
    time_steps_shared_by_previous_traj = min([len(previous_video_segment_all_traj_all_object_features[x]) for x in previous_video_segment_all_traj_all_object_features])
    if time_steps_shared_by_previous_traj <= 2:
        return False, None
    most_similar_baseids_along_time_steps = []
    for time_step in range(time_steps_shared_by_previous_traj):
        most_similar_baseids_along_time_steps.append(np.argmin([compute_reid_vector_distance(current_video_segment_representative_frames_current_tracklet_id, previous_video_segment_all_traj_all_object_features[previous_id][random.sample([x for x in previous_video_segment_all_traj_all_object_features[previous_id]], time_steps_shared_by_previous_traj)[time_step]]) for previous_id in previous_video_segment_all_traj_all_object_features]))
    if most_similar_baseids_along_time_steps.count(max(most_similar_baseids_along_time_steps, key=most_similar_baseids_along_time_steps.count)) > int(len(most_similar_baseids_along_time_steps) / len([x for x in previous_video_segment_all_traj_all_object_features])) + 1:
        return True, [x for x in previous_video_segment_all_traj_all_object_features.keys()][np.argmin([compute_reid_vector_distance(current_video_segment_representative_frames_current_tracklet_id, np.mean([previous_video_segment_all_traj_all_object_features[previous_id][x] for x in previous_video_segment_all_traj_all_object_features[previous_id]], axis=0)) for previous_id in previous_video_segment_all_traj_all_object_features])]
        # [x for x in previous_video_segment_all_traj_all_object_features.keys()][max(most_similar_baseids_along_time_steps, key=most_similar_baseids_along_time_steps.count)]
    else:
        return False, None

def linear_assignment_ori(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True) # _:最优指派的代价 x:为一个长度为 N行数的数组，指定每行分配给哪一列 y:为长度为列数的数组，指定每列分配给哪一行。
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix) # x:行索引 y:列索引
    return np.array(list(zip(x, y)))

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh) # cost_limit:代价矩阵当中每个元素上限值，超过cost_limit的元素不参与分配
    # cost_limit 设置过小会漏掉能匹配上的目标，设置过大会增加错误的匹配
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def cosine_similarity(vec1,vec2):
    num = float(np.dot(vec1, vec2.T))
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos = num / denom  # cosine similarity
    return cos

def stitching_tracklets(current_video_segment_predicted_tracks, previous_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, previous_video_segment_predicted_tracks_bboxes,current_video_segment_predicted_tracks_bboxes_test,previous_video_segment_predicted_tracks_bboxes_test):
    # split one trajectory into two if box in t-1 has iou with box in t lower than split_single_trajectory_thresh if split one into three ?
    # for current_video_segment_predicted_tracks_bboxes_key in [x for x in current_video_segment_predicted_tracks_bboxes.keys()]:
    #     for time_idx in range(len(current_video_segment_predicted_tracks_bboxes[current_video_segment_predicted_tracks_bboxes_key]) - 1):
    #         if current_video_segment_predicted_tracks_bboxes[current_video_segment_predicted_tracks_bboxes_key][time_idx] -
    # compute_iou_single_box
    # Notice that the indices of trajectories start with 1
    '''
    node_matching_dict:当前batch第一帧与之前batch最后一帧匹配node_id匹配关系
    tracklet_inner_cnt: the index of the last frame in current batch
    current_video_segment_predicted_tracks: a dict describing current tracklet (e.g. frames 1-10), each key is the index of a trajectory, each value is a dict where each key is a frame index and corresponding value is human center at that time
    previous_video_segment_predicted_tracks: same as above, for previous tracklet (e.g. frames 0-9)
    current_video_segment_predicted_tracks_bboxes: a dict describing current tracklet (e.g. frames 1-10), each key is the index of a trajectory, each value is a dict where each key is a frame index and corresponding value is human bbox at that time
    previous_video_segment_predicted_tracks_bboxes: same as above, for previous tracklet (e.g. frames 0-9)
    current_video_segment_representative_frames: for each human trajectory, we use one representative instant to represent the visual feature of the human. The visual feature denotes reid feature, the way of selecting representative feature is achieved with the bbox in one trajectory with highest confidence score provided by yolov5. So each trajectory has one representative feature vector. This is used only when trajectories cannot be matched according to spatial consistency and requires appearance consistency
    current_video_segment_all_traj_all_object_features: the reid features of all humans in all trajectories
    previous_video_segment_representative_frames: same as above
    previous_video_segment_all_traj_all_object_features: same as above
    average_sampling_density_hori_vert: useless
    '''
    global entropy_one_stage
    global entropy_two_stage_1
    global entropy_two_stage_2
    frames_height = frame_height
    frames_width = frame_width
    # (0,boder_x_min) (boder_x_max,frames_width)
    gap = 60  # gap设置太大会把最后一段轨迹给分开来
    border_x_min,border_x_max =  gap ,frames_width-gap
    border_y_min,boder_y_max = gap,frames_height-gap
    tracklets_similarity_matrix = np.zeros((len(previous_video_segment_predicted_tracks), len(current_video_segment_predicted_tracks)))
    direction_similarity_matrix = np.zeros((len(previous_video_segment_predicted_tracks), len(current_video_segment_predicted_tracks)))
    predicted_bbox_based_on_historical_traj = {}
    predicted_bbox_based_on_historical_traj2 = {}
    ## delete the people who disappear from sides of images
    whether_use_consistency_in_traj = True
    # max([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]):上一个batch最大的frameid
    # [max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]表示所有轨迹最后一帧的frameid
    if max([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]) - \
        min([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]) > large_temporal_stride_thresh: #100
        whether_use_consistency_in_traj = False
    def appearance_update(alpha,track_all_test,det_thresh):
        '''
        alpha = 0.95
        track_all_test:previous_video_segment_predicted_tracks_bboxes_test
        det_thresh = 0.4
        '''
        video_feature = {}
        for track in track_all_test:
            for node in track_all_test[track]:
                feat = np.array(track_all_test[track][node][3]) # 需要转化为array才能相乘
                conf = np.array(track_all_test[track][node][2])
                if track not in video_feature:
                    video_feature[track] = feat
                    continue
                trust = (conf - det_thresh) / (1-det_thresh)
                det_alpha = alpha + (1-alpha) * (1-trust)
                update_emb = det_alpha * video_feature[track] + (1-det_alpha)*feat # self.emb = alpha * self.emb + (1 - alpha) * emb
                update_emb /= np.linalg.norm(update_emb)
                video_feature[track] = update_emb
        return video_feature
    previous_video_feature = appearance_update(0.95,previous_video_segment_predicted_tracks_bboxes_test,0.4)
    current_video_feature = appearance_update(0.95,current_video_segment_predicted_tracks_bboxes_test,0.4)
    tracklets_reid_similarity_matrix = np.zeros((len(previous_video_segment_predicted_tracks), len(current_video_segment_predicted_tracks)))
    previous_track_information = {}
    current_track_information = {}
    previous_track_fitter = {} # previous batch的方向信息
    current_track_fitter = {} # current batch的轨迹方向信息
    first_priority_inds = []
    second_priority_inds = []
    current_priority_dict = {}
    for previous_tracklet_id in previous_video_segment_predicted_tracks_bboxes:
        # ### 终止该条轨迹，不进行reid 匹配 ###
        # if previous_tracklet_id in terminate_track_list:
        #     tracklets_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id),:] = 1
        #     continue
        previous_single_track_information = {}
        prev_traj_conf = [previous_video_segment_predicted_tracks_bboxes_test[previous_tracklet_id][node][2] for node in previous_video_segment_predicted_tracks_bboxes_test[previous_tracklet_id]]
        previous_single_track_information['conf'] = copy.deepcopy(prev_traj_conf)
        trajectory_from_prev = previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id] # 前一个batch的一条轨迹
        independent_variable = [float(x[:-4]) for x in trajectory_from_prev]  # 自变量 后四个为图片名称,数目为frame数
        # independent_variable_mean = independent_variable[0]# np.mean(independent_variable)
        # independent_variable = [x-independent_variable_mean for x in independent_variable]
        # If we fit left, top, right, bottom independently, the predicted left may be larger than right
        existing_left_coordinates = [trajectory_from_prev[x][0][0] for x in trajectory_from_prev]   # 可能left全在0附近
        existing_top_coordinates = [trajectory_from_prev[x][0][1] for x in trajectory_from_prev]
        existing_right_coordinates = [trajectory_from_prev[x][1][0] for x in trajectory_from_prev]
        existing_bottom_coordinates = [trajectory_from_prev[x][1][1] for x in trajectory_from_prev]
        existing_horicenter_coordinates = ((np.array(existing_left_coordinates) + np.array(existing_right_coordinates)) / 2.0).tolist() # 水平中心点
        existing_vertcenter_coordinates = ((np.array(existing_top_coordinates) + np.array(existing_bottom_coordinates)) / 2.0).tolist() # 垂直中心点
        existing_largest_width = np.max(np.array(existing_right_coordinates) - np.array(existing_left_coordinates)) # 物体出现过程当中bbox会逐渐变大，使用最大的确保能够框到
        existing_largest_height = np.max(np.array(existing_bottom_coordinates) - np.array(existing_top_coordinates))
        ##### 前一帧轨迹只有1个点的时候无法进行拟合 ########
        predicted_bbox_based_on_historical_traj[previous_tracklet_id] = {} # 需要和previous_video_segment_predicted_tracks_bboxes的key保持一致
        if len(independent_variable) >= 2:
            horicenter_fitter_coefficients = np.polyfit(independent_variable, existing_horicenter_coordinates, 1)
            vertcenter_fitter_coefficients = np.polyfit(independent_variable, existing_vertcenter_coordinates, 1)
            horicenter_fitter = np.poly1d(horicenter_fitter_coefficients) # np.poly1d根据数组生成一个多项式
            vertcenter_fitter = np.poly1d(vertcenter_fitter_coefficients)
            previous_track_fitter[previous_tracklet_id] = [horicenter_fitter,vertcenter_fitter]
            previous_track_direction = [horicenter_fitter_coefficients,vertcenter_fitter_coefficients]

            #previous_single_track_information['poly_error'] = np.polyfit(independent_variable, existing_horicenter_coordinates, 1,full=True)[1][0] + np.polyfit(independent_variable, existing_vertcenter_coordinates, 1,full=True)[1][0]
            previous_single_track_information['direction'] = [horicenter_fitter,vertcenter_fitter]
            previous_track_information[previous_tracklet_id] = previous_single_track_information
        previous_track_reid = previous_video_feature[previous_tracklet_id]
        for idx,current_tracklet_id in enumerate(current_video_segment_predicted_tracks_bboxes): # 遍历当前batch的轨迹
            if current_tracklet_id not in current_priority_dict:
                conf_first_bbox = current_video_segment_predicted_tracks_bboxes_test[current_tracklet_id][list(current_video_segment_predicted_tracks_bboxes_test[current_tracklet_id].keys())[0]][2]
                if conf_first_bbox < config["track_high_thresh"]:
                    second_priority_inds.append(idx)
                else:
                    first_priority_inds.append(idx)
                current_priority_dict[current_tracklet_id] = conf_first_bbox
            if current_tracklet_id not in current_track_information:
                current_single_track_information = {}
                curr_traj_conf = [current_video_segment_predicted_tracks_bboxes_test[current_tracklet_id][node][2] for node in current_video_segment_predicted_tracks_bboxes_test[current_tracklet_id]]
                current_single_track_information['conf'] = copy.deepcopy(curr_traj_conf)
                current_track_information[current_tracklet_id] = current_single_track_information

            trajectory_from_curr = current_video_segment_predicted_tracks_bboxes[current_tracklet_id] # 当前batch的一条轨迹
            sum_error_trajectories = 0.0 # 前一个batch中轨迹和当前batch轨迹的误差
            sum_error_trajectories_num_value_cnt = 0.0 #
            if current_tracklet_id not in current_track_fitter:
                ### backward regression ###
                curr_independent_variable = [float(x[:-4]) for x in trajectory_from_curr]  # 自变量 后四个为图片名称,数目为frame数
                curr_independent_variable_mean = curr_independent_variable[0]# np.mean(independent_variable)
                curr_independent_variable = [x-curr_independent_variable_mean for x in curr_independent_variable]
                # If we fit left, top, right, bottom independently, the predicted left may be larger than right
                curr_left_coordinates = [trajectory_from_curr[x][0][0] for x in trajectory_from_curr]   # 可能left全在0附近
                curr_top_coordinates = [trajectory_from_curr[x][0][1] for x in trajectory_from_curr]
                curr_right_coordinates = [trajectory_from_curr[x][1][0] for x in trajectory_from_curr]
                curr_bottom_coordinates = [trajectory_from_curr[x][1][1] for x in trajectory_from_curr]
                curr_horicenter_coordinates = ((np.array(curr_left_coordinates) + np.array(curr_right_coordinates)) / 2.0).tolist() # 水平中心点
                curr_vertcenter_coordinates = ((np.array(curr_top_coordinates) + np.array(curr_bottom_coordinates)) / 2.0).tolist() # 垂直中心点
                ##### 前一帧轨迹只有1个点的时候无法进行拟合 ########
                if len(curr_independent_variable) < 2 or len(independent_variable) < 2:
                    previous_last_bbox = previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][list(previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id].keys())[-1]]
                    current_first_bbox = current_video_segment_predicted_tracks_bboxes[current_tracklet_id][list(current_video_segment_predicted_tracks_bboxes[current_tracklet_id].keys())[0]]
                    tracklets_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id),[x for x in current_video_segment_predicted_tracks].index(current_tracklet_id)] = 1 - compute_iou_single_box([previous_last_bbox[0][1], previous_last_bbox[1][1], previous_last_bbox[0][0], previous_last_bbox[1][0]],[current_first_bbox[0][1], current_first_bbox[1][1], current_first_bbox[0][0], current_first_bbox[1][0]]) # (y1y2x1x2)
                    continue
                curr_horicenter_fitter_coefficients = np.polyfit(curr_independent_variable, curr_horicenter_coordinates, 1)
                curr_vertcenter_fitter_coefficients = np.polyfit(curr_independent_variable, curr_vertcenter_coordinates, 1)

                current_track_fitter[current_tracklet_id] = [curr_horicenter_fitter_coefficients,curr_vertcenter_fitter_coefficients]
            current_track_direction = current_track_fitter[current_tracklet_id]
            vec1 = np.array([previous_track_direction[0][0], previous_track_direction[1][0]])
            vec2 = np.array([current_track_direction[0][0], current_track_direction[1][0]])
            direction = cosine_similarity(vec1, vec2)
            direction_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id), [x for x in current_video_segment_predicted_tracks].index(current_tracklet_id)] = direction
            if direction < 0. and np.linalg.norm(vec2) > 4 and np.linalg.norm(vec1) > 4:
                tracklets_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id), [x for x in current_video_segment_predicted_tracks].index(current_tracklet_id)] = 1
                tracklets_reid_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id), [x for x in current_video_segment_predicted_tracks].index(current_tracklet_id)] = 1
                continue
            for trajectory_from_curr_key in trajectory_from_curr: # 当前轨迹的所有frameid # [x for x in trajectory_from_curr if (x not in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id])]
                # Order: top, bottom, left, right
                if trajectory_from_curr_key not in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id]: # in 判断键是否存在于字典当中,对于当前batch最后一帧的预测,使用之前的中心点平均值线性拟合结果
                    # sum_error_trajectories += np.sqrt((vertcenter_fitter(float(trajectory_from_curr_key[:-4])-independent_variable_mean) - (trajectory_from_curr[trajectory_from_curr_key][0][1] + trajectory_from_curr[trajectory_from_curr_key][1][1]) / 2.0) ** 2 + \
                    #                                   (horicenter_fitter(float(trajectory_from_curr_key[:-4])-independent_variable_mean) - (trajectory_from_curr[trajectory_from_curr_key][0][0] + trajectory_from_curr[trajectory_from_curr_key][1][0]) / 2.0) ** 2)
                    # 顺序为y1y2x1x2
                    sum_error_trajectories += 1.0 - compute_iou_single_box([max([vertcenter_fitter(float(trajectory_from_curr_key[:-4])) - existing_largest_height / 2.0, 0]), \
                                                                            min([vertcenter_fitter(float(trajectory_from_curr_key[:-4])) + existing_largest_height / 2.0, frames_height]), \
                                                                            max([horicenter_fitter(float(trajectory_from_curr_key[:-4])) - existing_largest_width / 2.0, 0]), \
                                                                            min([horicenter_fitter(float(trajectory_from_curr_key[:-4])) + existing_largest_width / 2.0, frames_width])], \
                                                                           [trajectory_from_curr[trajectory_from_curr_key][0][1], trajectory_from_curr[trajectory_from_curr_key][1][1], \
                                                                            trajectory_from_curr[trajectory_from_curr_key][0][0], trajectory_from_curr[trajectory_from_curr_key][1][0]])
                    sum_error_trajectories_num_value_cnt += 1.0
                else:
                    # sum_error_trajectories += np.sqrt((previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][0][1] + previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][1][1] - trajectory_from_curr[trajectory_from_curr_key][0][1] - trajectory_from_curr[trajectory_from_curr_key][1][1]) ** 2 / 4.0 + \
                    #                                   (previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][0][0] + previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][1][0] - trajectory_from_curr[trajectory_from_curr_key][0][0] - trajectory_from_curr[trajectory_from_curr_key][1][0]) ** 2 / 4.0)
                    sum_error_trajectories += 1.0 - compute_iou_single_box([previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][0][1], previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][1][1], \
                                                                            previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][0][0], previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][trajectory_from_curr_key][1][0]], \
                                                                           [trajectory_from_curr[trajectory_from_curr_key][0][1], trajectory_from_curr[trajectory_from_curr_key][1][1], \
                                                                            trajectory_from_curr[trajectory_from_curr_key][0][0], trajectory_from_curr[trajectory_from_curr_key][1][0]])
                    sum_error_trajectories_num_value_cnt += 1.0
                try:
                    if (trajectory_from_curr_key in [x for x in trajectory_from_curr if (x not in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id])]) and (trajectory_from_curr_key not in predicted_bbox_based_on_historical_traj[previous_tracklet_id]):
                        # trajectory_from_curr_key in [x for x in trajectory_from_curr if (x not in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id])]) 在当前batch frames但是不在前一个batch frames
                        # 注意存放次序为y1y2x1x2
                        predicted_bbox_based_on_historical_traj[previous_tracklet_id][trajectory_from_curr_key] = \
                            [max([vertcenter_fitter(float(trajectory_from_curr_key[:-4])) - existing_largest_height / 2.0, 0]), min([vertcenter_fitter(float(trajectory_from_curr_key[:-4])) + existing_largest_height / 2.0, frames_height]), \
                             max([horicenter_fitter(float(trajectory_from_curr_key[:-4])) - existing_largest_width / 2.0, 0]), min([horicenter_fitter(float(trajectory_from_curr_key[:-4])) + existing_largest_width / 2.0, frames_width])]
                except:
                    print(trajectory_from_curr_key,previous_tracklet_id,list(previous_video_segment_predicted_tracks_bboxes.keys()),list(predicted_bbox_based_on_historical_traj.keys()))

            sum_error_trajectories = sum_error_trajectories / sum_error_trajectories_num_value_cnt
            # tracklets_similarity_matrix存放对应轨迹的匹配误差
            tracklets_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id), [x for x in current_video_segment_predicted_tracks].index(current_tracklet_id)] = sum_error_trajectories
            current_track_reid = current_video_feature[current_tracklet_id]
            tracklets_reid_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id), [x for x in current_video_segment_predicted_tracks].index(current_tracklet_id)] = 1 -  cosine_similarity(np.array(previous_track_reid),np.array(current_track_reid))
        # if previous_tracklet_id in current_video_segment_predicted_tracks_bboxes:
        # If a person has gone out of image, do not use trajectory consistency, if whether_use_consistency_in_traj is already false, do not change it any more
        if whether_use_consistency_in_traj == 1 and previous_tracklet_id in predicted_bbox_based_on_historical_traj and len(predicted_bbox_based_on_historical_traj[previous_tracklet_id]) > 0:
        # if the center of a person has gone out of image, we regard him as gone out
        # one question left here: if a person goes out of image from left, he might re-enter from right
            previous_tracklet_id_predictions = predicted_bbox_based_on_historical_traj[previous_tracklet_id] # 前一个batch当中previous_tracklet_id轨迹预测得到的结果
            # not( 垂直中心点最小值<0,垂直中心点最大值>= frames_height,水平中心点最小值<0,水平中心点最大值>= frames_width) 如果预测值出现异常则不使用consistency in traj
            whether_use_consistency_in_traj = not (min(((np.array([previous_tracklet_id_predictions[x][0] for x in previous_tracklet_id_predictions])+np.array([previous_tracklet_id_predictions[x][1] for x in previous_tracklet_id_predictions])) / 2.0).tolist()) < 0 or \
                                                   max(((np.array([previous_tracklet_id_predictions[x][0] for x in previous_tracklet_id_predictions])+np.array([previous_tracklet_id_predictions[x][1] for x in previous_tracklet_id_predictions])) / 2.0).tolist()) >= frames_height or \
                                                   min(((np.array([previous_tracklet_id_predictions[x][2] for x in previous_tracklet_id_predictions])+np.array([previous_tracklet_id_predictions[x][3] for x in previous_tracklet_id_predictions])) / 2.0).tolist()) < 0 or \
                                                   max(((np.array([previous_tracklet_id_predictions[x][2] for x in previous_tracklet_id_predictions])+np.array([previous_tracklet_id_predictions[x][3] for x in previous_tracklet_id_predictions])) / 2.0).tolist()) >= frames_width)
    # rows = np.min(tracklets_similarity_matrix,1)
    # min_index = np.argmin(tracklets_similarity_matrix,1) # 每行最小值索引,index表示其行数
    result_dict = {}

    # 需要计算匹配的轨迹以及当前帧当中新出现的轨迹以及上一帧中没有匹配到的轨迹
    # 设置阈值
    iou_mask = np.where(tracklets_similarity_matrix > 0.7)
    ### sbs ###
    tracklets_reid_similarity_matrix /= 2.0
    tracklets_reid_similarity_matrix [tracklets_reid_similarity_matrix > config["appearance_thresh"]] = 1.0
    tracklets_reid_similarity_matrix[iou_mask] = 1.0
    fused_similarity_matrix = np.minimum(tracklets_reid_similarity_matrix,tracklets_similarity_matrix) # 混合距离
    # ### osnet ###
    # tracklets_reid_similarity_matrix[iou_mask] = 1.0
    # fused_similarity_matrix = np.minimum(100*tracklets_reid_similarity_matrix,tracklets_similarity_matrix) # 混合距离
    # fused_similarity_matrix = np.minimum(tracklets_reid_similarity_matrix,tracklets_similarity_matrix) # 混合距离
    # binary_similarity_matrix = (np.array(tracklets_similarity_matrix) < 0.6).astype(np.int32)
    # if binary_similarity_matrix.sum(1).max() == 1 and binary_similarity_matrix.sum(0).max() == 1: #轨迹之间一一匹配
    #     matched_indices = np.stack(np.where(binary_similarity_matrix),axis=1)  # [:,0] 行索引  改为键值
    #     # result_dict
    # else:
    #     matched_indices = linear_assignment_ori(fused_similarity_matrix)  #  损失矩阵,fused_similarity_matrix ,tracklets_similarity_matrix

    previous_tracks_id = np.array(list(previous_video_segment_predicted_tracks_bboxes.keys())) # previous track number
    current_tracks_id = np.array(list(current_video_segment_predicted_tracks_bboxes.keys())) # current track number
    # 第一次关联
    dists_first = fused_similarity_matrix[:,first_priority_inds] #
    matches,u_track,u_detection = matching.linear_assignment(dists_first,thresh = 0.7)
    dets_first = current_tracks_id[first_priority_inds]
    dets_second = current_tracks_id[second_priority_inds]
    # print('first stage:',direction_similarity_matrix[matches[:,0],matches[:,1]])
    # abnor = matches[direction_similarity_matrix[matches[:,0],matches[:,1]] < 0,:]
    # for ab in abnor:
    #     print('previous:',previous_track_information[previous_tracks_id[ab[0]]]['direction'],'current:',current_track_fitter[dets_first[ab[1]]])
    for itracked,idet in matches: # 40,39

        result_dict[dets_first[idet]] = previous_tracks_id[itracked]
    # second association
    # r_tracked_stracks = np.array([previous_tracks_id[i] for i in u_track]) # 前一次没有匹配到的轨迹
    r_tracked_stracks = previous_tracks_id[u_track]
    if len(u_track) > 0 and len(second_priority_inds) > 0: # 包含没有匹配到的轨迹才进行下一次关联
        if len(u_track) == 1 and len(second_priority_inds) > 1:
            dists_second = fused_similarity_matrix[u_track][:,second_priority_inds] #
            #dists_second = np.expand_dims(dists_second, 0)
        elif len(second_priority_inds) == 1 and len(u_track) > 1:
            dists_second = fused_similarity_matrix[u_track][:,second_priority_inds]
            #dists_second = np.expand_dims(dists_second,0)
        elif len(second_priority_inds) > 1 and len(u_track) > 1:
            dists_second = fused_similarity_matrix[u_track][:,second_priority_inds] #
        else:
            dists_second = fused_similarity_matrix[[u_track], [second_priority_inds]] #
            #dists_second = np.expand_dims(dists_second, 0)

        matches_second,u_track_second,u_detection_second = matching.linear_assignment(dists_second,thresh=0.5)
        # if len(matches_second.shape) == 2:
        #     print('second stage:', direction_similarity_matrix[matches_second[:,0],matches_second[:,1]])
        for itracked, idet in matches_second:  # 40,39
            det = dets_second[idet]
            track = r_tracked_stracks[itracked]
            result_dict[dets_second[idet]] = r_tracked_stracks[itracked]

    previous_unmatched_tracks = []  # 前一个batch当中未匹配的
    curr_unmatched_tracks = []  # 当前batch未匹配的tracks的key
    for track_id in previous_video_segment_predicted_tracks_bboxes:
        if track_id not in list(result_dict.values()):
            previous_unmatched_tracks.append(track_id)

    for track_id in current_video_segment_predicted_tracks_bboxes:
        if track_id not in list(result_dict.keys()):
            curr_unmatched_tracks.append(track_id)

    ## curr_unmatched与其余curr_tracks计算相似度来判断
    dulplicate_track_list = []
    definite_track_list = list(set(current_tracks_id) - set(curr_unmatched_tracks)) # curr当中匹配上的tracks
    # 需要选择两者当中不重合部分进行比较
    for id1 in definite_track_list:
        for id2 in curr_unmatched_tracks:
            if len(current_video_segment_predicted_tracks_bboxes[id1]) < 2 or len(current_video_segment_predicted_tracks_bboxes[id2]) < 2:  # 无法进行回归的情况直接跳过
                continue
            ### 计算common frames当中的overlap ###
            track1 = current_video_segment_predicted_tracks_bboxes[id1]
            track2 = current_video_segment_predicted_tracks_bboxes[id2]
            frame_span1 = [int(frame.split('.')[0]) for frame in track1]
            frame_span2 = [int(frame.split('.')[0]) for frame in track2]
            frame_bbox1 = {int(frame.split('.')[0]):track1[frame] for frame in track1}
            frame_bbox2 = {int(frame.split('.')[0]):track2[frame] for frame in track2}
            ## 如果两个集合为包含关系 ##
            if set(frame_span1).issubset(set(frame_span2)) or set(frame_span2).issubset(set(frame_span1)):
                common_frames = set(frame_span1).intersection(set(frame_span2))  #
                tracklet_bbox1 = np.array([frame_bbox1[frame] for frame in common_frames])
                tracklet_bbox2 = np.array([frame_bbox2[frame] for frame in common_frames])
                tracklet_overlap_matrix = compute_iou_between_bbox_list(tracklet_bbox1.reshape(-1, 2, 2),tracklet_bbox2.reshape(-1, 2, 2))
                overlap = np.diagonal(tracklet_overlap_matrix)
                if np.mean(overlap) > 0.8:  # 0.66
                    print('track{0} and track{1} overlap is {2}'.format(id1, id2, np.mean(overlap)))
                    dulplicate_track_list.append(id2)  # 可能是split当中的轨迹
    dulplicate_track_list = np.unique(dulplicate_track_list).tolist()  # 需要唯一
    print('dulplicate_track_list', dulplicate_track_list)
    # for list need to use remove, for dict need to use pop
    [curr_unmatched_tracks.remove(trackid) for trackid in dulplicate_track_list if trackid in curr_unmatched_tracks]
    [current_video_segment_predicted_tracks_bboxes.pop(trackid) for trackid in dulplicate_track_list if trackid in current_video_segment_predicted_tracks_bboxes]
    [current_video_segment_predicted_tracks.pop(trackid) for trackid in dulplicate_track_list if trackid in current_video_segment_predicted_tracks]
    new_track_list = []
    terminate_track_list = []
    curr_batch_invalid_list = [] # invalid tracklet in current batch
    for track_id in previous_unmatched_tracks:
        prev_track = previous_video_segment_predicted_tracks_bboxes[track_id]
        (x1,y1),(x2,y2) = prev_track[max(list(prev_track.keys()))]
        flag = whether_on_border(x1,y1,x2,y2)
        if flag:
            terminate_track_list.append(track_id)
    print('terminate track id(global) is {0}'.format(terminate_track_list))
    [previous_unmatched_tracks.remove(track) for track in terminate_track_list]
    for track_id in curr_unmatched_tracks:
        curr_track = current_video_segment_predicted_tracks_bboxes[track_id]
        (x1,y1),(x2,y2) = curr_track[min(list(curr_track.keys()))]
        max_conf = np.max([current_video_segment_predicted_tracks_bboxes_test[track_id][node][2] for node in current_video_segment_predicted_tracks_bboxes_test[track_id]])
        if max_conf < config["track_high_thresh"] + 0.1 :
            curr_batch_invalid_list.append(track_id)
        flag = whether_on_border(x1,y1,x2,y2)
        if flag:
            new_track_list.append(track_id)
    print('new track id is {0}'.format(new_track_list))
    # # [previous_unmatched_tracks.remove(track) for track in terminate_track_list] # 去掉未匹配的轨迹避免对其回归造成id_switch
    # # [curr_unmatched_tracks.remove(track) for track in new_track_list]
    # ## 第三次匹配  使用reid信息 ###
    # ## 使用列表记录新开始的轨迹以及终止的轨迹，对于不是新开始或者终止的轨迹采用reid信息进行关联 ###
    # second_prev_track_list = list(set(copy.deepcopy(previous_unmatched_tracks)) - set(terminate_track_list))
    # second_curr_track_list = list(set(copy.deepcopy(curr_unmatched_tracks)) - set(new_track_list))
    # #### reid ####
    # second_prev_track_list = copy.deepcopy(previous_unmatched_tracks)
    # second_curr_track_list = copy.deepcopy(curr_unmatched_tracks)
    # reid_similarity = np.zeros((len(second_prev_track_list),len(second_curr_track_list)))
    # for previous_id in second_prev_track_list:
    #      ### 超过10帧之后开始使用reid信息进行匹配 ###
    #     prev_track = previous_video_segment_predicted_tracks_bboxes_test[previous_id]
    #     nodes = list(prev_track.keys())
    #     prev_track_feat = previous_video_feature[previous_id]
    #     for current_id in second_curr_track_list:
    #         curr_track = current_video_segment_predicted_tracks_bboxes_test[current_id]
    #         curr_nodes = list(curr_track.keys())
    #         curr_track_feat = current_video_feature[current_id]
    #         reid_similarity[second_prev_track_list.index(previous_id),second_curr_track_list.index(current_id)] = (1 - cosine_similarity(np.array(prev_track_feat),np.array(curr_track_feat))) /2.
    # matched_indices, previous_unmatched_ids, curr_unmatched_ids = linear_assignment(reid_similarity, thresh=0.4)  # ???
    # for match in matched_indices:
    #     result_dict[second_curr_track_list[match[1]]] = second_prev_track_list[match[0]]
    #
    #     previous_unmatched_tracks.remove(second_prev_track_list[match[0]])
    #     curr_unmatched_tracks.remove(second_curr_track_list[match[1]])
    #     print('matched_reid previous track id(global) is {0},current track id(local) is {1}'.format(second_prev_track_list[match[0]],second_curr_track_list[match[1]]))
    #     print('iou similarity:',tracklets_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(second_prev_track_list[match[0]]), [x for x in current_video_segment_predicted_tracks].index(second_curr_track_list[match[1]])])

    print('previous_unmatched(global) track {0},curr_unmatched_track(local) track {1} '.format(previous_unmatched_tracks,set(curr_unmatched_tracks)-set(curr_batch_invalid_list)))
    return  result_dict,previous_unmatched_tracks,curr_unmatched_tracks,terminate_track_list,curr_batch_invalid_list

    # # iou_statistics = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, \
    # #                   1.0, 0.0, 0.0, 2.0, 3.0, 22.0, 12.0, 7.0, 11.0, 10.0, 11.0, 54.0, 42.0, 63.0, 55.0, 141.0, 169.0, 243.0, 390.0, 644.0, \
    # #                   1179.0, 2445.0, 5040.0, 11486.0, 25227.0, 48708.0, 91280.0, 164452.0, 246454.0, 261649.0, 474782.0]
    # if tracklets_similarity_matrix.shape[0] == 1 and tracklets_similarity_matrix.shape[1] == 1:
    #     tracklets_similarity_matrix_without_postprocess = np.array([[1.0 - tracklets_similarity_matrix[0][0]]])
    # else:
    #     tracklets_similarity_matrix_without_postprocess = (np.max(tracklets_similarity_matrix) - tracklets_similarity_matrix) / np.max(tracklets_similarity_matrix) # 与tracklets_similarity_matrix含义相反,copy.deepcopy(1.0 - tracklets_similarity_matrix)
    #     tracklets_similarity_matrix = tracklets_similarity_matrix / np.max(tracklets_similarity_matrix[np.where(tracklets_similarity_matrix != 0)]) # 归一化
    #
    # person_to_person_matching_matrix = np.ones((len(previous_video_segment_representative_frames), len(current_video_segment_representative_frames))) * maximum_possible_number
    # for previous_tracklet_id in previous_video_segment_representative_frames:
    #     for current_tracklet_id in current_video_segment_representative_frames:
    #         time_start = time.time()
    #         prev_person_features = previous_video_segment_representative_frames[previous_tracklet_id][2]
    #         curr_person_features = current_video_segment_representative_frames[current_tracklet_id][2]
    #         person_to_person_matching_matrix[[x for x in previous_video_segment_representative_frames].index(previous_tracklet_id), [x for x in current_video_segment_representative_frames].index(current_tracklet_id)] = \
    #             1 - np.dot(curr_person_features, prev_person_features) / (np.linalg.norm(curr_person_features) * np.linalg.norm(prev_person_features))
    #         if tracklets_similarity_matrix_without_postprocess[[x for x in previous_video_segment_representative_frames].index(previous_tracklet_id), [x for x in current_video_segment_representative_frames].index(current_tracklet_id)] == 0:
    #             person_to_person_matching_matrix[[x for x in previous_video_segment_representative_frames].index(previous_tracklet_id), [x for x in current_video_segment_representative_frames].index(current_tracklet_id)] = maximum_possible_number
    #         time_end = time.time()
    # # if val == 'DJI_0579' and tracklet_inner_cnt == 1749:
    # #     person_to_person_matching_matrix[0, 0] -= 0.002
    # person_to_person_matching_matrix_without_postprocess = copy.deepcopy(1.0 - person_to_person_matching_matrix)
    # if np.max(person_to_person_matching_matrix) > 0:
    #     person_to_person_matching_matrix = person_to_person_matching_matrix / np.max(person_to_person_matching_matrix[np.where(person_to_person_matching_matrix != 0)])
    #
    # if whether_use_consistency_in_traj is True:
    #     tracklets_similarity_matrix = tracklets_similarity_matrix # (tracklets_similarity_matrix + person_to_person_matching_matrix) / 2 # tracklets_similarity_matrix #
    # else:
    #     tracklets_similarity_matrix = copy.deepcopy(person_to_person_matching_matrix)
    #
    # if tracklets_similarity_matrix.shape[0] > tracklets_similarity_matrix.shape[1]:
    #     add_width = tracklets_similarity_matrix.shape[0] - tracklets_similarity_matrix.shape[1]
    #     tracklets_similarity_matrix = np.concatenate((tracklets_similarity_matrix, np.ones((tracklets_similarity_matrix.shape[0], add_width))*np.max(tracklets_similarity_matrix)*2), axis=1)
    #     ot_src = [1.0] * tracklets_similarity_matrix.shape[0]
    #     ot_dst = [1.0] * tracklets_similarity_matrix.shape[1]
    #     transportation_array = ot.emd(ot_src, ot_dst, tracklets_similarity_matrix)
    #     transportation_array = transportation_array[:, 0:tracklets_similarity_matrix.shape[1]-add_width]
    # elif tracklets_similarity_matrix.shape[0] < tracklets_similarity_matrix.shape[1]:
    #     add_height = tracklets_similarity_matrix.shape[1] - tracklets_similarity_matrix.shape[0]
    #     tracklets_similarity_matrix = np.concatenate((tracklets_similarity_matrix, np.ones((add_height, tracklets_similarity_matrix.shape[1]))*np.max(tracklets_similarity_matrix)*2), axis=0)
    #     ot_src = [1.0] * tracklets_similarity_matrix.shape[0]
    #     ot_dst = [1.0] * tracklets_similarity_matrix.shape[1]
    #     transportation_array = ot.emd(ot_src, ot_dst, tracklets_similarity_matrix)
    #     transportation_array = transportation_array[0:tracklets_similarity_matrix.shape[0]-add_height, :]
    # else:
    #     ot_src = [1.0] * tracklets_similarity_matrix.shape[0]
    #     ot_dst = [1.0] * tracklets_similarity_matrix.shape[1]
    #     transportation_array = ot.emd(ot_src, ot_dst, tracklets_similarity_matrix)
    #
    # if dump_stitching_tracklets_switch == 1:
    #     transportation_array_dump = []
    #     for transportation_array_row_idx in range(transportation_array.shape[0]):
    #         transportation_array_dump.append(transportation_array[transportation_array_row_idx, :].tolist())
    #     out_file = os.path.join(dump_curr_video_name, \
    #                             'inside_stitching_tracklets_transportation_array_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
    #     json.dump(transportation_array_dump, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

    # result_dict = {} # key为当前轨迹id, value为上一个batch轨迹id ????
    # current_video_segment_predicted_tracks_backup = {}
    # current_video_segment_predicted_tracks_bboxes_backup = {}
    # current_video_segment_representative_frames_backup = {}
    # current_video_segment_all_traj_all_object_features_backup = {}
    # # backup the tracklets that are not visible in current tracklet
    # if len(current_video_segment_predicted_tracks) <= len(previous_video_segment_predicted_tracks):
    #     for current_tracklet_id in current_video_segment_predicted_tracks:
    #         # assert(np.max(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]) == 1)
    #         # determine location similarity (tracklets_similarity_matrix_without_postprocess) and appearance similarity (person_to_person_matching_matrix_without_postprocess)
    #         # in tracklets_similarity_matrix and person_to_person_matching_matrix, the higher, the less similar
    #         # in tracklets_similarity_matrix_without_postprocess and person_to_person_matching_matrix_without_postprocess, the higher, the more similar
    #         # iou_between_curr_identity_and_historical_identity represents consistency in trajectory
    #         # even if two trajectories are matched by ot, may be due to one person disappears while another enters, their sparial traj has absolute no overlap
    #         iou_between_curr_identity_and_historical_identity = np.max(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])
    #         # determine whether the person has gone out of image  top, bottom, left, right, historic_traj_predictions denotes consistency in trajectory
    #         historic_traj_predictions = predicted_bbox_based_on_historical_traj[[x for x in previous_video_segment_predicted_tracks][np.argmax(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])]]
    #         # if the center of a person has gone out of image, we regard him as gone out
    #         # whether_gone_outofimg = min(((np.array([historic_traj_predictions[x][0] for x in historic_traj_predictions])+np.array([historic_traj_predictions[x][1] for x in historic_traj_predictions])) / 2.0).tolist()) < 0 or \
    #         #                         max(((np.array([historic_traj_predictions[x][0] for x in historic_traj_predictions])+np.array([historic_traj_predictions[x][1] for x in historic_traj_predictions])) / 2.0).tolist()) >= frames_height or \
    #         #                         min(((np.array([historic_traj_predictions[x][2] for x in historic_traj_predictions])+np.array([historic_traj_predictions[x][3] for x in historic_traj_predictions])) / 2.0).tolist()) < 0 or \
    #         #                         max(((np.array([historic_traj_predictions[x][2] for x in historic_traj_predictions])+np.array([historic_traj_predictions[x][3] for x in historic_traj_predictions])) / 2.0).tolist()) >= frames_width
    #         # two cases: new people continuously coming in  or  A come in, B come in, B goes out, C comes in
    #         # case 1: whether_use_consistency_in_traj is 0 for 3 cases: A. a person is in image but has not been detected for over 100 frames; B. a person has gone out of image;
    #         # C. a person is going out of image. But in current branch, len(current_video_segment_predicted_tracks) <= len(previous_video_segment_predicted_tracks), so
    #         # possible case: 1. no people coming in while part of people are disappearing or are out of image 2. new people coming in but more old people are disappearing or are out of image
    #         # we need to determine whether new people are coming in (curr traj and prev traj not share common and curr object similar to multiple historical identities)
    #         if iou_between_curr_identity_and_historical_identity > 0.0:
    #             if iou_between_curr_identity_and_historical_identity > spatialconsistency_reidsimilarity_debate_thresh and whether_use_consistency_in_traj == 0 and \
    #                 np.argmax(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]) != np.argmax(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]): # and \
    #                 transportation_array[np.argmax(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]), :] = 0
    #                 result_dict[current_tracklet_id] = [x for x in previous_video_segment_predicted_tracks][np.argmax(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])]
    #                 transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)] = 0
    #                 transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)][np.argmax(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])] = 1
    #             #    if np.sum(transportation_array[np.argmax(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]), :]) > 1:
    #             #        for current_tracklet_other_idx in [x for x in current_video_segment_predicted_tracks if x != current_tracklet_id]:
    #             #            if transportation_array[np.argmax(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]), [y for y in current_video_segment_predicted_tracks].index(current_tracklet_other_idx)] == 1:
    #             #                transportation_array[np.argmax(tracklets_similarity_matrix_without_postprocess[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]), [y for y in current_video_segment_predicted_tracks].index(current_tracklet_other_idx)] = 0
    #             #                 if current_tracklet_other_idx in result_dict:
    #             #                     del result_dict[current_tracklet_other_idx]
    #             else:
    #                 result_dict[current_tracklet_id] = [x for x in previous_video_segment_predicted_tracks][np.argmax(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])]
    #         # Even if current id is not overlapped with any of existing previous predictions, as long as its most similar previous trajectory has gone out of image, we won't regard it as new coming
    #         # Whether_use_consistency_in_traj == 1 indicates a case: multiple people are always in image but are occluded or not detected, after a long time with re-appearance, iou is 0
    #         # whether_use_consistency_in_traj cannot determine new comer because if 3 people are in image but two are continuously occluded, one new person comes in
    #         # if current identity has no overlapping with existing trajectories in spatial domain, use reid similarity
    #         else:
    #             most_frequent_similarity, most_similar_id = information_gain(current_video_segment_representative_frames[current_tracklet_id][2], previous_video_segment_all_traj_all_object_features)
    #             if iou_between_curr_identity_and_historical_identity == 0.0 and most_frequent_similarity:
    #                 result_dict[current_tracklet_id] = most_similar_id # [x for x in previous_video_segment_predicted_tracks][np.argmin(person_to_person_matching_matrix[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])]
    #             # 3 people, 1 is out of image, 1 is not detected for a long time, the latter re-appears
    #             elif iou_between_curr_identity_and_historical_identity == 0.0 and (not most_frequent_similarity):
    #                 # if len(previous_video_segment_predicted_tracks) < maximum_number_people:
    #                 result_dict[current_tracklet_id] = len(previous_video_segment_predicted_tracks) + 1
    #                 transportation_array[np.argmax(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]), [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)] = 0
    #             # else:
    #             #     result_dict[current_tracklet_id] = [x for x in previous_video_segment_predicted_tracks][np.argmax(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])]
    #     # if more than one
    #     # if len(list(set([x for x in result_dict]))) != len(list(set([result_dict[x] for x in result_dict]))):
    #     #     list_curr_id_matched_to_same_prev_id = []
    #     #     for matched_previous_element in list(set([y for y in [result_dict[x] for x in result_dict] if [result_dict[x] for x in result_dict].count(y) > 1])):
    #     #         list_curr_id_matched_to_same_prev_id += [x for x in result_dict if result_dict[x] == matched_previous_element]
    #     #     for list_curr_id_matched_to_same_prev_id_item in list_curr_id_matched_to_same_prev_id:
    #     #         del result_dict[list_curr_id_matched_to_same_prev_id_item]
    #     #         transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(list_curr_id_matched_to_same_prev_id_item)] = 0
    #     #         del current_video_segment_predicted_tracks[list_curr_id_matched_to_same_prev_id_item]
    #     #         del current_video_segment_predicted_tracks_bboxes[list_curr_id_matched_to_same_prev_id_item]
    #     #         del current_video_segment_representative_frames[list_curr_id_matched_to_same_prev_id_item]
    #     #         del current_video_segment_all_traj_all_object_features[list_curr_id_matched_to_same_prev_id_item]
    #
    #     for occluded_track_id_in_prev in np.where(np.sum(transportation_array, axis=1) == 0)[0].tolist():
    #         current_video_segment_predicted_tracks_backup[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]] = \
    #             previous_video_segment_predicted_tracks[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]]
    #         current_video_segment_predicted_tracks_bboxes_backup[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]] = \
    #             previous_video_segment_predicted_tracks_bboxes[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]]
    #         current_video_segment_representative_frames_backup[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]] = \
    #             previous_video_segment_representative_frames[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]]
    #         current_video_segment_all_traj_all_object_features_backup[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]] = \
    #             previous_video_segment_all_traj_all_object_features[[x for x in previous_video_segment_predicted_tracks][occluded_track_id_in_prev]]
    # else:
    #     # if the number of
    #     additional_identity_cnt = 0
    #     for current_tracklet_id in current_video_segment_predicted_tracks:
    #         if np.max(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]) == 1:
    #             result_dict[current_tracklet_id] = [x for x in previous_video_segment_predicted_tracks][np.argmax(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)])]
    #         elif np.max(transportation_array[:, [y for y in current_video_segment_predicted_tracks].index(current_tracklet_id)]) == 0:
    #             # most_frequent_similarity, most_similar_id = information_gain(current_video_segment_representative_frames[current_tracklet_id][2], previous_video_segment_all_traj_all_object_features)
    #             # if most_frequent_similarity:
    #             #     result_dict[current_tracklet_id] = most_similar_id
    #             # else:
    #             additional_identity_cnt += 1  # revised on 20201231
    #             result_dict[current_tracklet_id] = len(previous_video_segment_predicted_tracks) + additional_identity_cnt
    #
    # if whether_use_consistency_in_traj == 0:
    #     return result_dict, current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_backup, current_video_segment_predicted_tracks_bboxes_backup, current_video_segment_representative_frames_backup, current_video_segment_all_traj_all_object_features_backup, False
    #
    #return result_dict, current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_backup, current_video_segment_predicted_tracks_bboxes_backup, current_video_segment_representative_frames_backup, current_video_segment_all_traj_all_object_features_backup, True

# reorganize all people in current batch of frames into a tensor for reidentitification
def convert_list_dict_to_np(tracklet_pose_collection_input, fixed_height, fixed_width):
    # This is a dict, each key is str(frame idx + [(left, top), (right, bottom)])
    mapping_frameid_bbox_to_features = {}
    curr_tracklet_bbox_cnt = 0 # 最近10帧的bbox个数
    for tracklet_pose_collection_input_item in tracklet_pose_collection_input:
        curr_tracklet_bbox_cnt += len(tracklet_pose_collection_input_item['bbox_list'])
    result_array = np.zeros((curr_tracklet_bbox_cnt, 3, fixed_height, fixed_width)).astype('uint8')

    result_center_coords_array = np.zeros((curr_tracklet_bbox_cnt, 2)).astype('float32')
    curr_tracklet_bbox_cnt = 0
    for tracklet_pose_collection_input_item in tracklet_pose_collection_input:
        curr_img = cv2.imread(tracklet_pose_collection_input_item['img_dir']) # cv2.imread(tracklet_pose_collection_input_item['img_dir'].split(tracklet_pose_collection_input_item['img_dir'].split('/')[-1])[0][:-1].split('_debug')[0]+'_HR'+'/'+tracklet_pose_collection_input_item['img_dir'].split('/')[-1]) # cv2.imread(tracklet_pose_collection_input_item['img_dir'])
        for bbox in tracklet_pose_collection_input_item['bbox_list']:
            ## previous
            curr_crop = curr_img[max(0,int(bbox[0][1])):min(int(bbox[1][1]),curr_img.shape[0]), max(0,int(bbox[0][0])):min(int(bbox[1][0]),curr_img.shape[1]), :]
            if curr_crop.shape[0] > curr_crop.shape[1] * 2:
                # fix by height
                curr_img_resized = cv2.resize(curr_crop, (fixed_width, int(fixed_width / curr_crop.shape[1] * curr_crop.shape[0])), interpolation=cv2.INTER_AREA)
                curr_img_resized = curr_img_resized[int((curr_img_resized.shape[0] - fixed_height) / 2):int((curr_img_resized.shape[0] - fixed_height) / 2) + fixed_height, :, :]
            elif curr_crop.shape[0] < curr_crop.shape[1] * 2:
                curr_img_resized = cv2.resize(curr_crop, (int(fixed_height / curr_crop.shape[0] * curr_crop.shape[1]), fixed_height), interpolation=cv2.INTER_AREA)
                curr_img_resized = curr_img_resized[:, int((curr_img_resized.shape[1] - fixed_width) / 2):int((curr_img_resized.shape[1] - fixed_width) / 2) + fixed_width, :]
            else:
                curr_img_resized = cv2.resize(curr_crop, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
            # curr_crop = curr_img[max(0,int(bbox[0][1])):min(int(bbox[1][1]),curr_img.shape[0]), max(0,int(bbox[0][0])):min(int(bbox[1][0]),curr_img.shape[1]), :]
            # if curr_crop.shape[0] > curr_crop.shape[1] * 2:
            #     # fix by height
            #     curr_img_resized = cv2.resize(curr_crop, (int(fixed_height / curr_crop.shape[0] * curr_crop.shape[1]), fixed_height), interpolation=cv2.INTER_AREA)
            #     pad_w = fixed_width - int(fixed_height / curr_crop.shape[0] * curr_crop.shape[1])
            #     curr_img_resized = np.concatenate((np.zeros((fixed_height,pad_w,3)),curr_img_resized),axis= 1)
            # elif curr_crop.shape[0] < curr_crop.shape[1] * 2:
            #     curr_img_resized = cv2.resize(curr_crop, (fixed_width, int(fixed_width / curr_crop.shape[1] * curr_crop.shape[0])), interpolation=cv2.INTER_AREA)
            #     pad_h = fixed_height - int(fixed_width / curr_crop.shape[1] * curr_crop.shape[0])
            #     curr_img_resized = np.concatenate((np.zeros((pad_h,fixed_width,3)),curr_img_resized),axis= 0 )
            # else:
            #     curr_img_resized = cv2.resize(curr_crop, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
            # crop = cv2.cvtColor(curr_img_resized.astype('uint8'), cv2.COLOR_BGR2RGB)
            # plt.figure(1)
            # plt.imshow(crop)
            # plt.show()

            # curr_img_resized (256,128,3)
            # normalize
            # curr_img_resized = curr_img_resized / 255.0
            # norm_mean = [0.485, 0.456, 0.406]
            # norm_std = [0.229, 0.224, 0.225]
            # curr_img_resized[:, :, 0] = (curr_img_resized[:, :, 0] - norm_mean[0]) / norm_std[0]
            # curr_img_resized[:, :, 1] = (curr_img_resized[:, :, 1] - norm_mean[1]) / norm_std[1]
            # curr_img_resized[:, :, 2] = (curr_img_resized[:, :, 2] - norm_mean[2]) / norm_std[2]

            result_array[curr_tracklet_bbox_cnt, :, :, :] = np.transpose(curr_img_resized, (2,0,1))
            result_center_coords_array[curr_tracklet_bbox_cnt, 0] = (bbox[0][0] + bbox[1][0]) / 2 # 中心点x坐标
            result_center_coords_array[curr_tracklet_bbox_cnt, 1] = (bbox[0][1] + bbox[1][1]) / 2 # 中心点y坐标
            mapping_frameid_bbox_to_features[tracklet_pose_collection_input_item['img_dir'].split('/')[-1][:-4] + \
                                             '[(' + str(bbox[0][0]) + ', ' + str(bbox[0][1]) + '), (' + str(bbox[1][0]) + ', ' + str(bbox[1][1]) + ')]'
                                             ] = curr_tracklet_bbox_cnt # example: '图像名[(1145.0, 229.0), (1201.0, 399.0)]'
            # mapping_frameid_bbox_to_features[tracklet_pose_collection_input_item['img_dir'].split('/')[-1][:-4] + \
            #                                  '[[' + str(bbox[0][0]) + ', ' + str(bbox[0][1]) + '], [' + str(bbox[1][0]) + ', ' + str(bbox[1][1]) + ']]'
            #                                  ] = curr_tracklet_bbox_cnt # example: '图像名[(1145.0, 229.0), (1201.0, 399.0)]'
            curr_tracklet_bbox_cnt += 1
            time_end = time.time()
    return result_array, mapping_frameid_bbox_to_features, result_center_coords_array
# result_array (79,3,256,128) 返回最近10帧所有resized截取的bbox
# mapping_frameid_bbox_to_features dict:79 '图像名[(1145.0, 229.0), (1201.0, 399.0)]'与检测框id对应
# result_center_coords_array:(79,2)　检测框中心坐标

################################################## functions for evaluating similarity #########################################
def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'
###############################################################################################################################
# This function converts the dict split_each_track into a string
# input: result is a list with two elements, result[0] is a string, result[1] is None, the string in result[0] is 'Predicted tracks\n' followed by a string of node indices
# str(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) is the number of trajectories
# split_each_track is a dict of lists, each list has nodes each of which is a list of two characters
# output: a string
def convert_dict_to_str(result, split_each_track):
    if 0 in split_each_track: # 删除0组成的轨迹
        split_each_track.pop(0)
    result_string = ''
    result_string += str(len(split_each_track)) # str(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0])
    result_string += '~~'
    # traverse trajectories, x denotes the key of a trajectory, y denotes the list describing the trajectory
    split_each_track = dict(sorted(split_each_track.items(), key=lambda d: d[0]))  # 按照key排序
    for x in split_each_track:
        # all tracks start with a common node
        result_string += result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[1].split('~')[0] + '~'
        result_list = []
        y = split_each_track[x]
        # result_list stores the unique nodes in the trajectory
        for z in y:
            if z[0] not in result_list:
                result_list.append(z[0])
            if z[1] not in result_list:
                result_list.append(z[1])
        # add the node names to result_string
        for result_list_item in result_list[::-1]:
            result_string += str(result_list_item) + '~'
        # all tracks end with a common node
        result_string += result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[1].split('~')[-1] + '~~'
    return result_string[:-2]

################################################################################################################################################################
################################################## use a string to produce tracks and valid masks of each track ################################################
################################################################################################################################################################
# input:
# result is a string, an example is "Predicted tracks\n3~~61~56~55~50~49~44~43~40~39~36~35~30~29~24~23~18~17~10~9~2~1~0~~61~58~57~52~51~46~45~38~37~32~31~26~25~22~21~14~13~8~7~4~3~0~~61~60~59~54~53~48~47~42~41~34~33~28~27~20~19~16~15~12~11~6~5~0\n", null
# '~~' splits different tracks, '~' splits different nodes inside one track, the number before the first '~~' denotes the number of tracks
# output:
# split_each_track and split_each_track are both lists
# example:
# for an input string 61~56~55~50~49~44~43~40~39~36~35~30~29~24~23~18~17~10~9~2~1~0 whose starting and ending node are shared across all trajectories
# split_each_track is {1:[1, 2], 2:[2, 9], 3:[9, 10], 4:[10, 17], 5:[17, 18], ...}
# split_each_track_valid_mask is {1: 1 if [1, 2] is valid else 0, 2: 1 if .. else 0, 3: 1 if .. else 0, 4:1 if .. else 0, 5:1 if .. else 0, ...}

def update_split_each_track_valid_mask_second(result):
    '''
    check out valid node or dulplicate node, the former tracklet has higher priority
    '''
    # split_each_track is a dict, each key is an index of a trajectory, the value is the list of nodes in the trajectory, each node is a list with two numbers
    # split_each_track_valid_mask is a dict, the keys are the same as split_each_track_valid, the value under each key is a bool number indicating the validity of the node
    split_each_track = {}
    split_each_track_valid_mask = {}
    # traverse trajectories
    for idx_track in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1): # result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[1]表示轨迹数
        # for instance, '61~56~55~50~49~46~45~42~41~36~35~30~29~24~23~16~15~8~7~4~3~0' is converted to
        # pair_former: ['3', '4', '7', '8', '15', '16', '23', '24', '29', '30', '35', '36', '41', '42', '45', '46', '49', '50', '55']
        # pair_latter: ['4', '7', '8', '15', '16', '23', '24', '29', '30', '35', '36', '41', '42', '45', '46', '49', '50', '55', '56']
        pair_former = result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[idx_track].split('~')[1:-1][::-1][:-1] # [1:-1]表示去掉首尾，[::-1]:反序
        pair_latter = result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[idx_track].split('~')[1:-1][::-1][1:]
        # build the nodes in current trajectory, the elements with odd indices indicates a bounding box, those with even indices indicate edges, starting index is 0
        # split_each_track[idx_track]: [['3', '4'], ['4', '7'], ['7', '8'], ['8', '15'], ['15', '16'], ['16', '23'], ['23', '24'], ['24', '29'], ['29', '30'], ['30', '35'], ['35', '36'], ['36', '41'], ['41', '42'], ['42', '45'], ['45', '46'], ['46', '49'], ['49', '50'], ['50', '55'], ['55', '56']]
        split_each_track[idx_track] = [[x, pair_latter[pair_former.index(x)]] for x in pair_former]
        # for each element in split_each_track, split_each_track_valid_mask provides a bool valid number, initialized to be all ones
        split_each_track_valid_mask[idx_track] = [1] * len(split_each_track[idx_track])

    # traverse trajectories
    for idx_track in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1):
        # each node pair denotes one bounding box or one edge, if it is a bounding box, it starts with an odd number and ends with an even number
        for idx,node_pair in enumerate(split_each_track[idx_track]):
            # This element is a node
            if idx % 2 == 0:#(int(node_pair[0]) % 2 == 1) and (int(node_pair[1]) % 2 == 0): # 表示节点
                # regular nodes should be composed of two adjacent numbers, if not, the corresponding element in split_each_track_valid_mask should be set to -1
                # Note that an element correponding to an irregular bounding box in split_each_track_valid_mask, an element correponding to an edge which is adjacent to an irregular bounding box in
                # split_each_track_valid_mask is set to 0
                if int(node_pair[1]) - int(node_pair[0]) != 1:
                    split_each_track_valid_mask[idx_track][split_each_track[idx_track].index(node_pair)] = -1  # curr element
                    if split_each_track[idx_track].index(node_pair) != 0: # 在track当中的索引
                        split_each_track_valid_mask[idx_track][max([split_each_track[idx_track].index(node_pair) - 1, 0])] = 0  # prev element 前一个元素代表连边
                    if split_each_track[idx_track].index(node_pair) != len(split_each_track[idx_track]) - 1:
                        split_each_track_valid_mask[idx_track][min([split_each_track[idx_track].index(node_pair) + 1, len(split_each_track[idx_track]) - 1])] = 0  # next element 后一个元素置0
    # judge whether two trajectories share common parts when the same node appears in two trajectories，同一个节点出现在两条轨迹当中
    for idx_track in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1): # idx_track：key
        for idx1,node_pair in enumerate(split_each_track[idx_track]):
            # traverse another track, idx_track and idx_another_track indicate two different tracks
            for idx_another_track in [x for x in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1) if x != idx_track]: # 另外一条id不等于idx_track的轨迹
                if idx_another_track < idx_track:
                    continue
                # if the node with content node_pair appears in two trajectories: split_each_track[idx_track] and split_each_track[idx_another_track]
                if idx1 % 2 == 0: # node
                    if node_pair in split_each_track[idx_another_track]:
                        split_each_track_valid_mask[idx_another_track][split_each_track[idx_another_track].index(node_pair)] = -1  # invalid node
                        if split_each_track[idx_another_track].index(node_pair) != 0: # 在track当中的索引
                            split_each_track_valid_mask[idx_another_track][max([split_each_track[idx_another_track].index(node_pair) - 1, 0])] = 0  # prev element 前一个元素代表连边
                        if split_each_track[idx_another_track].index(node_pair) != len(split_each_track[idx_another_track]) - 1:
                            split_each_track_valid_mask[idx_another_track][min([split_each_track[idx_another_track].index(node_pair) + 1, len(split_each_track[idx_another_track]) - 1])] = 0  # next element 后一个元素置0

    return split_each_track, split_each_track_valid_mask

def update_split_each_track_valid_mask(result):
    '''
    fix the intersection situation
    '''
    # split_each_track is a dict, each key is an index of a trajectory, the value is the list of nodes in the trajectory, each node is a list with two numbers
    # split_each_track_valid_mask is a dict, the keys are the same as split_each_track_valid, the value under each key is a bool number indicating the validity of the node
    split_each_track = {}
    split_each_track_valid_mask = {}
    # traverse trajectories
    for idx_track in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1): # result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[1]表示轨迹数
        # for instance, '61~56~55~50~49~46~45~42~41~36~35~30~29~24~23~16~15~8~7~4~3~0' is converted to
        # pair_former: ['3', '4', '7', '8', '15', '16', '23', '24', '29', '30', '35', '36', '41', '42', '45', '46', '49', '50', '55']
        # pair_latter: ['4', '7', '8', '15', '16', '23', '24', '29', '30', '35', '36', '41', '42', '45', '46', '49', '50', '55', '56']
        pair_former = result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[idx_track].split('~')[1:-1][::-1][:-1] # [1:-1]表示去掉首尾，[::-1]:反序
        pair_latter = result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[idx_track].split('~')[1:-1][::-1][1:]
        # build the nodes in current trajectory, the elements with odd indices indicates a bounding box, those with even indices indicate edges, starting index is 0
        # split_each_track[idx_track]: [['3', '4'], ['4', '7'], ['7', '8'], ['8', '15'], ['15', '16'], ['16', '23'], ['23', '24'], ['24', '29'], ['29', '30'], ['30', '35'], ['35', '36'], ['36', '41'], ['41', '42'], ['42', '45'], ['45', '46'], ['46', '49'], ['49', '50'], ['50', '55'], ['55', '56']]
        split_each_track[idx_track] = [[x, pair_latter[pair_former.index(x)]] for x in pair_former]
        # for each element in split_each_track, split_each_track_valid_mask provides a bool valid number, initialized to be all ones
        split_each_track_valid_mask[idx_track] = [1] * len(split_each_track[idx_track])

    # traverse trajectories
    for idx_track in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1):
        # each node pair denotes one bounding box or one edge, if it is a bounding box, it starts with an odd number and ends with an even number
        for idx,node_pair in enumerate(split_each_track[idx_track]):
            # This element is a node
            if idx % 2 == 0:#(int(node_pair[0]) % 2 == 1) and (int(node_pair[1]) % 2 == 0): # 表示节点
                # regular nodes should be composed of two adjacent numbers, if not, the corresponding element in split_each_track_valid_mask should be set to -1
                # Note that an element correponding to an irregular bounding box in split_each_track_valid_mask, an element correponding to an edge which is adjacent to an irregular bounding box in
                # split_each_track_valid_mask is set to 0
                if int(node_pair[1]) - int(node_pair[0]) != 1:
                    split_each_track_valid_mask[idx_track][split_each_track[idx_track].index(node_pair)] = -1  # curr element
                    if split_each_track[idx_track].index(node_pair) != 0: # 在track当中的索引
                        split_each_track_valid_mask[idx_track][max([split_each_track[idx_track].index(node_pair) - 1, 0])] = 0  # prev element 前一个元素代表连边
                    if split_each_track[idx_track].index(node_pair) != len(split_each_track[idx_track]) - 1:
                        split_each_track_valid_mask[idx_track][min([split_each_track[idx_track].index(node_pair) + 1, len(split_each_track[idx_track]) - 1])] = 0  # next element 后一个元素置0

            elif idx % 2 == 1:
                if int(node_pair[0]) > int(node_pair[1]) : # invalid edge
                    split_each_track_valid_mask[idx_track][split_each_track[idx_track].index(node_pair)] = 0  # invalid edge
                    if split_each_track[idx_track].index(node_pair) != -1:
                        split_each_track_valid_mask[idx_track][max([split_each_track[idx_track].index(node_pair) - 1, 0])] = -1
                    if split_each_track[idx_track].index(node_pair) != len(split_each_track[idx_track]) - 1:
                        split_each_track_valid_mask[idx_track][min([split_each_track[idx_track].index(node_pair) + 1, len(split_each_track[idx_track]) - 1])] = -1  # next element 后一个元素置-1,next_node

    # judge whether two trajectories share common parts when the same node appears in two trajectories，同一个节点出现在两条轨迹当中
    for idx_track in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1): # idx_track：key
        for idx,node_pair in enumerate(split_each_track[idx_track]):
            # traverse another track, idx_track and idx_another_track indicate two different tracks
            for idx_another_track in [x for x in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1) if x != idx_track]: # 另外一条id不等于idx_track的轨迹
                # if the node with content node_pair appears in two trajectories: split_each_track[idx_track] and split_each_track[idx_another_track]
                if node_pair in split_each_track[idx_another_track] or [node_pair[1],node_pair[0]] in split_each_track[idx_another_track]:
                    # length_common_curr_track_first_half: a list of indices in split_each_track[idx_track], starting from the previous one of node_pair and ending at the index of the first
                    # common node shared by split_each_track[idx_track] and split_each_track[idx_another_track]
                    # range前两个参数表示范围（，-1）到-1但是不包括，最后-1表示步长即倒序
                    if node_pair not in split_each_track[idx_another_track]:
                        node_pair_tmp = [node_pair[1],node_pair[0]]
                        # if (int(node_pair_tmp[0]) % 2 == 1) and (int(node_pair[1]) % 2 == 0):
                        if split_each_track[idx_another_track].index(node_pair_tmp) % 2 == 0:
                            split_each_track_valid_mask[idx_another_track][split_each_track[idx_another_track].index(node_pair_tmp)] = -1  # invalid node
                            if split_each_track[idx_another_track].index(node_pair_tmp) != 0: # 在track当中的索引
                                split_each_track_valid_mask[idx_another_track][max([split_each_track[idx_another_track].index(node_pair_tmp) - 1, 0])] = 0  # prev element 前一个元素代表连边
                            if split_each_track[idx_another_track].index(node_pair_tmp) != len(split_each_track[idx_another_track]) - 1:
                                split_each_track_valid_mask[idx_another_track][min([split_each_track[idx_another_track].index(node_pair_tmp) + 1, len(split_each_track[idx_another_track]) - 1])] = 0  # next element 后一个元素置0
                        else:
                            split_each_track_valid_mask[idx_another_track][split_each_track[idx_another_track].index(node_pair_tmp)] = 0  # invalid edge
                            if split_each_track[idx_another_track].index(node_pair_tmp) != -1:
                                split_each_track_valid_mask[idx_another_track][max([split_each_track[idx_another_track].index(node_pair_tmp) - 1, 0])] = -1
                            if split_each_track[idx_another_track].index(node_pair_tmp) != len(split_each_track[idx_another_track]) - 1:
                                split_each_track_valid_mask[idx_another_track][min([split_each_track[idx_another_track].index(node_pair_tmp) + 1, len(split_each_track[idx_another_track]) - 1])] = -1  # next element 后一个元素置-1,next_node
                    else:
                        node_pair_tmp = node_pair # node_pair_tmp for idx_another,node_pair for idx_track
                    if idx % 2 == 0 and split_each_track_valid_mask[idx_track][split_each_track[idx_track].index(node_pair)] == 1:
                        split_each_track_valid_mask[idx_track][split_each_track[idx_track].index(node_pair)] = -1  # curr element
                        if split_each_track[idx_track].index(node_pair) != 0: # 在track当中的索引
                            split_each_track_valid_mask[idx_track][max([split_each_track[idx_track].index(node_pair) - 1, 0])] = 0  # prev element 前一个元素代表连边
                        if split_each_track[idx_track].index(node_pair) != len(split_each_track[idx_track]) - 1:
                            split_each_track_valid_mask[idx_track][min([split_each_track[idx_track].index(node_pair) + 1, len(split_each_track[idx_track]) - 1])] = 0  # next element 后一个元素置0
                    elif idx % 2 == 1 and split_each_track_valid_mask[idx_track][split_each_track[idx_track].index(node_pair)] == 1: # invalid common edge
                        split_each_track_valid_mask[idx_track][split_each_track[idx_track].index(node_pair)] = 0  # invalid edge
                        if split_each_track[idx_track].index(node_pair) != -1:
                            split_each_track_valid_mask[idx_track][max([split_each_track[idx_track].index(node_pair) - 1, 0])] = -1
                        if split_each_track[idx_track].index(node_pair) != len(split_each_track[idx_track]) - 1:
                            split_each_track_valid_mask[idx_track][min([split_each_track[idx_track].index(node_pair) + 1, len(split_each_track[idx_track]) - 1])] = -1  # next element 后一个元素置-1,next_node

    return split_each_track, split_each_track_valid_mask

def compute_reid_vector_distance(pre_vector, post_vector):
    pre_vector = np.array(pre_vector)
    post_vector = np.array(post_vector)
    num = float(np.dot(pre_vector, post_vector.T))
    denom = np.linalg.norm(pre_vector) * np.linalg.norm(post_vector)
    cos = num / denom
    return 1.0 - cos


###########################################################################################################################################################
# input: a list of lists in a trajectory, example: [[A, B], [C, D], [E, F], ..., [X, Y]]
# output: a list of lists, example: [[A, B], [B, C], [C, D], [D, E], [E, F], ..., [W, X], [X, Y]]
def interpolate_to_obtain_traj(input_list):
    output_list = []
    for input_list_idx in range(len(input_list)):
        if input_list_idx < len(input_list) - 1:
            output_list.append(input_list[input_list_idx])
            output_list.append([input_list[input_list_idx][1], input_list[input_list_idx + 1][0]])
        elif input_list_idx == len(input_list) - 1:
            output_list.append(input_list[input_list_idx])
    return output_list


############################################################################################################################################################
# input:
# result - a list with two elements, the first element is a string, an example of the string is
# 'Predicted tracks
# 3~~61~56~55~50~49~44~43~40~39~36~35~30~29~24~23~18~17~10~9~2~1~0~~61~58~57~52~51~46~45~38~37~32~31~26~25~22~21~14~13~8~7~4~3~0~~61~60~59~54~53~48~47~42~41~34~33~28~27~20~19~16~15~12~11~6~5~0'
# the second element is None
# mapping_edge_id_to_cost - a dict, each key is a string, the value for each key is float, an example is provided in supplementary_materials_for_MOT_postprocessing
# mapping_node_id_to_bbox - a dict, each key is an integer, the value for each key is a list with three elements: bounding box coordinates in float, confidence in float and a string indicating frame id, an example is shown in supplementary_materials_for_MOT_postprocessing
# mapping_node_id_to_features - a dict, each key is an integer, the value for each key is a 512-dim numpy array with floating numbers


######################################################################################################################################################
################################################ part for anomaly detection ##########################################################################
######################################################################################################################################################

def compute_accuracy(score_list, gt_list, thresh):
    score_list_binarized = []
    for score_ele in score_list:
        if score_ele > thresh:
            score_list_binarized.append(1)
        else:
            score_list_binarized.append(0)
    false_pos = 0.0
    for x in range(len(score_list_binarized)):
        if score_list_binarized[x] == 1 and gt_list[x] == 0:
            false_pos += 1.0
    false_neg = 0.0
    for x in range(len(score_list_binarized)):
        if score_list_binarized[x] == 0 and gt_list[x] == 1:
            false_neg += 1.0
    return [1.0 - np.sum(false_pos) / float(len(gt_list)-np.sum(gt_list)), 1.0 - np.sum(false_neg) / float(np.sum(gt_list))]

def check_num_people_consistency(single_person_low):
    for single_person_low_ele in single_person_low:
        if (single_person_low_ele is None) or (len(single_person_low_ele) != len(single_person_low[0])):
            return False
    return True

def all_distinct_keys(single_person_low_buffer):
    result_list = []
    for single_person_low_buffer_item in single_person_low_buffer:
        for single_person_low_buffer_item_key in single_person_low_buffer_item.keys():
            if single_person_low_buffer_item_key not in result_list:
                result_list.append(single_person_low_buffer_item_key)
    return result_list

def obtain_single_person_traj(single_person_low_buffer, single_person_low_buffer_person_id):
    result_list = []
    for single_person_low_buffer_frame_dict in single_person_low_buffer:
        if single_person_low_buffer_person_id in single_person_low_buffer_frame_dict:
            result_list.append(single_person_low_buffer_frame_dict[single_person_low_buffer_person_id])
    return result_list

def tkImage(frame):
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Img.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height), Img.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage

def merge_curr_tracklet_json_with_curr_tracklet_json_conf_bbox(curr_tracklet_json, curr_tracklet_json_conf):
    assert(len(curr_tracklet_json) == len(curr_tracklet_json_conf))
    for idx in range(len(curr_tracklet_json)):
        curr_tracklet_json[idx] = {'bbox_list': curr_tracklet_json[idx]['bbox_list'], 'bbox_list_confidence': curr_tracklet_json_conf[idx]['bbox_list']}
    return curr_tracklet_json

def interpolation_fix_missed_detections(previous_video_segment_predicted_tracks, previous_video_segment_predicted_tracks_bboxes, previous_video_segment_all_traj_all_object_features, tracklet_pose_collection):
    '''
    对tracklet_pose_collection不进行修改
    '''
    donot_interpolate_dict = {}
    num_bits = len([y for y in previous_video_segment_predicted_tracks_bboxes[[x for x in previous_video_segment_predicted_tracks_bboxes][0]]][0].split('.')[0]) # frame_bite,eg:000001.jpg以‘.’分隔开之后num_bits = 6
    # 对previous_video_segment_predicted_tracks_bboxes进行插值
    for previous_video_segment_predicted_tracks_bboxes_key in previous_video_segment_predicted_tracks_bboxes:
        time_key_list = [x for x in previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key]] # frame_list
        time_value_list = [] # convert the time_key_list into float list
        horicenter_list = []
        vertcenter_list = []
        if ('.jpg' in str(time_key_list[0])) or ('.jpg' in str(time_key_list[0])):
            time_value_list = [float(x[:-4]) for x in time_key_list]
        else:
            time_value_list = time_key_list
        horicenter_list = [(previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][0][0] + previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][1][0]) / 2.0 for x in time_key_list] # x center
        vertcenter_list = [(previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][0][1] + previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][1][1]) / 2.0 for x in time_key_list] # y center
        width_list = [abs(previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][1][0] - previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][0][0]) for x in time_key_list]
        height_list = [abs(previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][1][1] - previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][x][0][1]) for x in time_key_list]
        if max(time_value_list) - min(time_value_list) + 1 > len(time_value_list): # need to interpolate
            for time_value_idx in range(len(sorted(time_value_list))): #
                if (time_value_idx > 0) and (abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) > 1): # 前后时间差 > 1
                    for interpolated_time_value in (sorted(time_value_list)[time_value_idx - 1] + 1, sorted(time_value_list)[time_value_idx]):
                        interpolated_horicenter = (horicenter_list[time_value_idx] - horicenter_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + horicenter_list[time_value_idx - 1]
                        interpolated_vertcenter = (vertcenter_list[time_value_idx] - vertcenter_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + vertcenter_list[time_value_idx - 1]
                        interpolated_width = (width_list[time_value_idx] - width_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + width_list[time_value_idx - 1]
                        interpolated_height = (height_list[time_value_idx] - height_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + height_list[time_value_idx - 1]
                        # if max([compute_iou_between_body_and_head([[(interpolated_horicenter - interpolated_width / 2, interpolated_vertcenter - interpolated_height / 2), (interpolated_vertcenter + interpolated_height / 2, interpolated_horicenter + interpolated_width / 2)]], [[(y[0][0], y[0][1]), (y[1][0], y[1][1])]])[0][0] for y in tracklet_pose_collection[[tracklet_pose_collection.index(x) for x in tracklet_pose_collection if int(x['img_dir'].split('/')[-1][:-4]) == interpolated_time_value][0]]['bbox_list']]) < 0.8:
                        if [(interpolated_horicenter - interpolated_width / 2, interpolated_vertcenter - interpolated_height / 2), (interpolated_horicenter + interpolated_width / 2, interpolated_vertcenter + interpolated_height / 2)] not in tracklet_pose_collection[[tracklet_pose_collection.index(x) for x in tracklet_pose_collection if int(x['img_dir'].split('/')[-1][:-4]) == interpolated_time_value][0]]['bbox_list'] and \
                            [(int(round(interpolated_horicenter - interpolated_width / 2)), int(round(interpolated_vertcenter - interpolated_height / 2))), (int(round(interpolated_horicenter + interpolated_width / 2)), int(round(interpolated_vertcenter + interpolated_height / 2)))] not in tracklet_pose_collection[[tracklet_pose_collection.index(x) for x in tracklet_pose_collection if int(x['img_dir'].split('/')[-1][:-4]) == interpolated_time_value][0]]['bbox_list']:
                            if '.jpg' in str(time_key_list[0]):
                                previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][('%0' + str(num_bits) + 'd') % interpolated_time_value + '.jpg'] = [(interpolated_horicenter - interpolated_width / 2, interpolated_vertcenter - interpolated_height / 2), (interpolated_horicenter + interpolated_width / 2, interpolated_vertcenter + interpolated_height / 2)]
                                # %06d:整数输出，整数宽度不足6位的时候左边补数字0
                            else:
                                previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][interpolated_time_value] = [(interpolated_horicenter - interpolated_width / 2, interpolated_vertcenter - interpolated_height / 2), (interpolated_horicenter + interpolated_width / 2, interpolated_vertcenter + interpolated_height / 2)]
                            # tracklet_pose_collection[[tracklet_pose_collection.index(x) for x in tracklet_pose_collection if int(x['img_dir'].split('/')[-1][:-4]) == interpolated_time_value][0]]['bbox_list'].append([(interpolated_horicenter - interpolated_width / 2, interpolated_vertcenter - interpolated_height / 2), (interpolated_horicenter + interpolated_width / 2, interpolated_vertcenter + interpolated_height / 2)])
                            # tracklet_pose_collection[[tracklet_pose_collection.index(x) for x in tracklet_pose_collection if int(x['img_dir'].split('/')[-1][:-4]) == interpolated_time_value][0]]['box_confidence_scores'].append(1.1)
                        else:
                            donot_interpolate_dict[previous_video_segment_predicted_tracks_bboxes_key] = interpolated_time_value
            proxy_dict = {}
            for proxy_dict_key in sorted([x for x in previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key]]):
                proxy_dict[proxy_dict_key] = previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key][proxy_dict_key]
            previous_video_segment_predicted_tracks_bboxes[previous_video_segment_predicted_tracks_bboxes_key] = proxy_dict
    # 对previous_video_segment_predicted_tracks插值
    for previous_video_segment_predicted_tracks_key in previous_video_segment_predicted_tracks:
        time_key_list = [x for x in previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key]]
        time_value_list = []
        horicenter_list = []
        vertcenter_list = []
        if ('.jpg' in str(time_key_list[0])) or ('.jpg' in str(time_key_list[0])):
            time_value_list = [float(x[:-4]) for x in time_key_list]
        else:
            time_value_list = time_key_list
        horicenter_list = [previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key][x][0] for x in time_key_list]
        vertcenter_list = [previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key][x][1] for x in time_key_list]
        if max(time_value_list) - min(time_value_list) + 1 > len(time_value_list):
            for time_value_idx in range(len(sorted(time_value_list))): #
                if (time_value_idx > 0) and (abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) > 1):
                    for interpolated_time_value in (sorted(time_value_list)[time_value_idx - 1] + 1, sorted(time_value_list)[time_value_idx]):
                        if not ((previous_video_segment_predicted_tracks_key in donot_interpolate_dict) and (donot_interpolate_dict[previous_video_segment_predicted_tracks_key] == interpolated_time_value)):
                            if '.jpg' in str(time_key_list[0]):
                                previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key][('%0' + str(num_bits) + 'd') % interpolated_time_value + '.jpg'] = [ \
                                    (horicenter_list[time_value_idx] - horicenter_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + horicenter_list[time_value_idx - 1], \
                                    (vertcenter_list[time_value_idx] - vertcenter_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + vertcenter_list[time_value_idx - 1]]
                            else:
                                previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key][interpolated_time_value] = [ \
                                    (horicenter_list[time_value_idx] - horicenter_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + horicenter_list[time_value_idx - 1], \
                                    (vertcenter_list[time_value_idx] - vertcenter_list[time_value_idx - 1]) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + vertcenter_list[time_value_idx - 1]]
            proxy_dict = {}
            for proxy_dict_key in sorted([x for x in previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key]]):
                proxy_dict[proxy_dict_key] = previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key][proxy_dict_key]
            previous_video_segment_predicted_tracks[previous_video_segment_predicted_tracks_key] = proxy_dict
    # 对previous_video_segment_predicted_tracks插值
    for previous_video_segment_all_traj_all_object_features_key in previous_video_segment_all_traj_all_object_features:
        time_key_list = [x for x in previous_video_segment_all_traj_all_object_features[previous_video_segment_all_traj_all_object_features_key]]
        time_value_list = []
        horicenter_list = []
        vertcenter_list = []
        if ('.jpg' in str(time_key_list[0])) or ('.jpg' in str(time_key_list[0])):
            time_value_list = [float(x[:-4]) for x in time_key_list]
        else:
            time_value_list = time_key_list
        featurevector_list = [previous_video_segment_all_traj_all_object_features[previous_video_segment_all_traj_all_object_features_key][x] for x in time_key_list]
        if max(time_value_list) - min(time_value_list) + 1 > len(time_value_list):
            for time_value_idx in range(len(sorted(time_value_list))): #
                if (time_value_idx > 0) and (abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) > 1):
                    for interpolated_time_value in (sorted(time_value_list)[time_value_idx - 1] + 1, sorted(time_value_list)[time_value_idx]):
                        interpolated_horicenter = ((np.array(featurevector_list[time_value_idx]) - np.array(featurevector_list[time_value_idx - 1])) / abs(sorted(time_value_list)[time_value_idx] - sorted(time_value_list)[time_value_idx - 1]) * abs(interpolated_time_value - sorted(time_value_list)[time_value_idx - 1]) + np.array(featurevector_list[time_value_idx - 1])).tolist()
                        if not ((previous_video_segment_all_traj_all_object_features_key in donot_interpolate_dict) and (donot_interpolate_dict[previous_video_segment_all_traj_all_object_features_key] == interpolated_time_value)):
                            if '.jpg' in str(time_key_list[0]):
                                previous_video_segment_all_traj_all_object_features[previous_video_segment_all_traj_all_object_features_key][('%0' + str(num_bits) + 'd') % interpolated_time_value + '.jpg'] = interpolated_horicenter
                            else:
                                previous_video_segment_all_traj_all_object_features[previous_video_segment_all_traj_all_object_features_key][interpolated_time_value] = interpolated_horicenter
            proxy_dict = {}
            for proxy_dict_key in sorted([x for x in previous_video_segment_all_traj_all_object_features[previous_video_segment_all_traj_all_object_features_key]]):
                proxy_dict[proxy_dict_key] = previous_video_segment_all_traj_all_object_features[previous_video_segment_all_traj_all_object_features_key][proxy_dict_key]
            previous_video_segment_all_traj_all_object_features[previous_video_segment_all_traj_all_object_features_key] = proxy_dict

    return previous_video_segment_predicted_tracks, previous_video_segment_predicted_tracks_bboxes, previous_video_segment_all_traj_all_object_features, tracklet_pose_collection

def GetFaceSmdScore(img):
    # img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = 0
    size = img.shape[0] * img.shape[1]
    d = np.sum(abs(img_gray[:-1, :-1].astype(int) - img_gray[:-1, 1:].astype(int))) + np.sum(abs(img_gray[:-1, :-1].astype(int) - img_gray[1:, :-1].astype(int)))
    # for i in range(img_gray.shape[0] - 1):
    #     for j in range(img_gray.shape[1] - 1):
    #         d = d + abs(int(img_gray[i,j]) - int(img_gray[i + 1,j]))
    #         d = d + abs(int(img_gray[i,j]) - int(img_gray[i,j + 1]))
    score = 0.0
    d = float(d / size)
    if d < 0.0:
        score = 0.6
    elif d > 75.0:
        score = 1.0
    else:
        score = 0.6 + (0.4 / 74.0) * float(d - 1)
    return score

def x1y1wh2x1y1x2y2(bbox):
    x, y, w, h = bbox[0],bbox[1],bbox[2],bbox[3] # top,left,width.height
    x1,y1 = float(x),float(y)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    x1y1x2y2 = [(float(x1),float(y1)),(float(x2),float(y2))]
    return x1y1x2y2
def convert_track_to_processing_format(iou_thresh,iou_thresh_step,split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features,split_each_track_valid_mask):
    # iou_thresh = 0.55 # 0.789
    # iou_thresh_step = 0.017 # 0.017
    curr_predicted_tracks = {}
    curr_predicted_tracks_bboxes = {}
    curr_predicted_tracks_bboxes_test = {} # 测试使用
    curr_predicted_tracks_confidence_score = {}
    curr_representative_frames = {}
    mapping_frameid_to_human_centers = {}  # 暂存curr_predicted_tracks的value
    mapping_frameid_to_bbox = {}
    mapping_frameid_to_confidence_score = {}
    trajectory_node_dict = {}
    trajectory_idswitch_dict = {}
    trajectory_idswitch_reliability_dict = {} # 保存每条轨迹切断的每一段置信度  key:trajectory id value:list eg[1,3,5]  the sum of node_valid_mask
    trajectory_segment_nodes_dict = {}
    for track_id in split_each_track:
        # curr_predicted_tracks[track_id] = mapping_frameid_to_human_centers
        confidence_score_max = 0
        node_id_max = 0
        # print(track_id,' started!' )
        mapping_track_time_to_bbox = {}
        trajectory_node_list = []
        trajectory_idswitch_list = []
        trajectory_idswitch_reliability_list = []
        # trajectory_similarity_list = []
        trajectory_idswitch_reliability = 0
        trajectory_segment_list = []
        trajectory_segment = []
        for idx,node_pair in enumerate(split_each_track[track_id]):  # node，edge
            # if int(node_pair[1]) % 2 == 0:
            if idx % 2 == 0: # 偶数位置表示人的node
                node_id = int(int(node_pair[1]) / 2)
                trajectory_node_list.append(int(node_id))
                # print(node_id)
            else:
                continue
            # mapping_node_id_to_bbox[mapping_node_id_to_bbox.index(int(node_pair[0]))][2] # str:img
            frame_id = mapping_node_id_to_bbox[node_id][2]  # 转化为int进行加减
            bbox = mapping_node_id_to_bbox[node_id][0]
            # [bbox_pre[0][1], bbox_pre[1][1], bbox_pre[0][0], bbox_pre[1][0]]
            idx_tmp = 1  # initial value
            if idx >= 1:
                iou_similarity = compute_iou_single_box([bbox[0][1], bbox[1][1], bbox[0][0], bbox[1][0]],[bbox_pre[0][1], bbox_pre[1][1], bbox_pre[0][0], bbox_pre[1][0]])
                #print(iou_similarity)
                offset_x = abs((bbox[0][0] + bbox[1][0]) / 2 - (bbox_pre[0][0] + bbox_pre[1][0]) / 2) # x-axis
                offset_y = abs((bbox[0][1] + bbox[1][1]) / 2 - (bbox_pre[0][1] + bbox_pre[1][1]) / 2) # y-axis
                offset = offset_x + offset_y
                iou_thresh_tmp = iou_thresh + int((idx-idx_tmp)/2)*iou_thresh_step
                if iou_similarity < iou_thresh_tmp or offset > 15:
                        #or (np.sign(velocity_x*velocity_x_pre)+np.sign(velocity_y*velocity_y_pre)) == -2:
                    #print(track_id,idx,iou_similarity)
                    trajectory_idswitch_list.append(int(idx/2)) # id从0开始
                    idx_tmp = int(idx)
                    # iou_thresh_tmp = copy.deepcopy(iou_thresh)
                    trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
                    trajectory_segment_list.append(trajectory_segment[:])
                    trajectory_idswitch_reliability = 0
                    trajectory_segment = []
            if split_each_track_valid_mask[track_id][idx] == 1:
                trajectory_idswitch_reliability += 1
                trajectory_segment.append(int(node_id))
            bbox_pre = copy.deepcopy(bbox)

            confidence_score = mapping_node_id_to_bbox[node_id][1]
            if confidence_score > confidence_score_max:
                confidence_score_max = confidence_score
                node_id_max = node_id
            mapping_frameid_to_human_centers[int(frame_id.split('.')[0])] = [(bbox[0][0] + bbox[1][0]) / 2,
                                                                             (bbox[0][1] + bbox[1][1]) / 2]  # 同一帧中图片相连?
            mapping_frameid_to_bbox[frame_id] = bbox
            # mapping_frameid_to_bbox[frame_id] = [bbox,confidence_score]
            mapping_frameid_to_confidence_score[frame_id] = confidence_score
            mapping_track_time_to_bbox[int(node_id)] = [frame_id,bbox,confidence_score]
            # current_video_segment_all_traj_all_object_features[track_id] = [[node_id], mapping_node_id_to_features[node_id]]  # ???
        trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
        trajectory_segment_list.append(trajectory_segment)
        trajectory_node_dict[track_id] = copy.deepcopy(trajectory_node_list)
        trajectory_idswitch_dict[track_id] = copy.deepcopy(trajectory_idswitch_list)
        trajectory_idswitch_reliability_dict[track_id] = copy.deepcopy(trajectory_idswitch_reliability_list)
        trajectory_segment_nodes_dict[track_id] = copy.deepcopy(trajectory_segment_list)
        curr_predicted_tracks[track_id] = copy.deepcopy(mapping_frameid_to_human_centers) # 直接等于之后操作会影响到curr_predicted_tracks
        curr_predicted_tracks_bboxes[track_id] = copy.deepcopy(mapping_frameid_to_bbox)
        curr_predicted_tracks_bboxes_test[track_id] = copy.deepcopy(mapping_track_time_to_bbox)
        curr_predicted_tracks_confidence_score[track_id] = copy.deepcopy(mapping_frameid_to_confidence_score)
        # 可能刚好在第一个
        if node_id_max == 0:
            node_id_max =  list(mapping_track_time_to_bbox.keys())[0]
        curr_representative_frames[track_id] = [node_id_max,(bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0]),mapping_node_id_to_features[node_id_max]]  # 高度,宽度
        mapping_frameid_to_human_centers.clear()
        mapping_frameid_to_bbox.clear()
        mapping_frameid_to_confidence_score.clear()
    return curr_predicted_tracks_bboxes_test,trajectory_node_dict,trajectory_idswitch_dict,trajectory_idswitch_reliability_dict,trajectory_segment_nodes_dict

# def convert_track_to_processing_format(iou_thresh,iou_thresh_step,split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features,split_each_track_valid_mask):
#     # iou_thresh = 0.55 # 0.789
#     # iou_thresh_step = 0.017 # 0.017
#     curr_predicted_tracks = {}
#     curr_predicted_tracks_bboxes = {}
#     curr_predicted_tracks_bboxes_test = {} # 测试使用
#     curr_predicted_tracks_confidence_score = {}
#     curr_representative_frames = {}
#     mapping_frameid_to_human_centers = {}  # 暂存curr_predicted_tracks的value
#     mapping_frameid_to_bbox = {}
#     mapping_frameid_to_confidence_score = {}
#     trajectory_node_dict = {}
#     trajectory_idswitch_dict = {}
#     trajectory_idswitch_reliability_dict = {} # 保存每条轨迹切断的每一段置信度  key:trajectory id value:list eg[1,3,5]  the sum of node_valid_mask
#     trajectory_segment_nodes_dict = {}
#     for track_id in split_each_track:
#         # curr_predicted_tracks[track_id] = mapping_frameid_to_human_centers
#         confidence_score_max = 0
#         node_id_max = 0
#         # print(track_id,' started!' )
#         mapping_track_time_to_bbox = {}
#         trajectory_node_list = []
#         trajectory_idswitch_list = []
#         trajectory_idswitch_reliability_list = []
#         # trajectory_similarity_list = []
#         trajectory_idswitch_reliability = 0
#         trajectory_segment_list = []
#         trajectory_segment = []
#         for idx,node_pair in enumerate(split_each_track[track_id]):  # node，edge
#             # if int(node_pair[1]) % 2 == 0:
#             if idx % 2 == 0: # 偶数位置表示人的node
#                 node_id = int(int(node_pair[1]) / 2)
#                 trajectory_node_list.append(int(node_id))
#                 # print(node_id)
#             else:
#                 continue
#             # mapping_node_id_to_bbox[mapping_node_id_to_bbox.index(int(node_pair[0]))][2] # str:img
#             frame_id = mapping_node_id_to_bbox[node_id][2]  # 转化为int进行加减
#             bbox = mapping_node_id_to_bbox[node_id][0]
#             # [bbox_pre[0][1], bbox_pre[1][1], bbox_pre[0][0], bbox_pre[1][0]]
#             idx_tmp = 1  # initial value
#             if idx >= 1:
#                 iou_similarity = compute_iou_single_box([bbox[0][1], bbox[1][1], bbox[0][0], bbox[1][0]],[bbox_pre[0][1], bbox_pre[1][1], bbox_pre[0][0], bbox_pre[1][0]])
#                 #print(iou_similarity)
#                 offset_x = abs((bbox[0][0] + bbox[1][0]) / 2 - (bbox_pre[0][0] + bbox_pre[1][0]) / 2) # x-axis
#                 offset_y = abs((bbox[0][1] + bbox[1][1]) / 2 - (bbox_pre[0][1] + bbox_pre[1][1]) / 2) # y-axis
#                 offset = offset_x + offset_y
#                 iou_thresh_tmp = iou_thresh + int((idx-idx_tmp)/2)*iou_thresh_step
#                 if iou_similarity < iou_thresh_tmp or offset > 15:
#                         #or (np.sign(velocity_x*velocity_x_pre)+np.sign(velocity_y*velocity_y_pre)) == -2:
#                     #print(track_id,idx,iou_similarity)
#                     trajectory_idswitch_list.append(int(idx/2)) # id从0开始
#                     idx_tmp = int(idx)
#                     # iou_thresh_tmp = copy.deepcopy(iou_thresh)
#                     trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
#                     trajectory_segment_list.append(trajectory_segment[:])
#                     trajectory_idswitch_reliability = 0
#                     trajectory_segment = []
#             if split_each_track_valid_mask[track_id][idx] == 1:
#                 trajectory_idswitch_reliability += 1
#                 trajectory_segment.append(int(node_id))
#             bbox_pre = copy.deepcopy(bbox)
#
#             confidence_score = mapping_node_id_to_bbox[node_id][1]
#             if confidence_score > confidence_score_max:
#                 confidence_score_max = confidence_score
#                 node_id_max = node_id
#             mapping_frameid_to_human_centers[int(frame_id.split('.')[0])] = [(bbox[0][0] + bbox[1][0]) / 2,
#                                                                              (bbox[0][1] + bbox[1][1]) / 2]  # 同一帧中图片相连?
#             mapping_frameid_to_bbox[frame_id] = bbox
#             # mapping_frameid_to_bbox[frame_id] = [bbox,confidence_score]
#             mapping_frameid_to_confidence_score[frame_id] = confidence_score
#             mapping_track_time_to_bbox[int(node_id)] = [frame_id,bbox,confidence_score]
#             # current_video_segment_all_traj_all_object_features[track_id] = [[node_id], mapping_node_id_to_features[node_id]]  # ???
#         trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
#         trajectory_segment_list.append(trajectory_segment)
#         trajectory_node_dict[track_id] = copy.deepcopy(trajectory_node_list)
#         trajectory_idswitch_dict[track_id] = copy.deepcopy(trajectory_idswitch_list)
#         trajectory_idswitch_reliability_dict[track_id] = copy.deepcopy(trajectory_idswitch_reliability_list)
#         trajectory_segment_nodes_dict[track_id] = copy.deepcopy(trajectory_segment_list)
#         curr_predicted_tracks[track_id] = copy.deepcopy(mapping_frameid_to_human_centers) # 直接等于之后操作会影响到curr_predicted_tracks
#         curr_predicted_tracks_bboxes[track_id] = copy.deepcopy(mapping_frameid_to_bbox)
#         curr_predicted_tracks_bboxes_test[track_id] = copy.deepcopy(mapping_track_time_to_bbox)
#         curr_predicted_tracks_confidence_score[track_id] = copy.deepcopy(mapping_frameid_to_confidence_score)
#         # 可能刚好在第一个
#         if node_id_max == 0:
#             node_id_max =  list(mapping_track_time_to_bbox.keys())[0]
#         curr_representative_frames[track_id] = [node_id_max,(bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0]),mapping_node_id_to_features[node_id_max]]  # 高度,宽度
#         mapping_frameid_to_human_centers.clear()
#         mapping_frameid_to_bbox.clear()
#         mapping_frameid_to_confidence_score.clear()
#     return curr_predicted_tracks_bboxes_test,trajectory_node_dict,trajectory_idswitch_dict,trajectory_idswitch_reliability_dict,trajectory_segment_nodes_dict

def convert_track_to_stitch_format(split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features):
    '''
    evaluate the tracklet importance and interpolate missed detections,delete duplicate ones
    '''
    #unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
    curr_predicted_tracks = {}
    curr_predicted_tracks_bboxes = {}
    curr_predicted_tracks_bboxes_test = {} # 测试使用
    curr_predicted_tracks_confidence_score = {}
    curr_representative_frames = {}
    curr_video_segment_all_traj_all_object_features = {}
    mapping_frameid_to_human_centers = {}  # 暂存curr_predicted_tracks的value
    mapping_frameid_to_bbox = {}
    mapping_frameid_to_confidence_score = {}
    mapping_frameid_to_object_features = {}
    trajectory_similarity_dict = {}
    for track_id in split_each_track:
        trajectory_similarity_list = []
        node_id_pre = 0
        # curr_predicted_tracks[track_id] = mapping_frameid_to_human_centers
        confidence_score_max = 0
        node_id_max = 0
        # print(track_id,' started!' )
        mapping_track_time_to_bbox = {}
        for idx,node_pair in enumerate(split_each_track[track_id]):
            if int(node_pair[1]) % 2 == 0:
                node_id = int(node_pair[1]) / 2
                # print(node_id)
            else:
                continue
            # mapping_node_id_to_bbox[mapping_node_id_to_bbox.index(int(node_pair[0]))][2] # str:img
            frame_id = mapping_node_id_to_bbox[node_id][2]  # 转化为int进行加减
            bbox = mapping_node_id_to_bbox[node_id][0]
            confidence_score = mapping_node_id_to_bbox[node_id][1]
            if confidence_score > confidence_score_max:
                confidence_score_max = confidence_score
                node_id_max = node_id
            mapping_frameid_to_human_centers[int(frame_id.split('.')[0])] = [(bbox[0][0] + bbox[1][0]) / 2,
                                                                             (bbox[0][1] + bbox[1][1]) / 2]  # 同一帧中图片相连?
            mapping_frameid_to_bbox[frame_id] = bbox
            # mapping_frameid_to_bbox[frame_id] = [bbox,confidence_score]
            mapping_frameid_to_confidence_score[frame_id] = confidence_score
            mapping_frameid_to_object_features[frame_id] = mapping_node_id_to_features[node_id]
            mapping_track_time_to_bbox[int(node_id)] = [frame_id,bbox,confidence_score,mapping_node_id_to_features[int(node_id)]]
            if node_id_pre != 0:
                trajectory_similarity_list.append(cosine_similarity(np.array(mapping_node_id_to_features[node_id_pre]),np.array(mapping_node_id_to_features[int(node_id)])))
            node_id_pre = int(node_id)  # 前一个加入的点
            # current_video_segment_all_traj_all_object_features[track_id] = [[node_id], mapping_node_id_to_features[node_id]]  # ???

        curr_predicted_tracks[track_id] = copy.deepcopy(mapping_frameid_to_human_centers) # 直接等于之后操作会影响到curr_predicted_tracks
        curr_predicted_tracks_bboxes[track_id] = copy.deepcopy(mapping_frameid_to_bbox)
        curr_predicted_tracks_bboxes_test[track_id] = copy.deepcopy(mapping_track_time_to_bbox)
        curr_predicted_tracks_confidence_score[track_id] = copy.deepcopy(mapping_frameid_to_confidence_score)
        curr_video_segment_all_traj_all_object_features[track_id] = copy.deepcopy(mapping_frameid_to_object_features)
        trajectory_similarity_dict[track_id] = trajectory_similarity_list
        # 可能刚好在第一个
        if node_id_max == 0:
            node_id_max =  list(mapping_track_time_to_bbox.keys())[0]
        curr_representative_frames[track_id] = [node_id_max,(bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0]),mapping_node_id_to_features[node_id_max]]  # 高度,宽度
        mapping_frameid_to_human_centers.clear()
        mapping_frameid_to_bbox.clear()
        mapping_frameid_to_confidence_score.clear()
        mapping_frameid_to_object_features.clear()
    return curr_predicted_tracks,curr_predicted_tracks_confidence_score,curr_predicted_tracks_bboxes,curr_representative_frames,curr_predicted_tracks_bboxes_test,trajectory_similarity_dict,curr_video_segment_all_traj_all_object_features

def mapping_data_preparation(files,encoder,tracklet_pose_collection,tracklet_inner_cnt,whether_use_reid_similarity_or_not):
    '''
    Use fastreid to extract reid feature
    '''
    curr_tracklet_input_people, mapping_frameid_bbox_to_features, curr_tracklet_input_people_center_coords = convert_list_dict_to_np(tracklet_pose_collection, 256, 128)# 近10帧
    gc.collect()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=0, abbreviated=False))
    ## BoT使用模型 ##
    features = np.zeros((0, 2048))
    for tracklet_pose_collection_input_item in tracklet_pose_collection:
        curr_img = cv2.imread(tracklet_pose_collection_input_item['img_dir']) # cv2默认为BGR顺序
        dets = np.array(tracklet_pose_collection_input_item['bbox_list']).reshape(-1,4)
        if len(dets) == 0: # 注意second_mapping的时候不一定每个tracklets都有dets
            continue
        features_img = encoder.inference(curr_img,dets)
        features = np.vstack((features,features_img)) # 数据类型：ndarray,features.data.numpy()转化为features
    # #### OSnet ####
    # with torch.no_grad():
    #     features_torch = similarity_module(torch.from_numpy(curr_tracklet_input_people.astype('float32')).cuda()).data.cpu()
    #     features = features_torch.data.numpy()

    # 计算所有input people的特征向量，Tensor(79,512)
    # mapping_frameid_bbox_to_features 的值替换为bbox的特征
    for mapping_frameid_bbox_to_features_key in mapping_frameid_bbox_to_features.keys():
        mapping_frameid_bbox_to_features[mapping_frameid_bbox_to_features_key] = features[mapping_frameid_bbox_to_features[mapping_frameid_bbox_to_features_key], :].tolist()
    ############################################################################################################################################
    mapping_node_id_to_bbox = {} # dict:79 包含每个node的bbox位置置信度以及帧名
    mapping_node_id_to_features = {} #　dict: key=str(imgname,bbox) value=feature
    mapping_edge_id_to_cost = {} #　每个id与下一帧每个id的损失　
    mapping_node_id_to_keypoint = {}
    node_id_cnt = 1 #
    ###################################################################### multiprocessing for computing matching error between people
    # error_computing_start_time = time.time()
    # number of frame-to-frame pairs for matching
    num_of_person_to_person_matching_matrix_normalized_copies = 0
    # for multi-process, the matching between each pair of frames produces a matrix with number of rows equal to the number of people in the former frame and
    # number of cols equal to the number of people in the latter frame, this variable stores the maximum number of rows throughout all frame pairs
    max_row_num_of_person_to_person_matching_matrix_normalized = 0 #所有当前帧中最大值
    # for multi-process, the matching between each pair of frames produces a matrix with number of rows equal to the number of people in the former frame and
    # number of cols equal to the number of people in the latter frame, this variable stores the maximum number of cols throughout all frame pairs
    max_col_num_of_person_to_person_matching_matrix_normalized = 0 #所有下一帧中最大值
    tracklet_inner_idx_list = []
    node_id_cnt_list = []
    parallel_tasks_args_list = []
    for tracklet_inner_idx in range(len(tracklet_pose_collection)):
        curr_frame_dict = tracklet_pose_collection[tracklet_inner_idx]  # 当前处理帧
        if len(curr_frame_dict['bbox_list']) == 0:
            continue
        for idx_stride_between_frame_pair in range(1, 3): # 配对帧之间步长最多为2
            if tracklet_inner_idx + idx_stride_between_frame_pair >= len(tracklet_pose_collection): # 保证仍然在该batch内
                continue
    # current node idx (each node represents one person in a certain frame), collection of all human bounding boxes in current batch, stride between former and latter frames, an extremely large number
            next_frame_dict = tracklet_pose_collection[tracklet_inner_idx + idx_stride_between_frame_pair]
            if len(next_frame_dict['bbox_list']) == 0:
                continue
            parallel_tasks_args_list.append([tracklet_inner_idx, tracklet_inner_cnt - tracklet_len + 1, node_id_cnt, tracklet_pose_collection, idx_stride_between_frame_pair, maximum_possible_number]) # 当前batch起始帧位置
            num_of_person_to_person_matching_matrix_normalized_copies += 1
            max_row_num_of_person_to_person_matching_matrix_normalized = max([max_row_num_of_person_to_person_matching_matrix_normalized, len(curr_frame_dict['bbox_list'])])
            max_col_num_of_person_to_person_matching_matrix_normalized = max([max_col_num_of_person_to_person_matching_matrix_normalized, len(next_frame_dict['bbox_list'])])
        node_id_cnt += len(curr_frame_dict['bbox_list'])
    # The matching matrix storing the matching relations between all pairs of frames
    person_to_person_matching_matrix_normalized_collection = np.zeros((max_row_num_of_person_to_person_matching_matrix_normalized, max_col_num_of_person_to_person_matching_matrix_normalized, num_of_person_to_person_matching_matrix_normalized_copies))
    result_person_to_person_matching_matrix_normalized_collection = copy.deepcopy(person_to_person_matching_matrix_normalized_collection)
    # person_to_person_matching_matrix_normalized_collection = multiprocessing.RawArray('d', person_to_person_matching_matrix_normalized_collection.ravel())
    for parallel_tasks_args_list_item in parallel_tasks_args_list:
        tracklet_inner_idx_list.append(parallel_tasks_args_list_item[0])
        node_id_cnt_list.append(parallel_tasks_args_list_item[2])
        parallel_tasks_args_list[parallel_tasks_args_list.index(parallel_tasks_args_list_item)] = parallel_tasks_args_list_item + [max_row_num_of_person_to_person_matching_matrix_normalized, max_col_num_of_person_to_person_matching_matrix_normalized, num_of_person_to_person_matching_matrix_normalized_copies, node_id_cnt_list, features]
    # with multiprocessing.Pool(processes=8) as pool:
    whether_use_iou_similarity_or_not = True # (GetFaceSmdScore(im0s) > 0.7) # 如果清晰度大于阈值则使用iou_similarity

    for parallel_tasks_args_list_item in parallel_tasks_args_list:
        person_to_person_matching_matrix_normalized, idx_stride_between_frame_pair, node_id_cnt = compute_inter_person_similarity_worker(files,parallel_tasks_args_list_item, whether_use_iou_similarity_or_not,whether_use_reid_similarity_or_not)
        # collect the results from multiprocessing and store the matching error between each pair of frames into result_person_to_person_matching_matrix_normalized_collection
        result_person_to_person_matching_matrix_normalized_collection[0:person_to_person_matching_matrix_normalized.shape[0], \
                                                                      0:person_to_person_matching_matrix_normalized.shape[1], \
                                                                      [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0]] = person_to_person_matching_matrix_normalized
    node_id_cnt = 1
    for tracklet_inner_idx in range(len(tracklet_pose_collection)):
        # tracklet_pose_collection is a dict, each key is an integer of frame id, the value corresponding to the key is a dict with following keys:
        # 'bbox_list': a list of bboxes in the frame, each bbox with floating data [(left coordinate, top coordinate), (right coordinate, bottom coordinate)], body bboxes
        # 'head_bbox_list': a list of bboxes in the frame, each bbox with floating data [(left coordinate, top coordinate), (right coordinate, bottom coordinate)], head bboxes
        # 'box_confidence_scores': a list of confidences of bboxes, each element is a floating number within range [0, 1]
        # 'target_body_box_coord': a list with two elements, in the format of floating [(left coordinate, top coordinate), (right coordinate, bottom coordinate)], describe the bbox of the target person
        # 'img_dir': a string describing the location for storing the frame, ending with '.jpg'
        curr_frame_dict = tracklet_pose_collection[tracklet_inner_idx]
        if len(curr_frame_dict['bbox_list']) == 0:
            continue
        # matching is conducted between (frame t, frame t+1) and (frame t, frame t+2), that is (frame t, frame t+idx_stride_between_frame_pair) with idx_stride_between_frame_pair=1, 2
        for idx_stride_between_frame_pair in range(1, 3):
            # if frame t and frame t+idx_stride_between_frame_pair all have detections
            if tracklet_inner_idx + idx_stride_between_frame_pair < len(tracklet_pose_collection):
                next_frame_dict = tracklet_pose_collection[tracklet_inner_idx + idx_stride_between_frame_pair]
                if len(next_frame_dict['bbox_list']) == 0:
                    continue
                ############################################################# matching ######################################################
                # a floating matrix person_to_person_matching_matrix_normalized describing the matching relations between current batch of frames
                person_to_person_matching_matrix_normalized = result_person_to_person_matching_matrix_normalized_collection[0:len(curr_frame_dict['bbox_list']), 0:len(next_frame_dict['bbox_list']), [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0]]
                ########################################################## begin to prepare data for tracker ###################################################################
                time_start_prepare_costs = time.time()
                # mapping_frameid_bbox_to_features: a dict mapping string 'frameid_bbox coordinates' to a 512-D feature vector, the string 'frameid_bbox coordinates' has length 36, an example is '0930[(467.0, 313.0), (508.0, 424.0)]' where 0930 is frame id, the coordinates are in the format [(left, top), (right, bottom)]
                mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost = prepare_costs_for_tracking_alg(idx_stride_between_frame_pair, curr_frame_dict, tracklet_pose_collection, next_frame_dict, person_to_person_matching_matrix_normalized, node_id_cnt, tracklet_inner_idx, mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost, mapping_frameid_bbox_to_features)
                time_end_prepare_costs = time.time()
        node_id_cnt += len(curr_frame_dict['bbox_list'])
    if len(mapping_edge_id_to_cost) > 0:
        max_in_mapping_edge_id_to_cost = max([mapping_edge_id_to_cost[x] for x in mapping_edge_id_to_cost]) + 1e-6
        if max_in_mapping_edge_id_to_cost >= 0:
            for mapping_edge_id_to_cost_key in mapping_edge_id_to_cost:
                mapping_edge_id_to_cost[mapping_edge_id_to_cost_key] = mapping_edge_id_to_cost[mapping_edge_id_to_cost_key] - max_in_mapping_edge_id_to_cost

    return mapping_node_id_to_bbox,mapping_edge_id_to_cost,mapping_node_id_to_features
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def tracks_combination(tracklet_inner_cnt,remained_tracks,result,result_second,mapping_node_id_to_bbox, mapping_node_id_to_bbox_second,mapping_node_id_to_features,mapping_node_id_to_features_second,source,tracklet_len):
    '''
    进行第二次ssp结果与第一次ssp结果的合并
    n_clusters:最多可能的轨迹数目
    remained_tracks: high detection ssp track with error
    '''
    global batch_id,frame_width,frame_height
    # split_each_track:每条轨迹所包含的节点连接顺序，split_each_track_valid_mask:不正确的点标注为-1,不正确的边标注为0,正确结果标注为１
    split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
    split_each_track_second, split_each_track_valid_mask_second = update_split_each_track_valid_mask_second(result_second)
    # 使用SSP的轨迹结果
    # trajectory_idswitch_dict 对应于在node_list当中的索引
    current_video_segment_predicted_tracks_bboxes_test_SSP,_,_,_,_ = track_processing(split_each_track, mapping_node_id_to_bbox, mapping_node_id_to_features,split_each_track_valid_mask)
    ## low confidence results ##
    ssp_test_second,trajectory_node_dict,trajectory_idswitch_dict,trajectory_idswitch_reliability_dict,trajectory_segment_nodes_dict = track_processing(split_each_track_second, mapping_node_id_to_bbox_second, mapping_node_id_to_features_second,split_each_track_valid_mask_second)
    ## 取出low-confidence当中有用的点 ##
    indefinite_node_list = [] # 以二维列表的形式存储第二次ssp的结果,[[]]
    definite_node_list = []
    indefinite_node = []
    n_clusters = 0
    for track_id in trajectory_idswitch_reliability_dict:
        if len(trajectory_idswitch_reliability_dict[track_id]) == 1: # 此时有可能是单一的node
            segment_nodes = trajectory_segment_nodes_dict[track_id][0]
            if len(segment_nodes) <= 5:
                continue
            # mean_conf = np.mean([mapping_node_id_to_bbox_second[node][1] for node in segment_nodes])
            max_conf = np.max([mapping_node_id_to_bbox_second[node][1] for node in segment_nodes])
            position = np.array([np.array(mapping_node_id_to_bbox_second[node][0]) for node in segment_nodes])
            left,top,right,bottom = np.min(position[:,0,0]),np.min(position[:,0,1]),np.max(position[:,1,0]),np.max(position[:,1,1])
            border = whether_on_border(left, top,right,bottom)
            if max_conf < config["track_high_thresh"] / 2:# or border:
                continue
            indefinite_node_list.append(segment_nodes)
            indefinite_node += segment_nodes
    def show_second_definite_tracks(tracks_SSP):
        tracks = {}
        for track_id in tracks_SSP:
            if len(trajectory_idswitch_reliability_dict[track_id]) == 1:
                tracks[track_id] = copy.deepcopy(tracks_SSP[track_id])
        cluster_frame_list = sorted(np.unique([mapping_node_id_to_bbox_second[x][2] for x in mapping_node_id_to_bbox_second]))
        for frame_name in cluster_frame_list:
            curr_img = cv2.imread(os.path.join(source, frame_name))
            for track_id in tracks:
                for bboxid in tracks[track_id]:
                    if mapping_node_id_to_bbox_second[bboxid][2] == frame_name:
                        left, top = int(mapping_node_id_to_bbox_second[bboxid][0][0][0]), int(mapping_node_id_to_bbox_second[bboxid][0][0][1])
                        right, bottom = int(mapping_node_id_to_bbox_second[bboxid][0][1][0]), int(mapping_node_id_to_bbox_second[bboxid][0][1][1])
                        # cv2.putText(curr_img, str(getDictKey_1(cluster_tracks,bboxid)), (int((left+right)/2), int((top+bottom)/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        cv2.putText(curr_img, str(track_id), (left, top),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
            if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',source.split('/')[-1] + 'second_ssp_definite_tracks/')):
                os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',source.split('/')[-1] + 'second_ssp_definite_tracks/'))
            cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',source.split('/')[-1] + 'second_ssp_definite_tracks/') + frame_name, curr_img)
        return tracks
    definite_tracks = show_second_definite_tracks(ssp_test_second)
    # cluster_tracks,convert the second result into
    # indefinite_node_list = np.unique(indefinite_node_list).tolist() # 之后就不是双层而是单层列表
    cluster_tracks = {} # keys: remained tracks and new track
    for idx, track in enumerate(indefinite_node_list):
        mapping_dict = {}
        for node in track:
            mapping_dict[node] = mapping_node_id_to_bbox_second[node]
        if idx <= len(remained_tracks)-1:
            cluster_tracks[remained_tracks[idx]] = mapping_dict
        else:
            track_id = idx - (len(remained_tracks)-1) + list(current_video_segment_predicted_tracks_bboxes_test_SSP.keys())[-1] # 第一次ssp的keys
            # remained_tracks.append(track_id)
            cluster_tracks[track_id] = mapping_dict

    def my_distance(point1, point2):
        dimension = len(point1)
        result = 0.0
        for i in range(dimension - 1):
            result += abs(point1[i] - point2[i]) ** 2
        eps = 0.01
        if point1[dimension - 1] == point2[dimension - 1]:
            result += 1 / eps
        return result
    # source = '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1'
    def getDictKey_1(myDict, value): #  得到node属于的轨迹
        return [k for k, v in myDict.items() if value in list(v.keys())][0]
    def show_clusters(cluster_tracks,mapping_node_id_to_bbox):
        cluster_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in indefinite_node]))
        for frame_name in cluster_frame_list:
            curr_img = cv2.imread(os.path.join(source, frame_name))
            for bboxid in indefinite_node:
                if mapping_node_id_to_bbox[bboxid][2] == frame_name:
                    left, top = int(mapping_node_id_to_bbox[bboxid][0][0][0]), int(
                        mapping_node_id_to_bbox[bboxid][0][0][1])
                    right, bottom = int(mapping_node_id_to_bbox[bboxid][0][1][0]), int(
                        mapping_node_id_to_bbox[bboxid][0][1][1])
                    # cv2.putText(curr_img, str(getDictKey_1(cluster_tracks,bboxid)), (int((left+right)/2), int((top+bottom)/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.putText(curr_img, str(getDictKey_1(cluster_tracks, bboxid)), (left, top),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.putText(curr_img,str(round(mapping_node_id_to_bbox[bboxid][1],2)), (right, bottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # 字体大小为1
                    cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
            if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',
                                               source.split('/')[-1] + '_cluster_results/')):
                os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',
                                         source.split('/')[-1] + '_cluster_results/'))
            cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',
                                     source.split('/')[-1] + '_cluster_results/') + str(tracklet_inner_cnt) + '_'+frame_name, curr_img)
    def show_fix_clusters(tracks,mapping_node_id_to_bbox): # 显示clusters处理之后的结果
        cluster_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
        for frame_name in cluster_frame_list:
            curr_img = cv2.imread(os.path.join(source, frame_name))
            for track_id in tracks:
                for bboxid in tracks[track_id]:
                    if mapping_node_id_to_bbox[bboxid][2] == frame_name:
                        left, top = int(mapping_node_id_to_bbox[bboxid][0][0][0]), int(
                            mapping_node_id_to_bbox[bboxid][0][0][1])
                        right, bottom = int(mapping_node_id_to_bbox[bboxid][0][1][0]), int(
                            mapping_node_id_to_bbox[bboxid][0][1][1])
                        # cv2.putText(curr_img, str(getDictKey_1(cluster_tracks,bboxid)), (int((left+right)/2), int((top+bottom)/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        cv2.putText(curr_img, str(track_id), (left, top),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
            if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',
                                               source.split('/')[-1] + '_fixed_clusters/')):
                os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',
                                         source.split('/')[-1] + '_fixed_clusters/'))
            cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all',
                                     source.split('/')[-1] + '_fixed_clusters/') + str(tracklet_inner_cnt) + '_'+frame_name, curr_img)
        return tracks

    # kmeans_visualizer.show_clusters(sample, clusters, final_centers)
    #show_clusters(cluster_tracks,mapping_node_id_to_bbox_second)

    definite_track_list = set(split_each_track.keys()) - set(remained_tracks) # definite track in first ssp
    def results_return(definite_track_list,cluster_tracks):
        ##### 删除掉长度为1的轨迹并且进行结果的整理返回 #####
        split_each_track_refined = {}
        final_track_list = np.unique(list(definite_track_list) + list(cluster_tracks.keys())).tolist()
        base_node = max(list(mapping_node_id_to_bbox.keys()))
        for track_id in final_track_list:
            # if trajectory_idswitch_reliability_dict[track_id][0] == tracklet_len:
            if track_id not in list(cluster_tracks.keys()) and track_id in split_each_track: #  不在clusters当中并且在split_each_track当中
            ### 去除split_each_track当中唯一点形成的轨迹 ###
                # if int((len(split_each_track[track_id])+1)/2) > 1:
                split_each_track_refined[track_id] = copy.deepcopy(split_each_track[track_id])
            # 插入节点之间的连边形成轨迹
            elif track_id in cluster_tracks:
                for node_to_add in cluster_tracks[track_id]:
                    if track_id not in split_each_track_refined:
                        split_each_track_refined[track_id] = []
                    node_to_add_real = node_to_add + base_node
                    split_each_track_refined[track_id].append([str(2 * node_to_add_real - 1), str(2 * node_to_add_real)])
                split_each_track_refined[track_id] = interpolate_to_obtain_traj(split_each_track_refined[track_id])
            else:
                continue
        delete_list = []

        for split_each_track_refined_key in split_each_track_refined:
            #mean_score = np.mean([mapping_node_id_to_bbox[int(node_pair[1]) / 2][1] for node_pair in split_each_track_refined[split_each_track_refined_key] if int(node_pair[1]) % 2 == 0])
            #print(split_each_track_refined_key,'mean confidece',mean_score,'length',(len(split_each_track_refined[split_each_track_refined_key])+1)/2)
            #remain_flag = mean_score >= 0.75 or mean_score*math.sqrt((len(split_each_track_refined[split_each_track_refined_key])+1)/2) > 2 # 设置为1
            if (len(split_each_track_refined[split_each_track_refined_key])+1)/2 <= 1: # or mean_score*math.sqrt((len(split_each_track_refined[split_each_track_refined_key])+1)/2)< 2 : # 取消掉长度小于1的轨迹
            # # if (len(split_each_track_refined[split_each_track_refined_key])+1)/2 <= 1 or mean_score < 0.8: # 取消掉长度小于1的轨迹
            # ### 0.8 太大了 ###
                delete_list.append(split_each_track_refined_key)
        # del split_each_track_refined[0] 没有必要
        [split_each_track_refined.pop(track) for track in delete_list]
        result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track_refined)
        return result
    if len(cluster_tracks) == 1:
        return results_return(definite_track_list,cluster_tracks)

    cluster_all_keys = set(cluster_tracks.keys())
    ######## 在此之前cluster_tracks当中的key没有变化，但是之后会发生变化 #######
    frame_list = sorted(np.unique([mapping_node_id_to_bbox_second[x][2] for x in mapping_node_id_to_bbox_second]))
    frame_start,frame_end = int(frame_list[0].split('.')[0]),int(frame_list[-1].split('.')[0])
    frame_span = list(range(frame_start,frame_end+1)) # int类型表示frame
    ##### 处理掉 K-means当中重复的帧 #####
    def remove_dulplicate_frames(cluster_tracks):
        for track1 in cluster_tracks:
            frame_cnt_dict = {} # 统计当前track当中每帧出现的次数
            frame_node_dict = {} # 每帧当中出现的node,帧使用整数表示
            for node in cluster_tracks[track1]:
                if int(mapping_node_id_to_bbox_second[node][2].split('.')[0]) not in frame_cnt_dict:
                    frame_cnt_dict[int(mapping_node_id_to_bbox_second[node][2].split('.')[0])] = 1
                    frame_node_dict[int(mapping_node_id_to_bbox_second[node][2].split('.')[0])] = [node]
                else:
                    frame_cnt_dict[int(mapping_node_id_to_bbox_second[node][2].split('.')[0])] += 1
                    frame_node_dict[int(mapping_node_id_to_bbox_second[node][2].split('.')[0])].append(node)
            # 找出存在冲突的帧
            if max(list(frame_cnt_dict.values())) < 2:
                continue
            else:
                for frame in frame_cnt_dict:
                    if frame_cnt_dict[frame] > 1: # 存在冲突
                        # 利用前面或者后面的帧来解决冲突
                        Flag = False # 标记冲突是否可以解决
                        for frame_before in range(frame-1,frame_start-1,-1):
                            if (frame_before in frame_cnt_dict) and frame_cnt_dict[frame_before] == 1: # 需要判断是否存在
                                Flag = True
                                bbox_before = np.array(mapping_node_id_to_bbox_second[frame_node_dict[frame_before][0]][0])
                                bbox_now = np.array([mapping_node_id_to_bbox_second[node][0] for node in frame_node_dict[frame]])
                                matrix = compute_iou_between_bbox_list(bbox_before.reshape(-1, 2, 2), bbox_now.reshape(-1, 2, 2)) #[(left, top), (right, bottom)] (row，col)
                                col_idx = np.argmax(matrix)
                                final_node = frame_node_dict[frame][col_idx]
                                remove_node = set(frame_node_dict[frame]) - set([frame_node_dict[frame][col_idx]])
                                [cluster_tracks[track1].pop(node) for node in remove_node]
                                frame_cnt_dict[frame] = 1
                                frame_node_dict[frame] = [final_node] # list
                                break
                        if not Flag:
                            for frame_after in range(frame+1,frame_end+1):
                                if (frame_after in frame_cnt_dict) and frame_cnt_dict[frame_after] == 1:
                                    Flag = True
                                    bbox_after = np.array(mapping_node_id_to_bbox_second[frame_node_dict[frame_after][0]][0])
                                    bbox_now = np.array([mapping_node_id_to_bbox_second[node][0] for node in frame_node_dict[frame]])
                                    matrix = compute_iou_between_bbox_list(bbox_after.reshape(-1, 2, 2), bbox_now.reshape(-1, 2, 2)) #[(left, top), (right, bottom)] (row，col)
                                    col_idx = np.argmax(matrix)
                                    final_node = frame_node_dict[frame][col_idx]
                                    remove_node = set(frame_node_dict[frame]) - set([frame_node_dict[frame][col_idx]])
                                    [cluster_tracks[track1].pop(node) for node in remove_node]
                                    frame_cnt_dict[frame] = 1
                                    frame_node_dict[frame] = [final_node]
                                break
                        # 冲突无法解决的时候选取每帧当中置信度更高bbox
                        if not Flag:
                            confidence_list = [mapping_node_id_to_bbox_second[node][1] for node in frame_node_dict[frame]]
                            final_node = frame_node_dict[frame][np.argmax(confidence_list)]
                            remove_node = set(frame_node_dict[frame]) - set([final_node])
                            [cluster_tracks[track1].pop(node) for node in remove_node]
                            frame_cnt_dict[frame] = 1
                            frame_node_dict[frame] = [final_node]
        return cluster_tracks

    cluster_tracks = remove_dulplicate_frames(cluster_tracks)
    if config["debug"]:
        show_fix_clusters(cluster_tracks,mapping_node_id_to_bbox_second)
    return results_return(definite_track_list,cluster_tracks)

def whether_on_border(x1,y1,x2,y2):
    global frame_width,frame_height
    gap = 60  # gap设置太大会把最后一段轨迹给分开来
    border_x_min,border_x_max =  gap ,frame_width-gap
    border_y_min,boder_y_max = gap,frame_height-gap
    flag = x2 >= border_x_max or x1 <= border_x_min or y2 >= boder_y_max or y1 <= border_y_min
    if flag:
        return True
    else:
        return False

def track(exp,config):
    #### global parameters ####
    global current_video_segment_predicted_tracks
    global current_video_segment_predicted_tracks_bboxes
    global current_video_segment_representative_frames
    global current_video_segment_all_traj_all_object_features
    global previous_video_segment_predicted_tracks
    global previous_video_segment_predicted_tracks_bboxes
    global previous_video_segment_representative_frames
    global previous_video_segment_all_traj_all_object_features
    global current_video_segment_predicted_tracks_backup
    global current_video_segment_predicted_tracks_bboxes_backup
    global current_video_segment_representative_frames_backup
    global current_video_segment_all_traj_all_object_features_backup
    global stitching_tracklets_dict
    global batch_id,batch_stride,batch_stride_write
    global frame_width
    global frame_height
    global tracklet_len
    global gap
    tracklet_len = config["tracklet_len"]
    #### initialization ####
    output_dir = osp.join(exp.output_dir, config["benchmark"],str(config["split"]),config["seq_name"],time.strftime("%b%d_%H%M_")+'window_size_'+str(config["tracklet_len"]))#exp.output_dir='./YOLOX_outputs,benchmark:dataset name
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(config["cfg_file"], os.path.join(output_dir, "ml_memAE_sc_cfg.yaml"))

    file_name = os.path.join(exp.output_dir, config["benchmark"])

    ssp_path = os.path.join(output_dir,'vis_ssp_first') # 第一次ssp结果路径,不能以/结尾
    if not os.path.exists(ssp_path):
        os.makedirs(ssp_path,exist_ok=True)
    ssp_second_path = os.path.join(output_dir,'vis_ssp_second') # 第二次ssp结果路径
    if not os.path.exists(ssp_second_path):
        os.makedirs(ssp_second_path,exist_ok=True)
    fixed_result_path = os.path.join(output_dir,'vis_fixed') # 修正之后每个batch的结果
    if not os.path.exists(fixed_result_path):
        os.makedirs(fixed_result_path,exist_ok=True)
    vis_result_path = os.path.join(output_dir,'vis') # 最终结果
    if not os.path.exists(vis_result_path):
        os.makedirs(vis_result_path,exist_ok=True)

    device = config["device"]
    txtname = os.path.join(output_dir,config["seq_name"]+'.txt') # 写入的txt文件名称

    setup_logger(file_name, distributed_rank=1, filename="val_log.txt", mode="a")
    # if config["conf"] is not None:
    #     exp.test_conf = config["conf"]
    # if config["nms"] is not None:
    #     exp.nmsthre = config["nms"]
    # if config["tsize"] is not None:
    #     exp.test_size = (config["tsize"], config["tsize"])

    #### loading model ####
    model = exp.get_model()
    model.cuda()
    model.eval()
    ckpt_file = config["det_model_path"]
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location=config["device"])
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    is_distributed = False  # gpu个数大于1的时候is_distributed为True

    if config["fuse"]:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    model.eval()
    if config["fp16"]:  # True
        model = model.half()
    encoder = FastReIDInterface(config["fast_reid_config"], config["fast_reid_weights"], device)
    #gmc = GMC(method=config["cmc_method"], verbose=[config["name"], config["ablation"]])
    ######################################################### detection #####################################################################
    tracklet_pose_collection = [] # 第一次ssp collection
    tracklet_pose_collection_second = [] # 第二次ssp collection
    ################################################# the above variables need to be cleared for each tracklet ##############################
    ##### load files #####
    if osp.isdir(config["path"]):
        files = get_image_list(config["path"])
    else:
        files = [config["path"]]
    files.sort() # 对文件进行排序
    if config["specific_clip"]:
        files = files[:config["specific_clip"]]
    predictor = Predictor(model, exp, config["device"], config["fp16"])
    batch_id = 0 # window_id
    base_track_id = 0 # 写数据时的轨迹编号
    batch_stride =  tracklet_len - 1
    batch_stride_write =  tracklet_len - 1
    tracklet_inner_cnt =  tracklet_len - 1 # 只有跟踪之后才改变值,9
    current_video_segment_predicted_tracks_backup = {}
    current_video_segment_predicted_tracks_bboxes_backup = {}
    current_video_segment_representative_frames_backup = {}
    total_frames = len(files)
    batch_cnt = math.ceil((total_frames-1)/batch_stride)# 一共47+1个batch，429张图片, the last batch:5 frames
    unmatched_tracks_memory_dict = {} # 记忆轨迹unmatched的次数，超过一定次数之后舍弃轨迹
    # for path, img, im0s, vid_cap in dataset: # check whether in reasonable order
    img_size = exp.test_size # (896,1600)
    results = []
    tracker = SSPTracker(config)
    for frame_id,img_path in enumerate(files,1):
        # Detect objects
        outputs, img_info = predictor.inference(img_path, timer)
        frame_height = img_info['height']
        frame_width = img_info['width']
        # Process detections
        box_detected = []
        box_confidence_scores = []

        # 每次只保存当前batch的结果
        # tracklet_pose_collection = conduct_pose_estimation(webcam, path, out, im0s, pred, img, dataset, save_txt, save_img, view_img, box_detected, head_box_detected, foreignmatter_box_detected, box_confidence_scores, head_box_confidence_scores, foreignmatter_box_confidence_scores, centers, scales, vid_path, vid_writer, vid_cap, tracklet_pose_collection, names, colors, pose_transform, bbox_confidence_threshold, tracklet_inner_cnt, need_face_recognition_switch, face_verification_thresh, opt['iou_thres'])
        if outputs[0] is not None:
        ############ tracklet pose collection #########
            tracklet_pose_collection,tracklet_pose_collection_second = tracklet_collection(img_path,img_size,outputs, img_info, box_detected, box_confidence_scores, tracklet_pose_collection,tracklet_pose_collection_second, (config["track_high_thresh"],config["track_low_thresh"]))
        if total_frames <= tracklet_len: # 不足一个batch
            if len(tracklet_pose_collection) < total_frames:
                continue
        elif batch_id < batch_cnt - 1: # 中间的batch
            if len(tracklet_pose_collection) < tracklet_len:
                continue
            elif len(tracklet_pose_collection) > tracklet_len and (len(tracklet_pose_collection)-tracklet_len)% batch_stride != 0:
                continue
            elif len(tracklet_pose_collection) > tracklet_len and (len(tracklet_pose_collection)-tracklet_len)% batch_stride == 0:
                tracklet_pose_collection[0:batch_stride] = []
                tracklet_pose_collection_second[0:batch_stride] = []
        else: # 最后一个batch，之后的batch不足
            if len(tracklet_pose_collection) < tracklet_len + (total_frames - batch_id*batch_stride-1):
                continue
            else:
                tracklet_pose_collection[0:batch_stride] = []
                tracklet_pose_collection_second[0:batch_stride] = []
                tracklet_len = len(tracklet_pose_collection)

        ##### 确保在最开始的时候tracklet_pose_collection_second ##### 与前一个batch重合的帧 has no high confidence detection nodes
        for i in range(tracklet_len-batch_stride):
            remove_bbox = []
            remove_conf = []
            for idx,bbox in enumerate(tracklet_pose_collection_second[i]['bbox_list']):
                if tracklet_pose_collection_second[i]['box_confidence_scores'][idx] > config["track_high_thresh"]:
                    remove_bbox.append(bbox)
                    remove_conf.append(tracklet_pose_collection_second[i]['box_confidence_scores'][idx])
            [tracklet_pose_collection_second[i]['bbox_list'].remove(bbox) for bbox in remove_bbox]
            [tracklet_pose_collection_second[i]['box_confidence_scores'].remove(conf) for conf in remove_conf]

        if tracklet_pose_collection[-1] == []:
            if len([x for x in tracklet_pose_collection if x != []]) == 0: # starting frames are without objects
                tracklet_pose_collection = [x for x in tracklet_pose_collection if x != []]
                tracklet_inner_cnt = 0
                continue
            else: # if middle frames are without detections
                tracklet_pose_collection = [x for x in tracklet_pose_collection if x != []]
                tracklet_inner_cnt = len(tracklet_pose_collection)
                continue

        ######################################################## conduct tracking ######################################################################

        if ((tracklet_inner_cnt + 1) >= tracklet_len + batch_id*batch_stride or batch_id == batch_cnt-1) and ({} not in tracklet_pose_collection): # and (whether_conduct_tracking == 1):
            batch_id += 1
            print('batch', batch_id)
            # allen added to filter out frames without people or without head,进入到下一帧
            if min([len(x['bbox_list']) for x in tracklet_pose_collection]) == 0: # or min([len(x['head_bbox_list']) for x in tracklet_pose_collection[-tracklet_len:]]) == 0:
                tracklet_inner_cnt += 1
                continue
            ################################################ collect the vectors of bboxes #############################################################
            mapping_node_id_to_bbox,mapping_edge_id_to_cost,mapping_node_id_to_features = mapping_data_preparation(files,encoder,tracklet_pose_collection, tracklet_inner_cnt,True)
            ### 需要对第二次ssp当中的tracklet_pose_collection_second进行修正 ###
            ############################ organize graph and call #############################################################################

            # if int(mapping_node_id_to_bbox[max([x for x in mapping_node_id_to_bbox])][2][:-4]) == len(dataset.files) - 1 and len(mapping_node_id_to_bbox) == 0 and len(mapping_edge_id_to_cost) == 0:
            if int(mapping_node_id_to_bbox[max([x for x in mapping_node_id_to_bbox])][2][:-4]) == len(files) - 1 and len(mapping_node_id_to_bbox) == 0 and len(mapping_edge_id_to_cost) == 0:
                break

            result = tracking(mapping_node_id_to_bbox, mapping_edge_id_to_cost, tracklet_inner_cnt) # tracking函数为ssp算法实现

            indefinite_node = [] # 表示第一次ssp当中不确定的点
            error_tracks = [] # 第一次ssp当中出错的轨迹段
            split_each_track_SSP, split_each_track_valid_mask  = update_split_each_track_valid_mask(result)
            current_video_segment_predicted_tracks_bboxes_test_SSP,trajectory_node_dict,trajectory_idswitch_dict,trajectory_idswitch_reliability_dict,trajectory_segment_nodes_dict = track_processing(split_each_track_SSP, mapping_node_id_to_bbox, mapping_node_id_to_features,split_each_track_valid_mask) # (config["initial_iou_thresh"],config["iou_thresh_step"])
            n_clusters = 0 # 最大轨迹数目，第二次低置信度点不增加轨迹数目只改变ssp结果
            for track_id in trajectory_idswitch_reliability_dict:
                if len(trajectory_idswitch_reliability_dict[track_id]) > 1: # 发生了idswitch的轨迹都应当加入到error_track当中
                    indefinite_node += trajectory_node_dict[track_id]
                    n_clusters += 1
                    error_tracks.append(track_id)

                elif len(trajectory_idswitch_reliability_dict[track_id]) == 1 and len(current_video_segment_predicted_tracks_bboxes_test_SSP[track_id])< 3:
                    # if trajectory_idswitch_reliability_dict[track_id][0] == 1: # 对于只有一个valid_node的不加入进行修正
                    #     continue
                    position = np.array([np.array(mapping_node_id_to_bbox[node][0]) for node in trajectory_node_dict[track_id]])
                    left,top,right,bottom = np.min(position[:,0,0]),np.min(position[:,0,1]),np.max(position[:,1,0]),np.max(position[:,1,1])
                    border = whether_on_border(left, top,right,bottom)
                    # if border:
                    #     continue
                    indefinite_node += trajectory_node_dict[track_id]
                    n_clusters += 1
                    error_tracks.append(track_id)
            error_tracks = np.unique(error_tracks).tolist()
            indefinite_node = np.unique(indefinite_node).tolist()

            unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
            ##### 修正之前纯SSP算法结果 #####
            if config["debug"]:
                #### high confidence score ####
                for frame_name in unique_frame_list:
                    curr_img = cv2.imread(os.path.join(config["path"], frame_name))
                    # curr_img = cv2.imread('/media/allenyljiang/Seagate_Backup_Plus_Drive/usr/local/VIBE-master/data/neurocomputing/05_0019/' + frame_name)
                    for human_id in split_each_track_SSP: # 当前轨迹id
                        for node_idx in [x for x in range(len(split_each_track_SSP[human_id])) if (x % 2 == 0)]: #偶数表示人的节点 id
                            if mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][2] == frame_name: #  写入帧数，轨迹数，坐标
                                # curr_batch_txt.write(str(unique_frame_list.index(frame_name) + 1) + ',' + str(human_id) + ',' + \
                                #                      str(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][0][0]) + ',' + \
                                #                      str(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][0][1]) + ',' + \
                                #                      str(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][1][0]) + ',' + \
                                #                      str(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][1][1]) + ',-1,-1,-1,-1\n') # mot格式：必须10个数
                                left, top = int(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][0][0]), int(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][0][1])
                                right,bottom = int(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][1][0]), int(mapping_node_id_to_bbox[int(int(split_each_track_SSP[human_id][node_idx][1]) / 2)][0][1][1])
                                cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 2)
                                cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.imwrite(ssp_path + '/'+str(tracklet_inner_cnt)+ '_' + frame_name, curr_img)

            #### 把indefinite_node 当中的节点加入到 tracklet_pose_collection_second当中 ####
            for node in set(indefinite_node):
                bbox,conf,frame = mapping_node_id_to_bbox[node][0],mapping_node_id_to_bbox[node][1],mapping_node_id_to_bbox[node][2]
                tracklet_pose_collection_second[unique_frame_list.index(frame)]['bbox_list'].append(bbox)
                tracklet_pose_collection_second[unique_frame_list.index(frame)]['box_confidence_scores'].append(conf)
            ##### second ssp #####
            second_ssp_node = sum([len(tracklet_pose_collection_second[frame]['bbox_list']) for frame in range(len(tracklet_pose_collection_second))])
            Second_Flag = False
            if second_ssp_node > 0 and batch_id > 1: # 没有node的时候不用进行第二次的ssp,第一个batch不进行第二次的ssp
                mapping_node_id_to_bbox_second,mapping_edge_id_to_cost_second,mapping_node_id_to_features_second = mapping_data_preparation(files,encoder,tracklet_pose_collection_second, tracklet_inner_cnt,False)

                if len(mapping_edge_id_to_cost_second) > 0:
                    ##### 节点数目大于10的情况才能进行ssp #####
                    result_second = tracking(mapping_node_id_to_bbox_second, mapping_edge_id_to_cost_second, tracklet_inner_cnt)
                    if 'Predicted tracks' in result_second[0] and len(indefinite_node) > 0:
                    # if len(mapping_node_id_to_bbox_second) > 10:
                        split_each_track_SSP_second,split_each_track_valid_mask_second= update_split_each_track_valid_mask_second(result_second)
                        if config["debug"]:
                            ##### 低置信度框 ###
                            for frame_name in unique_frame_list:
                                curr_img = cv2.imread(os.path.join(config["path"], frame_name))
                                # curr_img = cv2.imread('/media/allenyljiang/Seagate_Backup_Plus_Drive/usr/local/VIBE-master/data/neurocomputing/05_0019/' + frame_name)
                                for human_id in split_each_track_SSP_second: # 当前轨迹id
                                    for node_idx in [x for x in range(len(split_each_track_SSP_second[human_id])) if (x % 2 == 0)]: #偶数表示人的节点 id
                                        if mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][2] == frame_name: #  写入帧数，轨迹数，坐标
                                            # curr_batch_txt.write(str(unique_frame_list.index(frame_name) + 1) + ',' + str(human_id) + ',' + \
                                            #                      str(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][0][0]) + ',' + \
                                            #                      str(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][0][1]) + ',' + \
                                            #                      str(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][1][0]) + ',' + \
                                            #                      str(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][1][1]) + ',-1,-1,-1,-1\n') # mot格式：必须10个数
                                            left, top = int(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][0][0]), int(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][0][1])
                                            right, bottom = int(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][1][0]), int(mapping_node_id_to_bbox_second[int(int(split_each_track_SSP_second[human_id][node_idx][1]) / 2)][0][1][1])
                                            cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 2)
                                            cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.imwrite(ssp_second_path + '/'+ str(tracklet_inner_cnt) + '_' + frame_name, curr_img)
                        ## 进行轨迹合并 ##
                        # result = tracks_combination(error_tracks,result_second,mapping_node_id_to_bbox_second, mapping_node_id_to_features_second, source,tracklet_len,n_clusters)
                        result = tracks_combination(tracklet_inner_cnt,error_tracks,result,result_second,mapping_node_id_to_bbox, mapping_node_id_to_bbox_second,mapping_node_id_to_features,mapping_node_id_to_features_second,config["path"],tracklet_len)
                        ### 修改mapping_node_id_to_bbox_second以及mapping_node_id_to_features_second的key ##
                        new_key = (np.array(list(mapping_node_id_to_bbox_second.keys())) + max(list(mapping_node_id_to_bbox.keys()))).astype(np.int).tolist()
                        new_bbox_values = list(mapping_node_id_to_bbox_second.values())
                        new_feature_values = list(mapping_node_id_to_features_second.values())
                        mapping_node_id_to_bbox_second = dict(zip(new_key,new_bbox_values))
                        mapping_node_id_to_features_second = dict(zip(new_key,new_feature_values))

                        ### 对mapping_node_id_to_second 与 mapping_node_id_to_bbox 进行合并 ###
                        mapping_node_id_to_bbox.update(mapping_node_id_to_bbox_second)
                        mapping_node_id_to_features.update(mapping_node_id_to_features_second)
                        Second_Flag = True
                # elif 'Predicted tracks' not in result_second[0] and len(indefinite_node) > 0:
            split_each_track, split_each_track_mask = update_split_each_track_valid_mask(result)
            if not Second_Flag:
                for track_id in error_tracks:
                    if len(trajectory_idswitch_reliability_dict[track_id]) == 1:
                        continue
                    #[split_each_track.pop(track) for track in error_tracks]
                # [valid_mask.pop(track) for track in error_tracks] # valid_mask 并没有使用
            current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_confidence_score, current_video_segment_predicted_tracks_bboxes, current_video_segment_representative_frames,current_video_segment_predicted_tracks_bboxes_test,current_trajectory_similarity_dict,current_video_segment_all_traj_all_object_features = convert_track_to_stitch_format(split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features)
            # ## 以上为添加第二次ssp的结果
            # current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_confidence_score, current_video_segment_predicted_tracks_bboxes, current_video_segment_representative_frames,current_video_segment_predicted_tracks_bboxes_test,current_trajectory_similarity_dict,current_video_segment_all_traj_all_object_features = convert_track_to_stitch_format(split_each_track_SSP,mapping_node_id_to_bbox,mapping_node_id_to_features)
            # split_each_track = split_each_track_SSP
            # # current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, current_video_segment_all_traj_all_object_features, _ = interpolation_fix_missed_detections(current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, current_video_segment_all_traj_all_object_features, tracklet_pose_collection)
            frame_list = np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox])
            curr_last_frame_node_list = [x for x in mapping_node_id_to_bbox if mapping_node_id_to_bbox[x][2] == frame_list[-1]]  # 当前batch
            unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
            if config["debug"]:
            ##### 修正之后当前batch的图片写入  ########
                for frame_name in unique_frame_list:
                    curr_img = cv2.imread(os.path.join(config["path"], frame_name))
                    for human_id in current_video_segment_predicted_tracks_bboxes: # 第一帧所有的human_id与轨迹id相同
                        for bbox in current_video_segment_predicted_tracks_bboxes[human_id]: # dict bbox:key
                            if bbox == frame_name: #  写入帧数，轨迹数，坐标
                                left, top = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1])
                                right, bottom = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][1])
                                cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 2)
                                cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                    cv2.imwrite(fixed_result_path + '/'+str(tracklet_inner_cnt) + '_' + frame_name, curr_img)
            # if previous_video_segment_predicted_tracks != {}:
            online_targets = tracker.update(current_video_segment_predicted_tracks_bboxes_test,config,frame_list)
            if tracker.batch_id == 1:
                valid_frames = tracker.frames
            elif tracker.batch_id == batch_cnt:
                valid_frames = tracker.frames[-(total_frames - (batch_id - 1) * batch_stride - 1):]
            else:
                valid_frames = tracker.frames[-tracker.update_len:] # int类型不可以迭代

            for frame in valid_frames: # 当前窗口需要写入的数据
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    if frame in t.frames:
                        try:
                            index = t.frames.index(frame)
                            xyxy = t.xyxys[index]
                            score = t.score[index]
                            tid = t.track_id
                            # if tlwh[2] * tlwh[3] > self.args.min_box_area:
                            online_tlwhs.append(xyxy)
                            online_ids.append(tid)
                            online_scores.append(score)
                        except IndexError: # OT-7
                            # 捕获IndexError异常并打印错误信息
                            print(f"IndexError: list index out of range at index {index},{frame},{t.frames}")
                            print(len(t.xyxys))
                # save results
                results.append((frame, online_tlwhs, online_ids, online_scores)) # 当前帧的所有
                img_path = os.path.join(config["path"], frame)
                write_vis_results(img_path, vis_result_path, results)

        tracklet_inner_cnt += batch_stride  # 步长
    write_results(txtname,results)



if __name__ == '__main__':
    cfg_file = "/home/allenyljiang/Documents/SSP_EM/configs/avenue.yaml"
    config = yaml.safe_load(open(cfg_file))
    ### set all seed for reproduction ###
    seed = 12345
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


    exp = get_exp(config["exp_file"], '')# 模型参数（键值对形式）
    mainTimer = Timer()
    mainTimer.tic()
    if config["seq_name"]:
        seqs = [config["seq_name"]]
    else:
        if config["split"] == "train":
            seqs = os.listdir(config["dataset_dir"]+'/'+'training_frames')
        else:
            seqs = os.listdir(config["dataset_dir"] + '/' + 'testing_frames')

    exp.test_conf = max(0.001, config["track_low_thresh"] - 0.01)  # 0.09
    for seq in seqs:
        config["seq_name"] = seq
        if config["split"] == "train":
            config["path"] = os.path.join(config["dataset_dir"],'training_frames',str(seq))
        else:
            config["path"] = os.path.join(config["dataset_dir"], 'testing_frames', str(seq))
        print('current seq',config["seq_name"])
        track(exp,config)
    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    # print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    # print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))
    # detect(opt,exp,args)

