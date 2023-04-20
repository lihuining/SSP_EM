from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import io
import tempfile

from loguru import logger
import gc
import tracemalloc
import warnings

from pympler import tracker

from memory_profiler import profile
import objgraph
import argparse
import os
import platform
import shutil
import time
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from PIL import Image as Img
from PIL import ImageTk
import cv2
import torch
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
from numpy import random
import numpy as np
import _thread
import tkinter as tk
from tkinter import Tk, Label
import PIL.Image
from tkinter import ttk
from PIL import Image, ImageTk
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from scipy.stats import multivariate_normal
import sys
import time
import math
import ot
# import ot.gpu
from subprocess import *
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import multiprocessing
import copy
import sys
import os.path as osp
import torch.nn as nn
from similarity_module import torchreid
from similarity_module.torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)
from similarity_module.scripts.default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)
import json
from sklearn import metrics
# from ab_det_realtime.ab_detect_utils import *
from tkinter import *
from itertools import permutations
#################################################### detector related import start ######################################################################
import torch
import torchvision
import matplotlib.pyplot as plt
import glob
import torch.backends.cudnn as cudnn
from det_model_new.models.experimental import *
from det_model_new.utils.datasets import *
from det_model_new.utils.torch_utils import *
from det_model_new.utils import torch_utils
from det_model_new.utils.general import *#
from yolov5_head_body_detector import *
# weights_file_dir = ['/home/experiments/demo_multi_object_tracking/weights/helmet_head_person_m.pt']
dump_curr_video_name = '/usr/local/SSP_EM/tracking_for_integration'
dump_switch = 0 # ??
dump_further_switch = 0
dump_further_switch_20211011 = 0
dump_further_switch_optimizer = 0
dump_stitching_tracklets_switch = 0
import codecs
from fix_trajs_v5 import *

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
##### yolox #####

from yolox.core import launch
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
from yolox.tracker import matching
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
curr_batch_img_buffer = {}
# result_dict = {}

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
whether_conduct_tracking = False
batch_id = 0
frame_height = 0
frame_width = 0
skeletons = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
batch_stride = tracklet_len - 1
batch_stride_write = tracklet_len - 1
foreign_matter_cls_id_dict = {
'bicycle': 1,
'motorcycle': 3,
'skateboard': 36
}
# tracemalloc.start() # 开始跟踪内存分配
# snapshot = tracemalloc.take_snapshot()
# np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore')

# np.warnings.filterwarnings('ignore',category=np.RankWarning)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

'''
-f
../exps/example/mot/yolox_x_mix_mot20_ch.py
-c
../pretrained/bytetrack_x_mot20.tar
-b
1
-d
1
--fp16
--fuse
--match_thresh
0.7
--mot20
'''
def track_processing(split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features,split_each_track_valid_mask):
    iou_thresh = 0.55 # 0.789
    iou_thresh_step = 0.017 # 0.017
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
            # [bbox_pre[0][1], bbox_pre[1][1], bbox_pre[0][0], bbox_pre[1][0]]
            idx_tmp = 1  # initial value
            if idx >= 1:
                iou_similarity = compute_iou_single_box([bbox[0][1], bbox[1][1], bbox[0][0], bbox[1][0]],[bbox_pre[0][1], bbox_pre[1][1], bbox_pre[0][0], bbox_pre[1][0]])
                #print(iou_similarity)
                #velocity_x = (bbox[0][0] + bbox[1][0]) / 2 - (bbox_pre[0][0] + bbox_pre[1][0]) / 2 # x-axis
                #velocity_y = (bbox[0][1] + bbox[1][1]) / 2 - (bbox_pre[0][1] + bbox_pre[1][1]) / 2 # y-axis
                iou_thresh_tmp = iou_thresh + int((idx-idx_tmp)/2)*iou_thresh_step
                if iou_similarity < iou_thresh_tmp:
                        #or (np.sign(velocity_x*velocity_x_pre)+np.sign(velocity_y*velocity_y_pre)) == -2:
                    #print(track_id,idx,iou_similarity)
                    trajectory_idswitch_list.append(int(idx/2)) # id从0开始
                    idx_tmp = int(idx)
                    # iou_thresh_tmp = copy.deepcopy(iou_thresh)
                    trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
                    trajectory_segment_list.append(trajectory_segment[:])
                    trajectory_idswitch_reliability = 0
                    trajectory_segment = []

            # if idx >= 1:
            #     iou_similarity = compute_iou_single_box([bbox[0][1], bbox[1][1], bbox[0][0], bbox[1][0]],[bbox_pre[0][1], bbox_pre[1][1], bbox_pre[0][0], bbox_pre[1][0]])
            #     # print(iou_similarity)
            # if idx >= 1 and iou_similarity < iou_thresh:
            #     # print(track_id,idx,iou_similarity)
            #     trajectory_idswitch_list.append(int(idx/2))
            #     trajectory_idswitch_reliability_list.append(trajectory_idswitch_reliability)
            #     trajectory_idswitch_reliability = 0
            #
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

def make_parser():
    # python tools/demo_track.py -h 查看帮助信息/--help
    # 使用时顺序无关紧要
    parser = argparse.ArgumentParser("ByteTrack Demo!") # 创建解析对象
    # -- 表示可选参数，其余表示必选参数
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )# action表示--save_result标志存在时赋值为"store_true"

    # exp file
    parser.add_argument(
        "-f",# 短选项，使用-f/--exp_file均可
        "--exp_file",
        default='exps/example/mot/yolox_x_mix_mot20_ch.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='weights/bytetrack_x_mot20.tar', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    # 设备
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default= True,
        # action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default= True,
        # action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    # 结果可视化
    parser.add_argument('--save-result', dest="save_result",default=False,
                        help='whether save the visualizition result')  # python demo.py --save-result 启用该参数
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def tracemalloc_snapshot(snapshot1):
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print('top 10 differences')
    for stat in top_stats[:10]:
        print(stat)

def single_snapshot():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:15]:
        print(stat)

def sampling_process(coord, side_of_sampling_equilateral_triangle_input, height, width, number_points_around_each_joint):
    results = []
    side_of_sampling_equilateral_triangle = side_of_sampling_equilateral_triangle_input

    results.append([coord[0], coord[1] + 1.732 * side_of_sampling_equilateral_triangle]) #
    results.append([coord[0] - 1.5 * side_of_sampling_equilateral_triangle,
                    coord[1] - 0.866 * side_of_sampling_equilateral_triangle]) #
    results.append([coord[0] + 1.5 * side_of_sampling_equilateral_triangle,
                    coord[1] - 0.866 * side_of_sampling_equilateral_triangle]) #

    results.append([coord[0], coord[1]])
    results.append([coord[0] - side_of_sampling_equilateral_triangle, coord[1]])
    results.append([coord[0] + side_of_sampling_equilateral_triangle, coord[1]])
    results.append([coord[0] - 0.5 * side_of_sampling_equilateral_triangle,
                    coord[1] + 0.866 * side_of_sampling_equilateral_triangle])
    results.append([coord[0] + 0.5 * side_of_sampling_equilateral_triangle,
                    coord[1] + 0.866 * side_of_sampling_equilateral_triangle])
    results.append([coord[0] - 0.5 * side_of_sampling_equilateral_triangle,
                    coord[1] - 0.866 * side_of_sampling_equilateral_triangle])
    results.append([coord[0] + 0.5 * side_of_sampling_equilateral_triangle,
                    coord[1] - 0.866 * side_of_sampling_equilateral_triangle])
    results.append([coord[0], coord[1] - 1.732 * side_of_sampling_equilateral_triangle])
    results.append([coord[0] - 1.5 * side_of_sampling_equilateral_triangle,
                    coord[1] + 0.866 * side_of_sampling_equilateral_triangle])
    results.append([coord[0] + 1.5 * side_of_sampling_equilateral_triangle,
                    coord[1] + 0.866 * side_of_sampling_equilateral_triangle])

    for item in results:
        item[0] = max([min([item[0], width - 1]), 0])
        item[1] = max([min([item[1], height - 1]), 0])
    return results[:number_points_around_each_joint]

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

def people_matching_error_average_sampling(curr_person_bbox_coord, next_person_bbox_coord, curr_img, next_img, average_sampling_density_hori_vert):
    time_start_sampling_step1 = time.time()
    bbox_left_curr = curr_person_bbox_coord[0][0]
    bbox_top_curr = curr_person_bbox_coord[0][1]
    bbox_right_curr = curr_person_bbox_coord[1][0]
    bbox_bottom_curr = curr_person_bbox_coord[1][1]
    bbox_left_next = next_person_bbox_coord[0][0]
    bbox_top_next = next_person_bbox_coord[0][1]
    bbox_right_next = next_person_bbox_coord[1][0]
    bbox_bottom_next = next_person_bbox_coord[1][1]

    # bbox_left_curr = bbox_left_curr * 0.8 + bbox_right_curr * 0.2
    # bbox_right_curr = bbox_left_curr * 0.25 + bbox_right_curr * 0.75
    # bbox_top_curr = bbox_top_curr * 0.8 + bbox_bottom_curr * 0.2
    # bbox_bottom_curr = bbox_top_curr * 0.25 + bbox_bottom_curr * 0.75
    # bbox_left_next = bbox_left_next * 0.8 + bbox_right_next * 0.2
    # bbox_right_next = bbox_left_next * 0.25 + bbox_right_next * 0.75
    # bbox_top_next = bbox_top_next * 0.8 + bbox_bottom_next * 0.2
    # bbox_bottom_next = bbox_top_next * 0.25 + bbox_bottom_next * 0.75

    curr_hori_coords = np.linspace(bbox_left_curr, bbox_right_curr, num=average_sampling_density_hori_vert, endpoint=False)
    curr_vert_coords = np.linspace(bbox_top_curr, bbox_bottom_curr, num=average_sampling_density_hori_vert, endpoint=False)
    next_hori_coords = np.linspace(bbox_left_next, bbox_right_next, num=average_sampling_density_hori_vert, endpoint=False)
    next_vert_coords = np.linspace(bbox_top_next, bbox_bottom_next, num=average_sampling_density_hori_vert, endpoint=False)

    batch_v_range_curr = np.repeat(curr_vert_coords, average_sampling_density_hori_vert).astype(int)
    batch_h_range_curr = np.tile(curr_hori_coords, average_sampling_density_hori_vert).astype(int)
    batch_curr = curr_img[batch_v_range_curr, batch_h_range_curr, :].flatten().astype(np.int32)

    batch_v_range_next = np.repeat(next_vert_coords, average_sampling_density_hori_vert).astype(int)
    batch_h_range_next = np.tile(next_hori_coords, average_sampling_density_hori_vert).astype(int)
    batch_next = next_img[batch_v_range_next, batch_h_range_next, :].flatten().astype(np.int32)

    time_start_sampling_method2 = time.time()
    joint_to_joint_matching_matrix = np.ones(
        (len(batch_v_range_curr), len(batch_v_range_next))) * maximum_possible_number
    for curr_img_pixel_idx in range(len(batch_v_range_curr)):
        batch_curr_rolled = np.roll(batch_curr, 3 * curr_img_pixel_idx)
        vert_coords = np.roll(range(len(batch_v_range_curr)), curr_img_pixel_idx)
        hori_coords = range(len(batch_v_range_next))
        joint_to_joint_matching_matrix[vert_coords, hori_coords] = np.sum((batch_curr_rolled.reshape((len(batch_v_range_curr), 3)) - batch_next.reshape((len(batch_v_range_next), 3))) * (batch_curr_rolled.reshape((len(batch_v_range_curr), 3)) - batch_next.reshape((len(batch_v_range_next), 3))), axis=1)
    time_end_sampling_method2 = time.time()
    return joint_to_joint_matching_matrix

def people_matching_error(joint_to_joint_matching_matrix, curr_person_all_joint_coord, joint_coord_curr_frame, next_person_all_joint_coord, joint_coord_next_frame, number_points_around_each_joint, curr_img, next_img):
    side_of_sampling_equilateral_triangle = 999.0
    for bone in skeletons:
        if curr_person_all_joint_coord.index(joint_coord_curr_frame) in bone:
            idx_of_neighbor_coord = [x for x in bone if x != curr_person_all_joint_coord.index(joint_coord_curr_frame)][0]
            collection_distance_with_neighbors = np.sqrt(
                pow((joint_coord_curr_frame[0] - curr_person_all_joint_coord[idx_of_neighbor_coord][0]), 2) + pow((joint_coord_curr_frame[1] - curr_person_all_joint_coord[idx_of_neighbor_coord][1]), 2))
            side_of_sampling_equilateral_triangle = min([max([(collection_distance_with_neighbors / 1.732 / 5.0), 3]), side_of_sampling_equilateral_triangle])
    keypoints_around_joint_curr_frame = sampling_process(joint_coord_curr_frame, side_of_sampling_equilateral_triangle, curr_img.shape[0], curr_img.shape[1], number_points_around_each_joint)
    #############################################################################################################
    side_of_sampling_equilateral_triangle = 999.0
    for bone in skeletons:
        if next_person_all_joint_coord.index(joint_coord_next_frame) in bone:
            idx_of_neighbor_coord = [x for x in bone if x != next_person_all_joint_coord.index(joint_coord_next_frame)][0]
            collection_distance_with_neighbors = np.sqrt(
                pow((joint_coord_next_frame[0] - next_person_all_joint_coord[idx_of_neighbor_coord][0]), 2) + pow((joint_coord_next_frame[1] - next_person_all_joint_coord[idx_of_neighbor_coord][1]), 2))
            side_of_sampling_equilateral_triangle = min([max([(collection_distance_with_neighbors / 1.732 / 5.0), 3]), side_of_sampling_equilateral_triangle])
    keypoints_around_joint_next_frame = sampling_process(joint_coord_next_frame, side_of_sampling_equilateral_triangle, next_img.shape[0], next_img.shape[1], number_points_around_each_joint)
    for number_points_around_each_joint_idx in range(number_points_around_each_joint):
        joint_to_joint_matching_matrix[3*curr_person_all_joint_coord.index(joint_coord_curr_frame)+number_points_around_each_joint_idx, \
                                       3*next_person_all_joint_coord.index(joint_coord_next_frame)+number_points_around_each_joint_idx] = \
            np.linalg.norm(curr_img[int(keypoints_around_joint_curr_frame[number_points_around_each_joint_idx][1]), int(keypoints_around_joint_curr_frame[number_points_around_each_joint_idx][0]), :] - \
                           next_img[int(keypoints_around_joint_next_frame[number_points_around_each_joint_idx][1]), int(keypoints_around_joint_next_frame[number_points_around_each_joint_idx][0]), :])

    return joint_to_joint_matching_matrix

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
    even_node_cost_add = -abs(2 * src_dst_node_cost + even_node_cost)-0.1  # -abs(2 * src_dst_node_cost + most_unreliable_edge_cost + 2 * even_node_cost) / 2 - 0.1

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

def compute_inter_person_similarity(curr_frame_dict, next_frame_dict, maximum_possible_number, curr_img, next_img):
    time_start_compute_inter_person_similarity = time.time()
    time_start_initial = time.time()
    person_to_person_matching_matrix = np.ones((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list']))) * maximum_possible_number
    person_to_person_matching_matrix_iou = np.zeros((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list'])))
    time_end_initial = time.time()
    for curr_person_bbox_coord in curr_frame_dict['bbox_list']:
        for next_person_bbox_coord in next_frame_dict['bbox_list']:
            time_start_sampling = time.time()
            joint_to_joint_matching_matrix = people_matching_error_average_sampling(curr_person_bbox_coord, next_person_bbox_coord, curr_img, next_img, average_sampling_density_hori_vert)
            time_end_sampling = time.time()
            ot_src = [1.0] * joint_to_joint_matching_matrix.shape[0]
            ot_dst = [1.0] * joint_to_joint_matching_matrix.shape[1]
            time_start_ot = time.time()
            transportation_array = ot.emd(ot_src, ot_dst, joint_to_joint_matching_matrix) # sinkhorn(ot_src, ot_dst, joint_to_joint_matching_matrix, 1, method='greenkhorn') # method='sinkhorn_stabilized')
            time_end_ot = time.time()
            time_start_sum = time.time()
            person_to_person_matching_matrix[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = np.sum(joint_to_joint_matching_matrix * transportation_array)
            time_end_sum = time.time()
            time_start_iou = time.time()
            person_to_person_matching_matrix_iou[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                compute_iou_single_box([curr_person_bbox_coord[0][1], curr_person_bbox_coord[1][1], curr_person_bbox_coord[0][0], curr_person_bbox_coord[1][0]], \
                                       [next_person_bbox_coord[0][1], next_person_bbox_coord[1][1], next_person_bbox_coord[0][0], next_person_bbox_coord[1][0]])
            time_end_iou = time.time()
    time_start_post = time.time()
    person_to_person_matching_matrix = 1.0 / person_to_person_matching_matrix / np.max(1.0 / person_to_person_matching_matrix) # similarity
    person_to_person_matching_matrix = person_to_person_matching_matrix * person_to_person_matching_matrix_iou
    person_to_person_matching_matrix_normalized = person_to_person_matching_matrix / np.min(person_to_person_matching_matrix[np.where(person_to_person_matching_matrix!=0)])
    time_end_post = time.time()
    time_end_compute_inter_person_similarity = time.time()
    return person_to_person_matching_matrix_normalized

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

    if dump_further_switch_optimizer == 1:
        out_file = open(os.path.join(dump_curr_video_name, 'mapping_node_id_to_bbox' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json'), "w")
        json.dump(mapping_node_id_to_bbox, out_file)
        out_file.close()
        out_file = open(os.path.join(dump_curr_video_name, 'mapping_edge_id_to_cost' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json'), "w")
        json.dump(mapping_edge_id_to_cost, out_file)
        out_file.close()
        out_file = os.path.join(dump_curr_video_name, 'result_mapping_node_id_to_bbox_str' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(result_mapping_node_id_to_bbox_str, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'result_mapping_edge_id_to_cost_str' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(result_mapping_edge_id_to_cost_str, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'transfer_data_to_tracker' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(transfer_data_to_tracker, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)

    tracking_start_time = time.time()
    gc.collect()
    p = Popen('call/call', stdin=PIPE, stdout=PIPE, encoding='gbk')
    result = p.communicate(input=transfer_data_to_tracker)
    p.terminate()
    p.wait()
    result = [result[0], result[1]]

    if dump_further_switch_optimizer == 1:
        out_file = open(os.path.join(dump_curr_video_name, 'result0_' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json'), "w")
        json.dump(result[0], out_file)
        out_file.close()

    tracking_end_time = time.time()
    print(str(tracking_end_time - tracking_start_time))
    return result

def vis_det_pose_results(save_img, dataset, save_path, vid_path, vid_writer, vid_cap, im0):
    if save_img:
        if dataset.mode == 'images':
            cv2.imwrite(save_path, im0)
        else:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(im0)

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

def tracklet_collection(dataloader,img_size,outputs, info_imgs, ids, box_detected, box_confidence_scores, tracklet_pose_collection,tracklet_pose_collection_second, bbox_confidence_threshold, tracklet_inner_cnt,source):
    # pred - predicted human bounding boxes with confidences
    # path - to current image for pose estimation
    # out - output directory
    # im0s - current image array, 1080x1920x3
    # img - current image array with shape 1x3x1088x1920
    # tracklet_inner_cnt - index of frame
    # pose_model, pose_transform - model for pose estimation
    img_h,img_w,img_id = info_imgs[0], info_imgs[1], ids
    output = outputs[0]
    output = output.cpu()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    scale = min(
        img_size[0] / float(img_h), img_size[1] / float(img_w)
    )
    bboxes /= scale # xyxy
    # bboxes = xyxy2xywh(bboxes)

    cls = output[:, 6] # 0
    scores = output[:, 4] * output[:, 5]
    for ind in range(bboxes.shape[0]):
        label = dataloader.dataset.class_ids[int(cls[ind])]
        # pred_data = {
        #     "image_id": int(img_id),
        #     "category_id": label,
        #     "bbox": bboxes[ind].numpy().tolist(),
        #     "score": scores[ind].numpy().item(),
        #     "segmentation": [],
        # }  # COCO json format
        width = abs(float(bboxes[ind][2].data.cpu().numpy()) - float(bboxes[ind][0].data.cpu().numpy()))
        heigth = abs(float(bboxes[ind][1].data.cpu().numpy()) - float(bboxes[ind][3].data.cpu().numpy()))
        aspect_ratio = width / heigth
        if aspect_ratio > 1.6 or width*heigth < 100:
            continue
        box_detected.append([(float(bboxes[ind][0].data.cpu().numpy()), float(bboxes[ind][1].data.cpu().numpy())), (float(bboxes[ind][2].data.cpu().numpy()), float(bboxes[ind][3].data.cpu().numpy()))])
        box_confidence_scores.append(float(scores[ind].data.cpu().numpy()) + 1e-4*random.random())

    box_detected_high = [box_detected[box_confidence_scores.index(x)] for x in box_confidence_scores if x >= bbox_confidence_threshold] # 0.4
    box_confidence_scores_high = [box_confidence_scores[box_confidence_scores.index(x)] for x in box_confidence_scores if x >= bbox_confidence_threshold]
    min_conf = 0.1
    box_detected_second = [box_detected[box_confidence_scores.index(x)] for x in box_confidence_scores if x < bbox_confidence_threshold and x > min_conf]
    box_confidence_scores_second = [box_confidence_scores[box_confidence_scores.index(x)] for x in box_confidence_scores if x < bbox_confidence_threshold and x > min_conf]
        # if len(box_detected) > maximum_number_people:
        #     lowest_confidence_idx = box_confidence_scores.index(min(box_confidence_scores))
        #     box_detected.pop(lowest_confidence_idx)
        #     box_confidence_scores.pop(lowest_confidence_idx)
    if len(box_detected) == 0:
        tracklet_pose_collection.append([])
        return tracklet_pose_collection
    path = os.path.join(source,info_imgs[4][0].split('/')[-1])  # info_imgs[4] 是一个list
    tracklet_pose_collection_tmp = {}
    tracklet_pose_collection_tmp['bbox_list'] = box_detected_high
    tracklet_pose_collection_tmp['box_confidence_scores'] = box_confidence_scores_high
    tracklet_pose_collection_tmp['img_dir'] = path
    tracklet_pose_collection_tmp['foreignmatter_bbox_list'] = []
    tracklet_pose_collection_tmp['foreignmatter_box_confidence_scores'] = []
    tracklet_pose_collection.append(tracklet_pose_collection_tmp)
    ## second thresh ##
    # if len(box_detected_second) > 0: # 只有大于0的时候才加入
    tracklet_pose_collection_second_tmp = {}
    tracklet_pose_collection_second_tmp['bbox_list'] = box_detected_second
    tracklet_pose_collection_second_tmp['box_confidence_scores'] = box_confidence_scores_second
    tracklet_pose_collection_second_tmp['img_dir'] = path
    tracklet_pose_collection_second.append(tracklet_pose_collection_second_tmp)
    

    return tracklet_pose_collection,tracklet_pose_collection_second
def conduct_pose_estimation(webcam, path, out, im0s, pred, img, dataset, save_txt, save_img, view_img, box_detected, head_box_detected, foreignmatter_box_detected, box_confidence_scores, head_box_confidence_scores, foreignmatter_box_confidence_scores, centers, scales, vid_path, vid_writer, vid_cap, tracklet_pose_collection, names, colors, pose_transform, bbox_confidence_threshold, tracklet_inner_cnt, need_face_recognition_switch, face_verification_thresh, nms_thresh):
    # pred - predicted human bounding boxes with confidences
    # path - to current image for pose estimation
    # out - output directory
    # im0s - current image array, 1080x1920x3
    # img - current image array with shape 1x3x1088x1920
    # tracklet_inner_cnt - index of frame
    # pose_model, pose_transform - model for pose estimation
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        else:
            p, s, im0 = path, '', im0s

        save_path = str(Path(out) / Path(p).name)
        txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                if (save_img or view_img) and (int(cls.data.cpu().numpy()) == 0):  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                ######################################################### pose estimation #########################################################
                if int(cls.data.cpu().numpy()) == 0:
                    # if min([float(xyxy[3].data.cpu().numpy()), float(xyxy[1].data.cpu().numpy())]) - 0.0 > abs(float(xyxy[3].data.cpu().numpy())-float(xyxy[1].data.cpu().numpy())) and im0s.shape[0] - max([float(xyxy[3].data.cpu().numpy()), float(xyxy[1].data.cpu().numpy())]) > abs(float(xyxy[3].data.cpu().numpy())-float(xyxy[1].data.cpu().numpy())) and \
                    #     min([float(xyxy[2].data.cpu().numpy()), float(xyxy[0].data.cpu().numpy())]) - 0.0 > abs(float(xyxy[2].data.cpu().numpy())-float(xyxy[0].data.cpu().numpy())) and im0s.shape[1] - max([float(xyxy[2].data.cpu().numpy()), float(xyxy[0].data.cpu().numpy())]) > abs(float(xyxy[2].data.cpu().numpy())-float(xyxy[0].data.cpu().numpy())) and \
                    #     max([abs(float(xyxy[3].data.cpu().numpy())-float(xyxy[1].data.cpu().numpy())), abs(float(xyxy[2].data.cpu().numpy())-float(xyxy[0].data.cpu().numpy()))]) < 50: # and \
                    #     continue
                    statistic_information_dict = {'aspect ratio': [0.51, 6.24], 'width': [33, 165], 'height': [34, 284]}
                    width = abs(xyxy[2] - xyxy[0])
                    height = abs(xyxy[3] - xyxy[1])
                    #### 此处统计的信息不准确 #####
                    # if width > statistic_information_dict['width'][1] or width < statistic_information_dict['width'][0] or height > statistic_information_dict['height'][1] or height < statistic_information_dict['height'][0]:
                    if width > statistic_information_dict['width'][1] or width < statistic_information_dict['width'][0]  or height < statistic_information_dict['height'][0]:
                        continue
                    box_detected.append([(float(xyxy[0].data.cpu().numpy()), float(xyxy[1].data.cpu().numpy())), (float(xyxy[2].data.cpu().numpy()), float(xyxy[3].data.cpu().numpy()))])
                    box_confidence_scores.append(float(conf.data.cpu().numpy()) + 1e-4*random.random())
                if int(cls.data.cpu().numpy()) == 1:#body
                    head_box_detected.append([(float(xyxy[0].data.cpu().numpy()), float(xyxy[1].data.cpu().numpy())), (float(xyxy[2].data.cpu().numpy()), float(xyxy[3].data.cpu().numpy()))])
                    head_box_confidence_scores.append(float(conf.data.cpu().numpy()) + 1e-4*random.random())
            box_detected = [box_detected[box_confidence_scores.index(x)] for x in box_confidence_scores if x > bbox_confidence_threshold] # 0.4
            box_confidence_scores = [box_confidence_scores[box_confidence_scores.index(x)] for x in box_confidence_scores if x > bbox_confidence_threshold]
            head_box_detected = [head_box_detected[head_box_confidence_scores.index(x)] for x in head_box_confidence_scores if x > head_bbox_confidence_threshold] # 0.55
            head_box_confidence_scores = [head_box_confidence_scores[head_box_confidence_scores.index(x)] for x in head_box_confidence_scores if x > head_bbox_confidence_threshold]
            # if len(box_detected) > maximum_number_people:
            #     lowest_confidence_idx = box_confidence_scores.index(min(box_confidence_scores))
            #     box_detected.pop(lowest_confidence_idx)
            #     box_confidence_scores.pop(lowest_confidence_idx)
            if len(box_detected) == 0:
                tracklet_pose_collection.append([])
                return tracklet_pose_collection

    ############################################ match head boxes with body boxes ###########################################################################
    if len(head_box_detected) > 0:
        head_body_association_matrix = compute_iou_between_body_and_head(head_box_detected, box_detected)
        head_body_association_matrix_backup = compute_iou_between_body_and_head(head_box_detected, box_detected)
        head_body_association_matrix = head_body_association_matrix + np.max(head_body_association_matrix)
        head_body_association_matrix = 1.0 / (head_body_association_matrix + 1e-6) # avoiding division by zero

        if head_body_association_matrix.shape[0] > head_body_association_matrix.shape[1]:
            add_width = head_body_association_matrix.shape[0] - head_body_association_matrix.shape[1]
            head_body_association_matrix = np.concatenate((head_body_association_matrix, np.ones((head_body_association_matrix.shape[0], add_width))*np.max(head_body_association_matrix)*2), axis=1)
            ot_src = [1.0] * head_body_association_matrix.shape[0]
            ot_dst = [1.0] * head_body_association_matrix.shape[1]
            head_body_transportation_array = ot.emd(ot_src, ot_dst, head_body_association_matrix)
            head_body_transportation_array = head_body_transportation_array[:, :head_body_transportation_array.shape[1] - add_width]
        elif head_body_association_matrix.shape[0] < head_body_association_matrix.shape[1]:
            add_height = head_body_association_matrix.shape[1] - head_body_association_matrix.shape[0]
            head_body_association_matrix = np.concatenate((head_body_association_matrix, np.ones((add_height, head_body_association_matrix.shape[1]))*np.max(head_body_association_matrix)*2), axis=0)
            ot_src = [1.0] * head_body_association_matrix.shape[0]
            ot_dst = [1.0] * head_body_association_matrix.shape[1]
            head_body_transportation_array = ot.emd(ot_src, ot_dst, head_body_association_matrix)
            head_body_transportation_array = head_body_transportation_array[:head_body_transportation_array.shape[0] - add_height, :]
        else:
            ot_src = [1.0] * head_body_association_matrix.shape[0]
            ot_dst = [1.0] * head_body_association_matrix.shape[1]
            head_body_transportation_array = ot.emd(ot_src, ot_dst, head_body_association_matrix)
        for idx_row_head in range(head_body_transportation_array.shape[0]):
            for idx_col_body in range(head_body_transportation_array.shape[1]):
                if head_body_transportation_array[idx_row_head, idx_col_body] == 1 and compute_iou_between_body_and_head([head_box_detected[idx_row_head]], [box_detected[idx_col_body]])[0, 0] == 0.0:
                    head_body_transportation_array[idx_row_head, idx_col_body] = 0

    # deal with highly overlapping people, if the iou between two people are higher than nms threshold, pull them away
    for idx_box_detected in range(len(box_detected)):
        for idx_another_box_detected in range(idx_box_detected + 1, len(box_detected)):
            iou_before_pulling_away = compute_iou_single_box([box_detected[idx_box_detected][0][1], box_detected[idx_box_detected][1][1], box_detected[idx_box_detected][0][0], box_detected[idx_box_detected][1][0]], \
                                                             [box_detected[idx_another_box_detected][0][1], box_detected[idx_another_box_detected][1][1], box_detected[idx_another_box_detected][0][0], box_detected[idx_another_box_detected][1][0]])
            if iou_before_pulling_away > nms_thresh: # keep center unchanged to make spatial consistency stable
                pull_away_scale = (1.0 - np.sqrt(nms_thresh / iou_before_pulling_away)) / 2.0
                common_width = sorted([box_detected[idx_box_detected][0][0], box_detected[idx_box_detected][1][0], box_detected[idx_another_box_detected][0][0], box_detected[idx_another_box_detected][1][0]])[-2] - \
                               sorted([box_detected[idx_box_detected][0][0], box_detected[idx_box_detected][1][0], box_detected[idx_another_box_detected][0][0], box_detected[idx_another_box_detected][1][0]])[1]
                common_height = sorted([box_detected[idx_box_detected][0][1], box_detected[idx_box_detected][1][1], box_detected[idx_another_box_detected][0][1], box_detected[idx_another_box_detected][1][1]])[-2] - \
                                sorted([box_detected[idx_box_detected][0][1], box_detected[idx_box_detected][1][1], box_detected[idx_another_box_detected][0][1], box_detected[idx_another_box_detected][1][1]])[1]
                # if idx_box_detected lies to the left of idx_another_box_detected
                if box_detected[idx_box_detected][0][0] + box_detected[idx_box_detected][1][0] < box_detected[idx_another_box_detected][0][0] + box_detected[idx_another_box_detected][1][0]:
                    box_detected[idx_box_detected] = [(min([max([box_detected[idx_box_detected][0][0], 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][0][1], 0]), im0.shape[0]])), \
                                                      (min([max([box_detected[idx_box_detected][1][0] - pull_away_scale * common_width, 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][1][1], 0]), im0.shape[0]]))]
                    box_detected[idx_another_box_detected] = [(min([max([box_detected[idx_another_box_detected][0][0] + pull_away_scale * common_width, 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][0][1], 0]), im0.shape[0]])), \
                                                              (min([max([box_detected[idx_another_box_detected][1][0], 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][1][1], 0]), im0.shape[0]]))]
                else:
                    box_detected[idx_box_detected] = [(min([max([box_detected[idx_box_detected][0][0] + pull_away_scale * common_width, 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][0][1], 0]), im0.shape[0]])), \
                                                      (min([max([box_detected[idx_box_detected][1][0], 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][1][1], 0]), im0.shape[0]]))]
                    box_detected[idx_another_box_detected] = [(min([max([box_detected[idx_another_box_detected][0][0], 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][0][1], 0]), im0.shape[0]])), \
                                                              (min([max([box_detected[idx_another_box_detected][1][0] - pull_away_scale * common_width, 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][1][1], 0]), im0.shape[0]]))]
                # if idx_box_detected lies to the top of idx_another_box_detected
                if box_detected[idx_box_detected][0][1] + box_detected[idx_box_detected][1][1] < box_detected[idx_another_box_detected][0][1] + box_detected[idx_another_box_detected][1][1]:
                    box_detected[idx_box_detected] = [(min([max([box_detected[idx_box_detected][0][0], 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][0][1], 0]), im0.shape[0]])), \
                                                      (min([max([box_detected[idx_box_detected][1][0], 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][1][1] - pull_away_scale * common_height, 0]), im0.shape[0]]))]
                    box_detected[idx_another_box_detected] = [(min([max([box_detected[idx_another_box_detected][0][0], 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][0][1] + pull_away_scale * common_height, 0]), im0.shape[0]])), \
                                                              (min([max([box_detected[idx_another_box_detected][1][0], 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][1][1], 0]), im0.shape[0]]))]
                else:
                    box_detected[idx_box_detected] = [(min([max([box_detected[idx_box_detected][0][0], 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][0][1] + pull_away_scale * common_height, 0]), im0.shape[0]])), \
                                                      (min([max([box_detected[idx_box_detected][1][0], 0]), im0.shape[1]]), \
                                                       min([max([box_detected[idx_box_detected][1][1], 0]), im0.shape[0]]))]
                    box_detected[idx_another_box_detected] = [(min([max([box_detected[idx_another_box_detected][0][0], 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][0][1], 0]), im0.shape[0]])), \
                                                              (min([max([box_detected[idx_another_box_detected][1][0], 0]), im0.shape[1]]), \
                                                               min([max([box_detected[idx_another_box_detected][1][1] - pull_away_scale * common_height, 0]), im0.shape[0]]))]

    tracklet_pose_collection_tmp = {}
    tracklet_pose_collection_tmp['bbox_list'] = box_detected
    # tracklet_pose_collection_tmp['head_bbox_list'] = head_box_detected
    tracklet_pose_collection_tmp['box_confidence_scores'] = box_confidence_scores
    tracklet_pose_collection_tmp['img_dir'] = path
    tracklet_pose_collection_tmp['foreignmatter_bbox_list'] = []
    tracklet_pose_collection_tmp['foreignmatter_box_confidence_scores'] = []
    tracklet_pose_collection.append(tracklet_pose_collection_tmp)

    return tracklet_pose_collection

# def compute_inter_person_similarity_init_pool(input_person_to_person_matching_matrix_normalized_collection):
#     global person_to_person_matching_matrix_normalized_collection
#     person_to_person_matching_matrix_normalized_collection = input_person_to_person_matching_matrix_normalized_collection

# comput the matching error between multiple people in current two frames
# input:
# tracklet_inner_idx: frame index of former frame for matching, int
# tracklet_inner_base_idx: the starting frame idx of current batch of frames, int
# node_id_cnt: node id of current bounding box, each bounding boxes in current batch of frames is regarded as a node, int
# idx_stride_between_frame_pair: stride between two frames for matching, int
# maximum_possible_number: math.exp(10), float
# max_row_num_of_person_to_person_matching_matrix_normalized: initialized to be 2, int
# max_col_num_of_person_to_person_matching_matrix_normalized: initialized to be 2, int
# num_of_person_to_person_matching_matrix_normalized_copies: number of frame pairs for matching, int
# node_id_cnt_list: list of integers, storing the starting node index of each pair of frames, will be used later
# all_people_features: the reid features of all people, 2d array, shape: number of people in current batch of frames x 512, float
# output:
# person_to_person_matching_matrix_copy_normalized: matching matrix between current pair of frames, #rows = #people in former frame #cols = #people in latter frame, float
# idx_stride_between_frame_pair: stride between current batch of frames, int
# node_id_cnt: int, same as input
# def compute_inter_person_similarity_worker(input_list, whether_use_iou_similarity_or_not):
#     # tracklet_inner_idx: 帧索引 tracklet_inner_base_idx:当前batch开始的帧    node_id_cnt
#
#     tracklet_inner_idx, tracklet_inner_base_idx, node_id_cnt, tracklet_pose_collection, idx_stride_between_frame_pair, maximum_possible_number, max_row_num_of_person_to_person_matching_matrix_normalized, \
#         max_col_num_of_person_to_person_matching_matrix_normalized, num_of_person_to_person_matching_matrix_normalized_copies, node_id_cnt_list, all_people_features = \
#         input_list[0], input_list[1], input_list[2], input_list[3], input_list[4], input_list[5], input_list[6], input_list[7], input_list[8], input_list[9], input_list[10]
#
#     # collect all bounding boxes in frame pairs
#     curr_frame_dict = tracklet_pose_collection[tracklet_inner_idx]  # 当前帧tracklet信息
#     next_frame_dict = tracklet_pose_collection[tracklet_inner_idx + idx_stride_between_frame_pair] # 下一个桢tracklet信息
#
#     # matrix storing the iou similarity between each box from previous frame and each box from next frame, each row corresponds to one box in prev, each col - one box in next
#     person_to_person_matching_matrix = np.ones((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list']))) * maximum_possible_number #  乘以max_number的含义?
#     # matrix storing the appearance similarity between ...
#     person_to_person_matching_matrix_iou = np.zeros((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list'])))
#     # ????
#     person_to_person_depth_matching_matrix_iou = np.ones((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list']))) * 0.5
#
#     # evaluate_time_start = time.time()
#     for curr_person_bbox_coord in curr_frame_dict['bbox_list']:
#         for next_person_bbox_coord in next_frame_dict['bbox_list']:
#             # to find the index of each bounding box in all people in current batch of frames
#             # if curr_frame_dict['box_confidence_scores'][curr_frame_dict['bbox_list'].index(curr_person_bbox_coord)] > 1.0 or next_frame_dict['box_confidence_scores'][next_frame_dict['bbox_list'].index(next_person_bbox_coord)] > 1.0:
#             #     person_to_person_matching_matrix[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = 1.0
#             # else:
#             vector1 = all_people_features.data.numpy()[int(np.sum([len(x['bbox_list']) for x in tracklet_pose_collection[0:tracklet_inner_idx]]) + curr_frame_dict['bbox_list'].index(curr_person_bbox_coord))] # 当前帧bbox的特征向量
#             vector2 = all_people_features.data.numpy()[int(np.sum([len(x['bbox_list']) for x in tracklet_pose_collection[0:(tracklet_inner_idx + idx_stride_between_frame_pair)]]) + next_frame_dict['bbox_list'].index(next_person_bbox_coord))] # 下一帧bbox的特征向量
#             person_to_person_matching_matrix[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
#                 1.0 - min([np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)), 1.0])
#
#             person_to_person_matching_matrix_iou[
#                 curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
#                 min([max([compute_iou_single_box([curr_person_bbox_coord[0][1], curr_person_bbox_coord[1][1], curr_person_bbox_coord[0][0], curr_person_bbox_coord[1][0]], \
#                     [next_person_bbox_coord[0][1], next_person_bbox_coord[1][1], next_person_bbox_coord[0][0], next_person_bbox_coord[1][0]]), 0.0]), 1.0])
#
#             person_to_person_depth_matching_matrix_iou[
#                 curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
#                 1.0 / max([abs(curr_person_bbox_coord[1][1] - next_person_bbox_coord[1][1]), \
#                            person_to_person_depth_matching_matrix_iou[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)]])
#
#     evaluate_time_end = time.time()
#     # corner case: only one person
#     if person_to_person_matching_matrix.shape[0] == 1 and person_to_person_matching_matrix.shape[1] == 1 and person_to_person_matching_matrix[0][0] == 0.0:
#         person_to_person_matching_matrix[0][0] = 1.0
#     else:
#         # replace zero entries in the matrix "person_to_person_matching_matrix" with half minimum value to facilitate division
#         person_to_person_matching_matrix[np.where(person_to_person_matching_matrix==0)] = np.min(person_to_person_matching_matrix[np.where(person_to_person_matching_matrix>0)]) / 2.0
#         # similarity is inversely proportional to matching error
#         person_to_person_matching_matrix = 1.0 / person_to_person_matching_matrix / np.max(
#             1.0 / person_to_person_matching_matrix)  # similarity
#     # similarity is the summation of appearance and iou similarity
#     if whether_use_iou_similarity_or_not:
#         person_to_person_matching_matrix = person_to_person_matching_matrix * person_to_person_matching_matrix_iou # * person_to_person_depth_matching_matrix_iou
#     person_to_person_matching_matrix_copy = copy.deepcopy(person_to_person_matching_matrix)
#     denominator = person_to_person_matching_matrix_copy.max(axis=1).reshape(person_to_person_matching_matrix_copy.max(axis=1).shape[0], 1)
#     denominator[np.where(denominator==0)] = 1.0
#     person_to_person_matching_matrix_copy_normalized = person_to_person_matching_matrix_copy / denominator
#     for idx_col in range(person_to_person_matching_matrix_copy_normalized.shape[1]):
#         if len(np.where(person_to_person_matching_matrix_copy_normalized[:, idx_col]==1.0)[0].tolist()) > 1:
#             list_idx_compete = np.where(person_to_person_matching_matrix_copy_normalized[:, idx_col] == 1.0)[0].tolist()
#             list_idx_compete_ori = np.argsort(person_to_person_matching_matrix_copy[:, idx_col][list_idx_compete]).tolist()
#             list_idx_compete_ordered = np.array(list_idx_compete)[list_idx_compete_ori].tolist()
#             for list_idx_compete_ordered_ele in list_idx_compete_ordered:
#                 person_to_person_matching_matrix_copy_normalized[list_idx_compete_ordered_ele, :] *= 0.99**(len(list_idx_compete_ordered)-1-list_idx_compete_ordered.index(list_idx_compete_ordered_ele))
#     person_to_person_matching_matrix_copy_normalized *= 200
#
#     return person_to_person_matching_matrix_copy_normalized, idx_stride_between_frame_pair, node_id_cnt

def compute_inter_person_similarity_worker(input_list, whether_use_iou_similarity_or_not,whether_use_reid_similarity_or_not):
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
    # ????
    person_to_person_depth_matching_matrix_iou = np.ones((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list']))) * 0.5

    # evaluate_time_start = time.time()
    for curr_person_bbox_coord in curr_frame_dict['bbox_list']:
        for next_person_bbox_coord in next_frame_dict['bbox_list']:
            # to find the index of each bounding box in all people in current batch of frames
            # if curr_frame_dict['box_confidence_scores'][curr_frame_dict['bbox_list'].index(curr_person_bbox_coord)] > 1.0 or next_frame_dict['box_confidence_scores'][next_frame_dict['bbox_list'].index(next_person_bbox_coord)] > 1.0:
            #     person_to_person_matching_matrix[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = 1.0
            # else:
            vector1 = all_people_features.data.numpy()[int(np.sum([len(x['bbox_list']) for x in tracklet_pose_collection[0:tracklet_inner_idx]]) + curr_frame_dict['bbox_list'].index(curr_person_bbox_coord))] # 当前帧bbox的特征向量
            vector2 = all_people_features.data.numpy()[int(np.sum([len(x['bbox_list']) for x in tracklet_pose_collection[0:(tracklet_inner_idx + idx_stride_between_frame_pair)]]) + next_frame_dict['bbox_list'].index(next_person_bbox_coord))] # 下一帧bbox的特征向量
            person_to_person_matching_matrix[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                1.0 - min([np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)), 1.0])

            person_to_person_matching_matrix_iou[
                curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                min([max([compute_iou_single_box([curr_person_bbox_coord[0][1], curr_person_bbox_coord[1][1], curr_person_bbox_coord[0][0], curr_person_bbox_coord[1][0]], \
                    [next_person_bbox_coord[0][1], next_person_bbox_coord[1][1], next_person_bbox_coord[0][0], next_person_bbox_coord[1][0]]), 0.0]), 1.0])

            person_to_person_depth_matching_matrix_iou[
                curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                1.0 / max([abs(curr_person_bbox_coord[1][1] - next_person_bbox_coord[1][1]), \
                           person_to_person_depth_matching_matrix_iou[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)]])

    evaluate_time_end = time.time()
    # corner case: only one person
    if person_to_person_matching_matrix.shape[0] == 1 and person_to_person_matching_matrix.shape[1] == 1 and person_to_person_matching_matrix[0][0] == 0.0:
        person_to_person_matching_matrix[0][0] = 1.0
    else:
        # replace zero entries in the matrix "person_to_person_matching_matrix" with half minimum value to facilitate division
        person_to_person_matching_matrix[np.where(person_to_person_matching_matrix==0)] = np.min(person_to_person_matching_matrix[np.where(person_to_person_matching_matrix>0)]) / 2.0
        # similarity is inversely proportional to matching error
        person_to_person_matching_matrix = 1.0 / person_to_person_matching_matrix / np.max(
            1.0 / person_to_person_matching_matrix)  # similarity
    # similarity is the summation of appearance and iou similarity
    if whether_use_iou_similarity_or_not and whether_use_reid_similarity_or_not:
        person_to_person_matching_matrix = person_to_person_matching_matrix * person_to_person_matching_matrix_iou # * person_to_person_depth_matching_matrix_iou
    elif whether_use_iou_similarity_or_not and not whether_use_reid_similarity_or_not:# 只使用iou信息
        person_to_person_matching_matrix = person_to_person_matching_matrix_iou
    # 默认情况下为只是用reid信息
    person_to_person_matching_matrix_copy = copy.deepcopy(person_to_person_matching_matrix)
    denominator = person_to_person_matching_matrix_copy.max(axis=1).reshape(person_to_person_matching_matrix_copy.max(axis=1).shape[0], 1)
    denominator[np.where(denominator==0)] = 1.0
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

def batch_track_regression(track1,frame_start,frame_end,frame_span1):
    '''
    track1:previous_video_segment_predicted_tracks_bboxes_test
    start,end:regression start and end
    frame_span:original frame span (true detections)
    '''
    frame_bbox1 = {int(track1[node][0].split('.')[0]): track1[node][1] for node in track1}  # key:frame(int) value:bbox
    if set(range(frame_start,frame_end+1)) < set(frame_bbox1) : # 此时不需要进行回归,start到end内的元素均已有，此时不需要回归
        frame_bbox1 = dict(sorted(frame_bbox1.items(), key=operator.itemgetter(0))) # in order
        return frame_bbox1
    width1 = np.mean([frame_bbox1[frame][1][0] - frame_bbox1[frame][0][0] for frame in frame_bbox1])
    height1 = np.mean([frame_bbox1[frame][1][1] - frame_bbox1[frame][0][1] for frame in frame_bbox1])
    horicenter_coordinates1 = (np.array(
        [frame_bbox1[frame][1][0] + frame_bbox1[frame][0][0] for frame in frame_bbox1]) / 2.0).tolist()  # 水平
    vertcenter_coordinates1 = (np.array(
        [frame_bbox1[frame][1][1] + frame_bbox1[frame][0][1] for frame in frame_bbox1]) / 2.0).tolist()  # 垂直
    horicenter_fitter_coefficients = np.polyfit(frame_span1, horicenter_coordinates1, 1)
    vertcenter_fitter_coefficients = np.polyfit(frame_span1, vertcenter_coordinates1, 1)
    horicenter_fitter = np.poly1d(horicenter_fitter_coefficients)  # np.poly1d根据数组生成一个多项式
    vertcenter_fitter = np.poly1d(vertcenter_fitter_coefficients)
    for frame in range(frame_start, frame_end + 1):
        if frame in frame_bbox1:  # 已经存在
            continue
        else:  #
            frame_bbox1[frame] = [
                (horicenter_fitter(float(frame)) - width1 / 2.0, vertcenter_fitter(float(frame)) - height1 / 2.0),
                (horicenter_fitter(float(frame)) + width1 / 2.0, vertcenter_fitter(float(frame)) + height1 / 2.0)]
    # frame_bbox1 = dict(sorted(frame_bbox1.items(),key=lambda d:d[0]))
    # frame_bbox1 = sorted(frame_bbox1)
    frame_bbox1 = dict(sorted(frame_bbox1.items(), key=operator.itemgetter(0)))  # 按照key值升序,从而使得计算iou的时候帧数是对应的
    return frame_bbox1

def stitching_tracklets_revised(kmedoids_instance,current_video_segment_predicted_tracks_bboxes_test,current_trajectory_similarity_dict,previous_video_segment_predicted_tracks_bboxes_test,previous_trajectory_similarity_dict,node_matching_dict,mapping_node_id_to_features,mapping_node_id_to_bbox):
    prev_last_frame_node_list = list(node_matching_dict.values())
    curr_first_frame_node_list = list(node_matching_dict.keys())
    result_dict = {}
    previous_unmatched_tracks = []
    curr_unmatched_tracks = []
    for idx_prev,previous_tracklet_id in enumerate(previous_video_segment_predicted_tracks_bboxes_test):
        trajectory_from_prev = previous_video_segment_predicted_tracks_bboxes_test[previous_tracklet_id] # 前一个batch的一条轨迹
        prev_trajectory_nodes = list(trajectory_from_prev.keys()) # 该帧的node
        # if max(prev_trajectory_nodes) in prev_last_frame_node_list: # 表示该条轨迹包含前一个batch最后一帧
        prev_trajectory_similarity_list =  previous_trajectory_similarity_dict[previous_tracklet_id]
        for idx_curr,current_tracklet_id in enumerate(current_video_segment_predicted_tracks_bboxes_test): # 遍历当前batch的轨迹
            trajectory_from_curr = current_video_segment_predicted_tracks_bboxes_test[current_tracklet_id] # 当前batch的一条轨迹
            curr_trajectory_nodes = list(trajectory_from_curr.keys())
            curr_trajectory_similarity_list = current_trajectory_similarity_dict[current_tracklet_id]
            if current_tracklet_id in result_dict: # 如果已经匹配好则跳过
                continue
            if min(curr_trajectory_nodes) in curr_first_frame_node_list and node_matching_dict[min(curr_trajectory_nodes)] == max(prev_trajectory_nodes):
            # if max(prev_trajectory_nodes) in prev_last_frame_node_list and min(curr_trajectory_nodes) in curr_first_frame_node_list:
                result_dict[current_tracklet_id] = previous_tracklet_id
                reid_similarity = cosine_similarity(np.array(trajectory_from_curr[min(curr_trajectory_nodes)][3]),np.array(trajectory_from_prev[max(prev_trajectory_nodes)][3]))
                reid_similarity1 = cosine_similarity(np.array(trajectory_from_curr[curr_trajectory_nodes[-2]][3]),np.array(trajectory_from_prev[max(prev_trajectory_nodes)][3]))
                continue
            # else: # 采用reid信息进行匹配,当前batch第一帧与previous batch最后一帧
            #     reid_similarity = cosine_similarity(np.array(trajectory_from_curr[min(curr_trajectory_nodes)][3]),np.array(trajectory_from_prev[max(prev_trajectory_nodes)][3]))
            #     thresh = np.mean(prev_trajectory_similarity_list) - 3*np.std(prev_trajectory_similarity_list) #prev_trajectory_similarity_list只有一个元素
            #     if reid_similarity >= thresh:
            #         result_dict[current_tracklet_id] = previous_tracklet_id
            #         continue
            #     elif idx_curr == len(current_video_segment_predicted_tracks_bboxes_test) - 1:
            #         previous_unmatched_tracks.append(previous_tracklet_id)

    curr_unmatched_tracks = list(set(current_video_segment_predicted_tracks_bboxes_test.keys())-set(result_dict.keys()))
    previous_unmatched_tracks = list(set(previous_video_segment_predicted_tracks_bboxes_test.keys())-set(result_dict.values()))
    # node_list = curr_first_frame_node_list - 
    curr_track_first_node = [list(current_video_segment_predicted_tracks_bboxes_test[x].keys())[0] for x in curr_unmatched_tracks]
    # mapping2node_id_bbox_part2 = {list(current_video_segment_predicted_tracks_bboxes_test[x].keys())[0]:current_video_segment_predicted_tracks_bboxes_test[x].get(list(current_video_segment_predicted_tracks_bboxes_test[x].keys())[0]) for x in current_video_segment_predicted_tracks_bboxes_test}

    data = cluster_data(curr_track_first_node, mapping_node_id_to_features, mapping_node_id_to_bbox)
    labels = kmedoids_instance.predict(data)
    # result_dict: key:当前轨迹id，value： 前一个轨迹id
    # previous_unmatched_tracks: list,记录未匹配的轨迹
    # curr_unmatched_tracks:list,记录当前batch未匹配的的轨迹，表示新出现的轨迹
    return result_dict, previous_unmatched_tracks, curr_unmatched_tracks

def stitching_tracklets_revised_bidirectional(previous_bboxes_test,current_bboxes_test):
    remove_list = []
    dists = np.zeros((len(previous_bboxes_test), len(current_bboxes_test)))
    ###### 对于小于2的如何计算相似度 ######
    '''
    1 在上一个阶段删除小于2的tracklet
    2 小于2的tracklet采用和下一个track第1帧来计算相似度
    '''
    previous_tracks_id = list(previous_bboxes_test.keys())
    current_tracks_id = list(current_bboxes_test.keys())
    for id1 in previous_bboxes_test:
        for id2 in current_bboxes_test:
            track1 = previous_bboxes_test[id1]
            frame_span1 = [int(track1[node][0].split('.')[0]) for node in track1] # 自变量
            track2 = current_bboxes_test[id2]
            frame_span2 = [int(track2[node][0].split('.')[0]) for node in track2]
            if len(previous_bboxes_test[id1]) < 3 or len(current_bboxes_test[id2]) < 3:  # 小于2的时候无法进行回归
                '''
                无法回归的时候采用iou相似度
                '''
                bbox1 = track1[max(list(track1.keys()))][1]
                bbox2 = track2[min(list(track2.keys()))][1]
                # [former_bbox[0][1], former_bbox[1][1], former_bbox[0][0], former_bbox[1][0]]
                dists[previous_tracks_id.index(id1), current_tracks_id.index(id2)] = 1 - compute_iou_single_box([bbox1[0][1],bbox1[1][1],bbox1[0][0],bbox1[1][0]],[bbox2[0][1],bbox2[1][1],bbox2[0][0],bbox2[1][0]]) #
            # tmp_span = frame_span1 + frame_span2
            # frame_start,frame_end = min(frame_span1+frame_span2),max(frame_span1+frame_span2)
            frame_start,frame_end = max(frame_span1),min(frame_span2)
            frame_span = list(range(frame_start, frame_end + 1)) #  两个batch的frame范围
            frame_bbox1 = batch_track_regression(track1,frame_start,frame_end,frame_span1)
            frame_bbox2 = batch_track_regression(track2, frame_start, frame_end,frame_span2)
            ### regression ##
            ### 从最小的frame到最大的frame计算相似度 ###
            #### 注意两个轨迹段比较的帧一定要对应 ####
            tracklet1 = np.array([frame_bbox1[frame] for frame in frame_span])
            tracklet2 = np.array([frame_bbox2[frame] for frame in frame_span])
            tracklet_ious_matrix  = compute_iou_between_bbox_list(tracklet1.reshape(-1, 2, 2), tracklet2.reshape(-1, 2, 2))
            # tracklet_overlap_matrix = compute_overlap_between_bbox_list(tracklet1.reshape(-1, 2, 2), tracklet2.reshape(-1, 2, 2))
            ious = np.diagonal(tracklet_ious_matrix)
            # print('track{0} and track{1} iou is {2}'.format(id1,id2,np.mean(ious)))
            dists[previous_tracks_id.index(id1), current_tracks_id.index(id2)] = 1 - np.mean(ious[frame_span.index(max(frame_span1)):frame_span.index(min(frame_span2))+1])  # 前一个轨迹最后一帧与当前轨迹最前面一帧
    matched_indices, previous_unmatched_ids, curr_unmatched_ids = linear_assignment(dists, thresh= 0.3) # ???
    result_dict = {}
    previous_unmatched_tracks = []
    curr_unmatched_tracks = []
    for m in matched_indices:
        # 对matched_indices进行判断
        result_dict[current_tracks_id[m[1]]] = previous_tracks_id[m[0]]
    # print(np.mean(mean_similarity_list))
    for track_id in previous_unmatched_ids:
        previous_unmatched_tracks.append(previous_tracks_id[track_id])

    for track_id in curr_unmatched_ids:
        curr_unmatched_tracks.append(current_tracks_id[track_id])
    return  result_dict,previous_unmatched_tracks,curr_unmatched_tracks
def byte_stitching_tracklets(source,mapping_node_id_to_bbox,node_matching_dict,tracklet_inner_cnt, current_video_segment_predicted_tracks, previous_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, previous_video_segment_predicted_tracks_bboxes, current_video_segment_representative_frames, previous_video_segment_representative_frames,current_video_segment_predicted_tracks_bboxes_test,previous_video_segment_predicted_tracks_bboxes_test,unmatched_tracks_memory_dict):
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
    unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
    frames_height = frame_height
    frames_width = frame_width
    # (0,boder_x_min) (boder_x_max,frames_width)
    gap = 60  # gap设置太大会把最后一段轨迹给分开来
    border_x_min,border_x_max =  gap ,frames_width-gap
    border_y_min,boder_y_max = gap,frames_height-gap
    tracklets_similarity_matrix = np.zeros((len(previous_video_segment_predicted_tracks), len(current_video_segment_predicted_tracks)))
    predicted_bbox_based_on_historical_traj = {}
    whether_use_consistency_in_traj = True
    # max([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]):上一个batch最大的frameid
    # [max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]表示所有轨迹最后一帧的frameid
    if max([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]) - \
        min([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]) > large_temporal_stride_thresh: #100
        whether_use_consistency_in_traj = False
    tracks_all = np.array([previous_video_segment_predicted_tracks_bboxes[i][list(previous_video_segment_predicted_tracks_bboxes[i].keys())[-1]] for i in previous_video_segment_predicted_tracks_bboxes])#.reshape(-1,4)
    detections_all = np.array([current_video_segment_predicted_tracks_bboxes[i][list(current_video_segment_predicted_tracks_bboxes[i].keys())[0]] for i in current_video_segment_predicted_tracks_bboxes])#.reshape(-1,4)
    confidence =  np.array([current_video_segment_predicted_tracks_bboxes_test[i][list(current_video_segment_predicted_tracks_bboxes_test[i].keys())[0]][2] for i in current_video_segment_predicted_tracks_bboxes_test])
    inds_first = confidence > 0.6
    inds_low = confidence > 0.1
    inds_high = confidence <  0.6 # args.track_thresh
    inds_second = np.logical_and(inds_low,inds_high)
    dets_second = detections_all[inds_second]
    dets_first = detections_all[inds_first]
    # 结果参数初始化
    result_dict = {}
    previous_tracks_id = np.array(list(previous_video_segment_predicted_tracks_bboxes.keys()))
    current_tracks_id = np.array(list(current_video_segment_predicted_tracks_bboxes.keys()))
    previous_unmatched_tracks = []  # 前一个batch当中未匹配的
    curr_unmatched_tracks = []  # 当前batch未匹配的tracks的key
    # 第一次关联
    dists = 1 - compute_iou_between_bbox_list(tracks_all,dets_first)
    matches,u_track,u_detection = matching.linear_assignment(dists,thresh = 0.7)
    for itracked,idet in matches: # 40,39
        track = tracks_all[itracked]
        det = dets_first[idet]
        result_dict[current_tracks_id[inds_first][idet]] = previous_tracks_id[itracked]
    # second association
    r_tracked_stracks = np.array([tracks_all[i] for i in u_track])
    if len(dets_second) > 0:
        dists_second = 1 - compute_iou_between_bbox_list(r_tracked_stracks,dets_second)
        matches_second,u_track_second,u_detection_second = matching.linear_assignment(dists_second,thresh=0.5)
        for itracked, idet in matches_second:  # 40,39
            det = dets_second[idet]
            track = r_tracked_stracks[itracked]
            result_dict[current_tracks_id[inds_second][idet]] = previous_tracks_id[u_track][itracked]

    for track_id in previous_video_segment_predicted_tracks_bboxes:
        if track_id not in list(result_dict.values()):
            previous_unmatched_tracks.append(track_id)

    for track_id in current_video_segment_predicted_tracks_bboxes:
        if track_id not in list(result_dict.keys()):
            curr_unmatched_tracks.append(track_id)
    print('previous_unmatched(global) track {0},curr_unmatched_track(local) track {1} '.format(previous_unmatched_tracks,curr_unmatched_tracks))
    return  result_dict,previous_unmatched_tracks,curr_unmatched_tracks

def stitching_tracklets(source,mapping_node_id_to_bbox,node_matching_dict,tracklet_inner_cnt, current_video_segment_predicted_tracks, previous_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, previous_video_segment_predicted_tracks_bboxes, current_video_segment_representative_frames, previous_video_segment_representative_frames,current_video_segment_predicted_tracks_bboxes_test,previous_video_segment_predicted_tracks_bboxes_test,unmatched_tracks_memory_dict):
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
    # frames_height = previous_video_segment_representative_frames[[x for x in previous_video_segment_representative_frames][0]][0].shape[0]
    # frames_width = previous_video_segment_representative_frames[[x for x in previous_video_segment_representative_frames][0]][0].shape[1]

    # frames_height = previous_video_segment_representative_frames[[x for x in previous_video_segment_representative_frames][0]][1][0] # 表示第1个元素索引???
    # frames_width = previous_video_segment_representative_frames[[x for x in previous_video_segment_representative_frames][0]][1][1]
    frames_height = frame_height
    frames_width = frame_width
    # (0,boder_x_min) (boder_x_max,frames_width)
    gap = 60  # gap设置太大会把最后一段轨迹给分开来
    border_x_min,border_x_max =  gap ,frames_width-gap
    border_y_min,boder_y_max = gap,frames_height-gap
    tracklets_similarity_matrix = np.zeros((len(previous_video_segment_predicted_tracks), len(current_video_segment_predicted_tracks)))
    predicted_bbox_based_on_historical_traj = {}
    predicted_bbox_based_on_historical_traj2 = {}
    ## delete the people who disappear from sides of images
    whether_use_consistency_in_traj = True
    # max([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]):上一个batch最大的frameid
    # [max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]表示所有轨迹最后一帧的frameid
    if max([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]) - \
        min([max(previous_video_segment_predicted_tracks[x].keys()) for x in previous_video_segment_predicted_tracks.keys()]) > large_temporal_stride_thresh: #100
        whether_use_consistency_in_traj = False
    previous_video_feature = {}
    current_video_feature = {}
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
    for previous_tracklet_id in previous_video_segment_predicted_tracks_bboxes:
        # ### 终止该条轨迹，不进行reid 匹配 ###
        # if previous_tracklet_id in terminate_track_list:
        #     tracklets_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id),:] = 1
        #     continue
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
        if len(independent_variable) < 2:
            tracklets_similarity_matrix[[x for x in previous_video_segment_predicted_tracks].index(previous_tracklet_id), :] = 1
            continue
        horicenter_fitter_coefficients = np.polyfit(independent_variable, existing_horicenter_coordinates, 1)
        vertcenter_fitter_coefficients = np.polyfit(independent_variable, existing_vertcenter_coordinates, 1)
        horicenter_fitter = np.poly1d(horicenter_fitter_coefficients) # np.poly1d根据数组生成一个多项式
        vertcenter_fitter = np.poly1d(vertcenter_fitter_coefficients)
        # # ## 2d fitter ##
        # horicenter_fitter_coefficients2 = np.polyfit(independent_variable, existing_horicenter_coordinates, 3)
        # vertcenter_fitter_coefficients2 = np.polyfit(independent_variable, existing_vertcenter_coordinates, 3)
        # horicenter_fitter2 = np.poly1d(horicenter_fitter_coefficients2) # np.poly1d根据数组生成一个多项式
        # vertcenter_fitter2 = np.poly1d(vertcenter_fitter_coefficients2)
        # left_fitter_coefficients = np.polyfit(independent_variable, existing_left_coordinates, 1)
        # top_fitter_coefficients = np.polyfit(independent_variable, existing_top_coordinates, 1)
        # right_fitter_coefficients = np.polyfit(independent_variable, existing_right_coordinates, 1)
        # bottom_fitter_coefficients = np.polyfit(independent_variable, existing_bottom_coordinates, 1)
        # left_fitter = np.poly1d(left_fitter_coefficients)
        # top_fitter = np.poly1d(top_fitter_coefficients)
        # right_fitter = np.poly1d(right_fitter_coefficients)
        # bottom_fitter = np.poly1d(bottom_fitter_coefficients)
        predicted_bbox_based_on_historical_traj[previous_tracklet_id] = {}
        predicted_bbox_based_on_historical_traj2[previous_tracklet_id] = {}
        # polyfit_1d_error[previous_tracklet_id] = np.polyfit(independent_variable, existing_horicenter_coordinates, 1,full=True)[1][0] + np.polyfit(independent_variable, existing_vertcenter_coordinates, 1,full=True)[1][0]
        # polyfit_3d_error[previous_tracklet_id] = np.polyfit(independent_variable, existing_horicenter_coordinates, 3,full=True)[1][0] + np.polyfit(independent_variable, existing_vertcenter_coordinates, 3,full=True)[1][0]
        previous_track_reid = previous_video_feature[previous_tracklet_id]
        for current_tracklet_id in current_video_segment_predicted_tracks_bboxes: # 遍历当前batch的轨迹
            trajectory_from_curr = current_video_segment_predicted_tracks_bboxes[current_tracklet_id] # 当前batch的一条轨迹
            # if len([(x in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id] and current_video_segment_predicted_tracks_bboxes[current_tracklet_id][x] == previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][x]) for x in current_video_segment_predicted_tracks_bboxes[current_tracklet_id]]) > 0 and \
            #     (False not in [(x in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id] and current_video_segment_predicted_tracks_bboxes[current_tracklet_id][x] == previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id][x]) for x in current_video_segment_predicted_tracks_bboxes[current_tracklet_id]]):
            #     current_video_segment_predicted_tracks_bboxes[current_tracklet_id] = previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id]
            #     current_video_segment_predicted_tracks[current_tracklet_id] = previous_video_segment_predicted_tracks[previous_tracklet_id]
            #     current_video_segment_all_traj_all_object_features[current_tracklet_id] = previous_video_segment_all_traj_all_object_features[previous_tracklet_id]
            sum_error_trajectories = 0.0 # 前一个batch中轨迹和当前batch轨迹的误差
            sum_error_trajectories_num_value_cnt = 0.0 #
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
                if (trajectory_from_curr_key in [x for x in trajectory_from_curr if (x not in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id])]) and (trajectory_from_curr_key not in predicted_bbox_based_on_historical_traj[previous_tracklet_id]):
                    # trajectory_from_curr_key in [x for x in trajectory_from_curr if (x not in previous_video_segment_predicted_tracks_bboxes[previous_tracklet_id])]) 在当前batch frames但是不在前一个batch frames
                    # 注意存放次序为y1y2x1x2
                    predicted_bbox_based_on_historical_traj[previous_tracklet_id][trajectory_from_curr_key] = \
                        [max([vertcenter_fitter(float(trajectory_from_curr_key[:-4])) - existing_largest_height / 2.0, 0]), min([vertcenter_fitter(float(trajectory_from_curr_key[:-4])) + existing_largest_height / 2.0, frames_height]), \
                         max([horicenter_fitter(float(trajectory_from_curr_key[:-4])) - existing_largest_width / 2.0, 0]), min([horicenter_fitter(float(trajectory_from_curr_key[:-4])) + existing_largest_width / 2.0, frames_width])]
                    # predicted_bbox_based_on_historical_traj2[previous_tracklet_id][trajectory_from_curr_key] = \
                    #     [max([vertcenter_fitter2(float(trajectory_from_curr_key[:-4])) - existing_largest_height / 2.0, 0]), min([vertcenter_fitter2(float(trajectory_from_curr_key[:-4])) + existing_largest_height / 2.0, frames_height]), \
                    #      max([horicenter_fitter2(float(trajectory_from_curr_key[:-4])) - existing_largest_width / 2.0, 0]), min([horicenter_fitter2(float(trajectory_from_curr_key[:-4])) + existing_largest_width / 2.0, frames_width])]
            # if sum_error_trajectories_num_value_cnt == 0:
            #     sum_error_trajectories = 0
            # else:
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
    ##### 对轨迹预测结果进行可视化#####
    unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
    # if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_3dpolyfit_vis/')):
    #     os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_3dpolyfit_vis/'))
    # for frame_name in unique_frame_list:
    #     curr_img = cv2.imread(os.path.join(source, frame_name))
    #     for trackid in predicted_bbox_based_on_historical_traj:
    #         if frame_name in predicted_bbox_based_on_historical_traj[trackid]:
    #             top,bottom,left,right = predicted_bbox_based_on_historical_traj[trackid][frame_name]
    #             top2, bottom2, left2, right2 = predicted_bbox_based_on_historical_traj2[trackid][frame_name]
    #             cv2.rectangle(curr_img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2) # 蓝色为一阶拟合结果
    #             cv2.rectangle(curr_img, (int(left2), int(top2)), (int(right2), int(bottom2)), (0, 0, 255), 2) # 红色为二阶拟合结果
    #             cv2.putText(curr_img, str(trackid), (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    #     cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_3dpolyfit_vis/')  + frame_name, curr_img)
    # for i in range(np.size(tracklets_similarity_matrix,1)): # 行表示前一个batch,列表示当前batch
    #     result_dict_min[i+1] = min_index.tolist().index(i)+1 # 当前轨迹id与之前轨迹id对应关系  可能存在不对应的情况
    # 需要计算匹配的轨迹以及当前帧当中新出现的轨迹以及上一帧中没有匹配到的轨迹
    # 设置阈值
    iou_mask = np.where(tracklets_similarity_matrix > 0.7)
    tracklets_reid_similarity_matrix[iou_mask] = 1.0
    fused_similarity_matrix = np.minimum(100*tracklets_reid_similarity_matrix,tracklets_similarity_matrix) # 混合距离
    binary_similarity_matrix = (np.array(tracklets_similarity_matrix) < 0.6).astype(np.int32)
    if binary_similarity_matrix.sum(1).max() == 1 and binary_similarity_matrix.sum(0).max() == 1: #轨迹之间一一匹配
        matched_indices = np.stack(np.where(binary_similarity_matrix),axis=1)  # [:,0] 行索引  改为键值
        # result_dict
    else:
        matched_indices = linear_assignment_ori(tracklets_similarity_matrix)  #  损失矩阵，,fused_similarity_matrix fused_similarity_matrix
    # tracklets_similarity_matrix[matched_indices[:,0],matched_indices[:,1]]： 所有匹配的iou损失矩阵
    # tracklets_reid_similarity_matrix[matched_indices[:,0],matched_indices[:,1]]： 所有匹配的reid损失矩阵, > 0.02不是同一个人
    # tracklets_similarity_matrix[previous_tracks_id.index(36),:]
    previous_tracks_id = list(previous_video_segment_predicted_tracks_bboxes.keys())
    current_tracks_id = list(current_video_segment_predicted_tracks_bboxes.keys())
    previous_unmatched_tracks = []  # 前一个batch当中未匹配的
    curr_unmatched_tracks = []  # 当前batch未匹配的tracks的key
    mean_similarity_list = []
    for m in matched_indices: # 40,39
        # 对matched_indices进行判断
        # 对大于0.53的进行排除
        # print(tracklets_similarity_matrix[m[0],m[1]])
        mean_similarity_list.append(tracklets_similarity_matrix[m[0],m[1]])
        if tracklets_similarity_matrix[m[0],m[1]] >= 0.5:  # 3 and 40
            previous_unmatched_tracks.append(previous_tracks_id[m[0]])
            curr_unmatched_tracks.append(current_tracks_id[m[1]])
            continue
        result_dict[current_tracks_id[m[1]]] = previous_tracks_id[m[0]]
    # print(np.mean(mean_similarity_list))

    for track_id in previous_video_segment_predicted_tracks_bboxes:
        if (track_id not in np.array(previous_tracks_id)[matched_indices[:,0]]):
            previous_unmatched_tracks.append(track_id)

    for track_id in current_video_segment_predicted_tracks_bboxes:
        if (track_id not in np.array(current_tracks_id)[matched_indices[:,1]]):
            curr_unmatched_tracks.append(track_id)

    # new_track_list = []
    # terminate_track_list = []
    # for track_id in previous_unmatched_tracks:
    #     prev_track = previous_video_segment_predicted_tracks[track_id]
    #     center_x,center_y = prev_track[max(list(prev_track.keys()))]
    #     flag = center_x >= border_x_max or center_x <= border_x_min or center_y >= boder_y_max or center_y <= border_y_min
    #     if flag:
    #         terminate_track_list.append(track_id)
    # print('terminate track id(global) is {0}'.format(terminate_track_list))
    # for track_id in curr_unmatched_tracks:
    #     curr_track = current_video_segment_predicted_tracks[track_id]
    #     center_x,center_y = curr_track[min(list(curr_track.keys()))]
    #     flag = center_x >= border_x_max or center_x <= border_x_min or center_y >= boder_y_max or center_y <= border_y_min
    #     if flag:
    #         new_track_list.append(track_id)
    # print('new track id is {0}'.format(new_track_list))
    # # [previous_unmatched_tracks.remove(track) for track in terminate_track_list] # 去掉未匹配的轨迹避免对其回归造成id_switch
    # # [curr_unmatched_tracks.remove(track) for track in new_track_list]
    ### 第二次匹配  使用reid信息 ###
    ### 使用列表记录新开始的轨迹以及终止的轨迹，对于不是新开始或者终止的轨迹采用reid信息进行关联 ###
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
    #     prev_track_representative_conf = prev_track[nodes[np.argmax([prev_track[node][2] for node in prev_track])]][2]
    #     if prev_track_representative_conf < 0.8: # 如果置信度不够高说明提取出来的不够准确
    #         reid_similarity[second_prev_track_list.index(previous_id), :] = 1
    #         continue
    #     else:
    #         print('previous representative track {}'.format(previous_id))
    #         prev_track_representative_feature = prev_track[nodes[np.argmax([prev_track[node][2] for node in prev_track])]][3] # 该帧特征
    #     for current_id in second_curr_track_list:
    #         curr_track = current_video_segment_predicted_tracks_bboxes_test[current_id]
    #         curr_nodes = list(curr_track.keys())
    #         curr_track_representative_conf = curr_track[curr_nodes[np.argmax([curr_track[node][2] for node in curr_track])]][2]
    #         if curr_track_representative_conf < 0.8:
    #             reid_similarity[:, second_curr_track_list.index(current_id)] = 1
    #             continue
    #         print('current representative track {}'.format(current_id))
    #         similarity_list = [cosine_similarity(np.array(curr_track[node][3]),np.array(prev_track_representative_feature)) for node in curr_track]
    #         reid_similarity[second_prev_track_list.index(previous_id),second_curr_track_list.index(current_id)] = 1 - np.mean(similarity_list)
    # matched_indices, previous_unmatched_ids, curr_unmatched_ids = linear_assignment(reid_similarity, thresh=0.01)  # ???
    # for match in matched_indices:
    #     result_dict[second_curr_track_list[match[1]]] = second_prev_track_list[match[0]]
    #     previous_unmatched_tracks.remove(second_prev_track_list[match[0]])
    #     curr_unmatched_tracks.remove(second_curr_track_list[match[1]])
    #     print('matched_reid previous track id(global) is {0},current track id(local) is {1}'.format(second_prev_track_list[match[0]],second_curr_track_list[match[1]]))
    print('previous_unmatched(global) track {0},curr_unmatched_track(local) track {1} '.format(previous_unmatched_tracks,curr_unmatched_tracks))
    return  result_dict,previous_unmatched_tracks,curr_unmatched_tracks

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
            # time_start = time.time()
            # if curr_img.shape[0] == 1080:
            #     curr_img_resized = cv2.resize(curr_img[int(bbox[0][1]*2):int(bbox[1][1]*2), int(bbox[0][0]*2):int(bbox[1][0]*2), :], (int(fixed_height/(bbox[1][1]-bbox[0][1])*(bbox[1][0]-bbox[0][0])), fixed_height), interpolation=cv2.INTER_NEAREST)
            # else:

            # curr_img_resized = cv2.resize(curr_img[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0]), :], (int(fixed_height/(bbox[1][1]-bbox[0][1])*(bbox[1][0]-bbox[0][0])), fixed_height), interpolation=cv2.INTER_NEAREST)
            # curr_img_resized = curr_img_resized[:, int((curr_img_resized.shape[1] - fixed_width)/2):int((curr_img_resized.shape[1] - fixed_width)/2)+fixed_width, :] if \
            #                    curr_img_resized.shape[1] > fixed_width else np.pad(curr_img_resized, ((0,0), (int((fixed_width-curr_img_resized.shape[1])/2), fixed_width-curr_img_resized.shape[1]-int((fixed_width-curr_img_resized.shape[1])/2)), (0,0)), 'constant', constant_values=0)
            # curr_img_resized = cv2.resize(curr_img[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0]), :], (fixed_width, fixed_height), interpolation=cv2.INTER_CUBIC)

            # crop the regions containing current person
            # if curr_img is None:
            #     print(tracklet_pose_collection_input_item['img_dir'])
            #     break
            # curr_crop = curr_img[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0]), :]
            curr_crop = curr_img[max(0,int(bbox[0][1])):min(int(bbox[1][1]),curr_img.shape[0]), max(0,int(bbox[0][0])):min(int(bbox[1][0]),curr_img.shape[1]), :]
            if curr_crop.shape[0] > curr_crop.shape[1] * 2:
                curr_img_resized = cv2.resize(curr_crop, (fixed_width, int(fixed_width / curr_crop.shape[1] * curr_crop.shape[0])), interpolation=cv2.INTER_AREA)
                curr_img_resized = curr_img_resized[int((curr_img_resized.shape[0] - fixed_height) / 2):int((curr_img_resized.shape[0] - fixed_height) / 2) + fixed_height, :, :]
            elif curr_crop.shape[0] < curr_crop.shape[1] * 2:
                curr_img_resized = cv2.resize(curr_crop, (int(fixed_height / curr_crop.shape[0] * curr_crop.shape[1]), fixed_height), interpolation=cv2.INTER_AREA)
                curr_img_resized = curr_img_resized[:, int((curr_img_resized.shape[1] - fixed_width) / 2):int((curr_img_resized.shape[1] - fixed_width) / 2) + fixed_width, :]
            else:
                curr_img_resized = cv2.resize(curr_crop, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
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

def update_split_each_track_valid_mask(result):
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
        for node_pair in split_each_track[idx_track]:
            # This element is a node
            if (int(node_pair[0]) % 2 == 1) and (int(node_pair[1]) % 2 == 0): # 表示节点
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
        for node_pair in split_each_track[idx_track]:
            # traverse another track, idx_track and idx_another_track indicate two different tracks
            for idx_another_track in [x for x in range(1, int(result[0].split('Predicted tracks')[1].split('\n')[1].split('~~')[0]) + 1) if x != idx_track]: # 另外一条id不等于idx_track的轨迹
                # if the node with content node_pair appears in two trajectories: split_each_track[idx_track] and split_each_track[idx_another_track]
                if node_pair in split_each_track[idx_another_track]:
                    # length_common_curr_track_first_half: a list of indices in split_each_track[idx_track], starting from the previous one of node_pair and ending at the index of the first
                    # common node shared by split_each_track[idx_track] and split_each_track[idx_another_track]
                    # range前两个参数表示范围（，-1）到-1但是不包括，最后-1表示步长即倒序
                    length_common_curr_track_first_half = [x for x in range(split_each_track[idx_track].index(node_pair) - 1, -1, -1) \
                                                           if (split_each_track_valid_mask[idx_track][x] > 0 and split_each_track[idx_track][x] in split_each_track[idx_another_track])]
                    # length_common_curr_track_second_half: a list of indices in split_each_track[idx_track], starting from the index of node_pair and ending at the index of the last
                    # common node shared by split_each_track[idx_track] and split_each_track[idx_another_track]
                    length_common_curr_track_second_half = [x for x in range(split_each_track[idx_track].index(node_pair), len(split_each_track[idx_track])) \
                                                            if (split_each_track_valid_mask[idx_track][x] > 0 and split_each_track[idx_track][x] in split_each_track[idx_another_track])]
                    # the number of common nodes shared by split_each_track[idx_track] and split_each_track[idx_another_track]
                    length_common_curr_track = len(length_common_curr_track_first_half) + len(length_common_curr_track_second_half)
                    # length_common_another_track_first_half: the same as length_common_curr_track_first_half but in split_each_track[idx_another_track]
                    length_common_another_track_first_half = [x for x in range(split_each_track[idx_another_track].index(node_pair) - 1, -1, -1) \
                                                              if (split_each_track_valid_mask[idx_another_track][x] > 0 and split_each_track[idx_another_track][x] in split_each_track[idx_track])]
                    # length_common_another_track_second_half: the same as length_common_curr_track_second_half but in split_each_track[idx_another_track]
                    length_common_another_track_second_half = [x for x in range(split_each_track[idx_another_track].index(node_pair), len(split_each_track[idx_another_track])) \
                                                               if (split_each_track_valid_mask[idx_another_track][x] > 0 and split_each_track[idx_another_track][x] in split_each_track[idx_track])]
                    # the number of common nodes shared by split_each_track[idx_another_track] and split_each_track[idx_track]
                    length_common_another_track = len(length_common_another_track_first_half) + len(length_common_another_track_second_half)

                    # length_common_curr_track denotes the segment of nodes in current track all of which have appeared in another track,
                    # length_common_another_track denotes the segment of nodes in another track all of which have appeared in current track,
                    # if the former is shorter than the latter, this may happen in the case where other common nodes in current track appear in current track but not adjacent to the segment of nodes
                    # the shortest one in [length_common_curr_track, length_common_another_track] contains a subset of common nodes, their validity must be set to 0 and -1
                    if length_common_curr_track < length_common_another_track and len(length_common_curr_track_first_half) > 0 and len(length_common_curr_track_second_half) > 0:
                        for split_each_track_valid_mask_idx_track_idx in range(length_common_curr_track_first_half[-1], length_common_curr_track_second_half[-1] + 1):
                            if int(split_each_track[idx_track][split_each_track_valid_mask_idx_track_idx][0]) % 2 == 1 and int(split_each_track[idx_track][split_each_track_valid_mask_idx_track_idx][1]) % 2 == 0:
                                split_each_track_valid_mask[idx_track][split_each_track_valid_mask_idx_track_idx] = -1
                            else:
                                split_each_track_valid_mask[idx_track][split_each_track_valid_mask_idx_track_idx] = 0
                    elif length_common_curr_track < length_common_another_track and len(length_common_curr_track_first_half) == 0 and len(length_common_curr_track_second_half) > 0:
                        for split_each_track_valid_mask_idx_track_idx in range(split_each_track[idx_track].index(node_pair), length_common_curr_track_second_half[-1] + 1):
                            if int(split_each_track[idx_track][split_each_track_valid_mask_idx_track_idx][0]) % 2 == 1 and int(split_each_track[idx_track][split_each_track_valid_mask_idx_track_idx][1]) % 2 == 0:
                                split_each_track_valid_mask[idx_track][split_each_track_valid_mask_idx_track_idx] = -1
                            else:
                                split_each_track_valid_mask[idx_track][split_each_track_valid_mask_idx_track_idx] = 0
                    elif length_common_curr_track < length_common_another_track and len(length_common_curr_track_first_half) > 0 and len(length_common_curr_track_second_half) == 0:
                        for split_each_track_valid_mask_idx_track_idx in range(length_common_curr_track_first_half[-1], split_each_track[idx_track].index(node_pair) + 1):
                            if int(split_each_track[idx_track][split_each_track_valid_mask_idx_track_idx][0]) % 2 == 1 and int(split_each_track[idx_track][split_each_track_valid_mask_idx_track_idx][1]) % 2 == 0:
                                split_each_track_valid_mask[idx_track][split_each_track_valid_mask_idx_track_idx] = -1
                            else:
                                split_each_track_valid_mask[idx_track][split_each_track_valid_mask_idx_track_idx] = 0
                    elif length_common_curr_track > length_common_another_track and len(length_common_another_track_first_half) > 0 and len(length_common_another_track_second_half) > 0:
                        for split_each_track_valid_mask_idx_another_track_idx in range(length_common_another_track_first_half[-1], length_common_another_track_second_half[-1] + 1):
                            if int(split_each_track[idx_another_track][split_each_track_valid_mask_idx_another_track_idx][0]) % 2 == 1 and int(split_each_track[idx_another_track][split_each_track_valid_mask_idx_another_track_idx][1]) % 2 == 0:
                                split_each_track_valid_mask[idx_another_track][split_each_track_valid_mask_idx_another_track_idx] = -1
                            else:
                                split_each_track_valid_mask[idx_another_track][split_each_track_valid_mask_idx_another_track_idx] = 0
                    elif length_common_curr_track > length_common_another_track and len(length_common_another_track_first_half) == 0 and len(length_common_another_track_second_half) > 0:
                        for split_each_track_valid_mask_idx_another_track_idx in range(split_each_track[idx_another_track].index(node_pair), length_common_another_track_second_half[-1] + 1):
                            if int(split_each_track[idx_another_track][split_each_track_valid_mask_idx_another_track_idx][0]) % 2 == 1 and int(split_each_track[idx_another_track][split_each_track_valid_mask_idx_another_track_idx][1]) % 2 == 0: # 表示人的节点
                                split_each_track_valid_mask[idx_another_track][split_each_track_valid_mask_idx_another_track_idx] = -1
                            else: # 表示边的节点
                                split_each_track_valid_mask[idx_another_track][split_each_track_valid_mask_idx_another_track_idx] = 0
                    elif length_common_curr_track > length_common_another_track and len(length_common_another_track_first_half) > 0 and len(length_common_another_track_second_half) == 0:
                        for split_each_track_valid_mask_idx_another_track_idx in range(length_common_another_track_first_half[-1], split_each_track[idx_another_track].index(node_pair) + 1):
                            if int(split_each_track[idx_another_track][split_each_track_valid_mask_idx_another_track_idx][0]) % 2 == 1 and int(split_each_track[idx_another_track][split_each_track_valid_mask_idx_another_track_idx][1]) % 2 == 0:
                                split_each_track_valid_mask[idx_another_track][split_each_track_valid_mask_idx_another_track_idx] = -1
                            else:
                                split_each_track_valid_mask[idx_another_track][split_each_track_valid_mask_idx_another_track_idx] = 0

    return split_each_track, split_each_track_valid_mask

def compute_reid_vector_distance(pre_vector, post_vector):
    pre_vector = np.array(pre_vector)
    post_vector = np.array(post_vector)
    num = float(np.dot(pre_vector, post_vector.T))
    denom = np.linalg.norm(pre_vector) * np.linalg.norm(post_vector)
    cos = num / denom
    return 1.0 - cos

################################################################################################################################################
# Given two halves in different trajectories, determine whether need to hybridize
# Input:
# pairs_correction_relationships: a dict, each key is the index of one trajectory, each value is a list [ending node name (character) of the first half, starting node name of the second half]
# pairs_correction_two_halves: a dict, each key is the index of one trajectory, each value is a list [the first half, the second half]
#      the first half is a list of element each of which is a list with two characters of node names, the second half is in the same format
# split_each_track is a dict, each key is an index of a trajectory, the value is a list of nodes in one trajectory, each node is a list with two integers
# split_each_track_valid_mask is a dict, the keys are the same as split_each_track_valid, the value under each key is a bool number indicating the validity of the node
# need_to_fix_cnt is an integer indicating whether current sets of trajectories need hybridization, if need, need_to_fix_cnt is set to be larger than 0, if no fixing is required, need_to_fix_cnt is kept 0
# return_to_while is True when need_to_fix_cnt is set to be larger than 0
# result is the fixed string encoding all trajectories
# maximum_possible_number is math.exp(10)
# mapping_node_id_to_bbox: a dict, each key is a character of node name, each value is a list whose first element is bounding box coordinates, second
# element is confidence, third element is frame name string, refer to the supplementary materials
# mapping_node_id_to_features: a dict, each key is a character is node name, each value is an 512-d array storing a feature vector
# mapping_edge_id_to_cost: a dict, each key is a string with '_' connecting two characters each of which is a character of one node name, refer to the supplementary materials

def flat_area_optimize(pairs_correction_relationships, pairs_correction_two_halves, split_each_track, split_each_track_valid_mask, need_to_fix_cnt, return_to_while, result, maximum_possible_number, mapping_edge_id_to_cost, mapping_node_id_to_bbox, mapping_node_id_to_features):
    # For example, in
    # [... Ai, Ai+j,...]
    # [... Bi, Bi+j,...]
    # left_nodes is [Ai, Bi], right_nodes is [Ai+j, Bi+j]
    left_nodes = [pairs_correction_relationships[x][0] for x in pairs_correction_relationships.keys()]
    right_nodes = [pairs_correction_relationships[x][1] for x in pairs_correction_relationships.keys()]
    # Backup the two lists
    left_nodes_backup = copy.deepcopy(left_nodes)
    right_nodes_backup = copy.deepcopy(right_nodes)
    # print(left_nodes)
    # print(right_nodes)
    # replace left nodes with previous more confident ones
    # each trajectory has multiple nodes, each node is a bounding box with confidence, for each left node, we find the bounding box in its trajectory
    # with the highest confidence for representing its appearance
    # For example, in
    # [...Ak... Ai, Ai+j, ...Ah...]
    # [...Bp... Bi, Bi+j, ...Bq...]
    # we find that Ak has highest confidence in the first half of the first trajectory, Ah, Bp, Bq all have highest confidences in correponding halves,
    # we compute matching error between [Ak, Ah], [Ak, Bq], [Bp, Ah], [Bp, Bq]
    # left_nodes_prev_seqs is a list, each element is a cut half of an original trajectory, example:
    # [...Ak... Ai]
    # [...Bp... Bi]
    # each Ak, Ai, Bp, Bi is a list of two integers, if the former one is odd and latter one is even, then the element represents one bounding box, else it represents one edge
    # left_nodes_prev_seq_confidences is a list, each element is a cut half of an original trajectory, example:
    # [...Ak... Ai]
    # [...Bp... Bi]
    # each Ak, Ai, Bp, Bi is a floating confidence value
    # left_nodes stores the node which highest confidence in each trajectory, example:
    # [Ak]
    # [Bp]
    left_nodes_prev_seqs = [pairs_correction_two_halves[x][0] for x in pairs_correction_relationships.keys()]
    if None not in left_nodes_prev_seqs:
        for left_nodes_prev_seq_idx in range(len(left_nodes_prev_seqs)):
            left_nodes_prev_seq_confidences = []
            for candidate_node in left_nodes_prev_seqs[left_nodes_prev_seq_idx]:
                # each element in left_nodes_prev_seqs[left_nodes_prev_seq_idx] is a list of two integers, if the former one is odd and latter one is
                # even, then the element represents one bounding box, else it represents one edge
                if int(candidate_node[0]) % 2 == 1 and int(candidate_node[1]) % 2 == 0:
                    left_nodes_prev_seq_confidences.append(mapping_node_id_to_bbox[int(int(candidate_node[1]) / 2)][1])
                else:
                    left_nodes_prev_seq_confidences.append(0.0)
            left_nodes[left_nodes_prev_seq_idx] = left_nodes_prev_seqs[left_nodes_prev_seq_idx][np.argmax(left_nodes_prev_seq_confidences)][1]
    right_node_post_seqs = [pairs_correction_two_halves[x][1] for x in pairs_correction_relationships.keys()]
    if None not in right_node_post_seqs:
        for right_nodes_post_seq_idx in range(len(right_node_post_seqs)):
            right_nodes_post_seq_confidences = []
            for candidate_node in right_node_post_seqs[right_nodes_post_seq_idx]:
                if int(candidate_node[0]) % 2 == 1 and int(candidate_node[1]) % 2 == 0:
                    right_nodes_post_seq_confidences.append(mapping_node_id_to_bbox[int(int(candidate_node[1]) / 2)][1])
                else:
                    right_nodes_post_seq_confidences.append(0.0)
            right_nodes[right_nodes_post_seq_idx] = right_node_post_seqs[right_nodes_post_seq_idx][np.argmax(right_nodes_post_seq_confidences)][1]

    # assert (len(left_nodes) == len(np.unique(left_nodes)))
    # assert (len(right_nodes) == len(np.unique(right_nodes)))
    all_possible_correct_pairs = []
    all_possible_correct_pairs_ori = []
    # permutate to try all pairs for hybridization
    all_possible_pairing_orders = list(permutations(range(0, len(left_nodes))))
    for one_possible_pairing_orders in all_possible_pairing_orders:
        one_possible_correct_pairs = []
        # each element in one_possible_correct_pairs is a list of two nodes, the former from one left half, the latter from one right half
        for left_nodes_idx in range(len(left_nodes)):
            one_possible_correct_pairs.append([left_nodes[left_nodes_idx], right_nodes[one_possible_pairing_orders[left_nodes_idx]]])
        if len(one_possible_correct_pairs) > 0:
            all_possible_correct_pairs.append(one_possible_correct_pairs)
        # afraid of all_possible_correct_pairs being modified
        one_possible_correct_pairs_ori = []
        for left_nodes_idx in range(len(left_nodes_backup)):
            one_possible_correct_pairs_ori.append([left_nodes_backup[left_nodes_idx], right_nodes_backup[one_possible_pairing_orders[left_nodes_idx]]])
        if len(one_possible_correct_pairs_ori) > 0:
            all_possible_correct_pairs_ori.append(one_possible_correct_pairs_ori)
    # add additional possible combiations
    # ori: A-C, B-D, E-F,  revised: A-D-C, B-, E-F
    pairs_correction_two_halves_revised_list = []
    add_all_possible_correct_pairs = []
    for right_nodes_backup_idx in range(len(right_nodes_backup)):
        right_nodes_backup_key = [x for x in pairs_correction_two_halves][right_nodes_backup_idx]
        curr_right_node = right_nodes_backup[right_nodes_backup_idx]
        for left_nodes_backup_idx in [x for x in range(len(left_nodes_backup)) if x != right_nodes_backup_idx]:
            pairs_correction_two_halves_revised = {}
            for pairs_correction_two_halves_key in pairs_correction_two_halves:
                pairs_correction_two_halves_revised[pairs_correction_two_halves_key] = pairs_correction_two_halves[pairs_correction_two_halves_key]
            left_nodes_backup_key = [x for x in pairs_correction_two_halves][left_nodes_backup_idx]
            new_sequence_with_curr_right_node_inserted = pairs_correction_two_halves[left_nodes_backup_key]
            one_possible_correct_pairs = []
            if (None in pairs_correction_two_halves[right_nodes_backup_key]) or (None in pairs_correction_two_halves[left_nodes_backup_key]):
                continue
            if len([x for x in new_sequence_with_curr_right_node_inserted[0] if (int(pairs_correction_two_halves[right_nodes_backup_key][1][0][0]) > int(x[0]) and int(pairs_correction_two_halves[right_nodes_backup_key][1][-1][1]) < int(x[1]))]) > 0:
                new_sequence_with_curr_right_node_inserted_insert_location = new_sequence_with_curr_right_node_inserted[0].index([x for x in new_sequence_with_curr_right_node_inserted[0] if (int(pairs_correction_two_halves[right_nodes_backup_key][1][0][0]) > int(x[0]) and int(pairs_correction_two_halves[right_nodes_backup_key][1][-1][1]) < int(x[1]))][0])
                new_sequence_with_curr_right_node_inserted[0] = new_sequence_with_curr_right_node_inserted[0][:new_sequence_with_curr_right_node_inserted_insert_location] + \
                                                                [[new_sequence_with_curr_right_node_inserted[0][new_sequence_with_curr_right_node_inserted_insert_location][0], pairs_correction_two_halves[right_nodes_backup_key][1][0][0]]] + \
                                                                pairs_correction_two_halves[right_nodes_backup_key][1] + \
                                                                [[pairs_correction_two_halves[right_nodes_backup_key][1][-1][1], new_sequence_with_curr_right_node_inserted[0][new_sequence_with_curr_right_node_inserted_insert_location][1]]] + \
                                                                new_sequence_with_curr_right_node_inserted[0][new_sequence_with_curr_right_node_inserted_insert_location + 1:]
                one_possible_correct_pairs.append([new_sequence_with_curr_right_node_inserted[0][new_sequence_with_curr_right_node_inserted_insert_location][0], pairs_correction_two_halves[right_nodes_backup_key][1][0][0]])
                one_possible_correct_pairs.append([left_nodes_backup[left_nodes_backup_idx], right_nodes_backup[left_nodes_backup_idx]])
                pairs_correction_two_halves_revised[right_nodes_backup_key][1] = None
            elif len([x for x in new_sequence_with_curr_right_node_inserted[1] if (int(pairs_correction_two_halves[right_nodes_backup_key][1][0][0]) > int(x[0]) and int(pairs_correction_two_halves[right_nodes_backup_key][1][-1][1]) < int(x[1]))]) > 0:
                new_sequence_with_curr_right_node_inserted_insert_location = new_sequence_with_curr_right_node_inserted[1].index([x for x in new_sequence_with_curr_right_node_inserted[1] if (int(pairs_correction_two_halves[right_nodes_backup_key][1][0][0]) > int(x[0]) and int(pairs_correction_two_halves[right_nodes_backup_key][1][-1][1]) < int(x[1]))][0])
                new_sequence_with_curr_right_node_inserted[1] = new_sequence_with_curr_right_node_inserted[1][:new_sequence_with_curr_right_node_inserted_insert_location] + \
                                                                [[new_sequence_with_curr_right_node_inserted[1][new_sequence_with_curr_right_node_inserted_insert_location][0], pairs_correction_two_halves[right_nodes_backup_key][1][0][0]]] + \
                                                                pairs_correction_two_halves[right_nodes_backup_key][1] + \
                                                                [[pairs_correction_two_halves[right_nodes_backup_key][1][-1][1], new_sequence_with_curr_right_node_inserted[1][new_sequence_with_curr_right_node_inserted_insert_location][1]]] + \
                                                                new_sequence_with_curr_right_node_inserted[1][new_sequence_with_curr_right_node_inserted_insert_location + 1:]
                one_possible_correct_pairs.append([new_sequence_with_curr_right_node_inserted[1][new_sequence_with_curr_right_node_inserted_insert_location][0], pairs_correction_two_halves[right_nodes_backup_key][1][0][0]])
                one_possible_correct_pairs.append([left_nodes_backup[left_nodes_backup_idx], right_nodes_backup[left_nodes_backup_idx]])
                pairs_correction_two_halves_revised[right_nodes_backup_key][1] = None
            else:
                continue
            pairs_correction_two_halves_revised[left_nodes_backup_key] = new_sequence_with_curr_right_node_inserted
            for other_nodes_idx in [x for x in range(len(left_nodes_backup)) if (x != right_nodes_backup_idx and x != left_nodes_backup_idx)]:
                one_possible_correct_pairs.append([left_nodes_backup[other_nodes_idx], right_nodes_backup[other_nodes_idx]])
                other_nodes_key = [x for x in pairs_correction_two_halves][other_nodes_idx]
                pairs_correction_two_halves_revised[other_nodes_key] = pairs_correction_two_halves[other_nodes_key]
            add_all_possible_correct_pairs.append(one_possible_correct_pairs)
            pairs_correction_two_halves_revised_list.append(pairs_correction_two_halves_revised)
    # add additional possible combinations

    #### minimize loss
    # compute the matching error between all pairs
    all_possible_correct_pairs_loss = [maximum_possible_number * len(all_possible_correct_pairs[0])] * len(all_possible_correct_pairs)
    for all_possible_correct_pairs_idx in range(len(all_possible_correct_pairs)):
        curr_combination_loss = 0.0
        for all_possible_correct_pairs_idx_ele in range(len(all_possible_correct_pairs[all_possible_correct_pairs_idx])):
            pre_node = int(int(all_possible_correct_pairs[all_possible_correct_pairs_idx][all_possible_correct_pairs_idx_ele][0]) / 2)
            post_node = int((int(all_possible_correct_pairs[all_possible_correct_pairs_idx][all_possible_correct_pairs_idx_ele][1]) + 1) / 2)
            if pre_node in mapping_node_id_to_features and post_node in mapping_node_id_to_features:
                curr_combination_loss += compute_reid_vector_distance(mapping_node_id_to_features[pre_node], mapping_node_id_to_features[post_node])
            else:
                curr_combination_loss += maximum_possible_number
            # if str(pre_node) + '_' + str(post_node) in mapping_edge_id_to_cost:
            #     curr_combination_loss += mapping_edge_id_to_cost[str(pre_node) + '_' + str(post_node)]
            # else:
            #     curr_combination_loss += maximum_possible_number
        all_possible_correct_pairs_loss[all_possible_correct_pairs_idx] = curr_combination_loss

    # add additional possible combiations
    add_all_possible_correct_pairs_loss = []
    if len(add_all_possible_correct_pairs) > 0:
        add_all_possible_correct_pairs_loss = [maximum_possible_number * len(add_all_possible_correct_pairs[0])] * len(add_all_possible_correct_pairs)
        for all_possible_correct_pairs_idx in range(len(add_all_possible_correct_pairs)):
            curr_combination_loss = 0.0
            for all_possible_correct_pairs_idx_ele in range(len(add_all_possible_correct_pairs[all_possible_correct_pairs_idx])):
                pre_node = int(int(add_all_possible_correct_pairs[all_possible_correct_pairs_idx][all_possible_correct_pairs_idx_ele][0]) / 2)
                post_node = int((int(add_all_possible_correct_pairs[all_possible_correct_pairs_idx][all_possible_correct_pairs_idx_ele][1]) + 1) / 2)
                if pre_node in mapping_node_id_to_features and post_node in mapping_node_id_to_features:
                    curr_combination_loss += compute_reid_vector_distance(mapping_node_id_to_features[pre_node], mapping_node_id_to_features[post_node])
                else:
                    curr_combination_loss += maximum_possible_number
            add_all_possible_correct_pairs_loss[all_possible_correct_pairs_idx] = curr_combination_loss

    if len(add_all_possible_correct_pairs) > 0 and np.min(add_all_possible_correct_pairs_loss) < np.min(all_possible_correct_pairs_loss):
        pairs_correction_two_halves = pairs_correction_two_halves_revised_list[np.argmin(add_all_possible_correct_pairs_loss)]
        for pairs_correction_two_halves_key in pairs_correction_two_halves:
            if None not in pairs_correction_two_halves[pairs_correction_two_halves_key]:
                split_each_track[pairs_correction_two_halves_key] = pairs_correction_two_halves[pairs_correction_two_halves_key][0] + \
                                                                    [[pairs_correction_two_halves[pairs_correction_two_halves_key][0][-1][1], pairs_correction_two_halves[pairs_correction_two_halves_key][1][0][0]]] + \
                                                                    pairs_correction_two_halves[pairs_correction_two_halves_key][1]
                need_to_fix_cnt += 1
                return_to_while = True
                result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
                split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
            elif pairs_correction_two_halves[pairs_correction_two_halves_key][0] is not None and pairs_correction_two_halves[pairs_correction_two_halves_key][1] is None:
                split_each_track[pairs_correction_two_halves_key] = pairs_correction_two_halves[pairs_correction_two_halves_key][0]
                need_to_fix_cnt += 1
                return_to_while = True
                result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
                split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
            elif pairs_correction_two_halves[pairs_correction_two_halves_key][0] is None and pairs_correction_two_halves[pairs_correction_two_halves_key][1] is not None:
                split_each_track[pairs_correction_two_halves_key] = pairs_correction_two_halves[pairs_correction_two_halves_key][1]
                need_to_fix_cnt += 1
                return_to_while = True
                result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
                split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
    # add additional possible combiations end
    else:
        pairs_decision = all_possible_correct_pairs_ori[np.argmin(all_possible_correct_pairs_loss)]
        for trajectory_first_half_key in pairs_correction_two_halves:
            for trajectory_second_half_key in [y for y in pairs_correction_two_halves.keys() if y != trajectory_first_half_key]:
                # if current hybridization with two halves from two trajectories is pairs_decision with minimum loss
                if pairs_correction_two_halves[trajectory_first_half_key][0] is not None and \
                        pairs_correction_two_halves[trajectory_second_half_key][1] is not None and \
                        [pairs_correction_two_halves[trajectory_first_half_key][0][-1][1], pairs_correction_two_halves[trajectory_second_half_key][1][0][0]] in pairs_decision:
                    # the ending element of first half is a node representing a bbox, the starting element of second half is a node representing a bbox, we need to build a node
                    # representing an edge between the two bboxes
                    split_each_track[trajectory_first_half_key] = pairs_correction_two_halves[trajectory_first_half_key][0] + \
                                                                  [[pairs_correction_two_halves[trajectory_first_half_key][0][-1][1], pairs_correction_two_halves[trajectory_second_half_key][1][0][0]]] + \
                                                                  pairs_correction_two_halves[trajectory_second_half_key][1]
                    need_to_fix_cnt += 1
                    return_to_while = True
                    result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
                    split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
                elif pairs_correction_two_halves[trajectory_first_half_key][0] is not None and \
                        pairs_correction_two_halves[trajectory_second_half_key][1] is None and \
                        [pairs_correction_two_halves[trajectory_first_half_key][0][-1][1], str(int(max([int(x) for x in mapping_node_id_to_bbox.keys()])) * 4)] in pairs_decision:
                    split_each_track[trajectory_first_half_key] = pairs_correction_two_halves[trajectory_first_half_key][0]
                    need_to_fix_cnt += 1
                    return_to_while = True
                    result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
                    split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
                elif pairs_correction_two_halves[trajectory_first_half_key][0] is None and \
                        pairs_correction_two_halves[trajectory_second_half_key][1] is not None and \
                        [str(int(max([int(x) for x in mapping_node_id_to_bbox.keys()])) * 4), pairs_correction_two_halves[trajectory_second_half_key][1][0][0]] in pairs_decision:
                    split_each_track[trajectory_first_half_key] = pairs_correction_two_halves[trajectory_second_half_key][1]
                    need_to_fix_cnt += 1
                    return_to_while = True
                    result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
                    split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
    return split_each_track, split_each_track_valid_mask, need_to_fix_cnt, return_to_while, result

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

# output:
# result - a list with two elements, the first element is a string, an example of the string is
# # 'Predicted tracks
# # 3~~61~56~55~50~49~44~43~40~39~36~35~30~29~24~23~18~17~10~9~2~1~0~~61~58~57~52~51~46~45~38~37~32~31~26~25~22~21~14~13~8~7~4~3~0~~61~60~59~54~53~48~47~42~41~34~33~28~27~20~19~16~15~12~11~6~5~0'
def revise_result(result, mapping_edge_id_to_cost, mapping_node_id_to_bbox, mapping_node_id_to_features):
    # The reason for using while is we traverse all nodes of all trajectories to determine if a node also appears in another trajectory
    # When we have processed on duplicate node, we set return_to_while to True, inner loops are broken when seeing return_to_while == True
    # need_to_fix_cnt += accompanies return_to_while = True, if a while loop finds need_to_fix_cnt == 0, then no duplicate nodes appear, so the while is broken and this function returns

    # convert string 'result' which encodes all trajectories in a string to split_each_track which encodes all trajectories in a dict
    # split_each_track is a dict, each key is an index of a trajectory, the value is a list of nodes in one trajectory, each node is a list with two integers
    # split_each_track_valid_mask is a dict, the keys are the same as split_each_track_valid, the value under each key is a bool number indicating the validity of the node
    split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
    # assert the first nodes in all trajectories are valid
    # assert(0 not in [(int(split_each_track[x][0][0]) < int(split_each_track[x][0][1])) for x in split_each_track])
    # for split_each_track_key in split_each_track:
    #     if split_each_track[split_each_track_key][0][0] > split_each_track[split_each_track_key][0][1]:

    # Step 1: locate the first halves of all trajectories
    for split_each_track_key in split_each_track:
        for element_idx in range(0, len(split_each_track[split_each_track_key]), 2):
            if int(split_each_track[split_each_track_key][element_idx][0]) > int(split_each_track[split_each_track_key][element_idx][1]):
                for split_each_track_another_key in [x for x in split_each_track if x != split_each_track_key]:
                    if [split_each_track[split_each_track_key][element_idx][1], split_each_track[split_each_track_key][element_idx][0]] in split_each_track[split_each_track_another_key]:
                        split_each_track_valid_mask[split_each_track_another_key][split_each_track[split_each_track_another_key].index([split_each_track[split_each_track_key][element_idx][1], split_each_track[split_each_track_key][element_idx][0]])] = -1
                        if split_each_track[split_each_track_another_key].index([split_each_track[split_each_track_key][element_idx][1], split_each_track[split_each_track_key][element_idx][0]]) - 1 >= 0:
                            split_each_track_valid_mask[split_each_track_another_key][split_each_track[split_each_track_another_key].index([split_each_track[split_each_track_key][element_idx][1], split_each_track[split_each_track_key][element_idx][0]]) - 1] = 0
                        if split_each_track[split_each_track_another_key].index([split_each_track[split_each_track_key][element_idx][1], split_each_track[split_each_track_key][element_idx][0]]) - 2 >= 0: # < len(split_each_track_valid_mask[split_each_track_another_key]):
                            split_each_track_valid_mask[split_each_track_another_key][split_each_track[split_each_track_another_key].index([split_each_track[split_each_track_key][element_idx][1], split_each_track[split_each_track_key][element_idx][0]]) - 2] = 0

    # Allen: you must remove
    for split_each_track_valid_mask_key in split_each_track_valid_mask:
        for element_idx in range(1, len(split_each_track_valid_mask[split_each_track_valid_mask_key]) - 1):
            if split_each_track_valid_mask[split_each_track_valid_mask_key][element_idx] == -1 and len(split_each_track_valid_mask[split_each_track_valid_mask_key]) > element_idx + 1:
                split_each_track_valid_mask[split_each_track_valid_mask_key][element_idx + 1] = 0
            if split_each_track_valid_mask[split_each_track_valid_mask_key][element_idx] == -1 and element_idx - 1 >= 0:
                split_each_track_valid_mask[split_each_track_valid_mask_key][element_idx - 1] = 0
        if int(split_each_track[split_each_track_valid_mask_key][0][0]) % 2 == 0 and int(split_each_track[split_each_track_valid_mask_key][0][1]) % 2 == 1:
            split_each_track_valid_mask[split_each_track_valid_mask_key][0] = 0

    split_each_track_valid_first_halves = {}
    for split_each_track_key in split_each_track:
        split_each_track_valid_first_halves[split_each_track_key] = []
        start_valid_element_idx = [x for x in range(len(split_each_track_valid_mask[split_each_track_key])) if split_each_track_valid_mask[split_each_track_key][x] > 0][0]
        for element_idx in range(start_valid_element_idx, len(split_each_track_valid_mask[split_each_track_key])):
            if split_each_track_valid_mask[split_each_track_key][element_idx] > 0:
                split_each_track_valid_first_halves[split_each_track_key].append(split_each_track[split_each_track_key][element_idx])
            else:
                break

    independent_nodes_collection = []
    for split_each_track_key in split_each_track:
        for element_node in split_each_track[split_each_track_key]:
            if (element_node not in split_each_track_valid_first_halves[split_each_track_key]) and \
                    (int(element_node[0]) % 2 == 1) and (int(element_node[1]) % 2 == 0) and (int(element_node[0]) < int(element_node[1])):
                independent_nodes_collection.append(element_node)
    independent_nodes_collection = sorted(independent_nodes_collection)
    # the nodes in mapping_node_id_to_bbox but not in split_each_track should also be considered
    split_each_track_node_collection = []
    for split_each_track_key in split_each_track:
        split_each_track_node_collection += [x for x in split_each_track[split_each_track_key] if (int(x[0]) % 2 == 1 and int(x[1]) % 2 == 0)]
    for mapping_node_id_to_bbox_key in mapping_node_id_to_bbox:
        if [str(mapping_node_id_to_bbox_key * 2 - 1), str(mapping_node_id_to_bbox_key * 2)] not in split_each_track_node_collection:
            independent_nodes_collection.append([str(mapping_node_id_to_bbox_key * 2 - 1), str(mapping_node_id_to_bbox_key * 2)])

    for independent_node in independent_nodes_collection:
        curr_node_matching_scores_with_all_split_each_track_valid_first_halves = []
        curr_node_insert_locations_in_all_split_each_track_valid_first_halves = []
        for split_each_track_key in split_each_track_valid_first_halves:
            if len([x for x in range(len(split_each_track_valid_first_halves[split_each_track_key])) if (int(split_each_track_valid_first_halves[split_each_track_key][x][0]) < int(independent_node[0]) and int(split_each_track_valid_first_halves[split_each_track_key][x][1]) > int(independent_node[1]))]) > 0:
                curr_traj_insert_location = [x for x in range(len(split_each_track_valid_first_halves[split_each_track_key])) if (int(split_each_track_valid_first_halves[split_each_track_key][x][0]) < int(independent_node[0]) and int(split_each_track_valid_first_halves[split_each_track_key][x][1]) > int(independent_node[1]))][0]
                port_in_ori_traj = split_each_track_valid_first_halves[split_each_track_key][curr_traj_insert_location][0]
                port_in_curr_node = independent_node[0]
                if len([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) <= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key] and mapping_node_id_to_bbox[x][1] <= 1.0)]) == 0:
                    former_reid_feature = np.mean([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) <= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key])], axis=0)
                else:
                    former_reid_feature = np.mean([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) <= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key] and mapping_node_id_to_bbox[x][1] <= 1.0)], axis=0)  # np.array(mapping_node_id_to_features[int(int(port_in_ori_traj) / 2)])
                latter_reid_feature = np.array(mapping_node_id_to_features[int(int(independent_node[1]) / 2)])
                num = float(np.dot(former_reid_feature, latter_reid_feature.T))
                denom = np.linalg.norm(former_reid_feature) * np.linalg.norm(latter_reid_feature)
                cos = num / denom
                depth_similarity = 1.0 / max([0.5, abs(mapping_node_id_to_bbox[int(int(port_in_ori_traj) / 2)][0][1][1] - mapping_node_id_to_bbox[int(int(independent_node[1]) / 2)][0][1][1])])
                if mapping_node_id_to_bbox[int(int(independent_node[1]) / 2)][2] in [mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_features if (int(x) >= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key])]:
                    curr_node_matching_scores_with_all_split_each_track_valid_first_halves.append(0)
                else:
                    curr_node_matching_scores_with_all_split_each_track_valid_first_halves.append((0.5 + 0.5 * cos)) # * depth_similarity)
            elif int(split_each_track_valid_first_halves[split_each_track_key][-1][1]) < int(independent_node[0]):
                curr_traj_insert_location = len(split_each_track_valid_first_halves[split_each_track_key])
                port_in_ori_traj = split_each_track_valid_first_halves[split_each_track_key][-1][1]
                port_in_curr_node = independent_node[0]
                if len([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) <= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key] and mapping_node_id_to_bbox[x][1] <= 1.0)]) == 0:
                    former_reid_feature = np.mean([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) <= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key])], axis=0)
                else:
                    former_reid_feature = np.mean([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) <= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key] and mapping_node_id_to_bbox[x][1] <= 1.0)], axis=0)  # np.array(mapping_node_id_to_features[int(int(port_in_ori_traj) / 2)])
                latter_reid_feature = np.array(mapping_node_id_to_features[int(int(independent_node[1]) / 2)])
                num = float(np.dot(former_reid_feature, latter_reid_feature.T))
                denom = np.linalg.norm(former_reid_feature) * np.linalg.norm(latter_reid_feature)
                cos = num / denom
                depth_similarity = 1.0 / max([0.5, abs(mapping_node_id_to_bbox[int(int(port_in_ori_traj) / 2)][0][1][1] - mapping_node_id_to_bbox[int(int(independent_node[1]) / 2)][0][1][1])])
                if mapping_node_id_to_bbox[int(int(port_in_ori_traj) / 2)][2] == mapping_node_id_to_bbox[int(int(independent_node[1]) / 2)][2]:
                    curr_node_matching_scores_with_all_split_each_track_valid_first_halves.append(0)
                else:
                    curr_node_matching_scores_with_all_split_each_track_valid_first_halves.append((0.5 + 0.5 * cos)) # * depth_similarity)
            elif int(split_each_track_valid_first_halves[split_each_track_key][0][0]) > int(independent_node[1]):
                curr_traj_insert_location = 0
                port_in_ori_traj = split_each_track_valid_first_halves[split_each_track_key][0][1]
                port_in_curr_node = independent_node[1]
                if len([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) >= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key] and mapping_node_id_to_bbox[x][1] <= 1.0)]) == 0:
                    former_reid_feature = np.mean([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) >= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key])], axis=0)
                else:
                    former_reid_feature = np.mean([mapping_node_id_to_features[x] for x in mapping_node_id_to_features if (int(x) >= int(int(port_in_ori_traj) / 2) and [str(int(x) * 2 - 1), str(int(x) * 2)] in split_each_track_valid_first_halves[split_each_track_key] and mapping_node_id_to_bbox[x][1] <= 1.0)], axis=0)  # np.array(mapping_node_id_to_features[int(int(port_in_ori_traj) / 2)])
                latter_reid_feature = np.array(mapping_node_id_to_features[int(int(independent_node[1]) / 2)])
                num = float(np.dot(former_reid_feature, latter_reid_feature.T))
                denom = np.linalg.norm(former_reid_feature) * np.linalg.norm(latter_reid_feature)
                cos = num / denom
                depth_similarity = 1.0 / max([0.5, abs(mapping_node_id_to_bbox[int(int(port_in_ori_traj) / 2)][0][1][1] - mapping_node_id_to_bbox[int(int(independent_node[1]) / 2)][0][1][1])])
                if mapping_node_id_to_bbox[int(int(port_in_ori_traj) / 2)][2] == mapping_node_id_to_bbox[int(int(independent_node[1]) / 2)][2]:
                    curr_node_matching_scores_with_all_split_each_track_valid_first_halves.append(0)
                else:
                    curr_node_matching_scores_with_all_split_each_track_valid_first_halves.append((0.5 + 0.5 * cos))  # * depth_similarity)
            else:
                curr_traj_insert_location = -1
                curr_node_matching_scores_with_all_split_each_track_valid_first_halves.append(0)
            curr_node_insert_locations_in_all_split_each_track_valid_first_halves.append(curr_traj_insert_location)
        if np.max(curr_node_matching_scores_with_all_split_each_track_valid_first_halves) == 0:
            continue
        else:
            idx_of_trajectory_to_insert = np.argmax(curr_node_matching_scores_with_all_split_each_track_valid_first_halves)
            key_of_trajectory_to_insert = [x for x in split_each_track][np.argmax(curr_node_matching_scores_with_all_split_each_track_valid_first_halves)]
            location_to_insert = curr_node_insert_locations_in_all_split_each_track_valid_first_halves[idx_of_trajectory_to_insert]
            if location_to_insert == -1:
                continue
            elif location_to_insert == len(split_each_track_valid_first_halves[key_of_trajectory_to_insert]):
                split_each_track_valid_first_halves[key_of_trajectory_to_insert] = split_each_track_valid_first_halves[key_of_trajectory_to_insert] + \
                                                                                   [[split_each_track_valid_first_halves[key_of_trajectory_to_insert][-1][1], independent_node[0]]] + \
                                                                                   [independent_node]
            elif location_to_insert == 0:
                split_each_track_valid_first_halves[key_of_trajectory_to_insert] = [independent_node] + \
                                                                                   [[independent_node[1], split_each_track_valid_first_halves[key_of_trajectory_to_insert][0][0]]] + \
                                                                                   split_each_track_valid_first_halves[key_of_trajectory_to_insert]
            else:
                split_each_track_valid_first_halves[key_of_trajectory_to_insert] = split_each_track_valid_first_halves[key_of_trajectory_to_insert][:location_to_insert] + \
                                                                                   [[split_each_track_valid_first_halves[key_of_trajectory_to_insert][location_to_insert][0], independent_node[0]]] + \
                                                                                   [independent_node] + \
                                                                                   [[independent_node[1], split_each_track_valid_first_halves[key_of_trajectory_to_insert][location_to_insert][1]]] + \
                                                                                   split_each_track_valid_first_halves[key_of_trajectory_to_insert][location_to_insert + 1:]

        split_each_track = copy.deepcopy(split_each_track_valid_first_halves)
        result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
        split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)




    # if one trajectory in split_each_track has two nodes A and B which are both bounding boxes, A appears in t, B appears in t+1, but the iou between A and B is lower than split_single_trajectory_thresh==0.01
    # then the half ending at A and the other half beginning at B should not be in the same trajectory

    # while (True):
    #     need_to_fix_cnt = 0
    #     return_to_while = False
    #     for split_each_track_key in split_each_track.keys():
    #         if return_to_while == True:
    #             break
    #         curr_trajectory = split_each_track[split_each_track_key]
    #         ########### as mentioned above, each trajectory is a list, each element of a trajectory is a list with two numbers
    #         ############ if the index of an element in a trajectory is even (indices starting at 0), the element represents a bounding box, if odd, the element represents an edge
    #         ############ curr_trajectory includes only bounding boxes
    #         curr_trajectory = curr_trajectory[0:len(curr_trajectory):2]
    #         for curr_traj_time_step in range(0, len(curr_trajectory) - 1):
    #             if return_to_while == True:
    #                 break
    #             ############ for the same bounding box, its node ids in mapping_node_id_to_bbox is half of that in trajectories,
    #             ############ for instance, a bounding box has node ids 36 and 37 in trajectories, but its node id is 18 in mapping_node_id_to_bbox
    #             former_bbox = mapping_node_id_to_bbox[int(int(curr_trajectory[curr_traj_time_step][1]) / 2)][0]
    #             former_bbox_time = int(
    #                 mapping_node_id_to_bbox[int(int(curr_trajectory[curr_traj_time_step][1]) / 2)][2][:-4])
    #             latter_bbox = mapping_node_id_to_bbox[int(int(curr_trajectory[curr_traj_time_step + 1][1]) / 2)][0]
    #             latter_bbox_time = int(
    #                 mapping_node_id_to_bbox[int(int(curr_trajectory[curr_traj_time_step + 1][1]) / 2)][2][:-4])
    #             global max_movement_between_adjacent_frames
    #             if compute_iou_single_box([former_bbox[0][1], former_bbox[1][1], former_bbox[0][0], former_bbox[1][0]],
    #                                       [latter_bbox[0][1], latter_bbox[1][1], latter_bbox[0][0],
    #                                        latter_bbox[1][0]]) < split_single_trajectory_thresh and abs(
    #                     former_bbox_time - latter_bbox_time) == 1:
    #                 ############ the half ending at curr_traj_time_step and the other half beginning at curr_traj_time_step+1 should not be the same trajectory, set the second one to be a new trajectory with key being len(split_each_track) + 1
    #                 split_each_track[len(split_each_track) + 1] = interpolate_to_obtain_traj(
    #                     curr_trajectory[curr_traj_time_step + 1:])
    #                 split_each_track[split_each_track_key] = interpolate_to_obtain_traj(
    #                     curr_trajectory[:curr_traj_time_step + 1])
    #                 result[0] = 'Predicted tracks' + '\n' + convert_dict_to_str(result, split_each_track)
    #                 split_each_track, split_each_track_valid_mask = update_split_each_track_valid_mask(result)
    #                 need_to_fix_cnt += 1
    #                 return_to_while = True
    #                 break
    #         if return_to_while == True:
    #             break
    #     if need_to_fix_cnt == 0:
    #         break

    return result

def revise_format_for_bbox(curr_tracklet_json, mapping_node_id_to_bbox):
    result_list = [None]*tracklet_len
    distinct_frame_names = []
    for curr_tracklet_json_key in curr_tracklet_json.keys():
        distinct_frame_names += [x for x in curr_tracklet_json[curr_tracklet_json_key]]
    distinct_frame_names = list(set(distinct_frame_names))
    distinct_frame_names.sort()
    complete_frame_names = list(set([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox.keys()]))
    complete_frame_names.sort()
    for complete_frame_name in complete_frame_names:
        result_list[complete_frame_names.index(complete_frame_name)] = {'bbox_list': {}}
    for distinct_frame_name in distinct_frame_names:
        curr_frame_dict = {}
        for curr_tracklet_json_key in curr_tracklet_json.keys():
            if distinct_frame_name in curr_tracklet_json[curr_tracklet_json_key]:
                curr_frame_dict[str(curr_tracklet_json_key)] = curr_tracklet_json[curr_tracklet_json_key][distinct_frame_name]
        result_list[complete_frame_names.index(distinct_frame_name)] = {'bbox_list': curr_frame_dict}
    return result_list

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

def convert_track_to_stitch_format(split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features):
    '''
    evaluate the tracklet importance and interpolate missed detections,delete duplicate ones
    '''
    unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
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

def mapping_tracklet_data_preparation(tracklet_pose_collection,similarity_module,tracklet_inner_cnt,whether_use_reid_similarity_or_not):
    curr_tracklet_input_people, mapping_frameid_bbox_to_features, curr_tracklet_input_people_center_coords = convert_list_dict_to_np(tracklet_pose_collection, 256, 128)# 近10帧
    gc.collect()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=0, abbreviated=False))
    with torch.no_grad():
        features = similarity_module(torch.from_numpy(curr_tracklet_input_people.astype('float32')).cuda()).data.cpu()
    # 计算所有input people的特征向量
    # Tensor(79,512)
    reid_end_time = time.time()
    # mapping_frameid_bbox_to_features 的值替换为bbox的特征
    for mapping_frameid_bbox_to_features_key in mapping_frameid_bbox_to_features.keys():
        mapping_frameid_bbox_to_features[mapping_frameid_bbox_to_features_key] = features.data.numpy()[mapping_frameid_bbox_to_features[mapping_frameid_bbox_to_features_key], :].tolist()
    if dump_switch == 1:
        if not os.path.exists(os.path.join(dump_curr_video_name)):
            os.mkdir(os.path.join(dump_curr_video_name))
            # np.save(os.path.join(dump_curr_video_name, 'features: frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.npy'), features)
        out_file = os.path.join(dump_curr_video_name, 'tracklet_pose_collection_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(tracklet_pose_collection[tracklet_inner_cnt + 1 - tracklet_len: tracklet_inner_cnt + 1], codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'features_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(features.tolist(), codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'mapping_frameid_bbox_to_features_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(mapping_frameid_bbox_to_features, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
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
    for tracklet_inner_idx in range(0, tracklet_len):
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
        person_to_person_matching_matrix_normalized, idx_stride_between_frame_pair, node_id_cnt = compute_inter_person_similarity_worker(parallel_tasks_args_list_item, whether_use_iou_similarity_or_not,whether_use_reid_similarity_or_not)
        # collect the results from multiprocessing and store the matching error between each pair of frames into result_person_to_person_matching_matrix_normalized_collection
        result_person_to_person_matching_matrix_normalized_collection[0:person_to_person_matching_matrix_normalized.shape[0], \
                                                                      0:person_to_person_matching_matrix_normalized.shape[1], \
                                                                      [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0]] = person_to_person_matching_matrix_normalized
    if dump_further_switch == 1:
        if not os.path.exists(os.path.join(dump_curr_video_name)):
            os.mkdir(os.path.join(dump_curr_video_name))
        # '/usr/local/SSP_EM/tracking_for_integration/node_id_cnt_list_frame0to9.json'
        out_file = os.path.join(dump_curr_video_name, 'node_id_cnt_list_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(node_id_cnt_list, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        parallel_tasks_args_list_for_dump = copy.deepcopy(parallel_tasks_args_list)
        for parallel_tasks_args_list_for_dump_idx in range(len(parallel_tasks_args_list_for_dump)):
            parallel_tasks_args_list_for_dump[parallel_tasks_args_list_for_dump_idx][-1] = parallel_tasks_args_list_for_dump[parallel_tasks_args_list_for_dump_idx][-1].data.numpy().tolist() # Tensor转化
        out_file = os.path.join(dump_curr_video_name, 'parallel_tasks_args_list_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(parallel_tasks_args_list_for_dump, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    node_id_cnt = 1
    for tracklet_inner_idx in range(0, tracklet_len):
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
                # introduction
                # Now begin to obtain a matrix describing the matching error between people in two frames, each row corresponds to one person in current frame, each column corresponds to one person in next frame
                # each element of parallel_tasks_args_list corresponds to one frame pair, one element consists of: frame idx of the former frame, starting frame idx of current batch of frames,
                # starting node idx in former frame in a frame pair (each node represents one person in a certain frame), collection of all human bounding boxes in current batch, stride between former and latter frames, an extremely large number,
                # maximum number of people in former frames from all frame pairs, maximum number of people in latter frames from all frame pairs, number of frame pairs for matching in current batch of frames,
                # an integer list of starting node ids in former frame from all frame pairs, a floating array with shape num_people_in_current_batch_of_frames x 512 describing the reid features of all humans in current batch of frames
                # input
                # 0:len(curr_frame_dict['bbox_list']) - from 0 to the number of people in the former frame in frame pair
                # 0:len(next_frame_dict['bbox_list']) - from 0 to the number of people in the latter frame in frame pair
                # [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0] - the channel idx of the matrix describing matching relations between current frame pair in result_person_to_person_matching_matrix_normalized_collection
                # output
                # a floating matrix person_to_person_matching_matrix_normalized describing the matching relations between current batch of frames
                person_to_person_matching_matrix_normalized = result_person_to_person_matching_matrix_normalized_collection[0:len(curr_frame_dict['bbox_list']), 0:len(next_frame_dict['bbox_list']), [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0]]
                ########################################################## begin to prepare data for tracker ###################################################################
                time_start_prepare_costs = time.time()
                # mapping_frameid_bbox_to_features: a dict mapping string 'frameid_bbox coordinates' to a 512-D feature vector, the string 'frameid_bbox coordinates' has length 36, an example is '0930[(467.0, 313.0), (508.0, 424.0)]' where 0930 is frame id, the coordinates are in the format [(left, top), (right, bottom)]
                mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost = prepare_costs_for_tracking_alg(idx_stride_between_frame_pair, curr_frame_dict, tracklet_pose_collection, next_frame_dict, person_to_person_matching_matrix_normalized, node_id_cnt, tracklet_inner_idx, mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost, mapping_frameid_bbox_to_features)
                time_end_prepare_costs = time.time()
        node_id_cnt += len(curr_frame_dict['bbox_list'])
    # time_end = time.time()
    # error_computing_end_time = time.time()
    #print('error computing time: ' + str(error_computing_end_time - error_computing_start_time))
    max_in_mapping_edge_id_to_cost = max([mapping_edge_id_to_cost[x] for x in mapping_edge_id_to_cost]) + 1e-6
    if max_in_mapping_edge_id_to_cost >= 0:
        for mapping_edge_id_to_cost_key in mapping_edge_id_to_cost:
            mapping_edge_id_to_cost[mapping_edge_id_to_cost_key] = mapping_edge_id_to_cost[mapping_edge_id_to_cost_key] - max_in_mapping_edge_id_to_cost
    if dump_further_switch == 1:
        if not os.path.exists(os.path.join(dump_curr_video_name)):
            os.mkdir(os.path.join(dump_curr_video_name))
        out_file = os.path.join(dump_curr_video_name, 'input_to_track_mapping_node_id_to_bbox_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(mapping_node_id_to_bbox, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'input_to_track_mapping_node_id_to_features_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(mapping_node_id_to_features, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'input_to_track_mapping_edge_id_to_cost_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(mapping_edge_id_to_cost, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    return mapping_node_id_to_bbox,mapping_edge_id_to_cost,mapping_node_id_to_features

def mapping_data_preparation(tracklet_pose_collection,similarity_module,tracklet_inner_cnt,whether_use_reid_similarity_or_not):
    curr_tracklet_input_people, mapping_frameid_bbox_to_features, curr_tracklet_input_people_center_coords = convert_list_dict_to_np(tracklet_pose_collection, 256, 128)# 近10帧
    gc.collect()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=0, abbreviated=False))
    with torch.no_grad():
        features = similarity_module(torch.from_numpy(curr_tracklet_input_people.astype('float32')).cuda()).data.cpu()
    # 计算所有input people的特征向量
    # Tensor(79,512)
    reid_end_time = time.time()
    # mapping_frameid_bbox_to_features 的值替换为bbox的特征
    for mapping_frameid_bbox_to_features_key in mapping_frameid_bbox_to_features.keys():
        mapping_frameid_bbox_to_features[mapping_frameid_bbox_to_features_key] = features.data.numpy()[mapping_frameid_bbox_to_features[mapping_frameid_bbox_to_features_key], :].tolist()
    if dump_switch == 1:
        if not os.path.exists(os.path.join(dump_curr_video_name)):
            os.mkdir(os.path.join(dump_curr_video_name))
            # np.save(os.path.join(dump_curr_video_name, 'features: frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.npy'), features)
        out_file = os.path.join(dump_curr_video_name, 'tracklet_pose_collection_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(tracklet_pose_collection[tracklet_inner_cnt + 1 - tracklet_len: tracklet_inner_cnt + 1], codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'features_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(features.tolist(), codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        out_file = os.path.join(dump_curr_video_name, 'mapping_frameid_bbox_to_features_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(mapping_frameid_bbox_to_features, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
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
    for tracklet_inner_idx in range(0, tracklet_len):
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
        person_to_person_matching_matrix_normalized, idx_stride_between_frame_pair, node_id_cnt = compute_inter_person_similarity_worker(parallel_tasks_args_list_item, whether_use_iou_similarity_or_not,whether_use_reid_similarity_or_not)
        # collect the results from multiprocessing and store the matching error between each pair of frames into result_person_to_person_matching_matrix_normalized_collection
        result_person_to_person_matching_matrix_normalized_collection[0:person_to_person_matching_matrix_normalized.shape[0], \
                                                                      0:person_to_person_matching_matrix_normalized.shape[1], \
                                                                      [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0]] = person_to_person_matching_matrix_normalized
    if dump_further_switch == 1:
        if not os.path.exists(os.path.join(dump_curr_video_name)):
            os.mkdir(os.path.join(dump_curr_video_name))
        # '/usr/local/SSP_EM/tracking_for_integration/node_id_cnt_list_frame0to9.json'
        out_file = os.path.join(dump_curr_video_name, 'node_id_cnt_list_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(node_id_cnt_list, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
        parallel_tasks_args_list_for_dump = copy.deepcopy(parallel_tasks_args_list)
        for parallel_tasks_args_list_for_dump_idx in range(len(parallel_tasks_args_list_for_dump)):
            parallel_tasks_args_list_for_dump[parallel_tasks_args_list_for_dump_idx][-1] = parallel_tasks_args_list_for_dump[parallel_tasks_args_list_for_dump_idx][-1].data.numpy().tolist() # Tensor转化
        out_file = os.path.join(dump_curr_video_name, 'parallel_tasks_args_list_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
        json.dump(parallel_tasks_args_list_for_dump, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    node_id_cnt = 1
    for tracklet_inner_idx in range(0, tracklet_len):
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
                # introduction
                # Now begin to obtain a matrix describing the matching error between people in two frames, each row corresponds to one person in current frame, each column corresponds to one person in next frame
                # each element of parallel_tasks_args_list corresponds to one frame pair, one element consists of: frame idx of the former frame, starting frame idx of current batch of frames,
                # starting node idx in former frame in a frame pair (each node represents one person in a certain frame), collection of all human bounding boxes in current batch, stride between former and latter frames, an extremely large number,
                # maximum number of people in former frames from all frame pairs, maximum number of people in latter frames from all frame pairs, number of frame pairs for matching in current batch of frames,
                # an integer list of starting node ids in former frame from all frame pairs, a floating array with shape num_people_in_current_batch_of_frames x 512 describing the reid features of all humans in current batch of frames
                # input
                # 0:len(curr_frame_dict['bbox_list']) - from 0 to the number of people in the former frame in frame pair
                # 0:len(next_frame_dict['bbox_list']) - from 0 to the number of people in the latter frame in frame pair
                # [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0] - the channel idx of the matrix describing matching relations between current frame pair in result_person_to_person_matching_matrix_normalized_collection
                # output
                # a floating matrix person_to_person_matching_matrix_normalized describing the matching relations between current batch of frames
                person_to_person_matching_matrix_normalized = result_person_to_person_matching_matrix_normalized_collection[0:len(curr_frame_dict['bbox_list']), 0:len(next_frame_dict['bbox_list']), [parallel_tasks_args_list.index(x) for x in parallel_tasks_args_list if (x[2]==node_id_cnt and x[4]==idx_stride_between_frame_pair)][0]]
                ########################################################## begin to prepare data for tracker ###################################################################
                time_start_prepare_costs = time.time()
                # mapping_frameid_bbox_to_features: a dict mapping string 'frameid_bbox coordinates' to a 512-D feature vector, the string 'frameid_bbox coordinates' has length 36, an example is '0930[(467.0, 313.0), (508.0, 424.0)]' where 0930 is frame id, the coordinates are in the format [(left, top), (right, bottom)]
                mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost = prepare_costs_for_tracking_alg(idx_stride_between_frame_pair, curr_frame_dict, tracklet_pose_collection, next_frame_dict, person_to_person_matching_matrix_normalized, node_id_cnt, tracklet_inner_idx, mapping_node_id_to_bbox, mapping_node_id_to_features, mapping_edge_id_to_cost, mapping_frameid_bbox_to_features)
                time_end_prepare_costs = time.time()
        node_id_cnt += len(curr_frame_dict['bbox_list'])
    # time_end = time.time()
    # error_computing_end_time = time.time()
    #print('error computing time: ' + str(error_computing_end_time - error_computing_start_time))
    if len(mapping_edge_id_to_cost) > 0:
        max_in_mapping_edge_id_to_cost = max([mapping_edge_id_to_cost[x] for x in mapping_edge_id_to_cost]) + 1e-6
        if max_in_mapping_edge_id_to_cost >= 0:
            for mapping_edge_id_to_cost_key in mapping_edge_id_to_cost:
                mapping_edge_id_to_cost[mapping_edge_id_to_cost_key] = mapping_edge_id_to_cost[mapping_edge_id_to_cost_key] - max_in_mapping_edge_id_to_cost
        if dump_further_switch == 1:
            if not os.path.exists(os.path.join(dump_curr_video_name)):
                os.mkdir(os.path.join(dump_curr_video_name))
            out_file = os.path.join(dump_curr_video_name, 'input_to_track_mapping_node_id_to_bbox_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
            json.dump(mapping_node_id_to_bbox, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
            out_file = os.path.join(dump_curr_video_name, 'input_to_track_mapping_node_id_to_features_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
            json.dump(mapping_node_id_to_features, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
            out_file = os.path.join(dump_curr_video_name, 'input_to_track_mapping_edge_id_to_cost_frame' + str(tracklet_inner_cnt + 1 - tracklet_len) + 'to' + str(tracklet_inner_cnt) + '.json')
            json.dump(mapping_edge_id_to_cost, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True)
    return mapping_node_id_to_bbox,mapping_edge_id_to_cost,mapping_node_id_to_features

def detect(opt,exp,args):
    need_face_recognition_switch = 0
    face_verification_thresh = 0.0
    save_img = False
    out, track_out, source, weights, view_img, save_txt, imgsz, ab_detect_hyp,source = \
        opt['output'], opt['track_out'], opt['source'], opt['weights'], opt['view_img'], opt['save_txt'], opt['img_size'], opt['ab_detect_hyp'],opt['source']
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    if os.path.exists(track_out):
        shutil.rmtree(track_out)
    if os.path.exists(out):
        shutil.rmtree(out)

    similarity_module_cfg = get_default_config()
    similarity_module_cfg.use_gpu = torch.cuda.is_available()
    if opt['similarity_module_config_file']:
        similarity_module_cfg.merge_from_file(opt['similarity_module_config_file'])
    similarity_module_cfg.merge_from_list(opt['opts'])
    # set_random_seed(similarity_module_cfg.train.seed)
    check_cfg(similarity_module_cfg)
    if similarity_module_cfg.use_gpu:
        torch.backends.cudnn.benchmark = True
    similarity_module = torchreid.models.build_model(
        name=similarity_module_cfg.model.name,
        num_classes=751,
        loss=similarity_module_cfg.loss.name,
        pretrained=similarity_module_cfg.model.pretrained,
        use_gpu=similarity_module_cfg.use_gpu
    )
    if similarity_module_cfg.model.load_similarity_weights and check_isfile(
            similarity_module_cfg.model.load_similarity_weights):
        load_pretrained_weights(similarity_module, similarity_module_cfg.model.load_similarity_weights)
    if similarity_module_cfg.use_gpu:
        similarity_module = nn.DataParallel(similarity_module).cuda()
    similarity_module = similarity_module.eval()
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

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
    if not args.experiment_name:
        args.experiment_name = exp.exp_name # 'yolox_s_mix_det'

    output_dir = osp.join(exp.output_dir, args.experiment_name)#exp.output_dir='./YOLOX_outputs
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    rank = args.local_rank

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))
    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    model.cuda()
    model.eval()

    if not args.speed and not args.trt: # True
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else: # '../pretrained/bytetrack_x_mot20.tar'
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank) # cuda:0
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    is_distributed = False  # gpu个数大于1的时候is_distributed为True
    dataloader = exp.get_eval_loader(1, is_distributed, args.test)  # (1,False,False)

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    if args.trt: # False
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()

    # Initialize
    set_logging()
    device = select_device(opt['device'])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size


    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()


    # Set Dataloader
    vid_path, vid_writer = None, None

    # seq_path = '/home/allenyljiang/Documents/Dataset/MOT20'
    # phase = 'train' # 'test'
    # pattern = os.path.join(seq_path,phase,'*','img1')
    # for source in glob.glob(pattern):
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    model.eval()
    if half:  # True
        model = model.half()
    # Get names and colors
    names = ['person']
    # names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    ######################################################### detection #####################################################################
    tracklet_inner_cnt = 0  # tracklet_inner_cnt is an integer indicating the index of the last frame in current batch of frames for tracking, a batch of tracklet_len frames are processed together each time

    tracklet_pose_collection = []
    tracklet_pose_collection_second = []
    tracklet_pose_collection_backup = []
    if not os.path.exists(track_out):
        os.mkdir(track_out)
    if not os.path.exists(out):
        os.mkdir(out)
    ################################################# the above variables need to be cleared for each tracklet ##############################
    tracklet_id = 0
    trajectories_in_all_tracklets = {}
    single_person_low_buffer = []
    single_person_low_buffer_filtered = []
    single_person_high_buffer = []
    single_person_high_buffer_filtered = []
    mini_keys = []
    anomaly_detection_median_filter = {}
    most_updated_json_idx = 0


    # start_file = 260
    # dataset.files = dataset.files[start_file:]
    # dataset.video_flag = dataset.video_flag[start_file:]
    # dataset.nf -= start_file

    # det = open('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/det_tracklet.txt',encoding='utf-8')
    batch_id = 0 # window_id
    base_track_id = 0 # 写数据时的轨迹编号
    #all_video_predicted_tracks = {}
    batch_stride = tracklet_len - 1
    batch_stride_write = tracklet_len - 1
    tracklet_inner_cnt = tracklet_len - 1 # 只有跟踪之后才改变值,9
    current_video_segment_predicted_tracks_backup = {}
    current_video_segment_predicted_tracks_bboxes_backup = {}
    current_video_segment_representative_frames_backup = {}
    total_frames = len(dataloader)
    batch_cnt = math.ceil((total_frames-1)/batch_stride)# 一共47+1个batch，429张图片, the last batch:5 frames
    unmatched_tracks_memory_dict = {} # 记忆轨迹unmatched的次数，超过一定次数之后舍弃轨迹
    # for path, img, im0s, vid_cap in dataset: # check whether in reasonable order
    img_size = exp.test_size # (896,1600)
    data_list = []
    for imgs, _, info_imgs, ids in dataloader:
        # for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
        #     progress_bar(self.dataloader)
        # ): 
        with torch.no_grad():
            gc.collect()
            torch.cuda.empty_cache()

            frame_width,frame_height = info_imgs[1].item(),info_imgs[0].item()
            img_file_name = info_imgs[4]
            imgs = imgs.type(tensor_type)  # 对imgs类型进行转换
            # imgs_byte = torch.load('/home/allenyljiang/Documents/ByteTrack-main/saved_variables/imgs.pt')
            # print(imgs.equal(imgs_byte))

            # Inference
            t1 = time_synchronized()
            # pred = model(img, augment=opt['augment'])[0]
            pred = model(imgs) # (1,10710,6)
            num_classes = 1
            confthre = 0.01
            nmsthre = 0.7
            # byte_pred =  torch.load('/home/allenyljiang/Documents/ByteTrack-main/saved_variables/outputs.pt')
            # print(pred[0].equal(byte_pred))
            pred = postprocess(pred, num_classes, confthre, nmsthre)
            # byte_pred1 = torch.load('/home/allenyljiang/Documents/ByteTrack-main/saved_variables/outputs1.pt')
            # # pred[0].equal(byte_pred) # 比较两个张量是否相等
            # print(pred[0].equal(byte_pred1))# 逐元素判断

        border_x_min, border_x_max = gap, frame_width - gap
        border_y_min, boder_y_max = gap, frame_height - gap
        output_results = convert_to_coco_format(dataloader,pred, info_imgs, ids)  # 当前帧以COCO形式存储的数据
        data_list.extend(output_results)
        # Apply NMS
        # pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms']) # (32,6)
        t2 = time_synchronized()

        # Process detections
        box_detected = []
        head_box_detected = []
        box_confidence_scores = []
        head_box_confidence_scores = []
        tracklet_pose_collection_tmp = {}
        centers = []
        scales = []
        foreignmatter_box_detected = []
        foreignmatter_box_confidence_scores = []

        ### Suppose the 6th and 7th sample in the current tracklet are actually more than temporal_length_thresh_inside_tracklet away in temporal axis,
        ### then we integrate the last 4 samples from the last tracklet and the 1st to 6th samples in the current tracklet into a tracklet
        # global tracklet_pose_collection_large_temporal_stride_buffer
        # if len(tracklet_pose_collection_large_temporal_stride_buffer) > 0:
        #     tracklet_pose_collection.append(tracklet_pose_collection_large_temporal_stride_buffer[0])
        #     tracklet_pose_collection_large_temporal_stride_buffer = []
        #     tracklet_inner_cnt = len(tracklet_pose_collection)
        # 使用自带detector
        # 每次只保存当前batch的结果
        # tracklet_pose_collection = conduct_pose_estimation(webcam, path, out, im0s, pred, img, dataset, save_txt, save_img, view_img, box_detected, head_box_detected, foreignmatter_box_detected, box_confidence_scores, head_box_confidence_scores, foreignmatter_box_confidence_scores, centers, scales, vid_path, vid_writer, vid_cap, tracklet_pose_collection, names, colors, pose_transform, bbox_confidence_threshold, tracklet_inner_cnt, need_face_recognition_switch, face_verification_thresh, opt['iou_thres'])
        ############ tracklet pose collection #########
        tracklet_pose_collection,tracklet_pose_collection_second = tracklet_collection(dataloader,img_size,pred, info_imgs, ids, box_detected, box_confidence_scores, tracklet_pose_collection,tracklet_pose_collection_second, bbox_confidence_threshold, tracklet_inner_cnt,source)
        # tracklet_pose_collection_tmp = json.loads(det.readline())
        # tracklet_pose_collection.append(tracklet_pose_collection_tmp)
        if total_frames <= tracklet_len: # 不足一个batch
            if len(tracklet_pose_collection) < total_frames:
                continue
        if batch_id < batch_cnt - 1: # 中间的batch
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

    # eval_results = evaluate_prediction(dataloader,data_list)
            # tracklet_pose_collection.pop(0)
        # if len(tracklet_pose_collection) > tracklet_len and tracklet_pose_collection[-1] != [] and abs(int(tracklet_pose_collection[-1]['img_dir'].split('/')[-1][:-4]) - int(tracklet_pose_collection[-2]['img_dir'].split('/')[-1][:-4])) >= temporal_length_thresh_inside_tracklet:
        #     tracklet_pose_collection_large_temporal_stride_buffer.append(tracklet_pose_collection[-1])
        #     curr_tracklet_collected_len = len(tracklet_pose_collection[:-1]) - int(len(tracklet_pose_collection[:-1]) - tracklet_len + 1)
        #     tracklet_pose_collection = tracklet_pose_collection[:-1][:(len(tracklet_pose_collection[:-1]) - curr_tracklet_collected_len)] + \
        #                                tracklet_pose_collection[:-1][:(len(tracklet_pose_collection[:-1]) - curr_tracklet_collected_len)][-(tracklet_len - curr_tracklet_collected_len):] + \
        #                                tracklet_pose_collection[:-1][-curr_tracklet_collected_len:]
        #     tracklet_inner_cnt = len(tracklet_pose_collection) - 1
        if tracklet_pose_collection[-1] == []:
            if len([x for x in tracklet_pose_collection if x != []]) == 0: # starting frames are without objects
                tracklet_pose_collection = [x for x in tracklet_pose_collection if x != []]
                tracklet_inner_cnt = 0
                continue
            else: # if middle frames are without detections
                tracklet_pose_collection = [x for x in tracklet_pose_collection if x != []]
                tracklet_inner_cnt = len(tracklet_pose_collection)
                continue
        # if the bboxes in latest 10 frames are below 50% of image and the numbers of bboxes in latest 10 frames are the same
        # global whether_conduct_tracking

        ######################################################## conduct tracking ######################################################################

        if ((tracklet_inner_cnt + 1) >= tracklet_len + batch_id*batch_stride or batch_id == batch_cnt-1) and ({} not in tracklet_pose_collection): # and (whether_conduct_tracking == 1):
            # (batch_id+1)*
            batch_id += 1
            print('batch', batch_id)
            time_start = time.time()
            # allen added to filter out frames without people or without head,进入到下一帧
            if min([len(x['bbox_list']) for x in tracklet_pose_collection]) == 0: # or min([len(x['head_bbox_list']) for x in tracklet_pose_collection[-tracklet_len:]]) == 0:
                tracklet_inner_cnt += 1
                continue
            # if skipped intermediate frames without people
            # if len(tracklet_pose_collection) >= tracklet_len and abs(int(tracklet_pose_collection[-1]['img_dir'].split('/')[-1][:-4]) - int(tracklet_pose_collection[-tracklet_len]['img_dir'].split('/')[-1][:-4])) > tracklet_len - 1:
            #     tracklet_inner_cnt += 1
            #     continue
            ################################################ collect the vectors of bboxes #############################################################
            reid_start_time = time.time()
            # input:
            # tracklet_pose_collection: collection of details in all frames, tracklet_pose_collection is a list, each element of tracklet_pose_collection is a dict:
            # an example of an element of tracklet_pose_collection:
            # {'bbox_list': list of human body bounding boxes, each bbox [(left, top), (right, bottom)], float,
            # 'head_bbox_list': list of head bounding boxes, each bbox [(left, top), (right, bottom)], float,
            # 'box_confidence_scores': list of human body bounding boxes confidence scores, each one is a floating point number,
            # 'target_body_box_coord': the bounding box of target, each bbox [(left, top), (right, bottom)], float,
            # 'img_dir': the directory of current frame, str }
            # curr_tracklet_input_people: number of people in current batch x 3 x 256 x 128
            # output:
            # curr_tracklet_input_people: shape number of people in current batch of frames x 3 x 256 x 128
            # mapping_frameid_bbox_to_features: dict or lut(look up table can be implemented with C++) maps a string (frameid+'[(left, top), (right, bottom)]') to the idx of the bbox in current batch of frames, idx is integer
            # shape: number of people in current batch of frames x 2, 2 denotes horizontal and vertical coordinates of the center of each person, float
            mapping_node_id_to_bbox,mapping_edge_id_to_cost,mapping_node_id_to_features = mapping_data_preparation(tracklet_pose_collection, similarity_module, tracklet_inner_cnt,True)
            ### 需要对第二次ssp当中的tracklet_pose_collection_second进行修正 ###
            ##################################################################################################################################
            ############################ organize graph and call #############################################################################
            # time_start = time.time()
            # if int(mapping_node_id_to_bbox[max([x for x in mapping_node_id_to_bbox])][2][:-4]) == len(dataset.files) - 1 and len(mapping_node_id_to_bbox) == 0 and len(mapping_edge_id_to_cost) == 0:
            if int(mapping_node_id_to_bbox[max([x for x in mapping_node_id_to_bbox])][2][:-4]) == len(dataloader) - 1 and len(mapping_node_id_to_bbox) == 0 and len(mapping_edge_id_to_cost) == 0:
                break
            # time_start = time.time()
            result = tracking(mapping_node_id_to_bbox, mapping_edge_id_to_cost, tracklet_inner_cnt) # tracking函数为ssp算法实现
            
            time_end = time.time()
            # print('SSP computing time: ' + str(time_end - time_start))
            indefinite_node = [] # 表示第一次ssp当中不确定的点
            error_tracks = [] # 第一次ssp当中出错的轨迹段
            split_each_track_SSP, split_each_track_valid_mask  = update_split_each_track_valid_mask(result)
            current_video_segment_predicted_tracks_bboxes_test_SSP,trajectory_node_dict,trajectory_idswitch_dict,trajectory_idswitch_reliability_dict,trajectory_segment_nodes_dict = track_processing(split_each_track_SSP, mapping_node_id_to_bbox, mapping_node_id_to_features,split_each_track_valid_mask)
            n_clusters = 0 # 最大轨迹数目，第二次低置信度点不增加轨迹数目只改变ssp结果
            for track_id in trajectory_idswitch_reliability_dict:
                if len(trajectory_idswitch_reliability_dict[track_id]) > 1: # 发生了idswitch的轨迹都应当加入到error_track当中
                    error_tracks.append(track_id)
                    for i in range(len(trajectory_idswitch_reliability_dict[track_id])):
                        if trajectory_idswitch_reliability_dict[track_id][i] < 2:
                            continue
                        segment_nodes = trajectory_segment_nodes_dict[track_id][i]
                        mean_conf = np.mean([mapping_node_id_to_bbox[node][1] for node in segment_nodes])
                        # if mean_conf > 0.8:
                        n_clusters += 1
                        indefinite_node += segment_nodes

                elif len(trajectory_idswitch_reliability_dict[track_id]) == 1 and len(current_video_segment_predicted_tracks_bboxes_test_SSP[track_id])< tracklet_len:
                    # if trajectory_idswitch_reliability_dict[track_id][0] == 1: # 对于只有一个valid_node的不加入进行修正
                    #     continue
                    indefinite_node += trajectory_node_dict[track_id]
                    n_clusters += 1
                    error_tracks.append(track_id)
            error_tracks = np.unique(error_tracks).tolist()
            unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
            ##### 修正之前纯SSP算法结果 #####
            if args.save_result:
                #### high confidence score ####
                for frame_name in unique_frame_list:
                    curr_img = cv2.imread(os.path.join(source, frame_name))
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
                                cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
                                cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_ssp/')):
                        #os.mkdir(os.path.join(source.split(source.split('/')[-1])[0], 'results', source.split('/')[-1] + '_vis/'))
                        os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_ssp/'))
                    cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_ssp/') + str(tracklet_inner_cnt) + '_' + frame_name, curr_img)

            #### 把indefinite_node 当中的节点加入到 tracklet_pose_collection_second当中 ####
            for node in set(indefinite_node):
                bbox,conf,frame = mapping_node_id_to_bbox[node][0],mapping_node_id_to_bbox[node][1],mapping_node_id_to_bbox[node][2]
                tracklet_pose_collection_second[unique_frame_list.index(frame)]['bbox_list'].append(bbox)
                tracklet_pose_collection_second[unique_frame_list.index(frame)]['box_confidence_scores'].append(conf)
            ##### second ssp #####
            mapping_node_id_to_bbox_second,mapping_edge_id_to_cost_second,mapping_node_id_to_features_second = mapping_data_preparation(tracklet_pose_collection_second, similarity_module, tracklet_inner_cnt,False)
            if len(mapping_edge_id_to_cost_second) > 0:
                ##### 节点数目大于10的情况才能进行ssp #####
                Second_Flag = False
                result_second = tracking(mapping_node_id_to_bbox_second, mapping_edge_id_to_cost_second, tracklet_inner_cnt)
                if 'Predicted tracks' in result_second[0] and len(indefinite_node) > 0:
                # if len(mapping_node_id_to_bbox_second) > 10:
                    split_each_track_SSP_second,split_each_track_valid_mask_second= update_split_each_track_valid_mask(result_second)
                    if args.save_result:
                        ##### 低置信度框 ###
                        for frame_name in unique_frame_list:
                            curr_img = cv2.imread(os.path.join(source, frame_name))
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
                                        cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
                                        cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                            if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_ssp_second/')):
                                #os.mkdir(os.path.join(source.split(source.split('/')[-1])[0], 'results', source.split('/')[-1] + '_vis/'))
                                os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_ssp_second/'))
                            cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_ssp_second/') + str(tracklet_inner_cnt) + '_' + frame_name, curr_img)
                    ## 进行轨迹合并 ##
                    # result = tracks_combination(error_tracks,result_second,mapping_node_id_to_bbox_second, mapping_node_id_to_features_second, source,tracklet_len,n_clusters)
                    result = tracks_combination(error_tracks,result,result_second,mapping_node_id_to_bbox, mapping_node_id_to_bbox_second,mapping_node_id_to_features,mapping_node_id_to_features_second,source,tracklet_len)
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
            split_each_track, _ = update_split_each_track_valid_mask(result)
            if not Second_Flag:
                [split_each_track.pop(track) for track in error_tracks]
                # [valid_mask.pop(track) for track in error_tracks] # valid_mask 并没有使用
            current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_confidence_score, current_video_segment_predicted_tracks_bboxes, current_video_segment_representative_frames,current_video_segment_predicted_tracks_bboxes_test,current_trajectory_similarity_dict,current_video_segment_all_traj_all_object_features = convert_track_to_stitch_format(split_each_track,mapping_node_id_to_bbox,mapping_node_id_to_features)
            # current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, current_video_segment_all_traj_all_object_features, _ = interpolation_fix_missed_detections(current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, current_video_segment_all_traj_all_object_features, tracklet_pose_collection)
            frame_list = np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox])
            curr_last_frame_node_list = [x for x in mapping_node_id_to_bbox if mapping_node_id_to_bbox[x][2] == frame_list[-1]]  # 当前batch
            unique_frame_list = sorted(np.unique([mapping_node_id_to_bbox[x][2] for x in mapping_node_id_to_bbox]))
            if args.save_result:
            ##### 修正之后当前batch的图片写入  ########
                for frame_name in unique_frame_list:
                    curr_img = cv2.imread(os.path.join(source, frame_name))
                    for human_id in current_video_segment_predicted_tracks_bboxes: # 第一帧所有的human_id与轨迹id相同
                        for bbox in current_video_segment_predicted_tracks_bboxes[human_id]: # dict bbox:key
                            if bbox == frame_name: #  写入帧数，轨迹数，坐标
                                left, top = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1])
                                right, bottom = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][1])
                                cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
                                cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 3)
                    if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_fixed/')):
                        os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_fixed/'))
                    cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis_fixed/') + str(tracklet_inner_cnt) + '_' + frame_name, curr_img)
            # if previous_video_segment_predicted_tracks != {}:
            if batch_id > 1:
                curr_first_frame_node_list = [x for x in mapping_node_id_to_bbox if mapping_node_id_to_bbox[x][2] == frame_list[0]]
                node_matching_dict = {}
                # for idx,curr_node in enumerate(curr_first_frame_node_list):  # 对应关系？？？
                #     node_matching_dict[curr_node] = prev_last_frame_node_list[idx]
                # prev_last_frame_node_list , curr_first_frame_node_list
                result_dict,previous_unmatched_tracks,curr_unmatched_tracks = stitching_tracklets(source,mapping_node_id_to_bbox,node_matching_dict,tracklet_inner_cnt, current_video_segment_predicted_tracks, previous_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, previous_video_segment_predicted_tracks_bboxes, current_video_segment_representative_frames, previous_video_segment_representative_frames,current_video_segment_predicted_tracks_bboxes_test,previous_video_segment_predicted_tracks_bboxes_test,unmatched_tracks_memory_dict)
                # result_dict,previous_unmatched_tracks,curr_unmatched_tracks = stitching_tracklets_revised_bidirectional(previous_video_segment_predicted_tracks_bboxes_test,current_video_segment_predicted_tracks_bboxes_test)
                # result_dict,previous_unmatched_tracks,curr_unmatched_tracks = stitching_tracklets_revised(kmedoids_instance,current_video_segment_predicted_tracks_bboxes_test,current_trajectory_similarity_dict,previous_video_segment_predicted_tracks_bboxes_test,previous_trajectory_similarity_dict,node_matching_dict,mapping_node_id_to_features,mapping_node_id_to_bbox)
                # print('curr_unmatched_tracks',curr_unmatched_tracks)
                # print('previous_unmatched_tracks', previous_unmatched_tracks)
                # print('stitch result:',result_dict)
            # for track in [1,2,4,5]:
            #     print(np.mean([cosine_similarity(np.array(current_video_segment_predicted_tracks_bboxes_test[track][node][3]),np.array(track1[549][3])) for node in current_video_segment_predicted_tracks_bboxes_test[track]]))
            # split_each_track, _ = update_split_each_track_valid_mask(result)  # 得到单独每条轨迹
            #     result_dict, current_video_segment_predicted_tracks, current_video_segment_predicted_tracks_backup, current_video_segment_predicted_tracks_bboxes_backup, current_video_segment_representative_frames_backup, current_video_segment_all_traj_all_object_features_backup,_ = \
            # stitching_tracklets(tracklet_inner_cnt, current_video_segment_predicted_tracks, previous_video_segment_predicted_tracks, current_video_segment_predicted_tracks_bboxes, previous_video_segment_predicted_tracks_bboxes, current_video_segment_representative_frames, current_video_segment_all_traj_all_object_features, previous_video_segment_representative_frames, previous_video_segment_all_traj_all_object_features, average_sampling_density_hori_vert)

            start_frame_name = '%03d'%min([int(mapping_node_id_to_bbox[x][2].split('.')[0]) for x in mapping_node_id_to_bbox])+'.jpg' if \
                               min([int(mapping_node_id_to_bbox[x][2].split('.')[0]) for x in mapping_node_id_to_bbox]) < 1000 else \
                               '%03d' % min([int(mapping_node_id_to_bbox[x][2].split('.')[0]) for x in mapping_node_id_to_bbox]) + '.jpg' #???
            # curr_batch_txt = open(os.path.join(source.split(source.split('/')[-1])[0], 'results', source.split('/')[-1] + '_SSP_EM/') + start_frame_name.replace('.jpg', '.txt'), 'a')
            # curr_batch_txt = open('/media/allenyljiang/Seagate_Backup_Plus_Drive/usr/local/VIBE-master/data/neurocomputing/results/mot20_test2/05_0019_SSP_EM/' + '05_0019_' + start_frame_name.replace('.jpg', '.txt'), 'a')
            # #######   整个轨迹txt写入  ########
            # '''
            # 下一个batch写入上一个batch的数据
            # '''
            # # 1、如果是第一个batch则不进行数据写入
            # new_track_id = 0
            # if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all')):
            #     os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all'))
            # if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/')):
            #     os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/'))
            # if batch_id == 1:
            #     current_video_segment_predicted_tracks_backup = copy.deepcopy(current_video_segment_predicted_tracks)
            #     current_video_segment_predicted_tracks_bboxes_backup = copy.deepcopy(current_video_segment_predicted_tracks_bboxes)
            #     current_video_segment_representative_frames_backup = copy.deepcopy(current_video_segment_representative_frames)
            #     current_video_segment_predicted_tracks_bboxes_test_backup = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_test)
            #     current_trajectory_similarity_dict_backup = copy.deepcopy(current_trajectory_similarity_dict)
            #     #all_video_predicted_tracks = copy.deepcopy(current_video_segment_predicted_tracks_bboxes)
            #     total_txt = open(os.path.join(source.split(source.split('/')[-1])[0], 'results_all/') + (source.split('/')[-2]+'.txt'), 'w')
            #     base_track_id += len(split_each_track)  # 给没有匹配上的轨迹起始编号
            #     # for frame_name in unique_frame_list:
            #     # for frame_name in unique_frame_list:
            #     #     # curr_img = cv2.imread(os.path.join(source, frame_name))
            #     #     for human_id in current_video_segment_predicted_tracks_bboxes: # 第一帧所有的human_id与轨迹id相同
            #     #         for bbox in current_video_segment_predicted_tracks_bboxes[human_id]: # dict bbox:key
            #     #             if bbox == frame_name: #  写入帧数，轨迹数，坐标
            #     #                 left, top = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1])
            #     #                 cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 3)
            #         # if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/')):
            #         #     os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/'))
            #         # cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/') + str(tracklet_inner_cnt) + '_' + frame_name, curr_img)
            # else: # 写入上一个batch的数据
            #     total_txt = open(os.path.join(source.split(source.split('/')[-1])[0], 'results_all/') + (source.split('/')[-2]+'.txt'), 'a')
            #     # frame_list = [unique_frame_list[-1]] # int类型不可以迭代
            #     frame_list = unique_frame_list[-batch_stride_write:]
            #     if batch_id == batch_cnt:
            #         frame_list = unique_frame_list[-(total_frames - (batch_id - 1) * batch_stride - 1):]
            #     ### 对上一个batch的数据进行改正 ###
            #     previous_delete_track_dict = {}
            #     for human_id in previous_video_segment_predicted_tracks_bboxes:
            #         track = previous_video_segment_predicted_tracks_bboxes_test[human_id]
            #         track_importance = np.mean([track[node][2] for node in track])
            #         if batch_id == 2:
            #             if track_importance < 0.75 or track_importance*math.sqrt(len(track)) < 2:
            #                 previous_delete_track_dict[human_id] = track_importance
            #                 continue
            #         elif track_importance*math.sqrt(len(track)) < 2:
            #             previous_delete_track_dict[human_id] = track_importance * math.sqrt(len(track))
            #     previous_delete_track_list = list(previous_delete_track_dict.keys())
            #     [previous_video_segment_predicted_tracks_bboxes.pop(trackid) for trackid in previous_delete_track_list]
            #
            #     #### 以下为修正之后汇总的画图以及数据写入 ####
            #     if batch_id != 2: # 第2个batch的时候需要写入的数据是上一个batch的全部
            #         frame_list_iter1 = previous_frame_list[-batch_stride_write:]
            #         base_track_cnt = len(set(curr_unmatched_tracks_dict.keys()).intersection(set(previous_delete_track_list)))
            #         # base_track_id -= base_track_cnt # ???? 减去上一个batch新出现但是错误的
            #     else:
            #         frame_list_iter1 = previous_frame_list
            #     for frame_name in frame_list_iter1:
            #         curr_img = cv2.imread(os.path.join(source, frame_name))
            #         for human_id in previous_video_segment_predicted_tracks_bboxes:
            #             for bbox in previous_video_segment_predicted_tracks_bboxes[human_id]: # dict bbox:key
            #                 if bbox == frame_name: #  写入帧数，轨迹数，坐标
            #                     left, top = int(previous_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]), int(previous_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1])
            #                     cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 3)
            #                     total_txt.write(str(int(frame_name.split('.')[0])) + ',' + str(human_id) + ',' + \
            #                      str(previous_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]) + ',' + \
            #                      str(previous_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1]) + ',' + \
            #                      str(previous_video_segment_predicted_tracks_bboxes[human_id][bbox][1][0]-previous_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]) + ',' + \
            #                      str(previous_video_segment_predicted_tracks_bboxes[human_id][bbox][1][1]-previous_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1]) + ',-1,-1,-1,-1\n') # mot格式：必须10个数
            #         cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/') + str(tracklet_inner_cnt-batch_stride) + '_' + frame_name, curr_img) # 写入前一个batch的时候需要减去batch_stride
            #
            #     # [current_video_segment_representative_frames_backup.pop(track_id) for track_id in previous_delete_track_list]
            #     # [current_video_segment_predicted_tracks_bboxes_test_backup.pop(track_id) for track_id in previous_delete_track_list]
            #     # [current_trajectory_similarity_dict_backup.pop(track_id) for track_id in previous_delete_track_list]
            #     curr_unmatched_tracks_dict = {} #
            #     #### 此处进行id索引的变化 ####
            #     for human_id in current_video_segment_predicted_tracks_bboxes: # track_id,索引仍然按照该batch得到的轨迹来,只是改变写入到总的txt以及画的图当中的结果
            #         if human_id not in result_dict:
            #             # 如果是新增轨迹的话,之后的batch才可能有新增轨迹
            #             new_track_id += 1
            #             human_id_txt = base_track_id + new_track_id # 写入结果中的轨迹id
            #             result_dict[human_id] = human_id_txt
            #             if human_id in curr_unmatched_tracks:
            #                 curr_unmatched_tracks_dict[human_id_txt] = human_id
            #         else:
            #             human_id_txt = result_dict[human_id] # 写入的时候改为前一个batch当中轨迹值
            #         current_video_segment_predicted_tracks_bboxes_backup[human_id_txt] = current_video_segment_predicted_tracks_bboxes[human_id]
            #         current_video_segment_predicted_tracks_backup[human_id_txt] = current_video_segment_predicted_tracks[human_id]
            #         current_video_segment_representative_frames_backup[human_id_txt] = current_video_segment_representative_frames[human_id]
            #         current_video_segment_predicted_tracks_bboxes_test_backup[human_id_txt] = current_video_segment_predicted_tracks_bboxes_test[human_id]
            #         current_trajectory_similarity_dict_backup[human_id_txt] = current_trajectory_similarity_dict[human_id]
            #     # [current_video_segment_predicted_tracks_bboxes_backup.pop(track_id) for track_id in previous_delete_track_list]
            #     # [current_video_segment_predicted_tracks_backup.pop(track_id) for track_id in previous_delete_track_list]
            #     if batch_id == batch_cnt: # 最后一个batch还需要写入当前batch的数据
            #         for frame_name in frame_list:
            #             curr_img = cv2.imread(os.path.join(source, frame_name))
            #             for human_id in current_video_segment_predicted_tracks_bboxes_backup:
            #                 track = current_video_segment_predicted_tracks_bboxes_test_backup[human_id]
            #                 track_importance = np.mean([track[node][2] for node in track])
            #                 if track_importance*math.sqrt(len(track)) < 2:
            #                 # print('delete previous track {0}'.format(human_id))
            #                     continue
            #                 for bbox in current_video_segment_predicted_tracks_bboxes_backup[human_id]:
            #                     if bbox == frame_name: #  写入帧数，轨迹数，坐标
            #                         left, top = int(current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][0][0]), int(current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][0][1])
            #                         cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 3)
            #                         total_txt.write(str(int(frame_name.split('.')[0])) + ',' + str(human_id) + ',' + \
            #                          str(current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][0][0]) + ',' + \
            #                          str(current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][0][1]) + ',' + \
            #                          str(current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][1][0]-current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][0][0]) + ',' + \
            #                          str(current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][1][1]-current_video_segment_predicted_tracks_bboxes_backup[human_id][bbox][0][1]) + ',-1,-1,-1,-1\n') # mot格式：必须10个数
            #             cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/') + str(tracklet_inner_cnt) + '_' + frame_name, curr_img)
            #     total_txt.close()
            #     base_track_id += len(curr_unmatched_tracks) # 更新base_track_id
            #     for unmatch in (set(previous_unmatched_tracks)- set(previous_delete_track_list)): # 保留上一次未匹配的轨迹，每次会有很多重复的
            #         if unmatch not in unmatched_tracks_memory_dict:
            #             unmatched_tracks_memory_dict[unmatch] = 1 # 初始化未配对次数
            #         else:
            #             unmatched_tracks_memory_dict[unmatch] += 1 # 次数+1
            #         if unmatched_tracks_memory_dict[unmatch] >= 2: # 次数 >= 3之后忽略，认为轨迹终止
            #             ###  需要把该轨迹删除掉 ###
            #             if unmatch in current_video_segment_predicted_tracks_bboxes_backup:
            #                 current_video_segment_predicted_tracks_bboxes_backup.pop(unmatch)
            #             if unmatch in current_video_segment_predicted_tracks_backup:
            #                 current_video_segment_predicted_tracks_backup.pop(unmatch)
            #             continue
            #         current_video_segment_predicted_tracks_bboxes_backup[unmatch] = copy.deepcopy(previous_video_segment_predicted_tracks_bboxes[unmatch])
            #         current_video_segment_predicted_tracks_backup[unmatch] = copy.deepcopy(previous_video_segment_predicted_tracks[unmatch])
            # # 更新上一帧的值,注意都是使用backup来进行更新的
            # previous_frame_list = copy.deepcopy(unique_frame_list)
            # previous_video_segment_predicted_tracks = copy.deepcopy(current_video_segment_predicted_tracks_backup)
            # previous_video_segment_predicted_tracks_bboxes = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_backup)
            # previous_video_segment_representative_frames = copy.deepcopy(current_video_segment_representative_frames_backup)
            # previous_video_segment_predicted_tracks_bboxes_test = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_test_backup)
            # previous_trajectory_similarity_dict = copy.deepcopy(current_trajectory_similarity_dict_backup)
            # prev_last_frame_node_list = copy.deepcopy(curr_last_frame_node_list)
            # ## 清理内存
            # current_video_segment_predicted_tracks.clear()
            # current_video_segment_predicted_tracks_bboxes.clear()
            # current_video_segment_representative_frames.clear()
            # current_video_segment_all_traj_all_object_features.clear()
            # current_video_segment_predicted_tracks_bboxes_backup.clear()
            # current_video_segment_predicted_tracks_backup.clear()
            # current_video_segment_representative_frames_backup.clear()
            # current_video_segment_all_traj_all_object_features_backup.clear()

            #######   整个轨迹txt写入  ########
            '''
            按照每个batch来写
            '''
            # previous_unmatched_tracks,curr_unmatched_tracks
            # 如果是第一个batch则写入所有数据
            new_track_id = 0
            # current_video_segment_all_traj_all_object_features_backup = copy.deepcopy(current_video_segment_all_traj_all_object_features)
            if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all')):
                os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all'))
            if batch_id == 1:
                current_video_segment_predicted_tracks_backup = copy.deepcopy(current_video_segment_predicted_tracks)
                current_video_segment_predicted_tracks_bboxes_backup = copy.deepcopy(current_video_segment_predicted_tracks_bboxes)
                current_video_segment_representative_frames_backup = copy.deepcopy(current_video_segment_representative_frames)
                current_video_segment_predicted_tracks_bboxes_test_backup = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_test)
                current_trajectory_similarity_dict_backup = copy.deepcopy(current_trajectory_similarity_dict)
                #all_video_predicted_tracks = copy.deepcopy(current_video_segment_predicted_tracks_bboxes)
                total_txt = open(os.path.join(source.split(source.split('/')[-1])[0], 'results_all/') + (source.split('/')[-2]+'.txt'), 'w')
                base_track_id += len(split_each_track)  # 给没有匹配上的轨迹起始编号
                # frame_list = copy.deepcopy(unique_frame_list)
                # curr_batch_txt = open(os.path.join(source.split(source.split('/')[-1])[0], 'results',source.split('/')[-1] + '_SSP_EM/') + start_frame_name.replace('.jpg', '.txt'), 'r')
                # tracks = curr_batch_txt.readlines()
                # total_txt.writelines(tracks)
                # curr_batch_txt.close()
                # total_txt.close()
                for frame_name in unique_frame_list:
                    # curr_img = cv2.imread(os.path.join(source, frame_name))
                    for human_id in current_video_segment_predicted_tracks_bboxes: # 第一帧所有的human_id与轨迹id相同
                        for bbox in current_video_segment_predicted_tracks_bboxes[human_id]: # dict bbox:key
                            if bbox == frame_name: #  写入帧数，轨迹数，坐标
                                total_txt.write(str(int(frame_name.split('.')[0])) + ',' + str(human_id) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1]) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][0]-current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][1]-current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1]) + ',-1,-1,-1,-1\n') # mot格式：必须10个数
                    #             left, top = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1])
                    #             right, bottom = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][1])
                    #             cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
                    #             cv2.putText(curr_img, str(human_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 3)
                    # if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/')):
                    #     os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/'))
                    # cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/') + str(tracklet_inner_cnt) + '_' + frame_name, curr_img)

                total_txt.close()
            else:
                total_txt = open(os.path.join(source.split(source.split('/')[-1])[0], 'results_all/') + (source.split('/')[-2]+'.txt'), 'a')
                # frame_list = [unique_frame_list[-1]] # int类型不可以迭代
                frame_list = unique_frame_list[-batch_stride_write:] # int类型不可以迭代
                if batch_id == batch_cnt:
                    frame_list = unique_frame_list[-(total_frames - (batch_id-1)*batch_stride-1):]
                for frame_name in frame_list: # 只写入最后一帧,根据步长决定
                    # curr_img = cv2.imread(os.path.join(source, frame_name))
                    for human_id in current_video_segment_predicted_tracks_bboxes: # track_id,索引仍然按照该batch得到的轨迹来,只是改变写入到总的txt以及画的图当中的结果
                        if human_id not in result_dict:
                            # 如果是新增轨迹的话,之后的batch才可能有新增轨迹
                            new_track_id += 1
                            human_id_txt = base_track_id + new_track_id # 写入结果中的轨迹id
                            result_dict[human_id] = human_id_txt
                        else:
                            human_id_txt = result_dict[human_id] # 写入的时候改为前一个batch当中轨迹值
                        for bbox in current_video_segment_predicted_tracks_bboxes[human_id]: # dict bbox:key
                            if bbox == frame_name: #  写入帧数，轨迹数，坐标
                                total_txt.write(str(int(frame_name.split('.')[0])) + ',' + str(human_id_txt) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1]) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][0]-current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]) + ',' + \
                                                     str(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][1]-current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1]) + ',-1,-1,-1,-1\n') # mot格式：必须10个数
                                # left, top = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][0][1])
                                # right, bottom = int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][0]), int(current_video_segment_predicted_tracks_bboxes[human_id][bbox][1][1])
                                # cv2.rectangle(curr_img, (left, top), (right, bottom), (0, 255, 0), 3)
                                # cv2.putText(curr_img, str(human_id_txt), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 3)
                        current_video_segment_predicted_tracks_bboxes_backup[human_id_txt] = current_video_segment_predicted_tracks_bboxes[human_id]
                        current_video_segment_predicted_tracks_backup[human_id_txt] = current_video_segment_predicted_tracks[human_id]
                        current_video_segment_representative_frames_backup[human_id_txt] = current_video_segment_representative_frames[human_id]
                        current_video_segment_predicted_tracks_bboxes_test_backup[human_id_txt] = current_video_segment_predicted_tracks_bboxes_test[human_id]
                        current_trajectory_similarity_dict_backup[human_id_txt] = current_trajectory_similarity_dict[human_id]

                    # if not os.path.exists(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/')):
                    #     os.makedirs(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/'))
                    # cv2.imwrite(os.path.join(source.split(source.split('/')[-1])[0], 'results_all', source.split('/')[-1] + '_vis/') + str(tracklet_inner_cnt) + '_' + frame_name, curr_img)

                total_txt.close()
                base_track_id += len(curr_unmatched_tracks) # 更新base_track_id

                for unmatch in previous_unmatched_tracks: # 保留上一次未匹配的轨迹，每次会有很多重复的
                    if unmatch not in unmatched_tracks_memory_dict:
                        unmatched_tracks_memory_dict[unmatch] = 1 # 初始化未配对次数
                    else:
                        unmatched_tracks_memory_dict[unmatch] += 1 # 次数+1
                    if unmatched_tracks_memory_dict[unmatch] >= 10: # 次数 >= 3之后忽略，认为轨迹终止
                        continue
                    current_video_segment_predicted_tracks_bboxes_backup[unmatch] = copy.deepcopy(previous_video_segment_predicted_tracks_bboxes[unmatch])
                    current_video_segment_predicted_tracks_backup[unmatch] = copy.deepcopy(previous_video_segment_predicted_tracks[unmatch])

            # current_video_segment_predicted_tracks_bboxes = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_backup)
            # current_video_segment_predicted_tracks = copy.deepcopy(current_video_segment_predicted_tracks_backup)
            # current_video_segment_representative_frames = copy.deepcopy(current_video_segment_representative_frames_backup)
            # current_video_segment_all_traj_all_object_features = copy.deepcopy(current_video_segment_all_traj_all_object_features_backup)
            # 更新上一帧的值
            previous_video_segment_predicted_tracks = copy.deepcopy(current_video_segment_predicted_tracks_backup)
            previous_video_segment_predicted_tracks_bboxes = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_backup)
            previous_video_segment_representative_frames = copy.deepcopy(current_video_segment_representative_frames_backup)
            previous_video_segment_predicted_tracks_bboxes_test = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_test_backup)
            previous_trajectory_similarity_dict = copy.deepcopy(current_trajectory_similarity_dict_backup)

            prev_last_frame_node_list = copy.deepcopy(curr_last_frame_node_list)
            # previous_video_segment_predicted_tracks_bboxes_test = copy.deepcopy(current_video_segment_predicted_tracks_bboxes_test)
            # previous_video_segment_all_traj_all_object_features = copy.deepcopy(current_video_segment_all_traj_all_object_features_backup)
            ## 清理内存
            current_video_segment_predicted_tracks.clear()
            current_video_segment_predicted_tracks_bboxes.clear()
            current_video_segment_representative_frames.clear()
            current_video_segment_all_traj_all_object_features.clear()
            current_video_segment_predicted_tracks_bboxes_backup.clear()
            current_video_segment_predicted_tracks_backup.clear()
            current_video_segment_representative_frames_backup.clear()
            current_video_segment_all_traj_all_object_features_backup.clear()

        tracklet_inner_cnt += batch_stride  # 步长
            # print('after a batch')
            # objgraph.most_common_types(limit=50)
            # objgraph.show_growth()
            # print(str(time.time()))

def parse_opt():
    parser = argparse.ArgumentParser()#　照片要jpg格式
    # /usr/local/SSP_EM/05_0019
    # /home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1
    # parser.add_argument('--source', type=str, default='/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-03/img_test', help='file/dir/URL/glob, 0 for webcam')#/media/allenyljiang/Seagate_Backup_Plus_Drive/usr/local/VIBE-master/data/neurocomputing/05_0019

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    input_opt = parse_opt()
    opt = {}
    opt['name'] = 'yolox-s'
    exp_file = None
    exp = get_exp(exp_file,opt['name'])
    opt['cfg'] = r'/usr/local/lpn-pytorch-master/lpn-pytorch-master/experiments/coco/lpn/lpn101_256x192_gd256x2_gc.yaml'
    opt['source'] = '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-03/img1' # r'/media/allenyljiang/Seagate_Backup_Plus_Drive/usr/local/VIBE-master/data/neurocomputing/05_0019'
    # opt['source'] = input_opt.source # r'/media/allenyljiang/Seagate_Backup_Plus_Drive/usr/local/VIBE-master/data/neurocomputing/05_0019'
    opt['modelDir'] = ''
    opt['logDir'] = ''
    opt['weights'] = ['weights/bytetrack_x_mot20.tar']
    opt['ab_detect_hyp'] = ''
    opt['output'] = 'cache/'
    opt['track_out'] = 'cache/'
    opt['img_size'] = 960
    opt['conf_thres'] = 0.25
    opt['iou_thres'] = 0.4
    opt['device'] = '0'#'1'
    opt['view_img'] = False
    opt['save_txt'] = False
    opt['agnostic_nms'] = False
    opt['augment'] = False
    opt['update'] = False
    opt['classes'] = None
    opt['similarity_module_config_file'] = r'similarity_module/configs/im_osnet_x0_25_softmax_256x128_amsgrad.yaml'
    opt['show_anomaly_or_foreignmatter_or_tracking'] = 0
    opt['opts'] = ['model.load_similarity_weights',
                   r'similarity_module/model/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', \
                   'test.evaluate_similarity_module', 'True']
    args = make_parser().parse_args()#解析参数
    # args.save_result = True
    exp = get_exp(args.exp_file, args.name)# 模型参数（键值对形式）

    # seq_path = '/home/allenyljiang/Documents/Dataset/MOT20'
    # phase = 'train' # 'test'
    # pattern = os.path.join(seq_path,phase,'*','img1')
    # for source in glob.glob(pattern):
    #     opt['source'] = source
    #     detect(opt,exp,args)

    detect(opt,exp,args)


    # tracemalloc.start(25)
    # snapshot = tracemalloc.take_snapshot()
    # # global snapshot
    # gc.collect()
    # snapshot1 = tracemalloc.take_snapshot()
    # top_stats = snapshot1.compare_to(snapshot, 'lineno')
    # logger.warning("[ Top 20 differences ]")
    # for stat in top_stats[:20]:
    #     if stat.size_diff < 0:
    #         continue
    #     logger.warning(stat)
    # snapshot = tracemalloc.take_snapshot()
    # /home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1

    # seq_path = '/home/allenyljiang/Documents/Dataset/MOT20'
    # phase = 'test' # 'test'
    # pattern = os.path.join(seq_path,phase,'*','img1')
    # for source in glob.glob(pattern):
    #     opt['source'] = source
    #     detect(opt)

    #snapshot = tracemalloc.take_snapshot()
    # tracemalloc_snapshot(snapshot)
    # snapshot = tracemalloc.take_snapshot() #  快照,当前内存分配
    # # top_stats = snapshot.statistics('lineno') # 分开统计
    # top_stats = snapshot.statistics('MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref.py') # 统计整个文件内存vun
    # for stat in top_stats:
    #     print(stat)
# 运行时间分析
# pyinstrument -r html MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref_v7.py
# 内存监测
#  python -m memory_profiler MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref.py
'''
mprof run --multiprocess MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref.py
mprof plot
python -m memory_profiler --pdb-mmem=2000 MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref_v7.py
'''
