'''
neighbor frames reid similarity between same or different people
'''
import copy
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref_v4 import compute_iou_single_box, \
    compute_iou_between_bbox_list,cosine_similarity
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from similarity_module import torchreid
from similarity_module.torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)
from similarity_module.scripts.default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)
from fast_reid.fast_reid_interfece import FastReIDInterface
folder = 'MOT20-01'
gt_txt = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train', folder, 'gt/gt.txt')  # 前后不需要加/
# 存在以‘’/’’开始的参数，从最后一个以”/”开头的参数开始拼接，之前的参数全部丢弃。
img_list = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train', folder, 'img1')
img_blob = sorted(os.listdir(img_list),key = lambda x: int(x.split('.')[0]))
seq_tracks = np.loadtxt(gt_txt, delimiter=',')  # 注意加上delimiter
seq_tracks_valid_part3 = seq_tracks[seq_tracks[:, -2] == 7, :]  # static_person
seq_tracks = seq_tracks[seq_tracks[:, -3] == 1, :]  # pedestrain
classes = np.unique(seq_tracks[:, -2])  # the considered classes in gt

dst_path = os.path.join('/home/allenyljiang/Documents/SSP_EM/statistic_information')
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

num_person_per_frame = []
max_iou_between_people = []
fixed_height, fixed_width = 256,128
# reorganize all people in current batch of frames into a tensor for reidentitification
## similarity module import ##
def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


fast_reid_config = '/home/allenyljiang/Documents/SSP_EM/fast_reid/configs/MOT17/sbs_S50.yml'
fast_reid_weights = "/home/allenyljiang/Documents/SSP_EM/pretrained/mot17_sbs_S50.pth"
device = 'cuda'
encoder =  FastReIDInterface(fast_reid_config, fast_reid_weights, device)

# 相邻帧的reid相似度
conf_thresh = 0.9
reid_similarity_same_person = []
reid_similarity_diff_person = []
for frame in range(int(seq_tracks[:, 0].max())):
    img = cv2.imread(os.path.join(img_list, img_blob[frame]))
    img_dst = os.path.join(dst_path, img_blob[frame])
    track_id = seq_tracks[seq_tracks[:, 0] == (frame+1), 1]  # 轨迹id
    dets = seq_tracks[seq_tracks[:, 0] == (frame+1), 2:6]  # 检测框
    visibility = seq_tracks[seq_tracks[:, 0] == (frame+1), -1]  # visibility
    dets[:, 2:4] += dets[:, 0:2] # (xiy1x2y2)
    result_array = np.zeros((len(dets), 3, fixed_height, fixed_width)).astype('uint8')
    curr_tracklet_bbox_cnt = 0
    for idx,bbox in enumerate(dets):
        curr_crop = img[max(0,int(bbox[1])):min(int(bbox[3]),img.shape[0]), max(0,int(bbox[0])):min(int(bbox[2]),img.shape[1]), :]
        if curr_crop.shape[0] > curr_crop.shape[1] * 2:
            curr_img_resized = cv2.resize(curr_crop, (fixed_width, int(fixed_width / curr_crop.shape[1] * curr_crop.shape[0])), interpolation=cv2.INTER_AREA)
            curr_img_resized = curr_img_resized[int((curr_img_resized.shape[0] - fixed_height) / 2):int((curr_img_resized.shape[0] - fixed_height) / 2) + fixed_height, :, :]
        elif curr_crop.shape[0] < curr_crop.shape[1] * 2:
            curr_img_resized = cv2.resize(curr_crop, (int(fixed_height / curr_crop.shape[0] * curr_crop.shape[1]), fixed_height), interpolation=cv2.INTER_AREA)
            curr_img_resized = curr_img_resized[:, int((curr_img_resized.shape[1] - fixed_width) / 2):int((curr_img_resized.shape[1] - fixed_width) / 2) + fixed_width, :]
        else:
            curr_img_resized = cv2.resize(curr_crop, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
        result_array[curr_tracklet_bbox_cnt, :, :, :] = np.transpose(curr_img_resized, (2,0,1))
        curr_tracklet_bbox_cnt += 1
    with torch.no_grad():
        # features = similarity_module(torch.from_numpy(result_array.astype('float32')).cuda()).data.cpu()
        features = encoder.inference(img, dets)
    
    if frame > 0:
        for idx1, fea1 in enumerate(features_pre):
            for idx2,fea2 in enumerate(features):
                vis1 = visibility_pre[idx1]
                vis2 = visibility[idx2]
                if vis1 < conf_thresh or vis2 < conf_thresh:
                    continue
                if track_id_pre[idx1] == track_id[idx2]:
                    reid_similarity_same_person.append(cosine_similarity(fea1,fea2))
                else:
                    reid_similarity_diff_person.append(cosine_similarity(fea1,fea2))
        # reid_matrix = cosine_similarity(np.array(mapping_node_id_to_features[node_id_pre]),np.array(mapping_node_id_to_features[int(node_id)]))
    track_id_pre = copy.deepcopy(track_id)
    features_pre = copy.deepcopy(features)
    visibility_pre = copy.deepcopy(visibility)
    frame += 1
print('max similarity between different people',max(reid_similarity_diff_person))
print('min similarity between same people',min(reid_similarity_same_person))
plt.figure()
plt.subplot(1,2,1)
plt.hist(reid_similarity_same_person,label='same traj')
plt.subplot(1,2,2)
plt.hist(reid_similarity_diff_person,label='diff traj')
plt.savefig(os.path.join(dst_path,str(conf_thresh)+'reid.png'))
plt.show()

plt.figure()
plt.hist(reid_similarity_same_person,label= 'same traj')
plt.hist(reid_similarity_diff_person,label= 'diff traj')
plt.savefig(os.path.join(dst_path,'reid_all.png'))
plt.show()
