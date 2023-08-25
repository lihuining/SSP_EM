import copy
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
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
folder = 'MOT20-01'
gt_txt = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train', folder, 'gt/gt.txt')  # 检测结果的txt文件
# 存在以‘’/’’开始的参数，从最后一个以”/”开头的参数开始拼接，之前的参数全部丢弃。
img_list = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train', folder, 'img1') # 图像数据集路径
img_blob = sorted(os.listdir(img_list),key = lambda x: int(x.split('.')[0])) #
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

def cosine_similarity(vec1,vec2):
    num = float(np.dot(vec1, vec2.T))
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos = num / denom  # cosine similarity
    return cos
opt = {}
opt['cfg'] = r'/usr/local/lpn-pytorch-master/lpn-pytorch-master/experiments/coco/lpn/lpn101_256x192_gd256x2_gc.yaml'
opt['similarity_module_config_file'] = r'similarity_module/configs/im_osnet_x0_25_softmax_256x128_amsgrad.yaml'
opt['opts'] = ['model.load_similarity_weights',
               r'similarity_module/model/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', \
               'test.evaluate_similarity_module', 'True']
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
# 相邻帧的reid相似度
reid_similarity_same_person = []
reid_similarity_diff_person = []
for frame in range(int(seq_tracks[:, 0].max())):
    img = cv2.imread(os.path.join(img_list, img_blob[frame]))
    img_dst = os.path.join(dst_path, img_blob[frame])
    track_id = seq_tracks[seq_tracks[:, 0] == (frame+1), 1]  # 轨迹id
    dets = seq_tracks[seq_tracks[:, 0] == (frame+1), 2:6]  # 检测框
    visibility = seq_tracks[seq_tracks[:, 0] == (frame+1), -1]  # visibility
    dets[:, 2:4] += dets[:, 0:2] # (x1y1x2y2)
    result_array = np.zeros((len(dets), 3, fixed_height, fixed_width)).astype('uint8')
    curr_tracklet_bbox_cnt = 0
    #### 把每个img成为（256,128）大小 ####
    curr_crop = img
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
        features = similarity_module(torch.from_numpy(result_array.astype('float32')).cuda()).data.cpu()
        # reid_matrix = cosine_similarity(np.array(mapping_node_id_to_features[node_id_pre]),np.array(mapping_node_id_to_features[int(node_id)]))
    visibility_pre = copy.deepcopy(visibility)
    frame += 1
