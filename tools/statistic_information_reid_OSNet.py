'''
neighbor frames reid similarity between same or different people
'''
import copy
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
# print(sys.path)
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
gt_txt = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train', folder, 'gt/gt.txt')  # 前后不需要加/
# 存在以‘’/’’开始的参数，从最后一个以”/”开头的参数开始拼接，之前的参数全部丢弃。
img_list = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train', folder, 'img1')
img_blob = sorted(os.listdir(img_list),key = lambda x: int(x.split('.')[0]))
seq_tracks = np.loadtxt(gt_txt, delimiter=',')  # 注意加上delimiter
seq_tracks_valid_part3 = seq_tracks[seq_tracks[:, -2] == 7, :]  # static_person
seq_tracks = seq_tracks[seq_tracks[:, -3] == 1, :]  # pedestrain
classes = np.unique(seq_tracks[:, -2])  # the considered classes in gt
unique_track_id = np.unique(seq_tracks[:, 1])  # 总的轨迹数目
unique_frame_id = np.unique(seq_tracks[:,0]) # 总的frame数目
print(len(unique_track_id))
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

def appearance_update(alpha,track_all_test,det_thresh):
    '''
    alpha = 0.95
    track_all_test:previous_video_segment_predicted_tracks_bboxes_test
    det_thresh = 0.4
    '''
    video_feature = {}
    for track in track_all_test:
        for node in track_all_test[track]:
            feat = np.array(track_all_test[track][node][2]) # 需要转化为array才能相乘
            conf = np.array(track_all_test[track][node][3])
            if track not in video_feature:
                video_feature[track] = feat
                continue
            trust = (conf - det_thresh) / (1-det_thresh)
            if trust < 0: # 如果trust<0 则不进行更新
                continue
            det_alpha = alpha + (1-alpha) * (1-trust)
            update_emb = det_alpha * video_feature[track] + (1-det_alpha)*feat # self.emb = alpha * self.emb + (1 - alpha) * emb
            update_emb /= np.linalg.norm(update_emb)
            video_feature[track] = update_emb
    return video_feature
def det_resized(track_frames,dets,fixed_height,fixed_width):
    '''
    准备输入similarity module的数据，格式为[batch,channel,height,width]
    '''
    curr_tracklet_bbox_cnt = 0
    result_array = np.zeros((len(dets), 3, fixed_height, fixed_width)).astype('uint8')
    for idx,frame in enumerate(track_frames):
        img = cv2.imread('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1/'+ '%06d' %(frame)+".jpg")
        bbox = dets[idx]
        curr_crop = img[max(0, int(bbox[1])):min(int(bbox[3]), img.shape[0]),
                    max(0, int(bbox[0])):min(int(bbox[2]), img.shape[1]), :]
        #cv2.imshow('win', curr_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if curr_crop.shape[0] > curr_crop.shape[1] * 2:
            curr_img_resized = cv2.resize(curr_crop,
                                          (fixed_width, int(fixed_width / curr_crop.shape[1] * curr_crop.shape[0])),
                                          interpolation=cv2.INTER_AREA)
            curr_img_resized = curr_img_resized[int((curr_img_resized.shape[0] - fixed_height) / 2):int(
                (curr_img_resized.shape[0] - fixed_height) / 2) + fixed_height, :, :]
        elif curr_crop.shape[0] < curr_crop.shape[1] * 2:
            curr_img_resized = cv2.resize(curr_crop,
                                          (int(fixed_height / curr_crop.shape[0] * curr_crop.shape[1]), fixed_height),
                                          interpolation=cv2.INTER_AREA)
            curr_img_resized = curr_img_resized[:, int((curr_img_resized.shape[1] - fixed_width) / 2):int(
                (curr_img_resized.shape[1] - fixed_width) / 2) + fixed_width, :]
        else:
            curr_img_resized = cv2.resize(curr_crop, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
        plt.figure()
        crop = cv2.cvtColor(curr_crop,cv2.COLOR_BGR2RGB)
        resize = cv2.cvtColor(curr_img_resized,cv2.COLOR_BGR2RGB)
        plt.subplot(1,2,1)
        plt.imshow(crop)
        plt.subplot(1,2,2)
        plt.imshow(resize)
        plt.show()

        result_array[curr_tracklet_bbox_cnt, :, :, :] = np.transpose(curr_img_resized, (2, 0, 1))
        curr_tracklet_bbox_cnt += 1
    return result_array
            
opt = {}
opt['cfg'] = r'/usr/local/lpn-pytorch-master/lpn-pytorch-master/experiments/coco/lpn/lpn101_256x192_gd256x2_gc.yaml'
opt['similarity_module_config_file'] = r'../similarity_module/configs/im_osnet_x0_25_softmax_256x128_amsgrad.yaml'
opt['opts'] = ['model.load_similarity_weights',
               r'../similarity_module/model/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', \
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
################### 相邻帧的reid相似度 ###################
# conf_thresh = 0.9
# reid_similarity_same_person = []
# reid_similarity_diff_person = []
# for frame in range(int(seq_tracks[:, 0].max())):
#     img = cv2.imread(os.path.join(img_list, img_blob[frame]))
#     img_dst = os.path.join(dst_path, img_blob[frame])
#     track_id = seq_tracks[seq_tracks[:, 0] == (frame+1), 1]  # 轨迹id
#     dets = seq_tracks[seq_tracks[:, 0] == (frame+1), 2:6]  # 检测框
#     visibility = seq_tracks[seq_tracks[:, 0] == (frame+1), -1]  # visibility
#     dets[:, 2:4] += dets[:, 0:2] # (xiy1x2y2)
#     result_array = np.zeros((len(dets), 3, fixed_height, fixed_width)).astype('uint8')
#     curr_tracklet_bbox_cnt = 0
#     for idx,bbox in enumerate(dets):
#         curr_crop = img[max(0,int(bbox[1])):min(int(bbox[3]),img.shape[0]), max(0,int(bbox[0])):min(int(bbox[2]),img.shape[1]), :]
#         if curr_crop.shape[0] > curr_crop.shape[1] * 2:
#             curr_img_resized = cv2.resize(curr_crop, (fixed_width, int(fixed_width / curr_crop.shape[1] * curr_crop.shape[0])), interpolation=cv2.INTER_AREA)
#             curr_img_resized = curr_img_resized[int((curr_img_resized.shape[0] - fixed_height) / 2):int((curr_img_resized.shape[0] - fixed_height) / 2) + fixed_height, :, :]
#         elif curr_crop.shape[0] < curr_crop.shape[1] * 2:
#             curr_img_resized = cv2.resize(curr_crop, (int(fixed_height / curr_crop.shape[0] * curr_crop.shape[1]), fixed_height), interpolation=cv2.INTER_AREA)
#             curr_img_resized = curr_img_resized[:, int((curr_img_resized.shape[1] - fixed_width) / 2):int((curr_img_resized.shape[1] - fixed_width) / 2) + fixed_width, :]
#         else:
#             curr_img_resized = cv2.resize(curr_crop, (fixed_width, fixed_height), interpolation=cv2.INTER_AREA)
#         result_array[curr_tracklet_bbox_cnt, :, :, :] = np.transpose(curr_img_resized, (2,0,1))
#         curr_tracklet_bbox_cnt += 1
#     with torch.no_grad():
#         features = similarity_module(torch.from_numpy(result_array.astype('float32')).cuda()).data.cpu()
#
#     if frame > 0:
#         for idx1, fea1 in enumerate(features_pre):
#             for idx2,fea2 in enumerate(features):
#                 vis1 = visibility_pre[idx1]
#                 vis2 = visibility[idx2]
#                 if vis1 < conf_thresh or vis2 < conf_thresh:
#                     continue
#                 if track_id_pre[idx1] == track_id[idx2]:
#                     reid_similarity_same_person.append(cosine_similarity(fea1,fea2))
#                 else:
#                     reid_similarity_diff_person.append(cosine_similarity(fea1,fea2))
#         # reid_matrix = cosine_similarity(np.array(mapping_node_id_to_features[node_id_pre]),np.array(mapping_node_id_to_features[int(node_id)]))
#     track_id_pre = copy.deepcopy(track_id)
#     features_pre = copy.deepcopy(features)
#     visibility_pre = copy.deepcopy(visibility)
#     frame += 1
# print('max similarity between different people',max(reid_similarity_diff_person))
# print('min similarity between same people',min(reid_similarity_same_person))
# plt.figure()
# plt.subplot(1,2,1)
# plt.hist(reid_similarity_same_person,label='same traj')
# plt.subplot(1,2,2)
# plt.hist(reid_similarity_diff_person,label='diff traj')
# plt.savefig(os.path.join(dst_path,str(conf_thresh)+'reid.png'))
# plt.show()
#
# plt.figure()
# plt.hist(reid_similarity_same_person,label= 'same traj')
# plt.hist(reid_similarity_diff_person,label= 'diff traj')
# plt.savefig(os.path.join(dst_path,'reid_all.png'))
# plt.show()
#### 前后10帧同一条轨迹相似度 ####
'''
0           1       2   3   4     5      6     7     8
frame_id track_id left top width height flag   cls visibility
'''
batch_cnt = int(len(unique_frame_id) / 10)
prev_batch_tracklet_test = {}
same_traj_reid_similarity = []
diff_traj_reid_similarity = []
for batch in range(batch_cnt):
    frames = unique_frame_id[batch:(batch+1)*10]
    indices = np.where((seq_tracks[:,0] >= frames[0]) & (seq_tracks[:,0] <= frames[-1]))
    curr_batch_tracks = np.squeeze(seq_tracks[indices,:])
    curr_batch_tracks[:,4:6] += curr_batch_tracks[:,2:4]
    # curr_batch_dets = curr_batch_tracks[:,2:6]
    # curr_batch_dets[:, 2:4] += curr_batch_dets[:, 0:2]  # left,top,right,bottom--top,bottom,left,right
    visibility = curr_batch_tracks[:, -1] # 最后一维表示可见程度
    tracks = np.unique(curr_batch_tracks[:,1])
    curr_batch_tracklet_test = {} # 保存10帧内所有轨迹的信息
    for track_id in tracks:
        curr_tracklet_test = {}
        dets = curr_batch_tracks[curr_batch_tracks[:,1] == track_id,2:6] # 当前10帧内为该track id bbox
        nodes_length = len(dets)
        visibility = curr_batch_tracks[curr_batch_tracks[:,1] == track_id,-1]
        track_frames = curr_batch_tracks[curr_batch_tracks[:,1] == track_id,0] # 当前轨迹所在的frames
        det_fixed = det_resized(track_frames,dets, fixed_height, fixed_width)
        with torch.no_grad():
            features = similarity_module(torch.from_numpy(det_fixed.astype('float32')).cuda()).data.cpu()
        for idx,node in enumerate(list(range(nodes_length))):
            curr_tracklet_test[node] = [frames[idx], dets[idx],features.data.numpy()[idx,:],visibility[idx]]
        curr_batch_tracklet_test[track_id] = copy.deepcopy(curr_tracklet_test)
    if batch > 0:
        similarity_reid_matrix = np.zeros((len(prev_batch_tracklet_test),len(curr_batch_tracklet_test)))
        prev_track_list = list(prev_batch_tracklet_test.keys())
        curr_track_list = list(curr_batch_tracklet_test.keys())
        previous_video_feature = appearance_update(0.95,prev_batch_tracklet_test,0.4)
        current_video_feature = appearance_update(0.95,curr_batch_tracklet_test,0.4)
        for previous_track_id in prev_batch_tracklet_test:
            previous_track_feature = previous_video_feature[previous_track_id]
            for current_track_id in curr_batch_tracklet_test:
                current_track_feature = current_video_feature[current_track_id]
                reid_similarity = cosine_similarity(previous_track_feature,current_track_feature)
                similarity_reid_matrix[prev_track_list.index(previous_track_id),curr_track_list.index(current_track_id)] = reid_similarity
                if previous_track_id == current_track_id:
                    same_traj_reid_similarity.append(reid_similarity)
                else:
                    diff_traj_reid_similarity.append(reid_similarity)
        print(np.mean(same_traj_reid_similarity),np.mean(diff_traj_reid_similarity))
        plt.subplot(1,2,1)
        plt.hist(same_traj_reid_similarity,bins = 100, label='same traj',density=True,color='r')
        plt.subplot(1,2,2)
        plt.hist(diff_traj_reid_similarity,bins = 100, label='diff traj',density=True,color='g')
        plt.show()
        ####
        plt.hist(same_traj_reid_similarity,density=True,stacked=False,color='r')
        plt.hist(diff_traj_reid_similarity,density=True,stacked=False,color='g')
        plt.show()
    
    prev_batch_tracklet_test = copy.deepcopy(curr_batch_tracklet_test) # 0.995
plt.figure()
plt.subplot(1,2,1)
plt.hist(same_traj_reid_similarity,label= 'same traj')
plt.subplot(1,2,2)
plt.hist(diff_traj_reid_similarity,label= 'diff traj')
plt.savefig(os.path.join(dst_path,'reid_similarity_10_frames.png'))
plt.show()
'''
det_thresh = 0.4
0.9972154621810105 0.9673231021033444
0.9970187654371807 0.9680364867156663
0.9969737462371492 0.9681349159156649
0.9973423001735386 0.9676919939674434
0.997179748311444 0.967303628474228
0.9972318466355797 0.9670811071163353
0.9971523624663499 0.9668923985776434
'''