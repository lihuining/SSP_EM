import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2
# left,top,width,height
# compute_iou_between_bbox_list
# from MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref_v4 import *


def compute_overlap_between_bbox_list(head_box_detected, box_detected):#[(left, top), (right, bottom)]
    corresponding_coefficient_matrix = np.zeros((len(head_box_detected), len(box_detected)))
    for idx_row in range(len(head_box_detected)):
        for idx_col in range(len(box_detected)):
            corresponding_coefficient_matrix[idx_row, idx_col] = compute_overlap_single_box([head_box_detected[idx_row][0][1], head_box_detected[idx_row][1][1], head_box_detected[idx_row][0][0], head_box_detected[idx_row][1][0]], \
                                                                                        [box_detected[idx_col][0][1], box_detected[idx_col][1][1], box_detected[idx_col][0][0], box_detected[idx_col][1][0]])
    return corresponding_coefficient_matrix

def compute_overlap_single_box(curr_img_boxes, next_img_boxes):# Order: top, bottom, left, right
    intersect_vert = min([curr_img_boxes[1], next_img_boxes[1]]) - max([curr_img_boxes[0], next_img_boxes[0]])
    intersect_hori = min([curr_img_boxes[3], next_img_boxes[3]]) - max([curr_img_boxes[2], next_img_boxes[2]])
    union_vert = max([curr_img_boxes[1], next_img_boxes[1]]) - min([curr_img_boxes[0], next_img_boxes[0]])
    union_hori = max([curr_img_boxes[3], next_img_boxes[3]]) - min([curr_img_boxes[2], next_img_boxes[2]])
    area1 = abs(curr_img_boxes[0] - curr_img_boxes[1]) * abs(curr_img_boxes[2]-curr_img_boxes[3])
    area2 = abs(next_img_boxes[0] - next_img_boxes[1]) * abs(next_img_boxes[2]-next_img_boxes[3])
    if intersect_vert > 0 and intersect_hori > 0 and union_vert > 0 and union_hori > 0:
        corresponding_coefficient = float(intersect_vert) * float(intersect_hori) / min(area1,area2)
    else:
        corresponding_coefficient = 0.0
    return corresponding_coefficient


# 检测轨迹结果
track_seq_path = r'/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/gt/gt.txt'
track_seq = np.loadtxt(track_seq_path, delimiter=',')
track_seq = track_seq[track_seq[:, -3] == 1, :]  #
considered_cls = np.unique(track_seq[:, -2])  # 只有第一类
lines_length = int(track_seq[:, 0].max())
unique_track_id = np.unique(track_seq[:, 1])  # 总的轨迹数目
unique_frame_id = np.unique(track_seq[:,0]) # 总的frame数目
print(len(unique_track_id))
# img = cv2.imread(r'/home/allenyljiang/Documents/Dataset/MOT16/train/MOT16-02/img1/000001.jpg')
dst_path = r'/home/allenyljiang/Documents/Dataset/MOT16/train/MOT16-05/tracklet.jpg'
track_length_list = []
frame_dict = {}
frame_mask = []
velocity_hori = []
velocity_vert = []
iou_similarity_list = []
data_similarity_list = []
polyfit_error_list = []
min_aspect_ratio_list = []
max_aspect_ratio_list = []
overlap_list = []
min_height_list = []
max_height_list = []
min_width_list = []
max_width_list = []
aspect_ratio_var_list = []
for frame_id in range(len(unique_frame_id)):  # 每一条轨迹
    # 轨迹名称
    # unique_track_id[track_id]
    # print(unique_track_id[track_id])
    # img_id = len(track_seq[track_seq[:,1] == unique_track_id[track_id],0]) # 自变量
    dets = track_seq[track_seq[:, 0] == unique_frame_id[frame_id], 2:6]
    centers = 1 / 2 * dets[:, 2:4] + dets[:, 0:2]
    polyfit_y = centers[:, 0]
    polyfit_z = centers[:, 1]
    dets[:, 2:4] += dets[:, 0:2]  # left,top,right,bottom--top,bottom,left,right
    y = dets[:, [1, 3, 0, 2]]
    width = dets[:, 2] - dets[:, 0]

    min_width_list.append(np.min(width))
    max_width_list.append(np.max(width))
    height = dets[:, 3] - dets[:, 1]
    aspect_ratio = height / width

    min_height_list.append(np.min(height))
    max_height_list.append(np.max(height))

    min_aspect_ratio_list.append(np.min(aspect_ratio))
    max_aspect_ratio_list.append(np.max(aspect_ratio))
    aspect_ratio_var_list.append(np.std(aspect_ratio))
    ### same frame overlap information ###
    corresponding_coefficient_matrix = compute_overlap_between_bbox_list(dets.reshape(-1, 2, 2), dets.reshape(-1, 2, 2))
    overlap_in_same_frame = corresponding_coefficient_matrix - np.diag(np.diagonal(corresponding_coefficient_matrix))
    print(np.max(overlap_in_same_frame))
    overlap_list.append(np.max(overlap_in_same_frame))
    print(np.mean(overlap_in_same_frame))
print(overlap_list)
    # ### iou information ###
    # y = dets[:,[1,3,0,2]]
    # for i in range(len(y)-1):
    #     iou_similarity = compute_iou_single_box(y[i],y[i+3])
    #     iou_similarity_list.append(iou_similarity)
    # print('single mean',np.mean(iou_similarity_list))
    # print('single variance',np.var(iou_similarity_list))

print('min aspect ratio is {},min width is {},min height is {}'.format(np.min(min_aspect_ratio_list),
                                                                       np.min(min_width_list), np.min(min_height_list)))
print('max aspect ratio is {},max width is {},max height is {}'.format(np.max(max_aspect_ratio_list),
                                                                       np.max(max_width_list), np.max(max_height_list)))
