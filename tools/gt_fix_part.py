import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2
# left,top,width,height
# compute_iou_between_bbox_list
# from MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref_v4 import *
from MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref_v4 import compute_iou_single_box,compute_iou_between_bbox_list
### parameter defininition ###
frame_cnt = 500
# 检测轨迹结果
track_seq_path = r'/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/gt/gt.txt'
gt_part = track_seq_path.replace('gt.txt','gt_part.txt')
track_seq = np.loadtxt(track_seq_path,delimiter = ',')
track_seq_valid = track_seq[track_seq[:,-3] == 1,:] # choose valid gt
track_seq_part = track_seq_valid[track_seq_valid[:,0]<= frame_cnt,:]
np.savetxt(gt_part,track_seq_part,fmt='%i',delimiter=',') # fmt the format is integer %i %.1f ['%i','%i']
# src = open(gt_part,'w')
# src.write(track_seq_part)
# src.close()

