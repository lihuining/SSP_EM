import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import cv2
import os
# left,top,width,height
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
folder = 'MOT20-01'
# 检测轨迹结果
track_seq_path = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'gt/gt.txt')
track_seq = np.loadtxt(track_seq_path,delimiter = ',')
# track_seq = track_seq[0:100,:]
# track_seq = np.concatenate((track_seq_valid_part1,track_seq_valid_part2,track_seq_valid_part3),axis=0)
track_seq = track_seq[track_seq[:,-2] == 1,:] #
lines_length = int(track_seq[:,0].max())
unique_track_id = np.unique(track_seq[:,1]) # 总的轨迹数目
print(len(unique_track_id))
# img = cv2.imread(r'/home/allenyljiang/Documents/Dataset/MOT16/train/MOT16-02/img1/000001.jpg')
track_length_list = []
frame_dict = {}
frame_mask = []
velocity_hori = []
velocity_vert = []
iou_similarity_list = []
data_similarity_list = []
polyfit_error_list = []
track_polyfit_error = []
# single_traj_fit_error = []
for track_id in range(len(unique_track_id)): # 每一条轨迹
    # 轨迹名称
    # unique_track_id[track_id]
    # print(unique_track_id[track_id])
    single_traj_fit_error = []
    x = track_seq[track_seq[:,1] == unique_track_id[track_id],0] # 自变量,所在帧
    frame_dict[track_id] = x
    frame_mask.append(1 if int(sum(x[1:]-x[0:-1])) == (len(x)-1) else 0)
    # img_id = len(track_seq[track_seq[:,1] == unique_track_id[track_id],0]) # 自变量
    dets = track_seq[track_seq[:, 1]== unique_track_id[track_id], 2:6]
    centers = 1/2*dets[:, 2:4] + dets[:, 0:2]

    polyfit_y = centers[:, 0] # center_x
    polyfit_z = centers[:, 1] # center_y
    dets[:, 2:4] += dets[:, 0:2]  # left,top,right,bottom--top,bottom,left,right

    for i in range(len(dets)-10):
        polyfit_x = [variable-x[i] for variable in x[i:i+10]]
        # polyfit_error = np.polyfit(polyfit_x[i:i+10], polyfit_y[i:i+10], 1, full=True)[1][0] + np.polyfit(polyfit_x[i:i+10], polyfit_z[i:i+10], 1, full=True)[1][0]
        polyfit_error = np.polyfit(polyfit_x, polyfit_y[i:i+10], 2, full=True)[1][0] + np.polyfit(polyfit_x, polyfit_z[i:i+10], 2, full=True)[1][0]
        polyfit_error_list.append(polyfit_error)
        single_traj_fit_error.append(polyfit_error)
        # track_polyfit_error.append(polyfit_error)
    if len(single_traj_fit_error) == 0: # 轨迹长度 《= 10的时候无法计算
        continue
    print('mean error',np.mean(single_traj_fit_error))
    print('max error',np.max(single_traj_fit_error))
    print('error variance',np.var(single_traj_fit_error))
    # plt.figure()
    # plt.hist(single_traj_fit_error,label='single traj error')
    # plt.show()

plt.figure()
# cumulative=True 计算累计值
n,bins,patches = plt.hist(polyfit_error_list,250,(0,5),label='all polyfit error',density=True)
# n：落入每一个间隔的样本数，bins：每一个间隔的起点和终点，len(n)+1
plt.show()
print('mean',np.mean(polyfit_error_list))
print('max',np.max(polyfit_error_list))
print('variance',np.var(polyfit_error_list))
plt.savefig('statistic_information/polyfit/'+'2dhist.png')