'''
show pedestrain or static person in ground_truth
'''
import os

import cv2
import numpy as np
from MOT_demo_v14_based_on_reid_v22_notstandard_yolov5_add_face_remove_skeleton_ref_v4 import compute_iou_single_box,compute_iou_between_bbox_list
import matplotlib.pyplot as plt
folder = 'MOT20-01'
gt_txt =  os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'gt/gt.txt') # 前后不需要加/
# 存在以‘’/’’开始的参数，从最后一个以”/”开头的参数开始拼接，之前的参数全部丢弃。
img_list = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'img1')
img_blob = sorted(os.listdir(img_list))
seq_tracks = np.loadtxt(gt_txt,delimiter=',') # 注意加上delimiter
seq_tracks_valid_part3 = seq_tracks[seq_tracks[:,-2] == 7,:] # static_person
seq_tracks = seq_tracks[seq_tracks[:,-3] == 1,:] # pedestrain
classes = np.unique(seq_tracks[:,-2]) # the considered classes in gt
# seq_tracks_valid_part1 = seq_tracks[seq_tracks[:,-2] == 1,:] # pedestrain
# seq_tracks_valid_part2 = seq_tracks[seq_tracks[:,-2] == 2,:]
# seq_tracks_valid_part3 = seq_tracks[seq_tracks[:,-2] == 7,:] # static_person
# seq_tracks = np.concatenate((seq_tracks_valid_part1,seq_tracks_valid_part2,seq_tracks_valid_part3),axis=0)

dst_path = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'gt/vis')
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

# fig = plt.figure()
# ax1 = fig.add_subplot(111,aspect='equal')
num_person_per_frame = []
max_iou_between_people = []
for frame in range(int(seq_tracks[:,0].max())):
    img = cv2.imread(os.path.join(img_list,img_blob[frame]))
    img_dst = os.path.join(dst_path, img_blob[frame])
    frame += 1
    track_id = seq_tracks[seq_tracks[:,0] == frame,1] # 轨迹id
    dets = seq_tracks[seq_tracks[:,0] == frame,2:6] # 检测框
    visibility = seq_tracks[seq_tracks[:,0] == frame,-1] # visibility
    dets[:,2:4]+=dets[:,0:2]
    ## 对框进行缩放 ## 
    center_x, center_y = (dets[:,0] + dets[:,2]) / 2, (dets[:,1] + dets[:,3]) / 2
    ratio = 0.8
    width, height = (np.array(dets[:,2] - dets[:,0])*ratio), np.array((dets[:,3] - dets[:,1]))
    left, right, top, bottom = (center_x - width / 2), (center_x + width / 2), (center_y - height / 2), (center_y + height / 2)
    
    num_person_per_frame.append(len(dets))
    corresponding_coefficient_matrix = compute_iou_between_bbox_list(dets.reshape(-1,2,2),dets.reshape(-1,2,2))
    iou_in_same_frame = corresponding_coefficient_matrix - np.diag(np.diagonal(corresponding_coefficient_matrix)) # 同一帧当中不同人的iou相似度
    # print(np.max(iou_in_same_frame))
    max_iou_between_people.append(np.max(iou_in_same_frame))
    # print(np.where(iou_in_same_frame == np.max(iou_in_same_frame)))
    for i in range(len(track_id)):
        # left,top = int(dets[i,0]),int(dets[i,1])
        # cv2.putText(img, str(int(track_id[i])), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
        # cv2.rectangle(img,(int(dets[i,0]),int(dets[i,1])),(int(dets[i,2]),int(dets[i,3])),(0,255,0),1)
        cv2.rectangle(img,(int(left[i]),int(top[i])),(int(right[i]),int(bottom[i])),(0,255,0),2)
        cv2.putText(img,str(round(visibility[i],2)), (int(left[i]),int(top[i])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # ax1.imshow(img)
        # plt.imshow(img)
        # plt.show(img)
    cv2.imwrite(img_dst,img)
print(num_person_per_frame)
f = open(folder+".txt", "w")
f.write(str(num_person_per_frame))
f.close()
print('mean',np.mean(num_person_per_frame),'min',min(num_person_per_frame),'max',max(num_person_per_frame))
print('std',np.std(num_person_per_frame))
plt.figure()
plt.plot(list(range(len(num_person_per_frame))),num_person_per_frame)
plt.show()
plt.title('The numbers of person per frame')
plt.savefig(folder+'fig.png')
print('max_iou_between_people = {}'.format(np.max(max_iou_between_people)))