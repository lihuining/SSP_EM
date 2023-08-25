import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
folder = 'MOT20-03'
# 检测轨迹结果
track_seq_path = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'gt/gt.txt')
track_seq = np.loadtxt(track_seq_path,delimiter = ',')
# track_seq = track_seq[0:100,:]
track_seq = track_seq[track_seq[:,-3] == 1,:] # only pedestrains
lines_length = int(track_seq[:,0].max())
unique_track_id = np.unique(track_seq[:,1]) # 总的轨迹数目
print('轨迹数目',len(unique_track_id))

img = cv2.imread(os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'img1/000001.jpg'))
img2 = cv2.imread(os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'img1/000002.jpg'))
dst_path_track = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'tracklet.jpg')
dst_path_track_start_and_end = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'tracklet_start_and_end.jpg')
track_length_list = []
frame_dict = {}
frame_mask = []
velocity_hori = []
velocity_vert = []
velocity_hori_std_list = []
velocity_vert_std_list = []
max_angle_list = []
trajectory_x_list = []
trajectory_y_list = []
step = 2
for track_id in range(len(unique_track_id)): # 画出每一条轨迹
    # 轨迹名称
    # unique_track_id[track_id]
    # print(unique_track_id[track_id])
    x = track_seq[track_seq[:,1] == unique_track_id[track_id],0] # 自变量,所在帧
    frame_dict[track_id] = x
    frame_mask.append(1 if int(sum(x[1:]-x[0:-1])) == (len(x)-1) else 0)
    # print(frame_mask)
    # img_id = len(track_seq[track_seq[:,1] == unique_track_id[track_id],0]) # 自变量
    dets = track_seq[track_seq[:, 1]== unique_track_id[track_id], 2:6]

    y =  dets[:, 0:2] + 1/2*dets[:, 2:4] # 人体中心点,标准mot格式
    # # 进行下采样
    # y = y[::2,:]
    v_hori = y[1:,0]-y[0:-1,0] # 水平速度
    # print('standard error of horizontal velocity',np.std(v_hori))
    velocity_hori_std_list.append(np.std(v_hori))
    velocity_hori.append(np.mean(v_hori))
    v_vert = y[1:,0]-y[0:-1,0] # 垂直速度

    v_pre = np.concatenate((np.array(v_hori[0:-1]).reshape(1,-1),np.array(v_vert[0:-1]).reshape(1,-1)),axis=0)
    v_now = np.concatenate((np.array(v_hori[1:]).reshape(1,-1),np.array(v_vert[1:]).reshape(1,-1)),axis=0)
    cos_list = np.zeros(v_pre.shape[1])
    if len(cos_list) < 1 :
        continue
    for i in range(v_pre.shape[1]):
        # print(v_pre[:,i].dot(v_now[:,i]))
        # print(np.linalg.norm(v_pre[:,i])*np.linalg.norm(v_now[:,i]))
        if np.linalg.norm(v_pre[:,i])*np.linalg.norm(v_now[:,i]) == 0:
            cos_list[i] = 0
            continue
        # cos_angle =  np.clip(v_pre[:,i].dot(v_now[:,i])/(np.linalg.norm(v_pre[:,i])*np.linalg.norm(v_now[:,i])),0.1,0.99)
        cos_angle =  min(v_pre[:,i].dot(v_now[:,i])/(np.linalg.norm(v_pre[:,i])*np.linalg.norm(v_now[:,i])),1)
        cos_list[i] = np.arccos(cos_angle)*360/2/np.pi
    print('max angle',max(cos_list))
    max_angle_list.append(max(cos_list))
    # np.sign(v_hori[0:-1]*v_hori[1:])+np.sign(v_vert[0:-1]*v_vert[1:])
    # (np.sign(velocity_x*velocity_x_pre)+np.sign(velocity_y*velocity_y_pre))
    # print('standard error of vertical velocity', np.std(v_vert))
    velocity_vert_std_list.append(np.std(v_vert))
    velocity_vert.append(np.mean(v_vert))
    track_length_list.append(len(y)) # 全是连续的
    #y = 1/2*(dets[:, 0:2]+dets[:,2:4]) #  x1y1x2y2模式
    color = tuple(np.random.choice(range(256), size=3).tolist()) # tolist把Numpy数组转化为列表
    #color = (0,0,255)
    # print(y)
    # print(y)
    # plt.plot(y[:,0],y[:,1]) # plt.scatter
    # plt.show

    for i in range(len(y)):
        cv2.circle(img,(int(y[i,0]),int(y[i,1])),8,color,-1)  # height：1080 width：1920 宽，高
    cv2.putText(img,str(int(unique_track_id[track_id])),(int(y[0,0]),int(y[0,1])),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    cv2.circle(img2,(int(y[0,0]),int(y[0,1])),8,(0,255,0),-1)
    cv2.circle(img2,(int(y[-1,0]),int(y[-1,1])),8,(255,0,0),-1)
    trajectory_x_list.append(int(y[0,0]))
    trajectory_x_list.append(int(y[-1,0]))
    trajectory_y_list.append(int(y[0,1]))
    trajectory_y_list.append(int(y[-1,1]))
    # cv2.imshow('track',img)
    # cv2.waitKey(0)
cv2.imwrite(dst_path_track,img)
cv2.imwrite(dst_path_track_start_and_end,img2)
plt.figure()
plt.subplot(1,2,1)
plt.hist(trajectory_x_list)
plt.subplot(1,2,2)
plt.hist(trajectory_y_list)
plt.savefig(str(folder)+'tracklet_start_and_end.png')
print('max angle in all tracks',max(max_angle_list))
print(track_length_list)
print('水平速度:',velocity_hori)
print('垂直速度:',velocity_vert)
print(max(velocity_hori_std_list),max(velocity_vert_std_list))
print(frame_mask)
# ####  格式转换 ####
# considerd_set = {1,2,7}
# gt_file= '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/gt/gt.txt'
# gt_data = np.loadtxt(gt_file,delimiter=',')
# # gt_ped_data = gt_data[gt_data[:,-2] in considerd_set,:]
# gt_ped_data = gt_data[gt_data[:,-2] == 1 ,:]
# dst_file = '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/gt/gt_revised.txt'
# np.savetxt(dst_file,gt_ped_data,fmt='%i',delimiter=',')