import numpy as np
import matplotlib.pyplot as plt
import cv2
# 检测轨迹结果
track_seq_path = r'/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/det/det.txt'
track_seq = np.loadtxt(track_seq_path,delimiter = ',')
# track_seq = track_seq[0:100,:]
lines_length = int(track_seq[:,0].max())
unique_track_id = np.unique(track_seq[:,1]) # 总的轨迹数目
print('轨迹数目',len(unique_track_id))

img = cv2.imread(r'/home/allenyljiang/Documents/Dataset/MOT16/train/MOT16-02/img1/000001.jpg')

dst_path = r'/home/allenyljiang/Documents/Dataset/MOT16/train/MOT16-05/tracklet.jpg'
track_length_list = []
frame_dict = {}
frame_mask = []
velocity_hori = []
velocity_vert = []
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
    v_hori = y[1:,0]-y[0:-1,0] # 水平速度
    velocity_hori.append(np.mean(v_hori))
    v_vert = y[1:,0]-y[0:-1,0] # 垂直速度
    velocity_vert.append(np.mean(v_vert))
    track_length_list.append(len(y)) # 全是连续的
    #y = 1/2*(dets[:, 0:2]+dets[:,2:4]) #  x1y1x2y2模式
    # color = tuple(np.random.choice(range(256), size=3).tolist()) # tolist把Numpy数组转化为列表
    color = (0,0,255)
    # print(y)
    # print(y)
    # plt.plot(y[:,0],y[:,1]) # plt.scatter
    # plt.show

    for i in range(len(y)):
        cv2.circle(img,(int(y[i,0]),int(y[i,1])),8,color,-1)  # height：1080 width：1920 宽，高
    cv2.putText(img,str(int(unique_track_id[track_id])),(int(y[0,0]),int(y[0,1])),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    # cv2.imshow('track',img)
    # cv2.waitKey(0)
# cv2.imwrite(dst_path,img)
print(track_length_list)
print('水平速度:',velocity_hori)

print('垂直速度:',velocity_vert)
print(frame_mask)
# ####  格式转换 ####
# considerd_set = {1,2,7}
# gt_file= '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/gt/gt.txt'
# gt_data = np.loadtxt(gt_file,delimiter=',')
# # gt_ped_data = gt_data[gt_data[:,-2] in considerd_set,:]
# gt_ped_data = gt_data[gt_data[:,-2] == 1 ,:]
# dst_file = '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/gt/gt_revised.txt'
# np.savetxt(dst_file,gt_ped_data,fmt='%i',delimiter=',')