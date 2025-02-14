import numpy as np
import matplotlib.pyplot as plt

# 检测轨迹结果
track_seq_path = '/media/allenyljiang/564AFA804AFA5BE5/Dataset/MOT20/train/MOT20-03/gt/gt.txt'
track_seq = np.loadtxt(track_seq_path, delimiter=',')
track_seq = track_seq[track_seq[:, -3] == 1, :]  #
considered_cls = np.unique(track_seq[:, -2])  # 只有第一类
lines_length = int(track_seq[:, 0].max())
unique_track_id = np.unique(track_seq[:, 1])  # 总的轨迹数目
unique_frame_id = np.unique(track_seq[:,0]) # 总的frame数目
# 窗口长度，即帧的个数
window_length = 5

for track_id in unique_track_id:
    # 提取轨迹ID为1的目标的坐标
    trajectory_data = [item for item in track_seq if item[1] == track_id]

    # 计算每帧之间的移动距离
    distances = []

    for i in range(window_length, len(trajectory_data),window_length):
        x1, y1 = trajectory_data[i-window_length][2], trajectory_data[i-window_length][3]
        x2, y2 = trajectory_data[i][2], trajectory_data[i][3]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(distance)

# 绘制移动距离的直方图
plt.hist(distances, bins=5, edgecolor='black')
plt.title('Distribution of Movement Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()
