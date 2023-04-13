import copy
import numpy as np
import matplotlib.pyplot as plt
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
# #################################拟合优度R^2的计算######################################
def __sst(y_no_fitting):
    """
    计算SST(total sum of squares) 总平方和
    :param y_no_predicted: List[int] or array[int] 待拟合的y
    :return: 总平方和SST
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_no_fitting]
    sst = sum(s_list)
    return sst


def __ssr(y_fitting, y_no_fitting):
    """
    计算SSR(regression sum of squares) 回归平方和
    :param y_fitting: List[int] or array[int]  拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 回归平方和SSR
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_fitting]
    ssr = sum(s_list)
    return ssr


def __sse(y_fitting, y_no_fitting):
    """
    计算SSE(error sum of squares) 残差平方和
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 残差平方和SSE
    """
    s_list = [(y_fitting[i] - y_no_fitting[i])**2 for i in range(len(y_fitting))]
    sse = sum(s_list)
    return sse


def goodness_of_fit(y_fitting, y_no_fitting):
    """
    计算拟合优度R^2
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 拟合优度R^2
    """
    SSR = __ssr(y_fitting, y_no_fitting)
    SST = __sst(y_no_fitting)
    rr = SSR /SST
    return rr
folder = 'MOT20-01'
# 检测轨迹结果
track_seq_path = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'gt/gt.txt')
image_dir = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'img1')
pred_image_dir = os.path.join('/home/allenyljiang/Documents/Dataset/MOT20/train',folder,'img1_pred')
if not os.path.exists(pred_image_dir):
    os.makedirs(pred_image_dir,exist_ok=True)
track_seq = np.loadtxt(track_seq_path,delimiter = ',')
track_seq = track_seq[track_seq[:,-2] == 1,:] #
lines_length = int(track_seq[:,0].max())
unique_track_id = np.unique(track_seq[:,1]) # 总的轨迹数目
print(len(unique_track_id))
track_length_list = []
frame_dict = {}
frame_mask = []
velocity_hori = []
velocity_vert = []
iou_similarity_list = []
data_similarity_list = []
polyfit_error_list = []
track_polyfit_error = []
single_traj_fit_error = []

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
    mean_width,mean_height = np.mean(dets[:,2]),np.mean(dets[:,3])

    centers = 1/2*dets[:, 2:4] + dets[:, 0:2]
    polyfit_y = centers[:, 0] # center_x
    polyfit_z = centers[:, 1] # center_y
    dets[:, 2:4] += dets[:, 0:2]  # left,top,right,bottom--top,bottom,left,right

    for i in range(10,len(dets)-10,10):
        polyfit1d_list = []
        polyfit2d_list = []
        polyfit_x =  x[i-10:i]
        # polyfit_x = [variable for variable in x[i-10:i]]
        # polyfit_error = np.polyfit(polyfit_x[i:i+10], polyfit_y[i:i+10], 1, full=True)[1][0] + np.polyfit(polyfit_x[i:i+10], polyfit_z[i:i+10], 1, full=True)[1][0]
        ## 2d fitter ##
        horicenter_fitter_coefficients2 = np.polyfit(polyfit_x, polyfit_y[i-10:i], 2)
        vertcenter_fitter_coefficients2 = np.polyfit(polyfit_x, polyfit_z[i-10:i], 2)
        horicenter_fitter2 = np.poly1d(horicenter_fitter_coefficients2)  # np.poly1d根据数组生成一个多项式
        vertcenter_fitter2 = np.poly1d(vertcenter_fitter_coefficients2)
        ## 1d fitter ##
        horicenter_fitter_coefficients1 = np.polyfit(polyfit_x, polyfit_y[i-10:i], 1)
        vertcenter_fitter_coefficients1 = np.polyfit(polyfit_x, polyfit_z[i-10:i], 1)
        horicenter_fitter1 = np.poly1d(horicenter_fitter_coefficients1)  # np.poly1d根据数组生成一个多项式
        vertcenter_fitter1 = np.poly1d(vertcenter_fitter_coefficients1)
        if i > 0:
            for j in range(i,i+10): # 使用 i-10 ～ i-1 帧的预测 i 到 i+10
                img_name = '%06d' %(x[j])+".jpg"
                img = cv2.imread(os.path.join(image_dir,img_name))
                left1,top1,right1,bottom1 = horicenter_fitter1(j) - mean_width / 2.0 , vertcenter_fitter1(j) - mean_height/2 ,horicenter_fitter1(j) + mean_width / 2.0 , vertcenter_fitter1(j) + mean_height/2
                left2,top2,right2,bottom2 = horicenter_fitter2(j) - mean_width / 2.0 , vertcenter_fitter2(j) - mean_height/2 ,horicenter_fitter2(j) + mean_width / 2.0 , vertcenter_fitter2(j) + mean_height/2
                cv2.rectangle(img,(int(left1),int(top1)),(int(right1),int(bottom1)),[0,255,0]) # g
                cv2.rectangle(img,(int(left2),int(top2)),(int(right2),int(bottom2)),[255,0,0]) # b
                # polyfit1d_list.append(horicenter_fitter1(j))
                # polyfit2d_list.append(horicenter_fitter2(j))
                polyfit1d_list.append(left1)
                polyfit2d_list.append(left2)
                cv2.rectangle(img,(int(dets[j,0]),int(dets[j,1])),(int(dets[j,2]),int(dets[j,3])),[0,0,255]) # r

                cv2.imwrite(os.path.join(pred_image_dir,img_name),img)
        rr1 = goodness_of_fit(polyfit1d_list,dets[i:i+10,0])
        rr2 = goodness_of_fit(polyfit2d_list,dets[i:i+10,0])
        print('一阶拟合优度：{}，二阶拟合优度：{}'.format(rr1,rr2))



