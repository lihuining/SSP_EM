import cv2
import os
import json
def x1y1wh2x1y1x2y2(bbox):
    x, y, w, h = bbox[0],bbox[1],bbox[2],bbox[3] # top,left,width.height
    x1,y1 = float(x),float(y)
    x2 = x1 + float(w)
    y2 = y1 + float(h)
    x1y1x2y2 = [(float(x1),float(y1)),(float(x2),float(y2))]
    return x1y1x2y2

det = open('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/det/det.txt')
det_data = det.readlines()
frameid_pre = '0'
bbox_list = []
box_confidence_scores = []
tracklet_pose_collection = []
tracklet_pose_collection_tmp = {}
img = cv2.imread('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1/000001.jpg')
dst_path = '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/000001_result.jpg'
root_dir = '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1'
imgdir = sorted(os.listdir(root_dir))
txt_dst = '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/det_tracklet.txt'
dst = open(txt_dst,'w')
for line in det_data:
    frameid = line.split(',')[0]
    path = os.path.join(root_dir,imgdir[int(frameid)-1])
    bbox = line.split(',')[2:6]
    confidence_score = line.split(',')[6]
    if frameid == '1':
        bbox_list.append(x1y1wh2x1y1x2y2(bbox))
        cv2.rectangle(img,(int(bbox_list[-1][0][0]),int(bbox_list[-1][0][1])),(int(bbox_list[-1][1][0]),int(bbox_list[-1][1][1])),(0,255,0),2)
        box_confidence_scores.append(float(confidence_score))
        frameid_pre = frameid
        path_pre = os.path.join(root_dir,imgdir[int(frameid_pre)-1])
        continue
    elif frameid != frameid_pre: # 从此时更新
        if frameid == '2':
            cv2.imwrite(dst_path,img)
        tracklet_pose_collection_tmp['bbox_list'] = bbox_list
        tracklet_pose_collection_tmp['box_confidence_scores'] = box_confidence_scores
        tracklet_pose_collection_tmp['img_dir'] = path_pre
        tracklet_pose_collection.append(tracklet_pose_collection_tmp)
        dst.write(json.dumps(tracklet_pose_collection_tmp))
        dst.write('\n')
        tracklet_pose_collection_tmp = {}
        bbox_list = []
        box_confidence_scores = []
        frameid_pre = frameid
        path_pre = os.path.join(root_dir,imgdir[int(frameid_pre)-1])
    else:
        bbox_list.append(x1y1wh2x1y1x2y2(bbox))
        box_confidence_scores.append(float(confidence_score))
        frameid_pre = frameid
        path_pre = os.path.join(root_dir,imgdir[int(frameid_pre)-1])
# 写入最后一帧的数据
tracklet_pose_collection_tmp['bbox_list'] = bbox_list
tracklet_pose_collection_tmp['box_confidence_scores'] = box_confidence_scores
tracklet_pose_collection_tmp['img_dir'] = path_pre
tracklet_pose_collection.append(tracklet_pose_collection_tmp)
dst.write(json.dumps(tracklet_pose_collection_tmp))
dst.write('\n')

# tracklet_pose_collection_tmp = {}
# tracklet_pose_collection_tmp['bbox_list'] = box_detected
# tracklet_pose_collection_tmp['head_bbox_list'] = head_box_detected
# tracklet_pose_collection_tmp['box_confidence_scores'] = box_confidence_scores
# tracklet_pose_collection_tmp['img_dir'] = path
# tracklet_pose_collection_tmp['foreignmatter_bbox_list'] = []
# tracklet_pose_collection_tmp['foreignmatter_box_confidence_scores'] = []
# tracklet_pose_collection.append(tracklet_pose_collection_tmp)
# ## json 文件读取
# f_read = open(txt_dst,encoding='utf-8')
# for i in range(600):
#     line = f_read.readline()
#     js = json.loads(line)

