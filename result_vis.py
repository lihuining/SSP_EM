import numpy as np
import cv2
import os

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, frame_id=0):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    # cv2.putText(im, 'frame: %d num: %d' % (frame_id, len(tlwhs)),
    #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

result_txt = '/home/allenyljiang/Documents/SSP_EM/YOLOX_outputs/MOT20/window_size_10/MOT20-03.txt'
img_list = os.path.join('/media/allenyljiang/564AFA804AFA5BE5/Dataset/MOT20/train/MOT20-03/img1')
img_blob = sorted(os.listdir(img_list))
seq_tracks = np.loadtxt(result_txt,delimiter=',') # 注意加上delimiter

dst_path = '/home/allenyljiang/Documents/SSP_EM/YOLOX_outputs/MOT20/window_size_10/track_vis'
#dst_path = os.path.join('MOT20_03vis')
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

for frame in range(int(seq_tracks[:,0].max())):
    img = cv2.imread(os.path.join(img_list,img_blob[frame]))
    img_dst = os.path.join(dst_path, img_blob[frame])
    frame += 1
    track_id = seq_tracks[seq_tracks[:,0] == frame,1] # 轨迹id
    dets = seq_tracks[seq_tracks[:,0] == frame,2:6] # 检测框
    #dets[:,2:4]+=dets[:,0:2]
    img = plot_tracking(img, dets, track_id, frame_id=frame)
    # if frame % 10 == 0:
    cv2.imwrite(img_dst,img)
    if frame > 100:
        break


