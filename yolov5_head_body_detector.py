import torch
import torchvision
import matplotlib.pyplot as plt
import glob
import torch.backends.cudnn as cudnn
from det_model_new.models.experimental import *
from det_model_new.utils.datasets import *
from det_model_new.utils.torch_utils import *
from det_model_new.utils import torch_utils
from det_model_new.utils.general import *#

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def detect_face(img, curr_frame, device, half, model, names, conf_thres, iou_thres):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = torch_utils.time_synchronized()
    time_model_start = time.time()
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    time_model_end = time.time()
    t2 = torch_utils.time_synchronized()
    # print(str(t2 - t1))
    box_detected = []
    box_confidence_scores = []
    box_class = []
    result_array = []
    for i, det in enumerate(pred):
        gn = torch.tensor(curr_frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], curr_frame.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                if cls == 1:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    # if int(cls.data.cpu().numpy()) == 0:
                    box_detected.append([(float(xyxy[0].data.cpu().numpy()), float(xyxy[1].data.cpu().numpy())),
                                         (float(xyxy[2].data.cpu().numpy()), float(xyxy[3].data.cpu().numpy()))])
                    box_confidence_scores.append(float(conf.data.cpu().numpy()))
                    box_class.append(int(cls.data.cpu().numpy()))
                    result_array.append(np.array([float(xyxy[0].data.cpu().numpy()), float(xyxy[1].data.cpu().numpy()),
                                                  float(xyxy[2].data.cpu().numpy()), float(xyxy[3].data.cpu().numpy()),
                                                  float(conf.data.cpu().numpy())]))
    #             if int(cls.data.cpu().numpy()) == 1:
    #                 im0 = cv2.rectangle(im0, (int(xyxy[0].data.cpu().numpy()), int(xyxy[1].data.cpu().numpy())),
    #                                     (int(xyxy[2].data.cpu().numpy()), int(xyxy[3].data.cpu().numpy())), (0, 255, 0), 2)
    # cv2.imwrite(os.path.join(out, str(frame_cnt) + '.jpg'), im0)
    # frame_cnt += 1
    return np.array(result_array)