import cv2
import numpy as np

# 图片路径
img = cv2.imread('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-02/tracklet.jpg')
a = []
b = []


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL) # 解决图像显示不全的问题
        cv2.resizeWindow('image',960,540)
        # cv2.resizeWindow('image',None,fx=0.5,fy=0.5)
        cv2.imshow("image", img)


cv2.namedWindow("image",0)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# img =  cv2.resize(img,None,fx=0.5,fy=0.5) # 图片，输出图片尺寸，宽的比例，高的比例
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',600,500)
cv2.imshow("image", img)
cv2.waitKey(0)
print(a[0], b[0])