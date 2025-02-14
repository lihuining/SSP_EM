import os
import cv2

path = '/home/allenyljiang/Documents/SSP_EM/YOLOX_outputs/MOT20/window_size_10/track_vis/' # 由于直接+item需要最后有/
filelist = os.listdir(path)
get_key = lambda i:int(i.split('_')[-1].split('.')[0])
orderd_filelist = sorted(filelist,key = get_key)
# filelist.sort(key = lambda x :int (x[:-4]))
fps = 20  # 视频每秒25帧
size = (1173, 880)  # 需要注意的是在 VideoWriter 中指定的尺寸要和 write() 中写进去的一样，不然视频会存储失败的
# size = (960, 540)  # 需要转为视频的图片的尺寸
# 可以使用cv2.resize()进行修改

video = cv2.VideoWriter("../videos/MOT20-03.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
print()
# 视频保存在当前目录下

for item in orderd_filelist[:100]:
    if item.endswith('.jpg'):
        # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
        item = path + item
        img = cv2.imread(item)
        video.write(img)

video.release()
cv2.destroyAllWindows()
