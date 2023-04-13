import cv2
'''
1、得到图像清晰度
'''
#利用拉普拉斯
def getImageVar(imgPath):
    image = cv2.imread(imgPath)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar
## train
imageVar_01 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/img1/000001.jpg')
imageVar_02 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-02/img1/000001.jpg')
imageVar_03 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-03/img1/000001.jpg')
imageVar_05 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-05/img1/000001.jpg')
## test
imageVar_04 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/test/MOT20-04/img1/000001.jpg')
imageVar_06 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/test/MOT20-06/img1/000001.jpg')
imageVar_07 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/test/MOT20-07/img1/000001.jpg')
imageVar_08 = getImageVar('/home/allenyljiang/Documents/Dataset/MOT20/test/MOT20-08/img1/000001.jpg')

print('MOT20-01 :{},MOT20-02:{},MOT20-03 :{},MOT20-04:{},MOT20-05 :{},MOT20-06:{},MOT20-07 :{},MOT20-08:{}'.format(imageVar_01,imageVar_02,imageVar_03,imageVar_04,imageVar_05,imageVar_06,imageVar_07,imageVar_08))