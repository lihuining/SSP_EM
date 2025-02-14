# An Approach for Multi-Object Tracking with Two-Stage Min-Cost Flow

## Abstract
The minimum network flow algorithm is widely used in multi-target tracking. However, the majority of the
present methods concentrate exclusively on minimizing cost functions whose values may not indicate accurate solutions under occlusions. In this paper, by exploiting the properties of tracklets intersections and low-confidence detections, we develop a two stage tracking pipeline with an intersection mask that can
accurately locate inaccurate tracklets which are corrected in the second stage. Specifically, we employ the minimum network flow algorithm with high-confidence detections as input in the first stage to obtain the candidate tracklets that need correction. Then we leverage the intersection mask to accurately locate the
inaccurate parts of candidate tracklets. The second stage utilizes low-confidence detections that may be attributed to occlusions for correcting inaccurate tracklets. This process constructs a graph of nodes in inaccurate tracklets and low-confidence nodes and uses it for the second round of minimum network flow calculation. We perform sufficient experiments on popular MOT benchmark datasets and achieve 78.4 MOTA on the test set of MOT16, 79.2 on MOT17, and 76.4 on MOT20, which shows that the proposed
method is effective.

## Tracking Performance
### Results on MOT challenge test set



| Dataset   | MOTA | IDF1 | HOTA | MT   | ML   | AssA | DetA | LocA |
| --------- | :--: | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| **MOT16** | 78.4 | 72.9 | 61   | 407  | 92   | 58.4 | 64   | 83.1 |
| **MOT17** | 79.2 | 72.9 | 60.7 | 1236 | 309  | 57.9 | 64   | 83.2 |
| **MOT20** | 76.4 | 71.6 | 60.3 | 886  | 91   | 57.5 | 63.6 | 83.8 |

### Visualization results on MOT challenge test set

MOT20-03

<img src="assets/MOT20-03.gif" width="400"/>

## Installation

```
git clone https://github.com/lihuining/TSMCF.git
cd TSMCF

pip3 install -r requirements.
```

## Data structure
1. download [call](https://drive.google.com/file/d/1XZyZT1EE9tYjZCqsISoqGkgruUrXGJTj/view?usp=drive_link) and put it into call
2. download [bytetrack_x_mot17.pth.tar](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing) , [bytetrack_x_mot20.tar](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view) , [mot17_sbs_S50.pth](https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view?usp=sharing) ,[mot20_sbs_S50.pth](https://drive.google.com/file/d/1KqPQyj6MFyftliBHEIER7m_OrGpcrJwi/view?usp=sharing) and put them into models
````
--call # min-cost-flow C++算法
--exps # 每个数据集的检测配置文件
--fast_reid # 使用的reid库
	--configs
		--MOT17
		--MOT20
--figs # 论文中使用到的一些图
--models # 使用到的检测和reid模型
--similarity_module # 原始的reid模型
--statisctic_information(not used) # 一些统计信息
--tools # 可视化或者统计信息的脚本
	convert_frame_to_video.py # 将生成的图像转化为video
--tracker # bot-sort当中关于相机运动补偿的一些东西
--unused_previous_version # 之前使用的一些版本
<!-- --weights # 之前使用的权重,yolov5 -->
--yolox # yolox检测器

--call/call # SSP算法封装文件
````

## Inference

args默认使用default_parameters.文件当中default_parameters下修改路径

download detect and reid models to current directory models

```

python3 tracker_MOT20.py --path /media/allenyljiang/5234E69834E67DFB/Dataset/MOT20 -f exps/example/mot/yolox_x_mix_mot20_ch.py --eval train -c models/bytetrack_x_mot20.tar --benchmark MOT20 --fast-reid-weights models/mot20_sbs_S50.pth

```



### 待办事项

整理txt文件与vis文件路径

# Questions

1. If meet Permission denied using "call/call", try to run the following command in terminal:

```
chmod +x call/call
```

