from memory_profiler import profile
import motmetrics as mm  # 导入该库
import numpy as np

metrics = list(mm.metrics.motchallenge_metrics)  # # 即支持的所有metrics的名字列表
print(metrics)
acc = mm.MOTAccumulator(auto_id=True) # 创建accumulator
gt_file= "/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/gt/gt.txt"
# "/home/allenyljiang/Documents/Dataset/MOT16/train/MOT16-02/gt.txt"
"""  文件格式如下
1,0,1255,50,71,119,1,1,1
2,0,1254,51,71,119,1,1,1
3,0,1253,52,71,119,1,1,1
...
"""

ts_file= '/home/allenyljiang/Documents/Dataset/MOT20/train/MOT20-01/results_all/MOT20-01.txt'
# "/home/allenyljiang/Documents/Dataset/MOT16/train/MOT16-02/results_all/track.txt"
"""  文件格式如下
1,1,1240.0,40.0,120.0,96.0,0.999998,-1,-1,-1
2,1,1237.0,43.0,119.0,96.0,0.999998,-1,-1,-1
3,1,1237.0,44.0,117.0,95.0,0.999998,-1,-1,-1
...
"""

# gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)  # 读入GT   mot25-2d
gt = mm.io.loadtxt(gt_file, fmt="mot15-2D")  # 读入GT   mot25-2d
ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")  # 读入自己生成的跟踪结果

acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou',distth=0.5)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值

mh = mm.metrics.create()

# 打印单个accumulator
# summary = mh.compute(acc,
#                      metrics=['num_frames', 'mota', 'motp','mostly_tracked','mostly_lost','num_switches','num_fragmentations',
#                               'num_false_positives','partially_tracked','idf1'
#                               ],
#                      name='acc')
# 起个名
# metrics:一个list，里面装的是想打印的一些度量
# print(summary)

# # mh模块中有内置的显示格式
# summary = mh.compute_many([acc, acc.events.loc[0:1]],
#                           metrics=mm.metrics.motchallenge_metrics,
#                           names=['full', 'part'])

# mh模块中有内置的显示格式
summary = mh.compute(acc,metrics=mm.metrics.motchallenge_metrics,name='bo_ssp')

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)

# python -m memory_profiler eval_results.py









