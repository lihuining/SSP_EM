import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import sys
sys.path.append('./')
# import tracker
# 需要以最终运行文件的相对位置导入为准
from yolox.tracker import matching
from tracker.gmc import GMC
from yolox.tracker.basetrack import BaseTrack, TrackState
from yolox.tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


from collections import OrderedDict
class BBaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = []
    start_frame = 0
    frames = [] # 当前窗口的frame为一个列表
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return max(self.frames).split('.')[0]

    @staticmethod
    def next_id(): # 下一条轨迹的id
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STTrack(BBaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self,cur_frames, xyxys, score,segment_len=12, feat=None, feat_history=50):

     # xyxys:当前窗口的bbox batch，xyxy 格式[]
     # score:当前窗口的score,[]
     # feat: 当前窗口feat
     # wait activate
        self.segment_len = segment_len # 窗口长度
        self.frames = cur_frames
        if xyxys:
            self.xyxys = [[xyxy[0][0],xyxy[0][1],xyxy[1][0],xyxy[1][1]]for xyxy in xyxys]
        self.xyxys_pred = self.xyxys.copy() # 使用卡尔曼滤波器进行值的预测.
        self.score = score
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0 # 原始的变量

        self.smooth_feat = None
        self.curr_feat = None # 最后一次跟新的app_feature

        self.features = deque([], maxlen=feat_history) # 保存历史的外观特征
        self.alpha = 0.9

        if feat is not None:
            self.update_features_list(feat)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat) #平滑的app_feature
    def update_features_list(self,feat_list):
        # update_faet_list
        for i in range(len(feat_list)):
            self.update_features(feat_list[i])

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        '''
        多条轨迹同时使用卡尔曼滤波器进行预测，不需要修改
        '''
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2] # Translation part

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frames):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        #  初始的1帧用于初始化，其余跟新
        n = len(self.frames)
        new_xyxys = []
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlbr_to_xywh(tlbr=self.tlbr_f))
        new_xyxys.append(self.tlbr)
        for i in range(1,n):
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,self.tlbr_to_xywh(tlbr=self.xyxys[i]))
            new_xyxys.append(self.tlbr)
        self.xyxys_pred = new_xyxys
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if int(max(frames).split('.')[0]) <= self.segment_len: # Todo: 第一个batch全部设置为True
            self.is_activated = True
        self.frames = frames.copy()
        # self.score = new_track.score.copy()
        self.start_frame = frames[0].split('.')[0] # 从哪一帧开始被激活

    def re_activate(self, new_track, frames, new_id=False):
        n = len(frames)
        new_xyxys = []
        for i in range(n):
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlbr_to_xywh(new_track.xyxys[i]))
            new_xyxys.append(self.tlbr)
        #
        self.xyxys_pred = new_xyxys
        self.xyxys = new_track.xyxys
        if new_track.features is not None:
            self.update_features_list(new_track.features)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frames = frames.copy()
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score.copy()

    def update(self, new_track, frames):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frames = frames.copy()
        self.tracklet_len += len(frames)

        #new_tlwh = new_track.tlwh
        n = len(frames)
        new_xyxys = []
        for i in range(n):
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlbr_to_xywh(new_track.xyxys[i]))
            new_xyxys.append(self.tlbr)
        self.xyxys_pred = new_xyxys
        self.xyxys = new_track.xyxys
        if new_track.features is not None:
            self.update_features_list(new_track.features)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score.copy()

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret
    @property
    def tlbr_l(self):
        return np.asarray(self.xyxys_pred[-1], dtype=np.float) # 最后一帧的结果(last)，用于窗口之间的关联

    @property
    def tlbr_f(self):
        return np.asarray(self.xyxys_pred[0], dtype=np.float) #  第一帧的结果

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self): # 中心点坐标
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    @staticmethod
    def tlbr_to_xywh(tlbr):
        tlwh = np.asarray(tlbr).copy()
        tlwh[2:] -= tlwh[:2]
        xywh = np.asarray(tlwh).copy()
        xywh[:2] += xywh[2:] / 2
        return xywh
        # Todo: 静态方法之间应该如何调用?
        # # tlwh = self.tlbr_to_tlwh(tlbr)
        # xywh = self.tlwh_to_xywh(tlwh)
        # return xywh



    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)




def write_segment_data():
    pass



class SSPTracker(object):
    def __init__(self, config, frame_rate=30):
        self.tracked_stracks = []  # type: list[STTrack]
        self.lost_stracks = []  # type: list[STTrack]
        self.removed_stracks = []  # type: list[STTrack]

        self.frame_id = 0 # 当前应该跟踪的segment第一帧，调用update的时候进行更新
        self.frames = []
        self.batch_id = 0 # 滑动窗口id
        self.config = config
        self.tracklet_len = config["tracklet_len"]
        self.update_len = config["batch_stride_write"]
        self.track_high_thresh = config["track_high_thresh"] # 0.6
        self.track_low_thresh = config["track_low_thresh"] # 0.1
        self.new_track_thresh = config["new_track_thresh"] # 0.7

        self.buffer_size = int(frame_rate / 30.0 * config["track_buffer"]) # 30
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = config["proximity_thresh"] # 0.5
        self.appearance_thresh = config["appearance_thresh"] # 0.25
        # Iou
        self.iou_proximity_thresh = config["iou_proximity_thresh"]

        # if config["with_reid"]:
        #     self.encoder = FastReIDInterface(config["fast_reid_config"], config["fast_reid_weights"], config["device"])
        #     # '../fast_reid/configs/MOT17/sbs_S50.yml','../pretrained/mot17_sbs_S50.pth',cuda
        #
        # self.gmc = GMC(method=config["cmc_method"], verbose=[config["name"], config["ablation"]]) # method = file，verbose = [seq name, 是否ablation]

    def update(self,cur_segment,config,frames): # cur_video_segment_predicted_tracks_bboxes_test
        self.frame_id += self.update_len
        self.batch_id += 1
        self.frames = frames.copy() # 当前滑动窗口内的帧
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        # 当前窗口的信息
        cur_frames = [[cur_segment[key][node][0] for node in cur_segment[key]] for key in cur_segment]
        cur_detections = [[cur_segment[key][node][1] for node in cur_segment[key]] for key in cur_segment] # list, bboxes,list当中每个元素也是list
        cur_scores = [[cur_segment[key][node][2] for node in cur_segment[key]] for key in cur_segment]
        cur_features = [[cur_segment[key][node][3] for node in cur_segment[key]] for key in cur_segment]
        if len(cur_frames) > 0:
            '''Detections'''
            if self.config["with_reid"]:#cur_frames, xyxys, score,segment_len=12
                detections = [STTrack(frames, detections, scores,config["tracklet_len"],features) for
                              (frames, detections, scores,features) in zip(cur_frames, cur_detections,cur_scores,cur_features)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STTrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STTrack.multi_predict(strack_pool)

        # # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance_segments(strack_pool, detections) # 使用修正之后的与当前的检测进行匹配
        ious_dists_mask = (ious_dists > self.iou_proximity_thresh) # 0.5

        # if not self.config["mot20"]:
        ious_dists = matching.fuse_score_segment(ious_dists, detections)

        if self.config["with_reid"]:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0 # 0.5*cosinedist (36,35)
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0 # >0.25
            emb_dists[ious_dists_mask] = 1.0 # 之考虑iou小于0.5的emb距离
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.config["match_thresh"]) # 0.7

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], detections[idet].frames)
                activated_starcks.append(track)
            else:
                track.re_activate(det, det.frames, new_id=False)
                refind_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]

        ious_dists = matching.iou_distance_segments(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.iou_proximity_thresh ) # 0.5
        # if not self.config["mot20"]:
        ious_dists = matching.fuse_score_segment(ious_dists, detections)

        if self.config["with_reid"]:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0 # appearance_thresh = 0.25
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
            # emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            # emb_dists[emb_dists > self.appearance_thresh] = 1.0 # appearance_thresh = 0.25
            # dists = emb_dists
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], detections[idet].frames)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if max(track.score) < self.new_track_thresh: # 0.7
                continue

            track.activate(self.kalman_filter, track.frames)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    '''
    轨迹合并到一个集合，不需要修改
    '''
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    #pdist = matching.iou_distance(stracksa, stracksb) # iou_distance_segments
    pdist = matching.iou_distance_segments(stracksa, stracksb) # iou_distance_segments
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].end_frame - stracksa[p].start_frame
        timeq = stracksb[q].end_frame - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
import yaml
if __name__ == "__main__":
    cfg_file = "/home/allenyljiang/Documents/SSP_EM/configs/avenue.yaml"
    config = yaml.safe_load(open(cfg_file))
    tracker = SSPTracker(config)
    print(tracker)
