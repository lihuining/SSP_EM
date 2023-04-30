import time
import numpy as np
import cv2

def evaluate_prediction(dataloader, data_dict):
    if not is_main_process():
        return 0, 0, None

    logger.info("Evaluate in main process...")

    annType = ["segm", "bbox", "keypoints"]

    # inference_time = statistics[0].item()
    # track_time = statistics[1].item()
    # n_samples = statistics[2].item()

    # a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
    # a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

    # time_info = ", ".join(
    #     [
    #         "Average {} time: {:.2f} ms".format(k, v)
    #         for k, v in zip(
    #             ["forward", "track", "inference"],
    #             [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
    #         )
    #     ]
    # )

    # info = time_info + "\n"

    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_dict) > 0:
        cocoGt = dataloader.dataset.coco

        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)
        '''
        try:
            from yolox.layers import COCOeval_opt as COCOeval
        except ImportError:
            from pycocotools import cocoeval as COCOeval
            logger.warning("Use standard COCOeval.")
        '''
        #from pycocotools.cocoeval import COCOeval
        from yolox.layers import COCOeval_opt as COCOeval
        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        # info += redirect_string.getvalue()
        return cocoEval.stats[0], cocoEval.stats[1]
    else:
        return 0, 0

def people_matching_error_average_sampling(curr_person_bbox_coord, next_person_bbox_coord, curr_img, next_img, average_sampling_density_hori_vert):
    time_start_sampling_step1 = time.time()
    bbox_left_curr = curr_person_bbox_coord[0][0]
    bbox_top_curr = curr_person_bbox_coord[0][1]
    bbox_right_curr = curr_person_bbox_coord[1][0]
    bbox_bottom_curr = curr_person_bbox_coord[1][1]
    bbox_left_next = next_person_bbox_coord[0][0]
    bbox_top_next = next_person_bbox_coord[0][1]
    bbox_right_next = next_person_bbox_coord[1][0]
    bbox_bottom_next = next_person_bbox_coord[1][1]

    # bbox_left_curr = bbox_left_curr * 0.8 + bbox_right_curr * 0.2
    # bbox_right_curr = bbox_left_curr * 0.25 + bbox_right_curr * 0.75
    # bbox_top_curr = bbox_top_curr * 0.8 + bbox_bottom_curr * 0.2
    # bbox_bottom_curr = bbox_top_curr * 0.25 + bbox_bottom_curr * 0.75
    # bbox_left_next = bbox_left_next * 0.8 + bbox_right_next * 0.2
    # bbox_right_next = bbox_left_next * 0.25 + bbox_right_next * 0.75
    # bbox_top_next = bbox_top_next * 0.8 + bbox_bottom_next * 0.2
    # bbox_bottom_next = bbox_top_next * 0.25 + bbox_bottom_next * 0.75

    curr_hori_coords = np.linspace(bbox_left_curr, bbox_right_curr, num=average_sampling_density_hori_vert, endpoint=False)
    curr_vert_coords = np.linspace(bbox_top_curr, bbox_bottom_curr, num=average_sampling_density_hori_vert, endpoint=False)
    next_hori_coords = np.linspace(bbox_left_next, bbox_right_next, num=average_sampling_density_hori_vert, endpoint=False)
    next_vert_coords = np.linspace(bbox_top_next, bbox_bottom_next, num=average_sampling_density_hori_vert, endpoint=False)

    batch_v_range_curr = np.repeat(curr_vert_coords, average_sampling_density_hori_vert).astype(int)
    batch_h_range_curr = np.tile(curr_hori_coords, average_sampling_density_hori_vert).astype(int)
    batch_curr = curr_img[batch_v_range_curr, batch_h_range_curr, :].flatten().astype(np.int32)

    batch_v_range_next = np.repeat(next_vert_coords, average_sampling_density_hori_vert).astype(int)
    batch_h_range_next = np.tile(next_hori_coords, average_sampling_density_hori_vert).astype(int)
    batch_next = next_img[batch_v_range_next, batch_h_range_next, :].flatten().astype(np.int32)

    time_start_sampling_method2 = time.time()
    joint_to_joint_matching_matrix = np.ones(
        (len(batch_v_range_curr), len(batch_v_range_next))) * maximum_possible_number
    for curr_img_pixel_idx in range(len(batch_v_range_curr)):
        batch_curr_rolled = np.roll(batch_curr, 3 * curr_img_pixel_idx)
        vert_coords = np.roll(range(len(batch_v_range_curr)), curr_img_pixel_idx)
        hori_coords = range(len(batch_v_range_next))
        joint_to_joint_matching_matrix[vert_coords, hori_coords] = np.sum((batch_curr_rolled.reshape((len(batch_v_range_curr), 3)) - batch_next.reshape((len(batch_v_range_next), 3))) * (batch_curr_rolled.reshape((len(batch_v_range_curr), 3)) - batch_next.reshape((len(batch_v_range_next), 3))), axis=1)
    time_end_sampling_method2 = time.time()
    return joint_to_joint_matching_matrix



def compute_inter_person_similarity(curr_frame_dict, next_frame_dict, maximum_possible_number, curr_img, next_img):
    time_start_compute_inter_person_similarity = time.time()
    time_start_initial = time.time()
    person_to_person_matching_matrix = np.ones((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list']))) * maximum_possible_number
    person_to_person_matching_matrix_iou = np.zeros((len(curr_frame_dict['bbox_list']), len(next_frame_dict['bbox_list'])))
    time_end_initial = time.time()
    for curr_person_bbox_coord in curr_frame_dict['bbox_list']:
        for next_person_bbox_coord in next_frame_dict['bbox_list']:
            time_start_sampling = time.time()
            joint_to_joint_matching_matrix = people_matching_error_average_sampling(curr_person_bbox_coord, next_person_bbox_coord, curr_img, next_img, average_sampling_density_hori_vert)
            time_end_sampling = time.time()
            ot_src = [1.0] * joint_to_joint_matching_matrix.shape[0]
            ot_dst = [1.0] * joint_to_joint_matching_matrix.shape[1]
            time_start_ot = time.time()
            transportation_array = ot.emd(ot_src, ot_dst, joint_to_joint_matching_matrix) # sinkhorn(ot_src, ot_dst, joint_to_joint_matching_matrix, 1, method='greenkhorn') # method='sinkhorn_stabilized')
            time_end_ot = time.time()
            time_start_sum = time.time()
            person_to_person_matching_matrix[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = np.sum(joint_to_joint_matching_matrix * transportation_array)
            time_end_sum = time.time()
            time_start_iou = time.time()
            person_to_person_matching_matrix_iou[curr_frame_dict['bbox_list'].index(curr_person_bbox_coord), next_frame_dict['bbox_list'].index(next_person_bbox_coord)] = \
                compute_iou_single_box([curr_person_bbox_coord[0][1], curr_person_bbox_coord[1][1], curr_person_bbox_coord[0][0], curr_person_bbox_coord[1][0]], \
                                       [next_person_bbox_coord[0][1], next_person_bbox_coord[1][1], next_person_bbox_coord[0][0], next_person_bbox_coord[1][0]])
            time_end_iou = time.time()
    time_start_post = time.time()
    person_to_person_matching_matrix = 1.0 / person_to_person_matching_matrix / np.max(1.0 / person_to_person_matching_matrix) # similarity
    person_to_person_matching_matrix = person_to_person_matching_matrix * person_to_person_matching_matrix_iou
    person_to_person_matching_matrix_normalized = person_to_person_matching_matrix / np.min(person_to_person_matching_matrix[np.where(person_to_person_matching_matrix!=0)])
    time_end_post = time.time()
    time_end_compute_inter_person_similarity = time.time()
    return person_to_person_matching_matrix_normalized


def people_matching_error(joint_to_joint_matching_matrix, curr_person_all_joint_coord, joint_coord_curr_frame, next_person_all_joint_coord, joint_coord_next_frame, number_points_around_each_joint, curr_img, next_img):
    side_of_sampling_equilateral_triangle = 999.0
    for bone in skeletons:
        if curr_person_all_joint_coord.index(joint_coord_curr_frame) in bone:
            idx_of_neighbor_coord = [x for x in bone if x != curr_person_all_joint_coord.index(joint_coord_curr_frame)][0]
            collection_distance_with_neighbors = np.sqrt(
                pow((joint_coord_curr_frame[0] - curr_person_all_joint_coord[idx_of_neighbor_coord][0]), 2) + pow((joint_coord_curr_frame[1] - curr_person_all_joint_coord[idx_of_neighbor_coord][1]), 2))
            side_of_sampling_equilateral_triangle = min([max([(collection_distance_with_neighbors / 1.732 / 5.0), 3]), side_of_sampling_equilateral_triangle])
    keypoints_around_joint_curr_frame = sampling_process(joint_coord_curr_frame, side_of_sampling_equilateral_triangle, curr_img.shape[0], curr_img.shape[1], number_points_around_each_joint)
    #############################################################################################################
    side_of_sampling_equilateral_triangle = 999.0
    for bone in skeletons:
        if next_person_all_joint_coord.index(joint_coord_next_frame) in bone:
            idx_of_neighbor_coord = [x for x in bone if x != next_person_all_joint_coord.index(joint_coord_next_frame)][0]
            collection_distance_with_neighbors = np.sqrt(
                pow((joint_coord_next_frame[0] - next_person_all_joint_coord[idx_of_neighbor_coord][0]), 2) + pow((joint_coord_next_frame[1] - next_person_all_joint_coord[idx_of_neighbor_coord][1]), 2))
            side_of_sampling_equilateral_triangle = min([max([(collection_distance_with_neighbors / 1.732 / 5.0), 3]), side_of_sampling_equilateral_triangle])
    keypoints_around_joint_next_frame = sampling_process(joint_coord_next_frame, side_of_sampling_equilateral_triangle, next_img.shape[0], next_img.shape[1], number_points_around_each_joint)
    for number_points_around_each_joint_idx in range(number_points_around_each_joint):
        joint_to_joint_matching_matrix[3*curr_person_all_joint_coord.index(joint_coord_curr_frame)+number_points_around_each_joint_idx, \
                                       3*next_person_all_joint_coord.index(joint_coord_next_frame)+number_points_around_each_joint_idx] = \
            np.linalg.norm(curr_img[int(keypoints_around_joint_curr_frame[number_points_around_each_joint_idx][1]), int(keypoints_around_joint_curr_frame[number_points_around_each_joint_idx][0]), :] - \
                           next_img[int(keypoints_around_joint_next_frame[number_points_around_each_joint_idx][1]), int(keypoints_around_joint_next_frame[number_points_around_each_joint_idx][0]), :])

    return joint_to_joint_matching_matrix

def vis_det_pose_results(save_img, dataset, save_path, vid_path, vid_writer, vid_cap, im0):
    if save_img:
        if dataset.mode == 'images':
            cv2.imwrite(save_path, im0)
        else:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(im0)