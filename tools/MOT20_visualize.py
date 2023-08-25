import argparse
import json
import os

import cv2
from itertools import count
from tqdm import tqdm


parse = argparse.ArgumentParser(description="Visualize MOT20 dataset of standard format")
parse.add_argument("--input_path", default='/home/allenyljiang/Documents/Dataset/MOT20',type=str, help="path of standard format dataset")
parse.add_argument("--output_path", default='/home/allenyljiang/Documents/Dataset/result',type=str, help="path of output")
parse.add_argument("--fps", type=int, default=25, help="frames per second")
args = parse.parse_args()


def visualize_sub_dataset(sub_dataset_name, media, labels, ann_source):
    for i, sequence in tqdm(enumerate(media), total=len(media), desc=f"{sub_dataset_name}-{ann_source}"):
        visualize_sequence(sub_dataset_name, i, sequence, labels, ann_source)


def visualize_sequence(sub_dataset_name, seq_id, sequence, labels, ann_source):
    frames = []
    colors = ((243, 129, 129), (252, 227, 138), (234, 255, 208), (149, 225, 211))
    color_id_gen = count()
    instance2color = {}
    for media_path in sequence:
        media_label = labels[media_path]
        frame = cv2.imread(os.path.join(args.input_path, media_path))
        if "ground_truth" in media_label and "box2d" in media_label["ground_truth"]:
            for bbox in media_label["ground_truth"]["box2d"]:
                if bbox["attributes"]["source"] == ann_source:
                    x, y, w, h = list(map(int, bbox["bounding_box"]))
                    instance_id = bbox.get("instance_id", "0")
                    if instance_id not in instance2color:
                        instance2color[instance_id] = colors[next(color_id_gen) % len(colors)]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), instance2color[instance_id])
        frames.append(frame)
    video_path = os.path.join(args.output_path, f"{sub_dataset_name}-{seq_id}-{ann_source}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, args.fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        video.write(frame)


def main():
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    elif os.listdir(args.output_path):
        raise OSError(f"Directory is not empty: '{args.output_path}'")

    annotated_sub_datasets = ["train"]
    with open(os.path.join(args.input_path, "dataset_info.json"), "r", encoding="utf8") as f:
        dataset_info = json.load(f)
    for sub_dataset_name, sub_dataset in dataset_info["data"].items():
        media = sub_dataset["media"]
        with open(os.path.join(args.input_path, sub_dataset["label"]), "r", encoding="utf8") as f:
            labels = json.load(f)
        labels = {ann["media"]: ann for ann in labels[sub_dataset_name]}

        if sub_dataset_name in annotated_sub_datasets:
            visualize_sub_dataset(sub_dataset_name, media, labels, "gt")
        visualize_sub_dataset(sub_dataset_name, media, labels, "det")


if __name__ == "__main__":
    main()