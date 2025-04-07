import os
import sys
import pandas as pd
import json
import requests
import cv2
from pycocotools.coco import COCO
import ijmond_segmentation_dataset as ijmond_seg

# IJmond-SEG dataset (labeled)
def load_ijmond_segmented():
    json_path_annotations = "data/dataset/IJMOND_SEG/test/_annotations.coco.json"
    json_path_images = "data/dataset/IJMOND_SEG/test/"
    coco_data = COCO(json_path_annotations)
    imgIds = coco_data.getImgIds()
    ijmond_seg_dataset = ijmond_seg.COCOSegmentationDataset(coco_data, imgIds, json_path_images)
    return ijmond_seg_dataset

# IJmond-VID dataset (unlabeled)
def load_ijmond_video():
    base_path = "data/dataset/IJMOND_VID"
    image_files = []

    for root, dirs, files in os.walk(base_path):
        # Check if the current directory is a frame directory (numeric folder name)
        if os.path.basename(root).isdigit():
            for file in files:
                if file.endswith(".png"):
                    image_files.append(os.path.join(root, file))

    ijmond_vid_dataset = pd.DataFrame(image_files, columns=["image_path"])
    return ijmond_vid_dataset

# RISE (unlabeled)
def load_rise():
    metadata_file_path = 'data/RISE/metadata.json'
    output_dir = 'data/RISE/extracted_frames/'

    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)

    positive_videos = [
        video for video in metadata
        if video['label_state'] in [47, 23] or video['label_state_admin'] in [47, 23]
    ]

    def download_video(url_root, url_part, save_path):
        video_url = url_root + url_part
        response = requests.get(video_url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def extract_frame(video_path, frame_number=0):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if ret:
            return frame
        else:
            print(f"Failed to read frame {frame_number} from {video_path}")
            return None

    for video in positive_videos:
        file_name = video['file_name']
        url_root = video['url_root'].replace('/180/', '/320/')
        url_part = video['url_part'].replace('-180-180-', '-320-320-')

        # Determine the full path to save the video
        video_file_path = os.path.join(output_dir, f"{file_name}.mp4")

        # Download the video (if not already downloaded)
        if not os.path.exists(video_file_path):
            download_video(url_root, url_part, video_file_path)

        frame = extract_frame(video_file_path, frame_number=0)
        if frame is not None:
            frame_file_path = os.path.join(output_dir, f"{file_name}_frame.png")
            cv2.imwrite(frame_file_path, frame)


def combine_unlabelled_data(ijmond_vid, rise):
    ...
    # TODO: think of way to only include frames with smoke (or exclude those without)

def main():
    ijmond_seg = load_ijmond_segmented()
    ijmond_vid = load_ijmond_video()
    rise = load_rise()
    unlabeled_data = combine_unlabelled_data(ijmond_vid, rise)

if __name__ == "__main__":
    main(sys.argv)