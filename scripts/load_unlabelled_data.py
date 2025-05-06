import os
import sys
import pandas as pd
import json
import requests
import cv2
import shutil
import subprocess

# IJmond-VID dataset (unlabeled)
def load_ijmond_video():
    # Run download_videos.py
    try:
        subprocess.run(["python", "utils/download_videos.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running download_videos.py: {e}")
        return

    # Run preprocessing.py
    try:
        subprocess.run(["python", "utils/preprocessing.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running preprocessing.py: {e}")
        return

    # Now, let's put those frames in a single directory
    base_dir = '/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_VID/frames'
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Create new unique filename to avoid overwriting
                    new_filename = f"{folder}_{filename}"
                    src_path = os.path.join(folder_path, filename)
                    dst_path = os.path.join(base_dir, new_filename)

                    shutil.move(src_path, dst_path)
            # Optional: remove the now-empty subdirectory
            os.rmdir(folder_path)

    image_files = []
    for root, dirs, files in os.walk(base_dir):
        # Check if the current directory is a frame directory (numeric folder name)
        if os.path.basename(root).isdigit():
            for file in files:
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

if __name__ == "__main__":
    load_ijmond_video()
    load_rise()
