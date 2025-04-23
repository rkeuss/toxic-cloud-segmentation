import os
import json
import requests
import cv2

import os
import shutil

base_dir = '/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/RISE'
frames_dir = os.path.join(base_dir, 'frames')
videos_dir = os.path.join(base_dir, 'videos')
os.makedirs(videos_dir, exist_ok=True)

# Move .mp4 files
for filename in os.listdir(frames_dir):
    if filename.lower().endswith('.mp4'):
        src_path = os.path.join(frames_dir, filename)
        dst_path = os.path.join(videos_dir, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")
print("✅ Done moving all .mp4 files to /videos/")
