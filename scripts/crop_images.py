import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

INPUT_JSON = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG/_annotations.coco.json"
INPUT_IMG_DIR = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG"
OUTPUT_IMG_DIR = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG/cropped"
OUTPUT_JSON = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG/cropped/cropped_annotations.json"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Load COCO data
coco = COCO(INPUT_JSON)
images = coco.loadImgs(coco.getImgIds())
ann_id_counter = 1
new_images = []
new_annotations = []

for img in tqdm(images, desc="Processing images"):
    img_path = os.path.join(INPUT_IMG_DIR, img["file_name"])
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {img_path}")
        continue

    anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"]))

    for i, ann in enumerate(anns):
        x, y, w, h = map(int, ann["bbox"])

        min_crop_size = 650

        # Original object bounds
        x, y, w, h = map(int, ann["bbox"])
        x2 = x + w
        y2 = y + h

        # Calculate required extra width/height
        extra_w = max(min_crop_size - w, 0)
        extra_h = max(min_crop_size - h, 0)

        # Distribute extra size (can be asymmetric)
        x1 = max(x - extra_w // 3, 0)
        y1 = max(y - extra_h // 3, 0)
        x2 = min(x2 + (2 * extra_w) // 3, img["width"])
        y2 = min(y2 + (2 * extra_h) // 3, img["height"])

        # Adjust again if resulting size is still too small due to image edges
        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < min_crop_size:
            if x1 == 0:
                x2 = min(min_crop_size, img["width"])
            else:
                x1 = max(x2 - min_crop_size, 0)

        if crop_h < min_crop_size:
            if y1 == 0:
                y2 = min(min_crop_size, img["height"])
            else:
                y1 = max(y2 - min_crop_size, 0)

        # Final crop window
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cropped_img = image[y1:y2, x1:x2]

        # Skip empty crops
        if cropped_img.size == 0:
            continue

        # Save cropped image
        crop_filename = f"{os.path.splitext(img['file_name'])[0]}_obj{i}.jpg"
        crop_path = os.path.join(OUTPUT_IMG_DIR, crop_filename)
        cv2.imwrite(crop_path, cropped_img)

        # Adjust annotation to new coordinates
        new_bbox = [x - x1, y - y1, w, h]
        new_segmentation = []
        for seg in ann["segmentation"]:
            new_seg = []
            for j in range(0, len(seg), 2):
                new_seg.append(seg[j] - x1)
                new_seg.append(seg[j + 1] - y1)
            new_segmentation.append(new_seg)

        new_image_id = len(new_images)
        new_images.append({
            "id": new_image_id,
            "file_name": crop_filename,
            "width": cropped_img.shape[1],
            "height": cropped_img.shape[0]
        })

        new_annotations.append({
            "id": ann_id_counter,
            "image_id": new_image_id,
            "category_id": ann["category_id"],
            "bbox": new_bbox,
            "area": ann["area"],  # (not strictly valid after crop, but acceptable unless retraining)
            "segmentation": new_segmentation,
            "iscrowd": ann["iscrowd"]
        })

        ann_id_counter += 1

# Save new COCO JSON
output_json = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco.loadCats(coco.getCatIds())
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output_json, f, indent=2)

print(f"\n✅ Done! Cropped images saved to: {OUTPUT_IMG_DIR}")
print(f"📄 New annotation file: {OUTPUT_JSON}")
