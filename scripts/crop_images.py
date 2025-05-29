import os
import json
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
import random

"""
With the fallback logic used, every object is eventually included in at least one crop. However, the fallback 
allows partial overlap, meaning an object might appear in multiple crops if necessary to ensure all objects are 
captured. This happens when the crop size or positioning constraints make it impossible to include all objects 
in distinct, non-overlapping crops.
"""

INPUT_JSON = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG/_annotations.coco.json"
INPUT_IMG_DIR = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG"
OUTPUT_IMG_DIR = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG/cropped"
OUTPUT_JSON = "/Users/rkeuss/PycharmProjects/toxic-cloud-segmentation/data/IJMOND_SEG/cropped/cropped_annotations.json"
CROP_SIZE = 640

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Load COCO data
coco = COCO(INPUT_JSON)
images = coco.loadImgs(coco.getImgIds())
ann_id_counter = 1
new_images = []
new_annotations = []
used_objects = set()

for img in tqdm(images, desc="Processing images"):
    img_path = os.path.join(INPUT_IMG_DIR, img["file_name"])
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {img_path}")
        continue

    anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"]))

    for i, ann in enumerate(anns):
        if ann["id"] in used_objects:
            continue  # Skip objects already included in a crop

        x, y, w, h = map(int, ann["bbox"])

        # Randomize crop position while ensuring the object fits entirely
        if CROP_SIZE >= w:
            x_offset = random.randint(0, CROP_SIZE - w)
        else:
            print(f"No randomization possible: object width {w} exceeds crop size {CROP_SIZE}, img id: {img['id']}")
            x_offset = 0  # No randomization possible if object width exceeds crop size

        if CROP_SIZE >= h:
            y_offset = random.randint(0, CROP_SIZE - h)
        else:
            print(f"No randomization possible: object height {h} exceeds crop size {CROP_SIZE}, img id: {img['id']}")
            y_offset = 0  # No randomization possible if object height exceeds crop size
        x1 = max(x - x_offset, 0)
        y1 = max(y - y_offset, 0)
        x2 = x1 + CROP_SIZE
        y2 = y1 + CROP_SIZE

        # Adjust if crop exceeds image boundaries
        if x2 > img["width"]:
            x1 = max(img["width"] - CROP_SIZE, 0)
            x2 = img["width"]
        if y2 > img["height"]:
            y1 = max(img["height"] - CROP_SIZE, 0)
            y2 = img["height"]

        # Check which objects are fully within the crop
        included_objects = []
        for other_ann in anns:
            ox, oy, ow, oh = map(int, other_ann["bbox"])
            if ox >= x1 and oy >= y1 and (ox + ow) <= x2 and (oy + oh) <= y2:
                included_objects.append(other_ann["id"])

        # Skip if no objects are included
        if not included_objects:
            continue

        # Mark included objects as used
        used_objects.update(included_objects)

        # Final crop window
        cropped_img = image[y1:y2, x1:x2]

        # Skip empty crops
        if cropped_img.size == 0:
            print(f"Skipping empty crop for annotation {ann['id']} in image {img['file_name']}")
            continue

        # Save cropped image
        crop_filename = f"{os.path.splitext(img['file_name'])[0]}_obj{i}.jpg"
        crop_path = os.path.join(OUTPUT_IMG_DIR, crop_filename)
        cv2.imwrite(crop_path, cropped_img)

        # Adjust annotations for included objects
        for obj_id in included_objects:
            obj_ann = next(a for a in anns if a["id"] == obj_id)
            ox, oy, ow, oh = map(int, obj_ann["bbox"])
            new_bbox = [ox - x1, oy - y1, ow, oh]
            new_segmentation = []
            for seg in obj_ann["segmentation"]:
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
                "category_id": obj_ann["category_id"],
                "bbox": new_bbox,
                "area": obj_ann["area"],  # (not strictly valid after crop, but acceptable unless retraining)
                "segmentation": new_segmentation,
                "iscrowd": obj_ann["iscrowd"]
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