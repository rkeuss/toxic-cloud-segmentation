import json
import os
from sklearn.model_selection import train_test_split

def split_coco_annotations(coco_json_path, output_dir, train_size=0.8, random_state=42):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    image_ids = [img['id'] for img in coco_data['images']]
    train_ids, test_ids = train_test_split(image_ids, train_size=train_size, random_state=random_state)

    def filter_data(image_ids):
        images = [img for img in coco_data['images'] if img['id'] in image_ids]
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]
        return {
            'images': images,
            'annotations': annotations,
            'categories': coco_data['categories']
        }

    train_data = filter_data(train_ids)
    test_data = filter_data(test_ids)

    os.makedirs(output_dir, exist_ok=True)
    train_json_path = os.path.join(output_dir, 'train.json')
    test_json_path = os.path.join(output_dir, 'test.json')

    with open(train_json_path, 'w') as f:
        json.dump(train_data, f)
    with open(test_json_path, 'w') as f:
        json.dump(test_data, f)

    print(f"Train and test splits saved to {output_dir}")

coco_json_path = os.path.abspath('../data/IJMOND_SEG/cropped/cropped_annotations.json')
output_dir = '../data/IJMOND_SEG/cropped/splits'
split_coco_annotations(coco_json_path, output_dir)