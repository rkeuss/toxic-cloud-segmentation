import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image


class IJmondSegDataset(Dataset):
    def __init__(self, coco, imgIds, img_dir, transform=None):
        self.coco = coco
        self.imgIds = imgIds
        self.img_dir = img_dir
        self.transform = transform
        self.category_mapping = self._create_binary_mapping(coco)

    def _create_binary_mapping(self, coco):
        """
        Create a mapping of category IDs to binary labels: 1 for 'smoke', 0 for 'none'.
        """
        category_mapping = {}
        for category in coco.loadCats(coco.getCatIds()):
            category_mapping[category['id']] = 1 if category['supercategory'] == 'smoke' else 0
        return category_mapping

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img_id = self.imgIds[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)  # OpenCV loads images as (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            mask += self.coco.annToMask(ann) * self.category_mapping[ann['category_id']]

        # Apply augmentations if defined
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure PyTorch expects (C, H, W) format
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert (H, W, C) → (C, H, W)

        mask = torch.from_numpy(mask).long()

        _, H, W = image.shape  # Get current image dimensions
        new_H = ((H + 15) // 16) * 16  # Round up to nearest multiple of 16
        new_W = ((W + 15) // 16) * 16

        pad_H = new_H - H
        pad_W = new_W - W

        image = F.pad(image, (0, pad_W, 0, pad_H), mode="constant", value=0)
        mask = F.pad(mask, (0, pad_W, 0, pad_H), mode="constant", value=0)

        return image, mask

def get_ijmond_seg_dataloader(image_dir, split, batch_size, shuffle=True):
    """
    Load the COCO dataset for the specified split (train or test).
    """
    coco_annotations = f"{image_dir}/splits/{split}.json"  # Use the split JSON file
    coco = COCO(coco_annotations)
    imgIds = coco.getImgIds()
    transform = transforms.Compose([  # TODO: edit transformations
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = IJmondSegDataset(coco, imgIds, image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class UnlabelledDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        self.image_dirs = image_dirs
        self.transform = transform
        self.image_list = []
        self.pseudo_labels = [None] * len(self.image_list)  # Placeholder for pseudo-labels

        for image_dir in image_dirs:
            for img_name in os.listdir(image_dir):
                self.image_list.append(os.path.join(image_dir, img_name))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        pseudo_label = self.pseudo_labels[idx]  # Retrieve pseudo-label if available
        return image, pseudo_label

    def update_pseudo_labels(self, new_pseudo_labels):
        """Update pseudo-labels dynamically."""
        self.pseudo_labels = new_pseudo_labels

def get_unlabelled_dataloader(image_dirs, batch_size, shuffle=True):
    transform = transforms.Compose([  # TODO: edit transformations
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = UnlabelledDataset(image_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader