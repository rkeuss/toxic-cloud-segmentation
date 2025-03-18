from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCOSegmentationDataset(Dataset):
    def __init__(self, coco, imgIds, img_dir, transform=None):
        self.coco = coco
        self.imgIds = imgIds
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img_id = self.imgIds[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)  # OpenCV loads images as (H, W, C)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"DEBUG: Original image shape (H, W, C) = {image.shape}")  # Debugging

        # Load mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            mask += self.coco.annToMask(ann) * ann['category_id']

        # Apply augmentations if defined
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure PyTorch expects (C, H, W) format
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert (H, W, C) → (C, H, W)

        print(f"DEBUG: Final shape before padding (C, H, W) = {image.shape}")  # Debugging

        # ✅ Convert mask to PyTorch tensor before padding
        mask = torch.tensor(mask, dtype=torch.long)

        # ✅ Ensure Height & Width are Divisible by 16 (Padding Instead of Resizing)
        _, H, W = image.shape  # Get current image dimensions
        new_H = ((H + 15) // 16) * 16  # Round up to nearest multiple of 16
        new_W = ((W + 15) // 16) * 16

        pad_H = new_H - H
        pad_W = new_W - W

        # ✅ Pad image and mask
        image = F.pad(image, (0, pad_W, 0, pad_H), mode="constant", value=0)
        mask = F.pad(mask, (0, pad_W, 0, pad_H), mode="constant", value=0)

        print(f"DEBUG: Final shape after padding (C, H, W) = {image.shape}")  # Debugging

        return image, mask
