import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

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
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert (H, W, C) → (C, H, W)
        mask = torch.from_numpy(mask).long()

        _, H, W = image.shape  # Get current image dimensions
        new_H = ((H + 15) // 16) * 16  # Round up to nearest multiple of 16
        new_W = ((W + 15) // 16) * 16

        pad_H, pad_W = (new_H - H), (new_W - W)
        image = F.pad(image, (0, pad_W, 0, pad_H), mode="constant", value=0)
        mask = F.pad(mask, (0, pad_W, 0, pad_H), mode="constant", value=0)

        return image, mask


def get_ijmond_seg_dataset(image_dir, split):
    """
    Load the COCO dataset for the specified split (train or test).
    """
    coco_annotations = f"{image_dir}/splits/{split}.json"
    coco = COCO(coco_annotations)
    imgIds = coco.getImgIds()
    dataset = IJmondSegDataset(coco, imgIds, image_dir, transform=None)
    return dataset

def get_ijmond_seg_dataloader_train(train_idx, split, batch_size, shuffle=True):
    image_dir = 'data/IJMOND_SEG'
    coco_annotations = f"{image_dir}/splits/{split}.json"
    coco = COCO(coco_annotations)
    imgIds = coco.getImgIds()

    # Filter imgIds to only include those in train_idx
    train_imgIds = [imgIds[i] for i in train_idx]

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ToTensorV2(),
    ])

    dataset = IJmondSegDataset(coco, train_imgIds, image_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_ijmond_seg_dataloader_validation(val_idx, split, batch_size, shuffle=True):
    image_dir = 'data/IJMOND_SEG'
    coco_annotations = f"{image_dir}/splits/{split}.json"
    coco = COCO(coco_annotations)
    imgIds = coco.getImgIds()

    # Filter imgIds to only include those in val_idx
    val_imgIds = [imgIds[i] for i in val_idx]

    # For test/validation, no augmentation
    val_transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])

    dataset = IJmondSegDataset(coco, val_imgIds, image_dir, transform=val_transform)
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
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = ToTensorV2()(image=image)["image"]

        pseudo_label = self.pseudo_labels[idx]  # Retrieve pseudo-label if available
        return image, pseudo_label

    def update_pseudo_labels(self, new_pseudo_labels):
        """Update pseudo-labels dynamically."""
        self.pseudo_labels = new_pseudo_labels


def get_unlabelled_dataloader(image_dirs, batch_size, shuffle=True):
    strong_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.5),
        ToTensorV2(),
    ])

    dataset = UnlabelledDataset(image_dirs, transform=strong_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
