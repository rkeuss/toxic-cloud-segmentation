import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from ..base.base_dataset import BaseDataSet
from ..utils import palette
from .transforms import build_transform

class IJmondSegDataset(BaseDataSet):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1, end_percent=0.8,
                 color_jitter=1, random_grayscale=0.2, gaussian_blur=11, to_bgr=False, rotate=False,
                 high=512, scale_list=None, **kwargs):
        if scale_list is None:
            scale_list = [1.0]
        self.num_classes = 3
        self.mode = mode
        self.palette = palette  # Reuse CityScapes palette
        self.n_sup = n_sup
        self.split_seed = split_seed
        self.end_percent = end_percent
        self.color_jitter = color_jitter
        self.random_grayscale = random_grayscale
        self.gaussian_blur = gaussian_blur
        self.to_bgr = to_bgr
        self.split = kwargs['split']
        self.high = high
        self.crop_size = kwargs['crop_size']
        self.scale_list = scale_list
        self._set_transform()
        super(IJmondSegDataset, self).__init__(**kwargs)

    def _set_transform(self):
        if self.split == 'train':
            self.transform = build_transform(
                rotation=10,
                color_jitter=self.color_jitter,
                random_grayscale=self.random_grayscale,
                gaussian_blur=self.gaussian_blur,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                flip=True,
                high=self.high,
                scales=self.scale_list
            )
        else:
            self.transform = build_transform(
                rotation=0,
                color_jitter=0,
                random_grayscale=0,
                gaussian_blur=0,
                crop=self.crop_size,
                to_bgr=self.to_bgr,
                flip=False,
                high=self.high
            )

    def _set_files(self):
        self.coco = COCO(os.path.join(self.root, '_annotations.coco.json'))
        self.image_ids = self.coco.getImgIds()
        self.files = []

        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root, "frames", self.split, img_info['file_name'])
            self.files.append((img_path, img_id))

        print(f"Total samples in {self.split} split: {len(self.files)}")

    def _load_data(self, index):
        image_path, img_id = self.files[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)

        anns_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(anns_ids)

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            if 'segmentation' in ann:
                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                m = coco_mask.decode(rle)
                if m.ndim == 3:
                    m = np.any(m, axis=2)
                # Map category_id to label value (adjusted for your categories)
                if ann['category_id'] == 0:  # smoke
                    mask[m > 0] = 0
                elif ann['category_id'] == 1:  # high-opacity-smoke
                    mask[m > 0] = 1
                elif ann['category_id'] == 2:  # low-opacity-smoke
                    mask[m > 0] = 2

        return image, mask

    def _augmentation(self, image, label):
        image, label = self.transform(image, label)
        return image, label

