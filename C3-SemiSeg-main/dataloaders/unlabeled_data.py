from base import BaseDataSet
from glob import glob
import os
import numpy as np
from PIL import Image
from dataloaders.transforms import build_weak_strong_transform

class IJmondUnlabeledDataset(BaseDataSet):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1, end_percent=0.8,
                 color_jitter=1, random_grayscale=0.2, gaussian_blur=11, to_bgr=False, rotate=False,
                 high=512, scale_list=[1.0], **kwargs):
        self.num_classes = 2
        self.mode = mode
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
        super(IJmondUnlabeledDataset, self).__init__(**kwargs)

    def _set_transform(self):
        self.transform = build_weak_strong_transform(
            rotation=10,
            color_jitter=self.color_jitter,
            random_grayscale=self.random_grayscale,
            gaussian_blur=self.gaussian_blur,
            crop=self.crop_size,
            to_bgr=self.to_bgr,
            high=self.high,
            scales=self.scale_list
        )

    def _set_files(self):
        image_path = os.path.join(self.root, "frames", self.split)
        self.files = sorted(glob(os.path.join(image_path, "*.jpg")))
        print(f"Total unlabeled samples in {self.split} split: {len(self.files)}")

    def _load_data(self, index):
        image_path = self.files[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        return image

    def _augmentation(self, image):
        weak, strong, _ = self.transform(image, None)
        return weak, strong

class RiseDataset(BaseDataSet):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1, end_percent=0.8,
                 color_jitter=1, random_grayscale=0.2, gaussian_blur=11, to_bgr=False, rotate=False,
                 high=512, scale_list=[1.0], **kwargs):
        self.num_classes = 2
        self.mode = mode
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
        super(RiseDataset, self).__init__(**kwargs)

    def _set_transform(self):
        self.transform = build_weak_strong_transform(
            rotation=10,
            color_jitter=self.color_jitter,
            random_grayscale=self.random_grayscale,
            gaussian_blur=self.gaussian_blur,
            crop=self.crop_size,
            to_bgr=self.to_bgr,
            high=self.high,
            scales=self.scale_list
        )

    def _set_files(self):
        image_path = os.path.join(self.root, "frames", self.split) # root = 'data/RISE/frames'
        self.files = sorted(glob(os.path.join(image_path, "*.png")))
        print(f"Total unlabeled samples in {self.split} split: {len(self.files)}")

    def _load_data(self, index):
        image_path = self.files[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        return image

    def _augmentation(self, image):
        weak, strong, _ = self.transform(image, None)
        return weak, strong