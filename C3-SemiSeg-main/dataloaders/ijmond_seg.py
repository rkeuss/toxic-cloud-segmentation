from base import BaseDataSet
from utils import palette
from glob import glob
import os
import numpy as np
from PIL import Image
from dataloaders.transforms import build_transform

class IJmondSegDataset(BaseDataSet):
    def __init__(self, mode='fine', n_sup=-1, split_seed=1, end_percent=0.8,
                 color_jitter=1, random_grayscale=0.2, gaussian_blur=11, to_bgr=False, rotate=False,
                 high=512, scale_list=[1.0], **kwargs):
        self.num_classes = 2
        self.mode = mode
        self.palette = palette.CityScpates_palette  # Reuse CityScapes palette
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
        image_path = os.path.join(self.root, "frames", self.split)
        # label_path = os.path.join(self.root, "labels", self.split)
        self.files = list(zip(
            sorted(glob(os.path.join(image_path, "*.jpg"))),
            # sorted(glob(os.path.join(label_path, "*.png")))
        ))
        print(f"Total samples in {self.split} split: {len(self.files)}")

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        # label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label

    def _augmentation(self, image, label):
        image, label = self.transform(image, label)
        return image, label
