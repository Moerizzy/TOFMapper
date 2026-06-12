import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import rasterio
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *

CLASSES = ("Background", "Forest", "Patch", "Linear", "Tree")
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (5000, 5000)


class ColorJitterRGB(albu.ImageOnlyTransform):
    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(p=p)
        self._cj = albu.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=1.0,
        )

    def apply(self, img, **params):
        # img: H x W x C
        if img.ndim != 3:
            return img

        h, w, c = img.shape
        if c == 3:
            # ganz normal wie früher
            return self._cj(image=img)["image"]
        elif c > 3:
            rgb = img[:, :, :3]
            extra = img[:, :, 3:]
            rgb = self._cj(image=rgb)["image"]
            return np.concatenate([rgb, extra], axis=2)
        else:
            # 1-Kanal o.ä.: gar nichts machen
            return img


# ImageNet RGB stats. For any extra (NIR / nDSM / ...) channel we reuse the
# mean of the three RGB ImageNet means/stds ("Option A"). This keeps the
# RGB pretrained backbone consistent with its training distribution and
# treats inflated 4th+ input channels as RGB-like channels.
_IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
_EXTRA_MEAN = sum(_IMAGENET_RGB_MEAN) / 3.0  # ≈ 0.449
_EXTRA_STD = sum(_IMAGENET_RGB_STD) / 3.0  # ≈ 0.226


def imagenet_stats(num_channels):
    """Return (mean, std) tuples of length `num_channels` ("Option A").

    First three channels = ImageNet RGB, additional channels = average of
    the RGB stats. For 3 channels this is plain ImageNet.
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if num_channels <= 3:
        return (_IMAGENET_RGB_MEAN[:num_channels], _IMAGENET_RGB_STD[:num_channels])
    extra = num_channels - 3
    mean = _IMAGENET_RGB_MEAN + (_EXTRA_MEAN,) * extra
    std = _IMAGENET_RGB_STD + (_EXTRA_STD,) * extra
    return mean, std


def build_train_transform(num_channels):
    mean, std = imagenet_stats(num_channels)
    return albu.Compose(
        [
            ColorJitterRGB(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
            ),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(1, 256),
                hole_width_range=(1, 256),
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            albu.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ]
    )


def build_val_transform(num_channels):
    mean, std = imagenet_stats(num_channels)
    return albu.Compose(
        [
            albu.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ]
    )


def make_train_aug(num_channels):
    transform = build_train_transform(num_channels)

    def _aug(img, mask):
        img, mask = np.array(img), np.array(mask)
        out = transform(image=img.copy(), mask=mask.copy())
        return out["image"], out["mask"]

    return _aug


def make_val_aug(num_channels):
    transform = build_val_transform(num_channels)

    def _aug(img, mask):
        img, mask = np.array(img), np.array(mask)
        out = transform(image=img.copy(), mask=mask.copy())
        return out["image"], out["mask"]

    return _aug


# Backwards compatibility: the original config imported `train_aug` / `val_aug`
# directly. Default to 4 channels (RGB+NIR) as before.
train_aug = make_train_aug(4)
val_aug = make_val_aug(4)


def load_multiband_image(img_path, band_indices=(1, 2, 3, 4)):
    """Load specific bands from a multi-band raster.

    Parameters
    ----------
    img_path : str
        Path to a GeoTIFF (or any rasterio-readable raster).
    band_indices : sequence of int
        1-based band indices to keep, in output order. Defaults to RGBN.

    Returns
    -------
    np.ndarray of shape (H, W, len(band_indices)), dtype uint8 in [0, 255].
    """
    band_indices = list(band_indices)
    try:
        with rasterio.open(img_path) as src:
            available = src.count
            requested = [b for b in band_indices if 1 <= b <= available]
            img_data = src.read(indexes=requested)  # (C, H, W)

        if len(requested) < len(band_indices):
            # Pad missing bands by repeating the last available one.
            missing = len(band_indices) - len(requested)
            extra = np.repeat(img_data[-1:, :, :], missing, axis=0)
            img_data = np.concatenate([img_data, extra], axis=0)

        img = np.transpose(img_data, (1, 2, 0))  # (H, W, C)

        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            return (img // 256).astype(np.uint8)
        # Generic fallback: per-array min-max scale.
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)

    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        # Fallback: PIL RGBA, sliced to first len(band_indices) channels.
        img = np.array(Image.open(img_path).convert("RGBA"))
        n = len(band_indices)
        if img.shape[2] >= n:
            return img[:, :, :n]
        reps = n - img.shape[2]
        extra = np.repeat(img[:, :, -1:], reps, axis=2)
        return np.concatenate([img, extra], axis=2)


class TOFDataset(Dataset):
    def __init__(
        self,
        data_root="data/tof/test",
        mode="val",
        img_dir="images_1024",
        mask_dir="masks_1024",
        img_suffix=".tif",
        mask_suffix=".png",
        transform=val_aug,
        mosaic_ratio=0.0,
        img_size=ORIGIN_IMG_SIZE,
        band_indices=(1, 2, 3, 4),
    ):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.band_indices = tuple(band_indices)
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == "val" or self.mode == "test":
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return {"img": img, "gt_semantic_seg": mask, "img_id": self.img_ids[index]}

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        # Strip the full configured suffix (handles e.g. mask_suffix='_mask.tif').
        img_filename_list = [
            f[: -len(self.img_suffix)]
            for f in os.listdir(osp.join(data_root, img_dir))
            if f.endswith(self.img_suffix)
        ]
        mask_filename_list = [
            f[: -len(self.mask_suffix)]
            for f in os.listdir(osp.join(data_root, mask_dir))
            if f.endswith(self.mask_suffix)
        ]
        assert len(img_filename_list) == len(
            mask_filename_list
        ), "Number of images and masks should be the same"
        # Intersect to ensure 1:1 pairing in case of stray files.
        common = sorted(set(img_filename_list) & set(mask_filename_list))
        assert len(common) == len(
            img_filename_list
        ), "Image / mask basenames do not fully overlap after stripping suffixes"
        return common

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)

        img = load_multiband_image(img_name, band_indices=self.band_indices)
        mask = np.array(Image.open(mask_name))
        if mask.ndim == 3:
            # Single-band masks are expected; if a 3-channel mask sneaks in,
            # take the first channel (which by convention holds class indices).
            mask = mask[..., 0]

        return img, mask

    def load_mosaic_img_and_mask(self, index):
        # Mosaic augmentation implementation
        # ... (keep existing mosaic code)
        pass
