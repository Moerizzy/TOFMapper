import argparse
from pathlib import Path
import glob
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os
import rasterio


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(description="TOFSeg Inference Script")
    parser.add_argument(
        "-i",
        "--image_path",
        required=True,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "-o", "--output_path", required=True, help="Path to save the output masks."
    )
    parser.add_argument(
        "-c", "--config_path", required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "-st", "--stride", type=int, default=256, help="Stride size for sliding window."
    )
    parser.add_argument(
        "-ps", "--patch_size", type=int, default=1024, help="Patch size for inference."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=2, help="Batch size for inference."
    )
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".tif", ".png", ".jpg"))]
        )
        if not self.image_files:
            raise ValueError(f"No valid images found in directory: {image_dir}")

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        image = torch.tensor(image).permute(2, 0, 1).float()
        return {"image": image, "image_name": image_name}

    def __len__(self):
        return len(self.image_files)


def sliding_window_inference(model, image, num_classes, patch_size=1024, stride=256):
    _, _, H, W = image.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = image.shape

    prediction = torch.zeros((1, num_classes, padded_H, padded_W), device=image.device)
    count = torch.zeros((1, 1, padded_H, padded_W), device=image.device)

    for h in range(0, padded_H - patch_size + 1, stride):
        for w in range(0, padded_W - patch_size + 1, stride):
            window = image[:, :, h : h + patch_size, w : w + patch_size]
            with torch.no_grad():
                output = model(window)
            prediction[:, :, h : h + patch_size, w : w + patch_size] += output
            count[:, :, h : h + patch_size, w : w + patch_size] += 1

    prediction /= count
    return prediction[:, :, :H, :W]


def main():
    args = get_args()
    seed_everything(42)

    # Load configuration
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model.eval()

    # Prepare dataset and dataloader
    dataset = InferenceDataset(image_dir=args.image_path, transform=albu.Normalize())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Inference loop
    for batch in tqdm(dataloader, desc="Processing Images"):
        images = batch["image"].cuda()
        image_names = batch["image_name"]

        predictions = sliding_window_inference(
            model,
            images,
            num_classes=config.num_classes,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        predictions = nn.Softmax(dim=1)(predictions).argmax(dim=1)

        for i, prediction in enumerate(predictions):
            prediction_np = prediction.cpu().numpy().astype(np.uint8)
            output_file = os.path.join(args.output_path, image_names[i])

            # Save prediction as GeoTIFF if input is GeoTIFF
            input_path = os.path.join(args.image_path, image_names[i])
            with rasterio.open(input_path) as src:
                meta = src.meta
                meta.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(output_file, "w", **meta) as dst:
                    dst.write(prediction_np, 1)

            # Save the geotif as shapefile
            os.system(
                f"gdal_polygonize.py -q {output_image} -f 'ESRI Shapefile' {output_image.replace('.tif', '.shp')}"
            )


if __name__ == "__main__":
    main()
