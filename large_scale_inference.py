import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

# Scientific and numerical libraries
import numpy as np
import pandas as pd

# PyTorch and related
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast

# Image processing and augmentation
import cv2
import albumentations as albu

# Geospatial
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# Graphs
import networkx as nx

# Progress bar
from tqdm import tqdm

# Custom modules
from tools.cfg import py2cfg
from train_supervision import *


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
        "-ps", "--patch_size", type=int, default=1024, help="Patch size for inference."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=2, help="Batch size for inference."
    )
    parser.add_argument(
        "-kr",
        "--keep_ratio",
        type=float,
        default=0.7,
        help="Ratio of patch to keep in sliding window inference.",
    )
    return parser.parse_args()


def calculate_margin(patch_size, keep_ratio):
    return int((1 - keep_ratio) * patch_size / 2)


class InferenceDataset(Dataset):
    def __init__(self, image_dir, patch_size, keep_ratio, transform=None):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.keep_ratio = keep_ratio
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".tif", ".png", ".jpg"))]
        )
        if not self.image_files:
            raise ValueError(f"No valid images found in directory: {image_dir}")

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)

        # Get original image dimensions
        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            transform = src.transform

        neighbors = find_neighbors(image_path)

        # Calculate margin based on image size
        margin = calculate_margin(self.patch_size, self.keep_ratio)
        output_shape = (3, height + 2 * margin, width + 2 * margin)

        combined_image = combine_neighbors(
            neighbors, image_path, output_shape, nodata_value=0
        )

        image = np.moveaxis(combined_image, 0, -1).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image = torch.tensor(image).permute(2, 0, 1).float()

        print(f"📷 Loading: {image_name}")

        return {
            "image": image,
            "image_name": image_name,
            "original_size": (height, width),
            "transform": transform,
        }

    def __len__(self):
        return len(self.image_files)


def find_neighbors(image_path, radius=500):
    with rasterio.open(image_path) as src:
        bounds = src.bounds

    neighbors = []
    for file in glob.glob(os.path.join(os.path.dirname(image_path), "*.tif")):
        if file != image_path:
            with rasterio.open(file) as src:
                neighbor_bounds = src.bounds
                if (
                    neighbor_bounds.left <= bounds.right + radius
                    and neighbor_bounds.right >= bounds.left - radius
                    and neighbor_bounds.bottom <= bounds.top + radius
                    and neighbor_bounds.top >= bounds.bottom - radius
                ):
                    neighbors.append(file)
    return neighbors


def combine_neighbors(neighbors, center_image, output_shape, nodata_value=0):
    combined = np.full(output_shape, nodata_value, dtype=np.float32)

    # First, handle the center image
    with rasterio.open(center_image) as src:
        center_data = src.read()[:3, :, :]
        center_h = (output_shape[1] - center_data.shape[1]) // 2
        center_w = (output_shape[2] - center_data.shape[2]) // 2
        combined[
            :,
            center_h : center_h + center_data.shape[1],
            center_w : center_w + center_data.shape[2],
        ] = center_data

    # Handle neighbors if they exist
    valid_neighbors = [n for n in neighbors if os.path.exists(n)]
    if valid_neighbors:
        src_files = [rasterio.open(neighbor) for neighbor in valid_neighbors]
        try:
            mosaic, transform = merge(src_files)
            mosaic = mosaic[:3, :, :]  # force to RGB

            # Ensure mosaic doesn't exceed output dimensions
            mosaic_h = min(mosaic.shape[1], output_shape[1])
            mosaic_w = min(mosaic.shape[2], output_shape[2])

            # Calculate offset to center the mosaic
            offset_h = (output_shape[1] - mosaic_h) // 2
            offset_w = (output_shape[2] - mosaic_w) // 2

            # Create mask for the target region
            mask = (
                combined[
                    :, offset_h : offset_h + mosaic_h, offset_w : offset_w + mosaic_w
                ]
                == nodata_value
            )

            # Update only where mask is True
            combined[:, offset_h : offset_h + mosaic_h, offset_w : offset_w + mosaic_w][
                mask
            ] = mosaic[:, :mosaic_h, :mosaic_w][mask]
        finally:
            for src in src_files:
                src.close()

    return combined


def sliding_window_inference_batched(
    model, images, num_classes, patch_size=1024, keep_ratio=0.7
):
    inner_size = int(patch_size * keep_ratio)
    outer_margin = (patch_size - inner_size) // 2
    stride = inner_size

    batch_size, _, H, W = images.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    images = nn.functional.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = images.shape

    prediction = torch.zeros(
        (batch_size, num_classes, padded_H, padded_W), device=images.device
    )

    all_patches = []
    patch_targets = []  # (image_index, h, w)

    for b in range(batch_size):
        for h in range(0, padded_H - patch_size + 1, stride):
            for w in range(0, padded_W - patch_size + 1, stride):
                window = images[b : b + 1, :, h : h + patch_size, w : w + patch_size]
                all_patches.append(window)
                patch_targets.append((b, h, w))

    batch_patches = torch.cat(all_patches, dim=0)  # shape: [N, 3, 1024, 1024]

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            batch_output = model(batch_patches)  # shape: [N, num_classes, 1024, 1024]

    for i, (b, h, w) in enumerate(patch_targets):
        prediction[
            b,
            :,
            h + outer_margin : h + outer_margin + inner_size,
            w + outer_margin : w + outer_margin + inner_size,
        ] = batch_output[
            i,
            :,
            outer_margin : outer_margin + inner_size,
            outer_margin : outer_margin + inner_size,
        ]

    prediction = prediction[:, :, :H, :W]
    return prediction


import time


def fill_small_holes(geom, hole_area_threshold=100):
    if isinstance(geom, Polygon):
        outer = geom.exterior
        new_interiors = [
            r for r in geom.interiors if Polygon(r).area > hole_area_threshold
        ]
        return Polygon(outer, new_interiors)
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon(
            [fill_small_holes(p, hole_area_threshold) for p in geom.geoms]
        )
    return geom


def merge_touching_polygons_connected_components(gdf, area_threshold=100, min_area=1):
    print("⏱️ merging connected components...")
    t_start = time.time()

    gdf = gdf[gdf.is_valid].copy()
    gdf = gdf[gdf["DN"] != 0].copy()
    gdf["area"] = gdf.geometry.area
    gdf = gdf.reset_index(drop=True)

    sindex = gdf.sindex
    G = nx.Graph()

    for i, row in gdf.iterrows():
        G.add_node(i)
        geom_i = row.geometry
        area_i = row.area
        matches = list(sindex.intersection(geom_i.bounds))
        if i in matches:
            matches.remove(i)
        for j in matches:
            geom_j = gdf.geometry[j]
            area_j = gdf.area[j]
            if geom_i.buffer(0.5).intersects(geom_j):
                if not (area_i >= area_threshold and area_j >= area_threshold):
                    G.add_edge(i, j)

    merged_polygons = []
    merged_classes = []

    for component in nx.connected_components(G):
        group = gdf.loc[list(component)]
        if len(group) == 1:
            merged_geom = group.geometry.values[0]
        else:
            merged_geom = unary_union(group.geometry)
        largest = group.sort_values("area", ascending=False).iloc[0]
        merged_polygons.append(merged_geom)
        merged_classes.append(largest["DN"])

    result = gpd.GeoDataFrame(
        {"DN": merged_classes, "geometry": merged_polygons}, crs=gdf.crs
    )
    result["area"] = result.geometry.area
    result = result[result["area"] >= min_area].copy()

    print(f"✅ merging done in {time.time() - t_start:.2f}s")
    return result


def polygonize_raster_to_geopackage(
    tif_path, output_gpkg, area_threshold=100, min_area=1, hole_area_threshold=100
):
    print(f"📦 Processing {Path(tif_path).name}")
    t_total = time.time()

    t1 = time.time()
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        mask = image != 0
        results = (
            {"properties": {"DN": int(v)}, "geometry": s}
            for s, v in shapes(image, mask=mask, transform=src.transform)
        )
        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
    print(f"⏱️ polygonize (shapes): {time.time() - t1:.2f}s")

    t2 = time.time()
    gdf = gdf[gdf.is_valid].copy()
    gdf = gdf[gdf["DN"] != 0].copy()
    gdf["area"] = gdf.geometry.area
    gdf = gdf.reset_index(drop=True)
    print(f"⏱️ filtering & area calc: {time.time() - t2:.2f}s")

    result = merge_touching_polygons_connected_components(gdf, area_threshold, min_area)

    t3 = time.time()
    has_holes = result.geometry.apply(
        lambda g: isinstance(g, Polygon) and len(g.interiors) > 0
    )
    result.loc[has_holes, "geometry"] = result.loc[has_holes, "geometry"].apply(
        lambda g: fill_small_holes(g, hole_area_threshold)
    )
    print(f"⏱️ fill small holes: {time.time() - t3:.2f}s")

    t4 = time.time()
    result["area"] = result.geometry.area
    result = result[result["area"] >= min_area].copy()
    result.to_file(output_gpkg, driver="GPKG")
    print(f"⏱️ save to GPKG: {time.time() - t4:.2f}s")

    print(f"✅ Total time: {time.time() - t_total:.2f}s")


def process_image(image_path):
    output_gpkg = image_path.replace(".tif", ".gpkg")
    polygonize_raster_to_geopackage(
        tif_path=image_path,
        output_gpkg=output_gpkg,
        area_threshold=100,
        min_area=1,
        hole_area_threshold=100,
    )
    return output_gpkg


def run_parallel_polygonization(image_paths, max_workers=4):
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, image_paths))

    return results


def main():
    args = get_args()
    seed_everything(42)

    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    dataset = InferenceDataset(
        image_dir=args.image_path,
        patch_size=args.patch_size,
        keep_ratio=args.keep_ratio,
        transform=albu.Normalize(),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.output_path, exist_ok=True)

    for batch in tqdm(dataloader, desc="Processing Images"):

        start_time = time.time()

        # Move images to GPU
        images = batch["image"].cuda()
        image_names = batch["image_name"]

        # Store original metadata
        input_paths = [os.path.join(args.image_path, name) for name in image_names]

        start_time_prediction = time.time()

        predictions = sliding_window_inference_batched(
            model,
            images,
            num_classes=config.num_classes,
            patch_size=args.patch_size,
            keep_ratio=args.keep_ratio,
        )

        end_time_prediction = time.time()
        elapsed_prediction = end_time_prediction - start_time_prediction
        print(f"⏱️  Time taken for prediction: {elapsed_prediction:.2f} seconds")

        predictions = nn.Softmax(dim=1)(predictions).argmax(dim=1)
        written_tif_paths = []

        for i in range(len(images)):
            prediction = predictions[i]
            prediction_np = prediction.cpu().numpy().astype(np.uint8)

            # Use original input path to preserve exact geospatial reference
            input_path = input_paths[i]
            output_file = os.path.join(args.output_path, image_names[i])
            written_tif_paths.append(output_file)

            # Open original image to get exact metadata
            with rasterio.open(input_path) as src:
                # Get original image dimensions and transform
                original_transform = src.transform
                original_crs = src.crs

                # Determine crop parameters if needed
                height, width = src.height, src.width

                # Ensure prediction matches original image exactly
                if prediction_np.shape != (height, width):
                    # Center crop or pad to match original image
                    center_h = (prediction_np.shape[0] - height) // 2
                    center_w = (prediction_np.shape[1] - width) // 2
                    prediction_np = prediction_np[
                        center_h : center_h + height, center_w : center_w + width
                    ]

                # Write output with original geospatial metadata
                profile = src.profile
                profile.update(
                    dtype=rasterio.uint8, count=1, compress="lzw", photometric=None
                )

                with rasterio.open(output_file, "w", **profile) as dst:
                    dst.write(prediction_np.astype(rasterio.uint8), 1)

        run_parallel_polygonization(written_tif_paths)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️  Time taken for {image_names[0]}: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
