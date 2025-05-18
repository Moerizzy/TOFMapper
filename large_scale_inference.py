import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
import re

# Scientific and numerical libraries
import numpy as np
import pandas as pd

# PyTorch and related
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import torch.nn.functional as F


# Image processing and augmentation
import cv2
import albumentations as albu

# Geospatial
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from orthophotos_downloader.data_scraping.image_download import (
    ImageDownloader,
    ExtendedWebMapService,
)

# Progress bar
from tqdm import tqdm
import threading

# Custom modules
from tools.cfg import py2cfg
from train_supervision import *


def seed_everything(seed):
    """
    Sets random seeds across commonly used libraries and environment settings
    to ensure reproducibility of results.

    This function is essential for consistent behavior during development,
    evaluation, and debugging in deep learning pipelines.

    Args:
        seed (int): The seed value to use for all random generators.
    """
    # Python + NumPy
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior (important for reproducibility)
    torch.backends.cudnn.deterministic = True

    # Note: benchmark=True can improve performance for fixed input sizes,
    # but may introduce minor nondeterminism on some setups
    torch.backends.cudnn.benchmark = True


def get_args():
    """
    Parses command-line arguments for the TOFSeg inference script.

    This function defines expected CLI inputs such as input/output directories,
    model configuration, patch size, batch size, and keep ratio. It is used to
    flexibly configure inference behavior without hardcoding parameters.

    Returns:
        argparse.Namespace: Parsed arguments object with all input values.
    """
    parser = argparse.ArgumentParser(description="TOFSeg Inference Script")

    # Path to input images (directory containing .tif/.png/.jpg files)
    parser.add_argument(
        "-i",
        "--image_path",
        required=True,
        help="Path to the folder containing input images.",
    )

    # Path where output masks will be saved (GeoTIFF or vector files)
    parser.add_argument(
        "-o", "--output_path", required=True, help="Path to save the output masks."
    )

    # Path to UTM grid (GeoPackage) for downloading images
    parser.add_argument(
        "-u",
        "--utm_grid",
        required=True,
        help="Path to the UTM grid (GeoPackage) for downloading images.",
    )

    # Path to model/config YAML or Python config file
    parser.add_argument(
        "-c", "--config_path", required=True, help="Path to the configuration file."
    )

    # Size of patches to extract during sliding-window inference
    parser.add_argument(
        "-ps",
        "--patch_size",
        type=int,
        default=1024,
        help="Patch size (in pixels) for sliding window inference. Default: 1024",
    )

    # Number of input images to process in one forward pass (outer batch size)
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=2,
        help="Number of full images to process per batch. Default: 2",
    )

    return parser.parse_args()


import os
import torch
import numpy as np
import cv2
import rasterio
from torch.utils.data import Dataset

download_done = threading.Event()
processed = set()
inference_counter = 0
inference_lock = threading.Lock()


class DownloadState:
    def __init__(self, total):
        self.counter = 0
        self.lock = threading.Lock()
        self.total = total


class InferenceDataset(Dataset):
    """
    Custom PyTorch dataset for inference on geospatial image tiles.

    This dataset loads one image tile at a time, applies optional transformations,
    and prepares the image as a PyTorch tensor for model inference.

    Args:
        image_dir (str): Path to the directory containing input image tiles (GeoTIFF, PNG, JPG).
        patch_size (int): Patch size used for inference (used to calculate margin).
        transform (albumentations.BasicTransform, optional): Transformations to apply to the input image (e.g., normalization).
        subset (list[str], optional): Optional list of specific filenames to include.
    """

    def __init__(self, image_dir, patch_size, transform=None, subset=None):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.margin = patch_size // 2
        self.transform = transform

        # List image files with supported extensions, exclude files with "sub"
        all_images = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".tif", ".tiff", ".png", ".jpg"))
            and "sub" not in f.lower()
        ]

        if subset:
            # Only keep images that are both in subset and in directory
            self.image_files = sorted([f for f in all_images if f in subset])
        else:
            self.image_files = sorted(all_images)

        if not self.image_files:
            raise ValueError(f"No valid images found in directory: {image_dir}")

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)

        with rasterio.open(image_path) as src:
            image = src.read()  # (C, H, W)
            transform = src.transform
            height = src.height
            width = src.width

        # Convert to (H, W, C)
        image = np.moveaxis(image, 0, -1).astype(np.uint8)

        # Convert to RGB if needed (e.g., BGR → RGB)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply albumentations transform (e.g., normalization)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert to PyTorch tensor in (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1).float()

        return {
            "image": image,
            "image_name": image_name,
            "original_size": (height, width),
            "transform": transform,
        }

    def __len__(self):
        return len(self.image_files)


def fill_small_holes(geom, hole_area_threshold=100):
    """
    Removes small interior holes from a polygon or multipolygon geometry.

    For each polygon, holes (interior rings) smaller than the given area threshold
    are removed. This helps clean up noisy segmentation or classification artifacts
    where small voids are not meaningful.

    Args:
        geom (shapely.geometry.Polygon or MultiPolygon): The input geometry.
        hole_area_threshold (float): Minimum area (in projection units, e.g. m²) a hole must have to be kept.

    Returns:
        shapely.geometry.Polygon or MultiPolygon: Geometry with small holes removed.
    """
    if isinstance(geom, Polygon):
        # Keep only the outer boundary and holes above threshold
        outer = geom.exterior
        new_interiors = [
            ring for ring in geom.interiors if Polygon(ring).area > hole_area_threshold
        ]
        return Polygon(outer, new_interiors)

    elif isinstance(geom, MultiPolygon):
        # Recursively apply to each part
        return MultiPolygon(
            [fill_small_holes(part, hole_area_threshold) for part in geom.geoms]
        )

    # Return original geometry if not a polygonal type
    return geom


def merge_touching_polygons_spatial(gdf, area_threshold=100, min_area=1):
    """
    Merges spatially touching polygons if at least one of them is smaller than the area threshold.

    This function uses a spatial join to identify touching polygons and merges them into connected
    components (groups of mutually touching small polygons). Each group is unified into a single geometry,
    and the dominant class label (DN) is taken from the largest polygon in the group.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with at least a 'DN' column and valid polygon geometries.
        area_threshold (float): Maximum area threshold below which polygons are considered for merging.
        min_area (float): Minimum area threshold for the output geometries; smaller ones will be discarded.

    Returns:
        GeoDataFrame: Cleaned and merged geometries with updated class labels and filtered small areas.
    """
    t_start = time.time()

    # Step 1: Remove invalid geometries and background class (DN == 0)
    gdf = gdf[gdf.is_valid].copy()
    gdf = gdf[gdf["DN"] != 0].copy()
    gdf["area"] = gdf.geometry.area
    gdf = gdf.reset_index(drop=True)

    # Step 2: Find touching polygons using spatial join
    joined = gpd.sjoin(gdf, gdf, predicate="touches", how="left")
    joined = joined.dropna(subset=["index_right"])
    joined = joined[joined.index != joined["index_right"]]  # Remove self-matches

    # Step 3: Build pairwise groups where at least one polygon is below area threshold
    groups = {}
    for idx, row in joined.iterrows():
        i, j = idx, int(row["index_right"])
        area_i = gdf.loc[i, "area"]
        area_j = gdf.loc[j, "area"]
        if area_i < area_threshold or area_j < area_threshold:
            groups.setdefault(i, set()).add(i)
            groups[i].add(j)
            groups.setdefault(j, set()).add(j)
            groups[j].add(i)

    # Step 4: Extract connected components from groups (depth-first search)
    visited = set()
    components = []
    for node in groups:
        if node not in visited:
            stack = [node]
            comp = set()
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    comp.add(n)
                    stack.extend(groups.get(n, []))
            components.append(comp)

    # Step 5: Merge geometries in each component and assign DN from largest polygon
    merged_geoms = []
    merged_classes = []
    for comp in components:
        group = gdf.loc[list(comp)]
        if len(group) == 1:
            merged_geoms.append(group.geometry.values[0])
            merged_classes.append(group["DN"].values[0])
        else:
            merged_geoms.append(unary_union(group.geometry))
            largest = group.sort_values("area", ascending=False).iloc[0]
            merged_classes.append(largest["DN"])

    # Step 6: Combine merged with untouched geometries
    merged = gpd.GeoDataFrame(
        {"DN": merged_classes, "geometry": merged_geoms}, crs=gdf.crs
    )
    untouched = gdf.drop(index=set().union(*components), errors="ignore")
    result = pd.concat([merged, untouched], ignore_index=True)

    # Step 7: Remove very small leftover geometries
    result["area"] = result.geometry.area
    result = result[result["area"] >= min_area].copy()

    return result


def polygonize_raster_to_geopackage(
    tif_path, output_gpkg, area_threshold=100, min_area=1, hole_area_threshold=100
):
    """
    Converts a classified raster into polygons, merges small touching regions,
    removes tiny areas, and fills small interior holes. The final result is saved as a GeoPackage.

    Args:
        tif_path (str): Path to the input classified raster file (GeoTIFF).
        output_gpkg (str): Output path for the resulting GeoPackage file.
        area_threshold (float): Maximum area (in projection units) below which polygons are candidates for merging.
        min_area (float): Minimum area to retain in the final result; smaller features are discarded.
        hole_area_threshold (float): Interior holes smaller than this threshold will be removed.

    Returns:
        None. Saves the processed polygons directly to a `.gpkg` file.
    """

    # --- Step 1: Polygonize the raster ---
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        mask = image != 0  # Only polygonize non-zero regions
        results = (
            {"properties": {"DN": int(v)}, "geometry": s}
            for s, v in shapes(image, mask=mask, transform=src.transform)
        )
        geoms = list(results)
        gdf = gpd.GeoDataFrame.from_features(geoms)
        if "geometry" not in gdf.columns or gdf.empty:
            print(f"[Polygonize] No geometry found in {tif_path}, skipping.")
            return
        gdf.set_crs(src.crs, inplace=True)

    # --- Step 2: Pre-filtering and area computation ---
    gdf = gdf[gdf.is_valid].copy()
    gdf = gdf[gdf["DN"] != 0].copy()
    gdf["area"] = gdf.geometry.area
    gdf = gdf.reset_index(drop=True)

    # --- Step 3: Merge small touching polygons based on spatial connectivity ---
    result = merge_touching_polygons_spatial(gdf, area_threshold, min_area)

    # --- Step 4: Fill small interior holes in valid Polygons only ---
    has_holes = result.geometry.apply(
        lambda g: isinstance(g, Polygon) and len(g.interiors) > 0
    )
    result.loc[has_holes, "geometry"] = result.loc[has_holes, "geometry"].apply(
        lambda g: fill_small_holes(g, hole_area_threshold)
    )

    # --- Step 5: Final area filtering and export ---
    result["area"] = result.geometry.area
    result = result[result["area"] >= min_area].copy()
    result.to_file(output_gpkg, driver="GPKG")


def process_image(image_path):
    """
    Processes a single raster file by converting it to polygons and cleaning the result.

    This function:
    - Converts the input `.tif` classification raster to vector polygons
    - Merges small touching polygons
    - Removes tiny holes and features
    - Saves the cleaned output as a `.gpkg` GeoPackage file

    Args:
        image_path (str): Path to the input `.tif` raster file.

    Returns:
        str: Path to the output `.gpkg` file.
    """
    output_gpkg = image_path.replace(".tiff", ".gpkg")

    polygonize_raster_to_geopackage(
        tif_path=image_path,
        output_gpkg=output_gpkg,
        area_threshold=100,  # Merge if one polygon is < 100 m²
        min_area=1,  # Remove any polygons < 1 m²
        hole_area_threshold=100,  # Fill holes smaller than 100 m²
    )

    return output_gpkg


def run_parallel_polygonization(image_paths, max_workers=4):
    """
    Executes polygonization and postprocessing of raster files in parallel.

    This function distributes the workload across multiple processes using
    Python's ProcessPoolExecutor. It is especially useful when working with
    large collections of raster files (e.g., segmentation masks).

    Args:
        image_paths (List[str]): A list of `.tif` raster file paths to process.
        max_workers (int): Maximum number of parallel processes.
                           If None, it defaults to (number of CPUs - 1).

    Returns:
        List[str]: List of output GeoPackage paths generated from each input raster.
    """
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, image_paths))

    return results


def sliding_window_inference_entropy_hann(model, images, patch_size=1024, stride=512):
    """
    Sliding window inference with Hann-weighting over full patch area.
    Returns logits and entropy map without visible seams between tiles.

    Args:
        model (nn.Module): segmentation model.
        images (Tensor): (B, C, H, W)
        patch_size (int): size of square patches
        stride (int): sliding window step (typically patch_size // 2)

    Returns:
        pred_map (Tensor): (B, num_classes, H, W)
        entropy_map (Tensor): (B, H, W)
    """

    def create_hann_window(size, device):
        hann_1d = torch.hann_window(size, periodic=False, device=device)
        hann_2d = torch.outer(hann_1d, hann_1d)
        softened = hann_2d**1.5  # flacherer Verlauf zum Rand
        return softened.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)

    B, C, H, W = images.shape
    device = images.device

    # Pad image to ensure full coverage
    pad_h = (patch_size - H % stride) % stride
    pad_w = (patch_size - W % stride) % stride
    images = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, padded_H, padded_W = images.shape

    hann_window = create_hann_window(patch_size, device=device)

    # To be filled during aggregation
    pred_map = None  # initialized after first inference
    entropy_map = torch.zeros((B, padded_H, padded_W), device=device)
    count_map_logits = None
    count_map_entropy = torch.zeros_like(entropy_map)

    # Extract patches
    all_patches = []
    patch_targets = []

    for b in range(B):
        for y in range(0, padded_H - patch_size + 1, stride):
            for x in range(0, padded_W - patch_size + 1, stride):
                patch = images[b : b + 1, :, y : y + patch_size, x : x + patch_size]
                all_patches.append(patch)
                patch_targets.append((b, y, x))

    patches_tensor = torch.cat(all_patches, dim=0)

    # Run inference
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            logits = model(patches_tensor)  # (N, C, H, W)

    softmax_probs = F.softmax(logits, dim=1)
    entropy = -(softmax_probs * torch.log(softmax_probs + 1e-8)).sum(dim=1)  # (N, H, W)

    # Init aggregation maps
    num_classes = logits.shape[1]
    pred_map = torch.zeros((B, num_classes, padded_H, padded_W), device=device)
    count_map_logits = torch.zeros_like(pred_map)

    # Merge patches
    for i, (b, y, x) in enumerate(patch_targets):
        logits_i = logits[i : i + 1] * hann_window  # (1, C, H, W)
        entropy_i = entropy[i : i + 1].unsqueeze(1) * hann_window  # (1, 1, H, W)

        pred_map[b, :, y : y + patch_size, x : x + patch_size] += logits_i.squeeze(0)
        entropy_map[b, y : y + patch_size, x : x + patch_size] += entropy_i.squeeze()

        count_map_logits[
            b, :, y : y + patch_size, x : x + patch_size
        ] += hann_window.squeeze(0)
        count_map_entropy[
            b, y : y + patch_size, x : x + patch_size
        ] += hann_window.squeeze()

    # Normalize
    pred_map = pred_map / (count_map_logits + 1e-6)
    entropy_map = entropy_map / (count_map_entropy + 1e-6)

    # Crop back to original
    pred_map = pred_map[:, :, :H, :W]
    entropy_map = entropy_map[:, :H, :W]

    return pred_map, entropy_map


def extract_coords_from_filename(filename):
    """
    Extract x_sw and y_sw from filename like dop20_rgb_32709_5605_1.tiff
    """
    match = re.search(r"image_32_(\d{3})000_(\d{4})000.tiff", filename)
    if match:
        x_sw = int(match.group(1)) * 1000
        y_sw = int(match.group(2)) * 1000
        return x_sw, y_sw
    else:
        return None, None


def aggregate_uncertainty_by_filename(
    raster_path, tile_grid, threshold=0.5, prefix="entropy"
):
    """
    Aggregates uncertainty from a raster and adds stats to the matching tile based on filename coordinates.

    Args:
        raster_path (Path or str): Path to the raster file
        tile_grid (GeoDataFrame): Grid with x_sw, y_sw columns
        threshold (float): Threshold for high uncertainty
        prefix (str): Prefix for new columns

    Returns:
        GeoDataFrame: Updated tile_grid with new columns
    """
    from pathlib import Path

    raster_path = Path(raster_path)
    x_sw, y_sw = extract_coords_from_filename(raster_path.name)

    if x_sw is None:
        raise ValueError(f"Filename {raster_path.name} could not be parsed.")

    # Read raster
    with rasterio.open(raster_path) as src:
        entropy_map = src.read(1, masked=True)
        transform = src.transform

    # Mask whole tile (no geometry match needed)
    stats = {}
    mask = np.ones_like(entropy_map, dtype=bool)
    values = entropy_map.compressed()

    if values.size == 0:
        stats[f"{prefix}_median"] = np.nan
        stats[f"{prefix}_mean"] = np.nan
        stats[f"{prefix}_std"] = np.nan
        stats[f"{prefix}_min"] = np.nan
        stats[f"{prefix}_max"] = np.nan
        stats[f"{prefix}_high_frac"] = np.nan
    else:
        stats[f"{prefix}_median"] = float(np.median(values))
        stats[f"{prefix}_mean"] = float(np.mean(values))
        stats[f"{prefix}_std"] = float(np.std(values))
        stats[f"{prefix}_min"] = float(np.min(values))
        stats[f"{prefix}_max"] = float(np.max(values))
        stats[f"{prefix}_high_frac"] = float((values > threshold).sum() / len(values))

    # Update tile
    tile_grid = tile_grid.copy()
    idx = tile_grid[(tile_grid["x_sw"] == x_sw) & (tile_grid["y_sw"] == y_sw)].index
    if not idx.empty:
        for key, val in stats.items():
            if key not in tile_grid.columns:
                tile_grid[key] = np.nan
            tile_grid.loc[idx, key] = val
    else:
        print(f"Warning: No match in grid for {raster_path.name}")

    return tile_grid


def inference_watcher():
    global inference_counter

    args = get_args()
    seed_everything(42)
    tile_grid = gpd.read_file(args.utm_grid)
    args.tile_count = len(tile_grid)

    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    os.makedirs(args.output_path, exist_ok=True)
    entropy_subfolder = os.path.join(args.output_path, "entropy_maps")
    os.makedirs(entropy_subfolder, exist_ok=True)
    done_once = {
        f.replace("_entropy_basic.tiff", "")
        for f in os.listdir(entropy_subfolder)
        if f.endswith(".tiff")
    }

    # Globalen Fortschrittszähler setzen

    inference_counter = len(done_once)

    while True:
        all_files = sorted(
            f
            for f in os.listdir(args.image_path)
            if f.endswith(".tiff") and "sub" not in f
        )
        new_files = [
            f
            for f in all_files
            if os.path.splitext(f)[0] not in done_once and f not in processed
        ]

        if new_files:
            print(f"[Inference] Found {len(new_files)} new images.")

            dataset = InferenceDataset(
                image_dir=args.image_path,
                patch_size=args.patch_size,
                transform=albu.Normalize(),
                subset=new_files,
            )
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            for batch in tqdm(dataloader, desc="Processing Images"):
                images = batch["image"].cuda()
                image_names = batch["image_name"]
                input_paths = [
                    os.path.join(args.image_path, name) for name in image_names
                ]

                logits, entropy_map_basic = sliding_window_inference_entropy_hann(
                    model,
                    images,
                    patch_size=args.patch_size,
                    stride=args.patch_size // 2,
                )

                predictions = nn.Softmax(dim=1)(logits).argmax(dim=1)
                written_tif_paths = []

                for i in range(len(images)):
                    prediction = predictions[i].cpu().numpy().astype(np.uint8)
                    entropy_basic = (
                        entropy_map_basic[i].cpu().numpy().astype(np.float32)
                    )

                    input_path = input_paths[i]
                    base_name = os.path.splitext(image_names[i])[0]
                    output_pred_file = os.path.join(
                        args.output_path, f"{base_name}.tiff"
                    )
                    output_entropy_basic_file = os.path.join(
                        entropy_subfolder, f"{base_name}_entropy_basic.tiff"
                    )

                    with rasterio.open(input_path) as src:
                        height, width = 5000, 5000

                        def center_crop(arr):
                            if arr.shape != (height, width):
                                center_h = (arr.shape[0] - height) // 2
                                center_w = (arr.shape[1] - width) // 2
                                return arr[
                                    center_h : center_h + height,
                                    center_w : center_w + width,
                                ]
                            return arr

                        prediction = center_crop(prediction)
                        entropy_basic = (center_crop(entropy_basic) * 1000).astype(
                            np.uint16
                        )

                        # Extract UTM coordinates from filename
                        match = re.search(r"(\d{6})_(\d{7})", base_name)
                        if not match:
                            raise ValueError(
                                f"Filename {base_name} does not contain valid UTM coords."
                            )

                        ulx, uly = map(int, match.groups())
                        lrx, lry = ulx + 1000, uly + 1000

                        # Define resolution and dimensions
                        resolution = 0.2
                        width = height = int(1000 / resolution)  # → 5000 px

                        # Create transform from bounds
                        transform = from_bounds(ulx, uly, lrx, lry, width, height)

                        # Build raster profile
                        profile = {
                            "driver": "GTiff",
                            "height": height,
                            "width": width,
                            "count": 1,
                            "dtype": rasterio.uint8,
                            "crs": "EPSG:25832",
                            "transform": transform,
                            "compress": "lzw",
                            "photometric": "MINISBLACK",
                        }
                        with rasterio.open(output_pred_file, "w", **profile) as dst:
                            dst.write(prediction, 1)

                        profile.update(dtype=rasterio.uint16, photometric="MINISBLACK")
                        with rasterio.open(
                            output_entropy_basic_file, "w", **profile
                        ) as dst:
                            dst.write(entropy_basic, 1)

                    written_tif_paths.append(output_pred_file)
                    processed.add(image_names[i])
                    with inference_lock:
                        inference_counter += 1
                        print(
                            f"[Inference] {inference_counter} / {args.tile_count} tiles processed"
                        )

                run_parallel_polygonization(written_tif_paths)

                # --- Delete all processed TIFF files ---
                for tif_path in written_tif_paths:
                    try:
                        os.remove(tif_path)
                        print(f"[Cleanup] Deleted {tif_path}")
                        # TIFF im image_dir entfernen
                        # TIFF im image_dir entfernen
                        base_name = Path(tif_path).stem  # z. B. image_32_734000_5563000
                        original_input = Path(args.image_path) / f"{base_name}.tiff"
                        if original_input.exists():
                            os.remove(original_input)
                            print(f"[Cleanup] Deleted input: {original_input}")
                    except Exception as e:
                        print(f"[Cleanup] Failed to delete {tif_path}: {e}")

        elif download_done.is_set():
            print("[Inference] No more new files. Exiting.")
            break

        time.sleep(2)


def download_partition(
    tile_subset, args, wms, margin_m, downloader, image_path, state, skip_tiles
):
    for _, row in tile_subset.iterrows():
        try:
            utm_bounding_box = row.geometry
            ulx = int(round(utm_bounding_box.bounds[0] / 1000) * 1000)
            uly = int(round(utm_bounding_box.bounds[1] / 1000) * 1000)
            prefix = f"{ulx}_{uly}"
            tile_id = f"image_32_{prefix}"

            # ✅ Überspringe, wenn Ergebnis schon existiert
            if tile_id in skip_tiles:
                with state.lock:
                    state.counter += 1
                    print(
                        f"[Download] Skipped tile {tile_id} ({state.counter} / {state.total})"
                    )
                continue

            path = image_path / f"{tile_id}.tiff"

            minx, miny, maxx, maxy = utm_bounding_box.buffer(margin_m).bounds
            width_px = int((maxx - minx) / wms.resolution)
            height_px = int((maxy - miny) / wms.resolution)
            print(f"[Download] Downloading tile {tile_id} ({width_px} x {height_px})")
            polygon = box(minx, miny, maxx, maxy)

            downloader.download_single_image(
                img_path=path,
                bounding_box=polygon,
                wms=wms,
                width_px=width_px,
                height_px=height_px,
                driver="GTiff",
            )

            with state.lock:
                state.counter += 1
                print(f"[Download] {state.counter} / {state.total} tiles completed")

        except Exception as e:
            print(f"[Download] Error on tile {row.name}: {e}")


def download_wrapper():
    args = get_args()
    seed_everything(42)

    image_path = Path(args.image_path)
    os.makedirs(image_path, exist_ok=True)

    tile_grid = gpd.read_file(args.utm_grid)
    args.tile_count = len(tile_grid)

    tile_subsets = np.array_split(tile_grid, 3)

    wms = ExtendedWebMapService(
        url="https://geodienste.sachsen.de/wms_geosn_dop_2018_2020/guest",
        version="1.3.0",
        resolution=0.2,
        layer_name="dop_2018_2020_rgb",
        crs="EPSG:25832",
        format="image/tiff",
    )

    margin_m = (args.patch_size * wms.resolution) // 2
    downloader = ImageDownloader(wms, grid_spacing=1000)

    # ✅ Fertig vorhandene Ergebnisse anhand von entropy_maps prüfen
    entropy_dir = os.path.join(args.output_path, "entropy_maps")
    already_done = {
        f.replace("_entropy_basic.tiff", "")
        for f in os.listdir(entropy_dir)
        if f.endswith(".tiff")
    }
    already_downloaded = {
        f.replace(".tiff", "")
        for f in os.listdir(args.image_path)
        if f.endswith(".tiff")
    }
    skip_tiles = already_done.union(already_downloaded)

    # ✅ Fortschritt initialisieren
    state = DownloadState(total=args.tile_count)
    state.counter = len(already_done)

    threads = []
    for i, subset in enumerate(tile_subsets):
        t = threading.Thread(
            target=download_partition,
            args=(
                subset,
                args,
                wms,
                margin_m,
                downloader,
                image_path,
                state,
                skip_tiles,
            ),
            name=f"Downloader-{i+1}",
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("[Download] All tiles downloaded.")
    download_done.set()


if __name__ == "__main__":
    t1 = threading.Thread(target=download_wrapper)
    t2 = threading.Thread(target=inference_watcher)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("✅ Alle Bilder heruntergeladen und inferiert.")
