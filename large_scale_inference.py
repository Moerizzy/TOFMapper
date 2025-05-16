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
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# Progress bar
from tqdm import tqdm

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


class InferenceDataset(Dataset):
    """
    Custom PyTorch dataset for inference on geospatial image tiles with contextual neighborhood merging.

    This dataset loads one image tile at a time and optionally merges neighboring tiles
    into the margins to preserve spatial context at patch boundaries. It prepares the image
    for inference by applying a transformation (e.g., normalization), padding margins,
    and converting to a PyTorch tensor.

    Args:
        image_dir (str): Path to the directory containing input image tiles (GeoTIFF, PNG, JPG).
        patch_size (int): Patch size used for inference (used to calculate margin).
        transform (albumentations.BasicTransform, optional): Transformations to apply to the input image (e.g., normalization).
    """

    def __init__(self, image_dir, patch_size, transform=None):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.margin = patch_size // 2
        self.transform = transform

        # Load and sort image file names with supported extensions
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".tif", ".png", ".jpg"))]
        )
        if not self.image_files:
            raise ValueError(f"No valid images found in directory: {image_dir}")

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)

        # Load basic metadata of the center image
        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            transform = src.transform

        # Find neighboring images within spatial proximity
        neighbors = find_neighbors(image_path)

        # Compute output shape with margin based on patch size and keep ratio
        output_shape = (3, height + 2 * self.margin, width + 2 * self.margin)

        # Combine center image with neighbors into a padded RGB array
        combined_image = combine_neighbors(
            neighbors, image_path, output_shape, nodata_value=0
        )

        # Convert CHW (rasterio) to HWC and ensure RGB color space
        image = np.moveaxis(combined_image, 0, -1).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply optional normalization/augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert to tensor in CHW format
        image = torch.tensor(image).permute(2, 0, 1).float()

        return {
            "image": image,
            "image_name": image_name,
            "original_size": (height, width),
            "transform": transform,
        }

    def __len__(self):
        return len(self.image_files)


def find_neighbors(image_path, radius=500):
    """
    Finds neighboring image tiles (GeoTIFFs) that spatially overlap or are within a given buffer radius.

    This function scans the directory of the input image and identifies all other `.tif` files
    whose bounding boxes fall within a specified distance (`radius`) from the input image's bounds.
    It is commonly used for stitching together adjacent tiles in tile-based inference or preprocessing.

    Args:
        image_path (str): Path to the center image file (GeoTIFF).
        radius (int): Buffer distance in map units (e.g., meters) to consider as "neighboring".

    Returns:
        List[str]: List of file paths to neighboring image tiles within the radius.
    """
    # Read bounding box of the center image
    with rasterio.open(image_path) as src:
        bounds = src.bounds

    neighbors = []
    search_dir = os.path.dirname(image_path)

    # Iterate over all GeoTIFF files in the same directory
    for file in glob.glob(os.path.join(search_dir, "*.tif")):
        if file == image_path:
            continue  # Skip the input image itself

        with rasterio.open(file) as src:
            neighbor_bounds = src.bounds

            # Check whether neighbor image is within radius of the center image bounds
            if (
                neighbor_bounds.left <= bounds.right + radius
                and neighbor_bounds.right >= bounds.left - radius
                and neighbor_bounds.bottom <= bounds.top + radius
                and neighbor_bounds.top >= bounds.bottom - radius
            ):
                neighbors.append(file)

    return neighbors


def combine_neighbors(neighbors, center_image, output_shape, nodata_value=0):
    """
    Combines a center image with its spatially adjacent neighbor images into a single larger canvas.

    This function creates a new image of shape `output_shape`, places the center image in the middle,
    and fills in surrounding pixels using available neighboring orthophotos. Only non-overlapping regions
    (where `nodata_value` is present) are filled by neighbor data, preserving priority of the center tile.

    Args:
        neighbors (List[str]): Paths to neighboring image tiles (GeoTIFFs).
        center_image (str): Path to the center image (GeoTIFF) that should be prioritized.
        output_shape (Tuple[int, int, int]): Desired output image shape (C, H, W).
        nodata_value (int or float): Fill value used for uninitialized pixels (default: 0).

    Returns:
        np.ndarray: Combined image array of shape (C, H, W) with center and neighbor data merged.
    """
    # Initialize the output array with nodata_value
    combined = np.full(output_shape, nodata_value, dtype=np.float32)

    # Insert the center image in the middle of the canvas
    with rasterio.open(center_image) as src:
        center_data = src.read()[:3, :, :]  # Only take the first 3 bands (RGB)
        center_h = (output_shape[1] - center_data.shape[1]) // 2
        center_w = (output_shape[2] - center_data.shape[2]) // 2

        combined[
            :,
            center_h : center_h + center_data.shape[1],
            center_w : center_w + center_data.shape[2],
        ] = center_data

    # Load and merge neighbor images if any exist
    valid_neighbors = [n for n in neighbors if os.path.exists(n)]
    if valid_neighbors:
        src_files = [rasterio.open(neighbor) for neighbor in valid_neighbors]
        try:
            # Merge all neighbors into a single mosaic
            mosaic, transform = merge(src_files)
            mosaic = mosaic[:3, :, :]  # Keep only RGB bands

            # Clip mosaic to output shape if needed
            mosaic_h = min(mosaic.shape[1], output_shape[1])
            mosaic_w = min(mosaic.shape[2], output_shape[2])

            # Center-align the mosaic in the output canvas
            offset_h = (output_shape[1] - mosaic_h) // 2
            offset_w = (output_shape[2] - mosaic_w) // 2

            # Create a mask for areas in the combined image that are still nodata
            mask = (
                combined[
                    :, offset_h : offset_h + mosaic_h, offset_w : offset_w + mosaic_w
                ]
                == nodata_value
            )

            # Fill only nodata areas with the mosaic content (preserving center priority)
            combined[:, offset_h : offset_h + mosaic_h, offset_w : offset_w + mosaic_w][
                mask
            ] = mosaic[:, :mosaic_h, :mosaic_w][mask]
        finally:
            # Always close rasterio files
            for src in src_files:
                src.close()

    return combined


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
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

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
    output_gpkg = image_path.replace(".tif", ".gpkg")

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
        """Generates a 2D Hann window of shape (1, 1, H, W)"""
        hann_1d = torch.hann_window(size, periodic=False, device=device)
        window = torch.outer(hann_1d, hann_1d)  # (H, W)
        return window.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

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
    match = re.search(r"dop20_rgb_32(\d{3})_(\d{4})_", filename)
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


def main():
    """
    Main execution function for large-scale raster segmentation inference.

    This function:
    - Loads the trained model checkpoint
    - Runs inference on each image using sliding window with batched patches
    - Writes prediction results as GeoTIFFs with original metadata
    - Polygonizes predictions and postprocesses using parallel processing
    """
    args = get_args()
    seed_everything(42)

    # Load configuration and model from checkpoint
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"),
        config=config,
    )
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    # Prepare dataset and dataloader
    dataset = InferenceDataset(
        image_dir=args.image_path,
        patch_size=args.patch_size,
        transform=albu.Normalize(),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.output_path, exist_ok=True)

    # Process each batch of large images
    for batch in tqdm(dataloader, desc="Processing Images"):

        images = batch["image"].cuda()
        image_names = batch["image_name"]
        input_paths = [os.path.join(args.image_path, name) for name in image_names]

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
            entropy_basic = entropy_map_basic[i].cpu().numpy().astype(np.float32)

            input_path = input_paths[i]
            base_name = os.path.splitext(image_names[i])[0]
            output_pred_file = os.path.join(args.output_path, f"{base_name}.tif")
            # Define the entropy subfolder path
            entropy_subfolder = os.path.join(args.output_path, "entropy_maps")

            # Create the folder if it doesn't exist
            os.makedirs(entropy_subfolder, exist_ok=True)

            # Define the output file path within the entropy subfolder
            output_entropy_basic_file = os.path.join(
                entropy_subfolder, f"{base_name}_entropy_basic.tif"
            )

            written_tif_paths.append(output_pred_file)

            with rasterio.open(input_path) as src:
                height, width = src.height, src.width

                def center_crop(arr):
                    if arr.shape != (height, width):
                        center_h = (arr.shape[0] - height) // 2
                        center_w = (arr.shape[1] - width) // 2
                        return arr[
                            center_h : center_h + height, center_w : center_w + width
                        ]
                    return arr

                prediction = center_crop(prediction)
                entropy_basic = center_crop(entropy_basic)
                entropy_basic = (center_crop(entropy_basic) * 1000).astype(np.uint16)

                profile = src.profile
                profile.update(
                    dtype=rasterio.uint8, count=1, compress="lzw", photometric=None
                )
                with rasterio.open(output_pred_file, "w", **profile) as dst:
                    dst.write(prediction, 1)

                profile.update(dtype=rasterio.uint16)
                with rasterio.open(output_entropy_basic_file, "w", **profile) as dst:
                    dst.write(entropy_basic, 1)

        run_parallel_polygonization(written_tif_paths)

    gpkg_path = "data/utm_grid/Sachen_Grid_ETRS89-UTM32_1km.gpkg"
    tile_grid = gpd.read_file(gpkg_path)

    # Convert string path to Path object
    entropy_subfolder = Path(os.path.join(args.output_path, "entropy_maps"))

    for raster_path in entropy_subfolder.glob("*.tif"):
        tile_grid = aggregate_uncertainty_by_filename(
            raster_path, tile_grid, threshold=600, prefix="entropy"
        )

    tile_grid.to_file("results/grid_with_entropy.gpkg", driver="GPKG")


if __name__ == "__main__":
    main()
