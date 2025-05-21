import geopandas as gpd
import numpy as np
import rasterio
from pathlib import Path
import argparse
import re


def extract_coords_from_filename(filename):
    """
    Extract x_sw and y_sw from filename like 'image_32_327000_5605000.tiff'
    """
    match = re.search(r"image_32_(\d{3})000_(\d{4})000\entropy_basic.tiff", filename)
    if match:
        x_sw = int(match.group(1)) * 1000
        y_sw = int(match.group(2)) * 1000
        return x_sw, y_sw
    else:
        return None, None


def aggregate_uncertainty_by_filename(
    raster_path, tile_grid, threshold=0.5, prefix="entropy"
):
    raster_path = Path(raster_path)
    x_sw, y_sw = extract_coords_from_filename(raster_path.name)

    if x_sw is None:
        print(f"[WARN] Could not parse coordinates from filename: {raster_path.name}")
        return tile_grid

    with rasterio.open(raster_path) as src:
        entropy_map = src.read(1, masked=True)

    values = entropy_map.compressed()
    stats = {}

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

    tile_grid = tile_grid.copy()
    idx = tile_grid[(tile_grid["x_sw"] == x_sw) & (tile_grid["y_sw"] == y_sw)].index
    if not idx.empty:
        for key, val in stats.items():
            if key not in tile_grid.columns:
                tile_grid[key] = np.nan
            tile_grid.loc[idx, key] = val
    else:
        print(
            f"[WARN] No matching tile for {raster_path.name} (x_sw={x_sw}, y_sw={y_sw})"
        )

    return tile_grid


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate uncertainty maps into tile grid."
    )
    parser.add_argument("grid_path", help="Path to tile grid (Shapefile or GeoParquet)")
    parser.add_argument("raster_dir", help="Directory containing uncertainty rasters")
    parser.add_argument("output_path", help="Output path for updated tile grid")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for high uncertainty"
    )
    parser.add_argument(
        "--prefix", type=str, default="entropy", help="Prefix for output columns"
    )

    args = parser.parse_args()

    grid = gpd.read_file(args.grid_path)
    raster_dir = Path(args.raster_dir)

    raster_files = list(raster_dir.glob("*.tiff"))

    print(f"[INFO] Found {len(raster_files)} raster files in {raster_dir}")

    for i, raster in enumerate(raster_files, 1):
        print(f"[{i}/{len(raster_files)}] Processing {raster.name}")
        grid = aggregate_uncertainty_by_filename(
            raster, grid, threshold=args.threshold, prefix=args.prefix
        )

    # Save output
    if args.output_path.endswith(".gpkg"):
        grid.to_file(args.output_path, driver="GPKG")
    else:
        grid.to_file(args.output_path)

    print(f"[DONE] Saved updated grid with uncertainty stats to {args.output_path}")


if __name__ == "__main__":
    main()

# Example usage:
# python TOFMapper/aggregation_uncertainty.py data/utm_grid/Sachsen_Grid_ETRS89-UTM32_1km.gpkg results/entropy_maps data/utm_grid/Sachsen_Grid_ETRS89-UTM32_1km_uncertainty.gpkg --threshold 0.5 --prefix "entropy"
