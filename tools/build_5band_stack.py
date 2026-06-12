"""Build 5-band RGB + NIR + nDSM stacks per site.

Reads:
    data/sites/<STATE>/TOP/TOP_<id>.tif      (4 bands: R, G, B, NIR; uint8 or uint16)
    data/sites/<STATE>/nDSM/nDSM_<id>.tif    (1 band, float32 meters)

Writes:
    data/sites/<STATE>/RGBN_nDSM/<id>.tif    (5 bands uint8: R, G, B, NIR, nDSM)

The nDSM band is clipped to [ndsm_min, ndsm_max] meters (default 0..50) and
linearly mapped to 0..255 uint8. The TOP raster's CRS / transform / shape is
authoritative; the nDSM is reprojected onto that grid (no-op when already
aligned, but safe otherwise).

Usage:
    python TOFMapper/tools/build_5band_stack.py --state BB
    python TOFMapper/tools/build_5band_stack.py --state BB --ndsm-max 80
    python TOFMapper/tools/build_5band_stack.py --state BB \
        --top-dir TOP --ndsm-dir nDSM --out-dir RGBN_nDSM
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.errors import CRSError
from rasterio.warp import reproject, Resampling

try:
    # CPLE_NotSupportedError lives here in modern rasterio.
    from rasterio._err import CPLE_NotSupportedError  # type: ignore
except Exception:  # pragma: no cover - older rasterio
    CPLE_NotSupportedError = Exception  # noqa: N816


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state", type=str, required=True, help="Site folder name under data/sites/"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/sites",
        help="Root containing per-state subfolders.",
    )
    parser.add_argument(
        "--top-dir",
        type=str,
        default="TOP",
        help="RGBI subfolder name (files: TOP_<id>.tif).",
    )
    parser.add_argument(
        "--ndsm-dir",
        type=str,
        default="nDSM",
        help="nDSM subfolder name (files: nDSM_<id>.tif).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="RGBN_nDSM",
        help="Output subfolder name (files: <id>.tif).",
    )
    parser.add_argument(
        "--ndsm-min", type=float, default=0.0, help="Lower clip for nDSM in meters."
    )
    parser.add_argument(
        "--ndsm-max", type=float, default=50.0, help="Upper clip for nDSM in meters."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )
    return parser.parse_args()


def to_uint8_rgbn(arr):
    """Bring an (4, H, W) RGBI array into uint8 [0, 255]."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr // 256).astype(np.uint8)
    # generic float / signed: per-array min-max
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def ndsm_to_uint8(ndsm_m, lo=0.0, hi=50.0):
    """Clip nDSM in meters to [lo, hi] and map linearly to 0..255 uint8."""
    arr = np.clip(ndsm_m.astype(np.float32), lo, hi)
    if hi <= lo:
        raise ValueError("ndsm_max must be greater than ndsm_min")
    arr = (arr - lo) * (255.0 / (hi - lo))
    return arr.astype(np.uint8)


def build_one(top_path, ndsm_path, out_path, ndsm_min, ndsm_max, overwrite):
    if os.path.exists(out_path) and not overwrite:
        return "skip-exists"

    with rasterio.open(top_path) as src_top:
        if src_top.count < 4:
            return f"skip-only-{src_top.count}-bands"
        rgbn = src_top.read([1, 2, 3, 4])  # (4, H, W)
        H, W = src_top.height, src_top.width
        dst_transform = src_top.transform
        dst_crs = src_top.crs
        profile = src_top.profile.copy()

    rgbn_u8 = to_uint8_rgbn(rgbn)

    ndsm_aligned = np.zeros((H, W), dtype=np.float32)
    with rasterio.open(ndsm_path) as src_ndsm:
        # Fast path: if the nDSM is already on the TOP grid, just read it. This
        # also rescues files whose GeoTIFF CRS tags PROJ degrades to an
        # "EngineeringCRS" (no datum), which would otherwise crash reproject.
        same_grid = (
            src_ndsm.height == H
            and src_ndsm.width == W
            and src_ndsm.transform == dst_transform
        )
        if same_grid:
            ndsm_aligned = src_ndsm.read(1).astype(np.float32, copy=False)
        else:
            try:
                reproject(
                    source=rasterio.band(src_ndsm, 1),
                    destination=ndsm_aligned,
                    src_transform=src_ndsm.transform,
                    src_crs=src_ndsm.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
            except (CPLE_NotSupportedError, CRSError):
                # Fallback: PROJ can't build a transform between the two CRS
                # descriptions even though they should be equivalent. Resample
                # in pixel space, assuming the rasters share a CRS in practice.
                ndsm_aligned[:] = 0.0
                reproject(
                    source=rasterio.band(src_ndsm, 1),
                    destination=ndsm_aligned,
                    src_transform=src_ndsm.transform,
                    src_crs=dst_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

    ndsm_u8 = ndsm_to_uint8(ndsm_aligned, lo=ndsm_min, hi=ndsm_max)

    profile.update(
        count=5,
        dtype="uint8",
        driver="GTiff",
        compress="lzw",
        photometric=None,
        nodata=None,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgbn_u8[0], 1)
        dst.write(rgbn_u8[1], 2)
        dst.write(rgbn_u8[2], 3)
        dst.write(rgbn_u8[3], 4)
        dst.write(ndsm_u8, 5)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")
        dst.set_band_description(4, "NIR")
        dst.set_band_description(5, f"nDSM[{ndsm_min}-{ndsm_max}m]->uint8")
    return "ok"


def main():
    args = parse_args()

    top_dir = os.path.join(args.data_root, args.state, args.top_dir)
    ndsm_dir = os.path.join(args.data_root, args.state, args.ndsm_dir)
    out_dir = os.path.join(args.data_root, args.state, args.out_dir)

    if not os.path.isdir(top_dir):
        raise FileNotFoundError(f"TOP folder not found: {top_dir}")
    if not os.path.isdir(ndsm_dir):
        raise FileNotFoundError(f"nDSM folder not found: {ndsm_dir}")
    os.makedirs(out_dir, exist_ok=True)

    top_files = sorted(
        f for f in os.listdir(top_dir) if f.startswith("TOP_") and f.endswith(".tif")
    )
    if not top_files:
        raise RuntimeError(f"No TOP_*.tif files found in {top_dir}")

    print(f"[{args.state}] {len(top_files)} TOP files; out -> {out_dir}")

    n_ok = n_skip = n_missing = 0
    for i, top_name in enumerate(top_files, 1):
        tile_id = top_name[len("TOP_") : -len(".tif")]
        ndsm_name = f"nDOM_{tile_id}.tif"
        top_path = os.path.join(top_dir, top_name)
        ndsm_path = os.path.join(ndsm_dir, ndsm_name)
        out_path = os.path.join(out_dir, f"{tile_id}.tif")

        if not os.path.exists(ndsm_path):
            n_missing += 1
            print(f"  [{i}/{len(top_files)}] {tile_id}: MISSING nDSM ({ndsm_name})")
            continue

        status = build_one(
            top_path,
            ndsm_path,
            out_path,
            ndsm_min=args.ndsm_min,
            ndsm_max=args.ndsm_max,
            overwrite=args.overwrite,
        )
        if status == "ok":
            n_ok += 1
        else:
            n_skip += 1
        print(f"  [{i}/{len(top_files)}] {tile_id}: {status}")

    print(f"[{args.state}] done. ok={n_ok} skipped={n_skip} missing_ndsm={n_missing}")


if __name__ == "__main__":
    main()
