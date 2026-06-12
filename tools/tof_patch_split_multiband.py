"""Tile multiband images + masks into fixed-size patches, preserving all bands.

This is a rasterio-based replacement for the RGB-only `tof_patch_split.py`.
It works for any number of bands (3, 4, 5, ...), so the same script handles
RGB-only, RGB+NIR, RGB+NIR+nDSM, etc.

Inputs
------
--img-dir   folder with N-band GeoTIFFs (any band count, uint8 expected)
--mask-dir  folder with single-band GeoTIFFs (class indices)

Pairing rule
------------
Image and mask files are matched by basename. By default this means
`<id>.tif` <-> `<id>.tif`. If your masks have a different prefix/suffix
(e.g. `mask_<id>.tif`), pass `--mask-prefix mask_` and the script will
strip it when matching.

Outputs
-------
--output-img-dir   N-band <id>_<m>_<k>.tif tiles
--output-mask-dir  single-band <id>_<m>_<k>.tif tiles

Train mode also writes flipped/rotated copies (m=0..3); val/test writes m=0
only. Tiles whose mask is fully background can be discarded with
`--drop-empty`.

Usage
-----
    # train tiles (1024x1024, stride 1024)
    python TOFMapper/tools/tof_patch_split_multiband.py \
        --img-dir  data/tof/train_images \
        --mask-dir data/tof/train_masks  \
        --output-img-dir  data/tof/train/images_1024 \
        --output-mask-dir data/tof/train/masks_1024 \
        --mode train --split-size 1024 --stride 1024
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.windows import Window


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--output-img-dir", required=True)
    p.add_argument("--output-mask-dir", required=True)
    p.add_argument("--img-prefix", default="",
                   help="Optional prefix on image filenames to strip when matching ids.")
    p.add_argument("--mask-prefix", default="",
                   help="Optional prefix on mask filenames to strip when matching ids.")
    p.add_argument("--mode", choices=["train", "val", "test"], default="train")
    p.add_argument("--split-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--drop-empty", action="store_true",
                   help="Skip tiles whose mask is entirely background (class 0).")
    p.add_argument("--ignore-index", type=int, default=255,
                   help="Padding fill value for masks (default 255).")
    return p.parse_args()


def list_ids(folder, prefix):
    out = {}
    for f in os.listdir(folder):
        if not f.endswith(".tif"):
            continue
        stem = os.path.splitext(f)[0]
        if prefix and stem.startswith(prefix):
            tile_id = stem[len(prefix):]
        else:
            tile_id = stem
        out[tile_id] = os.path.join(folder, f)
    return out


def aug_variants(arr, m):
    """m: 0=identity, 1=hflip, 2=vflip, 3=rot90 (last axis)."""
    if m == 0:
        return arr
    if m == 1:
        return np.flip(arr, axis=-1).copy()
    if m == 2:
        return np.flip(arr, axis=-2).copy()
    if m == 3:
        return np.rot90(arr, k=1, axes=(-2, -1)).copy()
    raise ValueError(m)


def pad_to_multiple(arr, patch, fill):
    """Pad last two dims so they are multiples of `patch`."""
    h, w = arr.shape[-2], arr.shape[-1]
    ph = (patch - h % patch) % patch
    pw = (patch - w % patch) % patch
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(arr, pad, mode="constant", constant_values=fill)


def write_tile(out_path, arr, profile_template, dtype):
    """arr: (C, H, W) for image or (H, W) for mask."""
    if arr.ndim == 2:
        count = 1
        height, width = arr.shape
    else:
        count = arr.shape[0]
        height, width = arr.shape[1], arr.shape[2]
    profile = profile_template.copy()
    profile.update(
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        compress="lzw",
        photometric=None,
        transform=rasterio.Affine.identity(),
        crs=None,
        nodata=None,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr.astype(dtype), 1)
        else:
            for b in range(count):
                dst.write(arr[b].astype(dtype), b + 1)


def process_one(tile_id, img_path, mask_path, args, m_values):
    with rasterio.open(img_path) as src_img:
        img = src_img.read()  # (C, H, W)
        img_profile = src_img.profile.copy()
    with rasterio.open(mask_path) as src_msk:
        msk = src_msk.read(1)  # (H, W)
        msk_profile = src_msk.profile.copy()

    if img.shape[-2:] != msk.shape[-2:]:
        print(f"  [{tile_id}] shape mismatch img={img.shape} mask={msk.shape}, skipping")
        return 0

    img = pad_to_multiple(img, args.split_size, fill=0)
    msk = pad_to_multiple(msk, args.split_size, fill=args.ignore_index)

    n_written = 0
    for m in m_values:
        img_v = aug_variants(img, m)
        msk_v = aug_variants(msk, m)
        H, W = img_v.shape[-2], img_v.shape[-1]

        k = 0
        for y in range(0, H - args.split_size + 1, args.stride):
            for x in range(0, W - args.split_size + 1, args.stride):
                img_tile = img_v[:, y:y + args.split_size, x:x + args.split_size]
                msk_tile = msk_v[y:y + args.split_size, x:x + args.split_size]

                if args.drop_empty and not np.any(msk_tile != 0):
                    k += 1
                    continue

                out_img = os.path.join(
                    args.output_img_dir, f"{tile_id}_{m}_{k}.tif"
                )
                out_msk = os.path.join(
                    args.output_mask_dir, f"{tile_id}_{m}_{k}.tif"
                )
                write_tile(out_img, img_tile, img_profile, dtype="uint8")
                write_tile(out_msk, msk_tile, msk_profile, dtype="uint8")
                n_written += 1
                k += 1
    return n_written


def main():
    args = parse_args()
    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)

    img_map = list_ids(args.img_dir, args.img_prefix)
    msk_map = list_ids(args.mask_dir, args.mask_prefix)

    common = sorted(set(img_map) & set(msk_map))
    only_img = sorted(set(img_map) - set(msk_map))
    only_msk = sorted(set(msk_map) - set(img_map))

    print(f"images={len(img_map)} masks={len(msk_map)} paired={len(common)}")
    if only_img:
        print(f"  WARN images without mask: {only_img[:5]}{' ...' if len(only_img) > 5 else ''}")
    if only_msk:
        print(f"  WARN masks without image: {only_msk[:5]}{' ...' if len(only_msk) > 5 else ''}")

    m_values = [0, 1, 2, 3] if args.mode == "train" else [0]

    total = 0
    for i, tile_id in enumerate(common, 1):
        n = process_one(tile_id, img_map[tile_id], msk_map[tile_id], args, m_values)
        total += n
        print(f"  [{i}/{len(common)}] {tile_id}: {n} tiles")

    print(f"done. wrote {total} tiles to {args.output_img_dir}")


if __name__ == "__main__":
    main()
