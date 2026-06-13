import argparse
import shutil
import os

"""
Usage:
    python copy_files.py [--sites SITE [SITE ...]] [--dest_train_images PATH]
                         [--dest_train_masks PATH] [--dest_val_images PATH]
                         [--dest_val_masks PATH] [--dest_test_images PATH]
                         [--dest_test_masks PATH] [--img-subdir NAME]
                         [--img-prefix STR]

Description:
    This script copies image and mask files into designated training, validation,
    and testing folders. It reads file paths from text files located in each site's
    "data/sites/{site}/Stats" directory and then copies the corresponding files to
    the destination directories. Remaining files are copied to the training set.

    By default it now reads the 5-band stacks under "RGBN_nDSM/<id>.tif" produced
    by `tools/build_5band_stack.py`. The rest of the pipeline
    (tof_patch_split.py / multiband splitter / configs) is unchanged.

Arguments:
    --sites             List of site names to process (default: ["BB", "NRW_1", "NRW_3", "SH"])
    --dest_train_images Destination folder for training images   (default: data/tof/train_images)
    --dest_train_masks  Destination folder for training masks    (default: data/tof/train_masks)
    --dest_val_images   Destination folder for validation images (default: data/tof/val_images)
    --dest_val_masks    Destination folder for validation masks  (default: data/tof/val_masks)
    --dest_test_images  Destination folder for test images       (default: data/tof/test_images)
    --dest_test_masks   Destination folder for test masks        (default: data/tof/test_masks)
    --img-subdir        Source image subfolder per site          (default: RGBN_nDSM)
    --img-prefix        Prefix on the source image filenames     (default: "")

Examples:
    1. Run with default settings (reads RGBN_nDSM 5-band stacks):
        python copy_files.py

    2. Fall back to the original 4-band TOP folder:
        python copy_files.py --img-subdir TOP --img-prefix TOP_
"""


def _tile_id_from_mask(mask_filename):
    """Strip leading 'mask_' if present and the extension."""
    base = os.path.basename(mask_filename)
    stem, _ = os.path.splitext(base)
    if stem.startswith("mask_"):
        stem = stem[len("mask_") :]
    return stem


def _src_mask_path(site, tile_id):
    return os.path.join("data", "sites", site, "Masks", f"mask_{tile_id}.tif")


def _src_image_path(site, tile_id, img_subdir, img_prefix):
    return os.path.join("data", "sites", site, img_subdir, f"{img_prefix}{tile_id}.tif")


def copy_files(
    sites,
    dest_train_images,
    dest_train_masks,
    dest_val_images,
    dest_val_masks,
    dest_test_images,
    dest_test_masks,
    img_subdir,
    img_prefix,
):
    for site in sites:
        text_file_path_val = f"data/sites/{site}/Stats/selected_masks_val.txt"
        text_file_path_test = f"data/sites/{site}/Stats/selected_masks_test.txt"

        # Process test set
        if os.path.exists(text_file_path_test):
            with open(text_file_path_test, "r") as file:
                for line in file:
                    mask_path_in_txt = line.strip()
                    if not mask_path_in_txt:
                        continue
                    tile_id = _tile_id_from_mask(mask_path_in_txt)
                    img_path = _src_image_path(site, tile_id, img_subdir, img_prefix)
                    mask_path = _src_mask_path(site, tile_id)
                    print(f"Copying test image: {img_path}")
                    try:
                        shutil.copy2(img_path, dest_test_images)
                        shutil.copy2(mask_path, dest_test_masks)
                    except Exception as e:
                        print(f"Failed to copy {mask_path}: {e}")
        else:
            print(f"Test file not found for site {site}: {text_file_path_test}")

        # Process validation set
        if os.path.exists(text_file_path_val):
            with open(text_file_path_val, "r") as file:
                for line in file:
                    mask_path_in_txt = line.strip()
                    if not mask_path_in_txt:
                        continue
                    tile_id = _tile_id_from_mask(mask_path_in_txt)
                    img_path = _src_image_path(site, tile_id, img_subdir, img_prefix)
                    mask_path = _src_mask_path(site, tile_id)
                    print(f"Copying validation image: {img_path}")
                    try:
                        shutil.copy2(img_path, dest_val_images)
                        shutil.copy2(mask_path, dest_val_masks)
                    except Exception as e:
                        print(f"Failed to copy {mask_path}: {e}")
        else:
            print(f"Validation file not found for site {site}: {text_file_path_val}")

        # Copy remaining files to the train set
        print(f"Copying remaining files to train set for site {site}.")
        masks_dir = f"data/sites/{site}/Masks"
        img_dir = os.path.join("data", "sites", site, img_subdir)
        if os.path.exists(masks_dir) and os.path.exists(img_dir):
            test_files = set(os.listdir(dest_test_masks))
            val_files = set(os.listdir(dest_val_masks))
            for file in os.listdir(masks_dir):
                if file in test_files or file in val_files:
                    continue
                tile_id = _tile_id_from_mask(file)
                src_mask = os.path.join(masks_dir, file)
                src_img = _src_image_path(site, tile_id, img_subdir, img_prefix)
                try:
                    shutil.copy2(src_mask, dest_train_masks)
                    shutil.copy2(src_img, dest_train_images)
                except Exception as e:
                    print(f"Failed to copy training file {file}: {e}")
        else:
            print(f"Directories not found for site {site}: {masks_dir} or {img_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy files into designated training, validation, and testing folders."
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["BB", "NRW_N", "NRW_S", "SH"],
        help="List of site names to process.",
    )
    parser.add_argument(
        "--dest_train_images",
        default="data/tof/train_images",
        help="Destination folder for training images.",
    )
    parser.add_argument(
        "--dest_train_masks",
        default="data/tof/train_masks",
        help="Destination folder for training masks.",
    )
    parser.add_argument(
        "--dest_val_images",
        default="data/tof/val_images",
        help="Destination folder for validation images.",
    )
    parser.add_argument(
        "--dest_val_masks",
        default="data/tof/val_masks",
        help="Destination folder for validation masks.",
    )
    parser.add_argument(
        "--dest_test_images",
        default="data/tof/test_images",
        help="Destination folder for test images.",
    )
    parser.add_argument(
        "--dest_test_masks",
        default="data/tof/test_masks",
        help="Destination folder for test masks.",
    )
    parser.add_argument(
        "--img-subdir",
        default="RGBN_nDSM",
        help="Source image subfolder per site (default: RGBN_nDSM 5-band stacks).",
    )
    parser.add_argument(
        "--img-prefix",
        default="",
        help="Filename prefix on the source images (default: '').",
    )

    args = parser.parse_args()

    # Ensure destination directories exist
    for directory in [
        args.dest_train_images,
        args.dest_train_masks,
        args.dest_val_images,
        args.dest_val_masks,
        args.dest_test_images,
        args.dest_test_masks,
    ]:
        os.makedirs(directory, exist_ok=True)

    copy_files(
        sites=args.sites,
        dest_train_images=args.dest_train_images,
        dest_train_masks=args.dest_train_masks,
        dest_val_images=args.dest_val_images,
        dest_val_masks=args.dest_val_masks,
        dest_test_images=args.dest_test_images,
        dest_test_masks=args.dest_test_masks,
        img_subdir=args.img_subdir,
        img_prefix=args.img_prefix,
    )

    print("File copying complete.")


if __name__ == "__main__":
    main()
