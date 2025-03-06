import argparse
import shutil
import os

"""
Usage:
    python copy_files.py [--sites SITE [SITE ...]] [--dest_train_images PATH]
                         [--dest_train_masks PATH] [--dest_val_images PATH]
                         [--dest_val_masks PATH] [--dest_test_images PATH]
                         [--dest_test_masks PATH]

Description:
    This script copies image and mask files into designated training, validation, 
    and testing folders. It reads file paths from text files located in each site's 
    "data/sites/{site}/Stats" directory and then copies the corresponding files to the 
    destination directories. Remaining files are copied to the training set.

Arguments:
    --sites             List of site names to process (default: ["BB", "NRW_1", "NRW_3", "SH"])
    --dest_train_images Destination folder for training images (default: data/tof/train_images)
    --dest_train_masks  Destination folder for training masks (default: data/tof/train_masks)
    --dest_val_images   Destination folder for validation images (default: data/tof/val_images)
    --dest_val_masks    Destination folder for validation masks (default: data/tof/val_masks)
    --dest_test_images  Destination folder for test images (default: data/tof/test_images)
    --dest_test_masks   Destination folder for test masks (default: data/tof/test_masks)

Examples:
    1. Run with default settings:
        python copy_files.py

    2. Process only two sites and use custom destination directories:
        python copy_files.py --sites SITE_A SITE_B \
            --dest_train_images "custom/train_images" \
            --dest_train_masks "custom/train_masks" \
            --dest_val_images "custom/val_images" \
            --dest_val_masks "custom/val_masks" \
            --dest_test_images "custom/test_images" \
            --dest_test_masks "custom/test_masks"

    3. Process a different list of sites with default folder paths:
        python copy_files.py --sites SITE_X SITE_Y SITE_Z
"""


def copy_files(
    sites,
    dest_train_images,
    dest_train_masks,
    dest_val_images,
    dest_val_masks,
    dest_test_images,
    dest_test_masks,
):
    for site in sites:
        text_file_path_val = f"data/sites/{site}/Stats/selected_masks_val.txt"
        text_file_path_test = f"data/sites/{site}/Stats/selected_masks_test.txt"

        # Process test set
        if os.path.exists(text_file_path_test):
            with open(text_file_path_test, "r") as file:
                for line in file:
                    file_path = line.strip()
                    file_path2 = file_path.replace("mask", "TOP").replace(
                        "Masks", "TOP"
                    )
                    print(f"Copying test image: {file_path2}")

                    if file_path:
                        try:
                            shutil.copy2(file_path2, dest_test_images)
                            shutil.copy2(file_path, dest_test_masks)
                        except Exception as e:
                            print(f"Failed to copy {file_path}: {e}")
        else:
            print(f"Test file not found for site {site}: {text_file_path_test}")

        # Process validation set
        if os.path.exists(text_file_path_val):
            with open(text_file_path_val, "r") as file:
                for line in file:
                    file_path = line.strip()
                    file_path2 = file_path.replace("mask", "TOP").replace(
                        "Masks", "TOP"
                    )
                    print(f"Copying validation image: {file_path2}")

                    if file_path:
                        try:
                            shutil.copy2(file_path2, dest_val_images)
                            shutil.copy2(file_path, dest_val_masks)
                        except Exception as e:
                            print(f"Failed to copy {file_path}: {e}")
        else:
            print(f"Validation file not found for site {site}: {text_file_path_val}")

        # Copy remaining files to the train set
        print(f"Copying remaining files to train set for site {site}.")
        masks_dir = f"data/sites/{site}/Masks"
        top_dir = f"data/sites/{site}/TOP"
        if os.path.exists(masks_dir) and os.path.exists(top_dir):
            # Get lists of files already copied into test and val folders
            test_files = set(os.listdir(dest_test_masks))
            val_files = set(os.listdir(dest_val_masks))
            for file in os.listdir(masks_dir):
                if file not in test_files and file not in val_files:
                    try:
                        src_mask = os.path.join(masks_dir, file)
                        src_top = os.path.join(top_dir, file.replace("mask", "TOP"))
                        shutil.copy2(src_mask, dest_train_masks)
                        shutil.copy2(src_top, dest_train_images)
                    except Exception as e:
                        print(f"Failed to copy training file {file}: {e}")
        else:
            print(f"Directories not found for site {site}: {masks_dir} or {top_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy files into designated training, validation, and testing folders."
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["BB", "NRW_1", "NRW_3", "SH"],
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
    )

    print("File copying complete.")


if __name__ == "__main__":
    main()
