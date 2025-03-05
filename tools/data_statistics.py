import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for the TOF script.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=str,
        help="Set it to the foldernames of your sites as string",
    )
    return parser.parse_args()


def get_statistics(mask_path: str, num_classes: int) -> np.ndarray:
    """
    Calculate the class frequencies of a given mask image.

    Args:
        mask_path (str): The path to the mask image file.
        num_classes (int): The number of classes in the mask image.

    Returns:
        numpy.ndarray: An array containing the class frequencies.

    """
    mask = Image.open(mask_path)
    mask_crop = np.array(mask)
    bins = np.array(range(num_classes))
    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)
    cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])

    return cf


def main() -> None:
    args = parse_args()
    state = args.state
    masks_dir = Path(f"data/sites/{state}/Masks")
    mask_paths = [file for file in masks_dir.iterdir() if file.suffix == ".tif"]
    num_classes = 6
    cf_list = []
    for mask_path in mask_paths:
        cf = get_statistics(str(mask_path), num_classes)
        cf_list.append(cf)

    # Calculate average distribution
    cf_list = np.array(cf_list)
    cf_avg = np.mean(cf_list, axis=0)
    print("Average Distribution:", cf_avg)

    # Choose 5 Validation Masks with similar distribution as the average
    mask_paths_selected_val = []
    cf_avg_select_val = np.zeros(num_classes - 1)
    while not np.allclose(cf_avg, cf_avg_select_val, atol=0.01):
        cf_list = []
        mask_paths_selected_val = []
        while len(mask_paths_selected_val) < 5:
            mask_path = np.random.choice(mask_paths)
            if mask_path not in mask_paths_selected_val:
                mask_paths_selected_val.append(mask_path)
                cf = get_statistics(str(mask_path), num_classes)
                cf_list.append(cf)
        cf_list = np.array(cf_list)
        cf_avg_select_val = np.mean(cf_list, axis=0)

    print("Average Distribution Selected Validation Masks:", cf_avg_select_val)
    print("Selected Validation Masks:", mask_paths_selected_val)

    # Choose 5 Testing Masks with similar distribution as the average but not part of the validation set
    mask_paths_test = [file for file in masks_dir.iterdir() if file.suffix == ".tif"]
    mask_paths_test = [
        file for file in mask_paths_test if file not in mask_paths_selected_val
    ]
    # Choose 5 Validation Masks with similar distribution as the average
    mask_paths_selected_test = []
    cf_avg_select_test = np.zeros(num_classes - 1)
    while not np.allclose(cf_avg, cf_avg_select_test, atol=0.01):
        cf_list = []
        mask_paths_selected_test = []
        while len(mask_paths_selected_test) < 5:
            mask_path = np.random.choice(mask_paths)
            if mask_path not in mask_paths_selected_test:
                mask_paths_selected_test.append(mask_path)
                cf = get_statistics(str(mask_path), num_classes)
                cf_list.append(cf)
        cf_list = np.array(cf_list)
        cf_avg_select_test = np.mean(cf_list, axis=0)

    print("Average Distribution Selected Testing Masks:", cf_avg_select_test)
    print("Selected Testing Masks:", mask_paths_selected_test)

    # Save average distribution and selected masks and their distributions
    stats_dir = Path(f"data/sites/{state}/Stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(stats_dir / "average_distribution.txt", cf_avg)
    np.savetxt(stats_dir / "selected_masks_val.txt", mask_paths_selected_val, fmt="%s")
    np.savetxt(stats_dir / "selected_masks_distribution_val.txt", cf_avg_select_val)
    np.savetxt(
        stats_dir / "selected_masks_test.txt", mask_paths_selected_test, fmt="%s"
    )
    np.savetxt(stats_dir / "selected_masks_distribution_test.txt", cf_avg_select_test)


if __name__ == "__main__":
    main()
