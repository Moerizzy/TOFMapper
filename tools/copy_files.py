import shutil
import os

sites = ["BB", "NRW_1", "NRW_3", "SH"]

destination_directory_val_masks = "data/tif_tof/val_masks"
destination_directory_val_images = "data/tif_tof/val_images"
destination_directory_test_masks = "data/tif_tof/test_masks"
destination_directory_test_images = "data/tif_tof/test_images"
destination_directory_train_images = "data/tif_tof/train_images"
destination_directory_train_masks = "data/tif_tof/train_masks"


for site in sites:
    text_file_path_val = f"data/sites/{site}/Stats/selected_masks_val.txt"
    text_file_path_test = f"data/sites/{site}/Stats/selected_masks_test.txt"

    # Read the text file and copy each file
    with open(text_file_path_test, "r") as file:
        for line in file:
            # Get the path from the line and strip any surrounding whitespace
            file_path = line.strip()
            file_path2 = file_path.replace("mask", "TOP").replace("Masks", "TOP")
            # file_path = file_path.replace("Masks", "Masks_reclass")
            print(file_path2)

            # Only proceed if the path is not empty
            if file_path:
                # Copy the file to the destination directory
                try:
                    shutil.copy2(
                        file_path2,
                        destination_directory_test_images,
                    )
                    shutil.copy2(file_path, destination_directory_test_masks)
                except Exception as e:
                    print(f"Failed to copy {file_path}: {e}")

    with open(text_file_path_val, "r") as file:
        for line in file:
            # Get the path from the line and strip any surrounding whitespace
            file_path = line.strip()
            file_path2 = file_path.replace("mask", "TOP").replace("Masks", "TOP")
            # file_path = file_path.replace("Masks", "Masks_reclass")
            print(file_path2)

            # Only proceed if the path is not empty
            if file_path:
                # Copy the file to the destination directory
                try:
                    shutil.copy2(
                        file_path2,
                        destination_directory_val_images,
                    )
                    shutil.copy2(file_path, destination_directory_val_masks)
                except Exception as e:
                    print(f"Failed to copy {file_path}: {e}")

        # copying all the remaining files that are not part of the test set
        print("Copying remaining files to train set.")
        for file in os.listdir(f"data/sites/{site}/Masks"):

            if file not in os.listdir(
                destination_directory_test_masks
            ) and file not in os.listdir(destination_directory_val_masks):
                shutil.copy2(
                    f"data/sites/{site}/Masks/{file}",
                    destination_directory_train_masks,
                )
                shutil.copy2(
                    f"data/sites/{site}/TOP/{file.replace('mask', 'TOP')}",
                    destination_directory_train_images,
                )

print("File copying complete.")
