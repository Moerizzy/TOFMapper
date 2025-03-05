import shutil
import os

sites = ["BB", "NRW_1", "NRW_3", "SH"]


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def copy_files(file_path, destination_images, destination_masks):
    file_path = file_path.strip()
    file_path_image = file_path.replace("mask", "TOP").replace("Masks", "TOP")
    if file_path:
        try:
            shutil.copy2(file_path_image, destination_images)
            shutil.copy2(file_path, destination_masks)
        except Exception as e:
            print(f"Failed to copy {file_path}: {e}")


def process_sites(test_site, train_val_sites, base_dir):
    destination_directory_val_masks = os.path.join(base_dir, "val_masks")
    destination_directory_val_images = os.path.join(base_dir, "val_images")
    destination_directory_test_masks = os.path.join(base_dir, "test_masks")
    destination_directory_test_images = os.path.join(base_dir, "test_images")
    destination_directory_train_images = os.path.join(base_dir, "train_images")
    destination_directory_train_masks = os.path.join(base_dir, "train_masks")

    # Ensure all directories exist
    for dir in [
        destination_directory_val_masks,
        destination_directory_val_images,
        destination_directory_test_masks,
        destination_directory_test_images,
        destination_directory_train_images,
        destination_directory_train_masks,
    ]:
        ensure_directory(dir)

    # Process test site
    print(f"Processing test site: {test_site}")
    text_file_path_test = f"data/sites/{test_site}/Stats/selected_masks_test.txt"
    with open(text_file_path_test, "r") as file:
        for line in file:
            copy_files(
                line,
                destination_directory_test_images,
                destination_directory_test_masks,
            )

    # Process train and validation sites
    for site in train_val_sites:
        print(f"Processing train/val site: {site}")
        text_file_path_val = f"data/sites/{site}/Stats/selected_masks_val.txt"

        # Copy validation files
        with open(text_file_path_val, "r") as file:
            for line in file:
                copy_files(
                    line,
                    destination_directory_val_images,
                    destination_directory_val_masks,
                )

        # Copy remaining files to train set
        print(f"Copying remaining files from {site} to train set.")
        for file in os.listdir(f"data/sites/{site}/Masks"):
            if file not in os.listdir(destination_directory_val_masks):
                shutil.copy2(
                    f"data/sites/{site}/Masks/{file}",
                    destination_directory_train_masks,
                )
                shutil.copy2(
                    f"data/sites/{site}/TOP/{file.replace('mask', 'TOP')}",
                    destination_directory_train_images,
                )


# Main execution
for i, test_site in enumerate(sites):
    train_val_sites = [site for site in sites if site != test_site]
    base_dir = f"data/combination_{i+1}"

    print(f"\nCreating dataset combination {i+1}")
    print(f"Test site: {test_site}")
    print(f"Train and validation sites: {train_val_sites}")

    ensure_directory(base_dir)
    process_sites(test_site, train_val_sites, base_dir)

print("All dataset combinations created.")
