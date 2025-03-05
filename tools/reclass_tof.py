import os
import rasterio
import argparse


def reclass_and_save(input_folder, output_folder):
    """
    Reclassifies all TIFF rasters in the input folder to 2 where values are 2, 3, or 4,
    and saves the resulting rasters to the output folder.

    Args:
        input_folder (str): Path to the input folder containing TIFF rasters.
        output_folder (str): Path to the output folder for saving the reclassified rasters.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            print(f"Processing {filename}")
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with rasterio.open(input_path, "r+") as src:
                data = src.read(1)  # Read the first band
                data[data == 2] = 2
                data[data == 3] = 2
                data[data == 4] = 2
                src.write(data, 1)

            # Save the modified raster to the output folder
            with rasterio.open(output_path, "w", **src.meta) as dst:
                dst.write(data, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reclassify TIFF rasters")
    parser.add_argument("--input", help="Input folder path", required=True)
    parser.add_argument("--output", help="Output folder path", required=True)
    args = parser.parse_args()

    reclass_and_save(args.input, args.output)
