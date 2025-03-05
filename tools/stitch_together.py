import os
import sys
from PIL import Image
from collections import defaultdict
import rasterio
import zipfile


def stitch_images_grid(image_group, geospatial_path, output_path):
    # Sort images by their filename to ensure they are in order
    image_group.sort()

    # Open all images and find their dimensions
    images = [Image.open(img) for img in image_group]
    widths, heights = zip(*(img.size for img in images))

    # Determine the dimensions of the grid
    width = max(widths)
    height = max(heights)

    # # Calculate the total width and height for the final image
    # total_width = max_width * 2  # Two images per row
    # total_height = max_height * 2  # Two rows

    # Create a new blank image with the calculated dimensions
    stitched_image = Image.new("L", (width, height))  # 'L' mode for grayscale

    # Arrange the images in the order: 0, 1, 2, 3
    for i, img in enumerate(images):
        if i >= 1:
            break
        stitched_image.paste(img, (0, 0))

    # Cut the padding from the stitched image
    # stitched_image = stitched_image.crop((0, 0, total_width, total_height))

    # Use the geospatial path to turn the stitched image into a geotiff
    stitched_image.save(output_path.replace(".png", ".tif"))

    # stitched image to a np array
    stitched_image = rasterio.open(output_path.replace(".png", ".tif")).read(1)

    # Find the corresponding image to get geospatial information
    with rasterio.open(geospatial_path) as src:
        # Get the geospatial information from the corresponding image
        geospatial_info = src.meta

    # Save the stitched image as a geotiff with the updated geospatial information
    with rasterio.open(
        output_path.replace(".png", ".tif"), "w", **geospatial_info
    ) as dst:
        dst.write(stitched_image, indexes=1)

    # Save the geotif as shapefile
    os.system(
        f"gdal_polygonize.py {output_path.replace('.png', '.tif')} -f 'ESRI Shapefile' {output_path.replace('.png', '.shp')}"
    )


def create_zip_file(input_path, output_folder):
    zip_path = os.path.join(input_path, "stitched_output.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, input_path)
                zipf.write(file_path, arcname)
    return zip_path


def main(input_path):
    # Dictionary to group images by their identifier
    image_groups = defaultdict(list)
    # Read all PNG files in the specified directory
    for image in os.listdir(input_path):
        if image.endswith(".png"):
            # Extract the identifier from the filename
            identifier = "_".join(image.split("_")[:3])
            image_groups[identifier].append(os.path.join(input_path, image))

    # Create output folder
    output_folder = os.path.join(input_path, "stitched")
    os.makedirs(output_folder, exist_ok=True)

    # Process each group and stitch images together
    for identifier, images in image_groups.items():
        output_path = os.path.join(output_folder, f"{identifier}.png")
        geospatial_path = os.path.join("data/tof/test_masks", f"mask_{identifier}.tif")
        stitch_images_grid(images, geospatial_path, output_path)
        print(f"Stitched image saved as {output_path}")

    # Create a ZIP file containing all output files
    zip_path = create_zip_file(input_path, output_folder)
    print(f"All output files have been zipped to {zip_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_path>")
        sys.exit(1)
    input_path = sys.argv[1]
    if not os.path.isdir(input_path):
        print(f"Error: {input_path} is not a valid directory.")
        sys.exit(1)
    main(input_path)
