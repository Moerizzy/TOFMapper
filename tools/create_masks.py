import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import os
import argparse


def argparse_TOF():
    """Parse the command line arguments for the TOF script"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=str,
        help="Set it to the foldernames of your sites as string",
    )
    parser.add_argument(
        "--eroded",
        type=bool,
        default=False,
        help="Set it to True if you want to erode the mask boundaries",
    )
    parser.add_argument(
        "--epsg",
        type=str,
        default="EPSG:25832",
        help="Set the EPSG code of your data",
    )
    return parser.parse_args()


def get_file_list(directory, extension=".tif", type=None):
    """Get a list of files with a given extension in a directory and optionally rename them with a prefix"""
    files = os.listdir(directory)
    file_list = []
    for file in files:
        if type:
            if file.endswith(".tif"):
                filename, _ = os.path.splitext(file)
                new_filename = f"{type}_{filename[4:]}{extension}"
                new_directory = directory.replace("TOP", type)
                new_file_path = os.path.join(new_directory, new_filename)
                # Create the directory if it doesn't exist
                os.makedirs(new_directory, exist_ok=True)
                file_list.append(new_file_path)
        else:
            if file.endswith(extension):
                file_list.append(f"{directory}/{file}")

    return sorted(file_list)


if __name__ == "__main__":

    args = argparse_TOF()
    state = args.state
    epsg = args.epsg
    eroded = args.eroded

    list_path_RGBI = get_file_list(f"data/sites/{state}/TOP")
    path_Shape = f"data/sites/{state}/SHP/{state}_TOF.shp"

    gdf = gpd.read_file(path_Shape)
    gdf["classvalue"] = gdf["classvalue"].astype(int)

    # Check if mask folder exists
    if not os.path.exists(f"data/sites/{state}/Masks"):
        os.makedirs(f"data/sites/{state}/Masks")

    for i, path_RGBI in enumerate(list_path_RGBI):

        name = path_RGBI.split("/")[-1].replace("TOP", "mask")
        mask_path = f"data/sites/{state}/Masks/{name}"

        with rasterio.open(path_RGBI[:-4] + ".tif") as ref_raster:
            with rasterio.open(
                mask_path,
                "w",
                driver="GTiff",
                height=ref_raster.height,
                width=ref_raster.width,
                count=1,
                dtype=rasterio.uint16,
                crs=ref_raster.crs,
                transform=ref_raster.transform,
            ) as mask_raster:
                burned = rasterize(
                    [
                        (geometry, value)
                        for geometry, value in zip(gdf.geometry, gdf["classvalue"])
                    ],
                    out_shape=ref_raster.shape,
                    all_touched=False,
                    transform=ref_raster.transform,
                    fill=0,
                    dtype=rasterio.uint16,
                    default_value=1,
                )

                if eroded:
                    # Implement erosion here
                    pass

                mask_raster.write(burned, 1)

        print(f"{i+1}/{len(list_path_RGBI)}: {name} created at {mask_path}")
