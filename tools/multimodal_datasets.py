"""Shared dataset-construction helpers for the multimodal TOF configs.

All configs use the SAME on-disk 5-band stack (RGB + NIR + nDSM). They only
differ in:
    * `band_indices`    -- which bands to feed the model
    * `in_chans`        -- the model's input channel count
    * `weights_name`    -- the run identifier

This keeps the experiments perfectly comparable: same files, same splits.
"""

from torch.utils.data import ConcatDataset
from geoseg.datasets.tof_dataset_multiband_norm import (
    TOFDataset,
    make_train_aug,
    make_val_aug,
)

# Region folder names under DATA_ROOT.
REGIONS = ("BB_2022", "NRW_N_2022", "NRW_S_2023", "SH_2021")


def build_region_datasets(data_root, band_indices, regions=REGIONS):
    """Return (train_ds, val_ds, test_ds) ConcatDatasets across all regions."""
    n_chans = len(band_indices)
    train_aug_fn = make_train_aug(n_chans)
    val_aug_fn = make_val_aug(n_chans)

    train_list, val_list, test_list = [], [], []
    for region in regions:
        train_list.append(
            TOFDataset(
                data_root=f"{data_root}/{region}/train",
                mode="test",  # disables mosaic, matches existing convention
                transform=train_aug_fn,
                img_dir="images_1024",
                img_suffix=".tif",
                mask_dir="masks_1024",
                mask_suffix=".tif",
                img_size=(1024, 1024),
                band_indices=band_indices,
            )
        )
        val_list.append(
            TOFDataset(
                data_root=f"{data_root}/{region}/val",
                mode="val",
                transform=val_aug_fn,
                img_dir="images_1024",
                img_suffix=".tif",
                mask_dir="masks_1024",
                mask_suffix=".tif",
                img_size=(1024, 1024),
                band_indices=band_indices,
            )
        )
        test_list.append(
            TOFDataset(
                data_root=f"{data_root}/{region}/test",
                mode="test",
                transform=val_aug_fn,
                img_dir="images",
                img_suffix=".tif",
                mask_dir="masks",
                mask_suffix="_mask.tif",
                img_size=(5000, 5000),
                band_indices=band_indices,
            )
        )

    return (
        ConcatDataset(train_list),
        ConcatDataset(val_list),
        ConcatDataset(test_list),
    )
