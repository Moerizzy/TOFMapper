"""Shared dataset-construction helpers for the multimodal TOF configs.

All configs use the SAME on-disk 5-band stack (RGB + NIR + nDSM). They only
differ in:
    * `band_indices`    -- which bands to feed the model
    * `in_chans`        -- the model's input channel count
    * `weights_name`    -- the run identifier

This keeps the experiments perfectly comparable: same files, same splits.

Expected on-disk layout (matches what `tools/copy_files.py` +
`tools/tof_patch_split_multiband.py` produce):

    DATA_ROOT/train/images_1024/<id>.tif       DATA_ROOT/train/masks_1024/<id>.tif
    DATA_ROOT/val/images_1024/<id>.tif         DATA_ROOT/val/masks_1024/<id>.tif
    DATA_ROOT/test_images/<id>.tif             DATA_ROOT/test_masks/mask_<id>.tif
"""

from geoseg.datasets.tof_dataset_multiband_norm import (
    TOFDataset,
    make_train_aug,
    make_val_aug,
)


def build_datasets(data_root, band_indices):
    """Return (train_ds, val_ds, test_ds) for the flat data layout."""
    n_chans = len(band_indices)
    train_aug_fn = make_train_aug(n_chans)
    val_aug_fn = make_val_aug(n_chans)

    train_ds = TOFDataset(
        data_root=f"{data_root}/train",
        mode="test",  # disables mosaic, matches existing convention
        transform=train_aug_fn,
        img_dir="images_1024",
        img_suffix=".tif",
        mask_dir="masks_1024",
        mask_suffix=".tif",
        img_size=(1024, 1024),
        band_indices=band_indices,
    )
    val_ds = TOFDataset(
        data_root=f"{data_root}/val",
        mode="val",
        transform=val_aug_fn,
        img_dir="images_1024",
        img_suffix=".tif",
        mask_dir="masks_1024",
        mask_suffix=".tif",
        img_size=(1024, 1024),
        band_indices=band_indices,
    )
    test_ds = TOFDataset(
        data_root=data_root,
        mode="test",
        transform=val_aug_fn,
        img_dir="test_images",
        img_suffix=".tif",
        mask_dir="test_masks",
        mask_suffix=".tif",
        mask_prefix="mask_",
        img_size=(5000, 5000),
        band_indices=band_indices,
    )

    return train_ds, val_ds, test_ds
