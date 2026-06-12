"""Class-aware oversampling for the multimodal TOF training pipeline.

Builds a `WeightedRandomSampler` aligned with the ConcatDataset returned by
`tools.multimodal_datasets.build_region_datasets`, biasing sampling toward
tiles that contain rare TOF classes (Patch, Linear, Tree).

Per-tile class histograms are cached to disk (`_class_freq.npz` inside each
`masks_1024/` folder) so subsequent runs skip the I/O scan.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler


# Must match `CLASSES` in tof_dataset_multiband_norm.py
NUM_CLASSES = 5  # 0=Background, 1=Forest, 2=Patch, 3=Linear, 4=Tree

CACHE_NAME = "_class_freq.npz"


def _read_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def compute_tile_freqs(
    mask_dir: str,
    mask_suffix: str = ".tif",
    cache: bool = True,
    rebuild: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Return (sorted_ids, freqs) where freqs[i, c] = fraction of pixels in
    tile i belonging to class c. Pixels with values >= NUM_CLASSES (e.g. 255
    for ignore) do not contribute to any class but are counted in the
    denominator, so frequencies sum to <= 1.
    """
    cache_path = os.path.join(mask_dir, CACHE_NAME)
    if cache and not rebuild and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return list(data["ids"]), data["freqs"].astype(np.float32)

    files = sorted(f for f in os.listdir(mask_dir) if f.endswith(mask_suffix))
    if not files:
        raise RuntimeError(f"No '*{mask_suffix}' masks found in {mask_dir}")

    ids: list[str] = []
    freqs: list[np.ndarray] = []
    bins = np.arange(NUM_CLASSES + 1)  # [0,1),...,[4,5)
    for f in files:
        m = _read_mask(os.path.join(mask_dir, f))
        h, _ = np.histogram(m, bins=bins)
        ids.append(f[: -len(mask_suffix)])
        freqs.append(h.astype(np.float32) / float(m.size))

    freqs_arr = np.stack(freqs, axis=0)  # (N, NUM_CLASSES)
    if cache:
        try:
            np.savez(cache_path, ids=np.array(ids), freqs=freqs_arr)
        except OSError as e:
            print(f"  [oversampling] could not write cache to {cache_path}: {e}")
    return ids, freqs_arr


def _gather_region_freqs(
    data_root: str,
    regions: Sequence[str],
    mask_dir: str = "masks_1024",
    mask_suffix: str = ".tif",
    rebuild_cache: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Concatenate per-region per-tile freqs in the same order ConcatDataset
    will iterate them. Returns (ids, freqs)."""
    all_ids: list[str] = []
    chunks: list[np.ndarray] = []
    for region in regions:
        mdir = os.path.join(data_root, region, "train", mask_dir)
        ids, freqs = compute_tile_freqs(
            mdir, mask_suffix=mask_suffix, rebuild=rebuild_cache
        )
        all_ids.extend(f"{region}/{i}" for i in ids)
        chunks.append(freqs)
    return all_ids, np.concatenate(chunks, axis=0)


def make_weighted_sampler_for_concat(
    data_root: str,
    regions: Sequence[str],
    mask_dir: str = "masks_1024",
    mask_suffix: str = ".tif",
    oversample_classes: Iterable[int] = (2, 3, 4),  # Patch, Linear, Tree
    method: str = "inverse_freq",
    smooth: float = 1e-4,
    num_samples: int | None = None,
    rebuild_cache: bool = False,
    verbose: bool = True,
) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler whose length matches the train
    ConcatDataset built by `build_region_datasets(data_root, ..., regions)`.

    Parameters
    ----------
    oversample_classes : iterable of int
        Class indices (in 0..NUM_CLASSES-1) that should be upweighted.
    method : "inverse_freq" | "presence"
        See module docstring for details.
    smooth : float
        Lower bound for tile weights so no tile has zero probability.
    num_samples : int or None
        Number of samples drawn per epoch. Defaults to len(train_dataset).
    rebuild_cache : bool
        If True, recompute the on-disk per-tile histogram cache.
    """
    oversample_classes = list(oversample_classes)
    for c in oversample_classes:
        if not (0 <= c < NUM_CLASSES):
            raise ValueError(
                f"oversample_classes contains {c}, must be in 0..{NUM_CLASSES - 1}"
            )

    ids, freqs = _gather_region_freqs(
        data_root, regions, mask_dir=mask_dir, mask_suffix=mask_suffix,
        rebuild_cache=rebuild_cache,
    )
    n_total = freqs.shape[0]

    if method == "inverse_freq":
        mean_freq = freqs.mean(axis=0) + smooth
        class_weight = 1.0 / mean_freq
        # Zero-out classes we don't want to oversample.
        mask = np.zeros(NUM_CLASSES, dtype=np.float32)
        for c in oversample_classes:
            mask[c] = 1.0
        class_weight = class_weight * mask
        tile_weight = freqs @ class_weight  # (N,)
    elif method == "presence":
        tile_weight = np.zeros(n_total, dtype=np.float32)
        for c in oversample_classes:
            tile_weight = np.maximum(tile_weight, (freqs[:, c] > 0).astype(np.float32))
    else:
        raise ValueError(f"unknown method: {method}")

    # No tile should have zero probability — keep the long tail in the mix.
    tile_weight = tile_weight + smooth
    tile_weight = tile_weight / tile_weight.sum()

    if num_samples is None:
        num_samples = n_total

    if verbose:
        # Quick stats: effective per-class sampling probability.
        eff = (tile_weight[:, None] * freqs).sum(axis=0) / max(freqs.mean(axis=0).sum(), 1e-12)
        print(f"[oversampling] tiles={n_total}  method={method}")
        print(f"[oversampling] global class freq:        "
              + ", ".join(f"{i}:{v:.4f}" for i, v in enumerate(freqs.mean(axis=0))))
        # Effective class fraction expected per epoch.
        sampled_class_frac = (tile_weight[:, None] * freqs).sum(axis=0) * n_total
        sampled_class_frac = sampled_class_frac / max(sampled_class_frac.sum(), 1e-12)
        print(f"[oversampling] expected sampled freq:    "
              + ", ".join(f"{i}:{v:.4f}" for i, v in enumerate(sampled_class_frac)))

    return WeightedRandomSampler(
        weights=torch.from_numpy(tile_weight).double(),
        num_samples=int(num_samples),
        replacement=True,
    )
