from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.config import DatasetSplit, EyeMaskDatasetConfig
from fisheye.training import zarr_eye_mask_dataset as mod


def _write_eye_training_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_purpose"] = "training"

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "merged_export_test"
    crop = crop_parent.create_group("merged_export_test")
    crop.create_array("roi_images", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(2, 8, 8))
    crop.create_array("roi_coordinates_full", data=np.zeros((4, 2), dtype=np.int32), chunks=(4, 2))

    eye_parent = root.create_group("eye_masks_runs")
    eye_parent.attrs["latest"] = "merged_export_test"
    eye = eye_parent.create_group("merged_export_test")
    masks = np.zeros((4, 2, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:3, 1:3] = 1
    masks[:, 1, 4:6, 4:6] = 1
    eye.create_array("masks_roi", data=masks, chunks=(2, 2, 8, 8))
    eye.create_array("ellipse_success", data=np.ones((4, 2), dtype=np.bool_), chunks=(4, 2))

    splits = root.create_group("splits")
    splits.create_array("train_indices", data=np.array([3, 1], dtype=np.int64), chunks=(2,))
    splits.create_array("val_indices", data=np.array([0], dtype=np.int64), chunks=(1,))


def test_eye_mask_dataset_honors_training_artifact_splits(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_training.zarr"
    _write_eye_training_zarr(zarr_path)

    cfg = EyeMaskDatasetConfig(
        zarr_path=zarr_path,
        split=DatasetSplit(train=0.5, val=0.5),
    )
    train_entries, val_entries, stats, _cache_info = mod._build_entries_for_dataset(
        dataset_name="eye_training",
        cfg=cfg,
        default_split=DatasetSplit(train=0.5, val=0.5),
        rng=np.random.default_rng(123),
    )

    assert [entry.roi_index for entry in train_entries] == [3, 1]
    assert [entry.roi_index for entry in val_entries] == [0]
    assert stats.total_rois == 4
