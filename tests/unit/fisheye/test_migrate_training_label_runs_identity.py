from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.migrate_training_label_runs_identity import migrate_training_label_runs_identity


def _make_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crops = root.create_group("crop_runs")
    source_crop = crops.create_group("crop_old")
    target_crop = crops.create_group("crop_new")
    for crop in (source_crop, target_crop):
        crop.attrs["roi_size"] = [2, 2]
        crop.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int64), overwrite=True)
        crop.create_array("detection_indices", data=np.array([10, 11, 12], dtype=np.int64), overwrite=True)
        crop.create_array("detection_source", data=np.array([1, 1, 1], dtype=np.int32), overwrite=True)
        crop.create_array("roi_coordinates_full", data=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32), overwrite=True)
    target_crop.attrs["source_crop_run"] = "crop_old"
    target_crop.attrs["roi_pixel_contract"] = {"name": "orange_mono_pynvvc_luma_uint8_v1"}

    keypoints_parent = root.create_group("keypoints_runs")
    keypoints_parent.attrs["latest"] = "keypoints_old"
    keypoints = keypoints_parent.create_group("keypoints_old")
    keypoints.attrs["source_crop_run"] = "crop_old"
    keypoints.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int64), overwrite=True)
    keypoints.create_array("detection_indices", data=np.array([10, 11, 12], dtype=np.int64), overwrite=True)
    keypoints_roi = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    keypoints_roi[1, 1, 0] = np.nan
    keypoints.create_array("keypoints_roi", data=keypoints_roi, overwrite=True)

    masks_parent = root.create_group("refined_subject_masks_runs")
    masks_parent.attrs["latest"] = "masks_old"
    masks = masks_parent.create_group("masks_old")
    masks.attrs["source_crop_run"] = "crop_old"
    masks.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int64), overwrite=True)
    masks.create_array("detection_indices", data=np.array([10, 11, 12], dtype=np.int64), overwrite=True)
    masks.create_array("masks_roi", data=np.ones((3, 2, 2), dtype=np.uint8), overwrite=True)
    return zarr_path


def test_migrate_training_label_runs_identity_copies_arrays_and_repoints_crop(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)

    report = migrate_training_label_runs_identity(
        zarr_path=zarr_path,
        target_crop_run="crop_new",
        families=["keypoints_runs", "refined_subject_masks_runs"],
        run_suffix="_new_pixels",
    )

    assert report["status"] == "ok"
    assert len(report["migrations"]) == 2
    root = zarr.open_group(str(zarr_path), mode="r")
    assert root["keypoints_runs"].attrs["latest"] == "keypoints_old"
    keypoints = root["keypoints_runs/keypoints_old_new_pixels"]
    assert keypoints.attrs["source_crop_run"] == "crop_new"
    assert keypoints.attrs["source_label_run"] == "keypoints_old"
    assert keypoints.attrs["label_coordinate_transform"] == "identity"
    assert keypoints.attrs["source_roi_pixel_contract_name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert np.array_equal(
        root["keypoints_runs/keypoints_old/keypoints_roi"][:],
        keypoints["keypoints_roi"][:],
        equal_nan=True,
    )
    masks = root["refined_subject_masks_runs/masks_old_new_pixels"]
    assert masks.attrs["source_crop_run"] == "crop_new"
    assert np.array_equal(
        root["refined_subject_masks_runs/masks_old/masks_roi"][:],
        masks["masks_roi"][:],
    )


def test_migrate_training_label_runs_identity_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)

    report = migrate_training_label_runs_identity(
        zarr_path=zarr_path,
        target_crop_run="crop_new",
        families=["keypoints_runs"],
        run_suffix="_preview",
        dry_run=True,
    )

    assert report["status"] == "dry_run"
    root = zarr.open_group(str(zarr_path), mode="r")
    assert "keypoints_old_preview" not in root["keypoints_runs"]
