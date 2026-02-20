"""Tests for merged eye-mask-training Zarr validation and export scaffold."""

from pathlib import Path
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.export_eye_mask_training_zarr import (
    export_merged_eye_mask_training_zarr,
    validate_merged_eye_mask_training_zarr,
)


def _write_valid_merged_eye_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    root.attrs["training_task"] = "eye_masks"
    root.attrs["training_export"] = {"input_format": "gray", "label_mode": "lr"}

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "merged_export_smoke"
    crop = crop_parent.create_group("merged_export_smoke")
    crop.create_array(
        "roi_images",
        data=np.zeros((4, 16, 16), dtype=np.uint8),
        chunks=(2, 16, 16),
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "crop_bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "frame_indices",
        data=np.arange(4, dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "detection_source",
        data=np.array([0, 1, 0, 0], dtype=np.int8),
        chunks=(4,),
    )

    eye_parent = root.create_group("eye_masks_runs")
    eye_parent.attrs["latest"] = "merged_export_smoke"
    eye = eye_parent.create_group("merged_export_smoke")

    masks = np.zeros((4, 2, 16, 16), dtype=np.uint8)
    masks[0, 0, 4:8, 4:8] = 1
    masks[0, 1, 4:8, 9:13] = 1
    masks[1, 0, 4:8, 4:8] = 1
    masks[1, 1, 4:8, 9:13] = 1
    masks[2, 0, 4:8, 4:8] = 1
    masks[2, 1, 4:8, 9:13] = 1
    masks[3, 0, 4:8, 4:8] = 1
    masks[3, 1, 4:8, 9:13] = 1
    eye.create_array("masks_roi", data=masks, chunks=(2, 2, 16, 16))

    ellipse_params = np.full((4, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.array(
        [
            [True, True],
            [True, True],
            [True, True],
            [False, False],
        ],
        dtype=np.bool_,
    )
    ellipse_params[0, 0] = np.array([6.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[0, 1] = np.array([11.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[1, 0] = np.array([6.0, 6.0, 9.0, 6.0, 5.0], dtype=np.float32)
    ellipse_params[1, 1] = np.array([11.0, 6.0, 9.0, 6.0, 5.0], dtype=np.float32)
    ellipse_params[2, 0] = np.array([6.0, 6.0, 7.5, 5.5, 10.0], dtype=np.float32)
    ellipse_params[2, 1] = np.array([11.0, 6.0, 7.5, 5.5, 10.0], dtype=np.float32)
    eye.create_array("ellipse_params", data=ellipse_params, chunks=(4, 2, 5))
    eye.create_array("ellipse_success", data=ellipse_success, chunks=(4, 2))
    eye.create_array("eye_separation", data=np.array([5.0, 5.0, 5.0, np.nan], dtype=np.float32), chunks=(4,))
    eye.create_array("frame_indices", data=np.arange(4, dtype=np.int64), chunks=(4,))
    eye.create_array("detection_source", data=np.array([0, 1, 0, 0], dtype=np.int8), chunks=(4,))
    eye.create_array(
        "reason",
        data=np.asarray(["clean", "clean", "manual_correction|clean", "incomplete"], dtype=object),
        chunks=(4,),
    )

    splits = root.create_group("splits")
    splits.create_array("train_indices", data=np.array([0, 1], dtype=np.int64), chunks=(2,))
    splits.create_array("val_indices", data=np.array([2], dtype=np.int64), chunks=(1,))
    splits.create_array("test_indices", data=np.array([3], dtype=np.int64), chunks=(1,))

    source = root.create_group("source_index")
    source.create_array(
        "source_dataset_idx",
        data=np.array([0, 0, 0, 0], dtype=np.int32),
        chunks=(4,),
    )
    source.create_array(
        "source_frame_idx",
        data=np.array([10, 11, 12, 13], dtype=np.int64),
        chunks=(4,),
    )
    source.create_array(
        "source_roi_idx",
        data=np.arange(4, dtype=np.int64),
        chunks=(4,),
    )
    source.create_array(
        "source_dataset_id",
        data=np.asarray(["dataset_a"], dtype=object),
        chunks=(1,),
    )
    source.create_array(
        "source_zarr_path",
        data=np.asarray(["/tmp/source_a.zarr"], dtype=object),
        chunks=(1,),
    )


def _write_source_eye_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.create_array(
        "roi_images",
        data=np.zeros((4, 16, 16), dtype=np.uint8),
        chunks=(2, 16, 16),
    )
    crop.create_array(
        "bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "crop_bbox_norm_coords",
        data=np.zeros((4, 4), dtype=np.float32),
        chunks=(4, 4),
    )
    crop.create_array(
        "frame_indices",
        data=np.array([100, 101, 102, 103], dtype=np.int64),
        chunks=(4,),
    )
    crop.create_array(
        "detection_source",
        data=np.array([0, 1, 0, 0], dtype=np.int8),
        chunks=(4,),
    )

    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"
    refined = refined_parent.create_group("refined_eye_masks_001")
    refined.attrs["source_crop_run"] = "crop_001"
    refined.attrs["method"] = "traditional_eye_segmentation"
    refined.attrs["eye_labels"] = ["eye_left", "eye_right"]
    refined.attrs["source_keypoints_run"] = "refined_keypoints_001"
    refined.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    refined.create_array(
        "masks_roi",
        data=np.zeros((4, 2, 16, 16), dtype=np.uint8),
        chunks=(2, 2, 16, 16),
    )
    ellipse_params = np.full((4, 2, 5), np.nan, dtype=np.float32)
    ellipse_params[0, 0] = np.array([6.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[0, 1] = np.array([11.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[1, 0] = np.array([6.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    ellipse_params[1, 1] = np.array([11.0, 6.0, 8.0, 6.0, 0.0], dtype=np.float32)
    refined.create_array("ellipse_params", data=ellipse_params, chunks=(4, 2, 5))
    refined.create_array(
        "ellipse_success",
        data=np.array([[True, True], [True, True], [False, False], [False, False]], dtype=np.bool_),
        chunks=(4, 2),
    )
    refined.create_array("eye_separation", data=np.array([5.0, 5.0, np.nan, np.nan], dtype=np.float32), chunks=(4,))
    metrics = refined.create_group("metrics")
    metrics.create_array(
        "reason",
        data=np.asarray(["clean", "clean", "incomplete", "incomplete"], dtype=object),
        chunks=(4,),
    )


def test_validate_merged_eye_mask_training_zarr_passes(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_eye_ok.zarr"
    _write_valid_merged_eye_zarr(zarr_path)

    summary = validate_merged_eye_mask_training_zarr(
        zarr_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_mode="lr",
    )

    assert summary["total_samples"] == 4
    assert summary["channels"] == 2
    assert summary["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert summary["source_count"] == 1


def test_validate_merged_eye_mask_training_zarr_rejects_invalid_frame_indices(tmp_path: Path) -> None:
    zarr_path = tmp_path / "merged_eye_bad.zarr"
    _write_valid_merged_eye_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    latest = root["crop_runs"].attrs["latest"]
    root[f"crop_runs/{latest}/frame_indices"][:] = np.array([1, 2, 3, 4], dtype=np.int64)

    with pytest.raises(ValueError, match="frame_indices"):
        validate_merged_eye_mask_training_zarr(
            zarr_path,
            expected_input_format="gray",
            expected_total_samples=4,
            expected_label_mode="lr",
        )


def test_export_merged_eye_mask_training_zarr_then_validate(tmp_path: Path) -> None:
    source_path = tmp_path / "source_training.zarr"
    out_path = tmp_path / "merged_eye_export.zarr"
    _write_source_eye_zarr(source_path)

    summary = export_merged_eye_mask_training_zarr(
        source_path,
        out_path,
        eye_stage="refined_eye_masks_runs",
        eye_run="refined_eye_masks_001",
        overwrite=True,
    )

    assert summary["total_samples"] == 4
    assert summary["channels"] == 2

    recheck = validate_merged_eye_mask_training_zarr(
        out_path,
        expected_input_format="gray",
        expected_total_samples=4,
        expected_label_mode="lr",
    )
    assert recheck["total_samples"] == 4
