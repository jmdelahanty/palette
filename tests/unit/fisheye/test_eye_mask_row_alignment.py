from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.refinement.refine_eye_masks import _validate_input_row_alignment as validate_refine_alignment
from fisheye.segmentation.eye_segmentation import _validate_input_row_alignment as validate_traditional_alignment
from fisheye.segmentation.eye_segmentation_yolo import _validate_input_row_alignment as validate_yolo_alignment
from fisheye.segmentation.infer_unet_eye_masks import _validate_input_row_alignment as validate_unet_alignment
from fisheye.shared.row_alignment import assert_row_alignment


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def test_assert_row_alignment_reports_named_mismatch() -> None:
    with pytest.raises(ValueError, match="demo input row-alignment check failed"):
        assert_row_alignment(
            3,
            (
                ("good", np.zeros((3, 2), dtype=np.float32)),
                ("bad", np.zeros((2, 2), dtype=np.float32)),
            ),
            stage="demo input",
        )


def test_traditional_alignment_rejects_keypoint_row_mismatch() -> None:
    crop_group = _FakeGroup(
        {
            "roi_images": np.zeros((3, 10, 10), dtype=np.uint8),
            "frame_indices": np.zeros((3,), dtype=np.int32),
            "detection_indices": np.zeros((3,), dtype=np.int32),
            "detection_source": np.zeros((3,), dtype=np.int8),
        }
    )
    kp_group = _FakeGroup(
        {
            "keypoints_roi": np.zeros((2, 3, 2), dtype=np.float32),
            "detection_success": np.zeros((2,), dtype=bool),
        }
    )
    with pytest.raises(ValueError, match="keypoints_runs/kp_001/keypoints_roi=2"):
        validate_traditional_alignment(
            crop_group=crop_group,
            crop_run="crop_001",
            kp_group=kp_group,
            keypoint_group_name="keypoints_runs",
            keypoint_run="kp_001",
            success_dataset_name="detection_success",
            total_rois=3,
        )


def test_yolo_alignment_rejects_detection_indices_mismatch() -> None:
    crop_group = _FakeGroup(
        {
            "roi_images": np.zeros((3, 10, 10), dtype=np.uint8),
            "frame_indices": np.zeros((3,), dtype=np.int32),
            "detection_indices": np.zeros((4,), dtype=np.int32),
            "detection_source": np.zeros((3,), dtype=np.int8),
        }
    )
    with pytest.raises(ValueError, match="crop_runs/crop_001/detection_indices=4"):
        validate_yolo_alignment(
            crop_group=crop_group,
            crop_run_name="crop_001",
            total_rois=3,
        )


def test_unet_alignment_rejects_detection_source_mismatch() -> None:
    crop_group = _FakeGroup(
        {
            "frame_indices": np.zeros((3,), dtype=np.int32),
            "detection_indices": np.zeros((3,), dtype=np.int32),
            "detection_source": np.zeros((2,), dtype=np.int8),
        }
    )
    with pytest.raises(ValueError, match="crop_runs/crop_001/detection_source=2"):
        validate_unet_alignment(
            crop_group=crop_group,
            crop_run="crop_001",
            total_rois=3,
        )


def test_unet_alignment_allows_geometry_only_crop_group() -> None:
    crop_group = _FakeGroup(
        {
            "frame_indices": np.zeros((2,), dtype=np.int32),
            "detection_indices": np.zeros((2,), dtype=np.int32),
            "detection_source": np.zeros((2,), dtype=np.int8),
        }
    )

    validate_unet_alignment(
        crop_group=crop_group,
        crop_run="crop_geometry",
        total_rois=2,
    )


def test_refine_alignment_rejects_probability_row_mismatch() -> None:
    src_run = _FakeGroup(
        {
            "masks_roi": np.zeros((3, 2, 10, 10), dtype=np.uint8),
            "frame_indices": np.zeros((3,), dtype=np.int32),
            "detection_indices": np.zeros((3,), dtype=np.int32),
            "detection_source": np.zeros((3,), dtype=np.int8),
        }
    )
    kp_group = _FakeGroup(
        {
            "keypoints_roi": np.zeros((3, 3, 2), dtype=np.float32),
            "heading": np.zeros((3,), dtype=np.float32),
            "detection_success": np.ones((3,), dtype=bool),
        }
    )
    with pytest.raises(ValueError, match="loaded_mask_bundle/probs=2"):
        validate_refine_alignment(
            total_rois=3,
            src_run=src_run,
            src_run_name="eye_001",
            kp_group=kp_group,
            keypoint_group_name="refined_keypoints_runs",
            keypoint_run_name="kp_001",
            success_dataset_name="detection_success",
            binary_data=np.zeros((3, 2, 10, 10), dtype=np.uint8),
            probs_data=np.zeros((2, 2, 10, 10), dtype=np.float32),
        )
