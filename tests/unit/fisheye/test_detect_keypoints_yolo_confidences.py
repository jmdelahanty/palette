from __future__ import annotations

import numpy as np
import torch
import zarr

from fisheye.detection.detect_keypoints_yolo import (
    _create_output_arrays,
    _extract_keypoint_confidences,
    _extract_pose_bbox_xyxy_roi,
)


class _KeypointsWithConf:
    def __init__(self, conf: torch.Tensor | None) -> None:
        self.conf = conf


class _BoxesWithXyxy:
    def __init__(self, xyxy: torch.Tensor | None) -> None:
        self.xyxy = xyxy


def test_extract_keypoint_confidences_returns_values_when_present() -> None:
    keypoints = _KeypointsWithConf(
        torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.6, 0.7, 0.8],
            ],
            dtype=torch.float32,
        )
    )

    actual = _extract_keypoint_confidences(keypoints, 1, n_keypoints=3)

    np.testing.assert_allclose(actual, np.array([0.6, 0.7, 0.8], dtype=np.float64))


def test_extract_keypoint_confidences_returns_nan_when_missing() -> None:
    keypoints = _KeypointsWithConf(None)

    actual = _extract_keypoint_confidences(keypoints, 0, n_keypoints=3)

    assert actual.shape == (3,)
    assert np.isnan(actual).all()


def test_create_output_arrays_includes_keypoint_confidences(tmp_path) -> None:
    root = zarr.open_group(store=str(tmp_path / "test.zarr"), mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")

    arrays = _create_output_arrays(run, total_rois=10, chunk_hint=4, n_keypoints=5)

    assert "keypoint_confidences" in arrays
    assert arrays["keypoints_roi"].shape == (10, 5, 2)
    assert arrays["keypoint_confidences"].shape == (10, 5)
    assert arrays["keypoint_confidences"].dtype.name == "float64"
    assert "pose_bbox_xyxy_roi" in arrays
    assert arrays["pose_bbox_xyxy_roi"].shape == (10, 4)
    assert arrays["pose_bbox_xyxy_roi"].dtype.name == "float32"


def test_extract_pose_bbox_xyxy_roi_clips_to_roi_bounds() -> None:
    boxes = _BoxesWithXyxy(
        torch.tensor(
            [
                [-2.0, 1.5, 8.2, 12.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=torch.float32,
        )
    )

    actual = _extract_pose_bbox_xyxy_roi(boxes, 0, roi_height=6, roi_width=8)

    np.testing.assert_allclose(actual, np.array([0.0, 1.5, 7.0, 5.0], dtype=np.float32))


def test_extract_pose_bbox_xyxy_roi_returns_nan_when_missing() -> None:
    boxes = _BoxesWithXyxy(None)

    actual = _extract_pose_bbox_xyxy_roi(boxes, 0, roi_height=6, roi_width=8)

    assert actual.shape == (4,)
    assert np.isnan(actual).all()
