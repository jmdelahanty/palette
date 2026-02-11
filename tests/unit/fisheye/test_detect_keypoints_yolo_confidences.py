from __future__ import annotations

import numpy as np
import torch
import zarr

from fisheye.detection.detect_keypoints_yolo import (
    _create_output_arrays,
    _extract_keypoint_confidences,
)


class _KeypointsWithConf:
    def __init__(self, conf: torch.Tensor | None) -> None:
        self.conf = conf


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

    arrays = _create_output_arrays(run, total_rois=10, chunk_hint=4)

    assert "keypoint_confidences" in arrays
    assert arrays["keypoint_confidences"].shape == (10, 3)
    assert arrays["keypoint_confidences"].dtype.name == "float64"
