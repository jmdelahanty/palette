from __future__ import annotations

import numpy as np

from fisheye.shared import refined_eye_mask_metrics as mod


class _FakeGroup(dict):
    def create_array(
        self,
        name: str,
        *,
        data=None,
        shape=None,
        dtype=None,
        fill_value=0,
        **_kwargs,
    ):
        if data is None:
            arr = np.full(shape, fill_value, dtype=dtype)
        else:
            arr = np.asarray(data, dtype=dtype)
        self[name] = arr
        return arr


def test_compute_refined_eye_roi_metrics_known_answer() -> None:
    left = np.zeros((8, 8), dtype=np.uint8)
    right = np.zeros((8, 8), dtype=np.uint8)
    left[1:3, 1:3] = 1
    right[5:7, 5:7] = 1
    centroids = np.array([[1.5, 1.5], [5.5, 5.5]], dtype=np.float32)
    eye_left = np.array([1.5, 1.5], dtype=np.float32)
    eye_right = np.array([6.5, 5.5], dtype=np.float32)
    ellipse_params = np.array(
        [
            [1.5, 1.5, 4.0, 2.0, 0.0],
            [5.5, 5.5, 6.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    metrics = mod.compute_refined_eye_roi_metrics(
        left,
        right,
        left,
        right,
        left | right,
        centroids,
        eye_left,
        eye_right,
        True,
        float(np.linalg.norm(centroids[1] - centroids[0])),
        ellipse_params,
        (None, None),
        None,
    )

    np.testing.assert_array_equal(metrics["refined_areas"], np.array([4.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(metrics["source_areas"], np.array([4.0, 4.0], dtype=np.float32))
    assert metrics["refined_union_area"] == 8.0
    assert metrics["source_union_area"] == 8.0
    assert metrics["axis_ratio"].tolist() == [0.5, 0.5]
    assert metrics["keypoint_separation"] == float(np.linalg.norm(eye_right - eye_left))


def test_write_refined_eye_contours_from_masks_builds_packed_arrays() -> None:
    run_group = _FakeGroup()

    masks = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:5, 1:5] = 1
    masks[0, 1, 2:6, 2:6] = 1
    run_group.create_array("masks_roi", data=masks)

    mod.write_refined_eye_contours_from_masks(run_group, total_rois=2, chunk_rois=1)

    assert np.asarray(run_group["contour_left_ptr"][:], dtype=np.int64).tolist() == [0, -1]
    assert np.asarray(run_group["contour_right_ptr"][:], dtype=np.int64).tolist() == [0, -1]
    assert int(run_group["contour_left_len"][0]) > 0
    assert int(run_group["contour_right_len"][0]) > 0
    assert tuple(run_group["contours_left"].shape[1:]) == (2,)
    assert tuple(run_group["contours_right"].shape[1:]) == (2,)
