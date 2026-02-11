import numpy as np
import zarr

from fisheye.tune.keypoint_review import _update_postprocess_summary
from fisheye.utils.patch_keypoints_from_crops import _update_keypoints_summary


def test_update_keypoints_summary_writes_heading_fields_and_drops_heading_valid(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_keypoint_heading_fields.zarr", mode="w")
    keypoints_runs = root.create_group("keypoints_runs")
    keypoints = keypoints_runs.create_group("keypoints_001")

    keypoints.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float64))
    keypoints.create_array("detection_success", data=np.array([True, True, False, True], dtype=bool))
    keypoints.create_array("frame_indices", data=np.array([0, 0, 1, 2], dtype=np.int32))
    keypoints.create_array("frame_counts", data=np.array([2, 1, 1], dtype=np.int32))
    keypoints.create_array("heading", data=np.array([10.0, np.nan, 30.0, 40.0], dtype=np.float64))
    keypoints.create_array("detection_source", data=np.array([0, 0, 0, 1], dtype=np.int8))
    keypoints.create_array("heading_valid", data=np.array([True, False, False, False], dtype=bool))

    _update_keypoints_summary(root, keypoints)

    assert "heading_valid" not in keypoints
    np.testing.assert_array_equal(
        keypoints["heading_finite"][:],
        np.array([True, False, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        keypoints["heading_usable"][:],
        np.array([True, False, False, False], dtype=bool),
    )


def test_update_postprocess_summary_reports_heading_finite_and_usable(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "test_keypoint_postprocess_heading_fields.zarr", mode="w")
    refined = root.create_group("refined")

    refined.create_array("keypoints_roi", data=np.zeros((3, 3, 2), dtype=np.float64))
    refined.create_array("refined_success", data=np.array([True, True, False], dtype=bool))
    refined.create_array("usable_keypoints", data=np.array([True, False, False], dtype=bool))
    refined.create_array("confidence_valid", data=np.array([True, False, False], dtype=bool))
    refined.create_array("geometry_valid", data=np.array([True, True, False], dtype=bool))
    refined.create_array("flip_corrected", data=np.array([False, True, False], dtype=bool))
    refined.create_array("heading_finite", data=np.array([True, True, False], dtype=bool))
    refined.create_array("heading_usable", data=np.array([True, False, False], dtype=bool))
    refined.create_array("source_success", data=np.array([True, True, False], dtype=bool))

    stats = _update_postprocess_summary(refined, print_summary=False)

    assert stats["heading_finite"] == 2
    assert stats["heading_usable"] == 1
    assert "heading_valid" not in stats
