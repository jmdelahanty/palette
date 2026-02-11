import numpy as np
import zarr

from fisheye.utils.backfill_keypoint_heading_fields import _backfill_heading_columns


def test_backfill_heading_columns_raw_keypoints_writes_and_drops_legacy(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_heading_backfill_raw.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("heading", data=np.array([10.0, np.nan, 30.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True, True, False], dtype=bool))
    run.create_array("detection_source", data=np.array([0, 0, 1], dtype=np.int8))
    run.create_array("heading_valid", data=np.array([True, False, False], dtype=bool))

    result = _backfill_heading_columns(
        run,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    assert "heading_valid" not in run
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True, False, True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True, False, False], dtype=bool))


def test_backfill_heading_columns_refined_defaults_detection_source_to_real(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_heading_backfill_refined.zarr", mode="w")
    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.create_array("heading", data=np.array([1.0, np.nan, 2.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, False, True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True, False, True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True, False, True], dtype=bool))


def test_backfill_heading_columns_skips_when_fields_present_and_no_legacy(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_heading_backfill_skip.zarr", mode="w")
    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.create_array("heading", data=np.array([1.0, 2.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, True], dtype=bool))
    run.create_array("heading_finite", data=np.array([False, False], dtype=bool))
    run.create_array("heading_usable", data=np.array([False, False], dtype=bool))

    result = _backfill_heading_columns(
        run,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "skipped_existing"
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([False, False], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([False, False], dtype=bool))


def test_backfill_heading_columns_detects_shape_mismatch(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_heading_backfill_shape.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("heading", data=np.array([1.0, 2.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "shape_mismatch"
