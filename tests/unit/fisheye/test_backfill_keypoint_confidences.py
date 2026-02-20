import numpy as np
import zarr

from fisheye.utils.backfill_keypoint_confidences import _backfill_run_group


def test_backfill_keypoint_confidences_writes_missing_array(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_conf_missing.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("detection_success", data=np.array([True, False, True], dtype=bool))
    run.create_array("keypoints_roi", shape=(3, 3, 2), chunks=(2, 3, 2), dtype="f8", fill_value=np.nan)

    result = _backfill_run_group(
        run,
        confidence_value=0.95,
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    assert "keypoint_confidences" in run
    got = np.asarray(run["keypoint_confidences"][:], dtype=np.float64)
    expected = np.array(
        [
            [0.95, 0.95, 0.95],
            [np.nan, np.nan, np.nan],
            [0.95, 0.95, 0.95],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected, equal_nan=True)
    assert run.attrs.get("keypoint_confidence_labels") == ["swim_bladder", "eye_left", "eye_right"]


def test_backfill_keypoint_confidences_skips_existing_when_not_overwriting(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_conf_skip.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("detection_success", data=np.array([True, True], dtype=bool))
    run.create_array(
        "keypoint_confidences",
        data=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
    )

    result = _backfill_run_group(
        run,
        confidence_value=0.95,
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "skipped_existing"
    np.testing.assert_allclose(
        np.asarray(run["keypoint_confidences"][:], dtype=np.float64),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
    )


def test_backfill_keypoint_confidences_overwrites_existing_when_requested(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_conf_overwrite.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("detection_success", data=np.array([True, False], dtype=bool))
    run.create_array(
        "keypoint_confidences",
        data=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
    )

    result = _backfill_run_group(
        run,
        confidence_value=0.95,
        overwrite_existing=True,
        apply=True,
    )

    assert result.status == "ok"
    got = np.asarray(run["keypoint_confidences"][:], dtype=np.float64)
    expected = np.array([[0.95, 0.95, 0.95], [np.nan, np.nan, np.nan]], dtype=np.float64)
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_backfill_keypoint_confidences_requires_detection_success(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_conf_no_success.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("heading", data=np.array([1.0, 2.0], dtype=np.float64))

    result = _backfill_run_group(
        run,
        confidence_value=0.95,
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "no_success"
