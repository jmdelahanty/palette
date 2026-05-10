from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.eye_angle_io import (
    EyeAngleIOError,
    discover_eye_angle_run_options,
    first_array_length,
    load_eye_angle_run_tables,
    load_eye_gaze_frame_series,
    optional_1d_array,
)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _make_eye_angle_archive(tmp_path: Path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "eye_angle.zarr"), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("eye_angle_runs")
    parent.attrs["latest"] = "eye_angle_1"
    run = parent.create_group("eye_angle_1")
    run.attrs.update(
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 5,
            "preferred_angle_family": "gaze",
            "preferred_eye_axis": "ellipse_major",
            "row_axis": "keypoint_detection_rows",
            "fps": 100.0,
        }
    )
    angles = run.create_group("angles")
    roi = angles.create_group("roi")
    frame = angles.create_group("frame")
    qa = run.create_group("qa")
    qa_roi = qa.create_group("roi")
    qa_frame = qa.create_group("frame")
    support = run.create_group("support")

    _write_array(roi, "left_eye_angle_deg", np.asarray([10.0, 11.0, 12.0], dtype=np.float32))
    _write_array(roi, "left_gaze_signed_deg", np.asarray([-80.0, -79.0, -78.0], dtype=np.float32))
    _write_array(frame, "left_gaze_deg", np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    _write_array(frame, "right_gaze_deg", np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
    _write_array(frame, "vergence_gaze_deg", np.asarray([9.0, 10.0, 11.0, 12.0], dtype=np.float32))
    _write_array(frame, "vergence_gaze_signed_deg", np.asarray([-1.0, -2.0, -3.0, -4.0], dtype=np.float32))
    _write_array(qa_roi, "valid_frame", np.asarray([True, False, True], dtype=bool))
    _write_array(qa_frame, "valid_frame", np.asarray([True, False, True, True], dtype=bool))
    _write_array(support, "time_seconds", np.asarray([0.0, 0.01, 0.02], dtype=np.float64))
    _write_array(support, "frame_indices", np.asarray([0, 1, 2], dtype=np.int64))
    _write_array(support, "frame_time_seconds", np.asarray([0.0, 0.01, 0.02, 0.03], dtype=np.float64))
    return root


def test_discover_eye_angle_run_options_uses_latest_and_shape_metadata(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    options = discover_eye_angle_run_options(root)

    assert len(options) == 1
    option = options[0]
    assert option.run_name == "eye_angle_1"
    assert option.run_path == "analysis/eye_angle_runs/eye_angle_1"
    assert option.schema_version == 5
    assert option.preferred_angle_family == "gaze"
    assert option.preferred_eye_axis == "ellipse_major"
    assert option.n_rows == 3
    assert option.is_latest is True
    assert "latest" in option.label


def test_load_eye_angle_run_tables_reads_logical_groups(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    tables = load_eye_angle_run_tables(root, run_name="analysis/eye_angle_runs/eye_angle_1")

    assert tables.run_name == "eye_angle_1"
    assert tables.run_path == "analysis/eye_angle_runs/eye_angle_1"
    assert tables.schema_version == 5
    assert tables.row_axis == "keypoint_detection_rows"
    assert first_array_length(tables.roi) == 3
    assert optional_1d_array(tables.support, "frame_time_seconds", length=4) is not None
    np.testing.assert_allclose(tables.roi["left_eye_angle_deg"], [10.0, 11.0, 12.0])
    assert tables.qa_frame["valid_frame"].tolist() == [True, False, True, True]


def test_load_eye_gaze_frame_series_aligns_frames_and_validity(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)
    frames = np.asarray([0, 2, 3], dtype=np.int64)

    series, source_refs = load_eye_gaze_frame_series(
        root,
        eye_angle_run="latest",
        eye_angle_family="gaze",
        frames=frames,
        allowed_families=("gaze",),
    )

    np.testing.assert_allclose(series["left_gaze_deg"], [1.0, 3.0, 4.0])
    np.testing.assert_allclose(series["right_gaze_deg"], [5.0, 7.0, 8.0])
    np.testing.assert_allclose(series["vergence_gaze_signed_deg"], [-1.0, -3.0, -4.0])
    assert series["valid_frame"].tolist() == [True, True, True]
    assert source_refs["source_eye_angle_run"] == "eye_angle_1"
    assert source_refs["source_eye_angle_schema_version"] == 5
    assert source_refs["source_eye_angle_arrays"]["left_gaze_deg"].endswith(
        "/angles/frame/left_gaze_deg"
    )


def test_load_eye_gaze_frame_series_rejects_unsupported_family(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    with pytest.raises(EyeAngleIOError, match="Unsupported eye_angle_family"):
        load_eye_gaze_frame_series(
            root,
            eye_angle_run="latest",
            eye_angle_family="eye_frame",
            frames=np.asarray([0], dtype=np.int64),
            allowed_families=("gaze",),
        )


def test_load_eye_gaze_frame_series_checks_frame_bounds(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    with pytest.raises(EyeAngleIOError, match="cannot cover requested frame"):
        load_eye_gaze_frame_series(
            root,
            eye_angle_run="latest",
            eye_angle_family="gaze",
            frames=np.asarray([4], dtype=np.int64),
            allowed_families=("gaze",),
        )
