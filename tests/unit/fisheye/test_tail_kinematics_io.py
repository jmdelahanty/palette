from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis.tail_kinematics_io import (
    TailKinematicsIOError,
    catalog_tail_kinematics_run,
    discover_tail_kinematics_run_options,
    load_tail_kinematics_window,
)


def _array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _fixture(tmp_path, *, fps: float | None = 3.0) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "tail.zarr"), mode="w")
    if fps is not None:
        root.attrs["fps"] = fps

    tail_parent = root.require_group("analysis/tail_kinematics_runs")
    tail_parent.attrs["latest_complete"] = "tail_run"
    tail = tail_parent.require_group("tail_run")
    tail.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "schema_version": 2,
            "method": "body_frame_spline_tangent",
            "source_subject_shape_run": "analysis/subject_shape_runs/shape_run",
        }
    )
    frames = np.arange(0, 30, 3, dtype=np.int64)
    angles = np.arange(frames.size * 10, dtype=np.float32).reshape(frames.size, 10)
    _array(tail, "frame_index", frames)
    _array(tail, "valid", np.asarray([True, True, True, False, True] * 2))
    _array(tail, "tail_angle_deg", angles)
    _array(tail, "tail_angle_sample_s", np.linspace(0.05, 0.95, 10, dtype=np.float32))
    _array(tail, "tail_tip_angle_deg", angles[:, -1])
    _array(
        tail,
        "tail_tip_lateral_deflection_px",
        np.linspace(-2, 2, frames.size, dtype=np.float32),
    )
    _array(tail, "tail_angle_rms_deg", np.sqrt(np.mean(angles**2, axis=1)))

    shape = root.require_group("analysis/subject_shape_runs/shape_run")
    shape.attrs["palette_run_completion_status"] = "complete"
    row_index = shape.require_group("row_index")
    _array(row_index, "frame_indices", frames.copy())
    body = shape.require_group("components/subject_body")
    curvature = (
        np.arange(frames.size * 32, dtype=np.float32).reshape(frames.size, 32) / 100.0
    )
    _array(body, "tail_curvature_px_inv", curvature)
    _array(body, "tail_sample_s", np.linspace(0, 1, 32, dtype=np.float32))
    _array(body, "tail_sample_valid", np.asarray([True] * frames.size))
    return root


def test_discovers_catalogs_and_bounds_tail_run(tmp_path) -> None:
    root = _fixture(tmp_path)

    options = discover_tail_kinematics_run_options(root)
    assert [option.run_name for option in options] == ["tail_run"]
    assert options[0].is_latest
    assert options[0].sample_count == 10

    catalog = catalog_tail_kinematics_run(root)
    assert catalog.fps == 3.0
    assert catalog.fps_source == "root.attrs.fps"
    assert catalog.time_start_s == 0.0
    assert catalog.time_stop_s == 9.0
    assert catalog.source_shape_run_name == "shape_run"
    assert catalog.source_shape_run_path == "analysis/subject_shape_runs/shape_run"
    assert catalog.source_curvature_sample_count == 32

    window = load_tail_kinematics_window(
        root,
        start_s=2.0,
        stop_s=5.0,
        scalar_series=("tail_tip_angle_deg", "tail_tip_lateral_deflection_px"),
    )
    np.testing.assert_array_equal(window.frame_indices, [6, 9, 12, 15])
    np.testing.assert_allclose(window.time_seconds, [2, 3, 4, 5])
    assert window.angle_deg.shape == (4, 10)
    assert window.dense_curvature_px_inv.shape == (4, 32)
    assert np.isnan(window.angle_deg[1]).all()
    assert np.isnan(window.scalar_series["tail_tip_angle_deg"][1])
    assert window.source_paths["dense_curvature"].endswith("tail_curvature_px_inv")


def test_refuses_oversized_projection(tmp_path) -> None:
    root = _fixture(tmp_path)
    with pytest.raises(TailKinematicsIOError, match="viewer limit"):
        load_tail_kinematics_window(root, start_s=0, stop_s=9, max_rows=4)


def test_fails_closed_when_subject_shape_rows_are_misaligned(tmp_path) -> None:
    root = _fixture(tmp_path)
    root["analysis/subject_shape_runs/shape_run/row_index/frame_indices"][3] = 10
    with pytest.raises(TailKinematicsIOError, match="lineage does not align"):
        load_tail_kinematics_window(root, start_s=2, stop_s=5)


def test_requires_fps_for_time_projection(tmp_path) -> None:
    root = _fixture(tmp_path, fps=None)
    with pytest.raises(TailKinematicsIOError, match="requires positive recording fps"):
        load_tail_kinematics_window(root, start_s=0, stop_s=1)
