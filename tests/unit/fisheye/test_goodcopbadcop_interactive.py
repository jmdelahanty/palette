from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceResult,
    ChaserDistanceWindow,
    write_chaser_distance_run,
)
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID
from fisheye.visualization.goodcopbadcop_interactive import (
    DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
    GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID,
    discover_goodcopbadcop_chaser_dashboard_options,
    load_goodcopbadcop_interactive_data,
    to_distance_timeseries_dataframe,
    to_position_dataframe,
    to_window_dataframe,
)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=np.asarray(values), chunks=np.asarray(values).shape, overwrite=True)


def _bytes_array(values: list[str], *, width: int = 48) -> np.ndarray:
    out = np.zeros((len(values), width), dtype=np.uint8)
    for row_idx, value in enumerate(values):
        encoded = value.encode("utf-8")[: width - 1]
        out[row_idx, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return out


def _make_archive_with_detection_occupancy(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "goodcopbadcop_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["recording_id"] = "test_GoodCopBadCop"
    analysis = root.create_group("analysis")
    parent = analysis.create_group("detection_occupancy_runs")
    parent.attrs["latest"] = "occupancy_1"
    parent.attrs["latest_complete"] = "occupancy_1"
    run = parent.create_group("occupancy_1")
    run.attrs.update(
        {
            "source_stimulus_epoch_run": "epochs_1",
            "source_detection_path": "refined_detect_runs/refined_1/instances",
        }
    )
    windows = run.create_group("windows")
    _write_array(windows, "label_bytes", _bytes_array(["pre_event", "training_event"]))
    _write_array(windows, "start_frame", np.asarray([0, 3], dtype=np.int64))
    _write_array(windows, "end_frame", np.asarray([2, 5], dtype=np.int64))
    _write_array(windows, "start_time_s", np.asarray([0.0, 0.3], dtype=np.float64))
    _write_array(windows, "end_time_s", np.asarray([0.3, 0.6], dtype=np.float64))
    heatmaps = run.create_group("heatmaps")
    _write_array(heatmaps, "counts", np.ones((2, 2, 2), dtype=np.float32))
    _write_array(heatmaps, "normalized", np.asarray([[[0.0, 1.0], [0.5, 0.2]], [[0.2, 0.3], [1.0, 0.1]]], dtype=np.float32))
    _write_array(heatmaps, "x_edges", np.asarray([0.0, 10.0, 20.0], dtype=np.float64))
    _write_array(heatmaps, "y_edges", np.asarray([0.0, 10.0, 20.0], dtype=np.float64))
    return zarr_path


def _make_chaser_result(zarr_path: Path) -> ChaserDistanceResult:
    n = 6
    chasers = np.asarray([0, 1], dtype=np.uint8)
    camera_frame_id = np.arange(n, dtype=np.int64)
    fish_xy = np.asarray(
        [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 3.0], [5.0, 3.0], [6.0, 4.0]],
        dtype=np.float32,
    )
    chaser_xy = np.zeros((n, 2, 2), dtype=np.float32)
    chaser_xy[:, 0, :] = np.asarray([0.0, 0.0], dtype=np.float32)
    chaser_xy[:, 1, :] = np.asarray([10.0, 0.0], dtype=np.float32)
    distance_px = np.linalg.norm(fish_xy[:, None, :] - chaser_xy, axis=2).astype(np.float32)
    distance_mm = (distance_px / 2.0).astype(np.float32)
    windows = (
        ChaserDistanceWindow(0, "pre_event", 0, 2, 0.0, 0.3, 0.3),
        ChaserDistanceWindow(1, "training_event", 3, 5, 0.3, 0.6, 0.3),
    )
    hist_counts = np.asarray(
        [
            [[1, 2, 0], [0, 1, 2]],
            [[2, 1, 0], [1, 1, 1]],
        ],
        dtype=np.int64,
    )
    hist_density = hist_counts.astype(np.float32)
    hist_density /= np.maximum(hist_density.sum(axis=2, keepdims=True), 1)
    return ChaserDistanceResult(
        zarr_path=str(zarr_path),
        recording_id="test_GoodCopBadCop",
        run_name="chaser_distance_1",
        source_detection_path="refined_detect_runs/refined_1/instances",
        source_detection_kind="refined",
        source_stimulus_run="stimulus_1",
        source_stimulus_path="analysis/stimulus_runs/stimulus_1",
        source_stimulus_epoch_run="epochs_1",
        source_stimulus_epoch_path="analysis/stimulus_epoch_runs/epochs_1",
        fps=10.0,
        total_frames=n,
        pixels_per_mm_projector=2.0,
        coordinate_frame="arena_relative_canvas_px",
        coordinate_origin="top_left_of_active_arena",
        arena_origin_in_canvas_xy=(0.0, 0.0),
        chaser_indices=chasers,
        camera_frame_id=camera_frame_id,
        stimulus_frame_num=camera_frame_id,
        timestamp_ns=np.arange(n, dtype=np.int64),
        stimulus_epoch_window_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        fish_centroid_img_xy=fish_xy,
        fish_centroid_arena_xy=fish_xy,
        chaser_arena_xy=chaser_xy,
        fish_valid=np.asarray([True, True, True, True, False, True], dtype=bool),
        chaser_valid=np.ones((n, 2), dtype=bool),
        distance_px=distance_px,
        distance_mm=distance_mm,
        nearest_chaser_index=np.argmin(distance_mm, axis=1).astype(np.int16),
        nearest_distance_mm=np.min(distance_mm, axis=1).astype(np.float32),
        windows=windows,
        epoch_valid_frame_count=np.asarray([[3, 3], [2, 2]], dtype=np.int64),
        epoch_mean_distance_mm=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        epoch_min_distance_mm=np.asarray([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        epoch_p05_distance_mm=np.asarray([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        epoch_p50_distance_mm=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        epoch_p95_distance_mm=np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        epoch_fraction_within_threshold=np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        threshold_mm=20.0,
        distribution_bin_width_mm=2.0,
        histogram_bin_edges_mm=np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
        histogram_bin_centers_mm=np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
        histogram_counts=hist_counts,
        histogram_density=hist_density,
    )


def test_chaser_distance_writer_adds_goodcopbadcop_interactive_spec(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    result = _make_chaser_result(zarr_path)

    run_path = write_chaser_distance_run(zarr_path, result, overwrite=True)

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root[run_path]
    visualizations = run["visualizations"]
    spec_group = visualizations[DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT]
    assert spec_group.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec_group.attrs["snapshot_artifact"] == "chaser_distance_timeseries_png"
    assert spec_group.attrs["renderer"] == "palette-goodcopbadcop-chaser-dashboard-v1"

    spec_bytes = np.asarray(spec_group["spec_json"][:], dtype=np.uint8).tobytes()
    spec = json.loads(spec_bytes.decode("utf-8"))
    assert spec["schema_id"] == GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID
    assert spec["source_paths"]["fish_centroid_arena_xy"].endswith("/positions/fish_centroid_arena_xy")
    assert spec["source_paths"]["distance_mm"].endswith("/distances/distance_mm")
    assert spec["source_paths"]["detection_occupancy_heatmap_normalized"].endswith(
        "analysis/detection_occupancy_runs/occupancy_1/heatmaps/normalized"
    )
    assert spec["source_runs"]["detection_occupancy"] == "occupancy_1"
    assert spec["static_artifacts"]["detection_occupancy"].endswith(
        "analysis/detection_occupancy_runs/occupancy_1/visualizations/detection_occupancy_overview_png"
    )

    manifest = run.attrs["visualizations"]
    assert manifest[DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT]["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID


def test_goodcopbadcop_interactive_loader_builds_plot_dataframes(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    result = _make_chaser_result(zarr_path)
    write_chaser_distance_run(zarr_path, result, overwrite=True)

    options = discover_goodcopbadcop_chaser_dashboard_options(zarr_path)
    assert [option.run_name for option in options] == ["chaser_distance_1"]

    data = load_goodcopbadcop_interactive_data(zarr_path, run_path=options[0].run_path)
    assert data.fps == 10.0
    assert data.distance_mm.shape == (6, 2)
    assert data.occupancy_normalized is not None
    assert data.occupancy_normalized.shape == (2, 2, 2)

    windows_df = to_window_dataframe(data)
    assert windows_df["label"].tolist() == ["pre_event", "training_event"]

    distance_df = to_distance_timeseries_dataframe(data)
    assert "distance_mm_chaser_0" in distance_df.columns
    assert "distance_mm_chaser_1" in distance_df.columns
    assert distance_df["time_s"].tolist() == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    position_df = to_position_dataframe(data)
    assert position_df["fish_valid"].tolist() == [True, True, True, True, False, True]
    assert position_df.loc[0, "unit"] == "arena_relative_canvas_px"
