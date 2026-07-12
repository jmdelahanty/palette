from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceResult,
    ChaserDistanceWindow,
    EPOCH_DISTRIBUTION_PNG_ARTIFACT_NAME,
    EPOCH_DISTRIBUTION_VISUALIZATION_CONTRACT_ID,
    TIMESERIES_PNG_ARTIFACT_NAME,
    TIMESERIES_VISUALIZATION_CONTRACT_ID,
    write_chaser_distance_run,
)
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID
from fisheye.visualization.goodcopbadcop_interactive import (
    CHASER_DASHBOARD_RENDERER,
    CHASER_DASHBOARD_SPEC_SCHEMA_ID,
    DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
    discover_goodcopbadcop_chaser_dashboard_options,
    discover_chaser_dashboard_options,
    load_chaser_dashboard_data,
    load_goodcopbadcop_interactive_data,
    to_distance_timeseries_dataframe,
    to_position_dataframe,
    to_spatial_occupancy_dataframe,
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
    calibration = analysis.create_group("calibration")
    _write_array(calibration, "homography_matrix", np.eye(3, dtype=np.float64))
    stimulus_parent = analysis.create_group("stimulus_runs")
    stimulus = stimulus_parent.create_group("stimulus_1")
    stimulus.attrs["protocol_json"] = json.dumps(
        {
            "steps": [
                {
                    "parameters": {
                        "chasers": [
                            {
                                "color_r": 1.0,
                                "color_g": 0.0,
                                "color_b": 0.0,
                                "color_a": 1.0,
                                "enable_chase": True,
                                "enable_random_movement": False,
                            },
                            {
                                "color_r": 0.0,
                                "color_g": 0.0,
                                "color_b": 1.0,
                                "color_a": 1.0,
                                "enable_chase": False,
                                "enable_random_movement": False,
                            },
                        ]
                    }
                }
            ]
        }
    )
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
    _write_array(windows, "label_bytes", _bytes_array(["pre_event", "training_event", "post_event"]))
    _write_array(windows, "start_frame", np.asarray([0, 3, 6], dtype=np.int64))
    _write_array(windows, "end_frame", np.asarray([2, 5, 8], dtype=np.int64))
    _write_array(windows, "start_time_s", np.asarray([0.0, 0.3, 0.6], dtype=np.float64))
    _write_array(windows, "end_time_s", np.asarray([0.3, 0.6, 0.9], dtype=np.float64))
    heatmaps = run.create_group("heatmaps")
    _write_array(heatmaps, "counts", np.ones((3, 2, 2), dtype=np.float32))
    _write_array(
        heatmaps,
        "normalized",
        np.asarray(
            [
                [[0.0, 1.0], [0.5, 0.2]],
                [[0.2, 0.3], [1.0, 0.1]],
                [[0.1, 0.4], [0.6, 0.9]],
            ],
            dtype=np.float32,
        ),
    )
    _write_array(heatmaps, "x_edges", np.asarray([0.0, 10.0, 20.0], dtype=np.float64))
    _write_array(heatmaps, "y_edges", np.asarray([0.0, 10.0, 20.0], dtype=np.float64))
    spatial = run.create_group("spatial_occupancy")
    quadrants = spatial.create_group("image_quadrants_v1")
    quadrants.attrs.update(
        {
            "schema_id": "palette.spatial_occupancy_zones.v1",
            "zone_set_source": "predefined_spec:quadrants.v1",
            "coordinate_frame": "source_image_px",
            "coordinate_origin": "top_left",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
        }
    )
    zone_spec = quadrants.create_group("zone_spec")
    _write_array(zone_spec, "zone_id", _bytes_array(["top_left", "top_right", "bottom_left", "bottom_right"]))
    _write_array(zone_spec, "label_bytes", _bytes_array(["Top left", "Top right", "Bottom left", "Bottom right"]))
    _write_array(zone_spec, "display_order", np.arange(4, dtype=np.int16))
    _write_array(
        zone_spec,
        "bounds_xyxy",
        np.asarray(
            [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 0.0, 20.0, 10.0],
                [0.0, 10.0, 10.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )
    summary = quadrants.create_group("summary")
    _write_array(
        summary,
        "frame_count",
        np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int64),
    )
    _write_array(
        summary,
        "time_s",
        np.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=np.float32),
    )
    _write_array(
        summary,
        "fraction_of_epoch",
        np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.3, 0.3], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32),
    )
    _write_array(
        summary,
        "fraction_of_detected",
        np.asarray([[0.1, 0.2, 0.3, 0.4], [0.19, 0.23, 0.27, 0.31], [0.21, 0.22, 0.27, 0.3]], dtype=np.float32),
    )
    _write_array(summary, "detected_frame_count", np.asarray([10, 26, 30], dtype=np.int64))
    _write_array(summary, "missing_frame_count", np.asarray([0, 1, 0], dtype=np.int64))
    _write_array(summary, "total_span_frames", np.asarray([10, 27, 30], dtype=np.int64))
    _write_array(summary, "coverage_pct", np.asarray([100.0, 96.3, 100.0], dtype=np.float32))
    return zarr_path


def _make_chaser_result(zarr_path: Path) -> ChaserDistanceResult:
    n = 9
    chasers = np.asarray([0, 1], dtype=np.uint8)
    camera_frame_id = np.arange(n, dtype=np.int64)
    fish_xy = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 3.0],
            [6.0, 4.0],
            [7.0, 12.0],
            [8.0, 12.0],
            [9.0, 12.0],
        ],
        dtype=np.float32,
    )
    chaser_xy = np.zeros((n, 2, 2), dtype=np.float32)
    chaser_xy[:6, 0, :] = np.asarray([0.0, 0.0], dtype=np.float32)
    chaser_xy[:6, 1, :] = np.asarray([10.0, 0.0], dtype=np.float32)
    chaser_xy[6:, 0, :] = np.asarray([15.0, 15.0], dtype=np.float32)
    chaser_xy[6:, 1, :] = np.asarray([5.0, 15.0], dtype=np.float32)
    distance_px = np.linalg.norm(fish_xy[:, None, :] - chaser_xy, axis=2).astype(np.float32)
    distance_mm = (distance_px / 2.0).astype(np.float32)
    windows = (
        ChaserDistanceWindow(0, "pre_event", 0, 2, 0.0, 0.3, 0.3),
        ChaserDistanceWindow(1, "training_event", 3, 5, 0.3, 0.6, 0.3),
        ChaserDistanceWindow(2, "post_event", 6, 8, 0.6, 0.9, 0.3),
    )
    hist_counts = np.asarray(
        [
            [[1, 2, 0], [0, 1, 2]],
            [[2, 1, 0], [1, 1, 1]],
            [[0, 2, 1], [2, 0, 1]],
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
        chaser_behavior_class_id=np.asarray([1, 3], dtype=np.int8),
        chaser_behavior_labels=("aggressive", "inert"),
        camera_frame_id=camera_frame_id,
        stimulus_frame_num=camera_frame_id,
        timestamp_ns=np.arange(n, dtype=np.int64),
        stimulus_epoch_window_id=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32),
        fish_centroid_img_xy=fish_xy,
        fish_centroid_arena_xy=fish_xy,
        chaser_arena_xy=chaser_xy,
        fish_valid=np.asarray([True, True, True, True, False, True, True, True, True], dtype=bool),
        chaser_valid=np.ones((n, 2), dtype=bool),
        distance_px=distance_px,
        distance_mm=distance_mm,
        nearest_chaser_index=np.argmin(distance_mm, axis=1).astype(np.int16),
        nearest_distance_mm=np.min(distance_mm, axis=1).astype(np.float32),
        windows=windows,
        epoch_valid_frame_count=np.asarray([[3, 3], [2, 2], [3, 3]], dtype=np.int64),
        epoch_mean_distance_mm=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        epoch_min_distance_mm=np.asarray([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float32),
        epoch_p05_distance_mm=np.asarray([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float32),
        epoch_p50_distance_mm=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        epoch_p95_distance_mm=np.asarray([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]], dtype=np.float32),
        epoch_fraction_within_threshold=np.asarray([[0.2, 0.4], [0.6, 0.8], [0.7, 0.9]], dtype=np.float32),
        threshold_mm=20.0,
        distribution_bin_width_mm=2.0,
        histogram_bin_edges_mm=np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
        histogram_bin_centers_mm=np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
        histogram_counts=hist_counts,
        histogram_density=hist_density,
    )


def test_chaser_distance_writer_adds_chaser_protocol_interactive_spec(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    result = _make_chaser_result(zarr_path)

    run_path = write_chaser_distance_run(zarr_path, result, overwrite=True)

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root[run_path]
    assert run.attrs["coordinate_frame"] == "arena_relative_canvas_px"
    assert run.attrs["coordinate_origin"] == "top_left_of_active_arena"
    assert run.attrs["x_axis_direction"] == "right"
    assert run.attrs["y_axis_direction"] == "down"
    assert run["positions"].attrs["coordinate_frame"] == "arena_relative_canvas_px"
    assert run["positions"].attrs["coordinate_origin"] == "top_left_of_active_arena"
    assert run["positions"].attrs["x_axis_direction"] == "right"
    assert run["positions"].attrs["y_axis_direction"] == "down"
    assert run["positions"].attrs["fish_centroid_arena_xy_coordinate_origin"] == "top_left_of_active_arena"
    visualizations = run["visualizations"]
    assert (
        visualizations[TIMESERIES_PNG_ARTIFACT_NAME].attrs[
            "visualization_contract_id"
        ]
        == TIMESERIES_VISUALIZATION_CONTRACT_ID
    )
    assert (
        visualizations[EPOCH_DISTRIBUTION_PNG_ARTIFACT_NAME].attrs[
            "visualization_contract_id"
        ]
        == EPOCH_DISTRIBUTION_VISUALIZATION_CONTRACT_ID
    )
    spec_group = visualizations[DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT]
    assert spec_group.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec_group.attrs["snapshot_artifact"] == "chaser_distance_timeseries_png"
    assert spec_group.attrs["renderer"] == CHASER_DASHBOARD_RENDERER

    spec_bytes = np.asarray(spec_group["spec_json"][:], dtype=np.uint8).tobytes()
    spec = json.loads(spec_bytes.decode("utf-8"))
    assert spec["schema_id"] == CHASER_DASHBOARD_SPEC_SCHEMA_ID
    assert spec["artifact_family"] == "chaser_protocol_dashboard"
    assert spec["protocol_family"] == "chaser"
    assert spec["source_paths"]["fish_centroid_arena_xy"].endswith("/positions/fish_centroid_arena_xy")
    assert spec["source_paths"]["distance_mm"].endswith("/distances/distance_mm")
    assert spec["source_paths"]["detection_occupancy_heatmap_normalized"].endswith(
        "analysis/detection_occupancy_runs/occupancy_1/heatmaps/normalized"
    )
    assert spec["source_paths"]["detection_spatial_occupancy"].endswith(
        "analysis/detection_occupancy_runs/occupancy_1/spatial_occupancy"
    )
    assert spec["source_runs"]["detection_occupancy"] == "occupancy_1"
    assert spec["static_artifacts"]["detection_occupancy"].endswith(
        "analysis/detection_occupancy_runs/occupancy_1/visualizations/detection_occupancy_overview_png"
    )

    manifest = run.attrs["visualizations"]
    assert manifest[DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT]["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID


def test_chaser_protocol_interactive_loader_builds_plot_dataframes(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_detection_occupancy(tmp_path)
    result = _make_chaser_result(zarr_path)
    write_chaser_distance_run(zarr_path, result, overwrite=True)

    options = discover_chaser_dashboard_options(zarr_path)
    assert [option.run_name for option in options] == ["chaser_distance_1"]

    data = load_chaser_dashboard_data(zarr_path, run_path=options[0].run_path)
    assert data.fps == 10.0
    assert data.distance_mm.shape == (9, 2)
    assert data.occupancy_normalized is not None
    assert data.occupancy_normalized.shape == (3, 2, 2)
    assert data.chaser_color_hex == {0: "#ff0000", 1: "#0000ff"}
    assert data.chaser_source_img_xy is not None
    np.testing.assert_allclose(data.chaser_source_img_xy, data.chaser_arena_xy)
    assert [zone_set.zone_set_id for zone_set in data.spatial_occupancy] == ["image_quadrants_v1"]

    windows_df = to_window_dataframe(data)
    assert windows_df["label"].tolist() == ["pre_event", "training_event", "post_event"]
    spatial_df = to_spatial_occupancy_dataframe(data)
    assert spatial_df["zone_id"].tolist()[:4] == ["top_left", "top_right", "bottom_left", "bottom_right"]
    assert spatial_df[["x_min", "y_min", "x_max", "y_max"]].iloc[0].tolist() == [0.0, 0.0, 10.0, 10.0]
    assert spatial_df.loc[spatial_df["window_label"] == "training_event", "frame_count"].tolist() == [5, 6, 7, 8]
    assert spatial_df.loc[spatial_df["window_label"] == "post_event", "frame_count"].tolist() == [9, 10, 11, 12]

    distance_df = to_distance_timeseries_dataframe(data)
    assert "distance_mm_chaser_0" in distance_df.columns
    assert "distance_mm_chaser_1" in distance_df.columns
    assert distance_df["time_s"].tolist() == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    legacy_options = discover_goodcopbadcop_chaser_dashboard_options(zarr_path)
    assert [option.run_name for option in legacy_options] == ["chaser_distance_1"]
    legacy_data = load_goodcopbadcop_interactive_data(zarr_path, run_path=legacy_options[0].run_path)
    assert legacy_data.run_name == data.run_name

    position_df = to_position_dataframe(data)
    assert position_df["fish_valid"].tolist() == [True, True, True, True, False, True, True, True, True]
    assert position_df.loc[0, "unit"] == "arena_relative_canvas_px"
