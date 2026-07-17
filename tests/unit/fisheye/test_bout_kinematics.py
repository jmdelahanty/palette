from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest
import zarr

import fisheye.analysis.bout_kinematics as bout_kinematics_module
from fisheye.analysis.bout_kinematics import (
    BOUT_KINEMATICS_LAYOUT_DEFAULT,
    LAYOUT_COMPACT_TABULAR_V2,
    LAYOUT_HIERARCHICAL_V1,
    compute_and_save_bout_kinematics,
    normalize_heading_level,
    resolve_bout_kinematics_tables,
)
from fisheye.shared.zarr.columnar import load_structured_dataset, write_columnar_dataset


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    group.create_array(name, data=data, chunks=data.shape, overwrite=True)


def _make_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")

    tk_parent = analysis.create_group("track_kinematics_runs")
    offline = tk_parent.create_group("offline")
    offline.attrs["latest"] = "tk_1"
    tk = offline.create_group("tk_1")
    tk.attrs["fps"] = 10.0
    track = tk.create_group("tracks").create_group("id_0")
    frames = np.arange(10, dtype=np.int64)
    _write_array(track, "frame_indices", frames)
    _write_array(track, "time_seconds", frames.astype(np.float32) / 10.0)
    _write_array(
        track,
        "smoothed_heading_degrees",
        np.asarray([0, 10, 10, 20, 40, 20, 30, 30, 0, 0], dtype=np.float32),
    )
    _write_array(
        track,
        "heading_degrees",
        np.asarray([0, 0, 0, 20, 60, 20, 40, 40, 0, 0], dtype=np.float32),
    )
    _write_array(
        track,
        "positions_px",
        np.column_stack([frames.astype(np.float32), frames.astype(np.float32) * 2.0]),
    )
    _write_array(
        track,
        "positions_mm",
        np.column_stack([frames.astype(np.float32) * 0.1, frames.astype(np.float32) * 0.05]),
    )
    _write_array(
        track,
        "speed_filtered_mm",
        np.asarray([0.0, 0.0, 0.0, 0.2, 3.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    _write_array(
        track,
        "frame_path_distance_filtered_mm",
        np.asarray([0.0, 0.0, 0.0, 0.02, 0.30, 0.02, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    _write_array(
        track,
        "frame_path_distance_filtered_px",
        np.asarray([0.0, 0.0, 0.0, 0.2, 3.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    delta_seconds = np.zeros(frames.size, dtype=np.float32)
    delta_seconds[1:] = 0.1
    _write_array(track, "delta_seconds", delta_seconds)
    _write_array(track, "transition_valid", np.asarray([False, *([True] * 9)], dtype=bool))
    _write_array(track, "sample_valid", np.ones(10, dtype=bool))

    bout_parent = analysis.create_group("swim_bout_runs")
    bout_parent.attrs["latest"] = "bouts_1"
    bout_run = bout_parent.create_group("bouts_1")
    bout_run.attrs["source_track_kinematics_run"] = "tk_1"
    bout_run.attrs["track_id"] = 0
    bout_run.attrs["default_level"] = "speed_filtered"
    speed_group = bout_run.create_group("speed_filtered")
    bouts = np.asarray(
        [(1, 3, 5, 3, 5, 0.25, 0.55, 0.30, True, True)],
        dtype=[
            ("bout_id", "i4"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("core_start_frame", "i8"),
            ("core_end_frame", "i8"),
            ("core_start_time_s_interpolated", "f8"),
            ("core_end_time_s_interpolated", "f8"),
            ("core_duration_s_interpolated", "f8"),
            ("core_start_time_interpolated_valid", "?"),
            ("core_end_time_interpolated_valid", "?"),
        ],
    )
    write_columnar_dataset(speed_group, "bouts", bouts, {"n_bouts": 1})
    return zarr_path


def _add_eye_angle_run(root: zarr.Group) -> None:
    eye_parent = root["analysis"].create_group("eye_angle_runs")
    eye_parent.attrs["latest"] = "eye_1"
    eye_run = eye_parent.create_group("eye_1")
    eye_run.attrs["schema_id"] = "analysis.eye_angle_runs"
    eye_run.attrs["schema_version"] = 3
    eye_run.attrs["method"] = "ellipse_and_centroid_eye_angles"
    eye_run.attrs["preferred_angle_family"] = "gaze"
    eye_run.attrs["preferred_eye_axis"] = "ellipse_minor"
    frame_angles = eye_run.create_group("angles").create_group("frame")
    frames = np.arange(10, dtype=np.float32)
    _write_array(frame_angles, "left_gaze_deg", frames)
    _write_array(frame_angles, "right_gaze_deg", frames * 2.0)
    _write_array(frame_angles, "vergence_gaze_deg", frames * 3.0)
    _write_array(frame_angles, "vergence_gaze_signed_deg", frames * 4.0)
    frame_qa = eye_run.create_group("qa").create_group("frame")
    _write_array(frame_qa, "valid_frame", np.ones(10, dtype=bool))


def test_normalize_heading_level_accepts_aliases() -> None:
    assert normalize_heading_level("smoothed") == "heading_smoothed"
    assert normalize_heading_level("heading_raw") == "heading_raw"


def test_normalize_heading_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported heading level"):
        normalize_heading_level("median")


def test_bout_kinematics_layout_default_is_compact_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    assert BOUT_KINEMATICS_LAYOUT_DEFAULT == LAYOUT_COMPACT_TABULAR_V2
    assert (
        inspect.signature(compute_and_save_bout_kinematics).parameters["layout"].default
        == BOUT_KINEMATICS_LAYOUT_DEFAULT
    )
    assert (
        inspect.signature(compute_and_save_bout_kinematics)
        .parameters["include_eye_gaze"]
        .default
        is True
    )

    captured: dict[str, object] = {}

    def _fake_compute_and_save_bout_kinematics(**kwargs: object) -> str:
        captured.update(kwargs)
        return "run"

    monkeypatch.setattr(
        bout_kinematics_module,
        "compute_and_save_bout_kinematics",
        _fake_compute_and_save_bout_kinematics,
    )

    assert bout_kinematics_module.main(["/tmp/example.zarr"]) == 0
    assert captured["layout"] == BOUT_KINEMATICS_LAYOUT_DEFAULT
    assert captured["include_eye_gaze"] is True

    captured.clear()
    assert (
        bout_kinematics_module.main(
            ["/tmp/example.zarr", "--no-include-eye-gaze"]
        )
        == 0
    )
    assert captured["include_eye_gaze"] is False


def test_default_eye_gaze_fails_before_creating_output_without_eye_angles(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)

    with pytest.raises(ValueError, match="enabled by default"):
        compute_and_save_bout_kinematics(
            zarr_path,
            run_name="missing_eye_angles",
            track_kinematics_run="tk_1",
            track_id=0,
            swim_bout_run="bouts_1",
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "bout_kinematics_runs" not in root["analysis"]


def test_compute_can_write_separate_node_local_output_without_mutating_source(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)
    output_zarr = tmp_path / "node-local-output.zarr"

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        output_zarr_path=output_zarr,
        run_name="bout_local",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        include_eye_gaze=False,
        write_visualizations=False,
        output_shard_rows=8_192,
    )

    source = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "bout_kinematics_runs" not in source["analysis"]
    output = zarr.open_group(str(output_zarr), mode="r", use_consolidated=False)
    parent = output["analysis/bout_kinematics_runs"]
    assert parent.attrs["latest_complete"] == run_name
    run = parent[run_name]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["parameters"]["output_shard_rows"] == 8_192
    assert run.attrs["physical_storage_layout"]["shard_policy"] == (
        "multi_chunk_capped"
    )
    assert run.attrs["physical_storage_layout"]["array_count"] > 0


def test_compute_and_save_bout_kinematics_writes_hierarchical_heading_levels(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        run_name="bout_kinematics_1",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        pre_window_s=0.2,
        post_window_s=0.2,
        physical_active_threshold_mm_s=0.1,
        physical_active_boundary_margin_s=0.1,
        include_eye_gaze=False,
        write_visualizations=True,
        visualization_bins=8,
        layout=LAYOUT_HIERARCHICAL_V1,
        overwrite=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    parent = root["analysis"]["bout_kinematics_runs"]
    run = parent[run_name]

    assert parent.attrs["latest"] == "bout_kinematics_1"
    assert run.attrs["status"] == "complete"
    assert run.attrs["schema_id"] == "analysis.bout_kinematics_runs"
    assert run.attrs["schema_version"] == 7
    assert run.attrs["method_version"] == "bout_kinematics.v7"
    assert run.attrs["default_heading_level"] == "heading_smoothed"
    assert run.attrs["source_swim_bout_run"] == "bouts_1"
    assert run.attrs["source_swim_bout_speed_level"] == "speed_filtered"
    assert run.attrs["parameters"]["resolved_pre_window_frames"] == 2
    assert run.attrs["parameters"]["resolved_post_window_frames"] == 2
    assert run.attrs["parameters"]["source_interpolated_threshold_fields"] == [
        "core_start_time_s_interpolated",
        "core_end_time_s_interpolated",
        "core_duration_s_interpolated",
        "core_start_time_interpolated_valid",
        "core_end_time_interpolated_valid",
    ]
    assert run.attrs["parameters"]["source_peak_event_fields"] == []
    assert run.attrs["parameters"]["eye_gaze"]["enabled"] is False
    assert run.attrs["parameters"]["physical_active"] == {
        "enabled": True,
        "boundary_policy": "physical_active",
        "boundary_constraint": "search_with_margin",
        "boundary_margin_s": 0.1,
        "resolved_boundary_margin_frames": 1,
        "threshold_mm_s": 0.1,
        "measurement_signal_level": "speed_filtered",
        "measurement_signal_array": "speed_filtered_mm",
    }
    assert run.attrs["source_refs"]["zarr_path"] == str(zarr_path)
    assert run.attrs["source_refs"]["source_track_id"] == 0
    assert run.attrs["source_refs"]["source_swim_bout_candidate_id"] == 0
    assert run.attrs["source_refs"]["source_swim_bout_signal_id"] == 0
    assert run.attrs["source_refs"]["source_swim_bout_signal_role"] == "physical_estimator"
    assert run.attrs["source_refs"]["source_heading_arrays"] == {
        "heading_smoothed": (
            "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/"
            "smoothed_heading_degrees"
        ),
        "heading_raw": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/heading_degrees",
    }
    assert run.attrs["source_refs"]["source_position_arrays"] == {
        "positions_mm": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/positions_mm",
        "positions_px": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/positions_px",
    }
    assert run.attrs["source_refs"]["source_movement_arrays"] == {
        "physical_active_speed": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/speed_filtered_mm",
        "physical_active_path_distance_mm": (
            "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/"
            "frame_path_distance_filtered_mm"
        ),
        "physical_active_path_distance_px": (
            "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/"
            "frame_path_distance_filtered_px"
        ),
    }
    assert run.attrs["source_refs"]["source_validity_arrays"] == {
        "delta_seconds": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/delta_seconds",
        "transition_valid": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/transition_valid",
        "sample_valid": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/sample_valid",
    }
    assert run.attrs["provenance"]["stage"] == "bout_kinematics"
    assert run.attrs["provenance"]["inputs"]["zarr_path"] == str(zarr_path)
    assert run.attrs["provenance"]["inputs"]["source_heading_arrays"] == run.attrs["source_refs"][
        "source_heading_arrays"
    ]

    assert run.attrs["analysis_levels"] == ["movement", "heading_smoothed", "heading_raw"]
    movement = run["movement"]["per_bout_metrics"]
    movement_records, _ = load_structured_dataset(run["movement"], "per_bout_metrics")
    assert movement.attrs["schema_id"] == "analysis.bout_kinematics_runs.movement.per_bout_metrics"
    assert movement.attrs["physical_active_boundary_policy"] == "physical_active"
    assert movement.attrs["physical_active_boundary_constraint"] == "search_with_margin"
    assert movement.attrs["physical_active_signal_level"] == "speed_filtered"
    assert movement["physical_active_start_frame"][:].tolist() == [3]
    assert movement["physical_active_end_frame"][:].tolist() == [5]
    np.testing.assert_allclose(movement["physical_active_duration_s"][:], [0.3])
    np.testing.assert_allclose(movement["physical_active_observed_duration_s"][:], [0.3])
    np.testing.assert_allclose(movement["physical_active_start_time_s_interpolated"][:], [0.25])
    np.testing.assert_allclose(movement["physical_active_end_time_s_interpolated"][:], [0.55])
    np.testing.assert_allclose(movement["physical_active_duration_s_interpolated"][:], [0.30])
    np.testing.assert_allclose(movement["physical_active_path_length_mm"][:], [0.34])
    np.testing.assert_allclose(movement["physical_active_path_length_px"][:], [3.4])
    np.testing.assert_allclose(movement["physical_active_mean_speed_mm_s"][:], [0.34 / 0.3])
    np.testing.assert_allclose(movement["physical_active_peak_speed_mm_s"][:], [3.0])
    assert movement["physical_active_valid"][:].tolist() == [True]
    assert movement_records["physical_active_boundary_policy_bytes"].tolist() == [b"physical_active"]
    assert movement_records["physical_active_boundary_constraint_bytes"].tolist() == [b"search_with_margin"]
    assert movement_records["failure_reason_bytes"].tolist() == [b"ok"]

    smoothed = run["heading_smoothed"]["per_bout_metrics"]
    smoothed_records, _ = load_structured_dataset(run["heading_smoothed"], "per_bout_metrics")
    assert smoothed.attrs["heading_source_array"] == "smoothed_heading_degrees"
    assert smoothed.attrs["source_bout_count"] == 1
    assert smoothed.attrs["source_interpolated_threshold_fields"] == [
        "core_start_time_s_interpolated",
        "core_end_time_s_interpolated",
        "core_duration_s_interpolated",
        "core_start_time_interpolated_valid",
        "core_end_time_interpolated_valid",
    ]
    assert smoothed.attrs["source_peak_event_fields"] == []
    np.testing.assert_allclose(smoothed["pre_heading_mean_deg"][:], [10.0])
    np.testing.assert_allclose(smoothed["post_heading_mean_deg"][:], [30.0])
    np.testing.assert_allclose(smoothed["net_delta_heading_deg"][:], [20.0])
    np.testing.assert_allclose(smoothed["abs_net_delta_heading_deg"][:], [20.0])
    assert smoothed["pre_epoch_start_frame"][:].tolist() == [1]
    assert smoothed["pre_epoch_end_frame"][:].tolist() == [2]
    assert smoothed["post_epoch_start_frame"][:].tolist() == [6]
    assert smoothed["post_epoch_end_frame"][:].tolist() == [7]
    np.testing.assert_allclose(smoothed["source_core_start_time_s_interpolated"][:], [0.25])
    np.testing.assert_allclose(smoothed["source_core_end_time_s_interpolated"][:], [0.55])
    np.testing.assert_allclose(smoothed["source_core_duration_s_interpolated"][:], [0.30])
    assert smoothed["source_core_start_time_interpolated_valid"][:].tolist() == [True]
    assert smoothed["source_core_end_time_interpolated_valid"][:].tolist() == [True]
    assert smoothed["source_peak_frame"][:].tolist() == [-1]
    assert np.isnan(smoothed["source_peak_left_width_time_s"][:]).all()
    assert smoothed_records["source_peak_boundary_mode_bytes"].tolist() == [b""]
    np.testing.assert_allclose(smoothed["pre_position_mean_x_px"][:], [1.5])
    np.testing.assert_allclose(smoothed["pre_position_mean_y_px"][:], [3.0])
    np.testing.assert_allclose(smoothed["post_position_mean_x_px"][:], [6.5])
    np.testing.assert_allclose(smoothed["post_position_mean_y_px"][:], [13.0])
    np.testing.assert_allclose(smoothed["interbout_epoch_displacement_px"][:], [np.hypot(5.0, 10.0)])
    np.testing.assert_allclose(smoothed["pre_position_mean_x_mm"][:], [0.15])
    np.testing.assert_allclose(smoothed["pre_position_mean_y_mm"][:], [0.075])
    np.testing.assert_allclose(smoothed["post_position_mean_x_mm"][:], [0.65])
    np.testing.assert_allclose(smoothed["post_position_mean_y_mm"][:], [0.325])
    np.testing.assert_allclose(smoothed["interbout_epoch_displacement_mm"][:], [np.hypot(0.5, 0.25)])
    np.testing.assert_allclose(smoothed["within_heading_range_deg"][:], [20.0])
    np.testing.assert_allclose(smoothed["within_heading_peak_to_peak_deg"][:], [20.0])
    np.testing.assert_allclose(smoothed["within_heading_path_deg"][:], [40.0])
    np.testing.assert_allclose(smoothed["within_angular_velocity_mean_deg_s"][:], [0.0])
    np.testing.assert_allclose(smoothed["within_angular_speed_mean_deg_s"][:], [200.0])
    np.testing.assert_allclose(smoothed["within_angular_speed_max_deg_s"][:], [200.0])
    np.testing.assert_allclose(smoothed["within_angular_velocity_std_deg_s"][:], [200.0])
    assert smoothed["within_heading_zero_crossings"][:].tolist() == [1]
    assert smoothed["pre_window_valid"][:].tolist() == [True]
    assert smoothed["post_window_valid"][:].tolist() == [True]
    assert smoothed["pre_position_valid"][:].tolist() == [True]
    assert smoothed["post_position_valid"][:].tolist() == [True]
    assert smoothed["pre_position_sample_count"][:].tolist() == [2]
    assert smoothed["post_position_sample_count"][:].tolist() == [2]
    assert smoothed["within_window_valid"][:].tolist() == [True]
    assert smoothed["within_angular_velocity_valid"][:].tolist() == [True]
    assert smoothed["within_angular_velocity_transition_count"][:].tolist() == [2]
    assert smoothed["dominant_frequency_valid"][:].tolist() == [False]
    assert smoothed_records["failure_reason_bytes"].tolist() == [b"dominant_frequency_disabled"]

    raw = run["heading_raw"]["per_bout_metrics"]
    np.testing.assert_allclose(raw["pre_heading_mean_deg"][:], [0.0])
    np.testing.assert_allclose(raw["post_heading_mean_deg"][:], [40.0])
    np.testing.assert_allclose(raw["net_delta_heading_deg"][:], [40.0])
    np.testing.assert_allclose(raw["within_heading_range_deg"][:], [40.0])
    np.testing.assert_allclose(raw["within_heading_path_deg"][:], [80.0])
    np.testing.assert_allclose(raw["within_angular_velocity_mean_deg_s"][:], [0.0])
    np.testing.assert_allclose(raw["within_angular_speed_mean_deg_s"][:], [400.0])
    np.testing.assert_allclose(raw["within_angular_speed_max_deg_s"][:], [400.0])

    visualizations = run["visualizations"]
    png = visualizations["bout_kinematics_summary_track_0_png"]
    assert bytes(np.asarray(png[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert png.attrs["plot_schema_id"] == "palette.plot_spec.bout_kinematics_summary.v1"
    assert png.attrs["visualization_contract_id"] == (
        bout_kinematics_module.BOUT_HEADING_VISUALIZATION_CONTRACT_ID
    )
    assert png.attrs["renderer"] == bout_kinematics_module.BOUT_KINEMATICS_PLOT_RENDERER
    assert png.attrs["renderer_version"] == (
        bout_kinematics_module.BOUT_KINEMATICS_RENDERER_VERSION
    )
    assert png.attrs["parameters"]["bins"] == 8
    spec_artifact = visualizations["bout_kinematics_summary_track_0_interactive"]
    assert spec_artifact.attrs["snapshot_artifact"] == "bout_kinematics_summary_track_0_png"
    spec_payload = np.asarray(spec_artifact["spec_json"][:], dtype=np.uint8).tobytes()
    assert b"net_heading_change_histograms" in spec_payload
    assert b"within_bout_heading_histograms" in spec_payload

    movement_png = visualizations["bout_movement_summary_track_0_png"]
    assert bytes(np.asarray(movement_png[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert movement_png.attrs["plot_schema_id"] == "palette.plot_spec.bout_movement_summary.v1"
    assert movement_png.attrs["visualization_contract_id"] == (
        bout_kinematics_module.BOUT_MOVEMENT_VISUALIZATION_CONTRACT_ID
    )
    assert movement_png.attrs["renderer"] == bout_kinematics_module.BOUT_MOVEMENT_PLOT_RENDERER
    assert movement_png.attrs["parameters"]["physical_active"]["measurement_signal_level"] == "speed_filtered"
    movement_spec = visualizations["bout_movement_summary_track_0_interactive"]
    assert movement_spec.attrs["snapshot_artifact"] == "bout_movement_summary_track_0_png"
    assert movement_spec.attrs["plot_schema_id"] == "palette.plot_spec.bout_movement_summary.v1"
    assert movement_spec.attrs["renderer"] == bout_kinematics_module.BOUT_MOVEMENT_PLOT_RENDERER
    movement_payload = np.asarray(movement_spec["spec_json"][:], dtype=np.uint8).tobytes()
    assert b"bout_physical_movement_histograms" in movement_payload
    assert b"physical_active_peak_speed_mm_s" in movement_payload


def test_compute_and_save_bout_kinematics_writes_default_compact_v2_layout(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    _add_eye_angle_run(root)

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        run_name="bout_kinematics_compact_v2",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        heading_levels=("heading_smoothed", "heading_raw"),
        pre_window_s=0.2,
        post_window_s=0.2,
        physical_active_threshold_mm_s=0.1,
        physical_active_boundary_margin_s=0.1,
        eye_angle_run="eye_1",
        vergence_threshold_deg=10.0,
    )

    run = zarr.open_group(str(zarr_path), mode="r")["analysis"]["bout_kinematics_runs"][run_name]
    assert run.attrs["layout"] == "compact_tabular_v2"
    assert run.attrs["status"] == "complete"
    assert run.attrs["analysis_levels"] == ["movement", "heading_smoothed", "heading_raw", "eye_gaze"]
    assert run.attrs["parameters"]["layout"] == "compact_tabular_v2"
    assert "movement" not in run
    assert "heading_smoothed" not in run
    assert "heading_raw" not in run
    assert "eye_gaze" not in run
    assert {"level_index", "movement_metrics", "heading_metrics", "eye_gaze_metrics"}.issubset(set(run.keys()))

    level_index, level_index_attrs = load_structured_dataset(run, "level_index")
    assert level_index_attrs["layout"] == "compact_tabular_v2"
    assert level_index["analysis_level_bytes"].tolist() == [
        b"movement",
        b"heading_smoothed",
        b"heading_raw",
        b"eye_gaze",
    ]
    assert level_index["row_count"].tolist() == [1, 1, 1, 1]

    movement_metrics, movement_attrs = load_structured_dataset(run, "movement_metrics")
    assert movement_attrs["analysis_level"] == "movement"
    assert movement_metrics["analysis_level_bytes"].tolist() == [b"movement"]
    np.testing.assert_allclose(movement_metrics["physical_active_path_length_mm"], [0.34])

    heading_metrics, heading_attrs = load_structured_dataset(run, "heading_metrics")
    assert heading_attrs["heading_levels"] == ["heading_smoothed", "heading_raw"]
    assert heading_metrics["heading_level_bytes"].tolist() == [b"heading_smoothed", b"heading_raw"]
    np.testing.assert_allclose(heading_metrics["net_delta_heading_deg"], [20.0, 40.0])

    records_by_level, level_attrs_by_level, table_attrs_by_level = resolve_bout_kinematics_tables(run)
    assert set(records_by_level) == {"movement", "heading_smoothed", "heading_raw", "eye_gaze"}
    assert "analysis_level_bytes" not in records_by_level["heading_smoothed"].dtype.names
    np.testing.assert_allclose(records_by_level["movement"]["physical_active_path_length_mm"], [0.34])
    np.testing.assert_allclose(records_by_level["heading_smoothed"]["net_delta_heading_deg"], [20.0])
    np.testing.assert_allclose(records_by_level["heading_raw"]["net_delta_heading_deg"], [40.0])
    np.testing.assert_allclose(records_by_level["eye_gaze"]["post_vergence_gaze_mean_deg"], [19.5])
    assert level_attrs_by_level["heading_smoothed"]["is_default_heading_level"] is True
    assert table_attrs_by_level["heading_raw"]["layout"] == "compact_tabular_v2"

    filtered_records, _filtered_level_attrs, _filtered_table_attrs = resolve_bout_kinematics_tables(
        run,
        heading_level="raw",
    )
    assert set(filtered_records) == {"heading_raw"}
    np.testing.assert_allclose(filtered_records["heading_raw"]["net_delta_heading_deg"], [40.0])


def _read_interactive_spec_json(spec_artifact: zarr.Group) -> dict[str, object]:
    spec_payload = np.asarray(spec_artifact["spec_json"][:], dtype=np.uint8).tobytes()
    return json.loads(spec_payload.decode("utf-8"))


def test_compute_and_save_bout_kinematics_compact_v2_writes_zarr_artifacts(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    _add_eye_angle_run(root)

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        run_name="bout_kinematics_compact_v2_artifacts",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        heading_levels=("heading_smoothed", "heading_raw"),
        pre_window_s=0.2,
        post_window_s=0.2,
        physical_active_threshold_mm_s=0.1,
        physical_active_boundary_margin_s=0.1,
        include_eye_gaze=True,
        eye_angle_run="eye_1",
        vergence_threshold_deg=10.0,
        layout=LAYOUT_COMPACT_TABULAR_V2,
        write_visualizations=True,
        visualization_bins=8,
    )

    run = zarr.open_group(str(zarr_path), mode="r")["analysis"]["bout_kinematics_runs"][run_name]
    visualizations = run["visualizations"]
    png = visualizations["bout_kinematics_summary_track_0_png"]
    assert bytes(np.asarray(png[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert png.attrs["parameters"]["layout"] == LAYOUT_COMPACT_TABULAR_V2
    assert png.attrs["source_paths"]["heading_metrics"].endswith("/heading_metrics")
    assert png.attrs["source_filters"]["heading_smoothed"]["heading_level_bytes"] == "heading_smoothed"

    heading_spec = _read_interactive_spec_json(visualizations["bout_kinematics_summary_track_0_interactive"])
    assert heading_spec["layout"] == LAYOUT_COMPACT_TABULAR_V2
    assert heading_spec["source_paths"]["heading_metrics"].endswith("/heading_metrics")
    assert heading_spec["source_paths"]["heading_smoothed.net_delta_heading_deg"].endswith(
        "/heading_metrics/net_delta_heading_deg"
    )
    assert heading_spec["source_filters"]["heading_smoothed"] == {
        "table": "heading_metrics",
        "heading_level_bytes": "heading_smoothed",
    }

    movement_spec = _read_interactive_spec_json(visualizations["bout_movement_summary_track_0_interactive"])
    assert movement_spec["layout"] == LAYOUT_COMPACT_TABULAR_V2
    assert movement_spec["source_paths"]["movement_metrics"].endswith("/movement_metrics")
    assert movement_spec["source_paths"]["movement.physical_active_peak_speed_mm_s"].endswith(
        "/movement_metrics/physical_active_peak_speed_mm_s"
    )
    assert movement_spec["source_filters"]["movement"] == {
        "table": "movement_metrics",
        "analysis_level_bytes": "movement",
    }

    eye_spec = _read_interactive_spec_json(visualizations["bout_eye_gaze_summary_track_0_interactive"])
    assert eye_spec["layout"] == LAYOUT_COMPACT_TABULAR_V2
    assert eye_spec["source_paths"]["eye_gaze_metrics"].endswith("/eye_gaze_metrics")
    assert eye_spec["source_paths"]["eye_gaze.post_vergence_gaze_mean_deg"].endswith(
        "/eye_gaze_metrics/post_vergence_gaze_mean_deg"
    )
    assert eye_spec["source_filters"]["eye_gaze"] == {
        "table": "eye_gaze_metrics",
        "analysis_level_bytes": "eye_gaze",
    }


def test_compute_and_save_bout_kinematics_rejects_exponential_physical_source(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)

    with pytest.raises(ValueError, match="Unsupported physical_active_signal_level"):
        compute_and_save_bout_kinematics(
            zarr_path,
            run_name="bad_physical_source",
            track_kinematics_run="tk_1",
            track_id=0,
            swim_bout_run="bouts_1",
            physical_active_signal_level="exponential",
            include_eye_gaze=False,
        )


def test_compute_and_save_bout_kinematics_writes_optional_eye_gaze_metrics(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    _add_eye_angle_run(root)

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        run_name="bout_kinematics_eye_gaze",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        heading_levels=("heading_smoothed",),
        pre_window_s=0.2,
        post_window_s=0.2,
        include_eye_gaze=True,
        eye_angle_run="eye_1",
        eye_validity_min_fraction=1.0,
        vergence_threshold_deg=10.0,
        write_visualizations=True,
        visualization_bins=8,
        layout=LAYOUT_HIERARCHICAL_V1,
    )

    run = zarr.open_group(str(zarr_path), mode="r")["analysis"]["bout_kinematics_runs"][run_name]
    assert run.attrs["schema_version"] == 7
    assert run.attrs["analysis_levels"] == ["movement", "heading_smoothed", "eye_gaze"]
    assert run.attrs["parameters"]["eye_gaze"] == {
        "enabled": True,
        "eye_angle_run": "eye_1",
        "eye_angle_family": "gaze",
        "eye_validity_min_fraction": 1.0,
        "vergence_threshold_deg": 10.0,
    }
    assert run.attrs["source_refs"]["source_eye_angle_run"] == "eye_1"
    assert run.attrs["source_refs"]["source_eye_angle_path"] == "analysis/eye_angle_runs/eye_1"
    assert run.attrs["source_refs"]["source_eye_angle_schema_version"] == 3
    assert run.attrs["source_refs"]["source_eye_angle_arrays"]["vergence_gaze_deg"] == (
        "analysis/eye_angle_runs/eye_1/angles/frame/vergence_gaze_deg"
    )

    eye_group = run["eye_gaze"]
    assert eye_group.attrs["eye_angle_family"] == "gaze"
    metrics = eye_group["per_bout_metrics"]
    records, _ = load_structured_dataset(eye_group, "per_bout_metrics")
    assert metrics.attrs["schema_id"] == "analysis.bout_kinematics_runs.eye_gaze.per_bout_metrics"
    assert metrics.attrs["eye_angle_run"] == "eye_1"
    assert metrics.attrs["vergence_threshold_deg"] == 10.0

    assert metrics["pre_epoch_start_frame"][:].tolist() == [1]
    assert metrics["pre_epoch_end_frame"][:].tolist() == [2]
    assert metrics["post_epoch_start_frame"][:].tolist() == [6]
    assert metrics["post_epoch_end_frame"][:].tolist() == [7]
    assert metrics["within_epoch_start_frame"][:].tolist() == [3]
    assert metrics["within_epoch_end_frame"][:].tolist() == [5]
    np.testing.assert_allclose(metrics["pre_left_gaze_mean_deg"][:], [1.5])
    np.testing.assert_allclose(metrics["pre_right_gaze_mean_deg"][:], [3.0])
    np.testing.assert_allclose(metrics["pre_vergence_gaze_mean_deg"][:], [4.5])
    np.testing.assert_allclose(metrics["pre_vergence_gaze_signed_mean_deg"][:], [6.0])
    np.testing.assert_allclose(metrics["pre_vergence_gaze_std_deg"][:], [1.5])
    np.testing.assert_allclose(metrics["pre_converged_fraction"][:], [0.0])
    np.testing.assert_allclose(metrics["post_left_gaze_mean_deg"][:], [6.5])
    np.testing.assert_allclose(metrics["post_right_gaze_mean_deg"][:], [13.0])
    np.testing.assert_allclose(metrics["post_vergence_gaze_mean_deg"][:], [19.5])
    np.testing.assert_allclose(metrics["post_vergence_gaze_signed_mean_deg"][:], [26.0])
    np.testing.assert_allclose(metrics["post_converged_fraction"][:], [1.0])
    np.testing.assert_allclose(metrics["within_bout_left_gaze_mean_deg"][:], [4.0])
    np.testing.assert_allclose(metrics["within_bout_right_gaze_mean_deg"][:], [8.0])
    np.testing.assert_allclose(metrics["within_bout_vergence_gaze_mean_deg"][:], [12.0])
    np.testing.assert_allclose(metrics["within_bout_vergence_gaze_signed_mean_deg"][:], [16.0])
    np.testing.assert_allclose(metrics["within_bout_vergence_gaze_max_deg"][:], [15.0])
    np.testing.assert_allclose(metrics["within_bout_vergence_gaze_range_deg"][:], [6.0])
    np.testing.assert_allclose(metrics["within_bout_vergence_gaze_std_deg"][:], [np.std([9.0, 12.0, 15.0])])
    np.testing.assert_allclose(metrics["within_bout_converged_fraction"][:], [2.0 / 3.0])
    assert metrics["pre_eye_window_valid"][:].tolist() == [True]
    assert metrics["post_eye_window_valid"][:].tolist() == [True]
    assert metrics["within_eye_window_valid"][:].tolist() == [True]
    assert metrics["pre_eye_sample_count"][:].tolist() == [2]
    assert metrics["post_eye_sample_count"][:].tolist() == [2]
    assert metrics["within_eye_sample_count"][:].tolist() == [3]
    assert records["failure_reason_bytes"].tolist() == [b"ok"]

    visualizations = run["visualizations"]
    png = visualizations["bout_eye_gaze_summary_track_0_png"]
    assert bytes(np.asarray(png[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert png.attrs["plot_schema_id"] == "palette.plot_spec.bout_eye_gaze_summary.v1"
    assert png.attrs["visualization_contract_id"] == (
        bout_kinematics_module.BOUT_EYE_GAZE_VISUALIZATION_CONTRACT_ID
    )
    assert png.attrs["source_runs"]["eye_angle"] == "eye_1"
    assert png.attrs["renderer"] == bout_kinematics_module.BOUT_EYE_GAZE_PLOT_RENDERER
    spec_artifact = visualizations["bout_eye_gaze_summary_track_0_interactive"]
    assert spec_artifact.attrs["snapshot_artifact"] == "bout_eye_gaze_summary_track_0_png"
    assert spec_artifact.attrs["plot_schema_id"] == "palette.plot_spec.bout_eye_gaze_summary.v1"
    assert spec_artifact.attrs["renderer"] == bout_kinematics_module.BOUT_EYE_GAZE_PLOT_RENDERER
    spec_payload = np.asarray(spec_artifact["spec_json"][:], dtype=np.uint8).tobytes()
    assert b"bout_eye_gaze_histograms" in spec_payload


def test_compute_and_save_bout_kinematics_marks_angular_velocity_invalid_across_gaps(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/transition_valid"][4] = False

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        run_name="bout_kinematics_gap",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        include_eye_gaze=False,
        overwrite=False,
        layout=LAYOUT_HIERARCHICAL_V1,
    )

    run = zarr.open_group(str(zarr_path), mode="r")["analysis"]["bout_kinematics_runs"][run_name]
    smoothed = run["heading_smoothed"]["per_bout_metrics"]
    smoothed_records, _ = load_structured_dataset(run["heading_smoothed"], "per_bout_metrics")
    assert smoothed["within_angular_velocity_valid"][:].tolist() == [False]
    assert smoothed["within_angular_velocity_transition_count"][:].tolist() == [2]
    assert np.isnan(smoothed["within_angular_velocity_mean_deg_s"][:]).all()
    assert np.isnan(smoothed["within_angular_speed_mean_deg_s"][:]).all()
    assert b"heading_transition_contains_gap" in smoothed_records["failure_reason_bytes"][0]


def test_compute_and_save_bout_kinematics_copies_peak_event_boundary_context(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    speed_group = root["analysis/swim_bout_runs/bouts_1/speed_filtered"]
    peak_events = np.asarray(
        [(1, 4, 0.40, 12.0, 8.5, 0.18, 6.0, 2.25, 5.75, b"relative_prominence_width", b"none")],
        dtype=[
            ("bout_id", "i4"),
            ("peak_frame", "i8"),
            ("peak_time_s", "f8"),
            ("peak_signal_value_mm_s", "f8"),
            ("peak_prominence_mm_s", "f8"),
            ("peak_width_s", "f8"),
            ("peak_width_height_mm_s", "f8"),
            ("left_width_frame_interpolated", "f8"),
            ("right_width_frame_interpolated", "f8"),
            ("boundary_mode", "S32"),
            ("shape_split_policy", "S32"),
        ],
    )
    write_columnar_dataset(speed_group, "peak_events", peak_events)

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        run_name="bout_kinematics_peak_events",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        heading_levels=("heading_smoothed",),
        pre_window_s=0.2,
        post_window_s=0.2,
        include_eye_gaze=False,
        layout=LAYOUT_HIERARCHICAL_V1,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["bout_kinematics_runs"][run_name]
    assert run.attrs["source_refs"]["source_peak_events_path"] == (
        "analysis/swim_bout_runs/bouts_1/speed_filtered/peak_events"
    )
    assert run.attrs["parameters"]["source_peak_event_fields"] == [
        "bout_id",
        "peak_frame",
        "peak_time_s",
        "peak_signal_value_mm_s",
        "peak_prominence_mm_s",
        "peak_width_s",
        "peak_width_height_mm_s",
        "left_width_frame_interpolated",
        "right_width_frame_interpolated",
        "boundary_mode",
        "shape_split_policy",
    ]
    smoothed = run["heading_smoothed"]["per_bout_metrics"]
    smoothed_records, _ = load_structured_dataset(run["heading_smoothed"], "per_bout_metrics")
    assert smoothed["source_peak_frame"][:].tolist() == [4]
    np.testing.assert_allclose(smoothed["source_peak_time_s"][:], [0.40])
    np.testing.assert_allclose(smoothed["source_peak_signal_value_mm_s"][:], [12.0])
    np.testing.assert_allclose(smoothed["source_peak_prominence_mm_s"][:], [8.5])
    np.testing.assert_allclose(smoothed["source_peak_width_s"][:], [0.18])
    np.testing.assert_allclose(smoothed["source_peak_width_height_mm_s"][:], [6.0])
    np.testing.assert_allclose(smoothed["source_peak_left_width_frame_interpolated"][:], [2.25])
    np.testing.assert_allclose(smoothed["source_peak_right_width_frame_interpolated"][:], [5.75])
    np.testing.assert_allclose(smoothed["source_peak_left_width_time_s"][:], [0.225])
    np.testing.assert_allclose(smoothed["source_peak_right_width_time_s"][:], [0.575])
    assert smoothed_records["source_peak_boundary_mode_bytes"].tolist() == [b"relative_prominence_width"]
    assert smoothed_records["source_peak_shape_split_policy_bytes"].tolist() == [b"none"]


def test_compute_and_save_bout_kinematics_requires_overwrite(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    kwargs = dict(
        zarr_path=zarr_path,
        run_name="candidate",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        pre_window_s=0.2,
        post_window_s=0.2,
        include_eye_gaze=False,
    )

    compute_and_save_bout_kinematics(**kwargs)
    with pytest.raises(ValueError, match="Use --overwrite"):
        compute_and_save_bout_kinematics(**kwargs)
    compute_and_save_bout_kinematics(**kwargs, overwrite=True)


def test_compute_and_save_bout_kinematics_does_not_publish_latest_when_visualization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _make_archive(tmp_path)

    def _fail_visualization_artifacts(*_args, **_kwargs) -> None:
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(
        bout_kinematics_module,
        "write_bout_kinematics_visualization_artifacts",
        _fail_visualization_artifacts,
    )

    with pytest.raises(RuntimeError, match="synthetic plot failure"):
        compute_and_save_bout_kinematics(
            zarr_path,
            run_name="bout_kinematics_failed_plot",
            track_kinematics_run="tk_1",
            track_id=0,
            swim_bout_run="bouts_1",
            speed_level="filtered",
            include_eye_gaze=False,
            write_visualizations=True,
        )

    parent = zarr.open_group(str(zarr_path), mode="r")["analysis"]["bout_kinematics_runs"]
    assert parent.attrs.get("latest") is None
    run = parent["bout_kinematics_failed_plot"]
    assert run.attrs["status"] == "failed"
    assert run.attrs["failure_stage"] == "bout_kinematics_visualization"
    assert "synthetic plot failure" in run.attrs["failure_reason"]


def test_compute_and_save_bout_kinematics_interbout_epoch_mode(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    speed_group = root["analysis/swim_bout_runs/bouts_1/speed_filtered"]
    bouts = np.asarray(
        [(1, 1, 2, 1, 2), (2, 4, 5, 4, 5), (3, 8, 9, 8, 9)],
        dtype=[
            ("bout_id", "i4"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("core_start_frame", "i8"),
            ("core_end_frame", "i8"),
        ],
    )
    write_columnar_dataset(speed_group, "bouts", bouts, {"n_bouts": 3})

    run_name = compute_and_save_bout_kinematics(
        zarr_path,
        run_name="bout_kinematics_interbout",
        track_kinematics_run="tk_1",
        track_id=0,
        swim_bout_run="bouts_1",
        speed_level="filtered",
        heading_levels=("heading_smoothed",),
        pre_post_mode="interbout_epoch",
        include_eye_gaze=False,
        layout=LAYOUT_HIERARCHICAL_V1,
        overwrite=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["bout_kinematics_runs"][run_name]
    assert run.attrs["parameters"]["pre_post_mode"] == "interbout_epoch"
    smoothed = run["heading_smoothed"]["per_bout_metrics"]

    assert smoothed["pre_epoch_start_frame"][:].tolist() == [-1, 3, 6]
    assert smoothed["pre_epoch_end_frame"][:].tolist() == [-1, 3, 7]
    assert smoothed["post_epoch_start_frame"][:].tolist() == [3, 6, -1]
    assert smoothed["post_epoch_end_frame"][:].tolist() == [3, 7, -1]
    assert smoothed["pre_window_valid"][:].tolist() == [False, True, True]
    assert smoothed["post_window_valid"][:].tolist() == [True, True, False]
    assert smoothed["pre_position_valid"][:].tolist() == [False, True, True]
    assert smoothed["post_position_valid"][:].tolist() == [True, True, False]

    np.testing.assert_allclose(smoothed["pre_heading_mean_deg"][1:2], [20.0])
    np.testing.assert_allclose(smoothed["post_heading_mean_deg"][1:2], [30.0])
    np.testing.assert_allclose(smoothed["net_delta_heading_deg"][1:2], [10.0])
    np.testing.assert_allclose(smoothed["pre_position_mean_x_px"][1:2], [3.0])
    np.testing.assert_allclose(smoothed["post_position_mean_x_px"][1:2], [6.5])
    np.testing.assert_allclose(smoothed["interbout_epoch_displacement_px"][1:2], [np.hypot(3.5, 7.0)])
    np.testing.assert_allclose(smoothed["interbout_epoch_displacement_mm"][1:2], [np.hypot(0.35, 0.175)])
