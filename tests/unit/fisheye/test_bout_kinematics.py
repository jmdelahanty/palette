from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.bout_kinematics import (
    compute_and_save_bout_kinematics,
    normalize_heading_level,
)
from fisheye.analysis.chaser_state_interpolator import load_structured_dataset, write_columnar_dataset


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


def test_normalize_heading_level_accepts_aliases() -> None:
    assert normalize_heading_level("smoothed") == "heading_smoothed"
    assert normalize_heading_level("heading_raw") == "heading_raw"


def test_normalize_heading_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported heading level"):
        normalize_heading_level("median")


def test_compute_and_save_bout_kinematics_writes_heading_levels(tmp_path: Path) -> None:
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
        write_visualizations=True,
        visualization_bins=8,
        overwrite=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    parent = root["analysis"]["bout_kinematics_runs"]
    run = parent[run_name]

    assert parent.attrs["latest"] == "bout_kinematics_1"
    assert run.attrs["schema_id"] == "analysis.bout_kinematics_runs"
    assert run.attrs["schema_version"] == 5
    assert run.attrs["method_version"] == "bout_kinematics.v5"
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
    assert run.attrs["source_refs"]["zarr_path"] == str(zarr_path)
    assert run.attrs["source_refs"]["source_track_id"] == 0
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
    assert run.attrs["source_refs"]["source_validity_arrays"] == {
        "transition_valid": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/transition_valid",
        "sample_valid": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/sample_valid",
    }
    assert run.attrs["provenance"]["stage"] == "bout_kinematics"
    assert run.attrs["provenance"]["inputs"]["zarr_path"] == str(zarr_path)
    assert run.attrs["provenance"]["inputs"]["source_heading_arrays"] == run.attrs["source_refs"][
        "source_heading_arrays"
    ]

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
    assert png.attrs["parameters"]["bins"] == 8
    spec_artifact = visualizations["bout_kinematics_summary_track_0_interactive"]
    assert spec_artifact.attrs["snapshot_artifact"] == "bout_kinematics_summary_track_0_png"
    spec_payload = np.asarray(spec_artifact["spec_json"][:], dtype=np.uint8).tobytes()
    assert b"net_heading_change_histograms" in spec_payload
    assert b"within_bout_heading_histograms" in spec_payload


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
        overwrite=False,
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
    )

    compute_and_save_bout_kinematics(**kwargs)
    with pytest.raises(ValueError, match="Use --overwrite"):
        compute_and_save_bout_kinematics(**kwargs)
    compute_and_save_bout_kinematics(**kwargs, overwrite=True)


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
