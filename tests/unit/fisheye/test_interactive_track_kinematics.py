from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis import plot_track_kinematics as plot_mod
from fisheye.analysis.chaser_state_interpolator import store_array, write_columnar_dataset
from fisheye.visualization.interactive_track_kinematics import (
    DEFAULT_INTERACTIVE_ARTIFACT,
    bout_classification_records_to_dataframe,
    discover_bout_classification_run_options,
    discover_eye_angle_run_options,
    discover_swim_bout_run_options,
    discover_track_kinematics_run_options,
    load_bout_classification_records,
    load_eye_angle_timeseries_data,
    load_track_kinematics_interactive_data,
    to_inter_bout_interval_dataframe,
    to_position_dataframe,
    to_swim_bout_dataframe,
    to_timeseries_dataframe,
    to_validity_span_dataframe,
)
from tests.unit.fisheye.test_plot_track_kinematics_artifacts import _make_track_kinematics_archive


def _make_archive_with_interactive_artifact(tmp_path: Path) -> Path:
    zarr_path = _make_track_kinematics_archive(tmp_path)
    plot_mod.main(
        [
            str(zarr_path),
            "--offline-only",
            "--track-id",
            "0",
            "--swim-bout-run",
            "none",
            "--bins",
            "8",
            "--write-zarr-artifacts",
        ]
    )
    return zarr_path


def _add_hierarchical_swim_bouts(
    zarr_path: Path,
    *,
    source_track_kinematics_run: str = "track_kinematics_1",
    track_id: int = 0,
    detection_method: str = "threshold",
    include_peak_events: bool = False,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    swim_parent = root["analysis"].create_group("swim_bout_runs")
    swim_parent.attrs["latest"] = "swim_bout_1"

    run = swim_parent.create_group("swim_bout_1")
    run.attrs["detection_method"] = detection_method
    run.attrs["default_level"] = "speed_smoothed"
    run.attrs["fps"] = 200.0
    run.attrs["source_track_kinematics_run"] = source_track_kinematics_run
    run.attrs["track_id"] = track_id

    bouts = np.asarray(
        [
            (1, 0.010, 0.020, 0.010, 0.009, 1.20, 0.90, False),
            (2, 0.035, 0.045, 0.010, 0.010, 1.80, 1.10, True),
        ],
        dtype=[
            ("bout_id", "i4"),
            ("start_time_s", "f8"),
            ("end_time_s", "f8"),
            ("duration_s", "f8"),
            ("observed_duration_s", "f8"),
            ("path_length_mm", "f8"),
            ("net_displacement_mm", "f8"),
            ("gap_censored", "?"),
        ],
    )
    intervals = np.asarray(
        [(1, 2, 0.020, 0.035, 0.015)],
        dtype=[
            ("prev_bout_id", "i4"),
            ("next_bout_id", "i4"),
            ("prev_end_time_s", "f8"),
            ("next_start_time_s", "f8"),
            ("interval_s", "f8"),
        ],
    )
    peak_events = np.asarray(
        [
            (1, 2, 0.0125, 4.0, 3.0, 1.5, 4.5, b"relative_prominence_width", b"none"),
            (2, 8, 0.0400, 5.0, 4.0, 6.5, 9.5, b"relative_prominence_width", b"none"),
        ],
        dtype=[
            ("bout_id", "i4"),
            ("peak_frame", "i8"),
            ("peak_time_s", "f8"),
            ("peak_signal_value_mm_s", "f8"),
            ("peak_prominence_mm_s", "f8"),
            ("left_width_frame_interpolated", "f8"),
            ("right_width_frame_interpolated", "f8"),
            ("boundary_mode", "S32"),
            ("shape_split_policy", "S32"),
        ],
    )
    for level in ("speed_raw", "speed_filtered", "speed_smoothed", "speed_averaged"):
        level_group = run.create_group(level)
        write_columnar_dataset(level_group, "bouts", bouts)
        write_columnar_dataset(level_group, "inter_bout_intervals", intervals)
        if include_peak_events:
            write_columnar_dataset(level_group, "peak_events", peak_events)


def _add_compact_swim_bouts(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    analysis = root["analysis"]
    swim_parent = analysis.create_group("swim_bout_runs")
    swim_parent.attrs["latest"] = "swim_bout_compact"

    run = swim_parent.create_group("swim_bout_compact")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 7,
            "layout": "compact_tabular_v2",
            "source_track_kinematics_run": "track_kinematics_1",
            "track_id": 0,
            "detection_method": "peak_event",
            "default_candidate_id": 0,
            "default_signal_id": 1,
            "exponential_tau_s": 0.025,
        }
    )
    indexes = run.create_group("indexes")
    tables = run.create_group("tables")
    signals = run.create_group("signals")

    candidates = np.zeros(
        1,
        dtype=[
            ("candidate_id", "i4"),
            ("candidate_name", "S32"),
            ("is_default", "?"),
            ("detection_method", "S32"),
        ],
    )
    candidates[0] = (0, b"compact_candidate", True, b"peak_event")
    write_columnar_dataset(indexes, "candidates", candidates)

    signal_variants = np.zeros(
        2,
        dtype=[
            ("signal_id", "i4"),
            ("speed_level", "S32"),
            ("signal_name", "S32"),
            ("role", "S32"),
            ("source_level", "S32"),
            ("transform_type", "S32"),
            ("transform_source_signal_id", "i4"),
            ("tau_s", "f8"),
            ("units", "S16"),
            ("path_distance_source_level", "S32"),
        ],
    )
    signal_variants[0] = (
        0,
        b"speed_filtered",
        b"filtered",
        b"physical_estimator",
        b"speed_filtered",
        b"identity",
        -1,
        np.nan,
        b"mm/s",
        b"filtered",
    )
    signal_variants[1] = (
        1,
        b"speed_exponential",
        b"exponential",
        b"detector_response",
        b"speed_filtered",
        b"exponential",
        0,
        0.025,
        b"mm/s",
        b"filtered",
    )
    write_columnar_dataset(indexes, "signal_variants", signal_variants)

    bouts = np.zeros(
        3,
        dtype=[
            ("candidate_id", "i4"),
            ("signal_id", "i4"),
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
        ],
    )
    bouts[0] = (0, 0, 10, 1, 3, 0.01)
    bouts[1] = (0, 1, 20, 4, 8, 0.02)
    bouts[2] = (0, 1, 21, 10, 16, 0.03)
    write_columnar_dataset(tables, "bouts", bouts)
    store_array(signals, "detector_signal_mm_s", np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32))
    store_array(signals, "detector_signal_signal_ids", np.asarray([1], dtype=np.int32))
    store_array(signals, "frame_indices", np.asarray([4, 5, 6], dtype=np.int64))


def _add_eye_angle_run(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    analysis = root["analysis"] if "analysis" in root else root.create_group("analysis")
    parent = analysis.create_group("eye_angle_runs")
    parent.attrs["latest"] = "eye_angle_1"
    run = parent.create_group("eye_angle_1")
    run.attrs.update(
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 4,
            "preferred_angle_family": "gaze",
            "preferred_eye_axis": "ellipse_minor",
            "row_axis": "keypoint_detection_rows",
            "fps": 200.0,
            "eye_angle_variant_schema": {
                "schema_id": "analysis.eye_angle_variant_schema",
                "schema_version": 1,
                "default_representation": "eye_frame",
                "representation_order": ["eye_frame", "gaze", "nasal_gaze"],
            },
        }
    )
    angles = run.create_group("angles")
    roi = angles.create_group("roi")
    frame = angles.create_group("frame")
    qa = run.create_group("qa")
    qa_roi = qa.create_group("roi")
    qa_frame = qa.create_group("frame")
    support = run.create_group("support")

    support.create_array("time_seconds", data=np.asarray([0.0, 0.005, 0.010], dtype=np.float64))
    support.create_array("frame_indices", data=np.asarray([0, 1, 2], dtype=np.int64))
    support.create_array("frame_time_seconds", data=np.asarray([0.0, 0.005, 0.010, 0.015], dtype=np.float64))

    roi.create_array("left_minor_signed_deg", data=np.asarray([-20.0, -10.0, -5.0], dtype=np.float32))
    roi.create_array("left_minor_signed_deg_smoothed", data=np.asarray([-18.0, -12.0, -6.0], dtype=np.float32))
    roi.create_array("left_eye_angle_deg", data=np.asarray([20.0, 10.0, 5.0], dtype=np.float32))
    roi.create_array("left_eye_angle_deg_smoothed", data=np.asarray([18.0, 12.0, 6.0], dtype=np.float32))
    roi.create_array("right_eye_angle_deg", data=np.asarray([15.0, 12.0, 10.0], dtype=np.float32))
    roi.create_array("right_eye_angle_deg_smoothed", data=np.asarray([14.0, 11.0, 9.0], dtype=np.float32))
    roi.create_array("vergence_eye_angle_deg", data=np.asarray([35.0, 22.0, 15.0], dtype=np.float32))
    roi.create_array("vergence_eye_angle_deg_smoothed", data=np.asarray([32.0, 23.0, 15.0], dtype=np.float32))
    roi.create_array("left_gaze_signed_deg", data=np.asarray([-20.0, -10.0, -5.0], dtype=np.float32))
    roi.create_array("left_nasal_gaze_deg_smoothed", data=np.asarray([70.0, 80.0, 85.0], dtype=np.float32))
    roi.create_array("mean_eye_vergence_gaze_deg_smoothed", data=np.asarray([30.0, 31.0, 32.0], dtype=np.float32))

    frame.create_array("left_eye_angle_deg_smoothed", data=np.asarray([18.0, 12.0, 6.0, 4.0], dtype=np.float32))
    frame.create_array("right_eye_angle_deg_smoothed", data=np.asarray([14.0, 11.0, 9.0, 7.0], dtype=np.float32))
    frame.create_array("vergence_eye_angle_deg_smoothed", data=np.asarray([32.0, 23.0, 15.0, 11.0], dtype=np.float32))
    frame.create_array("left_gaze_signed_deg_smoothed", data=np.asarray([-18.0, -12.0, -6.0, -4.0], dtype=np.float32))
    frame.create_array("right_gaze_signed_deg_smoothed", data=np.asarray([12.0, 10.0, 8.0, 6.0], dtype=np.float32))
    frame.create_array("mean_eye_vergence_gaze_deg_smoothed", data=np.asarray([30.0, 31.0, 32.0, 33.0], dtype=np.float32))
    frame.create_array("left_nasal_gaze_deg_smoothed", data=np.asarray([72.0, 78.0, 84.0, 86.0], dtype=np.float32))

    qa_roi.create_array("valid_frame", data=np.asarray([True, True, False], dtype=bool))
    qa_frame.create_array("valid_frame", data=np.asarray([True, True, True, False], dtype=bool))


def _add_bout_classification_run(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    analysis = root["analysis"] if "analysis" in root else root.create_group("analysis")
    parent = analysis.create_group("bout_classification_runs")
    parent.attrs["latest"] = "classifier_1"
    run = parent.create_group("classifier_1")
    run.attrs.update(
        {
            "schema_id": "analysis.bout_classification_runs",
            "schema_version": 1,
            "classifier_family": "megabouts",
            "classifier_name": "megabouts_transformer",
            "classifier_version": "test",
            "source_mode": "palette_bouts",
            "row_axis": "swim_bout_rows",
            "invalid_window_policy": "skip_invalid_windows",
            "source_bout_count": 2,
            "classified_bout_count": 1,
            "source_refs": {
                "swim_bout_level": "analysis/swim_bout_runs/swim_bout_1/speed_smoothed",
            },
            "parameters": {
                "swim_bout_run": "swim_bout_1",
                "speed_level": "speed_smoothed",
            },
        }
    )
    records = np.asarray(
        [
            (
                1,
                2,
                4,
                2,
                14,
                5,
                3,
                0,
                b"approach_swim",
                0,
                1,
                0.8,
                1.0,
                1.0,
                0,
                0,
                True,
                True,
                True,
                b"ok",
            ),
            (
                2,
                7,
                9,
                7,
                19,
                -1,
                -1,
                -1,
                b"skipped_invalid_window",
                -1,
                0,
                np.nan,
                0.3,
                1.0,
                8,
                0,
                False,
                False,
                False,
                b"tail_valid_fraction_below_threshold",
            ),
        ],
        dtype=[
            ("source_bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("window_start_frame", "i8"),
            ("window_end_frame", "i8"),
            ("HB1_frame", "i8"),
            ("HB1_offset_frames", "i8"),
            ("category_id", "i4"),
            ("category_label_bytes", "S64"),
            ("subcategory_id", "i4"),
            ("sign", "i4"),
            ("probability", "f4"),
            ("tail_valid_fraction", "f4"),
            ("traj_valid_fraction", "f4"),
            ("max_consecutive_tail_invalid", "i4"),
            ("max_consecutive_traj_invalid", "i4"),
            ("source_window_valid", "?"),
            ("classified", "?"),
            ("valid", "?"),
            ("failure_reason_bytes", "S64"),
        ],
    )
    write_columnar_dataset(run, "per_bout", records)


def test_load_track_kinematics_interactive_data_reads_spec_and_arrays(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)

    data = load_track_kinematics_interactive_data(
        zarr_path,
        run_path="analysis/track_kinematics_runs/offline/track_kinematics_1",
        artifact_name=DEFAULT_INTERACTIVE_ARTIFACT,
    )

    assert data.spec["schema_id"] == plot_mod.TRACK_KINEMATICS_PLOT_SPEC_SCHEMA_ID
    assert data.time_seconds.shape == (6,)
    assert data.frame_indices is not None
    assert data.positions is not None
    assert data.position_unit == "mm"
    assert "speed_smoothed_mm" in data.series
    assert "smoothed_acceleration_mm" in data.series
    assert "speed_filtered_acceleration_mm" in data.series
    assert "speed_filtered_smoothed_acceleration_mm" in data.series
    assert "angular_speed_smoothed_deg_s" in data.series
    np.testing.assert_allclose(data.series["smoothed_acceleration_mm"], np.full(6, 0.35))
    np.testing.assert_allclose(data.series["speed_filtered_acceleration_mm"], np.full(6, 0.2))
    np.testing.assert_allclose(data.series["speed_filtered_smoothed_acceleration_mm"], np.full(6, 0.25))
    assert data.source_paths["time_seconds"].endswith("/tracks/id_0/time_seconds")
    assert data.validity_source == "track_validity"
    assert data.validity_labels.tolist() == ["transition:frame_gap", "sample:keypoint_failed"]
    np.testing.assert_allclose(data.validity_spans, [[0.005, 0.010], [0.0125, 0.0175]])


def test_track_kinematics_interactive_dataframes(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)

    data = load_track_kinematics_interactive_data(
        zarr_path,
        run_path="analysis/track_kinematics_runs/offline/track_kinematics_1",
    )

    timeseries = to_timeseries_dataframe(data)
    positions = to_position_dataframe(data)
    validity = to_validity_span_dataframe(data)

    assert list(timeseries["frame_index"]) == [0, 1, 2, 3, 4, 5]
    assert "speed_smoothed_mm" in timeseries.columns
    assert "smoothed_acceleration_mm" in timeseries.columns
    assert "angular_velocity_smoothed_deg_s" in timeseries.columns
    assert "angular_speed_smoothed_deg_s" in timeseries.columns
    assert list(positions.columns) == ["time_s", "frame_index", "x", "y", "unit"]
    assert list(positions["frame_index"]) == [0, 1, 2, 3, 4, 5]
    assert positions["unit"].unique().tolist() == ["mm"]
    assert validity["reason"].tolist() == ["transition:frame_gap", "sample:keypoint_failed"]
    np.testing.assert_allclose(validity["duration_s"].to_numpy(), [0.005, 0.005])


def test_load_track_kinematics_interactive_data_reads_canonical_swim_bouts(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_hierarchical_swim_bouts(zarr_path)

    data = load_track_kinematics_interactive_data(
        zarr_path,
        run_path="analysis/track_kinematics_runs/offline/track_kinematics_1",
        swim_bout_run="latest",
        speed_level="smoothed",
    )
    swim_bouts = to_swim_bout_dataframe(data)
    inter_bout_intervals = to_inter_bout_interval_dataframe(data)

    assert data.swim_bout_source == "analysis/swim_bout_runs/swim_bout_1/speed_smoothed"
    assert data.swim_bout_label == "swim_bout_1 (speed_smoothed) (threshold)"
    assert swim_bouts["start_s"].tolist() == [0.010, 0.035]
    np.testing.assert_allclose(swim_bouts["duration_s"].to_numpy(), [0.010, 0.010])
    np.testing.assert_allclose(swim_bouts["path_length_mm"].to_numpy(), [1.20, 1.80])
    assert swim_bouts["gap_censored"].tolist() == [False, True]
    assert inter_bout_intervals["interval_s"].tolist() == [0.015]


def test_load_track_kinematics_interactive_data_merges_aligned_peak_event_boundaries(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_hierarchical_swim_bouts(
        zarr_path,
        detection_method="peak_event",
        include_peak_events=True,
    )

    data = load_track_kinematics_interactive_data(
        zarr_path,
        run_path="analysis/track_kinematics_runs/offline/track_kinematics_1",
        swim_bout_run="latest",
        speed_level="smoothed",
    )
    swim_bouts = to_swim_bout_dataframe(data)

    assert data.swim_bout_label == "swim_bout_1 (speed_smoothed) (peak_event)"
    assert swim_bouts["peak_event_boundary_mode"].tolist() == [
        "relative_prominence_width",
        "relative_prominence_width",
    ]
    np.testing.assert_allclose(swim_bouts["peak_event_peak_prominence_mm_s"].to_numpy(), [3.0, 4.0])
    np.testing.assert_allclose(swim_bouts["peak_event_left_width_time_s"].to_numpy(), [0.0075, 0.0325])
    np.testing.assert_allclose(swim_bouts["peak_event_right_width_time_s"].to_numpy(), [0.0225, 0.0475])


def test_discover_track_and_derived_swim_bout_options(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_hierarchical_swim_bouts(zarr_path)

    track_options = discover_track_kinematics_run_options(zarr_path)

    assert len(track_options) == 1
    assert track_options[0].run_name == "track_kinematics_1"
    assert track_options[0].run_path == "analysis/track_kinematics_runs/offline/track_kinematics_1"
    assert track_options[0].track_id == 0

    swim_options = discover_swim_bout_run_options(
        zarr_path,
        track_run_path=track_options[0].run_path,
        track_id=track_options[0].track_id,
    )

    assert len(swim_options) == 4
    assert [option.speed_level for option in swim_options] == [
        "smoothed",
        "filtered",
        "raw",
        "averaged",
    ]
    assert swim_options[0].run_name == "swim_bout_1"
    assert swim_options[0].layout == "hierarchical_v1"
    assert swim_options[0].candidate_id == 0
    assert swim_options[0].signal_id == 1
    assert swim_options[0].signal_role == "physical_estimator"
    assert swim_options[0].default_level == "speed_smoothed"
    assert swim_options[0].source_track_kinematics_run == "track_kinematics_1"
    assert swim_options[0].track_id == 0
    assert swim_options[0].n_bouts_by_level["speed_smoothed"] == 2
    assert "smoothed" in swim_options[0].label
    assert "default" in swim_options[0].label


def test_discover_compact_swim_bout_options_exposes_logical_identity(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_compact_swim_bouts(zarr_path)

    track_options = discover_track_kinematics_run_options(zarr_path)
    swim_options = discover_swim_bout_run_options(
        zarr_path,
        track_run_path=track_options[0].run_path,
        track_id=track_options[0].track_id,
    )

    assert len(swim_options) == 2
    assert [option.speed_level for option in swim_options] == ["exponential", "filtered"]
    assert swim_options[0].layout == "compact_tabular_v2"
    assert swim_options[0].candidate_id == 0
    assert swim_options[0].signal_id == 1
    assert swim_options[0].signal_role == "detector_response"
    assert swim_options[0].default_level == "speed_exponential"
    assert swim_options[0].n_bouts_by_level["speed_exponential"] == 2
    assert swim_options[1].signal_id == 0
    assert swim_options[1].signal_role == "physical_estimator"


def test_discover_and_load_bout_classification_options(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_hierarchical_swim_bouts(zarr_path)
    _add_bout_classification_run(zarr_path)

    options = discover_bout_classification_run_options(
        zarr_path,
        swim_bout_run="swim_bout_1",
        speed_level="smoothed",
    )

    assert len(options) == 1
    assert options[0].run_name == "classifier_1"
    assert options[0].classifier_family == "megabouts"
    assert options[0].source_swim_bout_run == "swim_bout_1"
    assert options[0].source_swim_bout_speed_level == "speed_smoothed"
    assert options[0].source_bout_count == 2
    assert options[0].classified_bout_count == 1
    assert options[0].skipped_bout_count == 1
    assert options[0].is_latest is True

    records, attrs = load_bout_classification_records(zarr_path, run_name="latest")
    frame = bout_classification_records_to_dataframe(records)

    assert attrs["classifier_name"] == "megabouts_transformer"
    assert frame["category_label"].tolist() == ["approach_swim", "skipped_invalid_window"]
    assert frame["failure_reason"].tolist() == ["ok", "tail_valid_fraction_below_threshold"]


def test_discover_and_load_eye_angle_timeseries(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_eye_angle_run(zarr_path)

    options = discover_eye_angle_run_options(zarr_path)

    assert len(options) == 1
    assert options[0].run_name == "eye_angle_1"
    assert options[0].run_path == "analysis/eye_angle_runs/eye_angle_1"
    assert options[0].schema_version == 4
    assert options[0].preferred_angle_family == "gaze"
    assert options[0].is_latest is True

    frame_data = load_eye_angle_timeseries_data(zarr_path, run_name="latest")

    assert frame_data.run_name == "eye_angle_1"
    assert frame_data.row_axis == "frame"
    assert frame_data.dataframe["time_s"].tolist() == [0.0, 0.005, 0.010, 0.015]
    assert "vergence_eye_angle_deg_smoothed" in frame_data.dataframe
    assert "left_eye_angle_deg_smoothed" in frame_data.dataframe
    assert "mean_eye_vergence_gaze_deg_smoothed" in frame_data.dataframe
    assert "left_gaze_signed_deg_smoothed" in frame_data.dataframe
    assert "left_minor_signed_deg" not in frame_data.dataframe
    np.testing.assert_allclose(frame_data.dataframe["vergence_eye_angle_deg_smoothed"], [32.0, 23.0, 15.0, 11.0])
    assert frame_data.dataframe["valid_frame"].tolist() == [True, True, True, False]

    roi_data = load_eye_angle_timeseries_data(
        zarr_path,
        run_name="analysis/eye_angle_runs/eye_angle_1",
        prefer_frame=False,
    )

    assert roi_data.row_axis == "roi"
    assert roi_data.dataframe["frame_index"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(roi_data.dataframe["left_eye_angle_deg"], [20.0, 10.0, 5.0])
    np.testing.assert_allclose(roi_data.dataframe["vergence_eye_angle_deg"], [35.0, 22.0, 15.0])
    np.testing.assert_allclose(roi_data.dataframe["left_minor_signed_deg"], [-20.0, -10.0, -5.0])
    assert roi_data.dataframe["valid_frame"].tolist() == [True, True, False]


def test_load_track_kinematics_interactive_data_skips_mismatched_swim_bout_run(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    _add_hierarchical_swim_bouts(zarr_path, track_id=1)

    data = load_track_kinematics_interactive_data(
        zarr_path,
        run_path="analysis/track_kinematics_runs/offline/track_kinematics_1",
        swim_bout_run="latest",
        speed_level="smoothed",
    )

    assert data.swim_bout_source is None
    assert to_swim_bout_dataframe(data).empty
    assert to_inter_bout_interval_dataframe(data).empty


def test_load_track_kinematics_interactive_data_rejects_wrong_schema(tmp_path: Path) -> None:
    zarr_path = _make_archive_with_interactive_artifact(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    artifact = root[
        "analysis/track_kinematics_runs/offline/track_kinematics_1/visualizations/"
        "track_kinematics_summary_track_0_interactive"
    ]
    del artifact["spec_json"]
    payload = b'{"schema_id":"wrong"}'
    artifact.create_array(
        "spec_json",
        data=np.frombuffer(payload, dtype=np.uint8),
        chunks=(len(payload),),
        overwrite=True,
    )

    try:
        load_track_kinematics_interactive_data(
            zarr_path,
            run_path="analysis/track_kinematics_runs/offline/track_kinematics_1",
        )
    except ValueError as exc:
        assert "Unsupported interactive spec schema" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
