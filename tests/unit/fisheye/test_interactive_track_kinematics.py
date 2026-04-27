from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis import plot_track_kinematics as plot_mod
from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.visualization.interactive_track_kinematics import (
    DEFAULT_INTERACTIVE_ARTIFACT,
    discover_swim_bout_run_options,
    discover_track_kinematics_run_options,
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
) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    swim_parent = root["analysis"].create_group("swim_bout_runs")
    swim_parent.attrs["latest"] = "swim_bout_1"

    run = swim_parent.create_group("swim_bout_1")
    run.attrs["detection_method"] = "threshold"
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
    for level in ("speed_raw", "speed_filtered", "speed_smoothed", "speed_averaged"):
        level_group = run.create_group(level)
        write_columnar_dataset(level_group, "bouts", bouts)
        write_columnar_dataset(level_group, "inter_bout_intervals", intervals)


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
    assert list(positions.columns) == ["time_s", "x", "y", "unit"]
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

    assert data.swim_bout_source == "analysis_swim_bout_runs"
    assert data.swim_bout_label == "swim_bout_1 (speed_smoothed) (threshold)"
    assert swim_bouts["start_s"].tolist() == [0.010, 0.035]
    np.testing.assert_allclose(swim_bouts["duration_s"].to_numpy(), [0.010, 0.010])
    np.testing.assert_allclose(swim_bouts["path_length_mm"].to_numpy(), [1.20, 1.80])
    assert swim_bouts["gap_censored"].tolist() == [False, True]
    assert inter_bout_intervals["interval_s"].tolist() == [0.015]


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

    assert len(swim_options) == 1
    assert swim_options[0].run_name == "swim_bout_1"
    assert swim_options[0].default_level == "speed_smoothed"
    assert swim_options[0].speed_level == "smoothed"
    assert swim_options[0].source_track_kinematics_run == "track_kinematics_1"
    assert swim_options[0].track_id == 0
    assert swim_options[0].n_bouts_by_level["speed_smoothed"] == 2


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
