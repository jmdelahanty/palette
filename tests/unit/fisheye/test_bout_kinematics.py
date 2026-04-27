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

    bout_parent = analysis.create_group("swim_bout_runs")
    bout_parent.attrs["latest"] = "bouts_1"
    bout_run = bout_parent.create_group("bouts_1")
    bout_run.attrs["source_track_kinematics_run"] = "tk_1"
    bout_run.attrs["track_id"] = 0
    bout_run.attrs["default_level"] = "speed_filtered"
    speed_group = bout_run.create_group("speed_filtered")
    bouts = np.asarray(
        [(1, 3, 5, 3, 5)],
        dtype=[
            ("bout_id", "i4"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("core_start_frame", "i8"),
            ("core_end_frame", "i8"),
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
    assert run.attrs["default_heading_level"] == "heading_smoothed"
    assert run.attrs["source_swim_bout_run"] == "bouts_1"
    assert run.attrs["source_swim_bout_speed_level"] == "speed_filtered"
    assert run.attrs["parameters"]["resolved_pre_window_frames"] == 2
    assert run.attrs["parameters"]["resolved_post_window_frames"] == 2
    assert run.attrs["source_refs"]["zarr_path"] == str(zarr_path)
    assert run.attrs["source_refs"]["source_track_id"] == 0
    assert run.attrs["source_refs"]["source_heading_arrays"] == {
        "heading_smoothed": (
            "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/"
            "smoothed_heading_degrees"
        ),
        "heading_raw": "analysis/track_kinematics_runs/offline/tk_1/tracks/id_0/heading_degrees",
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
    np.testing.assert_allclose(smoothed["pre_heading_mean_deg"][:], [10.0])
    np.testing.assert_allclose(smoothed["post_heading_mean_deg"][:], [30.0])
    np.testing.assert_allclose(smoothed["net_delta_heading_deg"][:], [20.0])
    np.testing.assert_allclose(smoothed["abs_net_delta_heading_deg"][:], [20.0])
    np.testing.assert_allclose(smoothed["within_heading_range_deg"][:], [20.0])
    np.testing.assert_allclose(smoothed["within_heading_peak_to_peak_deg"][:], [20.0])
    np.testing.assert_allclose(smoothed["within_heading_path_deg"][:], [40.0])
    assert smoothed["within_heading_zero_crossings"][:].tolist() == [1]
    assert smoothed["pre_window_valid"][:].tolist() == [True]
    assert smoothed["post_window_valid"][:].tolist() == [True]
    assert smoothed["within_window_valid"][:].tolist() == [True]
    assert smoothed["dominant_frequency_valid"][:].tolist() == [False]
    assert smoothed_records["failure_reason_bytes"].tolist() == [b"dominant_frequency_disabled"]

    raw = run["heading_raw"]["per_bout_metrics"]
    np.testing.assert_allclose(raw["pre_heading_mean_deg"][:], [0.0])
    np.testing.assert_allclose(raw["post_heading_mean_deg"][:], [40.0])
    np.testing.assert_allclose(raw["net_delta_heading_deg"][:], [40.0])
    np.testing.assert_allclose(raw["within_heading_range_deg"][:], [40.0])
    np.testing.assert_allclose(raw["within_heading_path_deg"][:], [80.0])

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
