from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import zarr

from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.analysis.swim_bout_visualization import (
    _load_swim_bout_run,
    create_swim_bout_dashboard,
)


def _bouts(ids: list[int]) -> np.ndarray:
    records = np.zeros(
        len(ids),
        dtype=[
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("distance_px", "f8"),
            ("mean_speed_px_s", "f8"),
        ],
    )
    records["bout_id"] = ids
    records["start_frame"] = np.arange(len(ids)) * 10
    records["end_frame"] = records["start_frame"] + 5
    records["duration_s"] = 0.1
    records["distance_px"] = np.arange(len(ids), dtype=np.float64) + 1.0
    records["mean_speed_px_s"] = 10.0
    return records


def _modern_bouts(ids: list[int]) -> np.ndarray:
    records = np.zeros(
        len(ids),
        dtype=[
            ("bout_id", "i8"),
            ("start_frame", "i8"),
            ("end_frame", "i8"),
            ("duration_s", "f8"),
            ("path_length_mm", "f8"),
            ("path_length_px", "f8"),
            ("mean_speed_mm_s", "f8"),
        ],
    )
    records["bout_id"] = ids
    records["start_frame"] = np.arange(len(ids)) * 10
    records["end_frame"] = records["start_frame"] + 5
    records["duration_s"] = 0.1
    records["path_length_mm"] = np.arange(len(ids), dtype=np.float64) + 1.0
    records["path_length_px"] = records["path_length_mm"] * 10.0
    records["mean_speed_mm_s"] = 20.0
    return records


def _global_metrics() -> np.ndarray:
    records = np.zeros(
        1,
        dtype=[
            ("total_bouts", "i8"),
            ("inter_bout_interval_count", "i8"),
            ("inter_bout_interval_mean_s", "f8"),
        ],
    )
    records["total_bouts"] = 2
    records["inter_bout_interval_count"] = 1
    records["inter_bout_interval_mean_s"] = 0.25
    return records


def _intervals() -> np.ndarray:
    records = np.zeros(
        1,
        dtype=[
            ("prev_bout_id", "i8"),
            ("next_bout_id", "i8"),
            ("interval_s", "f8"),
        ],
    )
    records["prev_bout_id"] = 0
    records["next_bout_id"] = 1
    records["interval_s"] = 0.25
    return records


def _interval_histogram() -> np.ndarray:
    records = np.zeros(
        1,
        dtype=[
            ("bin_left_edge_s", "f8"),
            ("bin_right_edge_s", "f8"),
            ("count", "i8"),
        ],
    )
    records["bin_left_edge_s"] = 0.2
    records["bin_right_edge_s"] = 0.3
    records["count"] = 1
    return records


def _trials() -> np.ndarray:
    records = np.zeros(
        1,
        dtype=[
            ("trial_id", "i8"),
            ("bout_rate_per_min", "f8"),
            ("percent_active", "f8"),
        ],
    )
    records["trial_id"] = 0
    records["bout_rate_per_min"] = 12.0
    records["percent_active"] = 34.0
    return records


def _make_archive(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_canary"
    run = parent.create_group("bouts_canary")
    run.attrs.update(
        {
            "default_level": "speed_exponential",
            "detection_method": "peak_event",
            "source_track_kinematics_run": "tk_hyst4_low2_s005",
            "track_id": 0,
        }
    )
    filtered = run.create_group("speed_filtered")
    filtered.attrs.update({"n_bouts": 1, "path_distance_source_level": "filtered"})
    write_columnar_dataset(filtered, "bouts", _modern_bouts([10]))

    exponential = run.create_group("speed_exponential")
    exponential.attrs.update(
        {
            "n_bouts": 2,
            "detection_signal_transform_type": "exponential",
            "detection_signal_source_level": "filtered",
        }
    )
    write_columnar_dataset(exponential, "bouts", _modern_bouts([0, 1]))
    write_columnar_dataset(exponential, "global_metrics", _global_metrics())
    write_columnar_dataset(exponential, "inter_bout_intervals", _intervals())
    write_columnar_dataset(exponential, "inter_bout_interval_histogram", _interval_histogram())
    return root


def test_visualization_loader_uses_resolver_for_exponential_speed(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _make_archive(zarr_path)

    attrs, datasets = _load_swim_bout_run(zarr_path, None, speed_level="exponential")

    assert attrs["run_name"] == "bouts_canary (speed_exponential)"
    assert attrs["speed_level"] == "speed_exponential"
    assert attrs["is_hierarchical"] is True
    assert attrs["source_swim_bout_run"] == "bouts_canary"
    assert attrs["source_swim_bout_signal_id"] == 1
    assert attrs["available_speed_levels"] == ["speed_filtered", "speed_exponential"]
    assert datasets["bouts"]["bout_id"].tolist() == [0, 1]
    assert datasets["inter_bout_interval_histogram"]["count"].tolist() == [1]
    assert datasets["trials"] is None


def test_visualization_loader_falls_back_to_default_speed(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _make_archive(zarr_path)

    attrs, datasets = _load_swim_bout_run(zarr_path, "bouts_canary", speed_level="smoothed")

    assert attrs["speed_level"] == "speed_exponential"
    assert datasets["bouts"]["bout_id"].tolist() == [0, 1]


def test_visualization_loader_reads_flat_legacy_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "legacy.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "flat_run"
    run = parent.create_group("flat_run")
    write_columnar_dataset(run, "bouts", _bouts([7]))
    write_columnar_dataset(run, "global_metrics", _global_metrics())
    write_columnar_dataset(run, "trials", _trials())
    write_columnar_dataset(run, "inter_bout_interval_histogram", _interval_histogram())

    attrs, datasets = _load_swim_bout_run(zarr_path, None, speed_level="filtered")

    assert attrs["run_name"] == "flat_run"
    assert attrs["speed_level"] == ""
    assert attrs["is_hierarchical"] is False
    assert datasets["bouts"]["bout_id"].tolist() == [7]
    assert datasets["trials"]["trial_id"].tolist() == [0]
    assert datasets["inter_bout_interval_histogram"]["count"].tolist() == [1]


def test_dashboard_accepts_modern_bout_fields(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _make_archive(zarr_path)
    attrs, datasets = _load_swim_bout_run(zarr_path, None, speed_level="exponential")

    fig = create_swim_bout_dashboard(attrs, datasets, title="test")

    assert fig.axes
    plt.close(fig)
