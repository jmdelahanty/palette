from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from fisheye.utils.export_cross_recording_analytics import export_sources


def _array(group, name: str, values) -> None:
    group.create_array(name, data=np.asarray(values), overwrite=True)


def _make_source_zarr(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w")
    analysis = root.create_group("analysis")

    stim_parent = analysis.create_group("stimulus_runs")
    stim_parent.attrs["latest"] = "stimulus_test"
    stim = stim_parent.create_group("stimulus_test")
    steps = stim.create_group("steps")
    step0 = steps.create_group("step_0")
    step0.attrs.update(
        {
            "step_index": 0,
            "step_name": "Moving Grating",
            "stimulus_mode": "MOVING_GRATING",
            "stimulus_mode_id": 3,
            "start_frame": 10,
            "end_frame": 110,
            "duration_s": 1.6667,
            "stimulus_params": {"direction_degrees": 0.0},
        }
    )
    moving = step0.create_group("moving_grating")
    moving.attrs["direction_degrees"] = 0.0
    moving.attrs["direction_mapping_status"] = "validated"

    step1 = steps.create_group("step_1")
    step1.attrs.update(
        {
            "step_index": 1,
            "step_name": "Concentric",
            "stimulus_mode": "CONCENTRIC_GRATING",
            "stimulus_mode_id": 6,
            "start_frame": 110,
            "end_frame": 210,
            "duration_s": 1.6667,
            "stimulus_params": {"is_expanding": False},
        }
    )
    concentric = step1.create_group("concentric_grating")
    concentric.attrs["stimulus_radial_polarity_authored"] = "contracting"

    resp_parent = analysis.create_group("stimulus_response_runs")
    resp_parent.attrs["latest"] = "stimulus_response_test"
    resp = resp_parent.create_group("stimulus_response_test")
    resp.attrs.update(
        {
            "source_stimulus_run": "stimulus_test",
            "source_track_kinematics_run": "tk_test",
            "source_track_kinematics_type": "offline",
            "source_bout_run": "bouts_test",
            "n_fish": 1,
            "n_steps": 2,
        }
    )
    global_group = resp.create_group("global")
    _array(global_group, "fish_id", [0])
    _array(global_group, "total_distance_mm", [25.0])
    _array(global_group, "mean_speed_mm_s", [3.5])
    _array(global_group, "total_active_s", [4.0])
    _array(global_group, "fraction_moving", [0.25])

    resp_steps = resp.create_group("steps")
    resp_step0 = resp_steps.create_group("step_0")
    resp_step0.attrs.update(dict(step0.attrs))
    pf0 = resp_step0.create_group("per_fish")
    _array(pf0, "fish_id", [0])
    _array(pf0, "total_distance_mm", [12.5])
    _array(pf0, "mean_speed_mm_s", [5.0])
    _array(pf0, "num_bouts", [2])
    grating = resp_step0.create_group("grating")
    omr = grating.create_group("omr")
    omr.attrs["method_version"] = "omr.v1"
    omr_pf = omr.create_group("per_fish")
    _array(omr_pf, "fish_id", [0])
    _array(omr_pf, "omr_path_index", [0.75])
    _array(omr_pf, "first_aligned_bout_latency_s", [np.nan])

    resp_step1 = resp_steps.create_group("step_1")
    resp_step1.attrs.update(dict(step1.attrs))
    pf1 = resp_step1.create_group("per_fish")
    _array(pf1, "fish_id", [0])
    _array(pf1, "total_distance_mm", [10.0])
    _array(pf1, "mean_speed_mm_s", [4.0])
    _array(pf1, "num_bouts", [1])
    conc = resp_step1.create_group("concentric_grating")
    radial = conc.create_group("radial_omr")
    radial.attrs["method_version"] = "radial.v1"
    radial_pf = radial.create_group("per_fish")
    _array(radial_pf, "fish_id", [0])
    _array(radial_pf, "omr_path_index", [0.5])
    _array(radial_pf, "radial_path_index", [-0.5])
    _array(radial_pf, "first_aligned_bout_latency_s", [0.2])

    swim_parent = analysis.create_group("swim_bout_runs")
    swim_parent.attrs["latest"] = "bouts_test"
    swim = swim_parent.create_group("bouts_test")
    swim.attrs.update(
        {
            "default_level": "speed_exponential",
            "track_id": 0,
            "source_track_kinematics_run": "tk_test",
            "source_track_kinematics_type": "offline",
        }
    )
    level = swim.create_group("speed_exponential")
    level.attrs.update(
        {
            "n_bouts": 2,
            "mean_bout_duration_s": 0.12,
            "total_path_length_mm": 8.0,
            "detection_method": "peak_event",
            "detection_signal_transform_type": "causal_exponential",
            "movement_metric_source_level": "filtered",
            "peak_prominence_mm_s": 4.0,
        }
    )
    bouts = level.create_group("bouts")
    _array(bouts, "bout_id", [0, 1])
    _array(bouts, "start_frame", [20, 140])
    _array(bouts, "end_frame", [30, 150])
    _array(bouts, "start_time_s", [0.3333, 2.3333])
    _array(bouts, "end_time_s", [0.5, 2.5])
    _array(bouts, "duration_s", [0.1667, 0.1667])
    _array(bouts, "path_length_mm", [3.0, 5.0])
    _array(bouts, "net_displacement_mm", [2.0, 4.0])
    _array(bouts, "mean_speed_mm_s", [18.0, 30.0])
    _array(bouts, "peak_physical_speed_mm_s", [25.0, 45.0])

    return path


def _read_dataset(output_root: Path, table: str, export_run_id: str):
    table_dir = output_root / "v1" / table / f"export_run_id={export_run_id}"
    files = sorted(table_dir.glob("*.parquet"))
    assert files, f"no parquet files for {table}"
    return pq.read_table(files).to_pylist()


def test_export_cross_recording_analytics_writes_first_tables(tmp_path: Path) -> None:
    source = _make_source_zarr(tmp_path / "recording_a_analysis.zarr")
    output = tmp_path / "exports" / "palette_analytics"

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="test_export",
        jobs=1,
    )

    assert manifest["row_counts_by_table"]["recording_summary"] == 1
    assert manifest["row_counts_by_table"]["stimulus_steps"] == 2
    assert manifest["row_counts_by_table"]["stimulus_step_summary"] == 2
    assert manifest["row_counts_by_table"]["stimulus_response_per_fish_step"] == 2
    assert manifest["row_counts_by_table"]["swim_bout_metrics"] == 2

    manifest_path = output / "v1" / "manifests" / "export_run_id=test_export.json"
    payload = json.loads(manifest_path.read_text())
    assert payload["source_recording_count"] == 1
    assert payload["row_counts_by_table"]["swim_bout_metrics"] == 2

    response_rows = _read_dataset(output, "stimulus_response_per_fish_step", "test_export")
    moving = next(row for row in response_rows if row["stimulus_mode"] == "MOVING_GRATING")
    assert moving["omr_family"] == "moving_grating_omr"
    assert moving["omr_path_index"] == 0.75
    assert moving["first_aligned_bout_latency_s"] is None

    radial = next(row for row in response_rows if row["stimulus_mode"] == "CONCENTRIC_GRATING")
    assert radial["omr_family"] == "concentric_radial_omr"
    assert radial["first_aligned_bout_latency_s"] == 0.2
    assert radial["radial_path_index"] == -0.5

    bout_rows = _read_dataset(output, "swim_bout_metrics", "test_export")
    assert bout_rows[0]["step_index"] == 0
    assert bout_rows[1]["step_index"] == 1
    assert bout_rows[0]["speed_level"] == "speed_exponential"


def test_export_cross_recording_analytics_can_limit_tables(tmp_path: Path) -> None:
    source = _make_source_zarr(tmp_path / "recording_b_analysis.zarr")
    output = tmp_path / "exports" / "palette_analytics"

    manifest = export_sources(
        [source],
        output_root=output,
        export_run_id="summary_only",
        tables=("recording_summary",),
        jobs=1,
    )

    assert manifest["row_counts_by_table"] == {"recording_summary": 1}
    assert (output / "v1" / "recording_summary" / "export_run_id=summary_only").is_dir()
    assert not (output / "v1" / "swim_bout_metrics").exists()
