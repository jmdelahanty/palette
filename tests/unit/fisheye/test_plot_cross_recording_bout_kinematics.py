from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.utils.plot_cross_recording_bout_kinematics import plot_export


def _write_parquet_export(root: Path, export_run_id: str) -> None:
    table_dir = root / "v1" / "bout_kinematics_metrics" / f"export_run_id={export_run_id}"
    table_dir.mkdir(parents=True)
    rows = [
        {
            "recording_id": "rec_a",
            "measurement_level": "heading_smoothed",
            "stimulus_mode": "MOVING_GRATING",
            "net_delta_heading_deg": 10.0,
            "abs_net_delta_heading_deg": 10.0,
            "within_heading_path_deg": 15.0,
            "within_heading_peak_to_peak_deg": 12.0,
            "within_angular_speed_mean_deg_s": 90.0,
            "within_angular_speed_max_deg_s": 250.0,
        },
        {
            "recording_id": "rec_a",
            "measurement_level": "heading_smoothed",
            "stimulus_mode": "MOVING_GRATING",
            "net_delta_heading_deg": -20.0,
            "abs_net_delta_heading_deg": 20.0,
            "within_heading_path_deg": 30.0,
            "within_heading_peak_to_peak_deg": 25.0,
            "within_angular_speed_mean_deg_s": 120.0,
            "within_angular_speed_max_deg_s": 300.0,
        },
        {
            "recording_id": "rec_a",
            "measurement_level": "heading_raw",
            "stimulus_mode": "MOVING_GRATING",
            "net_delta_heading_deg": 99.0,
            "abs_net_delta_heading_deg": 99.0,
            "within_heading_path_deg": 99.0,
            "within_heading_peak_to_peak_deg": 99.0,
            "within_angular_speed_mean_deg_s": 99.0,
            "within_angular_speed_max_deg_s": 99.0,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), table_dir / "part-00000.parquet")
    manifest_dir = root / "v1" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / f"export_run_id={export_run_id}.json").write_text(
        json.dumps({"export_run_id": export_run_id}) + "\n"
    )


def test_plot_cross_recording_bout_kinematics_writes_artifacts(tmp_path: Path) -> None:
    export_root = tmp_path / "exports"
    output_dir = tmp_path / "plots"
    _write_parquet_export(export_root, "run_test")

    summary = plot_export(
        export_root=export_root,
        export_run_id="latest",
        output_dir=output_dir,
        measurement_level="heading_smoothed",
        dpi=60,
    )

    assert summary["export_run_id"] == "run_test"
    assert summary["measurement_level"] == "heading_smoothed"
    assert summary["row_count"] == 2
    assert summary["metrics"]["net_delta_heading_deg"]["n"] == 2
    assert summary["metrics"]["net_delta_heading_deg"]["median"] == -5.0
    assert summary["net_delta_by_mode"]["MOVING_GRATING"]["n"] == 2

    expected = {
        "plot_summary.json",
        "swim_bout_heading_summary.tsv",
        "swim_bout_heading_histograms_overall.png",
        "swim_bout_net_heading_by_stimulus.png",
        "swim_bout_angular_speed_histograms.png",
    }
    assert expected == {path.name for path in output_dir.iterdir()}
    for path in output_dir.iterdir():
        assert path.stat().st_size > 0

    payload = json.loads((output_dir / "plot_summary.json").read_text())
    assert payload["source"] == "parquet:bout_kinematics_metrics"
