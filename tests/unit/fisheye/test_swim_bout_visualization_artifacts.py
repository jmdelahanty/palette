from __future__ import annotations

import numpy as np
import zarr

from fisheye.analysis import swim_bout_visualization as mod
from fisheye.shared.plot_artifacts import PNG_ARTIFACT_SCHEMA_ID


def test_write_swim_bout_visualization_artifact_persists_contract_and_histograms() -> None:
    root = zarr.group()
    run = root.require_group("analysis/swim_bout_runs/bouts_1")
    bouts = np.asarray(
        [
            (0.10, 1.0, 10.0),
            (0.20, 2.0, 20.0),
            (0.30, 3.0, 30.0),
        ],
        dtype=[
            ("duration_s", "f4"),
            ("path_length_mm", "f4"),
            ("mean_speed_mm_s", "f4"),
        ],
    )
    datasets = {
        "bouts": bouts,
        "global_metrics": None,
        "trials": None,
        "bout_points": None,
        "inter_bout_intervals": None,
        "inter_bout_interval_histogram": None,
    }
    attrs = {
        "run_name": "bouts_1 (speed_smoothed)",
        "source_swim_bout_run": "bouts_1",
        "source_swim_bout_path": "analysis/swim_bout_runs/bouts_1/series/smoothed",
        "source_track_kinematics_run": "tk_1",
        "track_id": 0,
    }

    artifact_path = mod.write_swim_bout_visualization_artifact(
        run_group=run,
        run_name="bouts_1",
        attrs=attrs,
        datasets=datasets,
        speed_level="smoothed",
        artifact_dpi=72,
    )
    assert artifact_path == "visualizations/swim_bout_summary_png"
    artifact = run[artifact_path]
    assert bytes(np.asarray(artifact[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert artifact.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert artifact.attrs["visualization_contract_id"] == (
        mod.SWIM_BOUT_SUMMARY_VISUALIZATION_CONTRACT_ID
    )
    assert artifact.attrs["renderer"] == mod.SWIM_BOUT_SUMMARY_RENDERER
    assert artifact.attrs["renderer_version"] == mod.SWIM_BOUT_SUMMARY_RENDERER_VERSION
    assert artifact.attrs["source_runs"]["track_id"] == 0

    histograms = run["report_tables/swim_bout_summary/smoothed/histograms"]
    assert set(histograms.group_keys()) == {"bout_distance", "duration_s", "mean_speed"}
    duration = histograms["duration_s"]
    assert duration.attrs["schema_id"] == mod.SWIM_BOUT_HISTOGRAM_SCHEMA_ID
    assert duration.attrs["bin_count"] == mod.SWIM_BOUT_HISTOGRAM_BIN_COUNT
    assert int(np.sum(duration["count"][:])) == 3
    assert np.isclose(float(np.sum(duration["fraction"][:])), 1.0)

    manifest = run.attrs["visualizations"]
    assert manifest["swim_bout_summary_png"]["visualization_contract_id"] == (
        mod.SWIM_BOUT_SUMMARY_VISUALIZATION_CONTRACT_ID
    )
