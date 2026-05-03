from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.plot_artifacts import (
    INTERACTIVE_SPEC_SCHEMA_ID,
    PNG_ARTIFACT_SCHEMA_ID,
    SPEC_MEDIA_TYPE,
    write_interactive_plot_spec_artifact,
    write_png_visualization_artifact,
)


def _make_run(tmp_path: Path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="w")
    return root.create_group("analysis").create_group("example_runs").create_group("example_1")


def test_write_png_visualization_artifact_stores_png_bytes_and_manifest(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    payload = b"\x89PNG\r\n\x1a\nfake-png"

    result = write_png_visualization_artifact(
        run,
        "example_summary_png",
        payload,
        description="Example summary",
        created_by="test_plot_artifacts",
        artifact_signature="sig-1",
        created_at_utc="2026-04-26T00:00:00+00:00",
        source_paths={"speed": "tracks/id_0/speed_smoothed_mm"},
        parameters={"dpi": np.int64(150), "missing_threshold": np.nan},
        extra_attrs={"plot_family": "example", "nonfinite_metric": np.float32(np.inf)},
    )

    assert result.path == "visualizations/example_summary_png"
    assert result.artifact_schema_id == PNG_ARTIFACT_SCHEMA_ID

    artifact = run["visualizations"]["example_summary_png"]
    assert np.asarray(artifact[:], dtype=np.uint8).tobytes() == payload
    assert artifact.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert artifact.attrs["artifact_role"] == "snapshot"
    assert artifact.attrs["mime"] == "image/png"
    assert artifact.attrs["content_sha256"] == hashlib.sha256(payload).hexdigest()
    assert artifact.attrs["source_paths"]["speed"] == "tracks/id_0/speed_smoothed_mm"
    assert artifact.attrs["parameters"]["dpi"] == 150
    assert artifact.attrs["parameters"]["missing_threshold"] is None
    assert artifact.attrs["plot_family"] == "example"
    assert artifact.attrs["nonfinite_metric"] is None
    json.dumps(dict(artifact.attrs), allow_nan=False)
    json.dumps(dict(run.attrs), allow_nan=False)

    manifest = run.attrs["visualizations"]
    entry = manifest["example_summary_png"]
    assert entry["path"] == "visualizations/example_summary_png"
    assert entry["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert entry["artifact_role"] == "snapshot"
    assert entry["content_sha256"] == hashlib.sha256(payload).hexdigest()


def test_write_interactive_plot_spec_artifact_stores_spec_group_and_manifest(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    spec = {
        "schema_id": "palette.plot_spec.track_kinematics.v1",
        "marks": [{"type": "line", "x": "time_seconds", "y": "speed_smoothed_mm"}],
        "data_sources": {
            "time_seconds": "tracks/id_0/time_seconds",
            "speed_smoothed_mm": "tracks/id_0/speed_smoothed_mm",
        },
    }

    result = write_interactive_plot_spec_artifact(
        run,
        "example_summary_interactive",
        spec,
        description="Example interactive summary",
        created_by="test_plot_artifacts",
        renderer="palette-timeseries-v1",
        artifact_signature="sig-2",
        created_at_utc="2026-04-26T00:00:00+00:00",
        snapshot_artifact="example_summary_png",
        source_paths={"run": "."},
        source_runs={"track_kinematics": "example_1"},
        parameters={"decimation": "viewer"},
    )

    assert result.path == "visualizations/example_summary_interactive"
    assert result.artifact_schema_id == INTERACTIVE_SPEC_SCHEMA_ID

    artifact = run["visualizations"]["example_summary_interactive"]
    assert artifact.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert artifact.attrs["artifact_role"] == "interactive_spec"
    assert artifact.attrs["media_type"] == SPEC_MEDIA_TYPE
    assert artifact.attrs["renderer"] == "palette-timeseries-v1"
    assert artifact.attrs["snapshot_artifact"] == "example_summary_png"

    spec_bytes = np.asarray(artifact["spec_json"][:], dtype=np.uint8).tobytes()
    assert json.loads(spec_bytes.decode("utf-8")) == spec
    assert artifact["spec_json"].attrs["content_sha256"] == hashlib.sha256(spec_bytes).hexdigest()

    manifest = run.attrs["visualizations"]
    entry = manifest["example_summary_interactive"]
    assert entry["path"] == "visualizations/example_summary_interactive"
    assert entry["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert entry["artifact_role"] == "interactive_spec"
    assert entry["snapshot_artifact"] == "example_summary_png"
