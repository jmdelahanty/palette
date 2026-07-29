from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.cluster.arena_geometry_campaign import (
    TARGET_MANIFEST_SCHEMA,
    load_target_manifest,
)


def _target_files(tmp_path: Path) -> dict[str, Path]:
    recording = tmp_path / "recording"
    analysis = recording / "zarr" / "recording_analysis.zarr"
    analysis.mkdir(parents=True)
    (analysis / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    files = {
        "video": recording / "cams" / "Cam1.mp4",
        "summary": recording / "cams" / "Cam1_external_summary.json",
        "keyframes": recording / "cams" / "Cam1_keyframe.json",
        "receipt": recording / "raw" / "recording_geometry_recovery.json",
        "observation": recording / "raw" / "observation.json",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    return {"recording": recording, "analysis": analysis, **files}


def _manifest_row(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "target_id": "recording_a",
        "recording_id": "recording-a",
        "recording_dir": str(paths["recording"]),
        "analysis_zarr": str(paths["analysis"]),
        "video": str(paths["video"]),
        "summary": str(paths["summary"]),
        "keyframes": str(paths["keyframes"]),
        "recovery_receipt": str(paths["receipt"]),
        "acquisition_observation": str(paths["observation"]),
    }


def test_target_manifest_preserves_exact_recording_bound_sources(
    tmp_path: Path,
) -> None:
    paths = _target_files(tmp_path)
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {"schema": TARGET_MANIFEST_SCHEMA, "targets": [_manifest_row(paths)]}
        )
    )

    (target,) = load_target_manifest(manifest)

    assert target.recording_id == "recording-a"
    assert target.video_path == paths["video"].resolve()
    assert target.keyframe_path == paths["keyframes"].resolve()
    assert target.acquisition_observation_path == paths["observation"].resolve()


def test_target_manifest_rejects_source_outside_recording(tmp_path: Path) -> None:
    paths = _target_files(tmp_path)
    outside = tmp_path / "other.mp4"
    outside.write_bytes(b"video")
    row = _manifest_row(paths)
    row["video"] = str(outside)
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps({"schema": TARGET_MANIFEST_SCHEMA, "targets": [row]})
    )

    with pytest.raises(ValueError, match="must belong to the recording"):
        load_target_manifest(manifest)


def test_target_manifest_rejects_non_v3_analysis_target(tmp_path: Path) -> None:
    paths = _target_files(tmp_path)
    (paths["analysis"] / "zarr.json").unlink()
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {"schema": TARGET_MANIFEST_SCHEMA, "targets": [_manifest_row(paths)]}
        )
    )

    with pytest.raises(FileNotFoundError, match="not Zarr v3"):
        load_target_manifest(manifest)
