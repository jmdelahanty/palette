from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.detection.clipped_native_binding import ClippedDetectionArtifactMember
from fisheye.utils import assemble_clipped_native_detection as assembler


def _member() -> ClippedDetectionArtifactMember:
    counts = np.asarray([1, 0, 1], dtype=np.int32)
    return ClippedDetectionArtifactMember(
        work_unit_id="clip_000000:camera_2010093",
        artifact_run_id="detect_artifact_0",
        clip_id="clip_000000",
        clip_index=0,
        camera_serial="2010093",
        source_width=640,
        source_height=480,
        artifact_manifest_sha256="a" * 64,
        run_group_tree_sha256="b" * 64,
        parent_frame_indices=np.arange(3, dtype=np.int64),
        frame_indices=np.asarray([0, 2], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [[0.25, 0.5, 0.1, 0.1], [0.75, 0.5, 0.1, 0.1]],
            dtype=np.float64,
        ),
        scores=np.asarray([0.9, 0.8], dtype=np.float32),
        class_ids=np.asarray([0, 0], dtype=np.int32),
        artifact_row_id=np.arange(2, dtype=np.uint64),
        frame_counts=counts,
        n_detections=counts.copy(),
    )


def _evidence(model_sha256: str) -> dict[str, object]:
    return {
        "report_path": "/reports/clip.json",
        "report_sha256": "c" * 64,
        "receipt_path": "/archive/.imports/clip.json",
        "receipt_sha256": "d" * 64,
        "artifact_group_path": (
            "clips/clip_000000/cameras/2010093/"
            "detection_artifact_runs/detect_artifact_0"
        ),
        "artifact_manifest_sha256": "a" * 64,
        "run_group_tree_sha256": "b" * 64,
        "run_provenance": {
            "input_artifacts": [
                {"role": "detect_model", "sha256": model_sha256}
            ]
        },
    }


def test_assembler_binds_then_publishes_one_native_v2_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        assembler,
        "load_clipped_detection_artifact_members",
        lambda *_args, **_kwargs: ((_member(),), (_evidence("3" * 64),)),
    )

    def fake_candidate(bound, **kwargs):
        calls["bound"] = bound
        calls["candidate_kwargs"] = kwargs
        return SimpleNamespace(
            output_path=kwargs["destination"],
            run_id=kwargs["run_id"],
            receipt={"native_run_manifest_schema_version": 2},
        )

    def fake_publish(**kwargs):
        calls["publish_kwargs"] = kwargs
        return {
            "status": "complete",
            "group_path": "detect_runs/detect_native",
            "selector_eligible": False,
            "registry_updated": False,
        }

    monkeypatch.setattr(
        assembler,
        "write_native_clipped_detection_candidate",
        fake_candidate,
    )
    monkeypatch.setattr(
        assembler,
        "publish_native_canonical_detection_candidate",
        fake_publish,
    )
    frame_authority = {"record_ref": "/frames@record", "record_sha256": "1" * 64}
    pixel_authority = {"record_ref": "/raw@record", "record_sha256": "2" * 64}

    result = assembler.assemble_and_publish_clipped_native_detection(
        analysis_zarr=tmp_path / "analysis.zarr",
        work_unit_reports=(tmp_path / "report.json",),
        recording_frame_index=tmp_path / "recording_frame_index.parquet",
        recording_identity="recording:fixture",
        n_frames=3,
        source_width=640,
        source_height=480,
        run_id="detect_native",
        candidate_zarr=tmp_path / "candidate.zarr",
        producer_id="fisheye.detection.detect_yolo",
        producer_version="commit123",
        source_frame_authority=frame_authority,
        source_pixel_authority=pixel_authority,
        model_artifact_sha256="3" * 64,
        workflow_id="workflow",
    )

    assert result["status"] == "complete"
    assert result["canonical_group_path"] == "detect_runs/detect_native"
    assert result["native_run_manifest_schema_version"] == 2
    assert result["logical_schema_version"] == 1
    bound = calls["bound"]
    assert bound.dimensions.n_instances == 2
    assert calls["candidate_kwargs"]["source_frame_authority"] == frame_authority
    assert calls["candidate_kwargs"]["source_pixel_authority"] == pixel_authority
    assert calls["publish_kwargs"]["run_id"] == "detect_native"


def test_assembler_rejects_artifact_from_a_different_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        assembler,
        "load_clipped_detection_artifact_members",
        lambda *_args, **_kwargs: ((_member(),), (_evidence("9" * 64),)),
    )

    with pytest.raises(ValueError, match="model digest differs"):
        assembler.assemble_and_publish_clipped_native_detection(
            analysis_zarr=tmp_path / "analysis.zarr",
            work_unit_reports=(tmp_path / "report.json",),
            recording_frame_index=tmp_path / "recording_frame_index.parquet",
            recording_identity="recording:fixture",
            n_frames=3,
            source_width=640,
            source_height=480,
            run_id="detect_native",
            candidate_zarr=tmp_path / "candidate.zarr",
            producer_id="fisheye.detection.detect_yolo",
            producer_version="commit123",
            source_frame_authority={
                "record_ref": "/frames@record",
                "record_sha256": "1" * 64,
            },
            source_pixel_authority={
                "record_ref": "/raw@record",
                "record_sha256": "2" * 64,
            },
            model_artifact_sha256="3" * 64,
            workflow_id="workflow",
        )
