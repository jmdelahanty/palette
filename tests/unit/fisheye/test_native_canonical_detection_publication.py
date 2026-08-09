from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.detection.clipped_native_binding import (
    ClippedDetectionArtifactMember,
    bind_clipped_detection_artifacts,
)
from fisheye.detection.native_canonical_candidate import (
    write_native_clipped_detection_candidate,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.analysis_workflows.native_canonical_detection_publication import (
    publish_native_canonical_detection_candidate,
)
from fisheye.analysis_workflows import (
    native_canonical_detection_publication as publication_mod,
)


RECORDING_IDENTITY = "recording:native-publication-fixture"
RUN_ID = "detect_native_clipped_1"


def _bound():
    counts = np.asarray([1, 0, 1], dtype=np.int32)
    return bind_clipped_detection_artifacts(
        [
            ClippedDetectionArtifactMember(
                work_unit_id="clip_work_0",
                artifact_run_id="artifact_0",
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
                    [[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]],
                    dtype=np.float64,
                ),
                scores=np.asarray([0.9, 0.8], dtype=np.float32),
                class_ids=np.asarray([0, 0], dtype=np.int32),
                artifact_row_id=np.arange(2, dtype=np.uint64),
                frame_counts=counts,
                n_detections=counts.copy(),
            )
        ],
        recording_identity=RECORDING_IDENTITY,
        n_frames=3,
        source_width=640,
        source_height=480,
    )


def _provenance() -> dict[str, object]:
    return {
        "schema": "palette.run_provenance.v1",
        "git_sha": "4" * 40,
        "config_hash": "5" * 64,
        "params": {"conf_threshold": 0.5},
        "input_run_ids": {},
        "input_artifacts": [{"role": "detect_model", "sha256": "3" * 64}],
        "command": "fisheye.detection.clipped_native_binding",
        "fisheye_version": None,
    }


def _archive(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = RECORDING_IDENTITY
    frame_node = root.create_group("analysis/acquisition_camera_frames/frame_axis")
    frame_record = {
        "schema_id": "palette.acquisition_frame_axis",
        "schema_version": 1,
        "n_frames": 3,
    }
    frame_digest = canonical_json_sha256(frame_record)
    frame_node.attrs.update(
        {"record": frame_record, "record_sha256": frame_digest}
    )
    pixel_node = root.create_group("raw_video")
    pixel_record = {
        "schema_id": "palette.pixel_frame_authority",
        "schema_version": 1,
        "width": 640,
        "height": 480,
        "units": "pixels",
    }
    pixel_digest = canonical_json_sha256(pixel_record)
    pixel_node.attrs.update(
        {
            "source_pixel_authority": pixel_record,
            "source_pixel_authority_sha256": pixel_digest,
        }
    )
    return (
        {
            "record_ref": "/analysis/acquisition_camera_frames/frame_axis@record",
            "record_sha256": frame_digest,
        },
        {
            "record_ref": "/raw_video@source_pixel_authority",
            "record_sha256": pixel_digest,
        },
    )


def _candidate(tmp_path: Path, frame: dict[str, str], pixel: dict[str, str]):
    return write_native_clipped_detection_candidate(
        _bound(),
        destination=tmp_path / "candidate.zarr",
        run_id=RUN_ID,
        recording_identity=RECORDING_IDENTITY,
        producer_id="fisheye.detection.detect_yolo",
        producer_version="e3936b9a",
        source_frame_authority=frame,
        source_pixel_authority=pixel,
        model_artifact_sha256="3" * 64,
        run_provenance=_provenance(),
    )


def test_native_candidate_is_atomically_published_but_not_selected(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)
    receipt_path = tmp_path / "publication.json"

    result = publish_native_canonical_detection_candidate(
        analysis_zarr=archive,
        candidate_zarr=candidate.output_path,
        run_id=RUN_ID,
        recording_identity=RECORDING_IDENTITY,
        result_json=receipt_path,
    )

    assert result["status"] == "complete"
    assert result["native_run_manifest_schema_version"] == 2
    assert result["logical_schema_version"] == 1
    assert result["selector_eligible"] is False
    assert result["registry_updated"] is False
    assert result["source_authorities_revalidated_after_copy"] is True
    assert receipt_path.is_file()

    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    family = root["detect_runs"]
    assert family.attrs.get("latest") is None
    assert family.attrs.get("latest_complete") is None
    run = family[RUN_ID]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["run_manifest"]["schema_version"] == 2
    assert np.asarray(run["instances/frame_row_offsets"][:]).tolist() == [0, 1, 1, 2]


def test_native_publication_tolerates_unrelated_legacy_root_infinity(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)
    root_metadata = archive / "zarr.json"
    payload = json.loads(root_metadata.read_text(encoding="utf-8"))
    payload["attributes"]["legacy_unrelated_limit"] = float("inf")
    root_metadata.write_text(json.dumps(payload), encoding="utf-8")

    result = publish_native_canonical_detection_candidate(
        analysis_zarr=archive,
        candidate_zarr=candidate.output_path,
        run_id=RUN_ID,
        recording_identity=RECORDING_IDENTITY,
    )

    assert result["status"] == "complete"
    assert result["selector_eligible"] is False


def test_native_postcopy_failure_tombstones_exact_owned_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)

    monkeypatch.setattr(
        publication_mod,
        "canonical_detection_metadata_declaration_maps",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("post-copy metadata failure")
        ),
    )

    with pytest.raises(ValueError, match="post-copy metadata failure"):
        publish_native_canonical_detection_candidate(
            analysis_zarr=archive,
            candidate_zarr=candidate.output_path,
            run_id=RUN_ID,
            recording_identity=RECORDING_IDENTITY,
        )

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    family = root["detect_runs"]
    run = family[RUN_ID]
    assert run.attrs["status"] == "failed"
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["atomic_publication_tombstone"]["schema_id"] == (
        "palette.native_canonical_detection.postcopy_failure"
    )
    assert family.attrs.get("latest") is None
    assert family.attrs.get("latest_complete") is None


def test_native_publication_fails_before_copy_when_authority_drifted(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root["raw_video"].attrs["source_pixel_authority"] = {
        "schema_id": "palette.pixel_frame_authority",
        "schema_version": 1,
        "width": 320,
        "height": 240,
        "units": "pixels",
    }

    with pytest.raises(ValueError, match="content digest differs"):
        publish_native_canonical_detection_candidate(
            analysis_zarr=archive,
            candidate_zarr=candidate.output_path,
            run_id=RUN_ID,
            recording_identity=RECORDING_IDENTITY,
        )

    assert not (archive / "detect_runs" / RUN_ID).exists()


def test_native_publication_rejects_manifest_v1_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)
    run = zarr.open_group(
        str(candidate.output_path / "detect_runs" / RUN_ID),
        mode="a",
        use_consolidated=False,
    )
    manifest = dict(run.attrs["run_manifest"])
    manifest["schema_version"] = 1
    run.attrs["run_manifest"] = manifest

    with pytest.raises(ValueError, match="run-manifest v2"):
        publish_native_canonical_detection_candidate(
            analysis_zarr=archive,
            candidate_zarr=candidate.output_path,
            run_id=RUN_ID,
            recording_identity=RECORDING_IDENTITY,
        )

    assert not (archive / "detect_runs" / RUN_ID).exists()


def test_native_publication_requires_candidate_receipt(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    frame, pixel = _archive(archive)
    candidate = _candidate(tmp_path, frame, pixel)
    (candidate.output_path / "native_detection_candidate_receipt.json").unlink()

    with pytest.raises(FileNotFoundError, match="candidate receipt"):
        publish_native_canonical_detection_candidate(
            analysis_zarr=archive,
            candidate_zarr=candidate.output_path,
            run_id=RUN_ID,
            recording_identity=RECORDING_IDENTITY,
        )

    assert not (archive / "detect_runs" / RUN_ID).exists()
