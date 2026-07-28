from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.detection.clipped_native_binding import (
    ClippedDetectionArtifactMember,
    bind_clipped_detection_artifacts,
)
from fisheye.detection.native_canonical_candidate import (
    validate_native_canonical_detection_candidate,
    write_native_clipped_detection_candidate,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1


def _bound():
    frame_indices = np.asarray([0, 2], dtype=np.int32)
    counts = np.asarray([1, 0, 1], dtype=np.int32)
    member = ClippedDetectionArtifactMember(
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
        frame_indices=frame_indices,
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
    return bind_clipped_detection_artifacts(
        [member],
        recording_identity="recording:native-fixture",
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


def test_native_candidate_uses_manifest_v2_and_logical_schema_v1(
    tmp_path: Path,
) -> None:
    bound = _bound()
    candidate = write_native_clipped_detection_candidate(
        bound,
        destination=tmp_path / "native-candidate.zarr",
        run_id="detect_native_clipped_1",
        recording_identity="recording:native-fixture",
        producer_id="fisheye.detection.detect_yolo",
        producer_version="e3936b9a",
        source_frame_authority={
            "record_ref": "analysis/acquisition_camera_frames/frame_axis@record",
            "record_sha256": "1" * 64,
        },
        source_pixel_authority={
            "record_ref": "raw_video@source_pixel_authority",
            "record_sha256": "2" * 64,
        },
        model_artifact_sha256="3" * 64,
        run_provenance=_provenance(),
    )

    assert candidate.manifest["schema_version"] == (
        CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert candidate.manifest["payload"]["logical_schema"]["schema_version"] == (
        CANONICAL_DETECTION_SCHEMA_V1.schema_version
    )
    assert candidate.receipt["native_run_manifest_schema_version"] == 2
    assert candidate.receipt["logical_schema_version"] == 1
    source_provenance = candidate.manifest["payload"]["source_evidence"][
        "run_provenance"
    ]["document"]
    assert source_provenance["clipped_detection_binding"]["digest"] == (
        bound.binding_evidence["digest"]
    )
    assert validate_native_canonical_detection_candidate(candidate) == ()

    root = zarr.open_group(
        str(candidate.output_path),
        mode="r",
        use_consolidated=True,
    )
    run = root["detect_runs"][candidate.run_id]
    assert run.attrs["stage_selector_eligible"] is False
    assert root["detect_runs"].attrs.get("latest") is None
    assert tuple(
        f"instances/{name}" for name in run["instances"].array_keys()
    ) != ()
    assert (
        candidate.output_path / "native_detection_candidate_receipt.json"
    ).is_file()


def test_native_candidate_rejects_different_binding_in_provenance(
    tmp_path: Path,
) -> None:
    provenance = _provenance()
    provenance["clipped_detection_binding"] = {
        "digest_algorithm": "sha256_canonical_json_v1",
        "digest": "f" * 64,
        "document": {},
    }
    with pytest.raises(ValueError, match="different clipped detection binding"):
        write_native_clipped_detection_candidate(
            _bound(),
            destination=tmp_path / "must-not-exist.zarr",
            run_id="detect_native_clipped_1",
            recording_identity="recording:native-fixture",
            producer_id="fisheye.detection.detect_yolo",
            producer_version="e3936b9a",
            source_frame_authority={
                "record_ref": "analysis/acquisition_camera_frames/frame_axis@record",
                "record_sha256": "1" * 64,
            },
            source_pixel_authority={
                "record_ref": "raw_video@source_pixel_authority",
                "record_sha256": "2" * 64,
            },
            model_artifact_sha256="3" * 64,
            run_provenance=provenance,
        )
    assert not (tmp_path / "must-not-exist.zarr").exists()


def test_native_candidate_reopen_validation_detects_manifest_mutation(
    tmp_path: Path,
) -> None:
    candidate = write_native_clipped_detection_candidate(
        _bound(),
        destination=tmp_path / "native-candidate.zarr",
        run_id="detect_native_clipped_1",
        recording_identity="recording:native-fixture",
        producer_id="fisheye.detection.detect_yolo",
        producer_version="e3936b9a",
        source_frame_authority={
            "record_ref": "analysis/acquisition_camera_frames/frame_axis@record",
            "record_sha256": "1" * 64,
        },
        source_pixel_authority={
            "record_ref": "raw_video@source_pixel_authority",
            "record_sha256": "2" * 64,
        },
        model_artifact_sha256="3" * 64,
        run_provenance=_provenance(),
    )
    run = zarr.open_group(
        str(candidate.output_path / "detect_runs" / candidate.run_id),
        mode="r+",
        use_consolidated=False,
    )
    mutated = copy.deepcopy(dict(run.attrs["run_manifest"]))
    mutated["payload"]["run_id"] = "tampered"
    run.attrs["run_manifest"] = mutated
    errors = validate_native_canonical_detection_candidate(candidate)
    assert "persisted native run manifest differs from the candidate" in errors
