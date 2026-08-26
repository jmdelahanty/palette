from __future__ import annotations

import copy

import numpy as np

from fisheye.shared.zarr.assignment_keypoint_rebinding import (
    ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
    ASSIGNMENT_KEYPOINT_REBINDING_POLICY,
    ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
    ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION,
    _assignment_collection_source_run,
    _chunked_equivalence,
    validate_assignment_keypoint_rebinding_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _collection() -> dict[str, object]:
    return {
        "schema_id": "palette.subject_mask.assignment_keypoint_collection",
        "schema_version": 1,
        "mode": "exact_worker_partition",
        "row_policy": "ordered_contiguous_recording_crop_rows_v1",
        "n_rois": 5,
        "workers": [
            {
                "global_row_interval": {"start_row": 0, "stop_row": 2},
                "assignment": {
                    "assignment_keypoint_group": "keypoints_runs",
                    "assignment_keypoints_run": "historical",
                    "assignment_keypoint_success_dataset": "detection_success",
                },
            },
            {
                "global_row_interval": {"start_row": 2, "stop_row": 5},
                "assignment": {
                    "assignment_keypoint_group": "keypoints_runs",
                    "assignment_keypoints_run": "historical",
                    "assignment_keypoint_success_dataset": "detection_success",
                },
            },
        ],
    }


def _manifest() -> dict[str, object]:
    payload = {
        "rebinding_run_id": "rebind_001",
        "policy": ASSIGNMENT_KEYPOINT_REBINDING_POLICY,
        "recording_identity": "recording",
        "camera_identity": "camera",
        "row_count": 5,
        "assignment_state": "used",
        "subject_mask_source": {
            "bundle_id": "bundle_001",
            "bundle_manifest_payload_digest": "1" * 64,
            "bundle_coordinate_authority_digest": "2" * 64,
            "refined_run_path": "refined_subject_masks_runs/refined_001",
            "assignment_collection_digest": "3" * 64,
            "historical_keypoint_run_path": "keypoints_runs/historical",
        },
        "canonical_keypoint_source": {
            "authority_profile": ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
            "run_path": "keypoints_runs/canonical",
            "run_manifest_payload_digest": "a" * 64,
            "run_manifest_document_digest": "b" * 64,
            "keypoint_bundle_authority_generation": 1,
            "keypoint_bundle_authority_digest": "c" * 64,
            "coordinate_successor_authority_digest": "d" * 64,
            "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
            "eye_keypoint_indices": {"eye_left": 1, "eye_right": 2},
            "keypoints_dataset": "keypoints_roi",
            "success_dataset": "pose_success",
        },
        "equivalence": {
            name: {
                "shape": [5, 3, 2] if name.startswith("keypoints_roi") else [5],
                "historical_dtype": (
                    "float64" if name.startswith("keypoints_roi") else "uint64"
                ),
                "canonical_dtype": (
                    "float32" if name.startswith("keypoints_roi") else "uint64"
                ),
                "normalization": (
                    "numpy_astype_float32_c_order_v1"
                    if name.startswith("keypoints_roi")
                    else "identity"
                ),
                "digest_algorithm": "sha256_c_contiguous_bytes_v1",
                "normalized_sha256": "e" * 64,
            }
            for name in (
                "source_crop_row_ids_to_source_crop_row_ids",
                "instance_key_to_instance_key",
                (
                    "source_acquisition_frame_index_to_"
                    "source_acquisition_frame_index"
                ),
                "keypoints_roi_to_keypoints_roi",
                "detection_success_to_pose_success",
            )
        },
        "selection_policy": "explicit_bundle_and_keypoint_run_no_fallback_v1",
        "stage_selector_eligible": False,
        "production_state_changes": [],
    }
    return {
        "schema_id": ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
        "schema_version": ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def test_assignment_collection_requires_one_gapless_recording_run() -> None:
    assert _assignment_collection_source_run(_collection()) == "historical"

    gap = _collection()
    gap["workers"][1]["global_row_interval"]["start_row"] = 3
    try:
        _assignment_collection_source_run(gap)
    except ValueError as exc:
        assert "partition" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("gapped assignment collection was accepted")

    mixed = _collection()
    mixed["workers"][1]["assignment"]["assignment_keypoints_run"] = "other"
    try:
        _assignment_collection_source_run(mixed)
    except ValueError as exc:
        assert "one recording-wide" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("mixed assignment collection was accepted")


def test_chunked_equivalence_seals_declared_float32_normalization() -> None:
    values64 = np.asarray(
        [[[1.25, 2.5]], [[np.nan, np.nan]], [[3.75, 4.0]]],
        dtype=np.float64,
    )
    values32 = values64.astype(np.float32)
    record = _chunked_equivalence(
        values64,
        values32,
        normalized_dtype=np.dtype("float32"),
        block_rows=2,
    )
    assert record["normalization"] == "numpy_astype_float32_c_order_v1"
    assert record["historical_dtype"] == "float64"
    assert record["canonical_dtype"] == "float32"

    changed = values32.copy()
    changed[-1, 0, 0] += 1
    try:
        _chunked_equivalence(
            values64,
            changed,
            normalized_dtype=np.dtype("float32"),
            block_rows=2,
        )
    except ValueError as exc:
        assert "values differ" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("changed keypoints were accepted")


def test_rebinding_manifest_is_closed_and_digest_sealed() -> None:
    manifest = _manifest()
    assert validate_assignment_keypoint_rebinding_manifest(manifest) == ()

    tampered = copy.deepcopy(manifest)
    tampered["payload"]["row_count"] = 6
    assert any(
        "payload digest differs" in error
        for error in validate_assignment_keypoint_rebinding_manifest(tampered)
    )

    expanded = copy.deepcopy(manifest)
    expanded["payload"]["unexpected"] = True
    expanded["payload_digest"] = canonical_json_sha256(expanded["payload"])
    assert any(
        "payload fields are not exact" in error
        for error in validate_assignment_keypoint_rebinding_manifest(expanded)
    )
