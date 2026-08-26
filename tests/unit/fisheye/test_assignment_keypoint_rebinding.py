from __future__ import annotations

import copy
import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np

from fisheye.shared.zarr.assignment_keypoint_rebinding import (
    ASSIGNMENT_CANONICAL_KEYPOINT_PROFILE,
    ASSIGNMENT_KEYPOINT_REBINDING_POLICY,
    ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
    ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_VERSION,
    _assignment_collection_source_run,
    _chunked_equivalence,
    inspect_assignment_keypoint_rebinding,
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


def test_inspection_uses_resolver_digests_for_immutable_documents(
    monkeypatch: object,
    tmp_path: object,
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()

    def digest(values: np.ndarray) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(values).tobytes(order="C")
        ).hexdigest()

    identity = {
        "source_crop_row_ids": np.asarray([0, 1], dtype=np.int64),
        "instance_key": np.asarray([10, 11], dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray([20, 21], dtype=np.int64),
    }
    keypoints = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    pose_success = np.asarray([True, False], dtype=np.bool_)
    canonical_arrays = {
        **identity,
        "keypoints_roi": keypoints,
        "pose_success": pose_success,
    }
    historical_arrays = {
        **identity,
        "keypoints_roi": keypoints.astype(np.float64),
        "detection_success": pose_success,
    }

    class Group(dict[str, object]):
        def __init__(
            self,
            values: dict[str, object],
            *,
            path: str,
            attrs: dict[str, object] | None = None,
        ) -> None:
            super().__init__(values)
            self.path = path
            self.attrs = attrs or {}

    labels = ["swim_bladder", "eye_left", "eye_right"]
    historical = Group(
        historical_arrays,
        path="keypoints_runs/historical",
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "keypoint_labels": labels,
        },
    )
    canonical = Group(
        canonical_arrays,
        path="keypoints_runs/canonical",
    )
    array_declarations = {
        name: {
            "digest_algorithm": "sha256_c_contiguous_bytes_v1",
            "sha256": digest(values),
        }
        for name, values in canonical_arrays.items()
    }
    manifest_document = {
        "payload_digest": "a" * 64,
        "payload": {
            "logical_content": {"document": {"arrays": array_declarations}},
            "source_crop_snapshot": {"run_path": "crop_runs/canonical"},
            "pose_model_schema_binding": {
                "pose_schema": {"keypoint_labels": labels}
            },
        },
    }
    manifest_digest = "b" * 64
    authority_digest = "c" * 64
    successor_digest = "d" * 64
    source = SimpleNamespace(
        run_group=canonical,
        manifest=MappingProxyType(manifest_document),
        manifest_digest=manifest_digest,
        active_keypoint_bundle_authority=MappingProxyType({"generation": 7}),
        active_keypoint_bundle_authority_digest=authority_digest,
        successor_authority_digest=successor_digest,
    )
    collection = MappingProxyType(
        {
            "schema_id": "palette.subject_mask.assignment_keypoint_collection",
            "schema_version": 1,
            "mode": "exact_worker_partition",
            "row_policy": "ordered_contiguous_recording_crop_rows_v1",
            "n_rois": 2,
            "workers": [
                {
                    "global_row_interval": {"start_row": 0, "stop_row": 2},
                    "assignment": {
                        "assignment_keypoint_group": "keypoints_runs",
                        "assignment_keypoints_run": "historical",
                        "assignment_keypoint_success_dataset": "detection_success",
                    },
                }
            ],
        }
    )
    bundle = SimpleNamespace(
        assignment_keypoint_collection=collection,
        crop_run_path="crop_runs/canonical",
        bundle_manifest={
            "payload_digest": "e" * 64,
            "payload": {
                "cross_binding": {
                    "raw_refined_identity_array_values_sha256": {
                        name: digest(values) for name, values in identity.items()
                    }
                }
            },
        },
        recording_identity="recording",
        camera_identity="camera",
        n_rois=2,
        bundle_id="bundle",
        authority_digest="f" * 64,
        refined_run_path="refined_subject_masks_runs/refined",
    )

    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_recording_subject_mask_coordinate_authority",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding."
        "load_keypoint_coordinate_successor_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        "fisheye.shared.zarr.assignment_keypoint_rebinding.open_zarr_root",
        lambda *_args, **_kwargs: {"keypoints_runs/historical": historical},
    )

    result = inspect_assignment_keypoint_rebinding(
        analysis_zarr=archive,
        subject_mask_bundle_id="bundle",
        keypoint_run_id="canonical",
        rebinding_run_id="rebind_001",
        block_rows=1,
    )

    keypoint_source = result["payload"]["canonical_keypoint_source"]
    assert keypoint_source["run_manifest_document_digest"] == manifest_digest
    assert keypoint_source["keypoint_bundle_authority_digest"] == authority_digest
    assert keypoint_source["coordinate_successor_authority_digest"] == successor_digest
    json.dumps(result, allow_nan=False)
