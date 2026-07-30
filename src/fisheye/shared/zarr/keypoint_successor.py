"""Prepare complete raw-keypoint successors after crop reconciliation.

The target remains a fresh immutable ``keypoints_runs/<run>`` snapshot.  This
module reuses parent pose payload only for crop rows proven unchanged by the
crop successor plan, and requires one explicit terminal inference result for
every added or invalidated observation.  A terminal failure is represented by
a real row with ``pose_success == False`` and NaN payloads; absence from the
batch means pending work and fails preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.keyed_delta import ACTION_CODE_MAP
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_shadow import validate_crop_geometry_shadow_publication
from fisheye.shared.zarr.crop_successor import (
    CropGeometrySuccessorPublication,
    validate_crop_geometry_successor_publication_receipt,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KeypointPreprocessingReference,
    keypoint_skeleton_digest,
)
from fisheye.shared.zarr.keypoint_publication import (
    DEFAULT_KEYPOINT_SHADOW_ROOT,
    KeypointShadowPublication,
    PreparedRawKeypointSnapshot,
    prepare_raw_keypoint_v2_snapshot,
    publish_selector_ineligible_keypoint_snapshot,
    validate_keypoint_shadow_publication,
)
from fisheye.shared.zarr.keypoint_schema import (
    KeypointDimensions,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile


RAW_KEYPOINT_SUCCESSOR_SCHEMA_ID = "palette.keypoint.raw_successor_preparation"
RAW_KEYPOINT_SUCCESSOR_SCHEMA_VERSION = 1
RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_ID = (
    "palette.keypoint.raw_successor_publication"
)
RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_VERSION = 1
RAW_KEYPOINT_SUCCESSOR_PUBLICATION_RECEIPT_NAME = (
    "raw_keypoint_successor_publication_receipt.json"
)
_INLINE_KEY_LIMIT = 64


class RawKeypointSuccessorError(ValueError):
    """Raised when a complete raw-keypoint successor cannot be prepared."""


@dataclass(frozen=True)
class TerminalKeypointInferenceBatch:
    """Terminal model results for exactly the rows requiring computation."""

    instance_key: np.ndarray
    keypoints_roi: np.ndarray
    keypoint_confidences: np.ndarray
    pose_confidence: np.ndarray
    pose_bbox_xyxy_roi: np.ndarray
    pose_success: np.ndarray

    def __post_init__(self) -> None:
        keys = np.asarray(self.instance_key)
        points = np.asarray(self.keypoints_roi)
        confidences = np.asarray(self.keypoint_confidences)
        pose_confidence = np.asarray(self.pose_confidence)
        bbox = np.asarray(self.pose_bbox_xyxy_roi)
        success = np.asarray(self.pose_success)
        if keys.dtype != np.dtype(np.uint64) or keys.ndim != 1:
            raise RawKeypointSuccessorError(
                "Inference instance_key must have exact uint64 shape [M]."
            )
        rows = int(keys.shape[0])
        if np.unique(keys).shape[0] != rows:
            raise RawKeypointSuccessorError(
                "Inference instance_key values must be unique."
            )
        if (
            points.dtype != np.dtype(np.float32)
            or points.ndim != 3
            or points.shape[0] != rows
            or points.shape[2] != 2
        ):
            raise RawKeypointSuccessorError(
                "Inference keypoints_roi must have exact float32 shape [M,K,2]."
            )
        keypoints = int(points.shape[1])
        if confidences.dtype != np.dtype(np.float32) or confidences.shape != (
            rows,
            keypoints,
        ):
            raise RawKeypointSuccessorError(
                "Inference keypoint_confidences must have exact float32 shape [M,K]."
            )
        if pose_confidence.dtype != np.dtype(np.float32) or pose_confidence.shape != (
            rows,
        ):
            raise RawKeypointSuccessorError(
                "Inference pose_confidence must have exact float32 shape [M]."
            )
        if bbox.dtype != np.dtype(np.float32) or bbox.shape != (rows, 4):
            raise RawKeypointSuccessorError(
                "Inference pose_bbox_xyxy_roi must have exact float32 shape [M,4]."
            )
        if success.dtype != np.dtype(bool) or success.shape != (rows,):
            raise RawKeypointSuccessorError(
                "Inference pose_success must have exact bool shape [M]."
            )
        finite_x = np.isfinite(points[..., 0])
        finite_y = np.isfinite(points[..., 1])
        if not np.array_equal(finite_x, finite_y):
            raise RawKeypointSuccessorError(
                "Inference landmarks must be wholly finite or wholly NaN."
            )
        if not np.array_equal(np.isfinite(confidences), finite_x):
            raise RawKeypointSuccessorError(
                "Inference confidence finiteness must match landmark finiteness."
            )
        if np.any(success & ~np.isfinite(pose_confidence)) or np.any(
            success & ~np.all(np.isfinite(bbox), axis=1)
        ):
            raise RawKeypointSuccessorError(
                "Successful inference rows require finite pose confidence and bbox."
            )
        failed = ~success
        if (
            np.any(np.isfinite(points[failed]))
            or np.any(np.isfinite(confidences[failed]))
            or np.any(np.isfinite(pose_confidence[failed]))
            or np.any(np.isfinite(bbox[failed]))
        ):
            raise RawKeypointSuccessorError(
                "Terminal failed inference rows require exact NaN pose payloads."
            )

    @property
    def n_rows(self) -> int:
        return int(self.instance_key.shape[0])

    @property
    def n_keypoints(self) -> int:
        return int(self.keypoints_roi.shape[1])


@dataclass(frozen=True)
class PreparedRawKeypointSuccessor:
    """Complete validated raw snapshot plus its reconciliation receipt."""

    prepared: PreparedRawKeypointSnapshot
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class RawKeypointSuccessorPublication:
    """One immutable raw-keypoint successor and its persisted receipt."""

    publication: KeypointShadowPublication
    prepared: PreparedRawKeypointSuccessor
    receipt: Mapping[str, Any]


def _values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _key_receipt(values: np.ndarray) -> dict[str, object]:
    keys = np.sort(np.asarray(values, dtype=np.uint64).reshape(-1))
    result: dict[str, object] = {
        "count": int(keys.shape[0]),
        "sha256": sha256_array(keys),
    }
    if keys.shape[0] <= _INLINE_KEY_LIMIT:
        result["values"] = [int(value) for value in keys]
    return result


def _row_lookup(
    *,
    requested_keys: np.ndarray,
    available_keys: np.ndarray,
    label: str,
) -> np.ndarray:
    requested = np.asarray(requested_keys, dtype=np.uint64).reshape(-1)
    available = np.asarray(available_keys, dtype=np.uint64).reshape(-1)
    order = np.argsort(available, kind="stable")
    sorted_keys = available[order]
    positions = np.searchsorted(sorted_keys, requested)
    in_bounds = positions < sorted_keys.shape[0]
    matched = np.zeros(requested.shape[0], dtype=bool)
    matched[in_bounds] = sorted_keys[positions[in_bounds]] == requested[in_bounds]
    if not np.all(matched):
        missing = np.sort(requested[~matched])
        raise RawKeypointSuccessorError(
            f"{label} lacks instance_key values {missing[:16].tolist()!r}."
        )
    return order[positions].astype(np.int64, copy=False)


def prepare_raw_keypoint_successor(
    parent: KeypointShadowPublication,
    crop_successor: CropGeometrySuccessorPublication,
    inference: TerminalKeypointInferenceBatch,
    *,
    pose_model_schema_binding: Mapping[str, Any] | None = None,
    preprocessing: KeypointPreprocessingReference | None = None,
) -> PreparedRawKeypointSuccessor:
    """Reconcile one complete crop successor into a complete raw-keypoint run."""

    if not isinstance(parent, KeypointShadowPublication):
        raise TypeError("parent must be a KeypointShadowPublication.")
    if not isinstance(crop_successor, CropGeometrySuccessorPublication):
        raise TypeError("crop_successor must be a CropGeometrySuccessorPublication.")
    parent_errors = validate_keypoint_shadow_publication(parent)
    if parent_errors:
        raise RawKeypointSuccessorError(
            "Parent raw-keypoint publication is invalid: "
            + "; ".join(parent_errors)
        )
    target_crop = crop_successor.publication
    crop_plan = crop_successor.plan
    crop_errors = validate_crop_geometry_shadow_publication(target_crop)
    if crop_errors:
        raise RawKeypointSuccessorError(
            "Target crop successor publication is invalid: "
            + "; ".join(crop_errors)
        )
    receipt_errors = validate_crop_geometry_successor_publication_receipt(
        crop_successor.receipt
    )
    if receipt_errors:
        raise RawKeypointSuccessorError(
            "Target crop successor receipt is invalid: "
            + "; ".join(receipt_errors)
        )
    crop_output = crop_successor.receipt["payload"]["output_crop"]
    if (
        crop_output["run_id"] != target_crop.run_id
        or crop_output["run_manifest_digest"]
        != target_crop.manifest["payload_digest"]
    ):
        raise RawKeypointSuccessorError(
            "Target crop successor receipt differs from its publication."
        )
    if (
        parent.prepared.crop_source.run_id != crop_plan.parent_crop_run_id
        or parent.prepared.source_crop_manifest.get("payload_digest")
        != crop_plan.parent_crop_manifest_digest
    ):
        raise RawKeypointSuccessorError(
            "Parent raw-keypoint crop binding differs from the crop successor parent."
        )
    target_crop_keys = _values(target_crop.arrays["instance_key"]).astype(
        np.uint64,
        copy=False,
    )
    if not np.array_equal(
        target_crop_keys,
        crop_plan.keyed_plan.target_instance_keys,
    ):
        raise RawKeypointSuccessorError(
            "Target crop rows differ from the crop successor plan."
        )

    binding = (
        parent.prepared.pose_model_schema_binding
        if pose_model_schema_binding is None
        else dict(pose_model_schema_binding)
    )
    preprocessing_ref = (
        parent.prepared.preprocessing if preprocessing is None else preprocessing
    )
    if dict(binding) != dict(parent.prepared.pose_model_schema_binding):
        raise RawKeypointSuccessorError(
            "Raw-keypoint successor v1 requires the exact parent pose-model binding."
        )
    if preprocessing_ref != parent.prepared.preprocessing:
        raise RawKeypointSuccessorError(
            "Raw-keypoint successor v1 requires the exact parent preprocessing."
        )
    keypoints = parent.prepared.dimensions.n_keypoints
    if inference.n_keypoints != keypoints:
        raise RawKeypointSuccessorError(
            "Inference landmark count differs from the parent keypoint contract."
        )

    target_rows = int(target_crop_keys.shape[0])
    reused_mask = crop_plan.keyed_plan.action_codes == ACTION_CODE_MAP["copy"]
    compute_mask = ~reused_mask
    reused_keys = target_crop_keys[reused_mask]
    compute_keys = target_crop_keys[compute_mask]
    inference_keys = np.asarray(inference.instance_key, dtype=np.uint64)
    if not np.array_equal(np.sort(inference_keys), np.sort(compute_keys)):
        raise RawKeypointSuccessorError(
            "Terminal inference keys must exactly equal added/invalidated crop keys."
        )

    parent_arrays = {
        path: _values(parent.prepared.arrays[path])
        for path in parent.prepared.arrays
    }
    parent_keys = np.asarray(parent_arrays["instance_key"], dtype=np.uint64)
    reused_parent_rows = _row_lookup(
        requested_keys=reused_keys,
        available_keys=parent_keys,
        label="Parent raw keypoint snapshot",
    )
    inference_rows = _row_lookup(
        requested_keys=compute_keys,
        available_keys=inference_keys,
        label="Terminal inference batch",
    )

    points_roi = np.full((target_rows, keypoints, 2), np.nan, dtype=np.float32)
    confidences = np.full((target_rows, keypoints), np.nan, dtype=np.float32)
    pose_confidence = np.full(target_rows, np.nan, dtype=np.float32)
    bbox_roi = np.full((target_rows, 4), np.nan, dtype=np.float32)
    pose_success = np.zeros(target_rows, dtype=bool)
    reused_target_rows = np.flatnonzero(reused_mask)
    computed_target_rows = np.flatnonzero(compute_mask)
    for path, destination in (
        ("keypoints_roi", points_roi),
        ("keypoint_confidences", confidences),
        ("pose_confidence", pose_confidence),
        ("pose_bbox_xyxy_roi", bbox_roi),
        ("pose_success", pose_success),
    ):
        destination[reused_target_rows] = parent_arrays[path][reused_parent_rows]
    points_roi[computed_target_rows] = inference.keypoints_roi[inference_rows]
    confidences[computed_target_rows] = inference.keypoint_confidences[inference_rows]
    pose_confidence[computed_target_rows] = inference.pose_confidence[inference_rows]
    bbox_roi[computed_target_rows] = inference.pose_bbox_xyxy_roi[inference_rows]
    pose_success[computed_target_rows] = inference.pose_success[inference_rows]

    keypoint_valid = (
        pose_success[:, None]
        & np.all(np.isfinite(points_roi), axis=2)
        & np.isfinite(confidences)
    )
    points_roi[~keypoint_valid] = np.float32(np.nan)
    confidences[~keypoint_valid] = np.float32(np.nan)
    pose_confidence[~pose_success] = np.float32(np.nan)
    bbox_roi[~pose_success] = np.float32(np.nan)

    crop_arrays = target_crop.arrays
    origins = _values(crop_arrays["roi_coordinates_full"]).astype(
        np.float32,
        copy=False,
    )
    points_img = points_roi + origins[:, None, :]
    bbox_img = bbox_roi + np.column_stack((origins, origins)).astype(
        np.float32,
        copy=False,
    )
    source_crop_signatures = _values(crop_arrays["source_row_signature"]).astype(
        np.uint8,
        copy=False,
    )
    arrays: dict[str, np.ndarray] = {
        "instance_key": np.array(target_crop_keys, copy=True),
        "source_crop_row_ids": np.arange(target_rows, dtype=np.int64),
        "source_acquisition_frame_index": np.array(
            _values(crop_arrays["source_acquisition_frame_index"]),
            dtype=np.int64,
            copy=True,
        ),
        "frame_indices": np.array(
            _values(crop_arrays["frame_indices"]),
            dtype=np.int64,
            copy=True,
        ),
        "frame_row_offsets": np.array(
            _values(crop_arrays["frame_row_offsets"]),
            dtype=np.int64,
            copy=True,
        ),
        "source_crop_row_signature": np.array(
            source_crop_signatures,
            copy=True,
        ),
        "keypoints_roi": points_roi,
        "keypoints_img": points_img,
        "keypoint_confidences": confidences,
        "keypoint_valid": keypoint_valid,
        "pose_confidence": pose_confidence,
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": pose_success,
    }
    skeleton_digest = keypoint_skeleton_digest(binding)
    arrays["keypoint_row_signature"] = derive_keypoint_row_signatures(
        instance_key=arrays["instance_key"],
        source_crop_row_signature=arrays["source_crop_row_signature"],
        keypoints_roi=arrays["keypoints_roi"],
        keypoint_valid=arrays["keypoint_valid"],
        skeleton_digest=skeleton_digest,
    )
    dimensions = KeypointDimensions(
        n_frames=target_crop.dimensions.n_frames,
        n_instances=target_rows,
        n_keypoints=keypoints,
        source_width=target_crop.dimensions.source_width,
        source_height=target_crop.dimensions.source_height,
    )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=target_crop.arrays,
        source_crop_manifest=target_crop.manifest,
        pose_model_schema_binding=binding,
        preprocessing=preprocessing_ref,
    )
    success_keys = compute_keys[inference.pose_success[inference_rows]]
    failure_keys = compute_keys[~inference.pose_success[inference_rows]]
    retired_keys = crop_plan.retired_instance_keys
    receipt = {
        "schema_id": RAW_KEYPOINT_SUCCESSOR_SCHEMA_ID,
        "schema_version": RAW_KEYPOINT_SUCCESSOR_SCHEMA_VERSION,
        "parent_keypoint_run_id": parent.run_id,
        "parent_keypoint_manifest_digest": parent.manifest["payload_digest"],
        "parent_crop_run_id": crop_plan.parent_crop_run_id,
        "parent_crop_manifest_digest": crop_plan.parent_crop_manifest_digest,
        "target_crop_run_id": target_crop.run_id,
        "target_crop_manifest_digest": target_crop.manifest["payload_digest"],
        "pose_model_binding": dict(binding),
        "preprocessing": preprocessing_ref.as_manifest(),
        "instance_keys": {
            "reused": _key_receipt(reused_keys),
            "inference_succeeded": _key_receipt(success_keys),
            "inference_failed": _key_receipt(failure_keys),
            "retired": _key_receipt(retired_keys),
        },
        "row_coverage": "complete_target_crop_rowset",
        "pending_row_count": 0,
        "publication_authorized": False,
        "selector_activation": "none_preparation_only",
        "production_state_changes": [],
    }
    return PreparedRawKeypointSuccessor(prepared=prepared, receipt=receipt)


def _validate_key_receipt(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return (f"{name} must be an object",)
    expected = {"count", "sha256"}
    if "values" in value:
        expected.add("values")
    errors: list[str] = []
    if set(value) != expected:
        errors.append(f"{name} has an unexpected field set")
    count = value.get("count")
    digest = value.get("sha256")
    if type(count) is not int or count < 0:
        errors.append(f"{name}.count must be a nonnegative integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        errors.append(f"{name}.sha256 must be a lowercase SHA-256 digest")
    values = value.get("values")
    if values is not None:
        if (
            not isinstance(values, list)
            or any(type(item) is not int or item < 0 for item in values)
            or values != sorted(set(values))
        ):
            errors.append(f"{name}.values must be sorted unique uint64 integers")
        elif type(count) is int and len(values) != count:
            errors.append(f"{name}.values cardinality differs from count")
        else:
            expected_digest = sha256_array(
                np.asarray(values, dtype=np.uint64)
            )
            if digest != expected_digest:
                errors.append(f"{name}.values digest mismatch")
    return tuple(errors)


def validate_raw_keypoint_successor_publication_receipt(
    receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the exact fail-closed envelope of a persisted successor."""

    errors: list[str] = []
    expected_outer = {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }
    if set(receipt) != expected_outer:
        errors.append("raw keypoint successor receipt has an unexpected field set")
    if (
        receipt.get("schema_id") != RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_ID
        or receipt.get("schema_version")
        != RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_VERSION
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("raw keypoint successor receipt schema header mismatch")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "raw keypoint successor receipt payload must be an object")
    expected_payload = {
        "status",
        "selector_eligible",
        "registry_registered",
        "parent_keypoint",
        "crop_successor",
        "output_keypoint",
        "reconciliation",
        "storage_profile_id",
        "selector_activation",
        "production_state_changes",
    }
    if set(payload) != expected_payload:
        errors.append(
            "raw keypoint successor receipt payload has an unexpected field set"
        )
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(
            f"raw keypoint successor receipt payload is not canonical JSON: {exc}"
        )
    else:
        if receipt.get("payload_digest") != expected_digest:
            errors.append("raw keypoint successor receipt payload digest mismatch")
    if payload.get("status") != "complete":
        errors.append("raw keypoint successor receipt status is not complete")
    if payload.get("selector_eligible") is not False:
        errors.append("raw keypoint successor receipt must remain selector-ineligible")
    if payload.get("registry_registered") is not False:
        errors.append("raw keypoint successor receipt must remain unregistered")
    if payload.get("selector_activation") != "none_direct_path_only":
        errors.append("raw keypoint successor selector activation is invalid")
    if payload.get("production_state_changes") != []:
        errors.append("raw keypoint successor reports production-state changes")
    parent = payload.get("parent_keypoint")
    if not isinstance(parent, Mapping) or set(parent) != {
        "run_id",
        "run_manifest_digest",
    }:
        errors.append("raw keypoint successor parent binding is invalid")
    crop = payload.get("crop_successor")
    if not isinstance(crop, Mapping) or set(crop) != {
        "run_id",
        "run_manifest_digest",
        "successor_receipt_digest",
    }:
        errors.append("raw keypoint crop-successor binding is invalid")
    output = payload.get("output_keypoint")
    if not isinstance(output, Mapping) or set(output) != {
        "path",
        "run_id",
        "run_manifest_digest",
        "logical_content_digest",
    }:
        errors.append("raw keypoint successor output binding is invalid")
    reconciliation = payload.get("reconciliation")
    expected_reconciliation = {
        "schema_id",
        "schema_version",
        "parent_keypoint_run_id",
        "parent_keypoint_manifest_digest",
        "parent_crop_run_id",
        "parent_crop_manifest_digest",
        "target_crop_run_id",
        "target_crop_manifest_digest",
        "pose_model_binding",
        "preprocessing",
        "instance_keys",
        "row_coverage",
        "pending_row_count",
        "publication_authorized",
        "selector_activation",
        "production_state_changes",
    }
    if not isinstance(reconciliation, Mapping):
        errors.append("raw keypoint reconciliation must be an object")
    else:
        if set(reconciliation) != expected_reconciliation:
            errors.append("raw keypoint reconciliation has an unexpected field set")
        if (
            reconciliation.get("schema_id") != RAW_KEYPOINT_SUCCESSOR_SCHEMA_ID
            or reconciliation.get("schema_version")
            != RAW_KEYPOINT_SUCCESSOR_SCHEMA_VERSION
            or reconciliation.get("row_coverage")
            != "complete_target_crop_rowset"
            or reconciliation.get("pending_row_count") != 0
            or reconciliation.get("publication_authorized") is not False
            or reconciliation.get("selector_activation")
            != "none_preparation_only"
            or reconciliation.get("production_state_changes") != []
        ):
            errors.append("raw keypoint reconciliation safety envelope is invalid")
        key_sets = reconciliation.get("instance_keys")
        expected_key_sets = {
            "reused",
            "inference_succeeded",
            "inference_failed",
            "retired",
        }
        if not isinstance(key_sets, Mapping) or set(key_sets) != expected_key_sets:
            errors.append("raw keypoint reconciliation key sets are invalid")
        else:
            for name in sorted(expected_key_sets):
                errors.extend(
                    _validate_key_receipt(
                        key_sets[name],
                        name=f"raw keypoint reconciliation {name}",
                    )
                )
        if isinstance(parent, Mapping) and (
            parent.get("run_id") != reconciliation.get("parent_keypoint_run_id")
            or parent.get("run_manifest_digest")
            != reconciliation.get("parent_keypoint_manifest_digest")
        ):
            errors.append("raw keypoint parent and reconciliation bindings differ")
        if isinstance(crop, Mapping) and (
            crop.get("run_id") != reconciliation.get("target_crop_run_id")
            or crop.get("run_manifest_digest")
            != reconciliation.get("target_crop_manifest_digest")
        ):
            errors.append("raw keypoint crop and reconciliation bindings differ")
    return tuple(errors)


def publish_selector_ineligible_raw_keypoint_successor(
    parent: KeypointShadowPublication,
    crop_successor: CropGeometrySuccessorPublication,
    inference: TerminalKeypointInferenceBatch,
    *,
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_KEYPOINT_SHADOW_ROOT,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "raw_keypoint_successor",
) -> RawKeypointSuccessorPublication:
    """Publish one complete successor without touching selectors or registries."""

    prepared = prepare_raw_keypoint_successor(
        parent,
        crop_successor,
        inference,
    )
    publication = publish_selector_ineligible_keypoint_snapshot(
        prepared.prepared,
        destination=destination,
        run_id=run_id,
        shadow_root=shadow_root,
        storage_profile=storage_profile,
        created_by=created_by,
    )
    publication_errors = validate_keypoint_shadow_publication(publication)
    if publication_errors:
        raise RawKeypointSuccessorError(
            "Published raw-keypoint successor is invalid: "
            + "; ".join(publication_errors)
        )
    crop_receipt_digest = crop_successor.receipt.get("payload_digest")
    payload = {
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "parent_keypoint": {
            "run_id": parent.run_id,
            "run_manifest_digest": parent.manifest["payload_digest"],
        },
        "crop_successor": {
            "run_id": crop_successor.publication.run_id,
            "run_manifest_digest": crop_successor.publication.manifest[
                "payload_digest"
            ],
            "successor_receipt_digest": crop_receipt_digest,
        },
        "output_keypoint": {
            "path": str(publication.output_path),
            "run_id": publication.run_id,
            "run_manifest_digest": publication.manifest["payload_digest"],
            "logical_content_digest": publication.manifest["payload"][
                "logical_content"
            ]["digest"],
        },
        "reconciliation": dict(prepared.receipt),
        "storage_profile_id": storage_profile.profile_id,
        "selector_activation": "none_direct_path_only",
        "production_state_changes": [],
    }
    receipt = {
        "schema_id": RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_ID,
        "schema_version": RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    receipt_errors = validate_raw_keypoint_successor_publication_receipt(receipt)
    if receipt_errors:
        raise RawKeypointSuccessorError(
            "Raw-keypoint successor receipt is invalid: "
            + "; ".join(receipt_errors)
        )
    receipt_path = (
        publication.output_path / RAW_KEYPOINT_SUCCESSOR_PUBLICATION_RECEIPT_NAME
    )
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(
            receipt,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        handle.write("\n")
    return RawKeypointSuccessorPublication(
        publication=publication,
        prepared=prepared,
        receipt=receipt,
    )


__all__ = [
    "RAW_KEYPOINT_SUCCESSOR_PUBLICATION_RECEIPT_NAME",
    "RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_ID",
    "RAW_KEYPOINT_SUCCESSOR_PUBLICATION_SCHEMA_VERSION",
    "RAW_KEYPOINT_SUCCESSOR_SCHEMA_ID",
    "RAW_KEYPOINT_SUCCESSOR_SCHEMA_VERSION",
    "PreparedRawKeypointSuccessor",
    "RawKeypointSuccessorPublication",
    "RawKeypointSuccessorError",
    "TerminalKeypointInferenceBatch",
    "prepare_raw_keypoint_successor",
    "publish_selector_ineligible_raw_keypoint_successor",
    "validate_raw_keypoint_successor_publication_receipt",
]
