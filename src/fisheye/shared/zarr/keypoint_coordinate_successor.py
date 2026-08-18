"""Republish immutable raw keypoints with the canonical-v2 coordinate graph.

This migration never reruns inference and never advances a selector.  Payload
objects are hard-linked from one sealed raw-keypoint bundle member, while all
metadata objects are copied before the successor is modified.  The original
bundle authority remains the scientific authority for the observation values;
the new child supplies only a freshly validated coordinate publication.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import zarr

from fisheye.shared.artifact_fingerprint import CONTENT_FINGERPRINT_SCHEME
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.keypoint_coordinate_publication import (
    KEYPOINT_COORDINATE_CONTEXT_ATTR,
    KEYPOINT_COORDINATE_DERIVATION_ATTR,
    KEYPOINT_PUBLICATION_OWNER_ATTR,
    load_persisted_ineligible_keypoint_coordinate_surfaces,
    prepare_keypoint_coordinate_context,
    publish_keypoint_coordinate_surfaces,
    require_bound_ineligible_keypoint_coordinate_surfaces,
)
from fisheye.shared.model_input_transform import model_input_transform_from_attrs
from fisheye.shared.pose_model_schema_binding import pose_schema_from_model_binding
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.coordinate_successor_authority import (
    KEYPOINT_COORDINATE_SUCCESSOR_KIND,
    build_coordinate_successor_authority,
    load_coordinate_successor_authority,
    stamp_coordinate_successor_authority,
)
from fisheye.shared.zarr.coordinate_successor_files import (
    copy_metadata_and_link_payload,
    metadata_tree_sha256,
)
from fisheye.shared.zarr.keypoint_bundle_activation import (
    KEYPOINT_BUNDLE_AUTHORITY_ATTR,
    resolve_active_keypoint_bundle_from_root,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    build_keypoint_coordinate_successor_manifest,
    keypoint_preprocessing_from_manifest,
    validate_keypoint_publication,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_publication import (
    keypoint_metadata_declaration_maps,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    ATOMIC_PUBLICATION_OWNER_ATTR,
)
from fisheye.shared.zarr.keypoint_schema import KEYPOINT_SCHEMA_V2, KeypointDimensions
from fisheye.shared.zarr.keypoint_storage import plan_keypoint_storage
from fisheye.shared.zarr.historical_geometry_only_crop_adapter import (
    HISTORICAL_BBOX_NORMALIZATION_ATTR,
    bind_historical_bbox_normalization_to_successor,
    bind_persisted_padded_placement_record,
    bind_persisted_run_attribute_record,
    bind_historical_geometry_only_crop_source,
    historical_geometry_only_crop_loader,
    stamp_persisted_padded_placement_provenance,
)
from fisheye.shared.zarr.array_contracts import ArrayContract, DTypeContract
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import AccessPattern, ArrayIntent, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)


KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID = (
    "palette.keypoint_coordinate_successor.production_publication"
)
KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION = 1
KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY = "sealed_raw_bundle_member_hardlink_logical_payload_with_successor_owned_coordinate_auxiliaries_v2"
KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_ID = (
    "palette.keypoint_coordinate_successor.derived_auxiliary"
)
KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_VERSION = 1
KEYPOINT_COORDINATE_AUXILIARY_ATTR = "coordinate_successor_auxiliary_materialization"
KEYPOINT_COORDINATE_AUXILIARY_POLICY = (
    "successor_owned_derived_coordinates_not_keypoint_v2_logical_payload_v2"
)
SELECTOR_ACTIVATION_DEFERRED = "deferred_separate_reviewed_change"

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


def _require_run_id(value: object, *, label: str) -> str:
    result = str(value or "").strip()
    if not _RUN_ID.fullmatch(result):
        raise ValueError(f"{label} must be one safe nonempty run ID.")
    return result


def _selector_snapshot(root: Any) -> dict[str, Any]:
    family = root["keypoints_runs"]
    return {
        "family": {
            name: {"present": name in family.attrs, "value": family.attrs.get(name)}
            for name in _SELECTOR_ATTRS
        },
        "root_current_keypoint_group_path": {
            "present": "current_keypoint_group_path" in root.attrs,
            "value": root.attrs.get("current_keypoint_group_path"),
        },
        "root_bundle_authority_digest": (
            canonical_json_sha256(root.attrs[KEYPOINT_BUNDLE_AUTHORITY_ATTR])
            if isinstance(root.attrs.get(KEYPOINT_BUNDLE_AUTHORITY_ATTR), Mapping)
            else None
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_artifact(path: Path, *, pose_binding: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Keypoint model artifact not found: {resolved}")
    model = pose_binding.get("model")
    expected = model.get("sha256") if isinstance(model, Mapping) else None
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError("Pose binding lacks an exact keypoint-model SHA-256.")
    actual = _sha256_file(resolved)
    if actual != expected:
        raise ValueError("Keypoint model bytes differ from the bound pose schema.")
    stat = resolved.stat()
    return {
        "role": "keypoint_model",
        "path": str(resolved),
        "fingerprint_scheme": CONTENT_FINGERPRINT_SCHEME,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": actual,
        "source": "coordinate_successor_revalidation",
        "pose_schema_binding": copy.deepcopy(dict(pose_binding)),
    }


def _source_bundle_authority(
    root: Any,
    *,
    run_path: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = resolve_active_keypoint_bundle_from_root(root)
    if not isinstance(resolved, Mapping) or not isinstance(
        resolved.get("authority"), Mapping
    ):
        raise ValueError("Archive lacks a committed keypoint bundle authority.")
    authority = dict(resolved["authority"])
    members = authority.get("members")
    raw = members.get("raw_keypoints") if isinstance(members, Mapping) else None
    logical = source_manifest.get("payload", {}).get("logical_content", {})
    if (
        not isinstance(raw, Mapping)
        or raw.get("run_path") != run_path
        or raw.get("manifest_payload_digest") != source_manifest.get("payload_digest")
        or raw.get("manifest_document_digest") != canonical_json_sha256(source_manifest)
        or raw.get("logical_content_digest") != logical.get("digest")
    ):
        raise ValueError(
            "Committed keypoint bundle authority does not bind the exact source run."
        )
    return authority


def _dimensions(payload: Mapping[str, Any]) -> KeypointDimensions:
    raw = payload["logical_schema"]["dimensions"]
    return KeypointDimensions(
        n_frames=raw["n_frames"],
        n_instances=raw["n_instances"],
        n_keypoints=raw["n_keypoints"],
        source_width=raw["source_width"],
        source_height=raw["source_height"],
    )


def _submitted_input_mode(preprocessing: Any) -> str:
    """Resolve the actual model submission mode, not the outer cache reader."""

    value = preprocessing.document.get("model_input_mode")
    if value is None and preprocessing.input_mode in {"numpy-list", "tensor"}:
        value = preprocessing.input_mode
    if value not in {"numpy-list", "tensor"}:
        raise ValueError(
            "Keypoint preprocessing lacks an exact submitted model input mode."
        )
    return str(value)


def _keypoint_semantic_attrs(
    payload: Mapping[str, Any],
    *,
    model_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover exact writer attrs from the digest-bound manifest authority."""

    model_sha256 = model_artifact.get("sha256")
    pose_schema, pose_schema_attrs, validated_binding = pose_schema_from_model_binding(
        payload["pose_model_schema_binding"],
        expected_model_sha256=str(model_sha256 or ""),
    )
    if validated_binding != payload["pose_model_schema_binding"]:
        raise ValueError(
            "Validated keypoint model-schema binding differs from the immutable source manifest."
        )
    labels = list(pose_schema.node_names)
    dimensions = payload["logical_schema"]["dimensions"]
    if (
        not labels
        or len(labels) != int(dimensions["n_keypoints"])
        or pose_schema_attrs.get("keypoint_labels") != labels
        or pose_schema_attrs.get("kpt_shape") != [len(labels), 2]
    ):
        raise ValueError(
            "Keypoint pose schema differs from the immutable logical dimensions."
        )
    model_kpt_shape = pose_schema_attrs.get("metadata", {}).get("model_kpt_shape")
    if (
        type(model_kpt_shape) is not list
        or len(model_kpt_shape) != 2
        or any(type(value) is not int for value in model_kpt_shape)
        or model_kpt_shape[0] != len(labels)
        or model_kpt_shape[1] <= 0
    ):
        raise ValueError(
            "Keypoint model-schema binding lacks an exact compatible model_kpt_shape."
        )
    return {
        "keypoint_labels": labels,
        "keypoint_confidence_labels": list(labels),
        "skeleton_id": str(pose_schema_attrs["skeleton_id"]),
        "kpt_shape": [len(labels), 2],
        "model_kpt_shape": copy.deepcopy(model_kpt_shape),
        "pose_schema": copy.deepcopy(pose_schema_attrs),
    }


@dataclass(frozen=True)
class _KeypointAuxiliaryMaterializationPlan:
    """Read-only values and storage contracts for successor-owned auxiliaries."""

    values: Mapping[str, np.ndarray]
    storage: Mapping[str, Mapping[str, Any]]
    source_existing: Mapping[str, Mapping[str, Any]]
    source_crop_row_ids_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_id": KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_ID,
            "schema_version": KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_VERSION,
            "policy": KEYPOINT_COORDINATE_AUXILIARY_POLICY,
            "write_ownership": "single_writer_immutable_materialization",
            "source_crop_row_ids_sha256": self.source_crop_row_ids_sha256,
            "arrays": {
                name: {
                    "action": "materialize_successor_owned",
                    "source": (
                        "exact_crop_rows_from_source_crop_row_ids"
                        if name == "source_crop_xywh"
                        else "validated_source_keypoints_img_or_pose_bbox_img"
                    ),
                    "shape": [int(value) for value in values.shape],
                    "dtype": values.dtype.str,
                    "sha256": sha256_array(values),
                    "storage": dict(self.storage[name]),
                }
                for name, values in self.values.items()
            },
            "validated_existing_source_arrays": {
                name: dict(value) for name, value in self.source_existing.items()
            },
            "logical_payload_scope": {
                "schema_id": KEYPOINT_SCHEMA_V2.schema_id,
                "schema_version": KEYPOINT_SCHEMA_V2.schema_version,
                "auxiliary_arrays_excluded": True,
                "source_schema_arrays_remain_hardlinked": True,
            },
        }


def _read_successor_source_array(run: Any, name: str) -> np.ndarray:
    try:
        node = run[name]
        values = np.asarray(node[...])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Keypoint source array {name!r} is unavailable: {exc}"
        ) from exc
    return np.ascontiguousarray(values)


def _derive_historical_image_coordinates(
    roi_values: np.ndarray,
    placement: np.ndarray,
    *,
    roi_width: int,
    roi_height: int,
) -> np.ndarray:
    values = np.asarray(roi_values)
    if values.dtype.kind != "f" or np.isinf(values).any():
        raise ValueError(
            "Historical ROI coordinates must be finite-or-NaN floating values."
        )
    bbox = values.ndim >= 2 and values.shape[-1] == 4
    shaped = values.reshape(*values.shape[:-1], 2, 2) if bbox else values
    finite = np.isfinite(shaped)
    filled = np.where(finite, shaped, np.zeros((), dtype=values.dtype))
    scale = placement[:, 2:].astype(np.float64) / np.asarray(
        [roi_width, roi_height], dtype=np.float64
    )
    shape = (placement.shape[0],) + (1,) * (filled.ndim - 2) + (2,)
    projected = filled.astype(np.float64) * scale.reshape(shape) + placement[
        :, :2
    ].reshape(shape)
    result = np.asarray(projected, dtype=values.dtype)
    result[~finite] = np.nan
    return result.reshape(values.shape) if bbox else result


def _normalize_historical_image_coordinates(
    values: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind != "f" or np.isinf(source).any():
        raise ValueError(
            "Historical image coordinates must be finite-or-NaN floating values."
        )
    factors = np.asarray(
        [width, height] if source.shape[-1] == 2 else [width, height, width, height],
        dtype=np.float64,
    )
    result = np.asarray(source.astype(np.float64) / factors, dtype=source.dtype)
    result[np.isnan(source)] = np.nan
    return result


def _auxiliary_array_contract_and_plan(
    name: str,
    values: np.ndarray,
    *,
    profile: Any,
) -> tuple[ArrayContract, Any]:
    shape = tuple(int(value) for value in values.shape)
    dimensions = {
        "n_instances": int(shape[0]),
        "n_keypoints": int(shape[1]) if len(shape) == 3 else 0,
    }
    schema_id = f"{KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_ID}.{name}"
    contract = ArrayContract(
        schema_id=schema_id,
        schema_version=KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_VERSION,
        dtype=DTypeContract(
            dtype_id=f"numpy:{values.dtype.str}",
            numpy_dtype=values.dtype.str,
        ),
        shape_template=(
            ("n_instances", "n_keypoints", 2) if len(shape) == 3 else ("n_instances", 4)
        ),
        axis_names=(
            ("observation_instance", "keypoint", "coordinate")
            if len(shape) == 3
            else ("observation_instance", "coordinate")
        ),
        description=(
            "Successor-owned derived coordinate auxiliary; excluded from frozen keypoint-v2 logical payload."
        ),
        units="px" if name == "source_crop_xywh" else None,
        coordinate_space=(
            "source_camera_xywh" if name == "source_crop_xywh" else "source_camera"
        ),
    )
    shape_errors = contract.validate_shape(shape, dimensions=dimensions)
    if shape_errors:
        raise ValueError(
            f"Auxiliary {name} shape is invalid: {'; '.join(shape_errors)}"
        )
    intent = ArrayIntent(
        shape=shape,
        dtype=values.dtype,
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        logical_schema_id=contract.schema_id,
        logical_schema_version=contract.schema_version,
        access_unit_shape=((1, shape[1], 2) if len(shape) == 3 else (1, shape[1])),
        name=name,
    )
    return contract, plan_storage(intent, profile)


def _plan_keypoint_auxiliaries(
    *,
    source_run: Any,
    source_manifest: Mapping[str, Any],
    historical_crop: Any,
) -> _KeypointAuxiliaryMaterializationPlan:
    rows = _read_successor_source_array(source_run, "source_crop_row_ids")
    placement_source = historical_crop.source._rowset_node["source_crop_xywh"]
    placement = np.ascontiguousarray(np.asarray(placement_source[rows]))
    expected_count = int(rows.shape[0])
    if rows.dtype != np.dtype("<i8") or rows.ndim != 1 or expected_count == 0:
        raise ValueError(
            "Keypoint source_crop_row_ids must be a nonempty little-endian int64 row array."
        )
    if placement.shape != (expected_count, 4):
        raise ValueError(
            "Historical crop placement does not cover every keypoint source row."
        )
    roi_width = int(historical_crop.source.roi_frame.endpoint.width)
    roi_height = int(historical_crop.source.roi_frame.endpoint.height)
    # The adapter has already validated the full crop. Re-check the selected
    # placement values here so this preflight is the exact apply input.
    if placement.dtype != np.dtype("<f4") or not np.isfinite(placement).all():
        raise ValueError(
            "Selected historical crop placement is not exact finite float32 xywh."
        )
    keypoints_roi = _read_successor_source_array(source_run, "keypoints_roi")
    keypoints_img = _read_successor_source_array(source_run, "keypoints_img")
    bbox_roi = _read_successor_source_array(source_run, "pose_bbox_xyxy_roi")
    bbox_img = _read_successor_source_array(source_run, "pose_bbox_xyxy_img")
    if keypoints_roi.shape[0] != expected_count or bbox_roi.shape != (
        expected_count,
        4,
    ):
        raise ValueError(
            "Historical keypoint ROI arrays do not cover the exact source rowset."
        )
    if keypoints_img.shape != keypoints_roi.shape or bbox_img.shape != bbox_roi.shape:
        raise ValueError(
            "Historical keypoint image arrays do not match ROI array shapes."
        )
    expected_keypoints_img = _derive_historical_image_coordinates(
        keypoints_roi, placement, roi_width=roi_width, roi_height=roi_height
    )
    expected_bbox_img = _derive_historical_image_coordinates(
        bbox_roi, placement, roi_width=roi_width, roi_height=roi_height
    )
    if keypoints_img.dtype != keypoints_roi.dtype or not np.array_equal(
        keypoints_img, expected_keypoints_img, equal_nan=True
    ):
        raise ValueError(
            "Historical keypoints_img is not an exact dtype-preserving ROI+placement derivation."
        )
    if bbox_img.dtype != bbox_roi.dtype or not np.array_equal(
        bbox_img, expected_bbox_img, equal_nan=True
    ):
        raise ValueError(
            "Historical pose_bbox_xyxy_img is not an exact dtype-preserving ROI+placement derivation."
        )
    keypoints_norm = _normalize_historical_image_coordinates(
        expected_keypoints_img,
        width=historical_crop.source.crop_geometry.source_geometry.frame_evidence.source_camera_frame.endpoint.width,
        height=historical_crop.source.crop_geometry.source_geometry.frame_evidence.source_camera_frame.endpoint.height,
    )
    bbox_norm = _normalize_historical_image_coordinates(
        expected_bbox_img,
        width=historical_crop.source.crop_geometry.source_geometry.frame_evidence.bbox_source_camera_frame.endpoint.width,
        height=historical_crop.source.crop_geometry.source_geometry.frame_evidence.bbox_source_camera_frame.endpoint.height,
    )
    profile = storage_profile_from_manifest(
        source_manifest["payload"]["storage_plan"]["storage_profile"]
    )
    values = {
        "source_crop_xywh": np.array(placement, copy=True, order="C"),
        "keypoints_norm": np.array(keypoints_norm, copy=True, order="C"),
        "pose_bbox_xyxy_norm": np.array(bbox_norm, copy=True, order="C"),
    }
    storage: dict[str, Mapping[str, Any]] = {}
    for name, value in values.items():
        _contract, plan = _auxiliary_array_contract_and_plan(
            name, value, profile=profile
        )
        storage[name] = plan.as_dict()
    source_existing = {
        name: {
            "shape": [int(value) for value in array.shape],
            "dtype": array.dtype.str,
            "sha256": sha256_array(array),
            "derivation": "validated_exact_roi_plus_requested_crop_placement_v1",
        }
        for name, array in {
            "keypoints_img": keypoints_img,
            "pose_bbox_xyxy_img": bbox_img,
        }.items()
    }
    return _KeypointAuxiliaryMaterializationPlan(
        values=values,
        storage=storage,
        source_existing=source_existing,
        source_crop_row_ids_sha256=sha256_array(rows),
    )


def inspect_keypoint_coordinate_successor_source(
    *,
    analysis_zarr: Path,
    source_run_id: str,
    successor_run_id: str,
    keypoint_model_path: Path,
) -> dict[str, Any]:
    """Validate one source and produce a read-only successor plan."""

    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    source_id = _require_run_id(source_run_id, label="source_run_id")
    successor_id = _require_run_id(successor_run_id, label="successor_run_id")
    if source_id == successor_id:
        raise ValueError("Keypoint coordinate successor cannot replace its source.")
    source_path = archive / "keypoints_runs" / source_id
    target_path = archive / "keypoints_runs" / successor_id
    if not source_path.is_dir():
        raise FileNotFoundError(f"Raw keypoint source not found: {source_path}")
    if target_path.exists():
        raise FileExistsError(f"Immutable successor target exists: {target_path}")

    root = open_zarr_root(archive, mode="r")
    source = root[f"keypoints_runs/{source_id}"]
    manifest = source.attrs.get(KEYPOINT_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError("Raw keypoint source lacks an immutable run manifest.")
    errors = validate_keypoint_run_manifest(manifest)
    if errors:
        raise ValueError(
            "Raw keypoint source manifest is invalid: " + "; ".join(errors)
        )
    payload = manifest["payload"]
    if payload.get("run_id") != source_id:
        raise ValueError("Raw keypoint source manifest binds another run ID.")
    if (
        source.attrs.get("status") != RUN_STATUS_COMPLETE
        or source.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or source.attrs.get("stage_selector_eligible") is not False
        or source.attrs.get("production_candidate") is not True
    ):
        raise ValueError("Raw keypoint source is not one complete sealed candidate.")
    authority = _source_bundle_authority(
        root,
        run_path=f"keypoints_runs/{source_id}",
        source_manifest=manifest,
    )
    dimensions = _dimensions(payload)
    profile = storage_profile_from_manifest(payload["storage_plan"]["storage_profile"])
    plans = plan_keypoint_storage(dimensions, profile=profile)
    direct, consolidated = keypoint_metadata_declaration_maps(
        archive, run_id=source_id, plans=plans
    )
    crop_manifest, crop_arrays = _source_crop_manifest_and_arrays(root, payload)
    source_errors = validate_keypoint_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays={name: source[name] for name in KEYPOINT_SCHEMA_V2.binding_paths},
        source_crop_arrays=crop_arrays,
        source_crop_manifest=crop_manifest,
    )
    if source_errors:
        raise ValueError(
            "Raw keypoint source publication is invalid: " + "; ".join(source_errors)
        )
    preprocessing = keypoint_preprocessing_from_manifest(payload["preprocessing"])
    transform_value = preprocessing.document.get("model_input_transform")
    if not isinstance(transform_value, Mapping):
        raise ValueError("Raw keypoint preprocessing lacks model_input_transform.")
    transform = model_input_transform_from_attrs(dict(transform_value))
    historical_crop = bind_historical_geometry_only_crop_source(
        analysis_zarr=archive,
        root=root,
        crop_reference=payload["source_crop_snapshot"],
        source_manifest=manifest,
        source_arrays={
            "source_crop_row_ids": source["source_crop_row_ids"],
            "instance_key": source["instance_key"],
            "source_acquisition_frame_index": source["source_acquisition_frame_index"],
            "source_crop_row_signature": source["source_crop_row_signature"],
        },
        source_run_path=f"keypoints_runs/{source_id}",
        model_input_transform=transform,
    )
    artifact = _model_artifact(
        keypoint_model_path,
        pose_binding=payload["pose_model_schema_binding"],
    )
    semantic_attrs = _keypoint_semantic_attrs(payload, model_artifact=artifact)
    auxiliary_plan = _plan_keypoint_auxiliaries(
        source_run=source,
        source_manifest=manifest,
        historical_crop=historical_crop,
    )
    crop = payload["source_crop_snapshot"]
    return json_attr_safe(
        {
            "schema_id": "palette.keypoint_coordinate_successor.source_inspection",
            "schema_version": 1,
            "status": "ready",
            "analysis_zarr": str(archive),
            "source_run_id": source_id,
            "source_run_path": f"keypoints_runs/{source_id}",
            "successor_run_id": successor_id,
            "successor_run_path": f"keypoints_runs/{successor_id}",
            "source_manifest_digest": canonical_json_sha256(manifest),
            "source_manifest_payload_digest": manifest["payload_digest"],
            "source_logical_content_digest": payload["logical_content"]["digest"],
            "source_metadata_tree_sha256": metadata_tree_sha256(source_path),
            "source_authority_digest": canonical_json_sha256(authority),
            "source_crop_path": crop["run_path"],
            "historical_crop_adapter": historical_crop.as_record(),
            "auxiliary_materialization": auxiliary_plan.as_record(),
            "preprocessing_input_mode": _submitted_input_mode(preprocessing),
            "model_input_transform": transform.to_attrs(),
            "model_artifact": artifact,
            "keypoint_semantic_attrs": semantic_attrs,
            "selectors_before": _selector_snapshot(root),
            "selector_eligible": False,
            "selector_activation": SELECTOR_ACTIVATION_DEFERRED,
            "registry_updated": False,
        }
    )


def _coordinate_record_pointers(surfaces: Any) -> dict[str, dict[str, str]]:
    values = {
        "context": surfaces.context.context_record,
        "row_identity": surfaces.context.row_identity,
        "temporal_authority": surfaces.context.temporal_authority,
        "derivation": surfaces.derivation,
    }
    return {
        name: {
            "record_ref": value.record_ref,
            "record_sha256": value.record_sha256,
        }
        for name, value in values.items()
    }


def _source_crop_manifest_and_arrays(
    root: Any, payload: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    crop_snapshot = payload["source_crop_snapshot"]
    crop_path = str(crop_snapshot["run_path"])
    crop = root[crop_path]
    manifest = crop.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Bound crop source lacks its run manifest.")
    arrays = {
        name: crop[name]
        for name in (
            "instance_key",
            "frame_indices",
            "source_acquisition_frame_index",
            "source_row_signature",
            "roi_coordinates_full",
            "roi_sizes_full",
        )
    }
    return manifest, arrays


def _materialize_keypoint_auxiliaries(
    run: Any,
    *,
    plan: _KeypointAuxiliaryMaterializationPlan,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create fresh target-owned auxiliary arrays from the read-only plan."""

    profile = storage_profile_from_manifest(
        source_manifest["payload"]["storage_plan"]["storage_profile"]
    )
    receipts: dict[str, Any] = {}
    for name, values in plan.values.items():
        if name in run:
            raise ValueError(
                f"Successor target already contains auxiliary array {name!r}; refusing reuse or overwrite."
            )
        contract, storage_plan = _auxiliary_array_contract_and_plan(
            name, values, profile=profile
        )
        if storage_plan.as_dict() != dict(plan.storage[name]):
            raise RuntimeError(
                f"Auxiliary storage plan changed between dry-run and apply for {name!r}."
            )
        target = create_array_from_plan(
            run,
            name=name,
            contract=contract,
            plan=storage_plan,
            fill_value=np.nan if values.dtype.kind == "f" else 0,
            attributes={
                "coordinate_successor_auxiliary": True,
                "coordinate_successor_auxiliary_policy": KEYPOINT_COORDINATE_AUXILIARY_POLICY,
                "source_crop_row_ids_sha256": plan.source_crop_row_ids_sha256,
            },
        )
        target[...] = values
        observed = np.asarray(target[...])
        if (
            observed.dtype != values.dtype
            or observed.shape != values.shape
            or not np.array_equal(observed, values, equal_nan=True)
        ):
            raise RuntimeError(
                f"Materialized keypoint auxiliary {name!r} failed exact readback validation."
            )
        receipts[name] = {
            "target_path": f"{run.path}/{name}",
            "shape": [int(value) for value in observed.shape],
            "dtype": observed.dtype.str,
            "sha256": sha256_array(observed),
            "source_plan_sha256": sha256_array(values),
            "materialization": "fresh_target_owned_single_writer_v1",
            "physical_chunks": {
                "chunk_shape": [int(value) for value in storage_plan.chunk_shape or ()],
                "shard_shape": [int(value) for value in storage_plan.shard_shape or ()]
                if storage_plan.shard_shape
                else None,
                "write_ownership": storage_plan.write_ownership,
            },
        }
    return {
        "schema_id": KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_ID,
        "schema_version": KEYPOINT_COORDINATE_AUXILIARY_SCHEMA_VERSION,
        "policy": KEYPOINT_COORDINATE_AUXILIARY_POLICY,
        "arrays": receipts,
    }


def publish_keypoint_coordinate_successor(
    *,
    analysis_zarr: Path,
    source_run_id: str,
    successor_run_id: str,
    keypoint_model_path: Path,
) -> dict[str, Any]:
    """Publish one complete selector-ineligible coordinate successor."""

    initial = inspect_keypoint_coordinate_successor_source(
        analysis_zarr=analysis_zarr,
        source_run_id=source_run_id,
        successor_run_id=successor_run_id,
        keypoint_model_path=keypoint_model_path,
    )
    archive = Path(initial["analysis_zarr"])
    source_id = str(initial["source_run_id"])
    successor_id = str(initial["successor_run_id"])
    source_path = archive / "keypoints_runs" / source_id
    target_path = archive / "keypoints_runs" / successor_id
    copy_receipt: dict[str, int] | None = None
    auxiliary_receipt: dict[str, Any] | None = None

    with archive_metadata_publication_lock(archive):
        checked = inspect_keypoint_coordinate_successor_source(
            analysis_zarr=archive,
            source_run_id=source_id,
            successor_run_id=successor_id,
            keypoint_model_path=keypoint_model_path,
        )
        if checked["selectors_before"] != initial["selectors_before"]:
            raise RuntimeError("Keypoint selectors changed after successor planning.")
        root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
        source = root[f"keypoints_runs/{source_id}"]
        source_manifest = copy.deepcopy(
            dict(source.attrs[KEYPOINT_RUN_MANIFEST_ATTRIBUTE])
        )
        source_authority = _source_bundle_authority(
            root,
            run_path=f"keypoints_runs/{source_id}",
            source_manifest=source_manifest,
        )
        payload = source_manifest["payload"]
        preprocessing = keypoint_preprocessing_from_manifest(payload["preprocessing"])
        transform = model_input_transform_from_attrs(
            dict(preprocessing.document["model_input_transform"])
        )
        artifact = _model_artifact(
            keypoint_model_path,
            pose_binding=payload["pose_model_schema_binding"],
        )
        semantic_attrs = _keypoint_semantic_attrs(payload, model_artifact=artifact)
        if semantic_attrs != checked["keypoint_semantic_attrs"]:
            raise RuntimeError(
                "Keypoint semantic attrs changed between dry-run and apply."
            )
        historical_crop = bind_historical_geometry_only_crop_source(
            analysis_zarr=archive,
            root=root,
            crop_reference=source_manifest["payload"]["source_crop_snapshot"],
            source_manifest=source_manifest,
            source_arrays={
                "source_crop_row_ids": source["source_crop_row_ids"],
                "instance_key": source["instance_key"],
                "source_acquisition_frame_index": source[
                    "source_acquisition_frame_index"
                ],
                "source_crop_row_signature": source["source_crop_row_signature"],
            },
            source_run_path=f"keypoints_runs/{source_id}",
            model_input_transform=transform,
        )
        auxiliary_plan = _plan_keypoint_auxiliaries(
            source_run=source,
            source_manifest=source_manifest,
            historical_crop=historical_crop,
        )
        if auxiliary_plan.as_record() != checked["auxiliary_materialization"]:
            raise RuntimeError(
                "Keypoint auxiliary materialization plan changed between dry-run and apply."
            )
        try:
            copy_receipt = copy_metadata_and_link_payload(source_path, target_path)
            run = root[f"keypoints_runs/{successor_id}"]
            attrs = dict(run.attrs)
            for name in (
                KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
                KEYPOINT_COORDINATE_CONTEXT_ATTR,
                KEYPOINT_COORDINATE_DERIVATION_ATTR,
                "coordinate_successor_authority",
                "coordinate_successor_authority_sha256",
                KEYPOINT_COORDINATE_AUXILIARY_ATTR,
                f"{KEYPOINT_COORDINATE_AUXILIARY_ATTR}_sha256",
                "coordinate_successor_padded_crop_lineage",
                "coordinate_successor_padded_crop_lineage_sha256",
                HISTORICAL_BBOX_NORMALIZATION_ATTR,
                f"{HISTORICAL_BBOX_NORMALIZATION_ATTR}_sha256",
                RUN_COMPLETED_AT_ATTR,
                "palette_run_failed_at_utc",
                "palette_run_error",
            ):
                attrs.pop(name, None)
            owner = uuid4().hex
            attrs.update(
                {
                    "status": "running",
                    "stage_selector_eligible": False,
                    "production_candidate": True,
                    "production_selector_activation": SELECTOR_ACTIVATION_DEFERRED,
                    "coordinate_contract": "coordinate_successor_preparing",
                    "coordinate_successor_policy": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY,
                    "coordinate_successor_source_run_path": f"keypoints_runs/{source_id}",
                    "coordinate_successor_historical_crop_adapter": historical_crop.as_record(),
                    "coordinate_successor_historical_crop_adapter_sha256": canonical_json_sha256(
                        historical_crop.as_record()
                    ),
                    **semantic_attrs,
                    ATOMIC_PUBLICATION_OWNER_ATTR: owner,
                    KEYPOINT_PUBLICATION_OWNER_ATTR: owner,
                }
            )
            run.attrs.put(attrs)
            mark_run_started(run, run_name=successor_id, stage="keypoints")

            auxiliary_receipt = _materialize_keypoint_auxiliaries(
                run,
                plan=auxiliary_plan,
                source_manifest=source_manifest,
            )
            run.attrs[KEYPOINT_COORDINATE_AUXILIARY_ATTR] = {
                **auxiliary_receipt,
                "plan": auxiliary_plan.as_record(),
                "authority": "successor_owned_derived_coordinate_auxiliaries",
                "logical_payload_exclusion": {
                    "schema_id": KEYPOINT_SCHEMA_V2.schema_id,
                    "schema_version": KEYPOINT_SCHEMA_V2.schema_version,
                    "arrays_excluded": [
                        "source_crop_xywh",
                        "keypoints_norm",
                        "pose_bbox_xyxy_norm",
                    ],
                },
            }
            run.attrs[f"{KEYPOINT_COORDINATE_AUXILIARY_ATTR}_sha256"] = (
                canonical_json_sha256(run.attrs[KEYPOINT_COORDINATE_AUXILIARY_ATTR])
            )
            stamp_persisted_padded_placement_provenance(run, historical_crop)
            publication_crop = bind_historical_bbox_normalization_to_successor(
                historical_crop,
                root=root,
                successor_run=run,
                successor_run_path=f"keypoints_runs/{successor_id}",
            )

            payload = source_manifest["payload"]
            preprocessing = keypoint_preprocessing_from_manifest(
                payload["preprocessing"]
            )
            transform = model_input_transform_from_attrs(
                dict(preprocessing.document["model_input_transform"])
            )
            with historical_geometry_only_crop_loader(publication_crop):
                prepare_keypoint_coordinate_context(
                    root,
                    f"keypoints_runs/{successor_id}",
                    crop_path=str(payload["source_crop_snapshot"]["run_path"]),
                    model_input_transform=transform,
                    preprocessing_input_mode=_submitted_input_mode(preprocessing),
                    model_artifact=artifact,
                )
                surfaces = publish_keypoint_coordinate_surfaces(
                    root, f"keypoints_runs/{successor_id}"
                )
            run.attrs["coordinate_successor_padded_crop_lineage"] = {
                "source_crop_adapter": historical_crop.as_record(),
                "placement_ownership": bind_persisted_padded_placement_record(run),
            }
            run.attrs["coordinate_successor_padded_crop_lineage_sha256"] = (
                canonical_json_sha256(
                    run.attrs["coordinate_successor_padded_crop_lineage"]
                )
            )
            padded_lineage_record = bind_persisted_run_attribute_record(
                run, attr_name="coordinate_successor_padded_crop_lineage"
            )
            auxiliary_record = bind_persisted_run_attribute_record(
                run, attr_name=KEYPOINT_COORDINATE_AUXILIARY_ATTR
            )
            bbox_normalization_record = bind_persisted_run_attribute_record(
                run, attr_name=HISTORICAL_BBOX_NORMALIZATION_ATTR
            )
            authority = build_coordinate_successor_authority(
                kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
                source_family="keypoints_runs",
                source_run_path=f"keypoints_runs/{source_id}",
                source_manifest=source_manifest,
                source_authority_kind="committed_root_keypoint_bundle_authority_v1",
                source_authority=source_authority,
                successor_family="keypoints_runs",
                successor_run_path=f"keypoints_runs/{successor_id}",
                payload_equivalence={
                    "policy": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY,
                    "source_logical_content_digest": payload["logical_content"][
                        "digest"
                    ],
                    "logical_payload": {
                        "schema_id": KEYPOINT_SCHEMA_V2.schema_id,
                        "schema_version": KEYPOINT_SCHEMA_V2.schema_version,
                        "byte_identical_hardlinked": True,
                        "source_schema_arrays_remain_hardlinked": True,
                        "auxiliary_arrays_excluded": [
                            "source_crop_xywh",
                            "keypoints_norm",
                            "pose_bbox_xyxy_norm",
                        ],
                    },
                    "auxiliary_materialization_record": auxiliary_record,
                    "auxiliary_materialization_receipt": auxiliary_receipt,
                    **dict(copy_receipt),
                },
                coordinate_records={
                    **_coordinate_record_pointers(surfaces),
                    "auxiliary_materialization": auxiliary_record,
                    "padded_crop_lineage": padded_lineage_record,
                    "historical_bbox_normalization": bbox_normalization_record,
                },
            )
            stamp_coordinate_successor_authority(run, authority)
            run.attrs["status"] = RUN_STATUS_COMPLETE
            mark_run_complete(run, run_name=successor_id)

            consolidate_metadata_capture_expected_warnings(archive)
            dimensions = _dimensions(payload)
            profile = storage_profile_from_manifest(
                payload["storage_plan"]["storage_profile"]
            )
            plans = plan_keypoint_storage(dimensions, profile=profile)
            direct, consolidated = keypoint_metadata_declaration_maps(
                archive, run_id=successor_id, plans=plans
            )
            successor_manifest = build_keypoint_coordinate_successor_manifest(
                source_manifest,
                run_id=successor_id,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
            )
            run.attrs[KEYPOINT_RUN_MANIFEST_ATTRIBUTE] = successor_manifest
            consolidate_metadata_capture_expected_warnings(archive)

            published = open_zarr_root(archive, mode="r")
            if _selector_snapshot(published) != initial["selectors_before"]:
                raise RuntimeError(
                    "Keypoint selectors changed during successor publication."
                )
            published_run = published[f"keypoints_runs/{successor_id}"]
            with historical_geometry_only_crop_loader(publication_crop):
                published_surfaces = (
                    require_bound_ineligible_keypoint_coordinate_surfaces(
                        load_persisted_ineligible_keypoint_coordinate_surfaces(
                            published, f"keypoints_runs/{successor_id}"
                        )
                    )
                )
            load_coordinate_successor_authority(
                published_run,
                expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
                expected_successor_run_path=f"keypoints_runs/{successor_id}",
            )
            direct, consolidated = keypoint_metadata_declaration_maps(
                archive, run_id=successor_id, plans=plans
            )
            crop_manifest, crop_arrays = _source_crop_manifest_and_arrays(
                published, payload
            )
            errors = validate_keypoint_publication(
                successor_manifest,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
                arrays={
                    name: published_run[name]
                    for name in KEYPOINT_SCHEMA_V2.binding_paths
                },
                source_crop_arrays=crop_arrays,
                source_crop_manifest=crop_manifest,
            )
            if errors:
                raise RuntimeError(
                    "Published keypoint successor failed validation: "
                    + "; ".join(errors)
                )
            del published_surfaces
            if (
                metadata_tree_sha256(source_path)
                != initial["source_metadata_tree_sha256"]
            ):
                raise RuntimeError("Immutable source keypoint metadata changed.")
        except BaseException as exc:
            if target_path.exists():
                try:
                    failed_root = zarr.open_group(
                        str(archive), mode="a", use_consolidated=False
                    )
                    failed = failed_root[f"keypoints_runs/{successor_id}"]
                    failed.attrs["status"] = "failed"
                    failed.attrs["stage_selector_eligible"] = False
                    mark_run_failed(failed, run_name=successor_id, error=str(exc))
                    consolidate_metadata_capture_expected_warnings(archive)
                except BaseException:
                    pass
            raise

    return json_attr_safe(
        {
            "schema_id": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID,
            "schema_version": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "published_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "source_run_path": f"keypoints_runs/{source_id}",
            "successor_run_path": f"keypoints_runs/{successor_id}",
            "source_manifest_digest": initial["source_manifest_digest"],
            "source_metadata_tree_sha256": initial["source_metadata_tree_sha256"],
            "historical_crop_adapter": initial["historical_crop_adapter"],
            "auxiliary_materialization_plan": initial["auxiliary_materialization"],
            "auxiliary_materialization": auxiliary_receipt,
            "copy": copy_receipt,
            "coordinate_contract": "canonical_v2",
            "selector_eligible": False,
            "selectors_before": initial["selectors_before"],
            "selectors_after": initial["selectors_before"],
            "selector_activation": SELECTOR_ACTIVATION_DEFERRED,
            "registry_updated": False,
        }
    )


__all__ = [
    "KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY",
    "KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID",
    "KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_VERSION",
    "inspect_keypoint_coordinate_successor_source",
    "publish_keypoint_coordinate_successor",
]
