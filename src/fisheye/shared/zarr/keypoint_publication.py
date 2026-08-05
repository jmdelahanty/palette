"""Selector-ineligible immutable publication for raw keypoint-v2 snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.keypoint_manifest import (
    KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    KeypointCropSourceReference,
    KeypointPreprocessingReference,
    build_keypoint_run_manifest,
    keypoint_crop_source_from_manifest,
    keypoint_skeleton_digest,
    validate_keypoint_publication,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
    derive_frame_row_offsets,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    KeypointPublicationDisposition,
)
from fisheye.shared.zarr.keypoint_storage import (
    KeypointStoragePlanSet,
    plan_keypoint_storage,
)
from fisheye.shared.zarr.storage_intent import StoragePlan
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)


KEYPOINT_SHADOW_SCHEMA_ID = "palette.keypoint.shadow_publication"
KEYPOINT_SHADOW_SCHEMA_VERSION = 1
DEFAULT_KEYPOINT_SHADOW_ROOT = Path(
    "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/keypoints"
)
_SELECTOR_ATTRIBUTES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


@dataclass(frozen=True)
class PreparedRawKeypointSnapshot:
    dimensions: KeypointDimensions
    arrays: Mapping[str, Any]
    source_crop_arrays: Mapping[str, Any]
    source_crop_manifest: Mapping[str, Any]
    crop_source: KeypointCropSourceReference
    pose_model_schema_binding: Mapping[str, Any]
    preprocessing: KeypointPreprocessingReference


@dataclass(frozen=True)
class YoloKeypointV2Preparation:
    """Exact v2 snapshot plus numerical evidence from a legacy YOLO payload."""

    prepared: PreparedRawKeypointSnapshot
    conversion_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class KeypointShadowPublication:
    output_path: Path
    run_id: str
    prepared: PreparedRawKeypointSnapshot
    plans: KeypointStoragePlanSet
    manifest: Mapping[str, Any]
    phase_seconds: Mapping[str, float]
    elapsed_seconds: float


def prepare_raw_keypoint_v2_snapshot(
    arrays: Mapping[str, Any],
    *,
    dimensions: KeypointDimensions,
    source_crop_arrays: Mapping[str, Any],
    source_crop_manifest: Mapping[str, Any],
    pose_model_schema_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
) -> PreparedRawKeypointSnapshot:
    """Validate one complete YOLO-facing result before any Zarr publication."""

    crop_source = keypoint_crop_source_from_manifest(source_crop_manifest)
    KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=keypoint_skeleton_digest(pose_model_schema_binding),
    )
    if (
        crop_source.n_frames != dimensions.n_frames
        or crop_source.n_instances != dimensions.n_instances
        or crop_source.source_width != dimensions.source_width
        or crop_source.source_height != dimensions.source_height
    ):
        raise ValueError("Crop and keypoint dimensions differ.")
    return PreparedRawKeypointSnapshot(
        dimensions=dimensions,
        arrays=dict(arrays),
        source_crop_arrays=dict(source_crop_arrays),
        source_crop_manifest=dict(source_crop_manifest),
        crop_source=crop_source,
        pose_model_schema_binding=dict(pose_model_schema_binding),
        preprocessing=preprocessing,
    )


def prepare_raw_keypoint_v2_from_yolo_arrays(
    yolo_arrays: Mapping[str, Any],
    *,
    dimensions: KeypointDimensions,
    source_crop_arrays: Mapping[str, Any],
    source_crop_manifest: Mapping[str, Any],
    pose_model_schema_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
    projection_tolerance_pixels: float = 0.001,
) -> YoloKeypointV2Preparation:
    """Normalize one complete legacy YOLO result into the strict v2 boundary.

    Historical heading, normalized-coordinate, count, and quality arrays are
    intentionally ignored.  Source-camera projections are recomputed from the
    float32 ROI authority and bound crop geometry.
    """

    if not np.isfinite(projection_tolerance_pixels) or projection_tolerance_pixels < 0:
        raise ValueError("projection_tolerance_pixels must be finite and nonnegative.")
    required = {
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "frame_indices",
        "keypoints_roi",
        "keypoint_confidences",
        "confidence",
        "detection_success",
        "pose_bbox_xyxy_roi",
    }
    missing = sorted(required - set(yolo_arrays))
    if missing:
        raise ValueError(
            f"YOLO keypoint payload is missing required arrays: {missing!r}."
        )

    def values(path: str) -> np.ndarray:
        value = yolo_arrays[path]
        try:
            return np.asarray(value[...])
        except (IndexError, KeyError, TypeError):
            return np.asarray(value)

    rows = values("source_crop_row_ids").astype(np.int64, copy=False)
    if rows.shape != (dimensions.n_instances,):
        raise ValueError("YOLO source_crop_row_ids has the wrong shape.")
    if np.any(rows < 0):
        raise ValueError("YOLO source_crop_row_ids cannot be negative.")
    crop_count = int(np.asarray(source_crop_arrays["instance_key"]).shape[0])
    if np.any(rows >= crop_count):
        raise ValueError("YOLO source_crop_row_ids exceeds the crop snapshot.")

    keys = values("instance_key").astype(np.uint64, copy=False)
    frame_indices = values("frame_indices").astype(np.int64, copy=False)
    acquisition_frames = values("source_acquisition_frame_index").astype(
        np.int64, copy=False
    )
    keypoints_roi_legacy = values("keypoints_roi")
    keypoints_roi = keypoints_roi_legacy.astype(np.float32)
    confidences = values("keypoint_confidences").astype(np.float32)
    pose_success = values("detection_success").astype(bool, copy=False)
    keypoint_valid = (
        pose_success[:, None]
        & np.all(np.isfinite(keypoints_roi), axis=2)
        & np.isfinite(confidences)
    )
    keypoints_roi[~keypoint_valid] = np.float32(np.nan)
    confidences[~keypoint_valid] = np.float32(np.nan)
    pose_confidence = values("confidence").astype(np.float32)
    pose_confidence[~pose_success] = np.float32(np.nan)
    bbox_roi_legacy = values("pose_bbox_xyxy_roi")
    bbox_roi = bbox_roi_legacy.astype(np.float32)
    bbox_roi[~pose_success] = np.float32(np.nan)

    origins = np.asarray(source_crop_arrays["roi_coordinates_full"])[rows]
    keypoints_img = keypoints_roi + origins.astype(np.float32)[:, None, :]
    bbox_img = bbox_roi + np.column_stack((origins, origins)).astype(np.float32)
    source_signatures = np.asarray(source_crop_arrays["source_row_signature"])[rows]
    skeleton_digest = keypoint_skeleton_digest(pose_model_schema_binding)
    arrays: dict[str, np.ndarray] = {
        "instance_key": keys,
        "source_crop_row_ids": rows,
        "source_acquisition_frame_index": acquisition_frames,
        "frame_indices": frame_indices,
        "frame_row_offsets": derive_frame_row_offsets(
            frame_indices, n_frames=dimensions.n_frames
        ),
        "source_crop_row_signature": source_signatures,
        "keypoints_roi": keypoints_roi,
        "keypoints_img": keypoints_img,
        "keypoint_confidences": confidences,
        "keypoint_valid": keypoint_valid,
        "pose_confidence": pose_confidence,
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_img,
        "pose_success": pose_success,
    }
    arrays["keypoint_row_signature"] = derive_keypoint_row_signatures(
        instance_key=keys,
        source_crop_row_signature=source_signatures,
        keypoints_roi=keypoints_roi,
        keypoint_valid=keypoint_valid,
        skeleton_digest=skeleton_digest,
    )

    def maximum_finite_difference(left: np.ndarray, right: np.ndarray) -> float:
        finite = np.isfinite(left) & np.isfinite(right)
        if not np.any(finite):
            return 0.0
        return float(np.max(np.abs(left[finite].astype(np.float64) - right[finite])))

    roi_cast_error = maximum_finite_difference(keypoints_roi_legacy, keypoints_roi)
    bbox_cast_error = maximum_finite_difference(bbox_roi_legacy, bbox_roi)
    projection_errors: dict[str, float | None] = {
        "keypoints_roi_float64_to_float32_max_abs_pixels": roi_cast_error,
        "pose_bbox_roi_float64_to_float32_max_abs_pixels": bbox_cast_error,
        "keypoints_img_reprojection_max_abs_pixels": None,
        "pose_bbox_img_reprojection_max_abs_pixels": None,
    }
    for legacy_path, output_path, receipt_key in (
        (
            "keypoints_img",
            "keypoints_img",
            "keypoints_img_reprojection_max_abs_pixels",
        ),
        (
            "pose_bbox_xyxy_img",
            "pose_bbox_xyxy_img",
            "pose_bbox_img_reprojection_max_abs_pixels",
        ),
    ):
        if legacy_path in yolo_arrays:
            error = maximum_finite_difference(values(legacy_path), arrays[output_path])
            projection_errors[receipt_key] = error
            if error > projection_tolerance_pixels:
                raise ValueError(
                    f"YOLO {legacy_path} differs from the v2 float32 projection "
                    f"by {error:.9g} pixels, above {projection_tolerance_pixels:.9g}."
                )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        source_crop_manifest=source_crop_manifest,
        pose_model_schema_binding=pose_model_schema_binding,
        preprocessing=preprocessing,
    )
    return YoloKeypointV2Preparation(
        prepared=prepared,
        conversion_receipt={
            "schema_id": "palette.keypoint.yolo_v1_to_v2_preparation",
            "schema_version": 1,
            "source_layout": "legacy_yolo_keypoint_payload_v1",
            "destination_layout": KEYPOINT_SCHEMA_V2.layout,
            "projection_tolerance_pixels": float(projection_tolerance_pixels),
            "numerical_comparison": projection_errors,
            "derived_arrays": [
                "frame_row_offsets",
                "keypoint_valid",
                "keypoint_row_signature",
                "keypoints_img",
                "pose_bbox_xyxy_img",
            ],
            "ignored_legacy_families": [
                "heading",
                "normalized_coordinates",
                "count_aliases",
                "embedded_quality",
            ],
        },
    )


def require_safe_keypoint_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_KEYPOINT_SHADOW_ROOT,
) -> Path:
    root = shadow_root.expanduser().resolve()
    output = destination.expanduser().resolve()
    if output == root:
        raise ValueError("Keypoint shadow destination cannot equal its root.")
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Keypoint shadow destination must be below shadow_root."
        ) from exc
    if output.exists():
        raise FileExistsError(f"Keypoint shadow destination already exists: {output}")
    return output


def _write_by_physical_units(
    destination: Any, values: np.ndarray, *, plan: StoragePlan
) -> None:
    unit_shape = plan.shard_shape or plan.chunk_shape
    if unit_shape is None:
        raise ValueError("Keypoint publication does not support scalar arrays.")
    unit_rows = max(1, int(unit_shape[0]))
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def keypoint_metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: KeypointStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    relative_paths = ("", *(entry.rule.path for entry in plans.entries))
    run_prefix = f"keypoints_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path /= relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")
    root = _read_strict_json(output_path / "zarr.json")
    envelope = root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ValueError("Keypoint shadow lacks root consolidated metadata.")
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ValueError("Keypoint consolidated metadata envelope is invalid.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Keypoint consolidated metadata map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Keypoint consolidated metadata lacks {full_path!r}.")
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def validate_keypoint_shadow_publication(
    publication: KeypointShadowPublication,
) -> tuple[str, ...]:
    try:
        direct, consolidated = keypoint_metadata_declaration_maps(
            publication.output_path,
            run_id=publication.run_id,
            plans=publication.plans,
        )
        run = zarr.open_group(
            str(publication.output_path / "keypoints_runs" / publication.run_id),
            mode="r",
            use_consolidated=False,
        )
        arrays = {path: run[path] for path in KEYPOINT_SCHEMA_V2.binding_paths}
    except (OSError, TypeError, ValueError) as exc:
        return (f"keypoint shadow reopen failed: {exc}",)
    errors = list(
        validate_keypoint_publication(
            publication.manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
            source_crop_arrays=publication.prepared.source_crop_arrays,
            source_crop_manifest=publication.prepared.source_crop_manifest,
        )
    )
    if run.attrs.get("status") != "complete":
        errors.append("keypoint shadow status is not complete")
    if run.attrs.get("stage_selector_eligible") is not False:
        errors.append("keypoint shadow is not selector-ineligible")
    family = zarr.open_group(
        str(publication.output_path / "keypoints_runs"),
        mode="r",
        use_consolidated=False,
    )
    selected = [
        name
        for name in _SELECTOR_ATTRIBUTES
        if family.attrs.get(name) == publication.run_id
    ]
    if selected:
        errors.append(f"keypoint shadow is selected by {selected!r}")
    return tuple(errors)


def publish_selector_ineligible_keypoint_snapshot(
    prepared: PreparedRawKeypointSnapshot,
    *,
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_KEYPOINT_SHADOW_ROOT,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "keypoint_v2_shadow",
    disposition: KeypointPublicationDisposition = KeypointPublicationDisposition(),
) -> KeypointShadowPublication:
    """Write and validate one immutable, selector-ineligible raw snapshot."""

    output_path = require_safe_keypoint_shadow_destination(
        destination, shadow_root=shadow_root
    )
    if not str(created_by).strip():
        raise ValueError("created_by cannot be empty.")
    if "/" in str(run_id) or not str(run_id).strip():
        raise ValueError("run_id must be one nonempty archive group name.")
    KEYPOINT_SCHEMA_V2.require(
        prepared.arrays,
        dimensions=prepared.dimensions,
        source_crop_arrays=prepared.source_crop_arrays,
        skeleton_digest=keypoint_skeleton_digest(prepared.pose_model_schema_binding),
    )
    plans = plan_keypoint_storage(prepared.dimensions, profile=storage_profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            **disposition.root_attributes(),
            "schema_id": KEYPOINT_SHADOW_SCHEMA_ID,
            "schema_version": KEYPOINT_SHADOW_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "created_by": str(created_by),
        }
    )
    family = require_runs_parent(
        root, "keypoints_runs", completion_epoch=COMPLETION_EPOCH_STRICT
    )
    family.attrs.update(disposition.family_attributes())
    run = family.create_group(str(run_id))
    run.attrs.update(
        {
            "status": "running",
            **disposition.run_attributes(),
            "artifact_class": "raw_keypoint_observations",
            "keypoint_authority": False,
            "logical_schema": KEYPOINT_SCHEMA_V2.as_manifest(
                dimensions=prepared.dimensions
            ),
            "storage_plan": plans.as_manifest(),
            "source_crop_snapshot": prepared.crop_source.as_manifest(),
            "pose_model_schema_binding": dict(prepared.pose_model_schema_binding),
            "preprocessing": prepared.preprocessing.as_manifest(),
        }
    )
    if disposition.production_candidate:
        mark_run_started(run, run_name=str(run_id), stage="keypoints")
    arrays: dict[str, Any] = {}
    bindings = {binding.path: binding for binding in KEYPOINT_SCHEMA_V2.bindings}
    phase_started = time.perf_counter()
    for entry in plans.entries:
        path = entry.rule.path
        values = np.asarray(prepared.arrays[path])
        binding = bindings[path]
        contract = KEYPOINT_SCHEMA_V2.contracts.resolve(
            binding.contract_id, binding.contract_version
        )
        array = create_array_from_plan(
            run,
            name=path,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
            attributes={
                **disposition.array_attributes(),
                "artifact_class": "raw_keypoint_observations",
            },
        )
        _write_by_physical_units(array, values, plan=entry.plan)
        arrays[path] = array
    phase_seconds["create_and_write_arrays"] = time.perf_counter() - phase_started
    KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=prepared.dimensions,
        source_crop_arrays=prepared.source_crop_arrays,
        skeleton_digest=keypoint_skeleton_digest(prepared.pose_model_schema_binding),
    )
    run.attrs["status"] = "complete"
    if disposition.production_candidate:
        mark_run_complete(
            run,
            run_name=str(run_id),
            run_provenance=disposition.run_provenance,
        )
    phase_started = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = keypoint_metadata_declaration_maps(
        output_path, run_id=str(run_id), plans=plans
    )
    phase_seconds["first_consolidation"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    manifest = build_keypoint_run_manifest(
        run_id=str(run_id),
        dimensions=prepared.dimensions,
        crop_source=prepared.crop_source,
        source_crop_manifest=prepared.source_crop_manifest,
        source_crop_arrays=prepared.source_crop_arrays,
        pose_binding=prepared.pose_model_schema_binding,
        preprocessing=prepared.preprocessing,
        storage_plan=plans,
        arrays=arrays,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
    )
    run.attrs[KEYPOINT_RUN_MANIFEST_ATTRIBUTE] = manifest
    phase_seconds["build_manifest"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = keypoint_metadata_declaration_maps(
        output_path, run_id=str(run_id), plans=plans
    )
    errors = validate_keypoint_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
        source_crop_arrays=prepared.source_crop_arrays,
        source_crop_manifest=prepared.source_crop_manifest,
    )
    phase_seconds["final_consolidation_and_gate"] = time.perf_counter() - phase_started
    if errors:
        run.attrs.update(
            {
                "status": "failed",
                "stage_selector_eligible": False,
                "publication_errors": list(errors),
            }
        )
        if disposition.production_candidate:
            mark_run_failed(
                run,
                run_name=str(run_id),
                error="; ".join(errors),
            )
        raise RuntimeError("Keypoint publication gate failed: " + "; ".join(errors))
    publication = KeypointShadowPublication(
        output_path=output_path,
        run_id=str(run_id),
        prepared=prepared,
        plans=plans,
        manifest=manifest,
        phase_seconds=phase_seconds,
        elapsed_seconds=time.perf_counter() - started,
    )
    reopen_errors = validate_keypoint_shadow_publication(publication)
    if reopen_errors:
        raise RuntimeError(
            "Reopened keypoint publication gate failed: " + "; ".join(reopen_errors)
        )
    return publication


__all__ = [
    "DEFAULT_KEYPOINT_SHADOW_ROOT",
    "KEYPOINT_SHADOW_SCHEMA_ID",
    "KEYPOINT_SHADOW_SCHEMA_VERSION",
    "KeypointShadowPublication",
    "PreparedRawKeypointSnapshot",
    "YoloKeypointV2Preparation",
    "keypoint_metadata_declaration_maps",
    "prepare_raw_keypoint_v2_snapshot",
    "prepare_raw_keypoint_v2_from_yolo_arrays",
    "publish_selector_ineligible_keypoint_snapshot",
    "require_safe_keypoint_shadow_destination",
    "validate_keypoint_shadow_publication",
]
