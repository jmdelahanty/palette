"""Immutable selector-ineligible publication for subject-mask core snapshots.

This adapter is deliberately independent of inference and refinement.  It
accepts one already complete logical core, validates it against the exact
subject-mask schema, rematerializes it through the shared byte planner, and
publishes a closed-world Zarr v3 store with consolidated metadata.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.subject_mask_worker_receipt import (
    SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_ID,
    SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_VERSION,
    SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_ID,
    SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION,
    build_recording_assignment_keypoint_collection,
    validate_recording_subject_mask_assembly_identity,
)
from fisheye.shared.zarr.coordinate_manifest import (
    build_coordinate_catalog_envelope,
    validate_coordinate_catalog_envelope,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_SCHEMA_ID,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.crop_schema import (
    CropPlacementMode,
    crop_geometry_policy_from_manifest,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_profiles import (
    PUBLISHED_HTTP_V1,
    StorageProfile,
    storage_profile_from_manifest,
)
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    RawSubjectMaskSchema,
    RefinedSubjectMaskCoreSchema,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    SubjectMaskProbabilityEncoding,
)
from fisheye.shared.zarr.subject_mask_final_layout_units import (
    FinalLayoutUnitAdoption,
    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM,
    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS,
    copy_encoded_physical_unit,
    prepare_subject_mask_final_layout_unit_adoption,
)
from fisheye.shared.zarr.subject_mask_storage import (
    SubjectMaskStoragePlanSet,
    plan_raw_subject_mask_storage,
    plan_refined_subject_mask_publication_storage,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM,
    SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
    SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_VERSION,
    validate_subject_mask_source_validation_receipt,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_COMPLETED_AT_ATTR,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)

SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID = "palette.subject_mask_core.run_manifest"
SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_VERSION = 2
SUBJECT_MASK_CORE_COMPOSABLE_RUN_MANIFEST_SCHEMA_VERSION = 3
SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION = 4
SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION = 5
SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR = "source_validation_receipt.json"
SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_ID = "palette.subject_mask_core.shadow_publication"
SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_VERSION = 1
SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM = SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM
SUBJECT_MASK_CORE_METADATA_DIGEST_SCOPE = (
    "exact_run_group_and_array_declarations_redacting_manifest_lifecycle_"
    "and_transport_publication_attrs"
)
_TRANSPORT_PUBLICATION_ATTRS = (
    "atomic_publication_owner_uuid",
    "atomic_publication_tombstone",
    "cluster_output_staging",
    "publication_status",
    "subject_mask_bundle_selector_eligible",
)
SUBJECT_MASK_PROB_MAX_CANONICALIZATION = "cpu_max_encoded_then_decode_float32_v1"
SUBJECT_MASK_PROB_MAX_SOURCE_MAX_ABS_TOLERANCE = float(np.finfo(np.float32).eps)
SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_ID = (
    "palette.subject_mask_core.coordinate_dependencies"
)
SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_VERSION = 3


class SubjectMaskCoreValidationMode(str, Enum):
    """Publication-time validation cost and evidence boundary."""

    REFERENCE_FULL = "reference_full_v1"
    PRODUCTION_STREAMING = "production_streaming_v1"
    PRODUCTION_COMPOSABLE = "production_composable_units_v1"


@dataclass(frozen=True)
class SubjectMaskCorePublication:
    output_path: Path
    family: str
    run_id: str
    kind: str
    dimensions: SubjectMaskDimensions
    components: SubjectMaskComponentRegistry
    plans: SubjectMaskStoragePlanSet
    validation_mode: SubjectMaskCoreValidationMode
    source_manifest: Mapping[str, Any]
    manifest: Mapping[str, Any]
    phase_seconds: Mapping[str, float]
    elapsed_seconds: float


def _core_coordinate_catalog(
    kind: str,
) -> tuple[RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema, dict[str, object]]:
    if kind == "raw_probability_uint8":
        schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema = (
            RAW_SUBJECT_MASK_UINT8_SCHEMA_V1
        )
    elif kind == "refined_dense_core":
        schema = REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
    else:
        raise ValueError(f"Unsupported subject-mask core kind {kind!r}.")
    return schema, build_coordinate_catalog_envelope(
        schema.coordinate_contract_manifest()
    )


def _coordinate_dependency_envelope(
    document: Mapping[str, Any],
) -> dict[str, object]:
    normalized = json.loads(canonical_json_bytes(document).decode("utf-8"))
    if not isinstance(normalized, dict):
        raise ValueError("Subject-mask coordinate dependencies must be an object.")
    return {
        "schema_id": SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "digest": canonical_json_sha256(normalized),
        "document": normalized,
    }


def _worker_array_reference_matches(
    record: Any,
    value: Any,
) -> bool:
    if not isinstance(record, Mapping) or set(record) != {
        "shape",
        "dtype",
        "sha256",
    }:
        return False
    array = np.ascontiguousarray(np.asarray(value))
    return (
        record.get("shape") == [int(item) for item in array.shape]
        and record.get("dtype") == str(array.dtype)
        and record.get("sha256") == hashlib.sha256(array.view(np.uint8)).hexdigest()
    )


def _validate_worker_crop_coordinate_bindings(
    producer_evidence: Mapping[str, Any],
    *,
    crop_run_path: str,
    crop_manifest: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    allow_signed_hybrid_crop_rebase: bool = False,
) -> None:
    """Join every worker's narrow row claims to the exact crop-v2 authority."""

    crop_payload = crop_manifest["payload"]
    crop_run_id = str(crop_payload["run_id"])
    normalized_path = str(crop_run_path).strip().strip("/")
    coverage = producer_evidence.get("work_unit_coverage")
    units = coverage.get("units") if isinstance(coverage, Mapping) else None
    if not isinstance(units, list):
        raise ValueError("Worker crop joins require authoritative work-unit coverage.")
    nonempty_units = [
        unit
        for unit in units
        if isinstance(unit, Mapping) and unit.get("row_stop") != unit.get("row_start")
    ]
    workers = producer_evidence["workers"]
    if len(nonempty_units) != len(workers):
        raise ValueError("Worker crop joins do not match the nonempty work units.")
    offsets = np.asarray(source_crop_arrays["frame_row_offsets"][:], dtype=np.int64)
    logical_schema = crop_payload.get("logical_schema")
    raw_policy = (
        logical_schema.get("crop_policy")
        if isinstance(logical_schema, Mapping)
        else None
    )
    try:
        crop_policy = crop_geometry_policy_from_manifest(raw_policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("Crop-v2 placement policy is invalid.") from exc
    signed_hybrid_authority = (
        crop_policy.placement_authority
        if crop_policy.placement_mode is CropPlacementMode.VERIFIED_EXPLICIT_PER_ROW
        and isinstance(crop_policy.placement_authority, Mapping)
        and crop_policy.placement_authority.get("authority_kind")
        == "signed_hybrid_crop_provider"
        else None
    )

    for worker, unit in zip(workers, nonempty_units, strict=True):
        interval = worker["global_row_interval"]
        start = int(interval["start_row"])
        stop = int(interval["stop_row"])
        frame_start = int(unit["frame_start"])
        frame_stop = int(unit["frame_stop"])
        if int(offsets[frame_start]) != start or int(offsets[frame_stop]) != stop:
            raise ValueError(
                "Worker frame/row interval differs from crop-v2 frame offsets."
            )
        science_payload = worker["scientific_identity"]["payload"]
        stage = science_payload["stage_kind"]
        row_identity = science_payload["row_identity"]
        row_arrays = row_identity["arrays"]
        expected_rows = np.arange(start, stop, dtype=np.int64)
        expected_values = {
            "source_crop_row_ids": expected_rows,
            "instance_key": np.ascontiguousarray(
                np.asarray(source_crop_arrays["instance_key"][start:stop])
            ),
            "source_acquisition_frame_index": np.ascontiguousarray(
                np.asarray(
                    source_crop_arrays["source_acquisition_frame_index"][start:stop]
                )
            ),
        }
        acquisition_frames = expected_values["source_acquisition_frame_index"]
        if np.any(acquisition_frames < frame_start) or np.any(
            acquisition_frames >= frame_stop
        ):
            raise ValueError(
                "Worker acquisition frames fall outside its authoritative window."
            )
        if stage == "refined_subject_mask":
            expected_values["source_crop_xywh"] = np.ascontiguousarray(
                np.asarray(source_crop_arrays["source_crop_xywh"][start:stop])
            )
        for name, expected in expected_values.items():
            if not _worker_array_reference_matches(row_arrays.get(name), expected):
                raise ValueError(
                    f"Worker {name!r} evidence differs from its exact crop-v2 slice."
                )

        crop = science_payload["crop"]
        worker_crop_run = str(crop.get("run_id") or "")
        direct_crop_binding = worker_crop_run == crop_run_id
        signed_hybrid_rebase = (
            allow_signed_hybrid_crop_rebase
            and not direct_crop_binding
            and signed_hybrid_authority is not None
            and worker_crop_run == signed_hybrid_authority.get("run_id")
        )
        if not direct_crop_binding and not signed_hybrid_rebase:
            raise ValueError("Worker crop run differs from the crop-v2 authority.")
        xywh = np.ascontiguousarray(
            np.asarray(source_crop_arrays["source_crop_xywh"][start:stop])
        )
        if stage == "raw_subject_mask":
            manifest_ref = crop.get("run_manifest")
            expected_ref = {
                "schema_id": crop_manifest.get("schema_id"),
                "schema_version": crop_manifest.get("schema_version"),
                "payload_digest": crop_manifest.get("payload_digest"),
            }
            # Older sealed inference workers recorded the exact run name here
            # rather than the canonical ``crop_runs/<run>`` group path.  Keep
            # those receipts readable only when the run id and manifest digest
            # still bind exactly to the current crop authority.
            recorded_path = str(crop.get("run_group_path") or "").strip().strip("/")
            accepted_paths = {normalized_path, crop_run_id}
            if signed_hybrid_rebase:
                accepted_provider_paths = {
                    worker_crop_run,
                    f"crop_runs/{worker_crop_run}",
                }
                pixel_contract = science_payload.get("pixels", {}).get("pixel_contract")
                if (
                    recorded_path not in accepted_provider_paths
                    or manifest_ref is not None
                    or not isinstance(pixel_contract, Mapping)
                    or pixel_contract.get("source_pixels")
                    != "hybrid_acquisition_crop_video_offline_supplement"
                ):
                    raise ValueError(
                        "Raw worker lacks the exact signed-hybrid crop rebase evidence."
                    )
            elif recorded_path not in accepted_paths or manifest_ref != expected_ref:
                raise ValueError(
                    "Raw worker does not bind the exact crop-v2 manifest and path."
                )
            coordinates = np.ascontiguousarray(xywh[:, :2].astype(np.int32))
            if not _worker_array_reference_matches(
                crop.get("roi_coordinates_full"), coordinates
            ):
                raise ValueError(
                    "Raw worker ROI placement differs from its crop-v2 slice."
                )
        elif signed_hybrid_rebase:
            snapshot = crop.get("source_crop_snapshot")
            signature = (
                snapshot.get("source_crop_signature")
                if isinstance(snapshot, Mapping)
                else None
            )
            expected_signature = {
                "schema_id": "palette.hybrid_crop_provider.signature",
                "schema_version": 1,
                "source_pixels": ("hybrid_acquisition_crop_video_offline_supplement"),
                "provider_record_sha256": signed_hybrid_authority.get(
                    "provider_record_sha256"
                ),
                "source_pixel_fingerprint": signed_hybrid_authority.get(
                    "source_pixel_fingerprint"
                ),
                "source_row_signature_spec_digest": signed_hybrid_authority.get(
                    "source_row_signature_spec_digest"
                ),
                "source_rowset_fingerprint": signed_hybrid_authority.get(
                    "source_rowset_fingerprint"
                ),
            }
            if signature != expected_signature:
                raise ValueError(
                    "Refined worker signed-hybrid identity differs from the "
                    "crop-v2 origin authority."
                )
        roi_shape = crop.get("roi_shape_hw")
        if (
            not isinstance(roi_shape, list)
            or len(roi_shape) != 2
            or np.any(xywh[:, 2] != roi_shape[1])
            or np.any(xywh[:, 3] != roi_shape[0])
        ):
            raise ValueError("Worker ROI shape differs from crop-v2 placement.")


def build_subject_mask_core_coordinate_dependencies(
    *,
    kind: str,
    crop_run_path: str,
    crop_manifest: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    source_run_path: str,
    source_validation_receipt: Mapping[str, Any],
    n_rois: int,
    raw_core_manifest: Mapping[str, Any] | None = None,
    allow_signed_hybrid_crop_rebase: bool = False,
) -> dict[str, object]:
    """Bind one recording core to crop, worker, and optional raw authorities."""

    crop_errors = validate_crop_run_manifest(crop_manifest)
    if crop_errors:
        raise ValueError("Invalid crop coordinate manifest: " + "; ".join(crop_errors))
    crop_payload = crop_manifest.get("payload")
    if (
        crop_manifest.get("schema_id") != CROP_RUN_MANIFEST_SCHEMA_ID
        or crop_manifest.get("schema_version")
        != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        or not isinstance(crop_payload, Mapping)
    ):
        raise ValueError("Subject-mask recording cores require crop manifest v2.")
    crop_coordinate = crop_payload.get("coordinate_contract")
    crop_logical = crop_payload.get("logical_content")
    crop_schema = crop_payload.get("logical_schema")
    crop_dimensions = (
        crop_schema.get("dimensions") if isinstance(crop_schema, Mapping) else None
    )
    if (
        not isinstance(crop_coordinate, Mapping)
        or not isinstance(crop_logical, Mapping)
        or not isinstance(crop_dimensions, Mapping)
    ):
        raise ValueError("Crop coordinate or logical-content authority is absent.")
    crop_n_frames = crop_dimensions.get("n_frames")
    if type(crop_n_frames) is not int or crop_n_frames <= 0:
        raise ValueError("Crop coordinate authority has an invalid frame domain.")
    crop_content_document = crop_logical.get("document")
    crop_content_arrays = (
        crop_content_document.get("arrays")
        if isinstance(crop_content_document, Mapping)
        else None
    )
    identity_paths = (
        "instance_key",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "source_crop_xywh",
    )
    if not isinstance(crop_content_arrays, Mapping):
        raise ValueError("Crop logical-content array evidence is absent.")
    for path in identity_paths:
        value = source_crop_arrays.get(path)
        record = crop_content_arrays.get(path)
        if value is None or not isinstance(record, Mapping):
            raise ValueError(f"Crop coordinate evidence lacks {path!r}.")
        shape, dtype = _shape_dtype(value)
        if (
            record.get("shape") != list(shape)
            or record.get("dtype") != str(dtype)
            or record.get("sha256") != _array_hash(value)
        ):
            raise ValueError(
                f"Live crop array {path!r} differs from its coordinate manifest."
            )
    crop_path = str(crop_run_path).strip().strip("/")
    expected_crop_path = f"crop_runs/{crop_payload.get('run_id')}"
    if crop_path != expected_crop_path:
        raise ValueError("Crop run path differs from its manifest run_id.")

    receipt_payload = source_validation_receipt.get("payload")
    producer_evidence = (
        receipt_payload.get("producer_evidence")
        if isinstance(receipt_payload, Mapping)
        else None
    )
    if (
        source_validation_receipt.get("schema_version")
        != SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_VERSION
        or not isinstance(receipt_payload, Mapping)
        or source_validation_receipt.get("payload_digest")
        != canonical_json_sha256(receipt_payload)
        or not isinstance(producer_evidence, Mapping)
    ):
        raise ValueError(
            "Coordinate-aware subject-mask cores require retained recording "
            "assembly evidence."
        )
    stage_kind = (
        "raw_subject_mask"
        if kind == "raw_probability_uint8"
        else "refined_subject_mask" if kind == "refined_dense_core" else ""
    )
    validated_evidence = validate_recording_subject_mask_assembly_identity(
        producer_evidence,
        kind=kind,
        stage_kind=stage_kind,
        source_run_path=source_run_path,
        n_rois=int(n_rois),
        n_frames=crop_n_frames,
    )
    if validated_evidence.get("schema_version") != (
        SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError(
            "Coordinate-aware subject-mask cores require recording assembly v3."
        )
    _validate_worker_crop_coordinate_bindings(
        validated_evidence,
        crop_run_path=crop_path,
        crop_manifest=crop_manifest,
        source_crop_arrays=source_crop_arrays,
        allow_signed_hybrid_crop_rebase=allow_signed_hybrid_crop_rebase,
    )

    raw_binding: dict[str, object] | None = None
    assignment_keypoints: dict[str, object] | None = None
    if kind == "refined_dense_core":
        if not isinstance(raw_core_manifest, Mapping):
            raise ValueError("Refined coordinate cores require the exact raw core.")
        raw_payload = raw_core_manifest.get("payload")
        raw_coordinate = (
            raw_payload.get("coordinate_contract")
            if isinstance(raw_payload, Mapping)
            else None
        )
        raw_dependencies = (
            raw_payload.get("coordinate_dependencies")
            if isinstance(raw_payload, Mapping)
            else None
        )
        raw_dependency_document = (
            raw_dependencies.get("document")
            if isinstance(raw_dependencies, Mapping)
            else None
        )
        raw_crop = (
            raw_dependency_document.get("crop")
            if isinstance(raw_dependency_document, Mapping)
            else None
        )
        raw_recording_assembly = (
            raw_dependency_document.get("recording_assembly")
            if isinstance(raw_dependency_document, Mapping)
            else None
        )
        refined_raw_source = validated_evidence.get("source_producer_binding")
        if (
            raw_core_manifest.get("schema_id")
            != SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID
            or raw_core_manifest.get("schema_version")
            not in {
                SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
                SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
            }
            or not isinstance(raw_payload, Mapping)
            or raw_payload.get("kind") != "raw_probability_uint8"
            or raw_core_manifest.get("payload_digest")
            != canonical_json_sha256(raw_payload)
            or not isinstance(raw_coordinate, Mapping)
            or not isinstance(raw_crop, Mapping)
            or not isinstance(raw_recording_assembly, Mapping)
            or not isinstance(refined_raw_source, Mapping)
            or raw_crop.get("manifest_payload_digest")
            != crop_manifest.get("payload_digest")
            or refined_raw_source.get("digest")
            != raw_recording_assembly.get("producer_evidence_digest")
            or refined_raw_source.get("source_run_path")
            != raw_payload.get("source", {}).get("run_path")
        ):
            raise ValueError("Refined coordinate core binds an invalid raw core.")
        raw_binding = {
            "run_id": raw_payload.get("run_id"),
            "manifest_payload_digest": raw_core_manifest.get("payload_digest"),
            "coordinate_catalog_digest": raw_coordinate.get("digest"),
        }
        assignment_keypoints = build_recording_assignment_keypoint_collection(
            validated_evidence,
            source_run_path=source_run_path,
            n_rois=int(n_rois),
            n_frames=crop_n_frames,
        )
    elif raw_core_manifest is not None:
        raise ValueError("Raw coordinate cores cannot bind another raw core.")

    return _coordinate_dependency_envelope(
        {
            "crop": {
                "run_path": crop_path,
                "manifest_payload_digest": crop_manifest.get("payload_digest"),
                "coordinate_catalog_digest": crop_coordinate.get("digest"),
                "logical_content_digest": crop_logical.get("digest"),
            },
            "recording_assembly": {
                "source_run_path": str(source_run_path).strip().strip("/"),
                "source_validation_receipt_payload_digest": (
                    source_validation_receipt.get("payload_digest")
                ),
                "producer_evidence_schema_id": validated_evidence.get("schema_id"),
                "producer_evidence_schema_version": validated_evidence.get(
                    "schema_version"
                ),
                "producer_evidence_digest": canonical_json_sha256(validated_evidence),
            },
            "raw_core": raw_binding,
            "assignment_keypoints": assignment_keypoints,
        }
    )


def _validate_core_coordinate_dependencies(
    value: Any,
    *,
    kind: str,
) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("subject-mask coordinate dependencies must be an object",)
    if set(value) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "digest",
        "document",
    }:
        return ("subject-mask coordinate dependency envelope is not exact",)
    document = value.get("document")
    if (
        value.get("schema_id") != SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_ID
        or value.get("schema_version")
        != SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_VERSION
        or value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(document, Mapping)
        or value.get("digest") != canonical_json_sha256(document)
    ):
        return ("subject-mask coordinate dependency envelope is stale",)
    if set(document) != {
        "crop",
        "recording_assembly",
        "raw_core",
        "assignment_keypoints",
    }:
        return ("subject-mask coordinate dependency document is not exact",)
    crop = document.get("crop")
    assembly = document.get("recording_assembly")
    raw = document.get("raw_core")
    assignment_keypoints = document.get("assignment_keypoints")
    if not isinstance(crop, Mapping) or set(crop) != {
        "run_path",
        "manifest_payload_digest",
        "coordinate_catalog_digest",
        "logical_content_digest",
    }:
        return ("subject-mask crop coordinate dependency is invalid",)
    if not isinstance(assembly, Mapping) or set(assembly) != {
        "source_run_path",
        "source_validation_receipt_payload_digest",
        "producer_evidence_schema_id",
        "producer_evidence_schema_version",
        "producer_evidence_digest",
    }:
        return ("subject-mask assembly coordinate dependency is invalid",)
    errors: list[str] = []
    for field_name in (
        "manifest_payload_digest",
        "coordinate_catalog_digest",
        "logical_content_digest",
    ):
        try:
            _require_sha256(crop.get(field_name), name=f"crop {field_name}")
        except ValueError as exc:
            errors.append(str(exc))
    for field_name in (
        "source_validation_receipt_payload_digest",
        "producer_evidence_digest",
    ):
        try:
            _require_sha256(assembly.get(field_name), name=field_name)
        except ValueError as exc:
            errors.append(str(exc))
    if (
        assembly.get("producer_evidence_schema_id")
        != SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_ID
        or assembly.get("producer_evidence_schema_version")
        != SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION
    ):
        errors.append("subject-mask assembly evidence identity mismatch")
    if kind == "raw_probability_uint8" and raw is not None:
        errors.append("raw coordinate core unexpectedly binds another raw core")
    if kind == "raw_probability_uint8" and assignment_keypoints is not None:
        errors.append("raw coordinate core unexpectedly binds assignment keypoints")
    if kind == "refined_dense_core":
        if not isinstance(raw, Mapping) or set(raw) != {
            "run_id",
            "manifest_payload_digest",
            "coordinate_catalog_digest",
        }:
            errors.append("refined coordinate core raw dependency is invalid")
        else:
            for field_name in (
                "manifest_payload_digest",
                "coordinate_catalog_digest",
            ):
                try:
                    _require_sha256(raw.get(field_name), name=f"raw {field_name}")
                except ValueError as exc:
                    errors.append(str(exc))
        if (
            not isinstance(assignment_keypoints, Mapping)
            or assignment_keypoints.get("schema_id")
            != SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_ID
            or assignment_keypoints.get("schema_version")
            != SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_VERSION
            or assignment_keypoints.get("mode")
            not in {"not_used", "exact_worker_partition"}
            or assignment_keypoints.get("row_policy")
            != "ordered_contiguous_recording_crop_rows_v1"
            or type(assignment_keypoints.get("n_rois")) is not int
            or not isinstance(assignment_keypoints.get("workers"), list)
            or not assignment_keypoints["workers"]
        ):
            errors.append(
                "refined coordinate core assignment-keypoint dependency is invalid"
            )
    return tuple(errors)


def _shape_dtype(value: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    return tuple(int(item) for item in value.shape), np.dtype(value.dtype)


def _array_hash(value: Any, *, row_bytes_budget: int = 64 * 1024 * 1024) -> str:
    shape, dtype = _shape_dtype(value)
    digest = hashlib.sha256()
    if not shape:
        digest.update(np.ascontiguousarray(np.asarray(value[...])).view(np.uint8))
        return digest.hexdigest()
    row_bytes = max(1, int(dtype.itemsize) * int(np.prod(shape[1:])))
    block_rows = max(1, int(row_bytes_budget) // row_bytes)
    trailing = (slice(None),) * (len(shape) - 1)
    for start in range(0, shape[0], block_rows):
        stop = min(shape[0], start + block_rows)
        block = np.ascontiguousarray(np.asarray(value[(slice(start, stop), *trailing)]))
        digest.update(block.view(np.uint8))
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _metadata_maps(
    output_path: Path,
    *,
    family: str,
    run_id: str,
    paths: tuple[str, ...],
    archive_root_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    prefix = f"{family}/{run_id}"
    direct: dict[str, dict[str, Any]] = {
        "": _strict_json(output_path / prefix / "zarr.json")
    }
    for path in paths:
        direct[path] = _strict_json(output_path / prefix / path / "zarr.json")
    root = (
        archive_root_metadata
        if archive_root_metadata is not None
        else _strict_json(output_path / "zarr.json")
    )
    envelope = root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping) or envelope.get("kind") != "inline":
        raise ValueError("Subject-mask core publication lacks inline consolidation.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Subject-mask consolidated metadata map is absent.")
    consolidated: dict[str, dict[str, Any]] = {}
    for path in ("", *paths):
        full = prefix if not path else f"{prefix}/{path}"
        value = flattened.get(full)
        if not isinstance(value, Mapping):
            raise ValueError(f"Consolidated metadata lacks {full!r}.")
        consolidated[path] = dict(value)
    return direct, consolidated


def _normalized_metadata_document(
    declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for path in sorted(declarations):
        value = metadata_without_empty_group_consolidation(
            declarations[path], path=path
        )
        value = dict(value)
        attributes = value.get("attributes")
        if isinstance(attributes, Mapping):
            attrs = dict(attributes)
            for name in (
                SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
                "status",
                RUN_COMPLETION_STATUS_ATTR,
                RUN_COMPLETED_AT_ATTR,
                *_TRANSPORT_PUBLICATION_ATTRS,
            ):
                attrs.pop(name, None)
            value["attributes"] = attrs
        result[path] = value
    return result


def _metadata_digest(
    direct: Mapping[str, Mapping[str, Any]],
    consolidated: Mapping[str, Mapping[str, Any]],
) -> str:
    direct_doc = _normalized_metadata_document(direct)
    consolidated_doc = _normalized_metadata_document(consolidated)
    if direct_doc != consolidated_doc:
        raise ValueError("Direct and consolidated subject-mask metadata differ.")
    return canonical_json_sha256(direct_doc)


def subject_mask_core_metadata_declaration_maps(
    output_path: Path,
    *,
    family: str,
    run_id: str,
    manifest: Mapping[str, Any],
    archive_root_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load the exact direct/consolidated declaration inventory for one core."""

    payload = manifest.get("payload")
    logical_content = (
        payload.get("logical_content") if isinstance(payload, Mapping) else None
    )
    document = (
        logical_content.get("document")
        if isinstance(logical_content, Mapping)
        else None
    )
    arrays = document.get("arrays") if isinstance(document, Mapping) else None
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("Subject-mask core manifest lacks its array inventory.")
    return _metadata_maps(
        output_path.expanduser().resolve(),
        family=str(family),
        run_id=str(run_id),
        paths=tuple(str(path) for path in arrays),
        archive_root_metadata=archive_root_metadata,
    )


def build_subject_mask_coordinate_successor_manifest(
    source_manifest: Mapping[str, Any],
    *,
    run_id: str,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    coordinate_dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebind an immutable core payload to successor coordinate metadata.

    The logical-content and source-production evidence remain byte-for-byte
    identical.  A refined successor may additionally replace only its exact raw
    core dependency with the already validated raw coordinate successor.
    """

    errors = validate_subject_mask_core_run_manifest(source_manifest)
    if errors:
        raise ValueError(
            "Subject-mask coordinate-successor source manifest is invalid: "
            + "; ".join(errors)
        )
    successor = copy.deepcopy(dict(source_manifest))
    payload = successor.get("payload")
    if not isinstance(payload, dict):  # pragma: no cover - validated above
        raise ValueError("Subject-mask source manifest payload is absent.")
    publication = payload.get("publication")
    if not isinstance(publication, dict):  # pragma: no cover - validated above
        raise ValueError("Subject-mask source publication record is absent.")
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id or "/" in normalized_run_id:
        raise ValueError("Subject-mask successor run_id must be one run name.")
    payload["run_id"] = normalized_run_id
    source = payload.get("source")
    if not isinstance(source, dict):  # pragma: no cover - validated above
        raise ValueError("Subject-mask source binding is absent.")
    validation_receipt = source.get("validation_receipt")
    if validation_receipt is not None:
        if not isinstance(validation_receipt, dict):  # pragma: no cover
            raise ValueError("Subject-mask source-validation binding is malformed.")
        validation_receipt["relative_path"] = (
            f"{payload['stage_family']}/{normalized_run_id}/"
            f"{SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR}"
        )
    publication["stage_selector_eligible"] = False
    publication["metadata_state"] = "direct_and_consolidated_validated"
    publication["metadata_digest"] = _metadata_digest(
        direct_metadata_declarations,
        consolidated_metadata_declarations,
    )
    if coordinate_dependencies is not None:
        normalized_dependencies = json.loads(
            canonical_json_bytes(coordinate_dependencies).decode("utf-8")
        )
        dependency_errors = _validate_core_coordinate_dependencies(
            normalized_dependencies,
            kind=str(payload.get("kind")),
        )
        if dependency_errors:
            raise ValueError(
                "Subject-mask successor dependencies are invalid: "
                + "; ".join(dependency_errors)
            )
        payload["coordinate_dependencies"] = normalized_dependencies
    successor["payload_digest"] = canonical_json_sha256(payload)
    canonical_json_bytes(successor)
    successor_errors = validate_subject_mask_core_run_manifest(successor)
    if successor_errors:
        raise ValueError(
            "Subject-mask coordinate-successor manifest is invalid: "
            + "; ".join(successor_errors)
        )
    return successor


def _validate_persisted_subject_mask_core_publication(
    output_path: Path,
    *,
    family: str,
    run_id: str,
    archive_root_metadata: Mapping[str, Any] | None = None,
    expected_manifest_payload_digest: str | None = None,
    recompute_bounded_samples: bool,
) -> tuple[str, ...]:
    errors: list[str] = []
    archive = output_path.expanduser().resolve()
    try:
        run = zarr.open_group(
            str(archive / str(family) / str(run_id)),
            mode="r",
            use_consolidated=False,
        )
        manifest = run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
        if not isinstance(manifest, Mapping):
            return ("subject-mask core run_manifest is absent",)
        if (
            expected_manifest_payload_digest is not None
            and manifest.get("payload_digest") != expected_manifest_payload_digest
        ):
            return ("subject-mask core manifest differs from bundle receipt",)
        errors.extend(validate_subject_mask_core_run_manifest(manifest))
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping):
            return tuple(errors)
        if payload.get("run_id") != str(run_id) or payload.get("stage_family") != str(
            family
        ):
            errors.append("subject-mask core persisted path differs from manifest")
        if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "complete":
            errors.append("subject-mask core completion status is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("subject-mask core is not selector-ineligible")
        logical = payload.get("logical_content")
        document = logical.get("document") if isinstance(logical, Mapping) else None
        arrays = document.get("arrays") if isinstance(document, Mapping) else None
        samples = payload.get("write_receipt")
        samples = (
            samples.get("bounded_reopen_samples")
            if isinstance(samples, Mapping)
            else None
        )
        if not isinstance(arrays, Mapping) or not isinstance(samples, Mapping):
            errors.append("subject-mask core validation inventory is absent")
        else:
            for path, declaration in arrays.items():
                if not isinstance(declaration, Mapping) or str(path) not in run:
                    errors.append(f"subject-mask core array is absent at {path}")
                    continue
                array = run[str(path)]
                if list(array.shape) != declaration.get("shape") or str(
                    np.dtype(array.dtype)
                ) != declaration.get("dtype"):
                    errors.append(f"subject-mask core array metadata differs at {path}")
                    continue
                path_samples = samples.get(path)
                if not isinstance(path_samples, list):
                    errors.append(
                        f"subject-mask core bounded samples are absent at {path}"
                    )
                    continue
                for sample in path_samples:
                    if not isinstance(sample, Mapping):
                        errors.append(
                            f"subject-mask core bounded sample is invalid at {path}"
                        )
                        continue
                    start = sample.get("start_row")
                    stop = sample.get("stop_row")
                    if (
                        type(start) is not int
                        or type(stop) is not int
                        or not 0 <= start < stop <= int(array.shape[0])
                    ):
                        errors.append(
                            f"subject-mask core bounded sample range is invalid at {path}"
                        )
                        continue
                    if not recompute_bounded_samples:
                        continue
                    trailing = (slice(None),) * (len(array.shape) - 1)
                    values = np.ascontiguousarray(
                        np.asarray(array[(slice(start, stop), *trailing)])
                    )
                    if hashlib.sha256(values.view(np.uint8)).hexdigest() != sample.get(
                        "sha256"
                    ):
                        errors.append(
                            f"subject-mask core bounded sample differs at {path}"
                        )
        if not errors:
            direct, consolidated = subject_mask_core_metadata_declaration_maps(
                archive,
                family=str(family),
                run_id=str(run_id),
                manifest=manifest,
                archive_root_metadata=archive_root_metadata,
            )
            observed_digest = _metadata_digest(direct, consolidated)
            publication = payload.get("publication")
            expected_digest = (
                publication.get("metadata_digest")
                if isinstance(publication, Mapping)
                else None
            )
            if observed_digest != expected_digest:
                errors.append("subject-mask core metadata digest differs")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return tuple(errors)


def validate_persisted_subject_mask_core_publication(
    output_path: Path,
    *,
    family: str,
    run_id: str,
    archive_root_metadata: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Boundedly revalidate one persisted core after transport/import.

    Logical array hashes were computed while the source was written and the
    atomic importer separately verifies every physical file.  This explicit
    audit gate checks exact metadata plus the manifest's bounded first/last
    physical row-band samples without another full decoded scan.
    """

    return _validate_persisted_subject_mask_core_publication(
        output_path,
        family=family,
        run_id=run_id,
        archive_root_metadata=archive_root_metadata,
        recompute_bounded_samples=True,
    )


def validate_receipt_bound_persisted_subject_mask_core_publication(
    output_path: Path,
    *,
    family: str,
    run_id: str,
    expected_manifest_payload_digest: str,
    archive_root_metadata: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Admit an outer-bundle-bound immutable core without payload reads."""

    return _validate_persisted_subject_mask_core_publication(
        output_path,
        family=family,
        run_id=run_id,
        archive_root_metadata=archive_root_metadata,
        expected_manifest_payload_digest=expected_manifest_payload_digest,
        recompute_bounded_samples=False,
    )


def _resolve_kind(
    kind: str,
    dimensions: SubjectMaskDimensions,
    *,
    include_threshold_cache: bool,
    profile: StorageProfile,
) -> tuple[
    str,
    RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    SubjectMaskStoragePlanSet,
]:
    normalized = str(kind).strip()
    if normalized == "raw_probability_uint8":
        return (
            "subject_mask_runs",
            RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
            plan_raw_subject_mask_storage(
                dimensions,
                encoding=SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255,
                include_threshold_cache=include_threshold_cache,
                profile=profile,
            ),
        )
    if normalized == "refined_dense_core":
        if include_threshold_cache:
            raise ValueError(
                "Refined dense publication has no optional threshold cache."
            )
        return (
            "refined_subject_masks_runs",
            REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
            plan_refined_subject_mask_publication_storage(
                dimensions,
                profile=profile,
            ),
        )
    raise ValueError(f"Unsupported subject-mask core kind {kind!r}.")


def _group_for_path(group: Any, path: str) -> tuple[Any, str]:
    parts = path.split("/")
    target = group
    for part in parts[:-1]:
        target = target.require_group(part)
    return target, parts[-1]


def _composable_identity_units(
    source: Any,
    adoption: FinalLayoutUnitAdoption,
) -> list[dict[str, object]]:
    """Assemble a storage-independent logical root from worker unit evidence.

    Complete global identity units are trusted from receipt-bound worker
    packages. Only identity units crossing worker boundaries are decoded and
    checked against every contributing package segment.
    """

    if (
        not adoption.composable_identity
        or adoption.logical_identity_unit_rows
        != SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
    ):
        raise ValueError("Composable publication requires current unit evidence.")
    shape, dtype = _shape_dtype(source)
    if not shape:
        raise ValueError("Composable publication requires a row axis.")
    row_bytes = int(dtype.itemsize) * int(np.prod(shape[1:]))
    complete = {int(unit.start_row): unit for unit in adoption.logical_identity_units}
    segments_by_unit: dict[int, list[Any]] = {}
    for segment in adoption.logical_identity_boundary_segments:
        segments_by_unit.setdefault(int(segment.unit_start_row), []).append(segment)
    trailing = (slice(None),) * (len(shape) - 1)
    result: list[dict[str, object]] = []
    for start in range(
        0,
        int(shape[0]),
        SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS,
    ):
        stop = min(
            int(shape[0]),
            start + SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS,
        )
        unit = complete.get(start)
        segments = sorted(
            segments_by_unit.get(start, []), key=lambda item: item.start_row
        )
        if unit is not None:
            if unit.stop_row != stop or segments:
                raise ValueError("Composable logical identity ownership is ambiguous.")
            result.append(
                {
                    "start_row": start,
                    "stop_row": stop,
                    "decoded_bytes": int(unit.decoded_bytes),
                    "sha256": str(unit.sha256),
                }
            )
            continue
        values = np.ascontiguousarray(
            np.asarray(source[(slice(start, stop), *trailing)])
        )
        cursor = start
        for segment in segments:
            if segment.start_row != cursor or segment.stop_row > stop:
                raise RuntimeError(
                    "Composable boundary evidence is not ordered and contiguous."
                )
            local_start = segment.start_row - start
            local_stop = segment.stop_row - start
            part = np.ascontiguousarray(values[local_start:local_stop])
            if (
                int(part.nbytes) != int(segment.decoded_bytes)
                or hashlib.sha256(part.view(np.uint8)).hexdigest() != segment.sha256
            ):
                raise RuntimeError(
                    "Composable boundary bytes differ from worker evidence."
                )
            cursor = segment.stop_row
        if cursor != stop:
            raise RuntimeError(
                "Composable boundary evidence does not cover its identity unit."
            )
        if int(values.nbytes) != (stop - start) * row_bytes:
            raise RuntimeError("Composable identity decoded byte count differs.")
        result.append(
            {
                "start_row": start,
                "stop_row": stop,
                "decoded_bytes": int(values.nbytes),
                "sha256": hashlib.sha256(values.view(np.uint8)).hexdigest(),
            }
        )
    if set(complete) | set(segments_by_unit) != {
        int(item["start_row"]) for item in result
    }:
        raise ValueError("Composable logical identity inventory differs.")
    return result


def _write_physical_units(
    destination: Any,
    source: Any,
    plan: Any,
    *,
    source_validation_record: Mapping[str, Any] | None = None,
    physical_unit_workers: int = 1,
    final_layout_adoption: FinalLayoutUnitAdoption | None = None,
    destination_array_path: Path | None = None,
    use_composable_identity: bool = False,
) -> dict[str, object]:
    """Write whole physical row bands and hash exact source bytes once.

    The driver reads and hashes first-axis bands in canonical order.  Optional
    worker threads receive only bands aligned to the outer-shard grid, or to
    the chunk grid for an unsharded array.  A band may contain several trailing
    physical units, but bands never share a chunk or shard.
    """

    if type(physical_unit_workers) is not int or physical_unit_workers <= 0:
        raise ValueError("physical_unit_workers must be a positive integer.")
    if final_layout_adoption is not None and destination_array_path is None:
        raise ValueError(
            "Final-layout unit adoption requires the destination array path."
        )
    if use_composable_identity and (
        final_layout_adoption is None or not final_layout_adoption.composable_identity
    ):
        raise ValueError(
            "Composable publication requires complete current final-layout units."
        )

    unit = plan.shard_shape or plan.chunk_shape
    if unit is None:
        raise ValueError("Subject-mask core arrays cannot be scalar.")
    shape, dtype = _shape_dtype(source)
    trailing = (slice(None),) * (len(shape) - 1)
    writes = 0
    encoded_unit_copies = 0
    boundary_reencodes = 0
    logical_digest = hashlib.sha256()
    composable_units = (
        _composable_identity_units(source, final_layout_adoption)
        if use_composable_identity and final_layout_adoption is not None
        else None
    )
    samples: list[dict[str, object]] = []
    expected_units: list[Mapping[str, Any]] | None = None
    expected_unit_index = 0
    expected_unit_digest = hashlib.sha256()
    if source_validation_record is not None and not use_composable_identity:
        algorithm = source_validation_record.get("digest_algorithm")
        if algorithm == SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM:
            raw_units = source_validation_record.get("units")
            if not isinstance(raw_units, list) or not raw_units:
                raise ValueError("Source-validation row units are absent.")
            expected_units = list(raw_units)
        elif algorithm != SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM:
            raise ValueError("Source-validation array digest algorithm is unsupported.")
    starts = tuple(range(0, shape[0], max(1, int(unit[0]))))
    effective_workers = min(int(physical_unit_workers), len(starts))
    executor = (
        ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="subject-mask-zarr-write",
        )
        if effective_workers > 1
        else None
    )
    pending: deque[Future[None]] = deque()
    try:
        for index, start in enumerate(starts):
            stop = min(shape[0], start + max(1, int(unit[0])))
            selection = (slice(start, stop), *trailing)
            encoded_unit = (
                final_layout_adoption.units.get(int(start))
                if final_layout_adoption is not None
                else None
            )
            values: np.ndarray[Any, Any] | None = None
            if not (use_composable_identity and encoded_unit is not None):
                values = np.ascontiguousarray(np.asarray(source[selection]))
                if values.dtype.hasobject:
                    raise TypeError("Subject-mask core arrays cannot use object dtype.")
                unit_bytes = values.view(np.uint8)
                if not use_composable_identity:
                    logical_digest.update(unit_bytes)
                unit_sha256 = hashlib.sha256(unit_bytes).hexdigest()
            else:
                unit_sha256 = str(encoded_unit.logical_sha256)
            if expected_units is not None:
                assert values is not None
                cursor = int(start)
                while cursor < int(stop):
                    if expected_unit_index >= len(expected_units):
                        raise RuntimeError(
                            "Bytes streamed into the subject-mask publication differ "
                            "from the validated source receipt."
                        )
                    expected = expected_units[expected_unit_index]
                    expected_start = int(expected["start_row"])
                    expected_stop = int(expected["stop_row"])
                    if not (expected_start <= cursor < expected_stop):
                        raise RuntimeError(
                            "Bytes streamed into the subject-mask publication differ "
                            "from the validated source receipt."
                        )
                    segment_stop = min(int(stop), expected_stop)
                    local_start = cursor - int(start)
                    local_stop = segment_stop - int(start)
                    segment = np.ascontiguousarray(values[local_start:local_stop])
                    expected_unit_digest.update(segment.view(np.uint8))
                    if segment_stop == expected_stop:
                        if expected_unit_digest.hexdigest() != expected.get("sha256"):
                            raise RuntimeError(
                                "Bytes streamed into the subject-mask publication "
                                "differ from the validated source receipt."
                            )
                        expected_unit_index += 1
                        expected_unit_digest = hashlib.sha256()
                    cursor = segment_stop
            if encoded_unit is not None:
                if encoded_unit.stop_row != int(stop) or (
                    not use_composable_identity
                    and encoded_unit.logical_sha256 != unit_sha256
                ):
                    raise RuntimeError(
                        "Encoded final-layout unit differs from the validated "
                        f"source bytes at rows {start}:{stop}."
                    )
                assert destination_array_path is not None
                encoded_unit_copies += 1
            else:
                boundary_reencodes += 1
            if executor is None:
                if encoded_unit is not None:
                    copy_encoded_physical_unit(
                        encoded_unit,
                        destination_array_path=destination_array_path,
                    )
                else:
                    assert values is not None
                    destination[selection] = values
            elif encoded_unit is not None:
                pending.append(
                    executor.submit(
                        copy_encoded_physical_unit,
                        encoded_unit,
                        destination_array_path=destination_array_path,
                    )
                )
                if len(pending) >= effective_workers:
                    pending.popleft().result()
            else:
                assert values is not None
                pending.append(
                    executor.submit(
                        _write_one_physical_row_band,
                        destination,
                        selection,
                        values,
                    )
                )
                if len(pending) >= effective_workers:
                    pending.popleft().result()
            if index == 0 or index == len(starts) - 1:
                samples.append(
                    {
                        "start_row": int(start),
                        "stop_row": int(stop),
                        "sha256": unit_sha256,
                    }
                )
            writes += 1
        for future in pending:
            future.result()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    logical_sha256 = logical_digest.hexdigest()
    if source_validation_record is not None and not use_composable_identity:
        if expected_units is not None:
            if expected_unit_index != len(expected_units):
                raise RuntimeError(
                    "Bytes streamed into the subject-mask publication differ from "
                    "the validated source receipt."
                )
        elif logical_sha256 != source_validation_record.get("sha256"):
            raise RuntimeError(
                "Bytes streamed into the subject-mask publication differ from "
                "the validated source receipt."
            )
    result: dict[str, object] = {
        "shape": list(shape),
        "dtype": str(dtype),
        "physical_write_count": writes,
        "encoded_physical_unit_copy_count": encoded_unit_copies,
        "boundary_reencode_count": boundary_reencodes,
        "encoded_object_copy_count": (
            sum(len(unit.objects) for unit in final_layout_adoption.units.values())
            if final_layout_adoption is not None
            else 0
        ),
        "encoded_bytes_copied": (
            int(final_layout_adoption.encoded_bytes)
            if final_layout_adoption is not None
            else 0
        ),
        "effective_physical_unit_workers": effective_workers,
        "bounded_reopen_samples": samples,
    }
    if composable_units is None:
        result.update(
            {
                "digest_algorithm": SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM,
                "sha256": logical_sha256,
            }
        )
    else:
        result.update(
            {
                "digest_algorithm": (
                    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM
                ),
                "identity_unit_rows": (
                    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
                ),
                "unit_count": len(composable_units),
                "units_digest": canonical_json_sha256(composable_units),
                "units": composable_units,
            }
        )
    return result


def _write_one_physical_row_band(
    destination: Any,
    selection: tuple[slice, ...],
    values: np.ndarray,
) -> None:
    destination[selection] = values


def _streamed_array_document(
    write_receipts: Mapping[str, Mapping[str, object]], paths: tuple[str, ...]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for path in paths:
        receipt = write_receipts[path]
        record: dict[str, object] = {
            "shape": list(write_receipts[path]["shape"]),
            "dtype": str(write_receipts[path]["dtype"]),
            "digest_algorithm": str(write_receipts[path]["digest_algorithm"]),
        }
        if receipt["digest_algorithm"] == SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM:
            record["sha256"] = str(receipt["sha256"])
        elif (
            receipt["digest_algorithm"]
            == SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM
        ):
            record.update(
                {
                    "identity_unit_rows": int(receipt["identity_unit_rows"]),
                    "unit_count": int(receipt["unit_count"]),
                    "units_digest": str(receipt["units_digest"]),
                    "units": list(receipt["units"]),
                }
            )
        else:  # pragma: no cover - writer owns the finite algorithm set
            raise RuntimeError("Subject-mask write receipt digest is unsupported.")
        result[path] = record
    return result


def _validate_reopened_metadata_and_samples(
    *,
    reopened_arrays: Mapping[str, Any],
    paths: tuple[str, ...],
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    dimensions: SubjectMaskDimensions,
    write_receipts: Mapping[str, Mapping[str, object]],
) -> None:
    bindings = {binding.path: binding for binding in schema.bindings}
    for path in paths:
        binding = bindings[path]
        contract = schema.contracts.resolve(
            binding.contract_id, binding.contract_version
        )
        errors = contract.validate_observation(
            reopened_arrays[path], dimensions=dimensions.contract_dimensions
        )
        if errors:
            raise RuntimeError(
                f"Reopened subject-mask array {path!r} violates its contract: "
                + "; ".join(errors)
            )
        shape, _dtype = _shape_dtype(reopened_arrays[path])
        trailing = (slice(None),) * (len(shape) - 1)
        samples = write_receipts[path].get("bounded_reopen_samples")
        if not isinstance(samples, list) or not samples:
            raise RuntimeError(
                f"Subject-mask write receipt lacks samples for {path!r}."
            )
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise RuntimeError(f"Subject-mask sample for {path!r} is malformed.")
            start = int(sample["start_row"])
            stop = int(sample["stop_row"])
            values = np.ascontiguousarray(
                np.asarray(reopened_arrays[path][(slice(start, stop), *trailing)])
            )
            if hashlib.sha256(values.view(np.uint8)).hexdigest() != sample.get(
                "sha256"
            ):
                raise RuntimeError(
                    f"Reopened subject-mask boundary sample differs for {path!r} "
                    f"rows {start}:{stop}."
                )


def _array_document(
    arrays: Mapping[str, Any], paths: tuple[str, ...]
) -> dict[str, object]:
    return {
        path: {
            "shape": list(_shape_dtype(arrays[path])[0]),
            "dtype": str(_shape_dtype(arrays[path])[1]),
            "digest_algorithm": SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM,
            "sha256": _array_hash(arrays[path]),
        }
        for path in paths
    }


def _canonical_probability_max(probabilities: Any) -> np.ndarray:
    shape, dtype = _shape_dtype(probabilities)
    if len(shape) != 4:
        raise ValueError("mask_probs_roi must have shape (N,C,H,W).")
    result = np.empty((shape[0], shape[1]), dtype=np.float32)
    row_bytes = max(1, int(dtype.itemsize) * int(np.prod(shape[1:])))
    block_rows = max(1, (64 * 1024 * 1024) // row_bytes)
    for start in range(0, shape[0], block_rows):
        stop = min(shape[0], start + block_rows)
        values = np.asarray(probabilities[start:stop])
        maxima = np.max(values, axis=(2, 3)).astype(np.float32, copy=False)
        if dtype == np.dtype(np.uint8):
            maxima = maxima / np.float32(255.0)
        result[start:stop] = maxima
    return result


def _canonicalize_raw_probability_max(
    arrays: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, object]]:
    canonical = _canonical_probability_max(arrays["mask_probs_roi"])
    source = np.asarray(arrays["metrics/prob_max"][...])
    if source.dtype != np.dtype(np.float32) or source.shape != canonical.shape:
        raise ValueError("Source metrics/prob_max must be exact float32[N,C].")
    if not np.all(np.isfinite(source)):
        raise ValueError("Source metrics/prob_max contains non-finite values.")
    differences = np.abs(source - canonical)
    mismatch_count = int(np.count_nonzero(source != canonical))
    max_abs = float(np.max(differences)) if differences.size else 0.0
    if max_abs > SUBJECT_MASK_PROB_MAX_SOURCE_MAX_ABS_TOLERANCE:
        raise ValueError(
            "Source metrics/prob_max differs materially from the canonical "
            f"stored-probability derivation (max_abs={max_abs!r})."
        )
    normalized = dict(arrays)
    normalized["metrics/prob_max"] = canonical
    return normalized, {
        "schema_id": "palette.subject_mask.prob_max_canonicalization",
        "schema_version": 1,
        "policy": SUBJECT_MASK_PROB_MAX_CANONICALIZATION,
        "source_mismatch_count": mismatch_count,
        "source_max_abs_difference": max_abs,
        "source_max_abs_tolerance": (SUBJECT_MASK_PROB_MAX_SOURCE_MAX_ABS_TOLERANCE),
        "canonical_dtype": "float32",
        "canonical_shape": list(canonical.shape),
    }


def _manifest_dimensions(value: object) -> SubjectMaskDimensions:
    if not isinstance(value, Mapping):
        raise ValueError("Subject-mask manifest dimensions must be an object.")
    dimensions = SubjectMaskDimensions(
        n_frames=value.get("n_frames"),
        n_rois=value.get("n_rois"),
        n_channels=value.get("n_channels"),
        roi_height=value.get("roi_height"),
        roi_width=value.get("roi_width"),
    )
    if dict(value) != dimensions.as_manifest():
        raise ValueError("Subject-mask manifest dimensions are not canonical.")
    return dimensions


def _manifest_components(value: object) -> SubjectMaskComponentRegistry:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "labels",
        "channel_axis",
        "ordering",
    }:
        raise ValueError("Subject-mask component registry is not exact.")
    labels = value.get("labels")
    if not isinstance(labels, list):
        raise ValueError("Subject-mask component labels must be an array.")
    components = SubjectMaskComponentRegistry(tuple(labels))
    if dict(value) != components.as_manifest():
        raise ValueError("Subject-mask component registry is not canonical.")
    return components


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
    return text


def validate_subject_mask_core_run_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply reconstruct and validate one persisted core manifest."""

    errors: list[str] = []
    expected_envelope = {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }
    if set(manifest) != expected_envelope:
        errors.append("subject-mask core manifest envelope has unexpected fields")
    manifest_schema_version = manifest.get("schema_version")
    composable_manifest = manifest_schema_version in {
        SUBJECT_MASK_CORE_COMPOSABLE_RUN_MANIFEST_SCHEMA_VERSION,
        SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    }
    if (
        manifest.get("schema_id") != SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID
        or manifest_schema_version
        not in {
            SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_VERSION,
            SUBJECT_MASK_CORE_COMPOSABLE_RUN_MANIFEST_SCHEMA_VERSION,
            SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
            SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
        }
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("subject-mask core manifest envelope identity mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "subject-mask core manifest payload must be an object")
    try:
        expected_payload_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"subject-mask core manifest is not strict JSON: {exc}")
    else:
        if manifest.get("payload_digest") != expected_payload_digest:
            errors.append("subject-mask core manifest payload_digest mismatch")
    expected_payload = {
        "run_id",
        "stage_family",
        "kind",
        "publication",
        "logical_schema",
        "storage_plan",
        "source",
        "write_receipt",
        "logical_content",
    }
    if manifest_schema_version in {
        SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
        SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    }:
        expected_payload.update({"coordinate_contract", "coordinate_dependencies"})
    if set(payload) != expected_payload:
        errors.append("subject-mask core manifest payload has unexpected fields")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id or "/" in run_id:
        errors.append("subject-mask core run_id is invalid")
    kind = payload.get("kind")
    expected_family = {
        "raw_probability_uint8": "subject_mask_runs",
        "refined_dense_core": "refined_subject_masks_runs",
    }.get(kind)
    if expected_family is None or payload.get("stage_family") != expected_family:
        errors.append("subject-mask core kind/stage family mismatch")

    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("subject-mask core publication must be an object")
    else:
        expected_publication = {
            "completion_contract": RUN_COMPLETION_CONTRACT,
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_digest_scope": SUBJECT_MASK_CORE_METADATA_DIGEST_SCOPE,
            "metadata_digest": publication.get("metadata_digest"),
        }
        if dict(publication) != expected_publication:
            errors.append("subject-mask core publication is not exact")
        try:
            _require_sha256(publication.get("metadata_digest"), name="metadata_digest")
        except ValueError as exc:
            errors.append(str(exc))

    dimensions: SubjectMaskDimensions | None = None
    components: SubjectMaskComponentRegistry | None = None
    logical = payload.get("logical_schema")
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema | None = None
    plans: SubjectMaskStoragePlanSet | None = None
    if not isinstance(logical, Mapping):
        errors.append("subject-mask core logical_schema must be an object")
    else:
        try:
            dimensions = _manifest_dimensions(logical.get("dimensions"))
            components = _manifest_components(logical.get("components"))
            storage = payload.get("storage_plan")
            if not isinstance(storage, Mapping):
                raise ValueError("Subject-mask storage_plan must be an object.")
            storage_profile_value = storage.get("storage_profile")
            if not isinstance(storage_profile_value, Mapping):
                raise ValueError("Subject-mask storage profile must be an object.")
            profile = storage_profile_from_manifest(storage_profile_value)
            if kind == "raw_probability_uint8":
                schema = RAW_SUBJECT_MASK_UINT8_SCHEMA_V1
                threshold = logical.get("threshold")
                if type(threshold) not in (int, float):
                    raise ValueError("Raw subject-mask threshold is invalid.")
                storage_arrays = storage.get("arrays")
                include_threshold_cache = bool(
                    isinstance(storage_arrays, list)
                    and any(
                        isinstance(item, Mapping) and item.get("path") == "masks_roi"
                        for item in storage_arrays
                    )
                )
                plans = plan_raw_subject_mask_storage(
                    dimensions,
                    encoding=SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255,
                    include_threshold_cache=include_threshold_cache,
                    profile=profile,
                )
                expected_logical = schema.as_manifest(
                    dimensions=dimensions,
                    components=components,
                    threshold=float(threshold),
                )
            elif kind == "refined_dense_core":
                schema = REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
                plans = plan_refined_subject_mask_publication_storage(
                    dimensions,
                    profile=profile,
                )
                expected_logical = schema.as_manifest(
                    dimensions=dimensions,
                    components=components,
                )
            else:
                raise ValueError("Unsupported subject-mask core kind.")
            if dict(logical) != expected_logical:
                errors.append("subject-mask core logical_schema differs from builder")
            if dict(storage) != plans.as_manifest():
                errors.append(
                    "subject-mask core storage plan differs from planner output"
                )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    if manifest_schema_version in {
        SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
        SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    }:
        if schema is None:
            errors.append("subject-mask coordinate catalog lacks a resolved schema")
        else:
            errors.extend(
                validate_coordinate_catalog_envelope(
                    payload.get("coordinate_contract"),
                    expected_document=schema.coordinate_contract_manifest(),
                )
            )
        errors.extend(
            _validate_core_coordinate_dependencies(
                payload.get("coordinate_dependencies"),
                kind=str(kind),
            )
        )

    source = payload.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "run_path",
        "manifest_digest",
        "manifest",
        "validation_receipt",
    }:
        errors.append("subject-mask core source binding is not exact")
    else:
        if not isinstance(source.get("run_path"), str) or not source.get("run_path"):
            errors.append("subject-mask core source run_path is invalid")
        try:
            source_manifest = source.get("manifest")
            if not isinstance(source_manifest, Mapping):
                raise ValueError("source manifest must be an object")
            if canonical_json_sha256(source_manifest) != _require_sha256(
                source.get("manifest_digest"), name="source manifest digest"
            ):
                errors.append("subject-mask core source manifest digest mismatch")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        receipt_binding = source.get("validation_receipt")
        if receipt_binding is not None:
            expected_receipt_fields = {
                "schema_id",
                "schema_version",
                "payload_digest",
                "storage",
                "relative_path",
                "document_sha256",
                "semantic_unit_count",
                "array_count",
            }
            if (
                not isinstance(receipt_binding, Mapping)
                or set(receipt_binding) != expected_receipt_fields
                or receipt_binding.get("storage") != "strict_json_sidecar_v1"
            ):
                errors.append("subject-mask source-validation binding is not exact")
            else:
                for name in ("payload_digest", "document_sha256"):
                    try:
                        _require_sha256(receipt_binding.get(name), name=name)
                    except ValueError as exc:
                        errors.append(str(exc))
                expected_relative = (
                    f"{expected_family}/{run_id}/"
                    f"{SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR}"
                )
                if receipt_binding.get("relative_path") != expected_relative:
                    errors.append(
                        "subject-mask source-validation receipt path mismatch"
                    )
                for name in ("semantic_unit_count", "array_count"):
                    if (
                        type(receipt_binding.get(name)) is not int
                        or int(receipt_binding[name]) <= 0
                    ):
                        errors.append(
                            f"subject-mask source-validation {name} is invalid"
                        )

    receipt = payload.get("write_receipt")
    expected_receipt_fields = {
        "validation_mode",
        "output_write_unit",
        "physical_write_counts",
        "logical_hash_timing",
        "reopen_validation",
        "bounded_reopen_samples",
        "parallel_write_policy",
        "derived_metric_canonicalization",
    }
    if composable_manifest:
        expected_receipt_fields.add("composable_identity")
    if not isinstance(receipt, Mapping) or set(receipt) != expected_receipt_fields:
        errors.append("subject-mask core write_receipt is not exact")
    elif plans is not None:
        if composable_manifest:
            payload_path = (
                "mask_probs_roi" if kind == "raw_probability_uint8" else "masks_roi"
            )
            expected_composable = {
                "policy": "ordered_fixed_global_row_units_v1",
                "payload_path": payload_path,
                "digest_algorithm": (
                    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM
                ),
                "identity_unit_rows": (
                    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
                ),
                "complete_units_source": "receipt_bound_worker_packages_v1",
                "boundary_units_source": "decoded_and_segment_verified_v1",
            }
            if (
                receipt.get("validation_mode")
                != (SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE.value)
                or receipt.get("composable_identity") != expected_composable
            ):
                errors.append("subject-mask composable write receipt differs")
        if receipt.get("parallel_write_policy") not in {
            "single_writer_v1_future_workers_require_disjoint_whole_shards",
            "bounded_threaded_disjoint_whole_physical_row_bands_v1",
        }:
            errors.append("subject-mask core parallel write policy is unsupported")
        paths = tuple(entry.rule.path for entry in plans.entries)
        counts = receipt.get("physical_write_counts")
        samples = receipt.get("bounded_reopen_samples")
        if not isinstance(counts, Mapping) or set(counts) != set(paths):
            errors.append("subject-mask core physical write counts are incomplete")
        elif any(
            type(counts[path]) is not int or int(counts[path]) <= 0 for path in paths
        ):
            errors.append("subject-mask core physical write count is invalid")
        if not isinstance(samples, Mapping) or set(samples) != set(paths):
            errors.append("subject-mask core reopen samples are incomplete")

    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping) or set(logical_content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("subject-mask core logical_content envelope is invalid")
    else:
        document = logical_content.get("document")
        if logical_content.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
            errors.append("subject-mask core logical_content algorithm mismatch")
        if not isinstance(document, Mapping):
            errors.append(
                "subject-mask core logical_content document must be an object"
            )
        else:
            try:
                if logical_content.get("digest") != canonical_json_sha256(document):
                    errors.append("subject-mask core logical_content digest mismatch")
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"subject-mask core logical_content is not strict JSON: {exc}"
                )
            if set(document) != {
                "schema_id",
                "schema_version",
                "kind",
                "dimensions",
                "components",
                "arrays",
            }:
                errors.append("subject-mask core logical_content has unexpected fields")
            if (
                document.get("schema_id") != "palette.subject_mask_core.logical_content"
                or document.get("schema_version") != (2 if composable_manifest else 1)
                or document.get("kind") != kind
            ):
                errors.append("subject-mask core logical_content identity mismatch")
            if (
                dimensions is not None
                and document.get("dimensions") != dimensions.as_manifest()
            ):
                errors.append("subject-mask core logical_content dimensions mismatch")
            if (
                components is not None
                and document.get("components") != components.as_manifest()
            ):
                errors.append("subject-mask core logical_content components mismatch")
            array_documents = document.get("arrays")
            if plans is not None:
                plan_by_path = {entry.rule.path: entry.plan for entry in plans.entries}
                if not isinstance(array_documents, Mapping) or set(
                    array_documents
                ) != set(plan_by_path):
                    errors.append(
                        "subject-mask core logical array declarations mismatch"
                    )
                else:
                    payload_path = (
                        "mask_probs_roi"
                        if kind == "raw_probability_uint8"
                        else "masks_roi"
                    )
                    for path, item in array_documents.items():
                        expected_fields = {
                            "shape",
                            "dtype",
                            "digest_algorithm",
                            "sha256",
                        }
                        if composable_manifest and path == payload_path:
                            expected_fields = {
                                "shape",
                                "dtype",
                                "digest_algorithm",
                                "identity_unit_rows",
                                "unit_count",
                                "units_digest",
                                "units",
                            }
                        if (
                            not isinstance(item, Mapping)
                            or set(item) != expected_fields
                        ):
                            errors.append(
                                f"subject-mask core logical declaration invalid at {path}"
                            )
                            continue
                        plan = plan_by_path[path]
                        if item.get("shape") != list(plan.logical_shape):
                            errors.append(f"subject-mask core shape mismatch at {path}")
                        if item.get("dtype") != plan.logical_dtype:
                            errors.append(f"subject-mask core dtype mismatch at {path}")
                        if composable_manifest and path == payload_path:
                            units = item.get("units")
                            if (
                                item.get("digest_algorithm")
                                != SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM
                                or item.get("identity_unit_rows")
                                != SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
                                or not isinstance(units, list)
                                or not units
                            ):
                                errors.append(
                                    f"subject-mask composable identity invalid at {path}"
                                )
                                continue
                            cursor = 0
                            row_bytes = int(
                                np.dtype(plan.logical_dtype).itemsize
                            ) * int(np.prod(plan.logical_shape[1:]))
                            for unit in units:
                                if not isinstance(unit, Mapping) or set(unit) != {
                                    "start_row",
                                    "stop_row",
                                    "decoded_bytes",
                                    "sha256",
                                }:
                                    errors.append(
                                        f"subject-mask composable unit invalid at {path}"
                                    )
                                    break
                                start = unit.get("start_row")
                                stop = unit.get("stop_row")
                                if (
                                    type(start) is not int
                                    or type(stop) is not int
                                    or start != cursor
                                    or stop
                                    != min(
                                        int(plan.logical_shape[0]),
                                        start
                                        + SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS,
                                    )
                                    or unit.get("decoded_bytes")
                                    != (stop - start) * row_bytes
                                ):
                                    errors.append(
                                        f"subject-mask composable coverage invalid at {path}"
                                    )
                                    break
                                try:
                                    _require_sha256(
                                        unit.get("sha256"),
                                        name=f"{path} unit sha256",
                                    )
                                except ValueError as exc:
                                    errors.append(str(exc))
                                cursor = int(stop)
                            if (
                                cursor != int(plan.logical_shape[0])
                                or item.get("unit_count") != len(units)
                                or item.get("units_digest")
                                != canonical_json_sha256(units)
                            ):
                                errors.append(
                                    f"subject-mask composable identity digest mismatch at {path}"
                                )
                        else:
                            if (
                                item.get("digest_algorithm")
                                != SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM
                            ):
                                errors.append(
                                    f"subject-mask core digest algorithm mismatch at {path}"
                                )
                            try:
                                _require_sha256(
                                    item.get("sha256"), name=f"{path} sha256"
                                )
                            except ValueError as exc:
                                errors.append(str(exc))
    return tuple(errors)


def publish_selector_ineligible_subject_mask_core_snapshot(
    source_arrays: Mapping[str, Any],
    *,
    source_crop_arrays: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    n_frames: int,
    components: SubjectMaskComponentRegistry,
    destination: Path,
    run_id: str,
    kind: str,
    source_run_path: str,
    source_attributes: Mapping[str, Any] | None = None,
    threshold: float = 0.5,
    include_threshold_cache: bool = False,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "subject_mask_core_shadow",
    validation_mode: SubjectMaskCoreValidationMode | str = (
        SubjectMaskCoreValidationMode.REFERENCE_FULL
    ),
    source_validation_receipt: Mapping[str, Any] | None = None,
    coordinate_dependencies: Mapping[str, Any] | None = None,
    physical_unit_workers: int = 1,
    final_layout_unit_packages: tuple[Path, ...] | None = None,
    require_complete_final_layout_units: bool = False,
) -> SubjectMaskCorePublication:
    """Validate and rematerialize one complete raw or refined core."""

    validation_mode = SubjectMaskCoreValidationMode(validation_mode)
    if type(physical_unit_workers) is not int or physical_unit_workers <= 0:
        raise ValueError("physical_unit_workers must be a positive integer.")
    package_paths = tuple(final_layout_unit_packages or ())
    if require_complete_final_layout_units and not package_paths:
        raise ValueError(
            "Complete final-layout unit adoption requires worker packages."
        )
    output_path = destination.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Subject-mask core destination exists: {output_path}")
    if not str(run_id).strip() or "/" in str(run_id):
        raise ValueError("run_id must be one nonempty group name.")
    if not str(source_run_path).strip():
        raise ValueError("source_run_path cannot be empty.")
    payload_path = "mask_probs_roi" if kind == "raw_probability_uint8" else "masks_roi"
    if payload_path not in source_arrays:
        raise ValueError(f"Subject-mask source lacks {payload_path}.")
    payload_shape, _payload_dtype = _shape_dtype(source_arrays[payload_path])
    if len(payload_shape) != 4:
        raise ValueError("Subject-mask payload must have shape (N,C,H,W).")
    dimensions = SubjectMaskDimensions(
        n_frames=int(n_frames),
        n_rois=int(payload_shape[0]),
        n_channels=int(payload_shape[1]),
        roi_height=int(payload_shape[2]),
        roi_width=int(payload_shape[3]),
    )
    family, schema, plans = _resolve_kind(
        kind,
        dimensions,
        include_threshold_cache=include_threshold_cache,
        profile=profile,
    )
    paths = tuple(entry.rule.path for entry in plans.entries)
    arrays = {path: source_arrays[path] for path in paths}
    canonicalization: dict[str, object] | None = None
    if validation_mode is SubjectMaskCoreValidationMode.REFERENCE_FULL:
        if source_validation_receipt is not None:
            raise ValueError(
                "Reference-full publication does not consume a source receipt."
            )
        if isinstance(schema, RawSubjectMaskSchema):
            arrays, canonicalization = _canonicalize_raw_probability_max(arrays)
            schema.require(
                arrays,
                dimensions=dimensions,
                components=components,
                threshold=float(threshold),
                source_crop_arrays=source_crop_arrays,
            )
        else:
            schema.require(
                arrays,
                dimensions=dimensions,
                components=components,
                source_crop_arrays=source_crop_arrays,
            )
    else:
        if source_validation_receipt is None:
            raise ValueError(
                "Production-streaming publication requires a source-validation receipt."
            )
        source_validation_receipt = validate_subject_mask_source_validation_receipt(
            source_validation_receipt,
            kind=kind,
            source_run_path=source_run_path,
            source_manifest=source_manifest,
            schema=schema,
            arrays=arrays,
            dimensions=dimensions,
            components=components,
            threshold=(
                float(threshold) if isinstance(schema, RawSubjectMaskSchema) else None
            ),
        )
        canonicalization = {
            "policy": "upstream_validated_exact_source_receipt_v1",
            "source_receipt_payload_digest": source_validation_receipt[
                "payload_digest"
            ],
        }
    if isinstance(schema, RawSubjectMaskSchema):
        logical_schema = schema.as_manifest(
            dimensions=dimensions,
            components=components,
            threshold=float(threshold),
        )
    else:
        logical_schema = schema.as_manifest(
            dimensions=dimensions,
            components=components,
        )
    manifest_schema_version = (
        SUBJECT_MASK_CORE_COMPOSABLE_RUN_MANIFEST_SCHEMA_VERSION
        if validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
        else SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_VERSION
    )
    coordinate_contract: dict[str, object] | None = None
    normalized_coordinate_dependencies: dict[str, object] | None = None
    if coordinate_dependencies is not None:
        if source_validation_receipt is None:
            raise ValueError(
                "Coordinate-aware core publication requires source validation evidence."
            )
        coordinate_contract = build_coordinate_catalog_envelope(
            schema.coordinate_contract_manifest()
        )
        normalized_coordinate_dependencies = json.loads(
            canonical_json_bytes(coordinate_dependencies).decode("utf-8")
        )
        coordinate_errors = _validate_core_coordinate_dependencies(
            normalized_coordinate_dependencies,
            kind=kind,
        )
        if coordinate_errors:
            raise ValueError(
                "Invalid subject-mask coordinate dependencies: "
                + "; ".join(coordinate_errors)
            )
        dependency_document = normalized_coordinate_dependencies["document"]
        assembly_dependency = dependency_document["recording_assembly"]
        if assembly_dependency[
            "source_validation_receipt_payload_digest"
        ] != source_validation_receipt.get("payload_digest"):
            raise ValueError(
                "Coordinate dependencies bind another source-validation receipt."
            )
        manifest_schema_version = (
            SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
            if validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
            else SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
        )

    started = time.perf_counter()
    phases: dict[str, float] = {}
    phase = time.perf_counter()
    final_layout_adoption: FinalLayoutUnitAdoption | None = None
    if package_paths:
        if (
            validation_mode
            not in {
                SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
                SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE,
            }
            or source_validation_receipt is None
        ):
            raise ValueError(
                "Final-layout unit packages require production-streaming source "
                "validation evidence."
            )
        payload_entries = [
            entry for entry in plans.entries if entry.rule.path == payload_path
        ]
        if len(payload_entries) != 1:
            raise RuntimeError("Subject-mask payload storage plan is ambiguous.")
        final_layout_adoption = prepare_subject_mask_final_layout_unit_adoption(
            package_paths,
            kind=kind,
            dimensions=dimensions,
            plan=payload_entries[0].plan,
            source_validation_receipt=source_validation_receipt,
            require_complete_eligible_units=require_complete_final_layout_units,
            profile=profile,
        )
        if (
            validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
            and not final_layout_adoption.composable_identity
        ):
            raise ValueError(
                "Composable publication requires current receipt-bound packages."
            )
    if (
        validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
        and final_layout_adoption is None
    ):
        raise ValueError(
            "Composable publication requires complete final-layout unit packages."
        )
    phases["final_layout_unit_preflight"] = time.perf_counter() - phase
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "schema_id": SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_VERSION,
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "created_at_utc": utc_now(),
            "created_by": str(created_by),
        }
    )
    parent = root.create_group(family)
    parent.attrs.update(
        {
            "benchmark_only": True,
            "selector_eligible": False,
            "selection_contract": "none_shadow_direct_path_only",
        }
    )
    run = parent.create_group(str(run_id))
    mark_run_started(run, run_name=str(run_id), stage=family.removesuffix("_runs"))
    run.attrs.update(
        {
            **dict(source_attributes or {}),
            "status": "running",
            "stage_selector_eligible": False,
            "shadow_only": True,
            "core_kind": kind,
            "source_run_path": str(source_run_path),
            "logical_schema": logical_schema,
            "storage_plan": plans.as_manifest(),
            "component_registry": components.as_manifest(),
            "validation_mode": validation_mode.value,
            "derived_metric_canonicalization": canonicalization,
            "physical_unit_workers_requested": int(physical_unit_workers),
            "final_layout_unit_adoption": {
                "enabled": final_layout_adoption is not None,
                "required_complete_eligible_units": bool(
                    require_complete_final_layout_units
                ),
                "package_count": (
                    int(final_layout_adoption.package_count)
                    if final_layout_adoption is not None
                    else 0
                ),
                "complete_unit_count": (
                    len(final_layout_adoption.units)
                    if final_layout_adoption is not None
                    else 0
                ),
                "boundary_unit_count": (
                    len(final_layout_adoption.boundary_starts)
                    if final_layout_adoption is not None
                    else 0
                ),
                "encoded_object_count": (
                    int(final_layout_adoption.encoded_object_count)
                    if final_layout_adoption is not None
                    else 0
                ),
                "encoded_bytes": (
                    int(final_layout_adoption.encoded_bytes)
                    if final_layout_adoption is not None
                    else 0
                ),
                "composable_identity": (
                    bool(final_layout_adoption.composable_identity)
                    if final_layout_adoption is not None
                    else False
                ),
                "logical_identity_unit_rows": (
                    final_layout_adoption.logical_identity_unit_rows
                    if final_layout_adoption is not None
                    else None
                ),
                "logical_identity_complete_unit_count": (
                    len(final_layout_adoption.logical_identity_units)
                    if final_layout_adoption is not None
                    else 0
                ),
                "logical_identity_boundary_segment_count": (
                    len(final_layout_adoption.logical_identity_boundary_segments)
                    if final_layout_adoption is not None
                    else 0
                ),
            },
        }
    )
    destination_arrays: dict[str, Any] = {}
    write_counts: dict[str, int] = {}
    write_receipts: dict[str, dict[str, object]] = {}
    source_array_validation: Mapping[str, Any] = {}
    if validation_mode in {
        SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE,
    }:
        assert source_validation_receipt is not None
        source_array_validation = source_validation_receipt["payload"]["arrays"]
    phase = time.perf_counter()
    bindings = {binding.path: binding for binding in schema.bindings}
    for entry in plans.entries:
        path = entry.rule.path
        binding = bindings[path]
        contract = schema.contracts.resolve(
            binding.contract_id, binding.contract_version
        )
        group, leaf = _group_for_path(run, path)
        destination_array = create_array_from_plan(
            group,
            name=leaf,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
            attributes={
                "benchmark_only": True,
                "selector_eligible": False,
                "artifact_class": "subject_mask_scientific_core",
            },
        )
        try:
            write_receipt = _write_physical_units(
                destination_array,
                arrays[path],
                entry.plan,
                source_validation_record=source_array_validation.get(path),
                physical_unit_workers=physical_unit_workers,
                final_layout_adoption=(
                    final_layout_adoption if path == payload_path else None
                ),
                destination_array_path=(
                    output_path / family / str(run_id) / Path(path)
                    if path == payload_path and final_layout_adoption is not None
                    else None
                ),
                use_composable_identity=(
                    validation_mode
                    is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
                    and path == payload_path
                ),
            )
        except Exception as exc:
            run.attrs["status"] = "failed"
            run.attrs["validation_failure"] = "physical_write_or_receipt_mismatch"
            mark_run_failed(
                run,
                run_name=str(run_id),
                error=f"Subject-mask physical publication failed: {exc}",
            )
            raise RuntimeError(
                f"Subject-mask physical publication failed for {path!r}: {exc}"
            ) from exc
        write_receipts[path] = write_receipt
        write_counts[path] = int(write_receipt["physical_write_count"])
        destination_arrays[path] = destination_array
    phases["physical_unit_publication"] = time.perf_counter() - phase
    effective_physical_unit_workers = max(
        int(receipt["effective_physical_unit_workers"])
        for receipt in write_receipts.values()
    )
    parallel_write_policy = (
        "single_writer_v1_future_workers_require_disjoint_whole_shards"
        if int(physical_unit_workers) == 1
        else "bounded_threaded_disjoint_whole_physical_row_bands_v1"
    )
    run.attrs.update(
        {
            "physical_unit_workers_effective_max": effective_physical_unit_workers,
            "parallel_write_policy": parallel_write_policy,
        }
    )
    streamed_array_document: dict[str, object] | None = None
    if validation_mode in {
        SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE,
    }:
        assert source_validation_receipt is not None
        streamed_array_document = _streamed_array_document(write_receipts, paths)
    run.attrs["physical_write_counts"] = write_counts

    phase = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = _metadata_maps(
        output_path, family=family, run_id=str(run_id), paths=paths
    )
    metadata_digest = _metadata_digest(direct, consolidated)
    phases["first_consolidation"] = time.perf_counter() - phase

    phase = time.perf_counter()
    if validation_mode is SubjectMaskCoreValidationMode.REFERENCE_FULL:
        array_document = _array_document(destination_arrays, paths)
    else:
        assert streamed_array_document is not None
        array_document = streamed_array_document
    content = {
        "schema_id": "palette.subject_mask_core.logical_content",
        "schema_version": (
            2
            if validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
            else 1
        ),
        "kind": kind,
        "dimensions": dimensions.as_manifest(),
        "components": components.as_manifest(),
        "arrays": array_document,
    }
    source_validation_binding: dict[str, object] | None = None
    source_validation_receipt_bytes: bytes | None = None
    source_validation_receipt_relative_path = (
        f"{family}/{run_id}/{SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR}"
    )
    if source_validation_receipt is not None:
        source_validation_receipt_bytes = canonical_json_bytes(
            source_validation_receipt
        )
        coverage = source_validation_receipt["payload"]["semantic_coverage"]
        receipt_arrays = source_validation_receipt["payload"]["arrays"]
        source_validation_binding = {
            "schema_id": source_validation_receipt["schema_id"],
            "schema_version": source_validation_receipt["schema_version"],
            "payload_digest": source_validation_receipt["payload_digest"],
            "storage": "strict_json_sidecar_v1",
            "relative_path": source_validation_receipt_relative_path,
            "document_sha256": hashlib.sha256(
                source_validation_receipt_bytes
            ).hexdigest(),
            "semantic_unit_count": coverage["unit_count"],
            "array_count": len(receipt_arrays),
        }
    manifest_payload: dict[str, object] = {
        "run_id": str(run_id),
        "stage_family": family,
        "kind": kind,
        "publication": {
            "completion_contract": RUN_COMPLETION_CONTRACT,
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_digest_scope": SUBJECT_MASK_CORE_METADATA_DIGEST_SCOPE,
            "metadata_digest": metadata_digest,
        },
        "logical_schema": logical_schema,
        "storage_plan": plans.as_manifest(),
        "source": {
            "run_path": str(source_run_path),
            "manifest_digest": canonical_json_sha256(source_manifest),
            "manifest": dict(source_manifest),
            "validation_receipt": source_validation_binding,
        },
        "write_receipt": {
            "validation_mode": validation_mode.value,
            "output_write_unit": "complete_outer_shard_or_unsharded_chunk",
            "physical_write_counts": write_counts,
            "logical_hash_timing": (
                "separate_postwrite_full_read_v1"
                if validation_mode is SubjectMaskCoreValidationMode.REFERENCE_FULL
                else (
                    "receipt_bound_composable_units_boundary_reads_only_v1"
                    if validation_mode
                    is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE
                    else "computed_during_required_publication_read_v1"
                )
            ),
            "reopen_validation": (
                "full_schema_and_full_logical_hash_v1"
                if validation_mode is SubjectMaskCoreValidationMode.REFERENCE_FULL
                else "metadata_plus_first_last_physical_row_band_samples_v1"
            ),
            "bounded_reopen_samples": {
                path: write_receipts[path]["bounded_reopen_samples"] for path in paths
            },
            "parallel_write_policy": parallel_write_policy,
            "derived_metric_canonicalization": canonicalization,
        },
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    if validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_COMPOSABLE:
        manifest_payload["write_receipt"]["composable_identity"] = {
            "policy": "ordered_fixed_global_row_units_v1",
            "payload_path": payload_path,
            "digest_algorithm": (SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM),
            "identity_unit_rows": (SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS),
            "complete_units_source": "receipt_bound_worker_packages_v1",
            "boundary_units_source": "decoded_and_segment_verified_v1",
        }
    if manifest_schema_version in {
        SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
        SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    }:
        assert coordinate_contract is not None
        assert normalized_coordinate_dependencies is not None
        manifest_payload["coordinate_contract"] = coordinate_contract
        manifest_payload["coordinate_dependencies"] = normalized_coordinate_dependencies
    manifest = {
        "schema_id": SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": manifest_schema_version,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(manifest_payload),
        "payload": manifest_payload,
    }
    canonical_json_bytes(manifest)
    manifest_errors = validate_subject_mask_core_run_manifest(manifest)
    if manifest_errors:
        mark_run_failed(
            run,
            run_name=str(run_id),
            error="; ".join(manifest_errors),
        )
        raise RuntimeError(
            "Subject-mask core manifest failed deep validation: "
            + "; ".join(manifest_errors)
        )
    run.attrs[SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE] = manifest
    phases["build_manifest"] = time.perf_counter() - phase

    phase = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = _metadata_maps(
        output_path, family=family, run_id=str(run_id), paths=paths
    )
    if _metadata_digest(direct, consolidated) != metadata_digest:
        raise RuntimeError(
            "Subject-mask metadata digest changed after manifest insertion."
        )
    if source_validation_receipt_bytes is not None:
        receipt_path = output_path / source_validation_receipt_relative_path
        receipt_path.write_bytes(source_validation_receipt_bytes)
    reopened = zarr.open_group(
        str(output_path / family / str(run_id)),
        mode="r",
        use_consolidated=False,
    )
    reopened_arrays = {path: reopened[path] for path in paths}
    if validation_mode is SubjectMaskCoreValidationMode.REFERENCE_FULL:
        if isinstance(schema, RawSubjectMaskSchema):
            schema.require(
                reopened_arrays,
                dimensions=dimensions,
                components=components,
                threshold=float(threshold),
                source_crop_arrays=source_crop_arrays,
            )
        else:
            schema.require(
                reopened_arrays,
                dimensions=dimensions,
                components=components,
                source_crop_arrays=source_crop_arrays,
            )
        if _array_document(reopened_arrays, paths) != array_document:
            raise RuntimeError("Reopened subject-mask logical content differs.")
    else:
        _validate_reopened_metadata_and_samples(
            reopened_arrays=reopened_arrays,
            paths=paths,
            schema=schema,
            dimensions=dimensions,
            write_receipts=write_receipts,
        )
        assert source_validation_receipt is not None
        persisted_receipt = _strict_json(
            output_path / source_validation_receipt_relative_path
        )
        if persisted_receipt != source_validation_receipt:
            raise RuntimeError("Persisted source-validation receipt differs.")
    # Completion is the last scientific-publication mutation.  Everything
    # above, including consolidation, manifest reconstruction, source receipt
    # persistence, and logical/sample validation, ran while the child remained
    # explicitly incomplete and selector-ineligible.
    writable_run = zarr.open_group(
        str(output_path / family / str(run_id)),
        mode="a",
        use_consolidated=False,
    )
    writable_run.attrs["status"] = "complete"
    mark_run_complete(writable_run, run_name=str(run_id))
    consolidate_metadata_capture_expected_warnings(output_path)
    final_direct, final_consolidated = _metadata_maps(
        output_path, family=family, run_id=str(run_id), paths=paths
    )
    if _metadata_digest(final_direct, final_consolidated) != metadata_digest:
        raise RuntimeError(
            "Subject-mask metadata declarations changed during completion."
        )
    reopened = zarr.open_group(
        str(output_path / family / str(run_id)),
        mode="r",
        use_consolidated=False,
    )
    if reopened.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        raise RuntimeError("Subject-mask completion contract is absent.")
    if reopened.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "complete":
        raise RuntimeError("Subject-mask completion status is not complete.")
    if reopened.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError("Subject-mask core publication is selector eligible.")
    if reopened.attrs.get("validation_mode") != validation_mode.value:
        raise RuntimeError("Subject-mask publication validation mode changed.")
    if reopened.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE) != manifest:
        raise RuntimeError("Persisted subject-mask manifest differs.")
    persisted_manifest_errors = validate_subject_mask_core_run_manifest(
        reopened.attrs[SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE]
    )
    if persisted_manifest_errors:
        raise RuntimeError(
            "Persisted subject-mask core manifest failed deep validation: "
            + "; ".join(persisted_manifest_errors)
        )
    phases["final_consolidation_and_reopen_gate"] = time.perf_counter() - phase
    return SubjectMaskCorePublication(
        output_path=output_path,
        family=family,
        run_id=str(run_id),
        kind=kind,
        dimensions=dimensions,
        components=components,
        plans=plans,
        validation_mode=validation_mode,
        source_manifest=dict(source_manifest),
        manifest=manifest,
        phase_seconds=phases,
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM",
    "SUBJECT_MASK_CORE_COMPOSABLE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_COMPOSABLE_RUN_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_ID",
    "SUBJECT_MASK_CORE_COORDINATE_DEPENDENCY_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_ID",
    "SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE",
    "SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID",
    "SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR",
    "SubjectMaskCorePublication",
    "SubjectMaskCoreValidationMode",
    "build_subject_mask_core_coordinate_dependencies",
    "build_subject_mask_coordinate_successor_manifest",
    "publish_selector_ineligible_subject_mask_core_snapshot",
    "subject_mask_core_metadata_declaration_maps",
    "validate_receipt_bound_persisted_subject_mask_core_publication",
    "validate_persisted_subject_mask_core_publication",
    "validate_subject_mask_core_run_manifest",
]
