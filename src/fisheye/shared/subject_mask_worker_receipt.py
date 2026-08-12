"""Bounded semantic receipts emitted by subject-mask compute workers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import hashlib
import numpy as np

from fisheye.shared.subject_mask_attempt import (
    SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
    validate_subject_mask_attempt,
    validate_subject_mask_scientific_identity,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
    build_subject_mask_source_run_manifest,
    build_subject_mask_source_validation_receipt,
    subject_mask_array_unit_document,
    subject_mask_semantic_units_from_array_document,
)
from fisheye.shared.zarr.subject_mask_schema import (
    RawSubjectMaskSchema,
    RefinedSubjectMaskCoreSchema,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)

SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_ID = (
    "palette.subject_mask.worker_semantic_receipt"
)
SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_VERSION = 1
RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS = (
    "mask_probs_roi",
    "available_channels",
    "metrics/prob_max",
    "metrics/mask_present",
    "metrics/area_px",
    "metrics/centroid_xy",
    "metrics/centroid_valid",
    "metrics/bbox_xyxy",
    "metrics/bbox_valid",
)
REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS = (
    "masks_roi",
    "available_channels",
    "metrics/mask_present",
    "metrics/area_px",
    "metrics/centroid_xy",
    "metrics/centroid_valid",
    "metrics/bbox_xyxy",
    "metrics/bbox_valid",
)
SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_ID = (
    "palette.subject_mask.recording_assembly_identity"
)
SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_LEGACY_SCHEMA_VERSION = 2
SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION = 3
SUBJECT_MASK_RECORDING_COMMON_AUTHORITY_SCHEMA_ID = (
    "palette.subject_mask.recording_common_scientific_authority"
)
SUBJECT_MASK_RECORDING_COMMON_AUTHORITY_SCHEMA_VERSION = 1
SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_ID = (
    "palette.subject_mask.assignment_keypoint_collection"
)
SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_VERSION = 1


def _strict_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict canonical JSON: {exc}.") from exc


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _without_fields(value: Any, names: set[str]) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {str(key): item for key, item in value.items() if str(key) not in names}


def _normalize_expected_work_units(
    value: Sequence[Mapping[str, Any]],
    *,
    n_frames: int,
    n_rois: int,
) -> list[dict[str, object]]:
    expected_fields = {
        "work_unit_id",
        "work_unit_index",
        "source_clip_id",
        "source_clip_index",
        "frame_start",
        "frame_stop",
        "row_start",
        "row_stop",
    }
    canonical = _strict_copy(list(value), name="expected subject-mask work units")
    if not isinstance(canonical, list) or not canonical:
        raise ValueError("Expected subject-mask work units are absent.")
    frame_cursor = 0
    row_cursor = 0
    seen_ids: set[str] = set()
    clip_ids_by_index: dict[int, str] = {}
    previous_clip_index = -1
    result: list[dict[str, object]] = []
    for ordinal, unit in enumerate(canonical):
        if not isinstance(unit, dict) or set(unit) != expected_fields:
            raise ValueError("Expected subject-mask work-unit fields are not exact.")
        work_unit_id = unit.get("work_unit_id")
        clip_id = unit.get("source_clip_id")
        clip_index = unit.get("source_clip_index")
        frame_start = unit.get("frame_start")
        frame_stop = unit.get("frame_stop")
        row_start = unit.get("row_start")
        row_stop = unit.get("row_stop")
        if (
            type(work_unit_id) is not str
            or not work_unit_id.strip()
            or work_unit_id != work_unit_id.strip()
            or work_unit_id in seen_ids
            or unit.get("work_unit_index") != ordinal
            or type(clip_id) is not str
            or not clip_id.strip()
            or clip_id != clip_id.strip()
            or type(clip_index) is not int
            or clip_index < 0
            or clip_index < previous_clip_index
            or type(frame_start) is not int
            or type(frame_stop) is not int
            or frame_start != frame_cursor
            or frame_stop <= frame_start
            or frame_stop > int(n_frames)
            or type(row_start) is not int
            or type(row_stop) is not int
            or row_start != row_cursor
            or not (row_start <= row_stop <= int(n_rois))
        ):
            raise ValueError(
                "Expected subject-mask work units must exactly and contiguously "
                "cover the frame and row domains."
            )
        seen_ids.add(work_unit_id)
        existing_clip_id = clip_ids_by_index.setdefault(clip_index, clip_id)
        if existing_clip_id != clip_id:
            raise ValueError(
                "One subject-mask source_clip_index maps to multiple clip IDs."
            )
        previous_clip_index = clip_index
        frame_cursor = frame_stop
        row_cursor = row_stop
        result.append(dict(unit))
    if frame_cursor != int(n_frames) or row_cursor != int(n_rois):
        raise ValueError(
            "Expected subject-mask work units do not cover the complete recording."
        )
    return result


def _refined_source_binding(
    scientific_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = scientific_identity.get("payload")
    model = payload.get("model") if isinstance(payload, Mapping) else None
    binding = model.get("source_input_binding") if isinstance(model, Mapping) else None
    if not isinstance(binding, Mapping):
        raise ValueError("Refined worker lacks its exact raw source-input binding.")
    return binding


def _validate_refined_worker_source_join(
    refined_worker: Mapping[str, Any],
    raw_worker: Mapping[str, Any],
) -> None:
    if refined_worker.get("global_row_interval") != raw_worker.get(
        "global_row_interval"
    ):
        raise ValueError("Refined and raw worker row intervals differ.")
    science = refined_worker.get("scientific_identity")
    raw_science = raw_worker.get("scientific_identity")
    raw_receipt = raw_worker.get("worker_receipt")
    if not isinstance(science, Mapping) or not isinstance(raw_science, Mapping):
        raise ValueError("Refined/raw worker scientific identity is absent.")
    binding = _refined_source_binding(science)
    receipt_binding = binding.get("worker_semantic_receipt_binding")
    expected_receipt_binding = {
        "schema_id": (
            raw_receipt.get("schema_id") if isinstance(raw_receipt, Mapping) else None
        ),
        "schema_version": (
            raw_receipt.get("schema_version")
            if isinstance(raw_receipt, Mapping)
            else None
        ),
        "payload_digest": (
            raw_receipt.get("payload_digest")
            if isinstance(raw_receipt, Mapping)
            else None
        ),
        "document_sha256": (
            canonical_json_sha256(raw_receipt)
            if isinstance(raw_receipt, Mapping)
            else None
        ),
        "storage": "strict_json_sidecar_v1",
    }
    raw_run_path = raw_worker.get("run_path")
    binding_run_path = binding.get("run_path")
    exact_run_path = binding_run_path == raw_run_path
    legacy_single_worker_collection_path = (
        binding_run_path == "subject_mask_shard_runs/<collection>"
        and isinstance(raw_run_path, str)
        and isinstance(receipt_binding, Mapping)
        and receipt_binding.get("relative_path")
        == f"{raw_run_path}/worker_semantic_receipt.json"
    )
    if (
        not (exact_run_path or legacy_single_worker_collection_path)
        or binding.get("scientific_identity_digest") != raw_science.get("digest")
        or not isinstance(receipt_binding, Mapping)
        or not isinstance(raw_receipt, Mapping)
        or any(
            receipt_binding.get(name) != expected
            for name, expected in expected_receipt_binding.items()
        )
    ):
        raise ValueError(
            "Refined worker does not bind the exact corresponding raw worker."
        )


def validate_recording_subject_mask_refined_source_join(
    *,
    raw_producer_evidence: Mapping[str, Any],
    refined_producer_evidence: Mapping[str, Any],
    raw_source_run_path: str,
    refined_source_run_path: str,
    n_frames: int,
    n_rois: int,
) -> None:
    """Prove the persisted refined assembly was derived from these raw workers."""

    raw = validate_recording_subject_mask_assembly_identity(
        raw_producer_evidence,
        kind="raw_probability_uint8",
        stage_kind="raw_subject_mask",
        source_run_path=raw_source_run_path,
        n_frames=n_frames,
        n_rois=n_rois,
    )
    refined = validate_recording_subject_mask_assembly_identity(
        refined_producer_evidence,
        kind="refined_dense_core",
        stage_kind="refined_subject_mask",
        source_run_path=refined_source_run_path,
        n_frames=n_frames,
        n_rois=n_rois,
    )
    source_binding = refined.get("source_producer_binding")
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("schema_id") != raw.get("schema_id")
        or source_binding.get("schema_version") != raw.get("schema_version")
        or source_binding.get("kind") != raw.get("kind")
        or source_binding.get("source_run_path") != raw.get("source_run_path")
        or source_binding.get("digest") != canonical_json_sha256(raw)
    ):
        raise ValueError("Refined recording assembly binds another raw producer.")
    raw_workers = raw["workers"]
    refined_workers = refined["workers"]
    if len(raw_workers) != len(refined_workers):
        raise ValueError("Refined/raw recording worker counts differ.")
    for refined_worker, raw_worker in zip(
        refined_workers,
        raw_workers,
        strict=True,
    ):
        _validate_refined_worker_source_join(refined_worker, raw_worker)


def _recording_common_scientific_authority(
    scientific_identity: Mapping[str, Any],
    *,
    stage_kind: str,
) -> dict[str, object]:
    """Project worker-local science onto recording-wide invariant semantics."""

    payload = scientific_identity.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Subject-mask scientific identity payload is absent.")
    stage = str(stage_kind)
    nested = {
        name: payload.get(name)
        for name in ("model", "crop", "pixels", "inference_contract")
    }
    if any(not isinstance(value, Mapping) for value in nested.values()):
        raise ValueError("Subject-mask scientific authority fields must be objects.")
    model = dict(nested["model"])
    crop = dict(nested["crop"])
    pixels = dict(nested["pixels"])
    inference = dict(nested["inference_contract"])
    if stage == "raw_subject_mask":
        pixel_local = {
            "decoded_shape",
            "decoded_pixels_sha256",
            "declared_pixels_sha256",
            "cache_key",
            "pixel_materialization_id",
        }
        normalized_model = model
        normalized_pixels = _without_fields(pixels, pixel_local)
        normalized_inference = inference
    elif stage == "refined_subject_mask":
        normalized_model = _without_fields(model, {"source_input_binding"})
        normalized_pixels = _without_fields(
            pixels,
            {"source_input_binding", "digest"},
        )
        assignment = inference.get("eye_assignment_contract")
        if isinstance(assignment, Mapping):
            assignment = _without_fields(
                assignment,
                {
                    "assignment_keypoint_group",
                    "assignment_keypoints_run",
                    "assignment_keypoint_run",
                    "assignment_keypoint_row_identity",
                    "assignment_keypoint_coordinate_run_path",
                    "assignment_keypoint_roi_descriptor_ref",
                    "assignment_keypoint_roi_descriptor_sha256",
                    "assignment_keypoint_coordinate_derivation_ref",
                    "assignment_keypoint_coordinate_derivation_sha256",
                    "assignment_keypoint_row_identity_ref",
                    "assignment_keypoint_row_identity_sha256",
                },
            )
        normalized_inference = dict(inference)
        if "eye_assignment_contract" in normalized_inference:
            normalized_inference["eye_assignment_contract"] = assignment
        component_sources = normalized_inference.get(
            "component_sources_and_policies"
        )
        if isinstance(component_sources, Mapping):
            worker_local_component_fields = {
                "assignment_summary",
                "source_created_at_utc",
                "source_roi_cache_canonical_path",
                "source_roi_cache_key",
                "source_roi_cache_path",
            }
            normalized_inference["component_sources_and_policies"] = {
                str(component): (
                    _without_fields(policy, worker_local_component_fields)
                    if isinstance(policy, Mapping)
                    else policy
                )
                for component, policy in component_sources.items()
            }
    else:
        raise ValueError(f"Unsupported recording assembly stage {stage!r}.")
    document = {
        "schema_id": SUBJECT_MASK_RECORDING_COMMON_AUTHORITY_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_RECORDING_COMMON_AUTHORITY_SCHEMA_VERSION,
        "stage_kind": stage,
        "model": normalized_model,
        "crop_pixel_domain": {
            "roi_shape_hw": crop.get("roi_shape_hw"),
            "storage_mode": crop.get("storage_mode"),
        },
        "pixels": normalized_pixels,
        "inference_contract": normalized_inference,
    }
    return _strict_copy(document, name="recording common scientific authority")


def _validate_array_document(
    document: Mapping[str, Any],
    *,
    paths: Sequence[str],
) -> dict[str, dict[str, object]]:
    canonical = _strict_copy(document, name="worker array document")
    expected = tuple(str(path) for path in paths)
    if not isinstance(canonical, dict) or set(canonical) != set(expected):
        raise ValueError("Worker receipt array inventory is not exact.")
    for path in expected:
        record = canonical[path]
        if not isinstance(record, dict) or set(record) != {
            "shape",
            "dtype",
            "digest_algorithm",
            "unit_count",
            "units_digest",
            "units",
        }:
            raise ValueError(f"Worker array record fields differ for {path!r}.")
        shape = record.get("shape")
        units = record.get("units")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(type(value) is int and value > 0 for value in shape)
            or not str(record.get("dtype") or "").strip()
            or record.get("digest_algorithm")
            != SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM
            or not isinstance(units, list)
            or not units
        ):
            raise ValueError(f"Worker array record is invalid for {path!r}.")
        cursor = 0
        for unit in units:
            if not isinstance(unit, dict) or set(unit) != {
                "start_row",
                "stop_row",
                "decoded_bytes",
                "sha256",
            }:
                raise ValueError(f"Worker array unit fields differ for {path!r}.")
            start = unit.get("start_row")
            stop = unit.get("stop_row")
            if (
                type(start) is not int
                or type(stop) is not int
                or start != cursor
                or not (start < stop <= shape[0])
                or type(unit.get("decoded_bytes")) is not int
                or unit.get("decoded_bytes") <= 0
                or not _valid_sha256(unit.get("sha256"))
            ):
                raise ValueError(f"Worker array unit coverage differs for {path!r}.")
            cursor = stop
        if (
            cursor != shape[0]
            or record.get("unit_count") != len(units)
            or record.get("units_digest") != canonical_json_sha256(units)
        ):
            raise ValueError(f"Worker array units are incomplete for {path!r}.")
    return canonical


def build_subject_mask_worker_semantic_receipt(
    *,
    stage_kind: str,
    run_path: str,
    scientific_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    scope: Mapping[str, Any],
    row_count: int,
    array_document: Mapping[str, Any],
    required_paths: Sequence[str],
    roi_aligned_paths: Sequence[str],
) -> dict[str, object]:
    """Seal one worker's exact outputs without rereading its output store."""

    science_errors = validate_subject_mask_scientific_identity(scientific_identity)
    attempt_errors = validate_subject_mask_attempt(attempt)
    if science_errors or attempt_errors:
        raise ValueError(
            "Worker receipt identity is invalid: "
            f"science={list(science_errors)}, attempt={list(attempt_errors)}"
        )
    stage = str(stage_kind).strip()
    if stage not in {"raw_subject_mask", "refined_subject_mask"}:
        raise ValueError(f"Unsupported subject-mask worker stage {stage!r}.")
    resolved_path = str(run_path).strip().strip("/")
    if not resolved_path or "/" not in resolved_path:
        raise ValueError("Worker receipt run_path must include a family and run.")
    rows = int(row_count)
    if type(row_count) is not int or rows <= 0:
        raise ValueError("Worker receipt row_count must be a positive exact integer.")
    paths = tuple(str(path) for path in required_paths)
    roi_paths = tuple(str(path) for path in roi_aligned_paths)
    if not roi_paths or not set(roi_paths) <= set(paths):
        raise ValueError("Worker ROI-aligned paths must be a nonempty path subset.")
    arrays = _validate_array_document(array_document, paths=paths)
    if any(arrays[path]["shape"][0] != rows for path in roi_paths):
        raise ValueError("Worker ROI-aligned array row count differs.")
    units = subject_mask_semantic_units_from_array_document(
        arrays,
        n_rois=rows,
        paths=roi_paths,
    )
    scope_document = _strict_copy(scope, name="worker scope")
    if not isinstance(scope_document, dict):
        raise ValueError("Worker scope must be one JSON object.")
    payload = {
        "stage_kind": stage,
        "result": "valid",
        "run_path": resolved_path,
        "scientific_identity_digest": scientific_identity["digest"],
        "attempt_payload_digest": attempt["payload_digest"],
        "scope": scope_document,
        "local_row_interval": {"start_row": 0, "stop_row": rows},
        "required_output_paths": list(paths),
        "roi_aligned_paths": list(roi_paths),
        "arrays": arrays,
        "semantic_coverage": {
            "axis": "local_roi_row",
            "complete_nonoverlapping": True,
            "unit_count": len(units),
            "units_digest": canonical_json_sha256(units),
            "units": list(units),
        },
    }
    envelope = {
        "schema_id": SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    validate_subject_mask_worker_semantic_receipt(
        envelope,
        scientific_identity=scientific_identity,
        attempt=attempt,
        required_paths=paths,
    )
    return envelope


def validate_subject_mask_worker_semantic_receipt(
    receipt: Mapping[str, Any],
    *,
    scientific_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    required_paths: Sequence[str],
) -> dict[str, object]:
    canonical = _strict_copy(receipt, name="worker semantic receipt")
    if not isinstance(canonical, dict) or set(canonical) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        raise ValueError("Worker semantic receipt fields are not exact.")
    payload = canonical.get("payload")
    if (
        canonical.get("schema_id") != SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_ID
        or canonical.get("schema_version")
        != SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_VERSION
        or canonical.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, dict)
        or canonical.get("payload_digest") != canonical_json_sha256(payload)
    ):
        raise ValueError("Worker semantic receipt is unsupported or stale.")
    if set(payload) != {
        "stage_kind",
        "result",
        "run_path",
        "scientific_identity_digest",
        "attempt_payload_digest",
        "scope",
        "local_row_interval",
        "required_output_paths",
        "roi_aligned_paths",
        "arrays",
        "semantic_coverage",
    }:
        raise ValueError("Worker semantic receipt payload fields are not exact.")
    paths = tuple(str(path) for path in required_paths)
    if (
        payload.get("result") != "valid"
        or payload.get("scientific_identity_digest")
        != scientific_identity.get("digest")
        or payload.get("attempt_payload_digest") != attempt.get("payload_digest")
        or payload.get("required_output_paths") != list(paths)
    ):
        raise ValueError("Worker semantic receipt identity binding changed.")
    arrays = _validate_array_document(payload.get("arrays"), paths=paths)
    interval = payload.get("local_row_interval")
    if not isinstance(interval, dict) or set(interval) != {"start_row", "stop_row"}:
        raise ValueError("Worker semantic local row interval is invalid.")
    rows = interval.get("stop_row")
    if interval.get("start_row") != 0 or type(rows) is not int or rows <= 0:
        raise ValueError("Worker semantic local row interval is invalid.")
    roi_paths = payload.get("roi_aligned_paths")
    if not isinstance(roi_paths, list) or not roi_paths:
        raise ValueError("Worker semantic ROI paths are absent.")
    units = subject_mask_semantic_units_from_array_document(
        arrays, n_rois=rows, paths=roi_paths
    )
    coverage = payload.get("semantic_coverage")
    expected_coverage = {
        "axis": "local_roi_row",
        "complete_nonoverlapping": True,
        "unit_count": len(units),
        "units_digest": canonical_json_sha256(units),
        "units": list(units),
    }
    if coverage != expected_coverage:
        raise ValueError("Worker semantic coverage changed.")
    return canonical


def validate_recording_subject_mask_assembly_identity(
    value: Mapping[str, Any],
    *,
    kind: str,
    stage_kind: str,
    source_run_path: str,
    n_rois: int,
    n_frames: int | None = None,
) -> dict[str, object]:
    """Deeply validate retained recording-assembly producer evidence."""

    canonical = _strict_copy(value, name="recording assembly identity")
    schema_version = (
        canonical.get("schema_version") if isinstance(canonical, dict) else None
    )
    legacy_fields = {
        "schema_id",
        "schema_version",
        "kind",
        "source_run_path",
        "row_policy",
        "context",
        "common_scientific_authority",
        "workers",
    }
    current_fields = legacy_fields | {
        "work_unit_coverage",
        "source_producer_binding",
    }
    expected_fields = (
        current_fields
        if schema_version == SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION
        else legacy_fields
    )
    if not isinstance(canonical, dict) or set(canonical) != expected_fields:
        raise ValueError("Recording assembly identity fields are not exact.")
    if (
        canonical.get("schema_id") != SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_ID
        or canonical.get("schema_version")
        not in {
            SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_LEGACY_SCHEMA_VERSION,
            SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION,
        }
        or canonical.get("kind") != str(kind)
        or canonical.get("source_run_path") != str(source_run_path).strip().strip("/")
        or canonical.get("row_policy") != "ordered_contiguous_real_work_units_v1"
        or not isinstance(canonical.get("context"), dict)
    ):
        raise ValueError("Recording assembly identity header changed.")
    stage = str(stage_kind).strip()
    required_paths = (
        RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS
        if stage == "raw_subject_mask"
        else (
            REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
            if stage == "refined_subject_mask"
            else ()
        )
    )
    if not required_paths:
        raise ValueError(f"Unsupported recording assembly stage {stage!r}.")
    workers = canonical.get("workers")
    common_authority = canonical.get("common_scientific_authority")
    if not isinstance(common_authority, dict):
        raise ValueError("Recording common scientific authority is absent.")
    if not isinstance(workers, list) or not workers:
        raise ValueError("Recording assembly workers are absent.")
    current = schema_version == SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION
    work_unit_coverage = canonical.get("work_unit_coverage")
    source_binding = canonical.get("source_producer_binding")
    expected_nonempty: list[dict[str, object]] = []
    if current:
        if not isinstance(work_unit_coverage, dict) or set(work_unit_coverage) != {
            "policy",
            "n_frames",
            "n_rois",
            "unit_count",
            "units_digest",
            "units",
        }:
            raise ValueError("Recording expected work-unit coverage is invalid.")
        declared_frames = work_unit_coverage.get("n_frames")
        if type(declared_frames) is not int or declared_frames <= 0:
            raise ValueError("Recording expected frame domain is invalid.")
        if n_frames is not None and declared_frames != int(n_frames):
            raise ValueError("Recording expected frame domain changed.")
        units = work_unit_coverage.get("units")
        normalized_units = _normalize_expected_work_units(
            units if isinstance(units, list) else [],
            n_frames=declared_frames,
            n_rois=int(n_rois),
        )
        if (
            work_unit_coverage.get("policy")
            != "authoritative_recording_plan_including_empty_windows_v1"
            or work_unit_coverage.get("n_rois") != int(n_rois)
            or work_unit_coverage.get("unit_count") != len(normalized_units)
            or work_unit_coverage.get("units_digest")
            != canonical_json_sha256(normalized_units)
        ):
            raise ValueError("Recording expected work-unit coverage changed.")
        expected_nonempty = [
            unit for unit in normalized_units if unit["row_stop"] > unit["row_start"]
        ]
        if len(expected_nonempty) != len(workers):
            raise ValueError(
                "Recording workers do not match the nonempty authoritative work units."
            )
        if stage == "raw_subject_mask" and source_binding is not None:
            raise ValueError("Raw recording assembly cannot bind another producer.")
        if stage == "refined_subject_mask":
            if not isinstance(source_binding, dict) or set(source_binding) != {
                "schema_id",
                "schema_version",
                "kind",
                "source_run_path",
                "digest",
            }:
                raise ValueError("Refined raw-producer binding is invalid.")
            if (
                source_binding.get("schema_id")
                != SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_ID
                or source_binding.get("schema_version")
                != SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION
                or source_binding.get("kind") != "raw_probability_uint8"
                or type(source_binding.get("source_run_path")) is not str
                or not _valid_sha256(source_binding.get("digest"))
            ):
                raise ValueError("Refined raw-producer binding changed.")
    cursor = 0
    for worker_index, worker in enumerate(workers):
        if not isinstance(worker, dict) or set(worker) != {
            "global_row_interval",
            "run_path",
            "scientific_identity_digest",
            "attempt_payload_digest",
            "worker_receipt_payload_digest",
            "scientific_identity",
            "attempt",
            "worker_receipt",
        }:
            raise ValueError("Recording assembly worker fields are not exact.")
        interval = worker.get("global_row_interval")
        if not isinstance(interval, dict) or set(interval) != {
            "start_row",
            "stop_row",
        }:
            raise ValueError("Recording assembly worker interval is invalid.")
        start = interval.get("start_row")
        stop = interval.get("stop_row")
        if (
            type(start) is not int
            or type(stop) is not int
            or start != cursor
            or not (start < stop <= int(n_rois))
        ):
            raise ValueError(
                "Recording assembly workers must cover ordered contiguous rows."
            )
        science = worker.get("scientific_identity")
        attempt = worker.get("attempt")
        receipt = worker.get("worker_receipt")
        if not isinstance(science, dict) or validate_subject_mask_scientific_identity(
            science
        ):
            raise ValueError("Recording worker scientific identity is invalid.")
        if current and science.get("schema_version") != (
            SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Current recording assembly requires scientific identity v2."
            )
        if science["payload"].get("stage_kind") != stage:
            raise ValueError("Recording worker scientific stage differs.")
        if (
            _recording_common_scientific_authority(science, stage_kind=stage)
            != common_authority
        ):
            raise ValueError("Recording workers have conflicting scientific authority.")
        if not isinstance(attempt, dict) or validate_subject_mask_attempt(attempt):
            raise ValueError("Recording worker attempt identity is invalid.")
        attempt_payload = attempt["payload"]
        if attempt_payload.get("scientific_identity_digest") != science.get("digest"):
            raise ValueError("Recording worker attempt binds another science identity.")
        if not isinstance(receipt, dict):
            raise ValueError("Recording worker semantic receipt is absent.")
        receipt_payload_value = receipt.get("payload")
        receipt_paths = (
            receipt_payload_value.get("required_output_paths")
            if isinstance(receipt_payload_value, Mapping)
            else None
        )
        allowed_paths = [list(required_paths)]
        if stage == "raw_subject_mask":
            allowed_paths.append([required_paths[0], "masks_roi", *required_paths[1:]])
        if receipt_paths not in allowed_paths:
            raise ValueError("Recording worker output inventory is unsupported.")
        validated_receipt = validate_subject_mask_worker_semantic_receipt(
            receipt,
            scientific_identity=science,
            attempt=attempt,
            required_paths=tuple(receipt_paths),
        )
        receipt_payload = validated_receipt["payload"]
        if (
            worker.get("run_path") != receipt_payload.get("run_path")
            or worker.get("scientific_identity_digest") != science.get("digest")
            or worker.get("attempt_payload_digest") != attempt.get("payload_digest")
            or worker.get("worker_receipt_payload_digest")
            != receipt.get("payload_digest")
            or int(receipt_payload["local_row_interval"]["stop_row"]) != stop - start
        ):
            raise ValueError("Recording worker retained bindings changed.")
        if current:
            expected_unit = expected_nonempty[worker_index]
            if start != expected_unit["row_start"] or stop != expected_unit["row_stop"]:
                raise ValueError(
                    "Recording worker interval differs from its authoritative work unit."
                )
            if stage == "raw_subject_mask":
                crop = science["payload"]["crop"]
                partition = crop.get("collection_partition_contract")
                partition_payload = (
                    partition.get("payload") if isinstance(partition, Mapping) else None
                )
                partition_collection = (
                    partition_payload.get("collection")
                    if isinstance(partition_payload, Mapping)
                    else None
                )
                partition_window = (
                    partition_payload.get("frame_window")
                    if isinstance(partition_payload, Mapping)
                    else None
                )
                partition_rows = (
                    partition_payload.get("crop_rows")
                    if isinstance(partition_payload, Mapping)
                    else None
                )
                if (
                    crop.get("source_work_unit_id") != expected_unit["work_unit_id"]
                    or crop.get("source_clip_id") != expected_unit["source_clip_id"]
                    or crop.get("source_clip_index")
                    != expected_unit["source_clip_index"]
                ):
                    raise ValueError(
                        "Raw worker identity differs from its authoritative work unit."
                    )
                if (
                    not isinstance(partition_collection, Mapping)
                    or not isinstance(partition_window, Mapping)
                    or not isinstance(partition_rows, Mapping)
                    or partition_collection.get("source_work_unit_id")
                    != expected_unit["work_unit_id"]
                    or partition_collection.get("source_clip_id")
                    != expected_unit["source_clip_id"]
                    or partition_collection.get("source_clip_index")
                    != expected_unit["source_clip_index"]
                    or partition_window.get("actual_start_frame")
                    != expected_unit["frame_start"]
                    or partition_window.get("end_frame_exclusive")
                    != expected_unit["frame_stop"]
                    or partition_rows.get("start") != expected_unit["row_start"]
                    or partition_rows.get("stop") != expected_unit["row_stop"]
                ):
                    raise ValueError(
                        "Raw worker partition proof differs from its authoritative "
                        "work unit."
                    )
        cursor = int(stop)
    if cursor != int(n_rois):
        raise ValueError("Recording assembly does not cover every ROI row.")
    return canonical


def build_recording_assignment_keypoint_collection(
    producer_evidence: Mapping[str, Any],
    *,
    source_run_path: str,
    n_rois: int,
    n_frames: int | None = None,
) -> dict[str, object]:
    """Retain the exact ordered keypoint-assignment inputs of refinement."""

    evidence = validate_recording_subject_mask_assembly_identity(
        producer_evidence,
        kind="refined_dense_core",
        stage_kind="refined_subject_mask",
        source_run_path=source_run_path,
        n_rois=n_rois,
        n_frames=n_frames,
    )
    workers: list[dict[str, object]] = []
    modes: set[str] = set()
    required = {
        "assignment_keypoints_run",
        "assignment_keypoint_group",
        "assignment_keypoint_contract",
        "assignment_keypoint_role",
        "assignment_keypoint_selection",
        "assignment_keypoint_success_dataset",
        "assignment_keypoint_row_identity",
        "assignment_keypoint_row_identity_check",
    }
    coordinate_fields = {
        "assignment_keypoint_coordinate_contract",
        "assignment_keypoint_coordinate_run_path",
        "assignment_keypoint_roi_descriptor_ref",
        "assignment_keypoint_roi_descriptor_sha256",
        "assignment_keypoint_coordinate_derivation_ref",
        "assignment_keypoint_coordinate_derivation_sha256",
        "assignment_keypoint_row_identity_ref",
        "assignment_keypoint_row_identity_sha256",
        "assignment_keypoint_eye_indices",
    }
    for worker in evidence["workers"]:
        interval = worker["global_row_interval"]
        start = int(interval["start_row"])
        stop = int(interval["stop_row"])
        inference = worker["scientific_identity"]["payload"]["inference_contract"]
        assignment = inference.get("eye_assignment_contract")
        if assignment is None:
            modes.add("not_used")
            workers.append(
                {
                    "global_row_interval": dict(interval),
                    "assignment": None,
                }
            )
            continue
        if not isinstance(assignment, Mapping):
            raise ValueError("Worker eye-assignment contract must be an object.")
        fields = set(assignment)
        if fields != required and fields != required | coordinate_fields:
            raise ValueError("Worker eye-assignment contract fields are not exact.")
        modes.add("exact_worker_partition")
        group = assignment.get("assignment_keypoint_group")
        run = assignment.get("assignment_keypoints_run")
        success = assignment.get("assignment_keypoint_success_dataset")
        selection = assignment.get("assignment_keypoint_selection")
        if (
            any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or "/" in value
                for value in (group, run, success)
            )
            or not isinstance(selection, str)
            or not selection.strip()
        ):
            raise ValueError("Worker assignment keypoint path fields are invalid.")
        if (
            assignment.get("assignment_keypoint_contract")
            != "subject_eyes_union_assignment_keypoints_v1"
            or assignment.get("assignment_keypoint_role") != "eyes_union_lr_assignment"
            or assignment.get("assignment_keypoint_row_identity_check")
            != "source_crop_row_ids_subset"
        ):
            raise ValueError("Worker assignment keypoint semantics changed.")
        row_identity = assignment.get("assignment_keypoint_row_identity")
        if not isinstance(row_identity, Mapping) or set(row_identity) != {
            "row_identity_check",
            "rows_checked",
            "keypoint_has_source_crop_row_ids",
            "mask_has_source_crop_row_ids",
            "keypoint_rows_available",
            "keypoint_rows_selected",
            "keypoint_selection_min_row",
            "keypoint_selection_max_row",
        }:
            raise ValueError("Worker assignment keypoint row identity is invalid.")
        if (
            row_identity.get("row_identity_check") != "source_crop_row_ids_subset"
            or row_identity.get("keypoint_has_source_crop_row_ids") is not True
            or row_identity.get("mask_has_source_crop_row_ids") is not True
            or row_identity.get("rows_checked") != stop - start
            or row_identity.get("keypoint_rows_selected") != stop - start
            or row_identity.get("keypoint_selection_min_row") != start
            or row_identity.get("keypoint_selection_max_row") != stop - 1
            or type(row_identity.get("keypoint_rows_available")) is not int
            or row_identity["keypoint_rows_available"] < stop
        ):
            raise ValueError("Worker assignment keypoint row coverage changed.")
        if coordinate_fields <= fields:
            if (
                assignment.get("assignment_keypoint_coordinate_contract")
                != "canonical_v2_exact"
                or assignment.get("assignment_keypoint_coordinate_run_path")
                != f"{group}/{run}"
            ):
                raise ValueError("Worker assignment coordinate source changed.")
            for name in (
                "assignment_keypoint_roi_descriptor_sha256",
                "assignment_keypoint_coordinate_derivation_sha256",
                "assignment_keypoint_row_identity_sha256",
            ):
                if not _valid_sha256(assignment.get(name)):
                    raise ValueError(f"Worker {name} is not a SHA-256 digest.")
            for name in (
                "assignment_keypoint_roi_descriptor_ref",
                "assignment_keypoint_coordinate_derivation_ref",
                "assignment_keypoint_row_identity_ref",
            ):
                value = assignment.get(name)
                if not isinstance(value, str) or not value.startswith("/"):
                    raise ValueError(f"Worker {name} is not an absolute record ref.")
            eye_indices = assignment.get("assignment_keypoint_eye_indices")
            if (
                not isinstance(eye_indices, Mapping)
                or set(eye_indices) != {"eye_left", "eye_right"}
                or any(
                    type(value) is not int or value < 0
                    for value in eye_indices.values()
                )
                or eye_indices["eye_left"] == eye_indices["eye_right"]
            ):
                raise ValueError("Worker assignment eye indices are invalid.")
        workers.append(
            {
                "global_row_interval": dict(interval),
                "assignment": _strict_copy(
                    assignment,
                    name="assignment keypoint contract",
                ),
            }
        )
    if len(modes) != 1:
        raise ValueError(
            "Refined recording workers cannot mix keypoint-assigned and "
            "unassigned eye semantics."
        )
    return {
        "schema_id": SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_VERSION,
        "mode": next(iter(modes)),
        "row_policy": "ordered_contiguous_recording_crop_rows_v1",
        "n_rois": int(n_rois),
        "workers": workers,
    }


def build_recording_subject_mask_source_receipt(
    *,
    kind: str,
    stage_kind: str,
    source_run_path: str,
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    arrays: Mapping[str, Any],
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    threshold: float | None,
    workers: Sequence[Mapping[str, Any]],
    identity_unit_rows: int = 131_072,
    assembly_context: Mapping[str, Any] | None = None,
    expected_work_units: Sequence[Mapping[str, Any]] | None = None,
    source_producer_evidence: Mapping[str, Any] | None = None,
    source_producer_run_path: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Aggregate ordered whole/real-clip receipts into publication evidence.

    Payload hashes are reused from workers. Narrow identity/index arrays are
    hashed once by the recording coordinator. Worker evidence may only be
    concatenated; reordering within or across worker intervals fails closed.
    """

    stage = str(stage_kind).strip()
    expected_base: tuple[str, ...]
    payload_paths: tuple[str, ...]
    if stage == "raw_subject_mask":
        expected_base = RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS
        payload_paths = tuple(
            path for path in expected_base if path != "available_channels"
        )
        if "masks_roi" in arrays:
            payload_paths = (payload_paths[0], "masks_roi", *payload_paths[1:])
    elif stage == "refined_subject_mask":
        expected_base = REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
        payload_paths = tuple(
            path for path in expected_base if path != "available_channels"
        )
    else:
        raise ValueError(f"Unsupported recording receipt stage {stage!r}.")
    expected_worker_paths = list(expected_base)
    if stage == "raw_subject_mask" and "masks_roi" in arrays:
        expected_worker_paths.insert(1, "masks_roi")
    schema_paths = tuple(
        binding.path
        for binding in schema.bindings
        if binding.required or binding.path in arrays
    )
    if set(arrays) != set(schema_paths):
        raise ValueError("Recording source array inventory differs from its schema.")

    ordered = sorted(
        [dict(item) for item in workers],
        key=lambda item: int(item.get("global_start_row", -1)),
    )
    if not ordered:
        raise ValueError("Recording source receipt requires worker evidence.")
    current_contract = expected_work_units is not None
    normalized_work_units: list[dict[str, object]] | None = None
    if current_contract:
        normalized_work_units = _normalize_expected_work_units(
            expected_work_units,
            n_frames=dimensions.n_frames,
            n_rois=dimensions.n_rois,
        )
        nonempty_count = sum(
            int(unit["row_stop"]) > int(unit["row_start"])
            for unit in normalized_work_units
        )
        if nonempty_count != len(ordered):
            raise ValueError(
                "Worker count differs from the nonempty authoritative work units."
            )
    validated_source_evidence: dict[str, object] | None = None
    if stage == "raw_subject_mask":
        if source_producer_evidence is not None or source_producer_run_path is not None:
            raise ValueError("Raw recording assembly cannot bind another producer.")
    elif current_contract:
        if source_producer_evidence is None or source_producer_run_path is None:
            raise ValueError(
                "Current refined recording assembly requires exact raw producer evidence."
            )
        validated_source_evidence = validate_recording_subject_mask_assembly_identity(
            source_producer_evidence,
            kind="raw_probability_uint8",
            stage_kind="raw_subject_mask",
            source_run_path=source_producer_run_path,
            n_rois=dimensions.n_rois,
            n_frames=dimensions.n_frames,
        )
        if validated_source_evidence.get("schema_version") != (
            SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Current refined recording assembly requires current raw evidence."
            )
    cursor = 0
    payload_units: dict[str, list[dict[str, object]]] = {
        path: [] for path in payload_paths
    }
    semantic_units: list[dict[str, object]] = []
    worker_bindings: list[dict[str, object]] = []
    available_actual = np.ascontiguousarray(np.asarray(arrays["available_channels"][:]))
    for item in ordered:
        if set(item) != {
            "global_start_row",
            "scientific_identity",
            "attempt",
            "receipt",
        }:
            raise ValueError("Recording worker evidence fields are not exact.")
        start = item["global_start_row"]
        if type(start) is not int or start != cursor:
            raise ValueError(
                "Recording worker intervals must be ordered, contiguous, and "
                "begin at row zero."
            )
        science = item["scientific_identity"]
        attempt = item["attempt"]
        receipt = validate_subject_mask_worker_semantic_receipt(
            item["receipt"],
            scientific_identity=science,
            attempt=attempt,
            required_paths=expected_worker_paths,
        )
        worker_payload = receipt["payload"]
        if worker_payload.get("stage_kind") != stage:
            raise ValueError("Recording worker stage differs.")
        interval = worker_payload["local_row_interval"]
        local_rows = int(interval["stop_row"])
        stop = start + local_rows
        if stop > dimensions.n_rois:
            raise ValueError("Recording worker interval exceeds the recording rows.")
        worker_arrays = worker_payload["arrays"]
        available_record = worker_arrays["available_channels"]
        available_units = available_record["units"]
        if (
            available_record["shape"] != list(available_actual.shape)
            or available_record["dtype"] != str(available_actual.dtype)
            or not available_units
        ):
            raise ValueError("Worker available-channel evidence differs.")
        for unit in available_units:
            unit_values = np.ascontiguousarray(
                available_actual[int(unit["start_row"]) : int(unit["stop_row"])]
            )
            if (
                int(unit["decoded_bytes"]) != int(unit_values.nbytes)
                or unit["sha256"]
                != hashlib.sha256(unit_values.view(np.uint8)).hexdigest()
            ):
                raise ValueError("Worker available-channel evidence differs.")
        for path in payload_paths:
            record = worker_arrays[path]
            actual_shape = tuple(int(value) for value in arrays[path].shape)
            local_shape = tuple(int(value) for value in record["shape"])
            if (
                local_shape[0] != local_rows
                or local_shape[1:] != actual_shape[1:]
                or record["dtype"] != str(np.dtype(arrays[path].dtype))
            ):
                raise ValueError(f"Worker output contract differs for {path!r}.")
            for unit in record["units"]:
                shifted = dict(unit)
                shifted["start_row"] = start + int(unit["start_row"])
                shifted["stop_row"] = start + int(unit["stop_row"])
                payload_units[path].append(shifted)
        for unit in worker_payload["semantic_coverage"]["units"]:
            shifted = dict(unit)
            shifted["start_row"] = start + int(unit["start_row"])
            shifted["stop_row"] = start + int(unit["stop_row"])
            shifted["evidence_digest"] = canonical_json_sha256(
                {
                    "global_start_row": shifted["start_row"],
                    "global_stop_row": shifted["stop_row"],
                    "worker_run_path": worker_payload["run_path"],
                    "worker_evidence_digest": unit["evidence_digest"],
                    "worker_receipt_payload_digest": receipt["payload_digest"],
                }
            )
            semantic_units.append(shifted)
        worker_bindings.append(
            {
                "global_row_interval": {
                    "start_row": start,
                    "stop_row": stop,
                },
                "run_path": worker_payload["run_path"],
                "scientific_identity_digest": science["digest"],
                "attempt_payload_digest": attempt["payload_digest"],
                "worker_receipt_payload_digest": receipt["payload_digest"],
                "scientific_identity": dict(science),
                "attempt": dict(attempt),
                "worker_receipt": dict(receipt),
            }
        )
        cursor = stop
    if cursor != dimensions.n_rois:
        raise ValueError("Recording worker evidence does not cover every ROI row.")

    array_document = subject_mask_array_unit_document(
        arrays,
        tuple(path for path in schema_paths if path not in payload_paths),
        unit_rows=max(1, int(identity_unit_rows)),
    )
    for path in payload_paths:
        units = payload_units[path]
        array_document[path] = {
            "shape": [int(value) for value in arrays[path].shape],
            "dtype": str(np.dtype(arrays[path].dtype)),
            "digest_algorithm": SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
            "unit_count": len(units),
            "units_digest": canonical_json_sha256(units),
            "units": units,
        }
    array_document = {path: array_document[path] for path in schema_paths}
    common_authority = _recording_common_scientific_authority(
        worker_bindings[0]["scientific_identity"],
        stage_kind=stage,
    )
    assembly_identity: dict[str, object] = {
        "schema_id": SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_ID,
        "schema_version": (
            SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION
            if current_contract
            else SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_LEGACY_SCHEMA_VERSION
        ),
        "kind": str(kind),
        "source_run_path": str(source_run_path).strip().strip("/"),
        "row_policy": "ordered_contiguous_real_work_units_v1",
        "context": _strict_copy(
            dict(assembly_context or {}), name="recording assembly context"
        ),
        "common_scientific_authority": common_authority,
        "workers": worker_bindings,
    }
    if current_contract:
        assert normalized_work_units is not None
        assembly_identity["work_unit_coverage"] = {
            "policy": "authoritative_recording_plan_including_empty_windows_v1",
            "n_frames": int(dimensions.n_frames),
            "n_rois": int(dimensions.n_rois),
            "unit_count": len(normalized_work_units),
            "units_digest": canonical_json_sha256(normalized_work_units),
            "units": normalized_work_units,
        }
        if stage == "raw_subject_mask":
            assembly_identity["source_producer_binding"] = None
        else:
            assert validated_source_evidence is not None
            raw_workers = validated_source_evidence["workers"]
            if len(raw_workers) != len(worker_bindings):
                raise ValueError("Refined/raw worker counts differ.")
            for refined_worker, raw_worker in zip(
                worker_bindings,
                raw_workers,
                strict=True,
            ):
                _validate_refined_worker_source_join(refined_worker, raw_worker)
            assembly_identity["source_producer_binding"] = {
                "schema_id": validated_source_evidence["schema_id"],
                "schema_version": validated_source_evidence["schema_version"],
                "kind": validated_source_evidence["kind"],
                "source_run_path": validated_source_evidence["source_run_path"],
                "digest": canonical_json_sha256(validated_source_evidence),
            }
    assembly_identity = validate_recording_subject_mask_assembly_identity(
        assembly_identity,
        kind=kind,
        stage_kind=stage,
        source_run_path=source_run_path,
        n_rois=dimensions.n_rois,
        n_frames=dimensions.n_frames,
    )
    source_manifest = build_subject_mask_source_run_manifest(
        kind=kind,
        run_path=source_run_path,
        schema=schema,
        dimensions=dimensions,
        components=components,
        threshold=threshold,
        producer_identity_schema_id=str(assembly_identity["schema_id"]),
        producer_identity_digest=canonical_json_sha256(assembly_identity),
        attempt_payload_digest=(
            str(worker_bindings[0]["attempt_payload_digest"])
            if len(worker_bindings) == 1
            else None
        ),
        array_document=array_document,
    )
    receipt = build_subject_mask_source_validation_receipt(
        kind=kind,
        source_run_path=source_run_path,
        source_manifest=source_manifest,
        schema=schema,
        arrays=arrays,
        dimensions=dimensions,
        components=components,
        threshold=threshold,
        array_document=array_document,
        semantic_units=semantic_units,
        producer_evidence=assembly_identity,
    )
    return source_manifest, receipt


__all__ = [
    "RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS",
    "REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS",
    "SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_ID",
    "SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_VERSION",
    "SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_ID",
    "SUBJECT_MASK_RECORDING_ASSEMBLY_IDENTITY_SCHEMA_VERSION",
    "SUBJECT_MASK_RECORDING_COMMON_AUTHORITY_SCHEMA_ID",
    "SUBJECT_MASK_RECORDING_COMMON_AUTHORITY_SCHEMA_VERSION",
    "SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_ID",
    "SUBJECT_MASK_ASSIGNMENT_KEYPOINT_COLLECTION_SCHEMA_VERSION",
    "build_recording_assignment_keypoint_collection",
    "build_subject_mask_worker_semantic_receipt",
    "build_recording_subject_mask_source_receipt",
    "validate_subject_mask_worker_semantic_receipt",
    "validate_recording_subject_mask_assembly_identity",
    "validate_recording_subject_mask_refined_source_join",
]
