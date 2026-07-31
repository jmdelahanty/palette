"""Bounded semantic receipts emitted by subject-mask compute workers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import hashlib
import numpy as np

from fisheye.shared.subject_mask_attempt import (
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
    assembly_identity = {
        "schema_id": "palette.subject_mask.recording_assembly_identity",
        "schema_version": 1,
        "kind": str(kind),
        "source_run_path": str(source_run_path).strip().strip("/"),
        "row_policy": "ordered_contiguous_real_work_units_v1",
        "context": _strict_copy(
            dict(assembly_context or {}), name="recording assembly context"
        ),
        "workers": worker_bindings,
    }
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
    )
    return source_manifest, receipt


__all__ = [
    "RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS",
    "REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS",
    "SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_ID",
    "SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_SCHEMA_VERSION",
    "build_subject_mask_worker_semantic_receipt",
    "build_recording_subject_mask_source_receipt",
    "validate_subject_mask_worker_semantic_receipt",
]
