"""Immutable selector-ineligible publication for subject-mask core snapshots.

This adapter is deliberately independent of inference and refinement.  It
accepts one already complete logical core, validates it against the exact
subject-mask schema, rematerializes it through the shared byte planner, and
publishes a closed-world Zarr v3 store with consolidated metadata.
"""

from __future__ import annotations

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
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    RawSubjectMaskSchema,
    RefinedSubjectMaskCoreSchema,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    SubjectMaskProbabilityEncoding,
)
from fisheye.shared.zarr.subject_mask_storage import (
    SubjectMaskStoragePlanSet,
    plan_raw_subject_mask_storage,
    plan_refined_subject_mask_publication_storage,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM,
    SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
    validate_subject_mask_source_validation_receipt,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)

SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID = "palette.subject_mask_core.run_manifest"
SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_VERSION = 2
SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR = "source_validation_receipt.json"
SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_ID = "palette.subject_mask_core.shadow_publication"
SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_VERSION = 1
SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM = SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM
SUBJECT_MASK_CORE_METADATA_DIGEST_SCOPE = (
    "exact_run_group_and_array_declarations_redacting_only_run_manifest"
)
SUBJECT_MASK_PROB_MAX_CANONICALIZATION = "cpu_max_encoded_then_decode_float32_v1"
SUBJECT_MASK_PROB_MAX_SOURCE_MAX_ABS_TOLERANCE = float(np.finfo(np.float32).eps)


class SubjectMaskCoreValidationMode(str, Enum):
    """Publication-time validation cost and evidence boundary."""

    REFERENCE_FULL = "reference_full_v1"
    PRODUCTION_STREAMING = "production_streaming_v1"


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
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    prefix = f"{family}/{run_id}"
    direct: dict[str, dict[str, Any]] = {
        "": _strict_json(output_path / prefix / "zarr.json")
    }
    for path in paths:
        direct[path] = _strict_json(output_path / prefix / path / "zarr.json")
    root = _strict_json(output_path / "zarr.json")
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
            attrs.pop(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE, None)
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


def _write_physical_units(
    destination: Any,
    source: Any,
    plan: Any,
    *,
    source_validation_record: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Write whole physical units and hash the exact source bytes once."""

    unit = plan.shard_shape or plan.chunk_shape
    if unit is None:
        raise ValueError("Subject-mask core arrays cannot be scalar.")
    shape, dtype = _shape_dtype(source)
    trailing = (slice(None),) * (len(shape) - 1)
    writes = 0
    logical_digest = hashlib.sha256()
    samples: list[dict[str, object]] = []
    expected_units: list[Mapping[str, Any]] | None = None
    expected_unit_index = 0
    expected_unit_digest = hashlib.sha256()
    if source_validation_record is not None:
        algorithm = source_validation_record.get("digest_algorithm")
        if algorithm == SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM:
            raw_units = source_validation_record.get("units")
            if not isinstance(raw_units, list) or not raw_units:
                raise ValueError("Source-validation row units are absent.")
            expected_units = list(raw_units)
        elif algorithm != SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM:
            raise ValueError("Source-validation array digest algorithm is unsupported.")
    starts = tuple(range(0, shape[0], max(1, int(unit[0]))))
    for index, start in enumerate(starts):
        stop = min(shape[0], start + max(1, int(unit[0])))
        selection = (slice(start, stop), *trailing)
        values = np.ascontiguousarray(np.asarray(source[selection]))
        if values.dtype.hasobject:
            raise TypeError("Subject-mask core arrays cannot use object dtype.")
        unit_bytes = values.view(np.uint8)
        logical_digest.update(unit_bytes)
        if expected_units is not None:
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
                            "Bytes streamed into the subject-mask publication differ "
                            "from the validated source receipt."
                        )
                    expected_unit_index += 1
                    expected_unit_digest = hashlib.sha256()
                cursor = segment_stop
        destination[selection] = values
        if index == 0 or index == len(starts) - 1:
            samples.append(
                {
                    "start_row": int(start),
                    "stop_row": int(stop),
                    "sha256": hashlib.sha256(unit_bytes).hexdigest(),
                }
            )
        writes += 1
    logical_sha256 = logical_digest.hexdigest()
    if source_validation_record is not None:
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
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "digest_algorithm": SUBJECT_MASK_CORE_ARRAY_DIGEST_ALGORITHM,
        "sha256": logical_sha256,
        "physical_write_count": writes,
        "bounded_reopen_samples": samples,
    }


def _streamed_array_document(
    write_receipts: Mapping[str, Mapping[str, object]], paths: tuple[str, ...]
) -> dict[str, object]:
    return {
        path: {
            "shape": list(write_receipts[path]["shape"]),
            "dtype": str(write_receipts[path]["dtype"]),
            "digest_algorithm": str(write_receipts[path]["digest_algorithm"]),
            "sha256": str(write_receipts[path]["sha256"]),
        }
        for path in paths
    }


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
) -> SubjectMaskCorePublication:
    """Validate and rematerialize one complete raw or refined core."""

    validation_mode = SubjectMaskCoreValidationMode(validation_mode)
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

    started = time.perf_counter()
    phases: dict[str, float] = {}
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
        }
    )
    destination_arrays: dict[str, Any] = {}
    write_counts: dict[str, int] = {}
    write_receipts: dict[str, dict[str, object]] = {}
    source_array_validation: Mapping[str, Any] = {}
    if validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_STREAMING:
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
            )
        except Exception as exc:
            run.attrs["status"] = "failed"
            run.attrs["validation_failure"] = "physical_write_or_receipt_mismatch"
            mark_run_failed(
                run,
                run_name=str(run_id),
                error=f"Subject-mask physical publication failed: {exc}",
            )
            raise
        write_receipts[path] = write_receipt
        write_counts[path] = int(write_receipt["physical_write_count"])
        destination_arrays[path] = destination_array
    phases["physical_unit_publication"] = time.perf_counter() - phase
    streamed_array_document: dict[str, object] | None = None
    if validation_mode is SubjectMaskCoreValidationMode.PRODUCTION_STREAMING:
        assert source_validation_receipt is not None
        streamed_array_document = _streamed_array_document(write_receipts, paths)
    run.attrs["physical_write_counts"] = write_counts
    run.attrs["status"] = "complete"
    mark_run_complete(run, run_name=str(run_id))

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
        "schema_version": 1,
        "kind": kind,
        "dimensions": dimensions.as_manifest(),
        "components": components.as_manifest(),
        "arrays": array_document,
    }
    source_validation_binding: dict[str, object] | None = None
    source_validation_receipt_bytes: bytes | None = None
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
            "relative_path": SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR,
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
                else "computed_during_required_publication_read_v1"
            ),
            "reopen_validation": (
                "full_schema_and_full_logical_hash_v1"
                if validation_mode is SubjectMaskCoreValidationMode.REFERENCE_FULL
                else "metadata_plus_first_last_physical_row_band_samples_v1"
            ),
            "bounded_reopen_samples": {
                path: write_receipts[path]["bounded_reopen_samples"] for path in paths
            },
            "parallel_write_policy": (
                "single_writer_v1_future_workers_require_disjoint_whole_shards"
            ),
            "derived_metric_canonicalization": canonicalization,
        },
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    manifest = {
        "schema_id": SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(manifest_payload),
        "payload": manifest_payload,
    }
    canonical_json_bytes(manifest)
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
        (output_path / SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR).write_bytes(
            source_validation_receipt_bytes
        )
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
            output_path / SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR
        )
        if persisted_receipt != source_validation_receipt:
            raise RuntimeError("Persisted source-validation receipt differs.")
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
    "SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_ID",
    "SUBJECT_MASK_CORE_PUBLICATION_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE",
    "SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_ID",
    "SUBJECT_MASK_CORE_RUN_MANIFEST_SCHEMA_VERSION",
    "SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR",
    "SubjectMaskCorePublication",
    "SubjectMaskCoreValidationMode",
    "publish_selector_ineligible_subject_mask_core_snapshot",
]
