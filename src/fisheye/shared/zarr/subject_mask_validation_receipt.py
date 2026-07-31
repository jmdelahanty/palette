"""Incremental validation receipts for subject-mask publication sources.

The receipt separates *when* scientific validation runs from *when* an
immutable physical layout is published.  Producers and finalizers may validate
bounded, non-overlapping row units while those values are already resident,
then seal one receipt proving complete logical coverage.  The publisher checks
the receipt against the exact bytes it streams into the destination; it does
not need to repeat the scientific computation over the full source surface.

``build_reference_subject_mask_validation_receipt`` is deliberately a
small-fixture/reference helper.  It performs the exhaustive schema validation
and full logical hashing that production workers are expected to accumulate
incrementally.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.subject_mask_schema import (
    RawSubjectMaskSchema,
    RefinedSubjectMaskCoreSchema,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)

SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_ID = (
    "palette.subject_mask.source_validation_receipt"
)
SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_VERSION = 1
SUBJECT_MASK_SOURCE_VALIDATION_MODE = "incremental_complete_row_coverage_v1"
SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID = "palette.subject_mask.source_semantics"
SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION = 1
SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM = "sha256_c_contiguous_row_units_v1"


def _strict_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict canonical JSON: {exc}.") from exc


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return str(value)


def _shape_dtype(value: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    return tuple(int(item) for item in value.shape), np.dtype(value.dtype)


def streaming_array_sha256(
    value: Any, *, row_bytes_budget: int = 64 * 1024 * 1024
) -> str:
    """Hash exact logical C-order bytes using bounded first-axis reads."""

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


def subject_mask_array_document(
    arrays: Mapping[str, Any], paths: Sequence[str]
) -> dict[str, dict[str, object]]:
    return {
        str(path): {
            "shape": list(_shape_dtype(arrays[str(path)])[0]),
            "dtype": str(_shape_dtype(arrays[str(path)])[1]),
            "digest_algorithm": SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM,
            "sha256": streaming_array_sha256(arrays[str(path)]),
        }
        for path in paths
    }


def subject_mask_array_unit_document(
    arrays: Mapping[str, Any],
    paths: Sequence[str],
    *,
    unit_rows: int,
) -> dict[str, dict[str, object]]:
    """Build ordered row-unit hashes for bounded fixtures.

    Parallel producers should emit equivalent records while their owned values
    are resident rather than call this helper over a completed full surface.
    """

    if type(unit_rows) is not int or unit_rows <= 0:
        raise ValueError("unit_rows must be one positive exact integer.")
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        shape, dtype = _shape_dtype(arrays[str(path)])
        trailing = (slice(None),) * (len(shape) - 1)
        units: list[dict[str, object]] = []
        for start in range(0, shape[0], unit_rows):
            stop = min(shape[0], start + unit_rows)
            values = np.ascontiguousarray(
                np.asarray(arrays[str(path)][(slice(start, stop), *trailing)])
            )
            units.append(
                {
                    "start_row": int(start),
                    "stop_row": int(stop),
                    "decoded_bytes": int(values.nbytes),
                    "sha256": hashlib.sha256(values.view(np.uint8)).hexdigest(),
                }
            )
        result[str(path)] = {
            "shape": list(shape),
            "dtype": str(dtype),
            "digest_algorithm": SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
            "unit_count": len(units),
            "units_digest": canonical_json_sha256(units),
            "units": units,
        }
    return result


def _receipt_paths(
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    arrays: Mapping[str, Any],
) -> tuple[str, ...]:
    declared = tuple(schema.binding_paths)
    required = {binding.path for binding in schema.bindings if binding.required}
    observed = set(arrays)
    if not required <= observed or not observed <= set(declared):
        raise ValueError(
            "Validation receipt arrays must contain every required schema path "
            "and no undeclared paths."
        )
    return tuple(path for path in declared if path in observed)


def _canonical_semantic_units(
    units: Sequence[Mapping[str, Any]], *, n_rois: int
) -> list[dict[str, object]]:
    canonical = _strict_copy(list(units), name="semantic validation units")
    if not isinstance(canonical, list) or not canonical:
        raise ValueError("Semantic validation requires at least one row unit.")
    expected_fields = {
        "start_row",
        "stop_row",
        "result",
        "validator_schema_id",
        "validator_schema_version",
        "evidence_digest",
    }
    cursor = 0
    normalized: list[dict[str, object]] = []
    for unit in canonical:
        if not isinstance(unit, dict) or set(unit) != expected_fields:
            raise ValueError("Semantic validation unit fields are not exact.")
        start = unit.get("start_row")
        stop = unit.get("stop_row")
        if (
            type(start) is not int
            or type(stop) is not int
            or start != cursor
            or not (0 <= start < stop <= n_rois)
            or unit.get("result") != "valid"
            or unit.get("validator_schema_id")
            != SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID
            or unit.get("validator_schema_version")
            != SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION
        ):
            raise ValueError(
                "Semantic validation units must be valid, ordered, contiguous, "
                "and use the frozen subject-mask validator."
            )
        _require_sha256(unit.get("evidence_digest"), name="unit evidence digest")
        cursor = int(stop)
        normalized.append(dict(unit))
    if cursor != n_rois:
        raise ValueError("Semantic validation units do not cover every ROI row.")
    return normalized


def _canonical_array_document(
    value: Mapping[str, Any],
    *,
    arrays: Mapping[str, Any],
    expected_paths: Sequence[str],
) -> dict[str, dict[str, object]]:
    canonical = _strict_copy(value, name="subject-mask array document")
    if not isinstance(canonical, dict) or set(canonical) != set(expected_paths):
        raise ValueError("Validation receipt array inventory is not exact.")
    result: dict[str, dict[str, object]] = {}
    for path in expected_paths:
        record = canonical.get(path)
        if not isinstance(record, dict):
            raise ValueError(
                f"Validation receipt array record for {path!r} is invalid."
            )
        shape, dtype = _shape_dtype(arrays[path])
        if record.get("shape") != list(shape) or record.get("dtype") != str(dtype):
            raise ValueError(
                f"Validation receipt shape or dtype changed for array {path!r}."
            )
        algorithm = record.get("digest_algorithm")
        if algorithm == SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM:
            if set(record) != {"shape", "dtype", "digest_algorithm", "sha256"}:
                raise ValueError(
                    f"Whole-array validation record fields changed for {path!r}."
                )
            _require_sha256(record.get("sha256"), name=f"{path} logical digest")
        elif algorithm == SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM:
            expected_unit_fields = {
                "shape",
                "dtype",
                "digest_algorithm",
                "unit_count",
                "units_digest",
                "units",
            }
            if set(record) != expected_unit_fields:
                raise ValueError(
                    f"Row-unit validation record fields changed for {path!r}."
                )
            units = record.get("units")
            if not isinstance(units, list) or not units:
                raise ValueError(f"Row-unit validation is absent for {path!r}.")
            cursor = 0
            row_bytes = int(dtype.itemsize) * int(np.prod(shape[1:]))
            for unit in units:
                if not isinstance(unit, dict) or set(unit) != {
                    "start_row",
                    "stop_row",
                    "decoded_bytes",
                    "sha256",
                }:
                    raise ValueError(f"Row-unit fields are invalid for {path!r}.")
                start = unit.get("start_row")
                stop = unit.get("stop_row")
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or start != cursor
                    or not (0 <= start < stop <= shape[0])
                    or unit.get("decoded_bytes") != (stop - start) * row_bytes
                ):
                    raise ValueError(
                        f"Row-unit coverage or byte count is invalid for {path!r}."
                    )
                _require_sha256(unit.get("sha256"), name=f"{path} row-unit digest")
                cursor = int(stop)
            if (
                cursor != shape[0]
                or record.get("unit_count") != len(units)
                or record.get("units_digest") != canonical_json_sha256(units)
            ):
                raise ValueError(
                    f"Row-unit validation does not cover all rows of {path!r}."
                )
        else:
            raise ValueError(f"Array digest algorithm is unsupported for {path!r}.")
        result[path] = dict(record)
    return result


def build_subject_mask_source_validation_receipt(
    *,
    kind: str,
    source_run_path: str,
    source_manifest: Mapping[str, Any],
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    arrays: Mapping[str, Any],
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    threshold: float | None,
    array_document: Mapping[str, Any],
    semantic_units: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Seal already-computed full-coverage scientific validation evidence."""

    paths = _receipt_paths(schema, arrays)
    units = _canonical_semantic_units(semantic_units, n_rois=dimensions.n_rois)
    arrays_doc = _canonical_array_document(
        array_document, arrays=arrays, expected_paths=paths
    )
    run_path = str(source_run_path).strip().strip("/")
    if not run_path:
        raise ValueError("Validation receipt source_run_path cannot be empty.")
    if threshold is not None and (
        not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("Validation receipt threshold must be finite within [0,1].")
    payload = {
        "validation_mode": SUBJECT_MASK_SOURCE_VALIDATION_MODE,
        "result": "valid",
        "kind": str(kind),
        "source": {
            "run_path": run_path,
            "manifest_digest": canonical_json_sha256(source_manifest),
        },
        "validator": {
            "schema_id": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION,
        },
        "logical_schema": {
            "schema_id": schema.schema_id,
            "schema_version": schema.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "components": components.as_manifest(),
        "threshold": None if threshold is None else float(threshold),
        "arrays": arrays_doc,
        "semantic_coverage": {
            "axis": "roi",
            "complete_nonoverlapping": True,
            "start_row": 0,
            "stop_row": dimensions.n_rois,
            "unit_count": len(units),
            "units_digest": canonical_json_sha256(units),
            "units": units,
        },
    }
    return {
        "schema_id": SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def validate_subject_mask_source_validation_receipt(
    document: Mapping[str, Any],
    *,
    kind: str,
    source_run_path: str,
    source_manifest: Mapping[str, Any],
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    arrays: Mapping[str, Any],
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    threshold: float | None,
) -> dict[str, object]:
    """Fail closed unless a receipt binds the exact source publication input."""

    canonical = _strict_copy(document, name="subject-mask validation receipt")
    expected_fields = {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }
    if not isinstance(canonical, dict) or set(canonical) != expected_fields:
        raise ValueError("Subject-mask validation receipt fields are not exact.")
    payload = canonical.get("payload")
    digest = canonical.get("payload_digest")
    if (
        canonical.get("schema_id") != SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_ID
        or canonical.get("schema_version")
        != SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_VERSION
        or canonical.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, dict)
        or not _is_sha256(digest)
        or digest != canonical_json_sha256(payload)
    ):
        raise ValueError("Subject-mask validation receipt is unsupported or stale.")
    payload_fields = {
        "validation_mode",
        "result",
        "kind",
        "source",
        "validator",
        "logical_schema",
        "dimensions",
        "components",
        "threshold",
        "arrays",
        "semantic_coverage",
    }
    if set(payload) != payload_fields:
        raise ValueError(
            "Subject-mask validation receipt payload fields are not exact."
        )
    expected_threshold = None if threshold is None else float(threshold)
    if (
        payload.get("validation_mode") != SUBJECT_MASK_SOURCE_VALIDATION_MODE
        or payload.get("result") != "valid"
        or payload.get("kind") != str(kind)
        or payload.get("source")
        != {
            "run_path": str(source_run_path).strip().strip("/"),
            "manifest_digest": canonical_json_sha256(source_manifest),
        }
        or payload.get("validator")
        != {
            "schema_id": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION,
        }
        or payload.get("logical_schema")
        != {"schema_id": schema.schema_id, "schema_version": schema.schema_version}
        or payload.get("dimensions") != dimensions.as_manifest()
        or payload.get("components") != components.as_manifest()
        or payload.get("threshold") != expected_threshold
    ):
        raise ValueError("Subject-mask validation receipt binding changed.")
    paths = _receipt_paths(schema, arrays)
    _canonical_array_document(
        payload.get("arrays"), arrays=arrays, expected_paths=paths
    )
    coverage = payload.get("semantic_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Subject-mask semantic coverage is absent.")
    coverage_fields = {
        "axis",
        "complete_nonoverlapping",
        "start_row",
        "stop_row",
        "unit_count",
        "units_digest",
        "units",
    }
    if set(coverage) != coverage_fields:
        raise ValueError("Subject-mask semantic coverage fields are not exact.")
    units = coverage.get("units")
    if not isinstance(units, list):
        raise ValueError("Subject-mask semantic validation units are absent.")
    normalized_units = _canonical_semantic_units(units, n_rois=dimensions.n_rois)
    if (
        coverage.get("axis") != "roi"
        or coverage.get("complete_nonoverlapping") is not True
        or coverage.get("start_row") != 0
        or coverage.get("stop_row") != dimensions.n_rois
        or coverage.get("unit_count") != len(normalized_units)
        or coverage.get("units_digest") != canonical_json_sha256(normalized_units)
    ):
        raise ValueError("Subject-mask semantic coverage is incomplete or stale.")
    return canonical


def build_reference_subject_mask_validation_receipt(
    *,
    kind: str,
    source_run_path: str,
    source_manifest: Mapping[str, Any],
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    arrays: Mapping[str, Any],
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    threshold: float | None,
    source_crop_arrays: Mapping[str, Any],
) -> dict[str, object]:
    """Exhaustively build a receipt for canaries and mode-equivalence tests."""

    if isinstance(schema, RawSubjectMaskSchema):
        if threshold is None:
            raise ValueError("Raw reference validation requires a threshold.")
        schema.require(
            arrays,
            dimensions=dimensions,
            components=components,
            threshold=float(threshold),
            source_crop_arrays=source_crop_arrays,
        )
    else:
        if threshold is not None:
            raise ValueError(
                "Refined reference validation does not accept a threshold."
            )
        schema.require(
            arrays,
            dimensions=dimensions,
            components=components,
            source_crop_arrays=source_crop_arrays,
        )
    paths = _receipt_paths(schema, arrays)
    array_document = subject_mask_array_document(arrays, paths)
    evidence = {
        "kind": str(kind),
        "dimensions": dimensions.as_manifest(),
        "components": components.as_manifest(),
        "threshold": None if threshold is None else float(threshold),
        "arrays": array_document,
    }
    unit = {
        "start_row": 0,
        "stop_row": dimensions.n_rois,
        "result": "valid",
        "validator_schema_id": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID,
        "validator_schema_version": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION,
        "evidence_digest": canonical_json_sha256(evidence),
    }
    return build_subject_mask_source_validation_receipt(
        kind=kind,
        source_run_path=source_run_path,
        source_manifest=source_manifest,
        schema=schema,
        arrays=arrays,
        dimensions=dimensions,
        components=components,
        threshold=threshold,
        array_document=array_document,
        semantic_units=(unit,),
    )


__all__ = [
    "SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM",
    "SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM",
    "SUBJECT_MASK_SOURCE_VALIDATION_MODE",
    "SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_ID",
    "SUBJECT_MASK_SOURCE_VALIDATION_RECEIPT_SCHEMA_VERSION",
    "SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID",
    "SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION",
    "build_reference_subject_mask_validation_receipt",
    "build_subject_mask_source_validation_receipt",
    "streaming_array_sha256",
    "subject_mask_array_document",
    "subject_mask_array_unit_document",
    "validate_subject_mask_source_validation_receipt",
]
