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
from dataclasses import dataclass, field
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
SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_ID = "palette.subject_mask.source_run_manifest"
SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_VERSION = 1


@dataclass
class SubjectMaskArrayUnitAccumulator:
    """Incrementally hash one logical array without rereading its output store.

    Appends must arrive in exact first-axis order.  Incoming blocks may use any
    row count; the accumulator splits them at the frozen ``unit_rows``
    boundaries so execution batch size does not become scientific identity.
    """

    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    unit_rows: int
    _cursor: int = 0
    _unit_start: int = 0
    _unit_rows_written: int = 0
    _unit_bytes: int = 0
    _digest: Any = field(default_factory=hashlib.sha256)
    _units: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.shape = tuple(int(value) for value in self.shape)
        self.dtype = np.dtype(self.dtype)
        if not self.shape or self.shape[0] <= 0:
            raise ValueError("Incremental array hashing requires a nonempty row axis.")
        if type(self.unit_rows) is not int or self.unit_rows <= 0:
            raise ValueError("unit_rows must be one positive exact integer.")

    @property
    def cursor(self) -> int:
        return int(self._cursor)

    def append(self, start_row: int, values: Any) -> None:
        start = int(start_row)
        block = np.asarray(values)
        if start != self._cursor:
            raise ValueError(
                f"Incremental array rows are not contiguous: expected {self._cursor}, "
                f"got {start}."
            )
        if block.dtype != self.dtype:
            raise ValueError(
                f"Incremental array dtype differs: expected {self.dtype}, got {block.dtype}."
            )
        if block.ndim != len(self.shape) or tuple(block.shape[1:]) != self.shape[1:]:
            raise ValueError(
                "Incremental array trailing shape differs: "
                f"expected {self.shape[1:]}, got {tuple(block.shape[1:])}."
            )
        stop = start + int(block.shape[0])
        if stop > self.shape[0]:
            raise ValueError("Incremental array append exceeds its declared shape.")
        offset = 0
        while offset < block.shape[0]:
            capacity = min(self.unit_rows, self.shape[0] - self._unit_start)
            take = min(capacity - self._unit_rows_written, block.shape[0] - offset)
            part = np.ascontiguousarray(block[offset : offset + take])
            self._digest.update(part.view(np.uint8))
            self._unit_rows_written += int(take)
            self._unit_bytes += int(part.nbytes)
            self._cursor += int(take)
            offset += int(take)
            if self._unit_rows_written == capacity:
                self._units.append(
                    {
                        "start_row": int(self._unit_start),
                        "stop_row": int(self._cursor),
                        "decoded_bytes": int(self._unit_bytes),
                        "sha256": self._digest.hexdigest(),
                    }
                )
                self._unit_start = int(self._cursor)
                self._unit_rows_written = 0
                self._unit_bytes = 0
                self._digest = hashlib.sha256()

    def as_document(self) -> dict[str, object]:
        if self._cursor != self.shape[0] or self._unit_rows_written != 0:
            raise ValueError(
                f"Incremental array hashing covered {self._cursor} of {self.shape[0]} rows."
            )
        units = [dict(unit) for unit in self._units]
        return {
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "digest_algorithm": SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
            "unit_count": len(units),
            "units_digest": canonical_json_sha256(units),
            "units": units,
        }


def subject_mask_semantic_units_from_array_document(
    array_document: Mapping[str, Mapping[str, Any]],
    *,
    n_rois: int,
    paths: Sequence[str],
) -> tuple[dict[str, object], ...]:
    """Derive exact semantic coverage evidence from aligned ROI-row hashes."""

    resolved_paths = tuple(str(path) for path in paths)
    if not resolved_paths or len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Semantic evidence paths must be unique and nonempty.")
    records = [array_document.get(path) for path in resolved_paths]
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("Semantic evidence array document is incomplete.")
    first_units = records[0].get("units")  # type: ignore[union-attr]
    if not isinstance(first_units, list) or not first_units:
        raise ValueError("Semantic evidence requires row-unit array hashes.")
    result: list[dict[str, object]] = []
    for index, first in enumerate(first_units):
        if not isinstance(first, Mapping):
            raise ValueError("Semantic evidence row unit is invalid.")
        start = first.get("start_row")
        stop = first.get("stop_row")
        evidence_arrays: dict[str, object] = {}
        for path, record in zip(resolved_paths, records, strict=True):
            assert isinstance(record, Mapping)
            units = record.get("units")
            if not isinstance(units, list) or len(units) != len(first_units):
                raise ValueError("Semantic evidence unit grids differ.")
            unit = units[index]
            if (
                not isinstance(unit, Mapping)
                or unit.get("start_row") != start
                or unit.get("stop_row") != stop
            ):
                raise ValueError("Semantic evidence row boundaries differ.")
            evidence_arrays[path] = {
                "decoded_bytes": unit.get("decoded_bytes"),
                "sha256": unit.get("sha256"),
            }
        result.append(
            {
                "start_row": start,
                "stop_row": stop,
                "result": "valid",
                "validator_schema_id": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID,
                "validator_schema_version": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION,
                "evidence_digest": canonical_json_sha256(
                    {
                        "start_row": start,
                        "stop_row": stop,
                        "arrays": evidence_arrays,
                    }
                ),
            }
        )
    _canonical_semantic_units(result, n_rois=int(n_rois))
    return tuple(result)


def build_subject_mask_source_run_manifest(
    *,
    kind: str,
    run_path: str,
    schema: RawSubjectMaskSchema | RefinedSubjectMaskCoreSchema,
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    threshold: float | None,
    producer_identity_schema_id: str,
    producer_identity_digest: str,
    attempt_payload_digest: str | None,
    array_document: Mapping[str, Any],
) -> dict[str, object]:
    """Build the immutable identity document consumed by source receipts."""

    resolved_path = str(run_path).strip().strip("/")
    producer_schema = str(producer_identity_schema_id).strip()
    if not resolved_path or not producer_schema:
        raise ValueError("Source run path and producer identity schema are required.")
    producer_digest = _require_sha256(
        producer_identity_digest, name="producer identity digest"
    )
    attempt_digest = (
        None
        if attempt_payload_digest is None
        else _require_sha256(attempt_payload_digest, name="attempt payload digest")
    )
    arrays = _strict_copy(array_document, name="source run array document")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("Source run array document cannot be empty.")
    payload = {
        "kind": str(kind),
        "run_path": resolved_path,
        "logical_schema": {
            "schema_id": schema.schema_id,
            "schema_version": schema.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "components": components.as_manifest(),
        "threshold": None if threshold is None else float(threshold),
        "producer_identity": {
            "schema_id": producer_schema,
            "digest": producer_digest,
        },
        "attempt_payload_digest": attempt_digest,
        "array_document_digest": canonical_json_sha256(arrays),
    }
    envelope = {
        "schema_id": SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    validate_subject_mask_source_run_manifest(envelope)
    return envelope


def validate_subject_mask_source_run_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    canonical = _strict_copy(manifest, name="subject-mask source run manifest")
    if not isinstance(canonical, dict) or set(canonical) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        raise ValueError("Subject-mask source run manifest fields are not exact.")
    payload = canonical.get("payload")
    if (
        canonical.get("schema_id") != SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_ID
        or canonical.get("schema_version")
        != SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_VERSION
        or canonical.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, dict)
        or canonical.get("payload_digest") != canonical_json_sha256(payload)
    ):
        raise ValueError("Subject-mask source run manifest is unsupported or stale.")
    if set(payload) != {
        "kind",
        "run_path",
        "logical_schema",
        "dimensions",
        "components",
        "threshold",
        "producer_identity",
        "attempt_payload_digest",
        "array_document_digest",
    }:
        raise ValueError(
            "Subject-mask source run manifest payload fields are not exact."
        )
    if not str(payload.get("run_path") or "").strip().strip("/"):
        raise ValueError("Subject-mask source run path is absent.")
    logical = payload.get("logical_schema")
    if not isinstance(logical, dict) or set(logical) != {
        "schema_id",
        "schema_version",
    }:
        raise ValueError("Subject-mask source logical schema is invalid.")
    producer = payload.get("producer_identity")
    if (
        not isinstance(producer, dict)
        or set(producer) != {"schema_id", "digest"}
        or not str(producer.get("schema_id") or "").strip()
    ):
        raise ValueError("Subject-mask producer identity is invalid.")
    _require_sha256(producer.get("digest"), name="producer identity digest")
    attempt = payload.get("attempt_payload_digest")
    if attempt is not None:
        _require_sha256(attempt, name="attempt payload digest")
    _require_sha256(payload.get("array_document_digest"), name="array document digest")
    return canonical


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
    "SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_ID",
    "SUBJECT_MASK_SOURCE_RUN_MANIFEST_SCHEMA_VERSION",
    "SubjectMaskArrayUnitAccumulator",
    "build_reference_subject_mask_validation_receipt",
    "build_subject_mask_source_run_manifest",
    "build_subject_mask_source_validation_receipt",
    "streaming_array_sha256",
    "subject_mask_array_document",
    "subject_mask_array_unit_document",
    "subject_mask_semantic_units_from_array_document",
    "validate_subject_mask_source_run_manifest",
    "validate_subject_mask_source_validation_receipt",
]
