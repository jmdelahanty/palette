"""Exact persisted manifest and source evidence for canonical detections.

The first consumer of this contract is the selector-ineligible shadow path.
The implementation is intentionally strict enough that a legacy detection run
cannot be described as canonical merely by supplying two plausible digests.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array, sha256_file
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
)
from fisheye.shared.zarr.detection_storage import (
    CanonicalDetectionStoragePlanSet,
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest


CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID = "palette.canonical_detection.run_manifest"
CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_VERSION = 1
CANONICAL_DETECTION_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
CANONICAL_DETECTION_RUN_MANIFEST_PERSISTED_PATH = (
    "detect_runs/<run>/zarr.json.attributes.run_manifest"
)
CANONICAL_DETECTION_LOGICAL_CONTENT_SCHEMA_ID = (
    "palette.canonical_detection.logical_content"
)
CANONICAL_DETECTION_LOGICAL_CONTENT_SCHEMA_VERSION = 1
CANONICAL_DETECTION_LEGACY_SOURCE_SCHEMA_ID = (
    "palette.canonical_detection.legacy_source_evidence"
)
CANONICAL_DETECTION_LEGACY_SOURCE_SCHEMA_VERSION = 1
CANONICAL_DETECTION_METADATA_DECLARATIONS_SCHEMA_ID = (
    "palette.canonical_detection.metadata_declarations"
)
CANONICAL_DETECTION_METADATA_DECLARATIONS_SCHEMA_VERSION = 1
CANONICAL_DETECTION_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
CANONICAL_DETECTION_METADATA_DIGEST_SCOPE = (
    "normalized_group_and_array_declarations_excluding_attributes"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_SOURCE_ARRAY_PATHS = (
    "frame_indices",
    "bbox_norm_coords",
    "scores",
    "class_ids",
)


def _require_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _require_text(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return normalized


def _require_run_id(value: object, *, name: str = "run_id") -> str:
    normalized = _require_text(value, name=name)
    if "/" in normalized:
        raise ValueError(f"{name} must be one path-safe group name.")
    return normalized


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def canonical_detection_logical_content_document(
    arrays: Mapping[str, Any],
    *,
    dimensions: CanonicalDetectionDimensions,
) -> dict[str, object]:
    """Describe every decoded canonical array with an exact content digest."""

    CANONICAL_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)
    expected = set(CANONICAL_DETECTION_SCHEMA_V1.binding_paths)
    if set(arrays) != expected:
        raise ValueError("Canonical logical content requires the exact array set.")
    declarations: dict[str, object] = {}
    for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
        values = _array_values(arrays[path])
        declarations[path] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "digest_algorithm": CANONICAL_DETECTION_ARRAY_DIGEST_ALGORITHM,
            "sha256": sha256_array(values),
        }
    document: dict[str, object] = {
        "schema_id": CANONICAL_DETECTION_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": CANONICAL_DETECTION_LOGICAL_CONTENT_SCHEMA_VERSION,
        "logical_schema": {
            "id": CANONICAL_DETECTION_SCHEMA_V1.schema_id,
            "version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "arrays": declarations,
    }
    canonical_json_bytes(document)
    return document


def canonical_detection_logical_content_digest(
    arrays: Mapping[str, Any],
    *,
    dimensions: CanonicalDetectionDimensions,
) -> str:
    return canonical_json_sha256(
        canonical_detection_logical_content_document(
            arrays,
            dimensions=dimensions,
        )
    )


def build_legacy_detection_source_evidence(
    source_group: Any,
    *,
    source_group_path: Path,
    source_run_id: str,
    recording_identity: str,
) -> dict[str, object]:
    """Bind a read-only legacy source before canonical conversion."""

    resolved_path = source_group_path.expanduser().resolve()
    metadata_path = resolved_path / "zarr.json"
    if not metadata_path.is_file():
        raise ValueError(f"Legacy source lacks direct metadata: {resolved_path}")
    arrays: dict[str, object] = {}
    for path in _LEGACY_SOURCE_ARRAY_PATHS:
        try:
            values = _array_values(source_group[path])
        except KeyError as exc:
            raise ValueError(f"Legacy source lacks {path!r}.") from exc
        arrays[path] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "digest_algorithm": CANONICAL_DETECTION_ARRAY_DIGEST_ALGORITHM,
            "sha256": sha256_array(values),
        }
    evidence: dict[str, object] = {
        "schema_id": CANONICAL_DETECTION_LEGACY_SOURCE_SCHEMA_ID,
        "schema_version": CANONICAL_DETECTION_LEGACY_SOURCE_SCHEMA_VERSION,
        "source_open_mode": "read_only_direct_metadata",
        "source_group_path": str(resolved_path),
        "source_run_id": _require_run_id(source_run_id, name="source_run_id"),
        "recording_identity": _require_text(
            recording_identity,
            name="recording_identity",
        ),
        "source_group_metadata_sha256": sha256_file(metadata_path),
        "source_arrays_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "source_arrays_digest": canonical_json_sha256(arrays),
        "source_arrays": arrays,
        "conversion": "legacy_sparse_detection_to_canonical_detection_v1",
    }
    validate_legacy_detection_source_evidence(evidence)
    return evidence


def validate_legacy_detection_source_evidence(
    evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_fields = {
        "schema_id",
        "schema_version",
        "source_open_mode",
        "source_group_path",
        "source_run_id",
        "recording_identity",
        "source_group_metadata_sha256",
        "source_arrays_digest_algorithm",
        "source_arrays_digest",
        "source_arrays",
        "conversion",
    }
    if set(evidence) != expected_fields:
        errors.append("legacy source evidence has an unexpected field set")
    if evidence.get("schema_id") != CANONICAL_DETECTION_LEGACY_SOURCE_SCHEMA_ID:
        errors.append("legacy source evidence schema_id mismatch")
    if (
        evidence.get("schema_version")
        != CANONICAL_DETECTION_LEGACY_SOURCE_SCHEMA_VERSION
    ):
        errors.append("legacy source evidence schema_version mismatch")
    if evidence.get("source_open_mode") != "read_only_direct_metadata":
        errors.append("legacy source evidence open mode mismatch")
    if evidence.get("conversion") != (
        "legacy_sparse_detection_to_canonical_detection_v1"
    ):
        errors.append("legacy source evidence conversion mismatch")
    for name in ("source_group_path", "source_run_id", "recording_identity"):
        try:
            _require_text(evidence.get(name), name=name)
        except ValueError as exc:
            errors.append(str(exc))
    try:
        _require_run_id(evidence.get("source_run_id"), name="source_run_id")
        _require_sha256(
            evidence.get("source_group_metadata_sha256"),
            name="source_group_metadata_sha256",
        )
    except ValueError as exc:
        errors.append(str(exc))
    arrays = evidence.get("source_arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(
        _LEGACY_SOURCE_ARRAY_PATHS
    ):
        errors.append("legacy source evidence array declarations are not exact")
    else:
        for path in _LEGACY_SOURCE_ARRAY_PATHS:
            item = arrays[path]
            if not isinstance(item, Mapping) or set(item) != {
                "shape",
                "dtype",
                "digest_algorithm",
                "sha256",
            }:
                errors.append(f"legacy source declaration is invalid at {path!r}")
                continue
            if item.get("digest_algorithm") != (
                CANONICAL_DETECTION_ARRAY_DIGEST_ALGORITHM
            ):
                errors.append(f"legacy source digest algorithm mismatch at {path!r}")
            if not isinstance(item.get("shape"), list) or not all(
                type(value) is int and value >= 0 for value in item.get("shape", [])
            ):
                errors.append(f"legacy source shape is invalid at {path!r}")
            if not str(item.get("dtype") or "").strip():
                errors.append(f"legacy source dtype is invalid at {path!r}")
            try:
                _require_sha256(item.get("sha256"), name=f"{path} sha256")
            except ValueError as exc:
                errors.append(str(exc))
        if evidence.get("source_arrays_digest_algorithm") != (
            CANONICAL_JSON_DIGEST_ALGORITHM
        ):
            errors.append("legacy source array digest algorithm mismatch")
        if evidence.get("source_arrays_digest") != canonical_json_sha256(arrays):
            errors.append("legacy source array digest mismatch")
    try:
        canonical_json_bytes(evidence)
    except (TypeError, ValueError) as exc:
        errors.append(f"legacy source evidence is not strict JSON: {exc}")
    return tuple(dict.fromkeys(errors))


def normalize_canonical_detection_metadata_declarations(
    direct_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_metadata_by_path: Mapping[str, Mapping[str, Any]],
    dimensions: CanonicalDetectionDimensions,
) -> dict[str, object]:
    """Normalize the exact run-relative Zarr-v3 declaration tree."""

    array_paths = CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    group_paths = ("", "instances")
    expected_paths = {*group_paths, *array_paths}
    if set(direct_metadata_by_path) != expected_paths:
        raise ValueError("Canonical direct metadata declaration paths are not exact.")
    if set(consolidated_metadata_by_path) != expected_paths:
        raise ValueError(
            "Canonical consolidated metadata declaration paths are not exact."
        )
    direct: dict[str, Mapping[str, Any]] = {}
    for path in sorted(expected_paths):
        declaration = direct_metadata_by_path[path]
        candidate = consolidated_metadata_by_path[path]
        if not isinstance(declaration, Mapping) or not isinstance(candidate, Mapping):
            raise TypeError(f"Zarr metadata declaration {path!r} must be an object.")
        canonical_json_bytes(declaration)
        canonical_json_bytes(candidate)
        if metadata_without_empty_group_consolidation(
            declaration,
            path=path,
        ) != metadata_without_empty_group_consolidation(
            candidate,
            path=path,
        ):
            raise ValueError(f"Direct and consolidated metadata differ at {path!r}.")
        direct[path] = declaration

    normalized: dict[str, dict[str, Any]] = {}
    for path in sorted(expected_paths):
        declaration = dict(direct[path])
        if declaration.get("zarr_format") != 3:
            raise ValueError(f"Zarr declaration {path!r} must use format 3.")
        if not isinstance(declaration.get("attributes"), Mapping):
            raise ValueError(f"Zarr declaration {path!r} requires attributes.")
        if path in group_paths:
            required = {"zarr_format", "node_type", "attributes"}
            optional = {"consolidated_metadata"}
            if not required.issubset(declaration) or not set(declaration).issubset(
                required | optional
            ):
                raise ValueError(f"Zarr group {path!r} has unexpected fields.")
            if declaration.get("node_type") != "group":
                raise ValueError(f"Zarr declaration {path!r} must be a group.")
            if declaration.get("consolidated_metadata") is not None and not isinstance(
                declaration["consolidated_metadata"], Mapping
            ):
                raise ValueError(f"Zarr group {path!r} has invalid consolidation.")
        else:
            required = {
                "zarr_format",
                "node_type",
                "shape",
                "data_type",
                "chunk_grid",
                "chunk_key_encoding",
                "fill_value",
                "codecs",
                "attributes",
                "storage_transformers",
            }
            optional = {"dimension_names"}
            if not required.issubset(declaration) or not set(declaration).issubset(
                required | optional
            ):
                raise ValueError(f"Zarr array {path!r} has unexpected fields.")
            if declaration.get("node_type") != "array":
                raise ValueError(f"Zarr declaration {path!r} must be an array.")
        declaration.pop("attributes")
        declaration.pop("consolidated_metadata", None)
        normalized[path] = declaration

    document: dict[str, object] = {
        "schema_id": CANONICAL_DETECTION_METADATA_DECLARATIONS_SCHEMA_ID,
        "schema_version": CANONICAL_DETECTION_METADATA_DECLARATIONS_SCHEMA_VERSION,
        "path_basis": "relative_to_detect_run_group_empty_string_is_root",
        "included_nodes": "exact_run_root_groups_and_schema_arrays",
        "excluded_fields": ["attributes", "consolidated_metadata"],
        "dimensions": dimensions.as_manifest(),
        "declarations": normalized,
    }
    canonical_json_bytes(document)
    return document


def canonical_detection_metadata_declarations_digest(
    direct_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_metadata_by_path: Mapping[str, Mapping[str, Any]],
    dimensions: CanonicalDetectionDimensions,
) -> str:
    return canonical_json_sha256(
        normalize_canonical_detection_metadata_declarations(
            direct_metadata_by_path,
            consolidated_metadata_by_path=consolidated_metadata_by_path,
            dimensions=dimensions,
        )
    )


def build_canonical_detection_run_manifest(
    *,
    run_id: str,
    dimensions: CanonicalDetectionDimensions,
    storage_plan: CanonicalDetectionStoragePlanSet,
    arrays: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    selector_eligible: bool = False,
) -> dict[str, object]:
    """Build the exact persisted canonical detection run manifest."""

    resolved_run_id = _require_run_id(run_id)
    if storage_plan.dimensions != dimensions:
        raise ValueError("Canonical storage plan dimensions do not match.")
    source_errors = validate_legacy_detection_source_evidence(source_evidence)
    if source_errors:
        raise ValueError("Invalid source evidence: " + "; ".join(source_errors))
    logical_content = canonical_detection_logical_content_document(
        arrays,
        dimensions=dimensions,
    )
    metadata_digest = canonical_detection_metadata_declarations_digest(
        direct_metadata_declarations,
        consolidated_metadata_by_path=consolidated_metadata_declarations,
        dimensions=dimensions,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "detect",
        "publication": {
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": bool(selector_eligible),
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": (
                CANONICAL_DETECTION_METADATA_DIGEST_SCOPE
            ),
            "metadata_declarations_digest_algorithm": (CANONICAL_JSON_DIGEST_ALGORITHM),
            "metadata_declarations_digest": metadata_digest,
        },
        "logical_schema": CANONICAL_DETECTION_SCHEMA_V1.as_manifest(
            dimensions=dimensions
        ),
        "storage_plan": storage_plan.as_manifest(),
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(logical_content),
            "document": logical_content,
        },
        "source_evidence": dict(source_evidence),
    }
    envelope: dict[str, object] = {
        "schema_id": CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": CANONICAL_DETECTION_RUN_MANIFEST_ATTRIBUTE,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def _dimensions_from_manifest(
    logical: Mapping[str, Any],
) -> CanonicalDetectionDimensions:
    raw = logical.get("dimensions")
    if not isinstance(raw, Mapping) or set(raw) != {
        "n_frames",
        "n_instances",
        "n_frame_boundaries",
        "source_width",
        "source_height",
    }:
        raise ValueError("Canonical logical dimensions are not exact.")
    dimensions = CanonicalDetectionDimensions(
        n_frames=raw.get("n_frames"),
        n_instances=raw.get("n_instances"),
        source_width=raw.get("source_width"),
        source_height=raw.get("source_height"),
    )
    if raw.get("n_frame_boundaries") != dimensions.n_frames + 1:
        raise ValueError("n_frame_boundaries must equal n_frames + 1.")
    return dimensions


def validate_canonical_detection_run_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the complete persisted envelope without reopening arrays."""

    errors: list[str] = []
    if set(manifest) != {
        "schema_id",
        "schema_version",
        "persisted_attribute",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("canonical run manifest envelope has unexpected fields")
    if manifest.get("schema_id") != CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID:
        errors.append("canonical run manifest schema_id mismatch")
    if manifest.get("schema_version") != (
        CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_VERSION
    ):
        errors.append("canonical run manifest schema_version mismatch")
    if manifest.get("persisted_attribute") != (
        CANONICAL_DETECTION_RUN_MANIFEST_ATTRIBUTE
    ):
        errors.append("canonical run manifest persisted_attribute mismatch")
    if manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("canonical run manifest digest_algorithm mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "canonical run manifest payload must be an object")
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        return (*errors, f"canonical run manifest is not strict JSON: {exc}")
    if manifest.get("payload_digest") != expected_digest:
        errors.append("canonical run manifest payload_digest mismatch")
    if set(payload) != {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "storage_plan",
        "logical_content",
        "source_evidence",
    }:
        errors.append("canonical run manifest payload has unexpected fields")
    if payload.get("stage") != "detect":
        errors.append("canonical run manifest stage mismatch")
    try:
        _require_run_id(payload.get("run_id"))
    except ValueError as exc:
        errors.append(str(exc))

    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("canonical publication must be an object")
    else:
        expected_publication = {
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": publication.get("stage_selector_eligible"),
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": (
                CANONICAL_DETECTION_METADATA_DIGEST_SCOPE
            ),
            "metadata_declarations_digest_algorithm": (CANONICAL_JSON_DIGEST_ALGORITHM),
            "metadata_declarations_digest": publication.get(
                "metadata_declarations_digest"
            ),
        }
        if dict(publication) != expected_publication:
            errors.append("canonical publication is not in exact persisted form")
        if type(publication.get("stage_selector_eligible")) is not bool:
            errors.append("canonical selector eligibility must be boolean")
        try:
            _require_sha256(
                publication.get("metadata_declarations_digest"),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))

    logical = payload.get("logical_schema")
    dimensions: CanonicalDetectionDimensions | None = None
    if not isinstance(logical, Mapping):
        errors.append("canonical logical_schema must be an object")
    else:
        try:
            dimensions = _dimensions_from_manifest(logical)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        if dimensions is not None and dict(logical) != (
            CANONICAL_DETECTION_SCHEMA_V1.as_manifest(dimensions=dimensions)
        ):
            errors.append("canonical logical_schema differs from frozen builder")

    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("canonical storage_plan must be an object")
    elif dimensions is not None:
        raw_profile = storage.get("storage_profile")
        if not isinstance(raw_profile, Mapping):
            errors.append("canonical storage_plan storage_profile must be an object")
        else:
            try:
                profile = storage_profile_from_manifest(raw_profile)
                expected_storage = plan_canonical_detection_storage(
                    dimensions,
                    profile=profile,
                ).as_manifest()
            except (TypeError, ValueError) as exc:
                errors.append(f"cannot reconstruct canonical storage_plan: {exc}")
            else:
                if dict(storage) != expected_storage:
                    errors.append(
                        "canonical storage_plan differs from byte planner output"
                    )

    content = payload.get("logical_content")
    if not isinstance(content, Mapping) or set(content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("canonical logical_content envelope is invalid")
    else:
        if content.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
            errors.append("canonical logical_content digest algorithm mismatch")
        document = content.get("document")
        if not isinstance(document, Mapping):
            errors.append("canonical logical_content document is invalid")
        elif content.get("digest") != canonical_json_sha256(document):
            errors.append("canonical logical_content digest mismatch")
        if dimensions is not None and isinstance(document, Mapping):
            expected_content_fields = {
                "schema_id",
                "schema_version",
                "logical_schema",
                "dimensions",
                "arrays",
            }
            if set(document) != expected_content_fields:
                errors.append(
                    "canonical logical_content document has unexpected fields"
                )
            if document.get("schema_id") != (
                CANONICAL_DETECTION_LOGICAL_CONTENT_SCHEMA_ID
            ) or document.get("schema_version") != (
                CANONICAL_DETECTION_LOGICAL_CONTENT_SCHEMA_VERSION
            ):
                errors.append("canonical logical_content schema identity mismatch")
            if document.get("logical_schema") != {
                "id": CANONICAL_DETECTION_SCHEMA_V1.schema_id,
                "version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
            }:
                errors.append("canonical logical_content logical schema mismatch")
            if document.get("dimensions") != dimensions.as_manifest():
                errors.append("canonical logical_content dimensions mismatch")
            arrays = document.get("arrays")
            if not isinstance(arrays, Mapping) or set(arrays) != set(
                CANONICAL_DETECTION_SCHEMA_V1.binding_paths
            ):
                errors.append("canonical logical_content array declarations mismatch")
            else:
                binding_by_path = {
                    binding.path: binding
                    for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings
                }
                for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
                    item = arrays[path]
                    if not isinstance(item, Mapping) or set(item) != {
                        "shape",
                        "dtype",
                        "digest_algorithm",
                        "sha256",
                    }:
                        errors.append(
                            f"canonical logical_content declaration invalid at {path!r}"
                        )
                        continue
                    binding = binding_by_path[path]
                    contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
                        binding.contract_id,
                        binding.contract_version,
                    )
                    expected_shape = [
                        (
                            dimension
                            if isinstance(dimension, int)
                            else dimensions.contract_dimensions[dimension]
                        )
                        for dimension in contract.shape_template
                    ]
                    expected_dtype = str(contract.dtype.numpy_dtype)
                    if item.get("shape") != expected_shape:
                        errors.append(
                            f"canonical logical_content shape mismatch at {path!r}"
                        )
                    if item.get("dtype") != expected_dtype:
                        errors.append(
                            f"canonical logical_content dtype mismatch at {path!r}"
                        )
                    if item.get("digest_algorithm") != (
                        CANONICAL_DETECTION_ARRAY_DIGEST_ALGORITHM
                    ):
                        errors.append(
                            f"canonical logical_content digest algorithm mismatch at {path!r}"
                        )
                    try:
                        _require_sha256(
                            item.get("sha256"),
                            name=f"logical_content {path} sha256",
                        )
                    except ValueError as exc:
                        errors.append(str(exc))

    source = payload.get("source_evidence")
    if not isinstance(source, Mapping):
        errors.append("canonical source_evidence must be an object")
    else:
        errors.extend(validate_legacy_detection_source_evidence(source))
    return tuple(dict.fromkeys(errors))


def validate_canonical_detection_publication(
    manifest: Mapping[str, Any],
    *,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recompute array, metadata, and manifest evidence before acceptance."""

    errors = list(validate_canonical_detection_run_manifest(manifest))
    payload = manifest.get("payload")
    logical = payload.get("logical_schema") if isinstance(payload, Mapping) else None
    if not isinstance(logical, Mapping):
        return (*errors, "canonical publication lacks logical_schema")
    try:
        dimensions = _dimensions_from_manifest(logical)
    except (TypeError, ValueError) as exc:
        return (*errors, f"canonical publication dimensions are invalid: {exc}")
    try:
        observed_content = canonical_detection_logical_content_document(
            arrays,
            dimensions=dimensions,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"canonical logical array validation failed: {exc}")
    else:
        content = payload.get("logical_content")
        expected_document = (
            content.get("document") if isinstance(content, Mapping) else None
        )
        if observed_content != expected_document:
            errors.append("canonical logical_content differs from decoded arrays")
    try:
        observed_metadata = canonical_detection_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_metadata_by_path=consolidated_metadata_declarations,
            dimensions=dimensions,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"canonical metadata declaration validation failed: {exc}")
    else:
        publication = payload.get("publication")
        expected_metadata = (
            publication.get("metadata_declarations_digest")
            if isinstance(publication, Mapping)
            else None
        )
        if observed_metadata != expected_metadata:
            errors.append("canonical metadata declaration digest mismatch")
    storage = payload.get("storage_plan")
    raw_profile = (
        storage.get("storage_profile") if isinstance(storage, Mapping) else None
    )
    try:
        if not isinstance(raw_profile, Mapping):
            raise ValueError("canonical storage profile is missing")
        profile = storage_profile_from_manifest(raw_profile)
        plans = plan_canonical_detection_storage(dimensions, profile=profile)
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct canonical physical plan: {exc}")
        return tuple(dict.fromkeys(errors))
    binding_by_path = {
        binding.path: binding for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings
    }
    for entry in plans.entries:
        declaration = direct_metadata_declarations.get(entry.rule.path)
        if not isinstance(declaration, Mapping):
            continue
        binding = binding_by_path[entry.rule.path]
        contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        physical_errors = validate_array_metadata_declaration_from_plan(
            declaration,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
        )
        errors.extend(
            f"canonical physical metadata at {entry.rule.path}: {error}"
            for error in physical_errors
        )
    return tuple(dict.fromkeys(errors))


def refined_source_identity_from_canonical_manifest(manifest: Mapping[str, Any]):
    """Return the exact refined-v1 source binding after manifest validation."""

    errors = validate_canonical_detection_run_manifest(manifest)
    if errors:
        raise ValueError("Invalid canonical source manifest: " + "; ".join(errors))
    payload = manifest["payload"]
    content = payload["logical_content"]
    from fisheye.shared.zarr.refined_detection_manifest import (  # local cycle guard
        RefinedDetectionSourceIdentity,
    )

    return RefinedDetectionSourceIdentity(
        run_id=payload["run_id"],
        run_manifest_digest=manifest["payload_digest"],
        logical_content_digest=content["digest"],
    )


__all__ = [
    "CANONICAL_DETECTION_ARRAY_DIGEST_ALGORITHM",
    "CANONICAL_DETECTION_LEGACY_SOURCE_SCHEMA_ID",
    "CANONICAL_DETECTION_LOGICAL_CONTENT_SCHEMA_ID",
    "CANONICAL_DETECTION_METADATA_DIGEST_SCOPE",
    "CANONICAL_DETECTION_RUN_MANIFEST_ATTRIBUTE",
    "CANONICAL_DETECTION_RUN_MANIFEST_PERSISTED_PATH",
    "CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID",
    "build_canonical_detection_run_manifest",
    "build_legacy_detection_source_evidence",
    "canonical_detection_logical_content_digest",
    "canonical_detection_logical_content_document",
    "canonical_detection_metadata_declarations_digest",
    "normalize_canonical_detection_metadata_declarations",
    "refined_source_identity_from_canonical_manifest",
    "validate_canonical_detection_publication",
    "validate_canonical_detection_run_manifest",
    "validate_legacy_detection_source_evidence",
]
