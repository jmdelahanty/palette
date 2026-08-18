"""Immutable metadata receipts for scanned subject-mask coordinate surfaces.

This module deliberately contains no publication or array-reading logic.  A
producer that has already completed a full scientific scan can use the
receipt to bind that fact to the exact source manifest, coordinate records,
and hard-link evidence used by a later coordinate successor publication.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)


SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_ID = (
    "palette.subject_mask.coordinate_surface_validation_receipt"
)
SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_VERSION = 1
SUBJECT_MASK_COORDINATE_VALIDATION_POLICY = "complete_coordinate_surface_scan_v1"
SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE = (
    "coordinate_surface_validation_receipt"
)
SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE = (
    f"{SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE}_sha256"
)

RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND = "raw_subject_mask"
REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND = "refined_subject_mask"
SUBJECT_MASK_COORDINATE_VALIDATION_KINDS = frozenset(
    {
        RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
        REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    }
)

_RUN_FAMILY_BY_KIND = {
    RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND: "subject_mask_runs",
    REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND: "refined_subject_masks_runs",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TOP_LEVEL_FIELDS = {
    "schema_id",
    "schema_version",
    "digest_algorithm",
    "payload_digest",
    "payload",
}
_PAYLOAD_FIELDS = {
    "kind",
    "successor_run_path",
    "source",
    "source_validation",
    "bundle_authority",
    "coordinate_records",
    "payload_equivalence",
    "validation_policy",
    "validator_identity",
}
_SOURCE_FIELDS = {
    "run_path",
    "core_manifest_payload_digest",
    "core_manifest_document_digest",
    "logical_content_digest",
}
_SOURCE_VALIDATION_FIELDS = {
    "schema_id",
    "schema_version",
    "payload_digest",
    "document_sha256",
    "semantic_unit_count",
}
_BUNDLE_AUTHORITY_FIELDS = {"kind", "document_digest"}
_PAYLOAD_EQUIVALENCE_FIELDS = {
    "schema_id",
    "schema_version",
    "receipt_digest",
    "inventory_digest",
    "payload_file_count",
}
_RECORD_POINTER_FIELDS = {"record_ref", "record_sha256"}


class SubjectMaskCoordinateValidationReceiptError(ValueError):
    """Raised when a coordinate-surface validation receipt is invalid or stale."""


def _strict_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be strict canonical JSON: {exc}."
        ) from exc


def _require_object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be an object."
        )
    if any(type(key) is not str for key in value):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} keys must be strings."
        )
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    if set(value) != expected:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} fields are not exact: expected {sorted(expected)}, "
            f"got {sorted(value)}."
        )


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be lowercase hexadecimal SHA-256."
        )
    return value


def _require_positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be one positive exact integer."
        )
    return value


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be one nonnegative exact integer."
        )
    return value


def _require_nonempty_text(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be one nonempty string without surrounding whitespace."
        )
    if "\x00" in value or any(ord(character) < 0x20 for character in value):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} contains a control character."
        )
    return value


def _require_safe_name(value: Any, *, name: str) -> str:
    value = _require_nonempty_text(value, name=name)
    if _SAFE_NAME_RE.fullmatch(value) is None or value in {".", ".."}:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} is not a safe name."
        )
    return value


def _require_kind(value: Any, *, name: str = "kind") -> str:
    if value not in SUBJECT_MASK_COORDINATE_VALIDATION_KINDS:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} is unsupported: {value!r}."
        )
    return str(value)


def _require_run_path(value: Any, *, kind: str, name: str) -> str:
    value = _require_nonempty_text(value, name=name)
    expected_family = _RUN_FAMILY_BY_KIND[kind]
    parts = value.split("/")
    if (
        len(parts) != 2
        or parts[0] != expected_family
        or _SAFE_NAME_RE.fullmatch(parts[1]) is None
        or parts[1] in {".", ".."}
        or "\\" in value
    ):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be one exact {expected_family}/<run> path."
        )
    return value


def _require_record_ref(value: Any, *, name: str) -> str:
    value = _require_nonempty_text(value, name=name)
    if value.count("@") != 1:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must contain exactly one @attr suffix."
        )
    archive_path, attr_name = value.split("@", 1)
    if not archive_path.startswith("/") or archive_path.endswith("/"):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must use an absolute archive path."
        )
    if (
        "\\" in archive_path
        or "\x00" in archive_path
        or any(ord(character) < 0x20 for character in archive_path)
    ):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} contains an unsafe archive path."
        )
    path_parts = archive_path.split("/")
    if any(part in {"", ".", ".."} for part in path_parts[1:]):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} contains an unsafe archive path component."
        )
    _require_safe_name(attr_name, name=f"{name} attribute")
    return value


def _normalize_expected_names(
    names: Iterable[str] | None, *, name: str
) -> frozenset[str] | None:
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be an explicit iterable of names, not a string."
        )
    try:
        values = tuple(names)
    except TypeError as exc:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be an explicit iterable of names."
        ) from exc
    if not values:
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must be nonempty."
        )
    normalized = frozenset(
        _require_safe_name(value, name=f"{name} member") for value in values
    )
    if len(normalized) != len(values):
        raise SubjectMaskCoordinateValidationReceiptError(
            f"{name} must not contain duplicate names."
        )
    return normalized


def _validate_validator_identity(value: Any) -> dict[str, Any]:
    identity = _strict_copy(value, name="validator_identity")
    if not isinstance(identity, dict):
        raise SubjectMaskCoordinateValidationReceiptError(
            "validator_identity must be an object."
        )
    if set(identity) == {"package", "version"}:
        _require_nonempty_text(identity["package"], name="validator package")
        _require_nonempty_text(identity["version"], name="validator version")
    elif set(identity) == {"commit"}:
        _require_nonempty_text(identity["commit"], name="validator commit")
    else:
        raise SubjectMaskCoordinateValidationReceiptError(
            "validator_identity must contain exactly package/version or commit."
        )
    return identity


def _validate_source(value: Any, *, kind: str) -> dict[str, Any]:
    source = _strict_copy(value, name="source binding")
    source_mapping = _require_object(source, name="source binding")
    _require_exact_fields(source_mapping, _SOURCE_FIELDS, name="source binding")
    result = {
        "run_path": _require_run_path(
            source_mapping["run_path"], kind=kind, name="source run_path"
        ),
        "core_manifest_payload_digest": _require_sha256(
            source_mapping["core_manifest_payload_digest"],
            name="source core-manifest payload digest",
        ),
        "core_manifest_document_digest": _require_sha256(
            source_mapping["core_manifest_document_digest"],
            name="source core-manifest document digest",
        ),
        "logical_content_digest": _require_sha256(
            source_mapping["logical_content_digest"],
            name="source logical-content digest",
        ),
    }
    return result


def _validate_source_validation(value: Any) -> dict[str, Any]:
    binding = _strict_copy(value, name="source validation binding")
    mapping = _require_object(binding, name="source validation binding")
    _require_exact_fields(
        mapping, _SOURCE_VALIDATION_FIELDS, name="source validation binding"
    )
    return {
        "schema_id": _require_nonempty_text(
            mapping["schema_id"], name="source validation schema id"
        ),
        "schema_version": _require_positive_integer(
            mapping["schema_version"], name="source validation schema version"
        ),
        "payload_digest": _require_sha256(
            mapping["payload_digest"], name="source validation payload digest"
        ),
        "document_sha256": _require_sha256(
            mapping["document_sha256"], name="source validation document sha256"
        ),
        "semantic_unit_count": _require_nonnegative_integer(
            mapping["semantic_unit_count"], name="semantic unit count"
        ),
    }


def _validate_bundle_authority(value: Any) -> dict[str, Any]:
    binding = _strict_copy(value, name="bundle authority binding")
    mapping = _require_object(binding, name="bundle authority binding")
    _require_exact_fields(
        mapping, _BUNDLE_AUTHORITY_FIELDS, name="bundle authority binding"
    )
    return {
        "kind": _require_nonempty_text(
            mapping["kind"], name="bundle authority kind"
        ),
        "document_digest": _require_sha256(
            mapping["document_digest"], name="bundle authority document digest"
        ),
    }


def _validate_payload_equivalence(value: Any) -> dict[str, Any]:
    binding = _strict_copy(value, name="payload-equivalence binding")
    mapping = _require_object(binding, name="payload-equivalence binding")
    _require_exact_fields(
        mapping, _PAYLOAD_EQUIVALENCE_FIELDS, name="payload-equivalence binding"
    )
    return {
        "schema_id": _require_nonempty_text(
            mapping["schema_id"], name="payload-equivalence schema id"
        ),
        "schema_version": _require_positive_integer(
            mapping["schema_version"], name="payload-equivalence schema version"
        ),
        "receipt_digest": _require_sha256(
            mapping["receipt_digest"], name="payload-equivalence receipt digest"
        ),
        "inventory_digest": _require_sha256(
            mapping["inventory_digest"], name="payload-equivalence inventory digest"
        ),
        "payload_file_count": _require_nonnegative_integer(
            mapping["payload_file_count"], name="payload file count"
        ),
    }


def _validate_coordinate_records(
    value: Any,
    *,
    expected_names: frozenset[str] | None,
) -> dict[str, dict[str, str]]:
    records = _strict_copy(value, name="coordinate records")
    mapping = _require_object(records, name="coordinate records")
    if not mapping:
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate records must be nonempty."
        )
    normalized: dict[str, dict[str, str]] = {}
    for name, pointer in mapping.items():
        safe_name = _require_safe_name(name, name="coordinate record name")
        pointer_mapping = _require_object(
            pointer, name=f"coordinate record {safe_name!r}"
        )
        _require_exact_fields(
            pointer_mapping,
            _RECORD_POINTER_FIELDS,
            name=f"coordinate record {safe_name!r}",
        )
        normalized[safe_name] = {
            "record_ref": _require_record_ref(
                pointer_mapping["record_ref"],
                name=f"coordinate record {safe_name!r} record_ref",
            ),
            "record_sha256": _require_sha256(
                pointer_mapping["record_sha256"],
                name=f"coordinate record {safe_name!r} record_sha256",
            ),
        }
    actual_names = frozenset(normalized)
    if expected_names is not None and actual_names != expected_names:
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate record names differ from the explicit closed key set."
        )
    return {name: normalized[name] for name in sorted(normalized)}


def _validate_document(
    document: Any,
    *,
    expected_kind: str | None = None,
    expected_successor_run_path: str | None = None,
    expected_coordinate_record_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    canonical = _strict_copy(
        document, name="subject-mask coordinate validation receipt"
    )
    top = _require_object(
        canonical, name="subject-mask coordinate validation receipt"
    )
    _require_exact_fields(
        top,
        _TOP_LEVEL_FIELDS,
        name="subject-mask coordinate validation receipt",
    )
    if (
        top["schema_id"] != SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_ID
        or top["schema_version"]
        != SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_VERSION
        or top["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt schema header is unsupported."
        )
    digest = _require_sha256(top["payload_digest"], name="receipt payload digest")
    body = {key: value for key, value in top.items() if key != "payload_digest"}
    if digest != canonical_json_sha256(body):
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt payload digest is stale."
        )
    payload = _strict_copy(top["payload"], name="coordinate validation payload")
    payload_mapping = _require_object(payload, name="coordinate validation payload")
    _require_exact_fields(
        payload_mapping, _PAYLOAD_FIELDS, name="coordinate validation payload"
    )
    kind = _require_kind(payload_mapping["kind"])
    if expected_kind is not None and kind != _require_kind(expected_kind):
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt kind differs from the requested kind."
        )
    successor_run_path = _require_run_path(
        payload_mapping["successor_run_path"],
        kind=kind,
        name="successor_run_path",
    )
    if expected_successor_run_path is not None:
        expected_path = _require_run_path(
            expected_successor_run_path,
            kind=kind,
            name="expected_successor_run_path",
        )
        if successor_run_path != expected_path:
            raise SubjectMaskCoordinateValidationReceiptError(
                "coordinate validation successor run path differs."
            )
    expected_names = _normalize_expected_names(
        expected_coordinate_record_names,
        name="expected_coordinate_record_names",
    )
    source = _validate_source(payload_mapping["source"], kind=kind)
    source_validation = _validate_source_validation(
        payload_mapping["source_validation"]
    )
    bundle_authority = _validate_bundle_authority(
        payload_mapping["bundle_authority"]
    )
    records = _validate_coordinate_records(
        payload_mapping["coordinate_records"], expected_names=expected_names
    )
    payload_equivalence = _validate_payload_equivalence(
        payload_mapping["payload_equivalence"]
    )
    if payload_mapping["validation_policy"] != SUBJECT_MASK_COORDINATE_VALIDATION_POLICY:
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation policy is unsupported."
        )
    validator_identity = _validate_validator_identity(
        payload_mapping["validator_identity"]
    )
    canonical_payload = {
        "kind": kind,
        "successor_run_path": successor_run_path,
        "source": source,
        "source_validation": source_validation,
        "bundle_authority": bundle_authority,
        "coordinate_records": records,
        "payload_equivalence": payload_equivalence,
        "validation_policy": SUBJECT_MASK_COORDINATE_VALIDATION_POLICY,
        "validator_identity": validator_identity,
    }
    if canonical_payload != payload_mapping:
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt is not canonically normalized."
        )
    return {
        "schema_id": top["schema_id"],
        "schema_version": top["schema_version"],
        "digest_algorithm": top["digest_algorithm"],
        "payload_digest": digest,
        "payload": canonical_payload,
    }


def build_subject_mask_coordinate_validation_receipt(
    *,
    kind: str,
    successor_run_path: str,
    source: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    bundle_authority: Mapping[str, Any],
    coordinate_records: Mapping[str, Mapping[str, Any]],
    coordinate_record_names: Iterable[str],
    payload_equivalence: Mapping[str, Any],
    validator_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one strict receipt from explicitly supplied immutable evidence.

    ``coordinate_record_names`` is intentionally required.  The builder never
    infers the closed key set from the mapping it receives; callers must state
    which kind-specific records were scanned and bound.
    """

    resolved_kind = _require_kind(kind)
    resolved_successor = _require_run_path(
        successor_run_path, kind=resolved_kind, name="successor_run_path"
    )
    expected_names = _normalize_expected_names(
        coordinate_record_names, name="coordinate_record_names"
    )
    assert expected_names is not None
    payload = {
        "kind": resolved_kind,
        "successor_run_path": resolved_successor,
        "source": _validate_source(source, kind=resolved_kind),
        "source_validation": _validate_source_validation(source_validation),
        "bundle_authority": _validate_bundle_authority(bundle_authority),
        "coordinate_records": _validate_coordinate_records(
            coordinate_records, expected_names=expected_names
        ),
        "payload_equivalence": _validate_payload_equivalence(payload_equivalence),
        "validation_policy": SUBJECT_MASK_COORDINATE_VALIDATION_POLICY,
        "validator_identity": _validate_validator_identity(validator_identity),
    }
    body = {
        "schema_id": SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload": payload,
    }
    document = {**body, "payload_digest": canonical_json_sha256(body)}
    return _validate_document(
        document,
        expected_kind=resolved_kind,
        expected_successor_run_path=resolved_successor,
        expected_coordinate_record_names=expected_names,
    )


def validate_subject_mask_coordinate_validation_receipt(
    document: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    expected_successor_run_path: str | None = None,
    expected_coordinate_record_names: Iterable[str] | None = None,
    expected_record_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate a receipt without opening or indexing any scientific array."""

    if expected_coordinate_record_names is not None and expected_record_names is not None:
        raise SubjectMaskCoordinateValidationReceiptError(
            "Provide only one expected coordinate-record name set."
        )
    names = (
        expected_coordinate_record_names
        if expected_coordinate_record_names is not None
        else expected_record_names
    )
    return _validate_document(
        document,
        expected_kind=expected_kind,
        expected_successor_run_path=expected_successor_run_path,
        expected_coordinate_record_names=names,
    )


def stamp_subject_mask_coordinate_validation_receipt(
    run: Any,
    receipt: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    expected_successor_run_path: str | None = None,
    expected_coordinate_record_names: Iterable[str] | None = None,
    expected_record_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Persist a validated receipt and a companion whole-document digest."""

    document = validate_subject_mask_coordinate_validation_receipt(
        receipt,
        expected_kind=expected_kind,
        expected_successor_run_path=expected_successor_run_path,
        expected_coordinate_record_names=expected_coordinate_record_names,
        expected_record_names=expected_record_names,
    )
    run_path = str(getattr(run, "path", "")).strip("/")
    if not run_path:
        raise SubjectMaskCoordinateValidationReceiptError(
            "run must expose a nonempty path."
        )
    if run_path != document["payload"]["successor_run_path"]:
        raise SubjectMaskCoordinateValidationReceiptError(
            "receipt successor run path differs from the target run."
        )
    attrs = getattr(run, "attrs", None)
    if attrs is None:
        raise SubjectMaskCoordinateValidationReceiptError(
            "run must expose mutable attrs."
        )
    persisted = copy.deepcopy(document)
    document_digest = canonical_json_sha256(persisted)
    existing = attrs.get(SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE)
    existing_digest = attrs.get(
        SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE
    )
    if existing is not None or existing_digest is not None:
        if existing != persisted or existing_digest != document_digest:
            raise SubjectMaskCoordinateValidationReceiptError(
                "coordinate validation receipt is already occupied by different "
                "or stale evidence."
            )
        return copy.deepcopy(persisted)
    attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE] = persisted
    attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE] = document_digest
    if (
        attrs.get(SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE) != persisted
        or attrs.get(SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE)
        != document_digest
    ):
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt did not persist exactly."
        )
    return copy.deepcopy(persisted)


def load_subject_mask_coordinate_validation_receipt(
    run: Any,
    *,
    expected_kind: str | None = None,
    expected_successor_run_path: str | None = None,
    expected_coordinate_record_names: Iterable[str] | None = None,
    expected_record_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load and validate only receipt metadata from a run's attrs."""

    attrs = getattr(run, "attrs", None)
    if attrs is None:
        raise SubjectMaskCoordinateValidationReceiptError(
            "run does not expose attrs."
        )
    value = attrs.get(SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE)
    if not isinstance(value, Mapping):
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt is absent."
        )
    document = validate_subject_mask_coordinate_validation_receipt(
        value,
        expected_kind=expected_kind,
        expected_successor_run_path=expected_successor_run_path,
        expected_coordinate_record_names=expected_coordinate_record_names,
        expected_record_names=expected_record_names,
    )
    expected_digest = canonical_json_sha256(document)
    if (
        attrs.get(SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE)
        != expected_digest
    ):
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt whole-document digest is stale."
        )
    run_path = str(getattr(run, "path", "")).strip("/")
    if run_path and run_path != document["payload"]["successor_run_path"]:
        raise SubjectMaskCoordinateValidationReceiptError(
            "coordinate validation receipt successor run path differs from the run."
        )
    return copy.deepcopy(document)


__all__ = [
    "RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND",
    "REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND",
    "SUBJECT_MASK_COORDINATE_VALIDATION_KINDS",
    "SUBJECT_MASK_COORDINATE_VALIDATION_POLICY",
    "SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE",
    "SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE",
    "SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_ID",
    "SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_SCHEMA_VERSION",
    "SubjectMaskCoordinateValidationReceiptError",
    "build_subject_mask_coordinate_validation_receipt",
    "load_subject_mask_coordinate_validation_receipt",
    "stamp_subject_mask_coordinate_validation_receipt",
    "validate_subject_mask_coordinate_validation_receipt",
]
