"""Digest-bound registry identity receipts for immutable analytics exports."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import UUID

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)


REGISTRY_IDENTITY_RECEIPT_SCHEMA_ID = (
    "palette.analytics_export.registry_identity_receipt"
)
REGISTRY_IDENTITY_RECEIPT_SCHEMA_VERSION = 2
REGISTRY_IDENTITY_SESSION_FIELD = "experimental_session_id"
REGISTRY_IDENTITY_SESSION_SNAPSHOT_FIELD = "experimental_session_snapshot_id"
REGISTRY_IDENTITY_SESSION_SCHEMA_ID_FIELD = "experimental_session_schema_id"
REGISTRY_IDENTITY_SESSION_SCHEMA_ID = "palette.registry.experimental_session.v1"
REGISTRY_IDENTITY_SESSION_SCHEMA_VERSION_FIELD = (
    "experimental_session_creation_registry_schema_version"
)
REGISTRY_IDENTITY_SESSION_SOURCE = (
    f"dataset_context_current.{REGISTRY_IDENTITY_SESSION_FIELD}"
)
REGISTRY_IDENTITY_ASSIGNMENT_SNAPSHOT_FIELD = (
    "experimental_session_assignment_snapshot_id"
)
REGISTRY_IDENTITY_ASSIGNMENT_BATCH_FIELD = "experimental_session_assignment_batch_id"
REGISTRY_IDENTITY_ASSIGNMENT_REVISION_FIELD = (
    "experimental_session_assignment_revision"
)
REGISTRY_IDENTITY_ASSIGNMENT_SUPERSEDES_FIELD = (
    "experimental_session_supersedes_assignment_snapshot_id"
)
REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID_FIELD = (
    "experimental_session_assignment_schema_id"
)
REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_VERSION_FIELD = (
    "experimental_session_assignment_registry_schema_version"
)
REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID = (
    "palette.registry.experimental_session_assignment.v1"
)
REGISTRY_IDENTITY_STATUS = "explicit"
REGISTRY_IDENTITY_STATUS_FIELD = "experimental_session_identity_status"
REGISTRY_IDENTITY_ASSIGNMENT_METHOD_FIELD = "experimental_session_assignment_method"
REGISTRY_IDENTITY_ASSIGNED_BY_FIELD = "experimental_session_assigned_by"
REGISTRY_IDENTITY_ASSIGNED_AT_FIELD = "experimental_session_assigned_at_utc"
REGISTRY_IDENTITY_SUBJECT_SOURCE = (
    "coalesce(dataset_context_current.subject_id,"
    "dataset_context_current.legacy_fish_id)"
)
REGISTRY_IDENTITY_CARDINALITY_POLICY = "exactly_one_subject_per_source_v1"

_RECEIPT_BODY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "registry_path",
        "session_id_source",
        "subject_id_source",
        "subject_cardinality_policy",
        "sources",
    }
)
_RECEIPT_FIELDS = _RECEIPT_BODY_FIELDS | {"payload_sha256"}
_SOURCE_BODY_FIELDS = frozenset(
    {
        "dataset_id",
        "zarr_path",
        "recording_id",
        "session_id",
        "experimental_session_id",
        "experimental_session_snapshot_id",
        "experimental_session_schema_id",
        "experimental_session_schema_version",
        "assignment_snapshot_id",
        "assignment_batch_id",
        "assignment_revision",
        "supersedes_assignment_snapshot_id",
        "assignment_schema_id",
        "assignment_schema_version",
        "experimental_session_identity_status",
        "assignment_method",
        "assigned_by",
        "assigned_at_utc",
        "subject_id",
        "subject_count",
    }
)
_SOURCE_FIELDS = _SOURCE_BODY_FIELDS | {"record_sha256"}


class RegistryIdentityReceiptError(ValueError):
    """Raised when registry identity evidence is missing or malformed."""


def _fail(message: str) -> None:
    raise RegistryIdentityReceiptError(message)


def _exact_mapping(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object.")
    result = {str(key): item for key, item in value.items()}
    if set(result) != fields:
        _fail(
            f"{label} must contain exactly {sorted(fields)!r}; "
            f"found {sorted(result)!r}."
        )
    return result


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        _fail(f"{label} must be a non-empty trimmed string.")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _uuid(value: object, *, label: str) -> str:
    text = _nonempty(value, label=label)
    try:
        parsed = UUID(text)
    except ValueError as exc:
        _fail(f"{label} must be a UUID: {exc}.")
    canonical = str(parsed)
    if text != canonical:
        _fail(f"{label} must be a canonical lowercase UUID string.")
    return canonical


def _canonical_path(value: object, *, label: str) -> str:
    text = _nonempty(value, label=label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        _fail(f"{label} must be an absolute path.")
    canonical = str(path.resolve())
    if text != canonical:
        _fail(f"{label} must already be canonical: expected {canonical!r}.")
    return canonical


def _parse_subject_ids(value: object, *, label: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            _fail(f"{label} must contain a JSON array: {exc}.")
    if not isinstance(parsed, list):
        _fail(f"{label} must be a JSON array.")
    return tuple(_nonempty(item, label=f"{label} item") for item in parsed)


def build_registry_identity_source(
    *,
    zarr_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve one exact single-subject source from registry query rows."""

    canonical_path = str(Path(zarr_path).expanduser().resolve())
    if not rows:
        _fail(f"No live registry row matches source Zarr: {canonical_path}")

    def unique(field: str) -> set[Any]:
        return {row.get(field) for row in rows}

    dataset_ids = unique("dataset_id")
    recording_ids = unique("recording_id")
    session_ids = unique(REGISTRY_IDENTITY_SESSION_FIELD)
    session_snapshot_ids = unique(REGISTRY_IDENTITY_SESSION_SNAPSHOT_FIELD)
    session_schema_ids = unique(REGISTRY_IDENTITY_SESSION_SCHEMA_ID_FIELD)
    session_schema_versions = unique(REGISTRY_IDENTITY_SESSION_SCHEMA_VERSION_FIELD)
    assignment_snapshot_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_SNAPSHOT_FIELD)
    assignment_batch_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_BATCH_FIELD)
    assignment_revisions = unique(REGISTRY_IDENTITY_ASSIGNMENT_REVISION_FIELD)
    assignment_supersedes_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_SUPERSEDES_FIELD)
    assignment_schema_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID_FIELD)
    assignment_schema_versions = unique(REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_VERSION_FIELD)
    identity_statuses = unique(REGISTRY_IDENTITY_STATUS_FIELD)
    assignment_methods = unique(REGISTRY_IDENTITY_ASSIGNMENT_METHOD_FIELD)
    assigned_by_values = unique(REGISTRY_IDENTITY_ASSIGNED_BY_FIELD)
    assigned_at_values = unique(REGISTRY_IDENTITY_ASSIGNED_AT_FIELD)
    subject_ids = unique("fish_id")
    subject_counts = unique("subject_count")
    if len(dataset_ids) != 1 or type(next(iter(dataset_ids))) is not int:
        _fail(f"Registry source has ambiguous or invalid dataset_id: {canonical_path}")
    if len(recording_ids) != 1:
        _fail(f"Registry source has ambiguous recording identity: {canonical_path}")
    if len(session_ids) != 1:
        _fail(
            "Registry source has ambiguous or missing experimental-session "
            f"identity: {canonical_path}"
        )
    for values, label in (
        (session_snapshot_ids, "experimental_session_snapshot_id"),
        (session_schema_ids, "experimental_session_schema_id"),
        (assignment_snapshot_ids, "assignment_snapshot_id"),
        (assignment_batch_ids, "assignment_batch_id"),
        (assignment_revisions, "assignment_revision"),
        (assignment_supersedes_ids, "supersedes_assignment_snapshot_id"),
        (assignment_schema_ids, "assignment_schema_id"),
        (identity_statuses, "experimental_session_identity_status"),
        (assignment_methods, "assignment_method"),
        (assigned_by_values, "assigned_by"),
        (assigned_at_values, "assigned_at_utc"),
    ):
        if len(values) != 1:
            _fail(f"Registry source has ambiguous or missing {label}: {canonical_path}")
    if (
        len(session_schema_versions) != 1
        or type(next(iter(session_schema_versions))) is not int
        or next(iter(session_schema_versions)) < 1
    ):
        _fail(
            "Registry source has ambiguous or invalid session_schema_version: "
            f"{canonical_path}"
        )
    if (
        len(assignment_schema_versions) != 1
        or type(next(iter(assignment_schema_versions))) is not int
        or next(iter(assignment_schema_versions)) < 1
    ):
        _fail(
            "Registry source has ambiguous or invalid assignment_schema_version: "
            f"{canonical_path}"
        )
    if len(subject_ids) != 1:
        _fail(
            f"Registry source has ambiguous or missing subject identity: {canonical_path}"
        )
    if subject_counts != {1}:
        _fail(
            "Analytics export currently requires exactly one registry subject per "
            f"source; found {sorted(map(str, subject_counts))!r} for {canonical_path}."
        )
    recording_id = _nonempty(next(iter(recording_ids)), label="recording_id")
    session_id = _nonempty(next(iter(session_ids)), label="session_id")
    subject_id = _nonempty(next(iter(subject_ids)), label="subject_id")
    for row in rows:
        declared_subjects = _parse_subject_ids(
            row.get("subject_ids_json"),
            label="subject_ids_json",
        )
        if declared_subjects and declared_subjects != (subject_id,):
            _fail(
                "Registry subject_ids_json does not identify exactly the effective "
                f"single subject for {canonical_path}."
            )
    body = {
        "dataset_id": next(iter(dataset_ids)),
        "zarr_path": canonical_path,
        "recording_id": recording_id,
        "session_id": session_id,
        "experimental_session_id": session_id,
        "experimental_session_snapshot_id": _uuid(
            next(iter(session_snapshot_ids)),
            label="experimental_session_snapshot_id",
        ),
        "experimental_session_schema_id": _nonempty(
            next(iter(session_schema_ids)),
            label="experimental_session_schema_id",
        ),
        "experimental_session_schema_version": next(iter(session_schema_versions)),
        "assignment_snapshot_id": _nonempty(
            next(iter(assignment_snapshot_ids)),
            label="assignment_snapshot_id",
        ),
        "assignment_batch_id": _nonempty(
            next(iter(assignment_batch_ids)),
            label="assignment_batch_id",
        ),
        "assignment_revision": next(iter(assignment_revisions)),
        "supersedes_assignment_snapshot_id": next(iter(assignment_supersedes_ids)),
        "assignment_schema_id": _nonempty(
            next(iter(assignment_schema_ids)),
            label="assignment_schema_id",
        ),
        "assignment_schema_version": next(iter(assignment_schema_versions)),
        "experimental_session_identity_status": _nonempty(
            next(iter(identity_statuses)),
            label="experimental_session_identity_status",
        ),
        "assignment_method": _nonempty(
            next(iter(assignment_methods)),
            label="assignment_method",
        ),
        "assigned_by": _nonempty(
            next(iter(assigned_by_values)),
            label="assigned_by",
        ),
        "assigned_at_utc": _nonempty(
            next(iter(assigned_at_values)),
            label="assigned_at_utc",
        ),
        "subject_id": subject_id,
        "subject_count": 1,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def build_registry_identity_receipt(
    *,
    registry_path: str | Path,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one sorted self-digested registry snapshot receipt."""

    canonical_registry_path = str(Path(registry_path).expanduser().resolve())
    normalized = [dict(validate_registry_identity_source(source)) for source in sources]
    normalized.sort(key=lambda source: source["zarr_path"])
    paths = [source["zarr_path"] for source in normalized]
    if len(paths) != len(set(paths)):
        _fail("Registry identity receipt source paths must be unique.")
    body = {
        "schema_id": REGISTRY_IDENTITY_RECEIPT_SCHEMA_ID,
        "schema_version": REGISTRY_IDENTITY_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "registry_path": canonical_registry_path,
        "session_id_source": REGISTRY_IDENTITY_SESSION_SOURCE,
        "subject_id_source": REGISTRY_IDENTITY_SUBJECT_SOURCE,
        "subject_cardinality_policy": REGISTRY_IDENTITY_CARDINALITY_POLICY,
        "sources": normalized,
    }
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def validate_registry_identity_source(value: object) -> Mapping[str, Any]:
    record = _exact_mapping(value, _SOURCE_FIELDS, label="registry identity source")
    body = {key: record[key] for key in _SOURCE_BODY_FIELDS}
    if type(body["dataset_id"]) is not int:
        _fail("registry identity dataset_id must be an integer.")
    body["zarr_path"] = _canonical_path(body["zarr_path"], label="source zarr_path")
    for field in (
        "recording_id",
        "session_id",
        "experimental_session_id",
        "experimental_session_schema_id",
        "assignment_snapshot_id",
        "assignment_batch_id",
        "assignment_schema_id",
        "experimental_session_identity_status",
        "assignment_method",
        "assigned_by",
        "assigned_at_utc",
        "subject_id",
    ):
        body[field] = _nonempty(body[field], label=field)
    body["experimental_session_snapshot_id"] = _uuid(
        body["experimental_session_snapshot_id"],
        label="experimental_session_snapshot_id",
    )
    body["assignment_snapshot_id"] = _uuid(
        body["assignment_snapshot_id"],
        label="assignment_snapshot_id",
    )
    body["assignment_batch_id"] = _uuid(
        body["assignment_batch_id"],
        label="assignment_batch_id",
    )
    if body["session_id"] != body["experimental_session_id"]:
        _fail("registry export session_id must equal experimental_session_id.")
    if body["assignment_schema_id"] != REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID:
        _fail("registry assignment_schema_id is invalid.")
    if body["experimental_session_schema_id"] != REGISTRY_IDENTITY_SESSION_SCHEMA_ID:
        _fail("registry experimental_session_schema_id is invalid.")
    if body["experimental_session_identity_status"] != REGISTRY_IDENTITY_STATUS:
        _fail("registry experimental-session identity must be explicit.")
    if (
        type(body["experimental_session_schema_version"]) is not int
        or body["experimental_session_schema_version"] < 1
    ):
        _fail("registry experimental_session_schema_version must be positive.")
    if (
        type(body["assignment_schema_version"]) is not int
        or body["assignment_schema_version"] < 1
    ):
        _fail("registry assignment_schema_version must be a positive integer.")
    if type(body["assignment_revision"]) is not int or body["assignment_revision"] < 1:
        _fail("registry assignment_revision must be a positive integer.")
    supersedes = body["supersedes_assignment_snapshot_id"]
    if body["assignment_revision"] == 1:
        if supersedes is not None:
            _fail("registry assignment revision 1 must not supersede a snapshot.")
    else:
        body["supersedes_assignment_snapshot_id"] = _uuid(
            supersedes,
            label="supersedes_assignment_snapshot_id",
        )
    if body["subject_count"] != 1 or type(body["subject_count"]) is not int:
        _fail("registry identity source must declare exactly one subject.")
    digest = _sha256(record["record_sha256"], label="source record_sha256")
    if canonical_json_sha256(body) != digest:
        _fail("registry identity source digest mismatch.")
    return MappingProxyType({**body, "record_sha256": digest})


def validate_registry_identity_receipt(
    value: object,
    *,
    expected_zarr_paths: Sequence[str | Path] | None = None,
) -> Mapping[str, Any]:
    """Deeply validate a receipt and optionally its exact source inventory."""

    record = _exact_mapping(value, _RECEIPT_FIELDS, label="registry identity receipt")
    if record["schema_id"] != REGISTRY_IDENTITY_RECEIPT_SCHEMA_ID:
        _fail("registry identity receipt schema_id is invalid.")
    if record["schema_version"] != REGISTRY_IDENTITY_RECEIPT_SCHEMA_VERSION:
        _fail("registry identity receipt schema_version is invalid.")
    if record["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM:
        _fail("registry identity receipt digest algorithm is invalid.")
    if record["session_id_source"] != REGISTRY_IDENTITY_SESSION_SOURCE:
        _fail("registry identity receipt session source is invalid.")
    if record["subject_id_source"] != REGISTRY_IDENTITY_SUBJECT_SOURCE:
        _fail("registry identity receipt subject source is invalid.")
    if record["subject_cardinality_policy"] != REGISTRY_IDENTITY_CARDINALITY_POLICY:
        _fail("registry identity receipt subject policy is invalid.")
    registry_path = _canonical_path(record["registry_path"], label="registry_path")
    raw_sources = record["sources"]
    if not isinstance(raw_sources, list):
        _fail("registry identity receipt sources must be an array.")
    sources = [dict(validate_registry_identity_source(source)) for source in raw_sources]
    paths = [source["zarr_path"] for source in sources]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail("registry identity receipt sources must be sorted and unique.")
    if expected_zarr_paths is not None:
        expected = sorted(
            str(Path(path).expanduser().resolve()) for path in expected_zarr_paths
        )
        if paths != expected:
            _fail(
                "registry identity receipt source set differs from requested sources: "
                f"expected={expected!r}, found={paths!r}."
            )
    body = {
        "schema_id": record["schema_id"],
        "schema_version": record["schema_version"],
        "digest_algorithm": record["digest_algorithm"],
        "registry_path": registry_path,
        "session_id_source": record["session_id_source"],
        "subject_id_source": record["subject_id_source"],
        "subject_cardinality_policy": record["subject_cardinality_policy"],
        "sources": sources,
    }
    digest = _sha256(record["payload_sha256"], label="receipt payload_sha256")
    if canonical_json_sha256(body) != digest:
        _fail("registry identity receipt payload digest mismatch.")
    return MappingProxyType({**body, "payload_sha256": digest})


def registry_identity_sources_by_path(
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    validated = validate_registry_identity_receipt(receipt)
    return {source["zarr_path"]: dict(source) for source in validated["sources"]}


__all__ = [
    "REGISTRY_IDENTITY_CARDINALITY_POLICY",
    "REGISTRY_IDENTITY_RECEIPT_SCHEMA_ID",
    "REGISTRY_IDENTITY_RECEIPT_SCHEMA_VERSION",
    "REGISTRY_IDENTITY_ASSIGNMENT_BATCH_FIELD",
    "REGISTRY_IDENTITY_ASSIGNED_AT_FIELD",
    "REGISTRY_IDENTITY_ASSIGNED_BY_FIELD",
    "REGISTRY_IDENTITY_ASSIGNMENT_METHOD_FIELD",
    "REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID_FIELD",
    "REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID",
    "REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_VERSION_FIELD",
    "REGISTRY_IDENTITY_ASSIGNMENT_SNAPSHOT_FIELD",
    "REGISTRY_IDENTITY_STATUS_FIELD",
    "REGISTRY_IDENTITY_STATUS",
    "REGISTRY_IDENTITY_SESSION_FIELD",
    "REGISTRY_IDENTITY_SESSION_SOURCE",
    "REGISTRY_IDENTITY_SUBJECT_SOURCE",
    "RegistryIdentityReceiptError",
    "build_registry_identity_receipt",
    "build_registry_identity_source",
    "registry_identity_sources_by_path",
    "validate_registry_identity_receipt",
    "validate_registry_identity_source",
]
