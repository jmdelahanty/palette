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
REGISTRY_IDENTITY_RECEIPT_SCHEMA_VERSION = 3
REGISTRY_IDENTITY_BATCH_FIELD = "acquisition_batch_id"
REGISTRY_IDENTITY_BATCH_SNAPSHOT_FIELD = "acquisition_batch_snapshot_id"
REGISTRY_IDENTITY_BATCH_SCHEMA_ID_FIELD = "acquisition_batch_schema_id"
REGISTRY_IDENTITY_BATCH_SCHEMA_ID = "palette.registry.acquisition_batch.v1"
REGISTRY_IDENTITY_BATCH_SCHEMA_VERSION_FIELD = (
    "acquisition_batch_creation_registry_schema_version"
)
REGISTRY_IDENTITY_BATCH_SOURCE = (
    f"dataset_context_current.{REGISTRY_IDENTITY_BATCH_FIELD}"
)
REGISTRY_IDENTITY_ASSIGNMENT_SNAPSHOT_FIELD = "acquisition_batch_assignment_snapshot_id"
REGISTRY_IDENTITY_ASSIGNMENT_BATCH_FIELD = "acquisition_batch_assignment_batch_id"
REGISTRY_IDENTITY_ASSIGNMENT_REVISION_FIELD = "acquisition_batch_assignment_revision"
REGISTRY_IDENTITY_ASSIGNMENT_SUPERSEDES_FIELD = (
    "acquisition_batch_supersedes_assignment_snapshot_id"
)
REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID_FIELD = "acquisition_batch_assignment_schema_id"
REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_VERSION_FIELD = (
    "acquisition_batch_assignment_registry_schema_version"
)
REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID = (
    "palette.registry.acquisition_batch_assignment.v1"
)
REGISTRY_IDENTITY_BATCH_STATUSES = frozenset({"explicit", "missing"})
REGISTRY_IDENTITY_STATUS_FIELD = "acquisition_batch_identity_status"
REGISTRY_IDENTITY_ASSIGNMENT_METHOD_FIELD = "acquisition_batch_assignment_method"
REGISTRY_IDENTITY_ASSIGNED_BY_FIELD = "acquisition_batch_assigned_by"
REGISTRY_IDENTITY_ASSIGNED_AT_FIELD = "acquisition_batch_assigned_at_utc"
REGISTRY_IDENTITY_SUBJECT_SOURCE = (
    "coalesce(dataset_context_current.subject_id,"
    "dataset_context_current.legacy_fish_id)"
)
REGISTRY_IDENTITY_CARDINALITY_POLICY = "exactly_one_subject_per_source_v1"
REGISTRY_IDENTITY_EXPERIMENTAL_UNIT_POLICY = "subject_is_experimental_unit_v1"
REGISTRY_IDENTITY_BATCH_POLICY = "optional_explicit_nuisance_block_v1"

_RECEIPT_BODY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "registry_path",
        "acquisition_batch_id_source",
        "acquisition_batch_policy",
        "subject_id_source",
        "subject_cardinality_policy",
        "experimental_unit_policy",
        "sources",
    }
)
_RECEIPT_FIELDS = _RECEIPT_BODY_FIELDS | {"payload_sha256"}
_SOURCE_BODY_FIELDS = frozenset(
    {
        "dataset_id",
        "zarr_path",
        "recording_id",
        "experimental_unit_id",
        "experimental_unit_kind",
        "acquisition_batch_id",
        "acquisition_batch_snapshot_id",
        "acquisition_batch_schema_id",
        "acquisition_batch_schema_version",
        "assignment_snapshot_id",
        "assignment_batch_id",
        "assignment_revision",
        "supersedes_assignment_snapshot_id",
        "assignment_schema_id",
        "assignment_schema_version",
        "acquisition_batch_identity_status",
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


def _optional_uuid(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _uuid(value, label=label)


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
    acquisition_batch_ids = unique(REGISTRY_IDENTITY_BATCH_FIELD)
    batch_snapshot_ids = unique(REGISTRY_IDENTITY_BATCH_SNAPSHOT_FIELD)
    batch_schema_ids = unique(REGISTRY_IDENTITY_BATCH_SCHEMA_ID_FIELD)
    batch_schema_versions = unique(REGISTRY_IDENTITY_BATCH_SCHEMA_VERSION_FIELD)
    assignment_snapshot_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_SNAPSHOT_FIELD)
    assignment_batch_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_BATCH_FIELD)
    assignment_revisions = unique(REGISTRY_IDENTITY_ASSIGNMENT_REVISION_FIELD)
    assignment_supersedes_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_SUPERSEDES_FIELD)
    assignment_schema_ids = unique(REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID_FIELD)
    assignment_schema_versions = unique(
        REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_VERSION_FIELD
    )
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
    for values, label in (
        (acquisition_batch_ids, "acquisition_batch_id"),
        (batch_snapshot_ids, "acquisition_batch_snapshot_id"),
        (batch_schema_ids, "acquisition_batch_schema_id"),
        (batch_schema_versions, "acquisition_batch_schema_version"),
        (assignment_snapshot_ids, "assignment_snapshot_id"),
        (assignment_batch_ids, "assignment_batch_id"),
        (assignment_revisions, "assignment_revision"),
        (assignment_supersedes_ids, "supersedes_assignment_snapshot_id"),
        (assignment_schema_ids, "assignment_schema_id"),
        (assignment_schema_versions, "assignment_schema_version"),
        (identity_statuses, "acquisition_batch_identity_status"),
        (assignment_methods, "assignment_method"),
        (assigned_by_values, "assigned_by"),
        (assigned_at_values, "assigned_at_utc"),
    ):
        if len(values) != 1:
            _fail(f"Registry source has ambiguous {label}: {canonical_path}")
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
    batch_status = next(iter(identity_statuses))
    if batch_status not in REGISTRY_IDENTITY_BATCH_STATUSES:
        _fail(
            "Registry source has invalid acquisition_batch_identity_status: "
            f"{canonical_path}"
        )
    raw_batch = {
        "acquisition_batch_id": next(iter(acquisition_batch_ids)),
        "acquisition_batch_snapshot_id": next(iter(batch_snapshot_ids)),
        "acquisition_batch_schema_id": next(iter(batch_schema_ids)),
        "acquisition_batch_schema_version": next(iter(batch_schema_versions)),
        "assignment_snapshot_id": next(iter(assignment_snapshot_ids)),
        "assignment_batch_id": next(iter(assignment_batch_ids)),
        "assignment_revision": next(iter(assignment_revisions)),
        "supersedes_assignment_snapshot_id": next(iter(assignment_supersedes_ids)),
        "assignment_schema_id": next(iter(assignment_schema_ids)),
        "assignment_schema_version": next(iter(assignment_schema_versions)),
        "assignment_method": next(iter(assignment_methods)),
        "assigned_by": next(iter(assigned_by_values)),
        "assigned_at_utc": next(iter(assigned_at_values)),
    }
    if batch_status == "missing":
        if any(value is not None for value in raw_batch.values()):
            _fail(
                "Missing acquisition-batch identity must not retain batch or "
                f"assignment provenance: {canonical_path}"
            )
        batch = raw_batch
    else:
        batch = {
            "acquisition_batch_id": _nonempty(
                raw_batch["acquisition_batch_id"],
                label="acquisition_batch_id",
            ),
            "acquisition_batch_snapshot_id": _uuid(
                raw_batch["acquisition_batch_snapshot_id"],
                label="acquisition_batch_snapshot_id",
            ),
            "acquisition_batch_schema_id": _nonempty(
                raw_batch["acquisition_batch_schema_id"],
                label="acquisition_batch_schema_id",
            ),
            "acquisition_batch_schema_version": raw_batch[
                "acquisition_batch_schema_version"
            ],
            "assignment_snapshot_id": _uuid(
                raw_batch["assignment_snapshot_id"],
                label="assignment_snapshot_id",
            ),
            "assignment_batch_id": _uuid(
                raw_batch["assignment_batch_id"],
                label="assignment_batch_id",
            ),
            "assignment_revision": raw_batch["assignment_revision"],
            "supersedes_assignment_snapshot_id": _optional_uuid(
                raw_batch["supersedes_assignment_snapshot_id"],
                label="supersedes_assignment_snapshot_id",
            ),
            "assignment_schema_id": _nonempty(
                raw_batch["assignment_schema_id"],
                label="assignment_schema_id",
            ),
            "assignment_schema_version": raw_batch["assignment_schema_version"],
            "assignment_method": _nonempty(
                raw_batch["assignment_method"],
                label="assignment_method",
            ),
            "assigned_by": _nonempty(raw_batch["assigned_by"], label="assigned_by"),
            "assigned_at_utc": _nonempty(
                raw_batch["assigned_at_utc"],
                label="assigned_at_utc",
            ),
        }
        if batch["acquisition_batch_schema_id"] != REGISTRY_IDENTITY_BATCH_SCHEMA_ID:
            _fail("Registry acquisition-batch schema_id is invalid.")
        if batch["assignment_schema_id"] != REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID:
            _fail("Registry acquisition-batch assignment schema_id is invalid.")
        for field in ("acquisition_batch_schema_version", "assignment_schema_version"):
            if type(batch[field]) is not int or batch[field] < 1:
                _fail(f"Registry {field} must be a positive integer.")
        if (
            type(batch["assignment_revision"]) is not int
            or batch["assignment_revision"] < 1
        ):
            _fail("Registry assignment_revision must be a positive integer.")
        if batch["assignment_revision"] == 1:
            if batch["supersedes_assignment_snapshot_id"] is not None:
                _fail("Registry assignment revision 1 must not supersede a snapshot.")
        elif batch["supersedes_assignment_snapshot_id"] is None:
            _fail("Registry corrected batch assignment must supersede a snapshot.")

    body = {
        "dataset_id": next(iter(dataset_ids)),
        "zarr_path": canonical_path,
        "recording_id": recording_id,
        "experimental_unit_id": subject_id,
        "experimental_unit_kind": "subject",
        **batch,
        "acquisition_batch_identity_status": batch_status,
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
        "acquisition_batch_id_source": REGISTRY_IDENTITY_BATCH_SOURCE,
        "acquisition_batch_policy": REGISTRY_IDENTITY_BATCH_POLICY,
        "subject_id_source": REGISTRY_IDENTITY_SUBJECT_SOURCE,
        "subject_cardinality_policy": REGISTRY_IDENTITY_CARDINALITY_POLICY,
        "experimental_unit_policy": REGISTRY_IDENTITY_EXPERIMENTAL_UNIT_POLICY,
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
        "experimental_unit_id",
        "experimental_unit_kind",
        "acquisition_batch_identity_status",
        "subject_id",
    ):
        body[field] = _nonempty(body[field], label=field)
    if body["experimental_unit_kind"] != "subject":
        _fail("registry experimental_unit_kind must be 'subject'.")
    if body["experimental_unit_id"] != body["subject_id"]:
        _fail("registry experimental_unit_id must equal subject_id.")
    status = body["acquisition_batch_identity_status"]
    if status not in REGISTRY_IDENTITY_BATCH_STATUSES:
        _fail("registry acquisition_batch_identity_status is invalid.")
    batch_fields = (
        "acquisition_batch_id",
        "acquisition_batch_snapshot_id",
        "acquisition_batch_schema_id",
        "acquisition_batch_schema_version",
        "assignment_snapshot_id",
        "assignment_batch_id",
        "assignment_revision",
        "supersedes_assignment_snapshot_id",
        "assignment_schema_id",
        "assignment_schema_version",
        "assignment_method",
        "assigned_by",
        "assigned_at_utc",
    )
    if status == "missing":
        if any(body[field] is not None for field in batch_fields):
            _fail("missing acquisition batch must have null batch provenance.")
    else:
        for field in (
            "acquisition_batch_id",
            "acquisition_batch_schema_id",
            "assignment_schema_id",
            "assignment_method",
            "assigned_by",
            "assigned_at_utc",
        ):
            body[field] = _nonempty(body[field], label=field)
        body["acquisition_batch_snapshot_id"] = _uuid(
            body["acquisition_batch_snapshot_id"],
            label="acquisition_batch_snapshot_id",
        )
        body["assignment_snapshot_id"] = _uuid(
            body["assignment_snapshot_id"],
            label="assignment_snapshot_id",
        )
        body["assignment_batch_id"] = _uuid(
            body["assignment_batch_id"],
            label="assignment_batch_id",
        )
        if body["assignment_schema_id"] != REGISTRY_IDENTITY_ASSIGNMENT_SCHEMA_ID:
            _fail("registry assignment_schema_id is invalid.")
        if body["acquisition_batch_schema_id"] != REGISTRY_IDENTITY_BATCH_SCHEMA_ID:
            _fail("registry acquisition_batch_schema_id is invalid.")
        for field in ("acquisition_batch_schema_version", "assignment_schema_version"):
            if type(body[field]) is not int or body[field] < 1:
                _fail(f"registry {field} must be a positive integer.")
        if (
            type(body["assignment_revision"]) is not int
            or body["assignment_revision"] < 1
        ):
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
    if record["acquisition_batch_id_source"] != REGISTRY_IDENTITY_BATCH_SOURCE:
        _fail("registry identity receipt acquisition-batch source is invalid.")
    if record["acquisition_batch_policy"] != REGISTRY_IDENTITY_BATCH_POLICY:
        _fail("registry identity receipt acquisition-batch policy is invalid.")
    if record["subject_id_source"] != REGISTRY_IDENTITY_SUBJECT_SOURCE:
        _fail("registry identity receipt subject source is invalid.")
    if record["subject_cardinality_policy"] != REGISTRY_IDENTITY_CARDINALITY_POLICY:
        _fail("registry identity receipt subject policy is invalid.")
    if record["experimental_unit_policy"] != REGISTRY_IDENTITY_EXPERIMENTAL_UNIT_POLICY:
        _fail("registry identity receipt experimental-unit policy is invalid.")
    registry_path = _canonical_path(record["registry_path"], label="registry_path")
    raw_sources = record["sources"]
    if not isinstance(raw_sources, list):
        _fail("registry identity receipt sources must be an array.")
    sources = [
        dict(validate_registry_identity_source(source)) for source in raw_sources
    ]
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
        "acquisition_batch_id_source": record["acquisition_batch_id_source"],
        "acquisition_batch_policy": record["acquisition_batch_policy"],
        "subject_id_source": record["subject_id_source"],
        "subject_cardinality_policy": record["subject_cardinality_policy"],
        "experimental_unit_policy": record["experimental_unit_policy"],
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
    "REGISTRY_IDENTITY_BATCH_POLICY",
    "REGISTRY_IDENTITY_EXPERIMENTAL_UNIT_POLICY",
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
    "REGISTRY_IDENTITY_BATCH_STATUSES",
    "REGISTRY_IDENTITY_BATCH_FIELD",
    "REGISTRY_IDENTITY_BATCH_SOURCE",
    "REGISTRY_IDENTITY_SUBJECT_SOURCE",
    "RegistryIdentityReceiptError",
    "build_registry_identity_receipt",
    "build_registry_identity_source",
    "registry_identity_sources_by_path",
    "validate_registry_identity_receipt",
    "validate_registry_identity_source",
]
