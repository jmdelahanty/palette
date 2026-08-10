"""Exact source authority for cross-recording chaser Arrow exports.

The authority set is an immutable invocation input.  It binds every source
archive to one verified chaser-distance base run and zero or more explicit,
self-digested component handles.  It grants no selector authority and never
permits ``latest`` or name-based component discovery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)


CHASER_EXPORT_AUTHORITY_SCHEMA_ID = "palette.chaser_export_authority_set.v1"
CHASER_EXPORT_AUTHORITY_SCHEMA_VERSION = 1
CHASER_EXPORT_AUTHORITY_RESOLUTION_POLICY = (
    "exact_base_and_component_handles_no_selector_fallback"
)
CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_ID = (
    "palette.analytics_export.chaser_authority_receipt"
)
CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_VERSION = 1

EPOCH_BEHAVIOR_FAMILY = "epoch_behavior_summary"
QUADRANT_OCCUPANCY_FAMILY = "chaser_quadrant_occupancy"
NEAR_FIELD_OCCUPANCY_FAMILY = "chaser_near_field_occupancy"
EGOCENTRIC_BEARING_FAMILY = "egocentric_bearing"
CHASER_EXPORT_COMPONENT_FAMILIES = frozenset(
    {
        EPOCH_BEHAVIOR_FAMILY,
        QUADRANT_OCCUPANCY_FAMILY,
        NEAR_FIELD_OCCUPANCY_FAMILY,
        EGOCENTRIC_BEARING_FAMILY,
    }
)

_ROOT_BODY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "resolution_policy",
        "sources",
    }
)
_ROOT_FIELDS = _ROOT_BODY_FIELDS | {"record_sha256"}
_SOURCE_BODY_FIELDS = frozenset(
    {
        "zarr_path",
        "recording_id",
        "base_run_name",
        "base_run_path",
        "base_publication_seal_sha256",
        "component_handles",
    }
)
_SOURCE_FIELDS = _SOURCE_BODY_FIELDS | {"record_sha256"}
_RECEIPT_BODY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "path",
        "file_sha256",
        "authority_record_sha256",
        "record",
        "resolved_sources",
    }
)
_RECEIPT_FIELDS = _RECEIPT_BODY_FIELDS | {"payload_sha256"}
_RESOLVED_SOURCE_FIELDS = frozenset(
    {
        "source_record_sha256",
        "base_run_name",
        "base_run_path",
        "base_publication_seal_sha256",
        "component_handles",
    }
)
_RESOLVED_COMPONENT_FIELDS = frozenset(
    {
        "component_path",
        "component_manifest_sha256",
        "handle_record_sha256",
    }
)


class ChaserExportAuthorityError(ValueError):
    """Raised when an exact chaser export authority is absent or malformed."""


def _fail(message: str) -> None:
    raise ChaserExportAuthorityError(message)


def _exact_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object.")
    record = {str(key): item for key, item in value.items()}
    if set(record) != fields:
        _fail(
            f"{label} must contain exactly {sorted(fields)!r}; "
            f"found {sorted(record)!r}."
        )
    return record


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail(f"{label} must be one non-empty trimmed string.")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _canonical_zarr_path(value: Any, *, label: str) -> str:
    text = _nonempty_string(value, label=label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        _fail(f"{label} must be an absolute path.")
    canonical = str(path.resolve())
    if text != canonical:
        _fail(f"{label} must already be canonical: expected {canonical!r}.")
    return canonical


def _base_run_path(run_name: str) -> str:
    return f"analysis/chaser_distance_runs/{run_name}"


def _validate_component_handles(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        _fail("chaser authority component_handles must be an object.")
    handles = {str(key): item for key, item in value.items()}
    unknown = sorted(set(handles) - CHASER_EXPORT_COMPONENT_FAMILIES)
    if unknown:
        _fail(f"chaser authority has unknown component families: {unknown!r}.")
    normalized: dict[str, dict[str, Any]] = {}
    for family in sorted(handles):
        handle = handles[family]
        if not isinstance(handle, Mapping):
            _fail(f"component handle {family!r} must be an object.")
        plain = {str(key): item for key, item in handle.items()}
        if plain.get("component_family") != family:
            _fail(f"component handle {family!r} belongs to a different family.")
        _sha256(plain.get("record_sha256"), label=f"{family} handle record_sha256")
        normalized[family] = plain
    return normalized


def build_chaser_export_source_authority(
    *,
    zarr_path: str | Path,
    recording_id: str,
    base_run_name: str,
    base_publication_seal_sha256: str,
    component_handles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one strict, self-digested source entry."""

    canonical_path = str(Path(zarr_path).expanduser().resolve())
    run_name = _nonempty_string(base_run_name, label="base_run_name")
    if "/" in run_name or run_name in {".", "..", "latest"}:
        _fail("base_run_name must be one explicit child name, never latest or a path.")
    body = {
        "zarr_path": canonical_path,
        "recording_id": _nonempty_string(recording_id, label="recording_id"),
        "base_run_name": run_name,
        "base_run_path": _base_run_path(run_name),
        "base_publication_seal_sha256": _sha256(
            base_publication_seal_sha256,
            label="base_publication_seal_sha256",
        ),
        "component_handles": _validate_component_handles(component_handles),
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def validate_chaser_export_source_authority(value: Any) -> Mapping[str, Any]:
    """Deeply validate one source entry without opening its archive."""

    record = _exact_mapping(value, _SOURCE_FIELDS, label="chaser authority source")
    body = {key: record[key] for key in _SOURCE_BODY_FIELDS}
    path = _canonical_zarr_path(body["zarr_path"], label="source zarr_path")
    recording_id = _nonempty_string(body["recording_id"], label="recording_id")
    run_name = _nonempty_string(body["base_run_name"], label="base_run_name")
    if "/" in run_name or run_name in {".", "..", "latest"}:
        _fail("base_run_name must be one explicit child name, never latest or a path.")
    expected_run_path = _base_run_path(run_name)
    if body["base_run_path"] != expected_run_path:
        _fail("base_run_path does not match base_run_name.")
    seal_sha256 = _sha256(
        body["base_publication_seal_sha256"],
        label="base_publication_seal_sha256",
    )
    handles = _validate_component_handles(body["component_handles"])
    normalized_body = {
        "zarr_path": path,
        "recording_id": recording_id,
        "base_run_name": run_name,
        "base_run_path": expected_run_path,
        "base_publication_seal_sha256": seal_sha256,
        "component_handles": handles,
    }
    digest = _sha256(record["record_sha256"], label="source record_sha256")
    if canonical_json_sha256(normalized_body) != digest:
        _fail("chaser authority source record digest mismatch.")
    return MappingProxyType({**normalized_body, "record_sha256": digest})


def build_chaser_export_authority_set(
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one sorted, self-digested authority set."""

    normalized = [dict(validate_chaser_export_source_authority(source)) for source in sources]
    normalized.sort(key=lambda source: source["zarr_path"])
    body = {
        "schema_id": CHASER_EXPORT_AUTHORITY_SCHEMA_ID,
        "schema_version": CHASER_EXPORT_AUTHORITY_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "resolution_policy": CHASER_EXPORT_AUTHORITY_RESOLUTION_POLICY,
        "sources": normalized,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def validate_chaser_export_authority_set(value: Any) -> Mapping[str, Any]:
    """Deeply validate one authority set and its exact source inventory."""

    record = _exact_mapping(value, _ROOT_FIELDS, label="chaser export authority set")
    if record["schema_id"] != CHASER_EXPORT_AUTHORITY_SCHEMA_ID:
        _fail("chaser export authority set has an incompatible schema_id.")
    if record["schema_version"] != CHASER_EXPORT_AUTHORITY_SCHEMA_VERSION:
        _fail("chaser export authority set has an incompatible schema_version.")
    if record["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM:
        _fail("chaser export authority set has an incompatible digest algorithm.")
    if record["resolution_policy"] != CHASER_EXPORT_AUTHORITY_RESOLUTION_POLICY:
        _fail("chaser export authority set has an incompatible resolution policy.")
    raw_sources = record["sources"]
    if not isinstance(raw_sources, list):
        _fail("chaser export authority sources must be an array.")
    sources = [dict(validate_chaser_export_source_authority(source)) for source in raw_sources]
    paths = [source["zarr_path"] for source in sources]
    if paths != sorted(paths):
        _fail("chaser export authority sources must be sorted by zarr_path.")
    if len(paths) != len(set(paths)):
        _fail("chaser export authority source zarr_path values must be unique.")
    recording_ids = [source["recording_id"] for source in sources]
    if len(recording_ids) != len(set(recording_ids)):
        _fail("chaser export authority recording_id values must be unique.")
    body = {
        "schema_id": record["schema_id"],
        "schema_version": record["schema_version"],
        "digest_algorithm": record["digest_algorithm"],
        "resolution_policy": record["resolution_policy"],
        "sources": sources,
    }
    digest = _sha256(record["record_sha256"], label="authority-set record_sha256")
    if canonical_json_sha256(body) != digest:
        _fail("chaser export authority-set record digest mismatch.")
    return MappingProxyType({**body, "record_sha256": digest})


@dataclass(frozen=True)
class LoadedChaserExportAuthoritySet:
    path: Path
    file_sha256: str
    record: Mapping[str, Any]
    sources_by_path: Mapping[str, Mapping[str, Any]]


def load_chaser_export_authority_set(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
) -> LoadedChaserExportAuthoritySet:
    """Load strict JSON, verify its file digest, then validate its record."""

    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None:
        expected = _sha256(expected_file_sha256, label="authority file SHA-256")
        if file_sha256 != expected:
            _fail("chaser export authority file SHA-256 mismatch.")
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: _fail(
            f"chaser export authority contains non-finite JSON token {token!r}."
        ))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"chaser export authority is not strict UTF-8 JSON: {exc}.")
    record = validate_chaser_export_authority_set(value)
    sources_by_path = MappingProxyType(
        {source["zarr_path"]: source for source in record["sources"]}
    )
    return LoadedChaserExportAuthoritySet(
        path=source,
        file_sha256=file_sha256,
        record=record,
        sources_by_path=sources_by_path,
    )


def write_chaser_export_authority_set(
    path: str | Path,
    authority_set: Mapping[str, Any],
) -> Path:
    """Write one already-valid authority set as canonical strict JSON."""

    record = validate_chaser_export_authority_set(authority_set)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(dict(record)) + b"\n")
    return destination


def build_chaser_export_authority_receipt(
    authority: LoadedChaserExportAuthoritySet,
    resolved_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the input authority bytes to the exact sources actually opened."""

    body = {
        "schema_id": CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_ID,
        "schema_version": CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_VERSION,
        "path": str(authority.path),
        "file_sha256": authority.file_sha256,
        "authority_record_sha256": authority.record["record_sha256"],
        "record": dict(authority.record),
        "resolved_sources": {
            str(path): dict(binding)
            for path, binding in sorted(resolved_sources.items())
        },
    }
    validate_chaser_export_authority_receipt(
        {**body, "payload_sha256": canonical_json_sha256(body)}
    )
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def validate_chaser_export_authority_receipt(
    value: object,
    *,
    expected_zarr_paths: Sequence[str | Path] | None = None,
) -> Mapping[str, Any]:
    """Deeply validate a persisted chaser-authority publication receipt."""

    receipt = _exact_mapping(
        value,
        _RECEIPT_FIELDS,
        label="chaser export authority receipt",
    )
    if receipt["schema_id"] != CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_ID:
        _fail("chaser export authority receipt has an incompatible schema_id.")
    if receipt["schema_version"] != CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_VERSION:
        _fail("chaser export authority receipt has an incompatible schema_version.")
    path = _canonical_zarr_path(receipt["path"], label="authority receipt path")
    file_sha256 = _sha256(receipt["file_sha256"], label="authority file_sha256")
    authority_record = validate_chaser_export_authority_set(receipt["record"])
    authority_digest = _sha256(
        receipt["authority_record_sha256"],
        label="authority_record_sha256",
    )
    if authority_digest != authority_record["record_sha256"]:
        _fail("authority receipt record digest does not match its embedded record.")
    raw_resolved = receipt["resolved_sources"]
    if not isinstance(raw_resolved, Mapping):
        _fail("authority receipt resolved_sources must be an object.")
    resolved = {str(key): item for key, item in raw_resolved.items()}
    authority_sources = {
        source["zarr_path"]: source for source in authority_record["sources"]
    }
    if set(resolved) != set(authority_sources):
        _fail("authority receipt resolved-source set differs from its authority set.")
    if expected_zarr_paths is not None:
        expected = {
            str(Path(item).expanduser().resolve()) for item in expected_zarr_paths
        }
        if set(resolved) != expected:
            _fail("authority receipt source set differs from export source_zarrs.")
    normalized_resolved: dict[str, dict[str, Any]] = {}
    for source_path, raw_binding in sorted(resolved.items()):
        canonical_path = _canonical_zarr_path(
            source_path,
            label="resolved authority source path",
        )
        binding = _exact_mapping(
            raw_binding,
            _RESOLVED_SOURCE_FIELDS,
            label=f"resolved authority source {source_path}",
        )
        source = authority_sources[canonical_path]
        if (
            binding["source_record_sha256"] != source["record_sha256"]
            or binding["base_run_name"] != source["base_run_name"]
            or binding["base_run_path"] != source["base_run_path"]
            or binding["base_publication_seal_sha256"]
            != source["base_publication_seal_sha256"]
        ):
            _fail(f"resolved authority source {source_path!r} differs from authority.")
        raw_components = binding["component_handles"]
        if not isinstance(raw_components, Mapping) or set(raw_components) != set(
            source["component_handles"]
        ):
            _fail(f"resolved authority source {source_path!r} has invalid components.")
        components: dict[str, dict[str, str]] = {}
        for family, raw_component in sorted(raw_components.items()):
            component = _exact_mapping(
                raw_component,
                _RESOLVED_COMPONENT_FIELDS,
                label=f"resolved component {family}",
            )
            source_handle = source["component_handles"][family]
            if (
                component["component_path"] != source_handle["component_path"]
                or component["component_manifest_sha256"]
                != source_handle["component_manifest_sha256"]
                or component["handle_record_sha256"] != source_handle["record_sha256"]
            ):
                _fail(f"resolved component {family!r} differs from authority handle.")
            components[str(family)] = {
                key: _nonempty_string(component[key], label=f"{family} {key}")
                for key in _RESOLVED_COMPONENT_FIELDS
            }
        normalized_resolved[canonical_path] = {
            "source_record_sha256": _sha256(
                binding["source_record_sha256"],
                label="source_record_sha256",
            ),
            "base_run_name": _nonempty_string(
                binding["base_run_name"], label="base_run_name"
            ),
            "base_run_path": _nonempty_string(
                binding["base_run_path"], label="base_run_path"
            ),
            "base_publication_seal_sha256": _sha256(
                binding["base_publication_seal_sha256"],
                label="base_publication_seal_sha256",
            ),
            "component_handles": components,
        }
    body = {
        "schema_id": receipt["schema_id"],
        "schema_version": receipt["schema_version"],
        "path": path,
        "file_sha256": file_sha256,
        "authority_record_sha256": authority_digest,
        "record": dict(authority_record),
        "resolved_sources": normalized_resolved,
    }
    digest = _sha256(receipt["payload_sha256"], label="receipt payload_sha256")
    if canonical_json_sha256(body) != digest:
        _fail("chaser export authority receipt payload digest mismatch.")
    return MappingProxyType({**body, "payload_sha256": digest})


__all__ = [
    "CHASER_EXPORT_AUTHORITY_RESOLUTION_POLICY",
    "CHASER_EXPORT_AUTHORITY_SCHEMA_ID",
    "CHASER_EXPORT_AUTHORITY_SCHEMA_VERSION",
    "CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_ID",
    "CHASER_EXPORT_AUTHORITY_RECEIPT_SCHEMA_VERSION",
    "CHASER_EXPORT_COMPONENT_FAMILIES",
    "EGOCENTRIC_BEARING_FAMILY",
    "EPOCH_BEHAVIOR_FAMILY",
    "ChaserExportAuthorityError",
    "LoadedChaserExportAuthoritySet",
    "NEAR_FIELD_OCCUPANCY_FAMILY",
    "QUADRANT_OCCUPANCY_FAMILY",
    "build_chaser_export_authority_set",
    "build_chaser_export_authority_receipt",
    "build_chaser_export_source_authority",
    "load_chaser_export_authority_set",
    "validate_chaser_export_authority_set",
    "validate_chaser_export_authority_receipt",
    "validate_chaser_export_source_authority",
    "write_chaser_export_authority_set",
]
