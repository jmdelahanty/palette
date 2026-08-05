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


__all__ = [
    "CHASER_EXPORT_AUTHORITY_RESOLUTION_POLICY",
    "CHASER_EXPORT_AUTHORITY_SCHEMA_ID",
    "CHASER_EXPORT_AUTHORITY_SCHEMA_VERSION",
    "CHASER_EXPORT_COMPONENT_FAMILIES",
    "EGOCENTRIC_BEARING_FAMILY",
    "EPOCH_BEHAVIOR_FAMILY",
    "ChaserExportAuthorityError",
    "LoadedChaserExportAuthoritySet",
    "NEAR_FIELD_OCCUPANCY_FAMILY",
    "QUADRANT_OCCUPANCY_FAMILY",
    "build_chaser_export_authority_set",
    "build_chaser_export_source_authority",
    "load_chaser_export_authority_set",
    "validate_chaser_export_authority_set",
    "validate_chaser_export_source_authority",
    "write_chaser_export_authority_set",
]
