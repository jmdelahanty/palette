"""Read-only index of exact producer-approved acquisition geometry references.

This catalog is neither an authority nor an acceptance receipt. Its digest
identifies an index, not the truth of its claims: validation reopens the producer
assets and Palette source-pixel authority through the existing native planner.
No selector, candidate, approval, recovery, or registration is written here.

The reference key is recording-independent; each occurrence retains its exact
recording-specific contract and pixel-frame binding. Producer field names and
inner-rim/detection-gate semantics are preserved, including the final acquisition
tolerance. A Citrus runtime application flag is not a gate for this folder
geometry supplier. H5-only and recovered references are outside this profile.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.shared.recording_geometry import RecordingGeometryError
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)

GEOMETRY_REFERENCE_CATALOG_SCHEMA_ID = "palette.recording_geometry_reference_catalog"
GEOMETRY_REFERENCE_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GeometryReferenceSource:
    """An explicit producer folder and its own Palette source-pixel authority.

    ``applicable_at_utc`` checks a declared registration expiration against the
    intended acquisition/use time. None means an inventory-only expiration
    check was not requested, never that this reference is suitable indefinitely.
    No wall-clock default is used for historical recordings.
    """

    recording_root: Path
    source_zarr: Path
    rig_id: str
    canvas_name: str
    arena_id: str
    camera_serial: str
    applicable_at_utc: str | None = None


@dataclass(frozen=True, order=True)
class GeometryReferenceKey:
    rig_id: str
    canvas_name: str
    arena_id: str
    camera_serial: str
    registration_id: str
    registration_sha256: str
    artifact_id: str
    source_observation_sha256: str
    coordinate_profile_id: str
    native_width_px: int
    native_height_px: int
    space_id: str
    pixel_convention: str
    units: str
    origin: str
    positive_x: str
    positive_y: str


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecordingGeometryError(f"{label} must be nonempty exact text.")
    return value


def _utc_time(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordingGeometryError(f"{label} must be an ISO-8601 timestamp.") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise RecordingGeometryError(f"{label} requires an explicit timezone.")
    return result


def _canonical_copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise RecordingGeometryError(
            "Catalog must contain strict finite JSON."
        ) from exc


def _source_record(source: GeometryReferenceSource) -> dict[str, Any]:
    if type(source) is not GeometryReferenceSource:
        raise RecordingGeometryError("An exact GeometryReferenceSource is required.")
    result = asdict(source)
    for name in ("rig_id", "canvas_name", "arena_id", "camera_serial"):
        _text(result[name], name)
    for name in ("recording_root", "source_zarr"):
        value = result[name]
        if not isinstance(value, (str, Path)):
            raise RecordingGeometryError(f"{name} must be a path.")
        result[name] = str(Path(value).expanduser().resolve())
    if source.applicable_at_utc is not None:
        _utc_time(source.applicable_at_utc, "applicable_at_utc")
    return result


def geometry_reference_source_from_record(value: Any) -> GeometryReferenceSource:
    """Parse the closed locator grammar; parsing alone grants no validation."""

    names = {field.name for field in fields(GeometryReferenceSource)}
    if not isinstance(value, Mapping) or set(value) != names:
        raise RecordingGeometryError("Geometry reference source has unexpected fields.")
    source = GeometryReferenceSource(**value)
    canonical = _source_record(source)
    if canonical_json_bytes(canonical) != canonical_json_bytes(value):
        raise RecordingGeometryError(
            "Catalog source paths must be exact absolute paths."
        )
    return GeometryReferenceSource(
        **{
            **canonical,
            "recording_root": Path(canonical["recording_root"]),
            "source_zarr": Path(canonical["source_zarr"]),
        }
    )


def _validity(record: Mapping[str, Any], applicable_at_utc: str | None) -> str:
    expires = record["acquisition_source"]["source_valid_until_utc"]
    deadline = (
        _utc_time(expires, "source_valid_until_utc") if expires is not None else None
    )
    if applicable_at_utc is None:
        return "not_requested"
    applicable = _utc_time(applicable_at_utc, "applicable_at_utc")
    if deadline is not None and applicable >= deadline:
        raise RecordingGeometryError(
            "Registration reference is expired at the declared applicability time."
        )
    return (
        "within_declared_validity" if deadline is not None else "no_declared_expiration"
    )


def _reference_parts(
    record: Mapping[str, Any],
) -> tuple[GeometryReferenceKey, dict[str, Any]]:
    arena = record["arena_binding"]
    acquisition = record["acquisition_source"]
    coordinate = record["coordinate_binding"]
    if (
        acquisition["source_kind"] != "orange_recording_folder"
        or acquisition["producer_contract_linkage_status"] != "producer_native"
        or acquisition["recovery_binding"] is not None
        or acquisition["producer_operator_accepted"] is not True
        or acquisition["materialized_asset_status"] != "complete"
    ):
        raise RecordingGeometryError(
            "Catalog requires complete producer-approved folder geometry."
        )
    registration_sha = _text(acquisition["registration_sha256"], "registration_sha256")
    key = GeometryReferenceKey(
        **arena,
        registration_id=acquisition["registration_id"],
        registration_sha256=registration_sha,
        artifact_id=acquisition["artifact_id"],
        source_observation_sha256=acquisition["source_observation_sha256"],
        coordinate_profile_id=coordinate["profile_id"],
        **{
            name: coordinate[name]
            for name in (
                "native_width_px",
                "native_height_px",
                "space_id",
                "pixel_convention",
                "units",
                "origin",
                "positive_x",
                "positive_y",
            )
        },
    )
    reference = {
        name: record[name] for name in ("physical_inner_rim", "valid_detection_region")
    }
    reference.update(
        {
            name: acquisition[name]
            for name in (
                "producer_operator_accepted",
                "producer_quality_flags",
                "source_valid_until_utc",
            )
        }
    )
    return key, _canonical_copy(reference)


def _inspect_source(
    source: GeometryReferenceSource,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Reuse the concrete producer/coordinate validator, not a copied geometry or
    # receipt implementation. Planning is read-only and does not create a run.
    from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
        plan_producer_native_acquisition_geometry_candidate,
    )

    locator = _source_record(source)
    plan = plan_producer_native_acquisition_geometry_candidate(
        source_zarr=locator["source_zarr"],
        recording_folder=locator["recording_root"],
        camera_serial=locator["camera_serial"],
        arena_id=locator["arena_id"],
    )
    record = _canonical_copy(plan.candidate_record)
    expected_arena = {
        name: locator[name]
        for name in (
            "rig_id",
            "canvas_name",
            "arena_id",
            "camera_serial",
        )
    }
    if record["arena_binding"] != expected_arena:
        raise RecordingGeometryError(
            "Reference source has the wrong exact rig/canvas/arena/camera binding."
        )
    occurrence = {
        "locator": locator,
        "candidate_id": plan.candidate_id,
        "candidate_record_sha256": plan.candidate_record_sha256,
        "acquisition_source": record["acquisition_source"],
        "coordinate_binding": record["coordinate_binding"],
        "validity_check": _validity(record, source.applicable_at_utc),
    }
    return record, occurrence


def _check_identity_conflicts(
    keys: Sequence[GeometryReferenceKey],
) -> None:
    identities: dict[tuple[Any, ...], Any] = {}
    for key in keys:
        claims = (
            (
                ("registration", key.rig_id, key.canvas_name, key.registration_id),
                key.registration_sha256,
            ),
            (
                (
                    "observation",
                    key.rig_id,
                    key.canvas_name,
                    key.arena_id,
                    key.camera_serial,
                    key.artifact_id,
                ),
                key.source_observation_sha256,
            ),
            (
                (
                    "registration_camera",
                    key.rig_id,
                    key.canvas_name,
                    key.arena_id,
                    key.camera_serial,
                    key.registration_sha256,
                ),
                key,
            ),
        )
        for identity, value in claims:
            if identity in identities and identities[identity] != value:
                raise RecordingGeometryError(
                    f"Reference identity conflict: {identity!r}."
                )
            identities[identity] = value


def build_geometry_reference_catalog(
    sources: Sequence[GeometryReferenceSource],
) -> dict[str, Any]:
    """Revalidate explicit sources and build a deterministic, non-authority index."""

    entries: dict[GeometryReferenceKey, dict[str, Any]] = {}
    seen_sources: dict[tuple[str, ...], dict[str, Any]] = {}
    for source in sources:
        locator = _source_record(source)
        occurrence_key = tuple(
            locator[name]
            for name in (
                "recording_root",
                "source_zarr",
                "camera_serial",
                "arena_id",
            )
        )
        previous = seen_sources.get(occurrence_key)
        if previous is not None:
            if canonical_json_bytes(previous) != canonical_json_bytes(locator):
                raise RecordingGeometryError("Reference source descriptor conflict.")
            continue
        record, occurrence = _inspect_source(source)
        key, reference = _reference_parts(record)
        common = {"key": asdict(key), "reference": reference}
        entry = {
            **common,
            "reference_sha256": canonical_json_sha256(common),
            "sources": [],
        }
        if key in entries:
            if canonical_json_bytes(entries[key]["reference"]) != canonical_json_bytes(
                reference
            ):
                raise RecordingGeometryError(
                    "Same exact reference key has conflicting geometry or producer evidence."
                )
        else:
            entries[key] = entry
        entries[key]["sources"].append(occurrence)
        seen_sources[occurrence_key] = locator
    if not entries:
        raise RecordingGeometryError(
            "Catalog requires at least one validated reference source."
        )
    _check_identity_conflicts(tuple(entries))
    for entry in entries.values():
        entry["sources"].sort(key=lambda value: canonical_json_bytes(value["locator"]))
    result = {
        "schema_id": GEOMETRY_REFERENCE_CATALOG_SCHEMA_ID,
        "schema_version": GEOMETRY_REFERENCE_CATALOG_SCHEMA_VERSION,
        "catalog_role": "non_authoritative_index",
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "metadata_read_mode": "unconsolidated_diagnostic",
        "entries": [entries[key] for key in sorted(entries)],
    }
    return {**result, "catalog_sha256": canonical_json_sha256(result)}


def _catalog_sources(value: Mapping[str, Any]) -> list[GeometryReferenceSource]:
    expected = {
        "schema_id",
        "schema_version",
        "catalog_role",
        "digest_algorithm",
        "metadata_read_mode",
        "entries",
        "catalog_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RecordingGeometryError("Catalog has missing or unexpected fields.")
    if (
        value["schema_id"] != GEOMETRY_REFERENCE_CATALOG_SCHEMA_ID
        or type(value["schema_version"]) is not int
        or value["schema_version"] != GEOMETRY_REFERENCE_CATALOG_SCHEMA_VERSION
        or value["catalog_role"] != "non_authoritative_index"
        or value["digest_algorithm"] != CANONICAL_JSON_DIGEST_ALGORITHM
        or value["metadata_read_mode"] != "unconsolidated_diagnostic"
    ):
        raise RecordingGeometryError("Unsupported geometry reference catalog contract.")
    body = {name: item for name, item in value.items() if name != "catalog_sha256"}
    if value["catalog_sha256"] != canonical_json_sha256(body):
        raise RecordingGeometryError("Catalog digest mismatch.")
    if not isinstance(value["entries"], list) or not value["entries"]:
        raise RecordingGeometryError("Catalog requires a nonempty entry list.")
    sources = []
    for entry in value["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "key",
            "reference",
            "reference_sha256",
            "sources",
        }:
            raise RecordingGeometryError(
                "Catalog entry has missing or unexpected fields."
            )
        if not isinstance(entry["sources"], list) or not entry["sources"]:
            raise RecordingGeometryError(
                "Catalog reference must retain its exact sources."
            )
        for occurrence in entry["sources"]:
            if not isinstance(occurrence, Mapping) or "locator" not in occurrence:
                raise RecordingGeometryError("Catalog source lacks an exact locator.")
            sources.append(geometry_reference_source_from_record(occurrence["locator"]))
    return sources


def validate_geometry_reference_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen every producer asset/pixel binding and reject stale or forged indexes.

    The return value is a validated snapshot, not a durable admission token. A
    later consumer must use ``resolve_geometry_reference`` to revalidate its
    exact selected source; changing the index digest never repairs stale data.
    """

    snapshot = _canonical_copy(value)
    sources = _catalog_sources(snapshot)
    rebuilt = build_geometry_reference_catalog(sources)
    if canonical_json_bytes(snapshot) != canonical_json_bytes(rebuilt):
        raise RecordingGeometryError(
            "Catalog is stale or conflicts with its exact live sources."
        )
    return rebuilt


def resolve_geometry_reference(
    value: Mapping[str, Any],
    *,
    key: GeometryReferenceKey,
    source_zarr: str | Path,
    applicable_at_utc: str | None = None,
) -> dict[str, Any]:
    """Return the unchanged candidate record after exact index/source revalidation.

    No nearest registration, newest observation, other recording, or current
    calibration fallback is permitted. The optional applicability time is an
    additional validity check and does not rewrite the indexed historical time.
    """

    if type(key) is not GeometryReferenceKey:
        raise RecordingGeometryError("An exact GeometryReferenceKey is required.")
    # Whole-index validation also detects conflicting identities, not just the
    # requested row. This intentionally favors assurance over catalog caching.
    catalog = validate_geometry_reference_catalog(value)
    matches = [
        entry
        for entry in catalog["entries"]
        if canonical_json_bytes(entry["key"]) == canonical_json_bytes(asdict(key))
    ]
    if len(matches) != 1:
        raise RecordingGeometryError(
            "Catalog has no single exact requested reference key."
        )
    path = str(Path(source_zarr).expanduser().resolve())
    occurrences = [
        row for row in matches[0]["sources"] if row["locator"]["source_zarr"] == path
    ]
    if len(occurrences) != 1:
        raise RecordingGeometryError(
            "Catalog has no single exact recording source for this reference."
        )
    selected = occurrences[0]
    record, current = _inspect_source(
        geometry_reference_source_from_record(selected["locator"])
    )
    if canonical_json_bytes(current) != canonical_json_bytes(selected):
        raise RecordingGeometryError("Catalog source changed during exact resolution.")
    _validity(record, applicable_at_utc)
    return record


__all__ = [
    "GEOMETRY_REFERENCE_CATALOG_SCHEMA_ID",
    "GEOMETRY_REFERENCE_CATALOG_SCHEMA_VERSION",
    "GeometryReferenceKey",
    "GeometryReferenceSource",
    "build_geometry_reference_catalog",
    "geometry_reference_source_from_record",
    "resolve_geometry_reference",
    "validate_geometry_reference_catalog",
]
