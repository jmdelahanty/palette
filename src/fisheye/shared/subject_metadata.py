"""Versioned acquisition subject-metadata authority."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import h5py

from .import_source_fingerprint import optional_source_stat_fingerprint_attrs
from .json_safety import json_attr_safe_mapping, strict_json_dumps
from .run_provenance import build_writer_run_provenance
from .zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
    resolve_latest_complete_run_group,
)


SUBJECT_METADATA_SCHEMA_ID = "palette.subject_metadata.v1"
SUBJECT_METADATA_SCHEMA_VERSION = 1
SUBJECT_METADATA_RUNS_PATH = "analysis/subject_metadata_runs"
SUBJECT_METADATA_RECORD_ATTR = "subject_metadata_record"
SUBJECT_METADATA_SHA256_ATTR = "subject_metadata_sha256"


class SubjectMetadataError(ValueError):
    """Base error for invalid subject-metadata authority."""


class MissingSubjectMetadataError(SubjectMetadataError):
    """Raised only when no modern or permitted legacy authority exists."""


@dataclass(frozen=True)
class ResolvedSubjectMetadata:
    metadata: Mapping[str, Any]
    subject_ids: tuple[str, ...]
    subject_identity_kind: str
    subject_identity_source_field: str
    record: Mapping[str, Any]
    record_sha256: str
    group_path: str
    run_name: str | None
    legacy: bool


def _explicit_subject_ids(metadata: Mapping[str, Any]) -> tuple[list[str], str]:
    raw_ids = metadata.get("subject_ids") or metadata.get("fish_ids")
    if isinstance(raw_ids, (list, tuple)):
        ids = list(
            dict.fromkeys(str(value).strip() for value in raw_ids if str(value).strip())
        )
        source_field = "subject_ids" if metadata.get("subject_ids") is not None else "fish_ids"
        return ids, source_field
    fish_id = str(metadata.get("fish_id") or "").strip()
    return ([fish_id] if fish_id else []), ("fish_id" if fish_id else "none")


def _identity_kind(subject_ids: list[str]) -> str:
    if not subject_ids:
        return "none"
    try:
        for subject_id in subject_ids:
            UUID(subject_id)
    except ValueError:
        return "opaque"
    return "uuid"


def subject_metadata_sha256(record: Mapping[str, Any]) -> str:
    return sha256(strict_json_dumps(record).encode("utf-8")).hexdigest()


def normalize_subject_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize source attrs without conflating source population and setup count."""

    canonical = json_attr_safe_mapping(metadata)
    if (
        canonical.get("source_dish_population_count") is None
        and canonical.get("fish_count") is not None
    ):
        canonical["source_dish_population_count"] = canonical["fish_count"]
    return canonical


def read_h5_subject_metadata(h5_path: str | Path) -> dict[str, Any]:
    """Read acquisition subject attrs without inventing absent values."""

    with h5py.File(Path(h5_path), "r") as h5:
        if "/subject_metadata" not in h5:
            return {}
        node = h5["/subject_metadata"]
        metadata = json_attr_safe_mapping(dict(node.attrs))
    return normalize_subject_metadata(metadata)


def build_subject_metadata_record(metadata: Mapping[str, Any]) -> dict[str, Any]:
    canonical = normalize_subject_metadata(metadata)
    subject_ids, source_field = _explicit_subject_ids(canonical)
    record = {
        "schema_id": SUBJECT_METADATA_SCHEMA_ID,
        "schema_version": SUBJECT_METADATA_SCHEMA_VERSION,
        "subject_metadata": canonical,
        "subject_ids": subject_ids,
        "subject_identity_kind": _identity_kind(subject_ids),
        "subject_identity_source_field": source_field,
    }
    return record


def _validate_record(record: Mapping[str, Any], digest: str | None = None) -> dict[str, Any]:
    canonical = json_attr_safe_mapping(record)
    if canonical.get("schema_id") != SUBJECT_METADATA_SCHEMA_ID:
        raise SubjectMetadataError("Subject metadata has an unsupported schema_id")
    if canonical.get("schema_version") != SUBJECT_METADATA_SCHEMA_VERSION:
        raise SubjectMetadataError("Subject metadata has an unsupported schema_version")
    metadata = canonical.get("subject_metadata")
    if not isinstance(metadata, dict):
        raise SubjectMetadataError("Subject metadata record has no metadata mapping")
    expected_ids, expected_source = _explicit_subject_ids(metadata)
    if canonical.get("subject_ids") != expected_ids:
        raise SubjectMetadataError("Normalized subject_ids disagree with source metadata")
    if canonical.get("subject_identity_source_field") != expected_source:
        raise SubjectMetadataError("Subject identity source field is inconsistent")
    if canonical.get("subject_identity_kind") != _identity_kind(expected_ids):
        raise SubjectMetadataError("Subject identity kind is inconsistent")
    actual = subject_metadata_sha256(canonical)
    if digest is not None and str(digest) != actual:
        raise SubjectMetadataError(
            f"Subject metadata digest mismatch: stored={digest!r}, computed={actual!r}"
        )
    return canonical


def _resolved(
    record: Mapping[str, Any],
    *,
    digest: str,
    group_path: str,
    run_name: str | None,
    legacy: bool,
) -> ResolvedSubjectMetadata:
    return ResolvedSubjectMetadata(
        metadata=dict(record["subject_metadata"]),
        subject_ids=tuple(str(value) for value in record["subject_ids"]),
        subject_identity_kind=str(record["subject_identity_kind"]),
        subject_identity_source_field=str(record["subject_identity_source_field"]),
        record=dict(record),
        record_sha256=digest,
        group_path=group_path,
        run_name=run_name,
        legacy=legacy,
    )


def publish_subject_metadata(
    root: Any,
    metadata: Mapping[str, Any],
    *,
    source_h5_path: str | Path | None = None,
) -> ResolvedSubjectMetadata:
    """Idempotently publish and select an immutable subject snapshot."""

    record = _validate_record(build_subject_metadata_record(metadata))
    digest = subject_metadata_sha256(record)
    run_name = f"subject_metadata_{digest[:16]}"
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "subject_metadata_runs")
    if run_name in parent:
        existing = parent[run_name]
        existing_record = existing.attrs.get(SUBJECT_METADATA_RECORD_ATTR)
        if not isinstance(existing_record, Mapping):
            raise SubjectMetadataError(f"Existing subject run {run_name!r} has no record")
        _validate_record(
            existing_record,
            str(existing.attrs.get(SUBJECT_METADATA_SHA256_ATTR) or ""),
        )
        if not is_run_complete_in_parent(parent, existing, legacy_default=False) or not is_run_selector_eligible(existing):
            raise SubjectMetadataError(
                f"Existing subject run {run_name!r} is not complete and selector eligible"
            )
        parent.attrs["latest_complete"] = run_name
        parent.attrs["latest"] = run_name
        return resolve_subject_metadata(root, allow_legacy=False)

    run = parent.create_group(run_name)
    mark_run_started(run, run_name=run_name, stage="subject_metadata")
    note_pending_latest(parent, run_name)
    run.attrs["stage_selector_eligible"] = False
    run.attrs["schema_id"] = SUBJECT_METADATA_SCHEMA_ID
    run.attrs["schema_version"] = SUBJECT_METADATA_SCHEMA_VERSION
    run.attrs[SUBJECT_METADATA_RECORD_ATTR] = record
    run.attrs[SUBJECT_METADATA_SHA256_ATTR] = digest
    run.attrs["subject_ids"] = record["subject_ids"]
    run.attrs["subject_identity_kind"] = record["subject_identity_kind"]
    run.attrs["subject_identity_source_field"] = record["subject_identity_source_field"]
    run.attrs["immutable"] = True
    _validate_record(
        run.attrs[SUBJECT_METADATA_RECORD_ATTR],
        str(run.attrs[SUBJECT_METADATA_SHA256_ATTR]),
    )
    run.attrs["stage_selector_eligible"] = True
    fingerprint = None
    if source_h5_path is not None:
        fingerprint = optional_source_stat_fingerprint_attrs(
            source_h5_path,
            attr_prefix="source_h5",
        ).get("source_h5_fingerprint")
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command="import_recording_analysis:publish_subject_metadata",
            params={"schema_id": SUBJECT_METADATA_SCHEMA_ID, "record_sha256": digest},
            input_run_ids={},
            input_artifacts=[
                {
                    "kind": "source_h5",
                    "path": str(source_h5_path) if source_h5_path is not None else None,
                    "stat_fingerprint": fingerprint,
                }
            ],
        ),
    )
    return resolve_subject_metadata(root, allow_legacy=False)


def resolve_subject_metadata(
    root: Any,
    *,
    allow_legacy: bool = True,
) -> ResolvedSubjectMetadata:
    analysis = root.get("analysis")
    parent = analysis.get("subject_metadata_runs") if analysis is not None else None
    if parent is not None:
        run_name, run = resolve_latest_complete_run_group(parent, legacy_default=False)
        if run_name is None or run is None:
            raise SubjectMetadataError(
                f"{SUBJECT_METADATA_RUNS_PATH} exists but has no selected complete run"
            )
        raw = run.attrs.get(SUBJECT_METADATA_RECORD_ATTR)
        if not isinstance(raw, Mapping):
            raise SubjectMetadataError(f"{SUBJECT_METADATA_RUNS_PATH}/{run_name} has no record")
        digest = str(run.attrs.get(SUBJECT_METADATA_SHA256_ATTR) or "")
        record = _validate_record(raw, digest)
        return _resolved(
            record,
            digest=digest,
            group_path=f"{SUBJECT_METADATA_RUNS_PATH}/{run_name}",
            run_name=run_name,
            legacy=False,
        )

    if not allow_legacy:
        raise MissingSubjectMetadataError(f"Missing canonical {SUBJECT_METADATA_RUNS_PATH}")
    singleton = root.get("analysis/subject_metadata")
    raw_metadata = singleton.attrs.get("subject_metadata") if singleton is not None else None
    group_path = "analysis/subject_metadata"
    if not isinstance(raw_metadata, Mapping):
        legacy = root.get("analysis_metadata")
        raw_metadata = legacy.attrs.get("subject_metadata") if legacy is not None else None
        group_path = "analysis_metadata@subject_metadata"
    if isinstance(raw_metadata, str):
        import json

        try:
            raw_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            raw_metadata = None
    if not isinstance(raw_metadata, Mapping):
        raise MissingSubjectMetadataError("Missing subject metadata")
    record = build_subject_metadata_record(raw_metadata)
    digest = subject_metadata_sha256(record)
    return _resolved(
        record,
        digest=digest,
        group_path=group_path,
        run_name=None,
        legacy=True,
    )


__all__ = [
    "MissingSubjectMetadataError",
    "ResolvedSubjectMetadata",
    "SUBJECT_METADATA_RECORD_ATTR",
    "SUBJECT_METADATA_RUNS_PATH",
    "SUBJECT_METADATA_SCHEMA_ID",
    "SUBJECT_METADATA_SCHEMA_VERSION",
    "SUBJECT_METADATA_SHA256_ATTR",
    "SubjectMetadataError",
    "build_subject_metadata_record",
    "normalize_subject_metadata",
    "publish_subject_metadata",
    "read_h5_subject_metadata",
    "resolve_subject_metadata",
    "subject_metadata_sha256",
]
