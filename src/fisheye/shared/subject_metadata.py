"""Versioned acquisition subject-metadata authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID

import h5py

from .import_source_fingerprint import optional_source_stat_fingerprint_attrs
from .json_safety import json_attr_safe_mapping, strict_json_dumps
from .type_conversions import normalize_attr
from .run_provenance import build_writer_run_provenance
from .subject_fields import TRANSLATORS, canonical_subject_fields
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
# v2 (docs/design/2026-10-06-canonical-subject-fields): v1's fields plus the
# canonical ``subject`` fields, translated once at publish by the named
# ``subject_translator``; ``subject_metadata`` keeps the raw source values.
SUBJECT_METADATA_V2_SCHEMA_ID = "palette.subject_metadata.v2"
SUBJECT_METADATA_V2_SCHEMA_VERSION = 2
_SCHEMAS = {
    SUBJECT_METADATA_SCHEMA_ID: SUBJECT_METADATA_SCHEMA_VERSION,
    SUBJECT_METADATA_V2_SCHEMA_ID: SUBJECT_METADATA_V2_SCHEMA_VERSION,
}
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
    # Canonical fields (fisheye.shared.subject_fields), translated from
    # ``metadata`` on read; consumers should read these, not ``metadata``.
    subject: Mapping[str, Any] = field(default_factory=dict)


def _explicit_subject_ids(
    metadata: Mapping[str, Any], *, legacy_rule: bool = False
) -> tuple[list[str], str]:
    """Citrus-local subject ids, never MetaZebrobot fish ids.

    ``subject_id`` is the identity field going forward; ``fish_id`` is the
    legacy Citrus spelling of the same local id (agent-contracts PR 52), and a
    source carrying both is refused. ``legacy_rule`` reproduces the earlier
    derivation, which ignored a singular ``subject_id``, so records published
    before this rule and legacy singleton reads stay byte-for-byte unchanged.
    """

    if not legacy_rule and _present(metadata.get("subject_id")) and _present(
        metadata.get("fish_id")
    ):
        raise SubjectMetadataError(
            "Subject metadata carries both subject_id and legacy fish_id"
        )
    raw_ids = metadata.get("subject_ids") or metadata.get("fish_ids")
    if isinstance(raw_ids, (list, tuple)):
        ids = list(
            dict.fromkeys(str(value).strip() for value in raw_ids if str(value).strip())
        )
        source_field = "subject_ids" if metadata.get("subject_ids") is not None else "fish_ids"
        return ids, source_field
    subject_id = "" if legacy_rule else str(metadata.get("subject_id") or "").strip()
    if subject_id:
        return [subject_id], "subject_id"
    fish_id = str(metadata.get("fish_id") or "").strip()
    return ([fish_id] if fish_id else []), ("fish_id" if fish_id else "none")


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def explicit_subject_ids(metadata: Mapping[str, Any]) -> list[str]:
    """Citrus-local subject ids from a raw subject mapping (the publish rule)."""

    return _explicit_subject_ids(metadata)[0]


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


def build_subject_metadata_record(
    metadata: Mapping[str, Any],
    *,
    legacy_rule: bool = False,
    translator: str | None = None,
) -> dict[str, Any]:
    """A v1 record, or with ``translator`` a v2 record carrying canonical fields."""

    canonical = normalize_subject_metadata(metadata)
    subject_ids, source_field = _explicit_subject_ids(canonical, legacy_rule=legacy_rule)
    record = {
        "schema_id": SUBJECT_METADATA_SCHEMA_ID,
        "schema_version": SUBJECT_METADATA_SCHEMA_VERSION,
        "subject_metadata": canonical,
        "subject_ids": subject_ids,
        "subject_identity_kind": _identity_kind(subject_ids),
        "subject_identity_source_field": source_field,
    }
    if translator is not None:
        if translator not in TRANSLATORS:
            raise SubjectMetadataError(f"Unknown subject translator {translator!r}")
        record.update(
            schema_id=SUBJECT_METADATA_V2_SCHEMA_ID,
            schema_version=SUBJECT_METADATA_V2_SCHEMA_VERSION,
            subject_translator=translator,
            subject=json_attr_safe_mapping(TRANSLATORS[translator](canonical, subject_ids)),
        )
    return record


def _stored_identity(
    record: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[list[str], str]:
    """Current identity rule, or the earlier one for an already-published record."""

    legacy = _explicit_subject_ids(metadata, legacy_rule=True)
    if (record.get("subject_ids"), record.get("subject_identity_source_field")) == legacy:
        return legacy
    return _explicit_subject_ids(metadata)


def _validate_record(record: Mapping[str, Any], digest: str | None = None) -> dict[str, Any]:
    canonical = json_attr_safe_mapping(record)
    schema_id = canonical.get("schema_id")
    if schema_id not in _SCHEMAS:
        raise SubjectMetadataError("Subject metadata has an unsupported schema_id")
    if canonical.get("schema_version") != _SCHEMAS[schema_id]:
        raise SubjectMetadataError("Subject metadata has an unsupported schema_version")
    if schema_id == SUBJECT_METADATA_V2_SCHEMA_ID:
        # The canonical fields are fixed at publish and covered by the record
        # digest; they are not recomputed, so later translator changes never
        # invalidate a stored record.
        if not isinstance(canonical.get("subject"), dict):
            raise SubjectMetadataError("Subject metadata v2 record has no subject fields")
        if not isinstance(canonical.get("subject_translator"), str):
            raise SubjectMetadataError("Subject metadata v2 record names no translator")
    metadata = canonical.get("subject_metadata")
    if not isinstance(metadata, dict):
        raise SubjectMetadataError("Subject metadata record has no metadata mapping")
    expected_ids, expected_source = _stored_identity(canonical, metadata)
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
    subject_ids = tuple(str(value) for value in record["subject_ids"])
    return ResolvedSubjectMetadata(
        metadata=dict(record["subject_metadata"]),
        subject_ids=subject_ids,
        subject=(
            dict(record["subject"])
            if record.get("schema_id") == SUBJECT_METADATA_V2_SCHEMA_ID
            else canonical_subject_fields(record["subject_metadata"], subject_ids)
        ),
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
    source_artifact: Mapping[str, Any] | None = None,
    provenance_command: str | None = None,
    provenance_params: Mapping[str, Any] | None = None,
    provenance_input_artifacts: Sequence[Mapping[str, Any]] | None = None,
    translator: str | None = None,
) -> ResolvedSubjectMetadata:
    """Idempotently publish and select an immutable subject snapshot.

    With ``translator`` (a ``fisheye.shared.subject_fields.TRANSLATORS`` name)
    the record is v2 and stores the canonical fields; without it, v1.
    """

    if source_h5_path is not None and source_artifact is not None:
        raise SubjectMetadataError(
            "Provide source_h5_path or source_artifact, not both"
        )

    record = _validate_record(build_subject_metadata_record(metadata, translator=translator))
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
    run.attrs["schema_id"] = record["schema_id"]
    run.attrs["schema_version"] = record["schema_version"]
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
    input_artifacts = list(provenance_input_artifacts or [])
    if source_h5_path is not None:
        input_artifacts.append(
            {
                "kind": "source_h5",
                "path": str(source_h5_path),
                "stat_fingerprint": fingerprint,
            }
        )
    if source_artifact is not None:
        input_artifacts.append(json_attr_safe_mapping(source_artifact))
    params = {
        "schema_id": record["schema_id"],
        "record_sha256": digest,
        **dict(provenance_params or {}),
    }
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command=(
                provenance_command
                or "import_recording_analysis:publish_subject_metadata"
            ),
            params=params,
            input_run_ids={},
            input_artifacts=input_artifacts,
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
        try:
            raw_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            raw_metadata = None
    if not isinstance(raw_metadata, Mapping):
        raise MissingSubjectMetadataError("Missing subject metadata")
    record = build_subject_metadata_record(raw_metadata, legacy_rule=True)
    digest = subject_metadata_sha256(record)
    return _resolved(
        record,
        digest=digest,
        group_path=group_path,
        run_name=None,
        legacy=True,
    )


# Dataset-profile composition (B4, docs/design/2026-10-07-intake-single-writer).
# The detection, keypoint and subject-mask profiles store this block; its
# field names, order and value types are part of their stored summaries.
PROFILE_COMPOSITION_FIELDS = (
    "rig_id",
    "camera_id",
    "arena_id",
    "dish_design",
    "canvas_name",
    "protocol_name",
    "genotype",
    "dpf_at_acquisition",
)


def _profile_attr_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        raw = value.decode("utf-8", "ignore")
    elif isinstance(value, str):
        raw = value
    else:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _profile_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _canonical_profile_subject_fields(root: Any) -> dict[str, Any]:
    """Genotype and age from the published subject record, or ``{}`` without one.

    Only the canonical ``subject_metadata_runs`` record counts here; archives
    without one keep their historical profile values (see
    :func:`_legacy_profile_composition`). A record that exists but does not
    resolve contributes nothing, so the profile keeps its historical values
    rather than failing; the record's own validators report that defect.
    """

    try:
        resolved = resolve_subject_metadata(root, allow_legacy=False)
    except SubjectMetadataError:
        return {}
    fields: dict[str, Any] = {}
    genotype = normalize_attr(resolved.subject.get("genotype"))
    if genotype is not None:
        fields["genotype"] = genotype
    dpf = resolved.subject.get("dpf_at_acquisition")
    if isinstance(dpf, int) and not isinstance(dpf, bool):
        fields["dpf_at_acquisition"] = dpf
    return fields


def _legacy_profile_composition(root: Any) -> dict[str, Any]:
    """Historical profile composition from root attrs and ``analysis_metadata``.

    This is the pre-B4 rule, kept value-for-value for archives that have no
    canonical subject record: root attrs win over ``session_context``, which
    wins over the ``subject_metadata``/``zebrobot_snapshot`` snapshot. It is not
    the owner's legacy path (``resolve_subject_metadata(allow_legacy=True)``),
    which derives age from dates and ignores root attrs, ``session_context``
    and ``zebrobot_snapshot``, so it would change historical profile values.
    """

    session_context: dict[str, Any] = {}
    snapshot: dict[str, Any] = {}
    analysis_meta = root.get("analysis_metadata")
    if analysis_meta is not None:
        session_context = _profile_attr_mapping(analysis_meta.attrs.get("session_context")) or {}
        for key in ("subject_metadata", "zebrobot_snapshot"):
            payload = _profile_attr_mapping(analysis_meta.attrs.get(key))
            if payload:
                snapshot = payload
                break
    dish = (_profile_attr_mapping(snapshot.get("dish")) if snapshot else None) or {}

    composition: dict[str, Any] = {}
    for key in PROFILE_COMPOSITION_FIELDS:
        if key == "protocol_name":
            value = normalize_attr(root.attrs.get("protocol_name")) or normalize_attr(
                session_context.get("protocol_name")
                or session_context.get("protocol_name_from_definition")
            )
        elif key == "genotype":
            value = (
                normalize_attr(root.attrs.get("genotype"))
                or normalize_attr(session_context.get("genotype"))
                or normalize_attr(dish.get("genotype") or snapshot.get("genotype"))
            )
        elif key == "dpf_at_acquisition":
            value = None
            for candidate in (
                root.attrs.get("dpf_at_acquisition"),
                session_context.get("dpf_at_acquisition"),
                session_context.get("days_post_fertilization"),
                snapshot.get("dpf_at_acquisition") or snapshot.get("days_post_fertilization"),
            ):
                value = _profile_int(candidate)
                if value is not None:
                    break
        else:
            value = normalize_attr(root.attrs.get(key)) or normalize_attr(session_context.get(key))
        if value is not None:
            composition[key] = value
    return composition


def read_profile_composition(root: Any) -> dict[str, Any]:
    """The ``composition`` block shared by the dataset-profile builders.

    ``genotype`` and ``dpf_at_acquisition`` come from the canonical subject
    record when one is published (the subject authority); a field the record
    does not carry, and every field of an archive without a record, keeps its
    historical value. Keys follow :data:`PROFILE_COMPOSITION_FIELDS` order and
    absent values are omitted.
    """

    composition = _legacy_profile_composition(root)
    canonical = _canonical_profile_subject_fields(root)
    if not canonical:
        return composition
    merged = {**composition, **canonical}
    return {key: merged[key] for key in PROFILE_COMPOSITION_FIELDS if key in merged}


__all__ = [
    "MissingSubjectMetadataError",
    "PROFILE_COMPOSITION_FIELDS",
    "ResolvedSubjectMetadata",
    "SUBJECT_METADATA_RECORD_ATTR",
    "SUBJECT_METADATA_RUNS_PATH",
    "SUBJECT_METADATA_SCHEMA_ID",
    "SUBJECT_METADATA_SCHEMA_VERSION",
    "SUBJECT_METADATA_SHA256_ATTR",
    "SUBJECT_METADATA_V2_SCHEMA_ID",
    "SUBJECT_METADATA_V2_SCHEMA_VERSION",
    "SubjectMetadataError",
    "build_subject_metadata_record",
    "explicit_subject_ids",
    "normalize_subject_metadata",
    "publish_subject_metadata",
    "read_h5_subject_metadata",
    "read_profile_composition",
    "resolve_subject_metadata",
    "subject_metadata_sha256",
]
