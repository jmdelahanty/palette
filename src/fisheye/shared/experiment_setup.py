"""Canonical experiment-setup publication and resolution.

``experiment_setup`` describes the acquisition plan (how many subjects and
arenas were expected).  It deliberately does not describe detections, tracks,
or source-dish population counts.  Modern archives publish immutable setup
runs below ``analysis/experiment_setup_runs``; root attrs remain a historical
compatibility projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional

from .import_source_fingerprint import optional_source_stat_fingerprint_attrs
from .json_safety import json_attr_safe_mapping, strict_json_dumps
from .run_provenance import build_writer_run_provenance
from .subject_metadata import resolve_subject_metadata
from .type_conversions import normalize_attr as _normalize_attr
from .zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
    resolve_latest_complete_run_group,
)


EXPERIMENT_SETUP_SCHEMA_ID = "palette.experiment_setup.v2"
EXPERIMENT_SETUP_SCHEMA_VERSION = 2
EXPERIMENT_SETUP_RUNS_PATH = "analysis/experiment_setup_runs"
EXPERIMENT_SETUP_RECORD_ATTR = "experiment_setup_record"
EXPERIMENT_SETUP_SHA256_ATTR = "experiment_setup_sha256"


class ExperimentSetupError(ValueError):
    """Base error for an absent, invalid, or contradictory setup authority."""


class MissingExperimentSetupError(ExperimentSetupError):
    """Raised only when no modern or permitted legacy setup exists."""


@dataclass(frozen=True)
class ExperimentSetupInfo:
    """Historical setup view retained for arena-assignment callers."""

    setup_type: str
    num_dishes: Optional[int]
    source: str
    has_experiment_setup: bool


@dataclass(frozen=True)
class ResolvedExperimentSetup:
    """Validated setup authority selected for one archive."""

    expected_subject_count: int
    expected_arena_count: Optional[int]
    expected_subjects_per_arena: Optional[int]
    assigned_subject_count: Optional[int]
    setup_type: str
    subject_assignment_status: str
    subject_metadata_ref: Optional[str]
    source: Mapping[str, Any]
    record: Mapping[str, Any]
    record_sha256: str
    group_path: str
    run_name: Optional[str]
    legacy: bool


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted


def _positive_int(value: Any, *, field: str, required: bool = False) -> Optional[int]:
    converted = _coerce_int(value)
    if converted is None:
        if required:
            raise ValueError(f"{field} must be a positive integer")
        return None
    if converted < 1:
        raise ValueError(f"{field} must be a positive integer")
    return converted


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    get_method = getattr(value, "get", None)
    if callable(get_method):
        return value  # type: ignore[return-value]
    return None


def experiment_setup_sha256(record: Mapping[str, Any]) -> str:
    """Return the canonical strict-JSON digest for one setup record."""

    return sha256(strict_json_dumps(record).encode("utf-8")).hexdigest()


def build_experiment_setup_record(
    subject_metadata: Mapping[str, Any],
    *,
    source_h5_path: str | Path | None = None,
    source: Mapping[str, Any] | None = None,
    subject_metadata_sha256: str | None = None,
    subject_metadata_ref: str | None = None,
) -> dict[str, Any]:
    """Build a v2 setup record from acquisition-time subject metadata.

    ``fish_count`` is intentionally retained only as
    ``source_dish_population_count``.  It is never used as the expected number
    of subjects in this recording.
    """

    metadata = json_attr_safe_mapping(subject_metadata)
    expected = _positive_int(
        metadata.get("subject_count"),
        field="subject_metadata.subject_count",
        required=True,
    )
    assert expected is not None
    if not subject_metadata_ref or not str(subject_metadata_ref).startswith(
        "analysis/subject_metadata_runs/"
    ):
        raise ExperimentSetupError(
            "A canonical subject_metadata_runs reference is required"
        )
    if (
        not subject_metadata_sha256
        or len(str(subject_metadata_sha256)) != 64
        or any(character not in "0123456789abcdef" for character in str(subject_metadata_sha256).lower())
    ):
        raise ExperimentSetupError("A canonical subject-metadata SHA-256 is required")

    raw_subject_ids = metadata.get("subject_ids") or metadata.get("fish_ids")
    if isinstance(raw_subject_ids, (list, tuple)):
        subject_ids = list(
            dict.fromkeys(
                str(value).strip() for value in raw_subject_ids if str(value).strip()
            )
        )
    else:
        fish_id = str(metadata.get("fish_id") or "").strip() or None
        subject_ids = [fish_id] if fish_id is not None else []
    assigned = len(subject_ids) or None
    if assigned is not None and assigned > expected:
        raise ExperimentSetupError(
            "Subject metadata assigns more explicit identities than the declared "
            f"subject_count: assigned={assigned}, expected={expected}"
        )
    assignment_status = (
        "explicit"
        if assigned == expected
        else ("partial" if assigned is not None else "count_only")
    )
    expected_arenas = 1
    subjects_per_arena = expected
    setup_type = (
        "single_subject_single_arena"
        if expected == 1
        else "multi_subject_single_arena"
    )

    if source is None:
        source_record: dict[str, Any] = {
            "kind": "h5_subject_metadata",
            "group_path": "/subject_metadata",
            "count_field": "subject_count",
        }
        if source_h5_path is not None:
            source_record["file_name"] = Path(source_h5_path).name
    else:
        source_record = json_attr_safe_mapping(source)
        if not str(source_record.get("kind") or "").strip():
            raise ExperimentSetupError("Experiment setup source requires kind")
        if not str(source_record.get("count_field") or "").strip():
            raise ExperimentSetupError("Experiment setup source requires count_field")

    population = _positive_int(
        metadata.get("source_dish_population_count", metadata.get("fish_count")),
        field="subject_metadata.source_dish_population_count",
        required=False,
    )
    record: dict[str, Any] = {
        "schema_id": EXPERIMENT_SETUP_SCHEMA_ID,
        "schema_version": EXPERIMENT_SETUP_SCHEMA_VERSION,
        "setup_type": setup_type,
        "expected_subject_count": expected,
        "expected_arena_count": expected_arenas,
        "expected_subjects_per_arena": subjects_per_arena,
        "assigned_subject_count": assigned,
        "subject_assignment_status": assignment_status,
        "subject_metadata_ref": str(subject_metadata_ref),
        "subject_metadata_sha256": str(subject_metadata_sha256),
        "source": source_record,
    }
    if population is not None:
        record["source_dish_population_count"] = population
    return record


def _validate_record(record: Mapping[str, Any], digest: str | None = None) -> dict[str, Any]:
    canonical = json_attr_safe_mapping(record)
    if canonical.get("schema_id") != EXPERIMENT_SETUP_SCHEMA_ID:
        raise ValueError(
            "Experiment setup has unsupported schema_id "
            f"{canonical.get('schema_id')!r}"
        )
    if _coerce_int(canonical.get("schema_version")) != EXPERIMENT_SETUP_SCHEMA_VERSION:
        raise ValueError("Experiment setup has unsupported schema_version")
    _positive_int(
        canonical.get("expected_subject_count"),
        field="expected_subject_count",
        required=True,
    )
    subject_ref = str(canonical.get("subject_metadata_ref") or "")
    if not subject_ref.startswith("analysis/subject_metadata_runs/"):
        raise ExperimentSetupError("Experiment setup has no canonical subject-metadata reference")
    subject_digest = str(canonical.get("subject_metadata_sha256") or "")
    if len(subject_digest) != 64 or any(
        character not in "0123456789abcdef" for character in subject_digest.lower()
    ):
        raise ExperimentSetupError("Experiment setup has no valid subject-metadata digest")
    for field in (
        "expected_arena_count",
        "expected_subjects_per_arena",
        "assigned_subject_count",
        "source_dish_population_count",
    ):
        _positive_int(canonical.get(field), field=field, required=False)
    actual = experiment_setup_sha256(canonical)
    if digest is not None and str(digest) != actual:
        raise ValueError(
            "Experiment setup digest mismatch: "
            f"stored={digest!r}, computed={actual!r}"
        )
    return canonical


def publish_experiment_setup(
    root: Any,
    record: Mapping[str, Any],
    *,
    source_h5_path: str | Path | None = None,
    source_artifact: Mapping[str, Any] | None = None,
    provenance_command: str = "import_recording_analysis:publish_experiment_setup",
) -> ResolvedExperimentSetup:
    """Idempotently publish and select one immutable setup run."""

    if source_h5_path is not None and source_artifact is not None:
        raise ExperimentSetupError(
            "Provide source_h5_path or source_artifact, not both"
        )

    canonical = _validate_record(record)
    digest = experiment_setup_sha256(canonical)
    run_name = f"experiment_setup_{digest[:16]}"
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "experiment_setup_runs")

    if run_name in parent:
        existing = parent[run_name]
        existing_record = _as_mapping(existing.attrs.get(EXPERIMENT_SETUP_RECORD_ATTR))
        if existing_record is None:
            raise ValueError(f"Existing experiment setup run {run_name!r} has no record")
        _validate_record(
            existing_record,
            str(existing.attrs.get(EXPERIMENT_SETUP_SHA256_ATTR) or ""),
        )
        if experiment_setup_sha256(existing_record) != digest:
            raise ValueError(f"Existing experiment setup run {run_name!r} conflicts")
        if not is_run_complete_in_parent(parent, existing, legacy_default=False) or not is_run_selector_eligible(existing):
            raise ValueError(
                f"Existing experiment setup run {run_name!r} is not a complete "
                "selector-eligible immutable artifact"
            )
        parent.attrs["latest_complete"] = run_name
        parent.attrs["latest"] = run_name
        _write_legacy_projection(root, canonical, run_name=run_name, digest=digest)
        return resolve_experiment_setup(root, allow_legacy=False)

    run = parent.create_group(run_name)
    mark_run_started(run, run_name=run_name, stage="experiment_setup")
    note_pending_latest(parent, run_name)
    run.attrs["stage_selector_eligible"] = False
    run.attrs["schema_id"] = EXPERIMENT_SETUP_SCHEMA_ID
    run.attrs["schema_version"] = EXPERIMENT_SETUP_SCHEMA_VERSION
    run.attrs[EXPERIMENT_SETUP_RECORD_ATTR] = canonical
    run.attrs[EXPERIMENT_SETUP_SHA256_ATTR] = digest
    run.attrs["immutable"] = True

    validated = _validate_record(
        run.attrs[EXPERIMENT_SETUP_RECORD_ATTR],
        str(run.attrs[EXPERIMENT_SETUP_SHA256_ATTR]),
    )
    if validated != canonical:
        raise ValueError("Experiment setup did not round-trip through Zarr attrs")
    run.attrs["stage_selector_eligible"] = True
    if source_artifact is not None:
        input_artifacts = [json_attr_safe_mapping(source_artifact)]
    else:
        source_fingerprint = None
        if source_h5_path is not None:
            source_fingerprint = optional_source_stat_fingerprint_attrs(
                source_h5_path,
                attr_prefix="source_h5",
            ).get("source_h5_fingerprint")
        input_artifacts = [
            {
                "kind": "source_h5",
                "path": str(source_h5_path) if source_h5_path is not None else None,
                "stat_fingerprint": source_fingerprint,
            }
        ]
    mark_run_complete(
        run,
        parent_group=parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command=provenance_command,
            params={
                "schema_id": EXPERIMENT_SETUP_SCHEMA_ID,
                "record_sha256": digest,
            },
            input_run_ids={},
            input_artifacts=input_artifacts,
        ),
    )

    # Historical projection only. Modern consumers bind the selected run and
    # digest returned by ``resolve_experiment_setup``.
    _write_legacy_projection(root, canonical, run_name=run_name, digest=digest)
    return resolve_experiment_setup(root, allow_legacy=False)


def _write_legacy_projection(
    root: Any,
    canonical: Mapping[str, Any],
    *,
    run_name: str,
    digest: str,
) -> None:
    root.attrs["experiment_setup"] = {
        "schema_id": EXPERIMENT_SETUP_SCHEMA_ID,
        "setup_type": canonical["setup_type"],
        "expected_subject_count": canonical["expected_subject_count"],
        "total_expected_fish": canonical["expected_subject_count"],
        "subject_count": canonical["expected_subject_count"],
        "num_dishes": canonical.get("expected_arena_count"),
        "fish_per_dish": canonical.get("expected_subjects_per_arena"),
        "canonical_run": run_name,
        "canonical_sha256": digest,
    }
    root.attrs["subject_count"] = canonical["expected_subject_count"]


def _resolved_from_record(
    record: Mapping[str, Any],
    *,
    digest: str,
    group_path: str,
    run_name: str | None,
    legacy: bool,
) -> ResolvedExperimentSetup:
    expected = _positive_int(
        record.get("expected_subject_count"),
        field="expected_subject_count",
        required=True,
    )
    assert expected is not None
    return ResolvedExperimentSetup(
        expected_subject_count=expected,
        expected_arena_count=_positive_int(
            record.get("expected_arena_count"), field="expected_arena_count"
        ),
        expected_subjects_per_arena=_positive_int(
            record.get("expected_subjects_per_arena"),
            field="expected_subjects_per_arena",
        ),
        assigned_subject_count=_positive_int(
            record.get("assigned_subject_count"), field="assigned_subject_count"
        ),
        setup_type=str(record.get("setup_type") or "unknown"),
        subject_assignment_status=str(
            record.get("subject_assignment_status") or "unknown"
        ),
        subject_metadata_ref=(
            str(record.get("subject_metadata_ref"))
            if record.get("subject_metadata_ref") not in (None, "")
            else None
        ),
        source=_as_mapping(record.get("source")) or {},
        record=dict(record),
        record_sha256=digest,
        group_path=group_path,
        run_name=run_name,
        legacy=legacy,
    )


def resolve_experiment_setup(
    root: Any,
    *,
    allow_legacy: bool = True,
) -> ResolvedExperimentSetup:
    """Resolve the selected complete setup run, optionally falling back to attrs."""

    analysis = root.get("analysis")
    parent = analysis.get("experiment_setup_runs") if analysis is not None else None
    if parent is not None:
        run_name, run = resolve_latest_complete_run_group(parent, legacy_default=False)
        if run_name is None or run is None:
            raise ExperimentSetupError(
                f"{EXPERIMENT_SETUP_RUNS_PATH} exists but has no selected complete run"
            )
        raw_record = _as_mapping(run.attrs.get(EXPERIMENT_SETUP_RECORD_ATTR))
        if raw_record is None:
            raise ExperimentSetupError(
                f"{EXPERIMENT_SETUP_RUNS_PATH}/{run_name} has no setup record"
            )
        digest = str(run.attrs.get(EXPERIMENT_SETUP_SHA256_ATTR) or "")
        record = _validate_record(raw_record, digest)
        subject = resolve_subject_metadata(root, allow_legacy=False)
        if (
            subject.group_path != record["subject_metadata_ref"]
            or subject.record_sha256 != record["subject_metadata_sha256"]
        ):
            raise ExperimentSetupError(
                "Selected experiment setup and subject-metadata authorities disagree"
            )
        return _resolved_from_record(
            record,
            digest=digest,
            group_path=f"{EXPERIMENT_SETUP_RUNS_PATH}/{run_name}",
            run_name=run_name,
            legacy=False,
        )

    if not allow_legacy:
        raise MissingExperimentSetupError(
            f"Missing canonical {EXPERIMENT_SETUP_RUNS_PATH}"
        )
    raw = _as_mapping(root.attrs.get("experiment_setup"))
    if raw is None:
        raise MissingExperimentSetupError("Missing experiment setup metadata")
    expected = (
        _positive_int(raw.get("expected_subject_count"), field="expected_subject_count")
        or _positive_int(raw.get("total_expected_fish"), field="total_expected_fish")
        or _positive_int(raw.get("subject_count"), field="subject_count")
    )
    if expected is None:
        raise ExperimentSetupError(
            "Legacy experiment_setup has no positive expected subject count"
        )
    arenas = _positive_int(raw.get("num_dishes"), field="num_dishes")
    per_arena = _positive_int(raw.get("fish_per_dish"), field="fish_per_dish")
    record = {
        "schema_id": "palette.experiment_setup.legacy",
        "schema_version": 0,
        "setup_type": str(raw.get("setup_type") or "unknown"),
        "expected_subject_count": expected,
        "expected_arena_count": arenas,
        "expected_subjects_per_arena": per_arena,
        "assigned_subject_count": None,
        "subject_assignment_status": "legacy_unknown",
        "subject_metadata_ref": None,
        "source": {"kind": "legacy_root_attr", "attr": "experiment_setup"},
    }
    digest = experiment_setup_sha256(record)
    return _resolved_from_record(
        record,
        digest=digest,
        group_path="@experiment_setup",
        run_name=None,
        legacy=True,
    )


def resolve_expected_subject_count(
    root: Any,
    explicit_count: Any = None,
    *,
    allow_legacy: bool = True,
) -> tuple[int, ResolvedExperimentSetup]:
    """Resolve expected count and reject an explicit contradiction."""

    setup = resolve_experiment_setup(root, allow_legacy=allow_legacy)
    if explicit_count is not None:
        explicit = _positive_int(
            explicit_count,
            field="explicit expected subject count",
            required=True,
        )
        if explicit != setup.expected_subject_count:
            raise ExperimentSetupError(
                "Explicit expected subject count contradicts experiment setup: "
                f"explicit={explicit}, setup={setup.expected_subject_count}, "
                f"setup_path={setup.group_path}"
            )
    return setup.expected_subject_count, setup


def infer_experiment_setup(attrs: Mapping[str, Any]) -> ExperimentSetupInfo:
    """Infer single- vs multi-arena mode using compatibility root attrs."""

    experiment_setup = attrs.get("experiment_setup")
    if experiment_setup:
        mapping = _as_mapping(experiment_setup)
        setup_type = None
        num_dishes = None
        if mapping is not None:
            setup_type = _normalize_attr(mapping.get("setup_type"))
            num_dishes = _coerce_int(
                mapping.get("num_dishes", mapping.get("expected_arena_count"))
            )
        if setup_type in {"single_dish", "single_subject_single_arena", "multi_subject_single_arena"} or num_dishes == 1:
            return ExperimentSetupInfo("single_dish", num_dishes or 1, "experiment_setup", True)
        if setup_type == "multi_dish" or (num_dishes is not None and num_dishes > 1):
            return ExperimentSetupInfo("multi_dish", num_dishes, "experiment_setup", True)
        return ExperimentSetupInfo(setup_type or "unknown", num_dishes, "experiment_setup", True)

    chamber = _normalize_attr(attrs.get("experimental_chamber"))
    if chamber:
        return ExperimentSetupInfo("single_dish", 1, "experimental_chamber", False)

    return ExperimentSetupInfo("unknown", None, "unknown", False)


def subdish_required(attrs: Mapping[str, Any]) -> bool:
    """Return True if sub-dish masks are required for spatial arena assignment."""

    info = infer_experiment_setup(attrs)
    return info.setup_type != "single_dish"


__all__ = [
    "EXPERIMENT_SETUP_RECORD_ATTR",
    "EXPERIMENT_SETUP_RUNS_PATH",
    "EXPERIMENT_SETUP_SCHEMA_ID",
    "EXPERIMENT_SETUP_SCHEMA_VERSION",
    "EXPERIMENT_SETUP_SHA256_ATTR",
    "ExperimentSetupError",
    "ExperimentSetupInfo",
    "MissingExperimentSetupError",
    "ResolvedExperimentSetup",
    "build_experiment_setup_record",
    "experiment_setup_sha256",
    "infer_experiment_setup",
    "publish_experiment_setup",
    "resolve_experiment_setup",
    "resolve_expected_subject_count",
    "subdish_required",
]
