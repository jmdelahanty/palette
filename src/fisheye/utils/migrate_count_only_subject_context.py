"""Migrate recording-local placeholder subjects to count-only authority.

This is a dry-run-first repair for historical manual subject-context backfills
that encoded an anonymous count as synthetic ``subject_id`` values.  It
publishes an immutable identity-free subject/setup authority on the canonical
analysis archive, refreshes every active source-recording sibling in the
registry, and removes only the exact placeholder memberships named by the
reviewed plan.

The command does not infer subjects from detections, tracks, filenames, or an
H5 population count.  Existing scientific context is accepted only when all
legacy placeholder snapshots for a recording agree.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import zarr

from fisheye.registry.db import Registry, RegistryPaths, SQLITE_BUSY_TIMEOUT_MS
from fisheye.registry.prune_stale_datasets import create_backup
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.experiment_setup import (
    MissingExperimentSetupError,
    build_experiment_setup_record,
    experiment_setup_sha256,
    publish_experiment_setup,
    resolve_experiment_setup,
)
from fisheye.shared.json_safety import json_attr_safe_mapping, strict_json_dumps
from fisheye.shared.subject_metadata import (
    MissingSubjectMetadataError,
    SubjectMetadataError,
    build_subject_metadata_record,
    publish_subject_metadata,
    resolve_subject_metadata,
    subject_metadata_sha256,
)

REPORT_SCHEMA_ID = "palette.count_only_subject_context_migration.v1"
MANUAL_AUTHORITY_SOURCE = "manual_operator_assertion"
ASSERTION_KIND = "legacy_recording_local_placeholder_migration"
LEGACY_SOURCE = "manual_subject_context_backfill"
LEGACY_IDENTITY_SCOPE = "recording_local_placeholder"
PROVENANCE_COMMAND = "migrate_count_only_subject_context"

_IDENTITY_FIELDS = frozenset(
    {
        "fish_id",
        "fish_ids",
        "subject_id",
        "subject_ids",
        "identity_scope",
    }
)
_CONTEXT_FIELDS = (
    "species",
    "subject_count",
    "subject_type",
    "dpf_at_acquisition",
    "days_post_fertilization",
    "date_of_fertilization",
    "genotype",
    "line_strain",
    "sex",
    "dish_id",
    "cross_id",
    "source_dish_population_count",
)


@dataclass(frozen=True)
class CountOnlyTarget:
    dataset_id: str
    recording_id: str
    zarr_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "recording_id": self.recording_id,
            "zarr_path": str(self.zarr_path),
        }


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    conn = sqlite3.connect(
        f"file:{quote(str(resolved), safe='/')}?mode=ro",
        uri=True,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA query_only = ON;")
    return conn


def _open_root(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode)


def _decode_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _is_placeholder_metadata(metadata: Mapping[str, Any]) -> bool:
    return (
        str(metadata.get("source") or "").strip() == LEGACY_SOURCE
        and str(metadata.get("identity_scope") or "").strip() == LEGACY_IDENTITY_SCOPE
    )


def _subject_ids(metadata: Mapping[str, Any]) -> list[str]:
    raw = metadata.get("subject_ids") or metadata.get("fish_ids")
    if isinstance(raw, (list, tuple)):
        values = [str(value).strip() for value in raw if str(value).strip()]
    else:
        value = str(metadata.get("subject_id") or metadata.get("fish_id") or "").strip()
        values = [value] if value else []
    return list(dict.fromkeys(values))


def _positive_subject_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("subject_count must be a positive integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("subject_count must be a positive integer") from exc
    if count < 1:
        raise ValueError("subject_count must be a positive integer")
    return count


def _normalize_context(metadata: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json_attr_safe_mapping(metadata)
    context = {
        key: canonical[key]
        for key in _CONTEXT_FIELDS
        if canonical.get(key) not in (None, "", [])
    }
    count = _positive_subject_count(context.get("subject_count"))
    context["subject_count"] = count
    context.setdefault("subject_type", "individual" if count == 1 else "group")

    dpf_values: list[int] = []
    for key in ("dpf_at_acquisition", "days_post_fertilization"):
        if context.get(key) not in (None, ""):
            try:
                dpf_values.append(int(context[key]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer") from exc
    if len(set(dpf_values)) > 1:
        raise ValueError("DPF fields disagree")
    if dpf_values:
        if dpf_values[0] < 0:
            raise ValueError("DPF cannot be negative")
        context["dpf_at_acquisition"] = dpf_values[0]
        context["days_post_fertilization"] = dpf_values[0]
    if "source_dish_population_count" in context:
        context["source_dish_population_count"] = _positive_subject_count(
            context["source_dish_population_count"]
        )
    return context


def _metadata_digest(metadata: Mapping[str, Any]) -> str:
    return sha256(strict_json_dumps(metadata).encode("utf-8")).hexdigest()


def _placeholder_membership(metadata_json: Any) -> bool:
    return _is_placeholder_metadata(_decode_json_mapping(metadata_json))


def select_targets(
    registry_path: Path,
    *,
    recording_ids: Sequence[str] = (),
    path_contains: Sequence[str] = (),
    all_placeholders: bool = False,
) -> list[CountOnlyTarget]:
    """Select one active source-recording analysis archive per recording."""

    if not recording_ids and not path_contains and not all_placeholders:
        raise ValueError(
            "Provide --recording-id/--path-contains or pass --all-placeholders."
        )
    sql = [
        "SELECT d.dataset_id, d.recording_id, d.zarr_path",
        "FROM datasets d",
        "WHERE d.status = 'active' COLLATE NOCASE",
        "  AND d.zarr_use = 'analysis' COLLATE NOCASE",
        "  AND d.artifact_kind = 'source_recording' COLLATE NOCASE",
        "  AND NULLIF(TRIM(d.recording_id), '') IS NOT NULL",
    ]
    params: list[Any] = []
    if recording_ids:
        placeholders = ", ".join("?" for _ in recording_ids)
        sql.append(f"AND d.recording_id IN ({placeholders})")
        params.extend(str(value) for value in recording_ids)
    if path_contains:
        predicates = " OR ".join("LOWER(d.zarr_path) LIKE ?" for _ in path_contains)
        sql.append(f"AND ({predicates})")
        params.extend(f"%{str(value).casefold()}%" for value in path_contains)
    sql.append("ORDER BY d.recording_id, d.dataset_id")

    with closing(_connect_read_only(registry_path)) as conn:
        rows = conn.execute("\n".join(sql), params).fetchall()
        placeholder_recordings = {
            str(row["recording_id"])
            for row in conn.execute(
                "SELECT recording_id, metadata_json FROM recording_subjects"
            ).fetchall()
            if row["recording_id"] is not None
            and _placeholder_membership(row["metadata_json"])
        }

    targets = [
        CountOnlyTarget(
            dataset_id=str(row["dataset_id"]),
            recording_id=str(row["recording_id"]),
            zarr_path=Path(str(row["zarr_path"])).expanduser().resolve(strict=False),
        )
        for row in rows
        if not all_placeholders or str(row["recording_id"]) in placeholder_recordings
    ]
    counts = Counter(target.recording_id for target in targets)
    ambiguous = sorted(key for key, count in counts.items() if count != 1)
    if ambiguous:
        raise ValueError(
            "Expected exactly one active analysis source per recording: "
            + ", ".join(ambiguous)
        )
    return targets


def _sibling_datasets(
    conn: sqlite3.Connection, recording_id: str
) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT dataset_id, zarr_path, zarr_use, artifact_kind, status
        FROM datasets
        WHERE recording_id = ?
          AND status = 'active' COLLATE NOCASE
          AND artifact_kind = 'source_recording' COLLATE NOCASE
        ORDER BY dataset_id;
        """,
        (recording_id,),
    ).fetchall()
    return [
        {
            "dataset_id": str(row["dataset_id"]),
            "zarr_path": str(
                Path(str(row["zarr_path"])).expanduser().resolve(strict=False)
            ),
            "zarr_use": str(row["zarr_use"] or ""),
            "artifact_kind": str(row["artifact_kind"] or ""),
            "status": str(row["status"] or ""),
        }
        for row in rows
    ]


def _placeholder_cleanup_plan(
    conn: sqlite3.Connection,
    *,
    recording_id: str,
    evidence_subject_ids: set[str],
) -> dict[str, Any]:
    memberships: list[dict[str, Any]] = []
    unexpected: list[str] = []
    non_placeholder: list[str] = []
    rows = conn.execute(
        """
        SELECT recording_id, subject_id, dataset_id, metadata_json
        FROM recording_subjects
        WHERE recording_id = ?
        ORDER BY subject_id;
        """,
        (recording_id,),
    ).fetchall()
    for row in rows:
        metadata = _decode_json_mapping(row["metadata_json"])
        if not _is_placeholder_metadata(metadata):
            non_placeholder.append(str(row["subject_id"]))
            continue
        subject_id = str(row["subject_id"])
        if subject_id not in evidence_subject_ids:
            unexpected.append(subject_id)
            continue
        memberships.append(
            {
                "recording_id": recording_id,
                "subject_id": subject_id,
                "dataset_id": (
                    str(row["dataset_id"]) if row["dataset_id"] is not None else None
                ),
                "metadata_sha256": _metadata_digest(metadata),
            }
        )

    subject_candidates: list[dict[str, Any]] = []
    planned_ids = {row["subject_id"] for row in memberships}
    for subject_id in sorted(planned_ids):
        subject = conn.execute(
            "SELECT metadata_json FROM subjects WHERE subject_id = ?",
            (subject_id,),
        ).fetchone()
        if subject is None:
            continue
        metadata = _decode_json_mapping(subject["metadata_json"])
        other_memberships = conn.execute(
            """
            SELECT COUNT(*)
            FROM recording_subjects
            WHERE subject_id = ? AND recording_id != ?;
            """,
            (subject_id, recording_id),
        ).fetchone()[0]
        if (
            str(metadata.get("source") or "").strip() == LEGACY_SOURCE
            and int(other_memberships) == 0
        ):
            subject_candidates.append(
                {
                    "subject_id": subject_id,
                    "metadata_sha256": _metadata_digest(metadata),
                }
            )
    return {
        "recording_subjects": memberships,
        "orphan_subjects": subject_candidates,
        "unexpected_placeholder_subject_ids": unexpected,
        "non_placeholder_subject_ids": non_placeholder,
    }


def _manual_assertion(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    assertion = metadata.get("manual_assertion")
    return assertion if isinstance(assertion, Mapping) else None


def _build_new_authorities(
    *,
    target: CountOnlyTarget,
    context: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    reviewer: str,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = {
        **dict(context),
        "recording_id": target.recording_id,
        "source": MANUAL_AUTHORITY_SOURCE,
        "status": "asserted",
        "manual_assertion": {
            "kind": ASSERTION_KIND,
            "reviewer": reviewer,
            "reason": reason,
            "evidence": list(evidence),
        },
    }
    for field in _IDENTITY_FIELDS:
        metadata.pop(field, None)
    subject_record = build_subject_metadata_record(metadata)
    subject_digest = subject_metadata_sha256(subject_record)
    subject_ref = (
        "analysis/subject_metadata_runs/" f"subject_metadata_{subject_digest[:16]}"
    )
    setup_source = {
        "kind": MANUAL_AUTHORITY_SOURCE,
        "assertion_kind": ASSERTION_KIND,
        "count_field": "subject_count",
        "reviewer": reviewer,
        "reason": reason,
    }
    setup_record = build_experiment_setup_record(
        metadata,
        source=setup_source,
        subject_metadata_ref=subject_ref,
        subject_metadata_sha256=subject_digest,
    )
    return subject_record, setup_record


def _desired_payload(
    subject_record: Mapping[str, Any], setup_record: Mapping[str, Any]
) -> dict[str, Any]:
    subject_digest = subject_metadata_sha256(subject_record)
    setup_digest = experiment_setup_sha256(setup_record)
    return {
        "subject_metadata_record": dict(subject_record),
        "subject_metadata_sha256": subject_digest,
        "subject_metadata_run": f"subject_metadata_{subject_digest[:16]}",
        "experiment_setup_record": dict(setup_record),
        "experiment_setup_sha256": setup_digest,
        "experiment_setup_run": f"experiment_setup_{setup_digest[:16]}",
        "expected_subject_count": int(setup_record["expected_subject_count"]),
        "assigned_subject_count": setup_record.get("assigned_subject_count"),
        "subject_assignment_status": setup_record["subject_assignment_status"],
    }


def plan_target(
    registry_path: Path,
    target: CountOnlyTarget,
    *,
    reviewer: str,
    reason: str,
) -> dict[str, Any]:
    """Build a mutation-free plan for one recording."""

    row: dict[str, Any] = {
        **target.as_dict(),
        "disposition": "blocked",
        "action": "none",
        "reason": None,
        "detail": None,
    }
    if not reviewer.strip() or not reason.strip():
        row.update(reason="reviewer_and_reason_required")
        return row
    if not target.zarr_path.is_dir():
        row.update(reason="analysis_zarr_missing")
        return row

    with closing(_connect_read_only(registry_path)) as conn:
        siblings = _sibling_datasets(conn, target.recording_id)
        row["source_datasets"] = siblings

        evidence: list[dict[str, Any]] = []
        evidence_contexts: list[dict[str, Any]] = []
        evidence_ids: set[str] = set()
        warnings: list[str] = []
        for sibling in siblings:
            path = Path(sibling["zarr_path"])
            if not path.is_dir():
                warnings.append(f"dataset_missing:{sibling['dataset_id']}")
                continue
            try:
                root = _open_root(path, mode="r")
                resolved = resolve_subject_metadata(root, allow_legacy=True)
            except MissingSubjectMetadataError:
                warnings.append(f"subject_metadata_missing:{sibling['dataset_id']}")
                continue
            except Exception as exc:
                row.update(
                    reason="subject_metadata_unreadable",
                    detail=f"{sibling['dataset_id']}: {type(exc).__name__}: {exc}",
                    warnings=warnings,
                )
                return row
            metadata = dict(resolved.metadata)
            if not _is_placeholder_metadata(metadata):
                continue
            ids = _subject_ids(metadata)
            if not ids:
                row.update(
                    reason="placeholder_snapshot_has_no_placeholder_ids",
                    detail=sibling["dataset_id"],
                    warnings=warnings,
                )
                return row
            try:
                context = _normalize_context(metadata)
            except ValueError as exc:
                row.update(
                    reason="placeholder_context_invalid",
                    detail=f"{sibling['dataset_id']}: {exc}",
                    warnings=warnings,
                )
                return row
            if len(ids) != int(context["subject_count"]):
                row.update(
                    reason="placeholder_identity_count_disagrees_with_subject_count",
                    detail=(
                        f"{sibling['dataset_id']}: ids={len(ids)}, "
                        f"subject_count={context['subject_count']}"
                    ),
                    warnings=warnings,
                )
                return row
            evidence_contexts.append(context)
            evidence_ids.update(ids)
            evidence.append(
                {
                    "dataset_id": sibling["dataset_id"],
                    "zarr_path": sibling["zarr_path"],
                    "subject_metadata_path": resolved.group_path,
                    "subject_metadata_sha256": _metadata_digest(metadata),
                    "legacy_source": LEGACY_SOURCE,
                    "legacy_identity_scope": LEGACY_IDENTITY_SCOPE,
                    "placeholder_subject_ids": ids,
                }
            )

        try:
            analysis_root = _open_root(target.zarr_path, mode="r")
            existing_subject = resolve_subject_metadata(
                analysis_root, allow_legacy=False
            )
        except MissingSubjectMetadataError:
            existing_subject = None
        except SubjectMetadataError as exc:
            row.update(
                reason="existing_subject_authority_invalid",
                detail=f"{type(exc).__name__}: {exc}",
                warnings=warnings,
            )
            return row
        except Exception as exc:
            row.update(
                reason="analysis_authority_unreadable",
                detail=f"{type(exc).__name__}: {exc}",
                warnings=warnings,
            )
            return row

        existing_setup = None
        try:
            existing_setup = resolve_experiment_setup(analysis_root, allow_legacy=False)
        except MissingExperimentSetupError:
            pass
        except Exception as exc:
            row.update(
                reason="existing_setup_authority_invalid",
                detail=f"{type(exc).__name__}: {exc}",
                warnings=warnings,
            )
            return row

        if existing_subject is not None:
            assertion = _manual_assertion(existing_subject.metadata)
            if (
                existing_subject.subject_identity_kind != "none"
                or str(existing_subject.metadata.get("source") or "")
                != MANUAL_AUTHORITY_SOURCE
                or assertion is None
                or assertion.get("kind") != ASSERTION_KIND
            ):
                row.update(
                    reason="existing_subject_authority_is_not_this_count_only_migration",
                    warnings=warnings,
                )
                return row
            if (
                str(assertion.get("reviewer") or "") != reviewer
                or str(assertion.get("reason") or "") != reason
            ):
                row.update(
                    reason="reviewer_or_reason_disagrees_with_existing_authority",
                    warnings=warnings,
                )
                return row
            stored_evidence = assertion.get("evidence")
            if not isinstance(stored_evidence, list) or not all(
                isinstance(item, Mapping) for item in stored_evidence
            ):
                row.update(
                    reason="existing_authority_evidence_is_invalid",
                    warnings=warnings,
                )
                return row
            evidence = [dict(item) for item in stored_evidence]
            for item in evidence:
                raw_ids = item.get("placeholder_subject_ids")
                if isinstance(raw_ids, (list, tuple)):
                    evidence_ids.update(
                        str(value).strip() for value in raw_ids if str(value).strip()
                    )
            desired_subject_record = dict(existing_subject.record)
            expected_context = _normalize_context(existing_subject.metadata)
            desired_setup_record = build_experiment_setup_record(
                existing_subject.metadata,
                source={
                    "kind": MANUAL_AUTHORITY_SOURCE,
                    "assertion_kind": ASSERTION_KIND,
                    "count_field": "subject_count",
                    "reviewer": reviewer,
                    "reason": reason,
                },
                subject_metadata_ref=existing_subject.group_path,
                subject_metadata_sha256=existing_subject.record_sha256,
            )
        else:
            if not evidence:
                row.update(
                    reason="legacy_placeholder_evidence_missing",
                    warnings=warnings,
                )
                return row
            expected_context = evidence_contexts[0]
            if any(context != expected_context for context in evidence_contexts[1:]):
                row.update(
                    reason="legacy_placeholder_context_disagreement",
                    detail=str(evidence_contexts),
                    warnings=warnings,
                )
                return row
            desired_subject_record, desired_setup_record = _build_new_authorities(
                target=target,
                context=expected_context,
                evidence=evidence,
                reviewer=reviewer,
                reason=reason,
            )

        if any(context != expected_context for context in evidence_contexts):
            row.update(
                reason="legacy_placeholder_context_disagrees_with_authority",
                warnings=warnings,
            )
            return row
        desired = _desired_payload(desired_subject_record, desired_setup_record)
        if desired["assigned_subject_count"] is not None:
            row.update(
                reason="desired_authority_is_not_identity_free", warnings=warnings
            )
            return row
        if desired["subject_assignment_status"] != "count_only":
            row.update(reason="desired_setup_is_not_count_only", warnings=warnings)
            return row
        if existing_setup is not None:
            if existing_setup.record_sha256 != desired["experiment_setup_sha256"]:
                row.update(
                    reason="existing_setup_authority_conflicts_with_plan",
                    warnings=warnings,
                )
                return row

        cleanup = _placeholder_cleanup_plan(
            conn,
            recording_id=target.recording_id,
            evidence_subject_ids=evidence_ids,
        )
        if cleanup["unexpected_placeholder_subject_ids"]:
            row.update(
                reason="unexpected_placeholder_membership",
                detail=str(cleanup["unexpected_placeholder_subject_ids"]),
                warnings=warnings,
            )
            return row
        if cleanup["non_placeholder_subject_ids"]:
            row.update(
                reason="explicit_or_unrelated_subject_membership_present",
                detail=str(cleanup["non_placeholder_subject_ids"]),
                warnings=warnings,
            )
            return row

    action = "publish"
    if existing_subject is not None and existing_setup is not None:
        action = "verify_existing"
    elif existing_subject is not None:
        action = "publish_missing_setup"
    row.update(
        disposition="eligible",
        action=action,
        desired=desired,
        cleanup=cleanup,
        evidence=evidence,
        warnings=warnings,
    )
    return row


def _plan_digest(plan: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    return sha256(strict_json_dumps(payload).encode("utf-8")).hexdigest()


def build_plan(
    registry_path: Path,
    targets: Sequence[CountOnlyTarget],
    *,
    reviewer: str,
    reason: str,
) -> dict[str, Any]:
    rows = [
        plan_target(
            registry_path,
            target,
            reviewer=reviewer.strip(),
            reason=reason.strip(),
        )
        for target in targets
    ]
    plan: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "created_utc": utc_now(),
        "mode": "dry_run",
        "registry_path": str(registry_path.expanduser().resolve(strict=False)),
        "reviewer": reviewer.strip(),
        "reason": reason.strip(),
        "dataset_count": len(rows),
        "recording_count": len({row["recording_id"] for row in rows}),
        "disposition_counts": dict(
            sorted(Counter(row["disposition"] for row in rows).items())
        ),
        "action_counts": dict(sorted(Counter(row["action"] for row in rows).items())),
        "reason_counts": dict(
            sorted(Counter(row["reason"] for row in rows if row["reason"]).items())
        ),
        "recordings": rows,
    }
    plan["plan_sha256"] = _plan_digest(plan)
    return plan


def _verify_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_id") != REPORT_SCHEMA_ID:
        raise ValueError("Apply plan has an unsupported schema_id")
    digest = str(plan.get("plan_sha256") or "")
    if digest != _plan_digest(plan):
        raise ValueError("Apply plan digest mismatch")
    if not str(plan.get("reviewer") or "").strip():
        raise ValueError("Apply plan has no reviewer")
    if not str(plan.get("reason") or "").strip():
        raise ValueError("Apply plan has no reason")


def _validate_published(root: zarr.Group, desired: Mapping[str, Any]) -> None:
    subject = resolve_subject_metadata(root, allow_legacy=False)
    setup = resolve_experiment_setup(root, allow_legacy=False)
    if subject.record_sha256 != desired["subject_metadata_sha256"]:
        raise ValueError("Published subject authority differs from the reviewed plan")
    if subject.subject_ids or subject.subject_identity_kind != "none":
        raise ValueError("Published count-only authority contains subject identity")
    if setup.record_sha256 != desired["experiment_setup_sha256"]:
        raise ValueError("Published setup authority differs from the reviewed plan")
    if setup.assigned_subject_count is not None:
        raise ValueError("Published count-only setup has assigned subjects")
    if setup.subject_assignment_status != "count_only":
        raise ValueError("Published setup is not count_only")


def _delete_planned_placeholders(
    registry: Registry,
    *,
    row: Mapping[str, Any],
) -> tuple[int, int]:
    cleanup = row.get("cleanup")
    if not isinstance(cleanup, Mapping):
        raise ValueError("Reviewed cleanup plan is missing")
    memberships = cleanup.get("recording_subjects")
    subjects = cleanup.get("orphan_subjects")
    if not isinstance(memberships, list) or not isinstance(subjects, list):
        raise ValueError("Reviewed cleanup plan is malformed")

    deleted_memberships = 0
    deleted_subjects = 0
    with registry._transaction_context():
        for planned in memberships:
            current = registry.conn.execute(
                """
                SELECT dataset_id, metadata_json
                FROM recording_subjects
                WHERE recording_id = ? AND subject_id = ?;
                """,
                (planned["recording_id"], planned["subject_id"]),
            ).fetchone()
            if current is None:
                continue
            metadata = _decode_json_mapping(current["metadata_json"])
            if (
                not _is_placeholder_metadata(metadata)
                or _metadata_digest(metadata) != planned["metadata_sha256"]
                or (
                    str(current["dataset_id"])
                    if current["dataset_id"] is not None
                    else None
                )
                != planned.get("dataset_id")
            ):
                raise ValueError(
                    "Placeholder membership changed after review: "
                    f"{planned['recording_id']} / {planned['subject_id']}"
                )
            cursor = registry.conn.execute(
                "DELETE FROM recording_subjects "
                "WHERE recording_id = ? AND subject_id = ?",
                (planned["recording_id"], planned["subject_id"]),
            )
            deleted_memberships += int(cursor.rowcount or 0)

        for planned in subjects:
            current = registry.conn.execute(
                "SELECT metadata_json FROM subjects WHERE subject_id = ?",
                (planned["subject_id"],),
            ).fetchone()
            if current is None:
                continue
            metadata = _decode_json_mapping(current["metadata_json"])
            remaining = registry.conn.execute(
                "SELECT COUNT(*) FROM recording_subjects WHERE subject_id = ?",
                (planned["subject_id"],),
            ).fetchone()[0]
            if (
                str(metadata.get("source") or "").strip() != LEGACY_SOURCE
                or _metadata_digest(metadata) != planned["metadata_sha256"]
                or int(remaining) != 0
            ):
                raise ValueError(
                    "Placeholder subject changed after review: "
                    f"{planned['subject_id']}"
                )
            cursor = registry.conn.execute(
                "DELETE FROM subjects WHERE subject_id = ?",
                (planned["subject_id"],),
            )
            deleted_subjects += int(cursor.rowcount or 0)
    return deleted_memberships, deleted_subjects


def _validate_registry_context(
    registry: Registry,
    *,
    row: Mapping[str, Any],
) -> None:
    desired = row["desired"]
    expected = int(desired["expected_subject_count"])
    for sibling in row.get("source_datasets", []):
        context = registry.conn.execute(
            """
            SELECT subject_count_snapshot, subject_count_recorded,
                   subject_count_effective, subject_identity_status,
                   subject_id, subject_ids_json
            FROM dataset_context_current
            WHERE dataset_id = ?;
            """,
            (sibling["dataset_id"],),
        ).fetchone()
        if context is None:
            raise ValueError(f"Registry context missing for {sibling['dataset_id']}")
        if int(context["subject_count_effective"] or 0) != expected:
            raise ValueError(
                f"Registry subject count mismatch for {sibling['dataset_id']}"
            )
        if context["subject_count_recorded"] is not None:
            raise ValueError(
                f"Registry still has explicit membership for {sibling['dataset_id']}"
            )
        if context["subject_identity_status"] != "count_only":
            raise ValueError(
                f"Registry identity status is not count_only for {sibling['dataset_id']}"
            )
        if context["subject_id"] is not None or context["subject_ids_json"] is not None:
            raise ValueError(
                f"Registry still exposes subject identity for {sibling['dataset_id']}"
            )


def apply_plan(registry_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a digest-bound reviewed plan; errors remain recoverable by replay."""

    _verify_plan(plan)
    reviewer = str(plan["reviewer"])
    reason = str(plan["reason"])
    registry = Registry(registry_path)
    rows: list[dict[str, Any]] = []
    try:
        for planned in plan.get("recordings", []):
            row = dict(planned)
            if row.get("disposition") != "eligible":
                rows.append(row)
                continue
            target = CountOnlyTarget(
                dataset_id=str(row["dataset_id"]),
                recording_id=str(row["recording_id"]),
                zarr_path=Path(str(row["zarr_path"])),
            )
            try:
                fresh = plan_target(
                    registry_path,
                    target,
                    reviewer=reviewer,
                    reason=reason,
                )
                for key in ("desired", "cleanup", "source_datasets"):
                    if fresh.get(key) != row.get(key):
                        raise ValueError(
                            f"Apply-time {key} differs from the reviewed plan"
                        )

                desired = row["desired"]
                metadata = desired["subject_metadata_record"]["subject_metadata"]
                evidence_artifacts = [
                    {
                        "kind": "legacy_subject_metadata_snapshot",
                        "path": item["zarr_path"],
                        "metadata_path": item["subject_metadata_path"],
                        "sha256": item["subject_metadata_sha256"],
                    }
                    for item in row.get("evidence", [])
                ]
                root = _open_root(target.zarr_path, mode="r+")
                subject = publish_subject_metadata(
                    root,
                    metadata,
                    provenance_command=PROVENANCE_COMMAND,
                    provenance_params={
                        "assertion_kind": ASSERTION_KIND,
                        "reviewer": reviewer,
                        "reason": reason,
                        "plan_sha256": plan["plan_sha256"],
                    },
                    provenance_input_artifacts=evidence_artifacts,
                )
                if subject.record_sha256 != desired["subject_metadata_sha256"]:
                    raise ValueError("Published subject digest differs from plan")
                publish_experiment_setup(
                    root,
                    desired["experiment_setup_record"],
                    provenance_command=PROVENANCE_COMMAND,
                    provenance_params={
                        "assertion_kind": ASSERTION_KIND,
                        "reviewer": reviewer,
                        "reason": reason,
                        "plan_sha256": plan["plan_sha256"],
                    },
                    provenance_input_artifacts=evidence_artifacts,
                )
                _validate_published(root, desired)

                for sibling in row.get("source_datasets", []):
                    sibling_path = Path(sibling["zarr_path"])
                    sibling_root = _open_root(sibling_path, mode="r")
                    refreshed_id = registry.register_from_root(
                        sibling_root, sibling_path
                    )
                    if refreshed_id != sibling["dataset_id"]:
                        raise ValueError(
                            "Registry refresh changed dataset identity: "
                            f"{sibling['dataset_id']} -> {refreshed_id}"
                        )
                deleted_memberships, deleted_subjects = _delete_planned_placeholders(
                    registry, row=row
                )
                _validate_registry_context(registry, row=row)
                integrity = registry.conn.execute("PRAGMA integrity_check;").fetchone()[
                    0
                ]
                foreign_keys = registry.conn.execute(
                    "PRAGMA foreign_key_check;"
                ).fetchall()
                if integrity != "ok" or foreign_keys:
                    raise ValueError(
                        "Registry integrity failed after placeholder cleanup"
                    )
                row.update(
                    disposition="applied",
                    action=(
                        "verified_existing"
                        if planned.get("action") == "verify_existing"
                        else "published"
                    ),
                    reason=None,
                    detail=None,
                    deleted_recording_subjects=deleted_memberships,
                    deleted_orphan_subjects=deleted_subjects,
                )
            except Exception as exc:
                row.update(
                    disposition="error",
                    action="none",
                    reason="apply_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
    finally:
        registry.close()

    report = {
        "schema_id": REPORT_SCHEMA_ID,
        "created_utc": utc_now(),
        "mode": "apply",
        "reviewed_plan_sha256": plan["plan_sha256"],
        "registry_path": str(registry_path.expanduser().resolve(strict=False)),
        "reviewer": reviewer,
        "reason": reason,
        "dataset_count": len(rows),
        "recording_count": len({row["recording_id"] for row in rows}),
        "disposition_counts": dict(
            sorted(Counter(row["disposition"] for row in rows).items())
        ),
        "action_counts": dict(sorted(Counter(row["action"] for row in rows).items())),
        "reason_counts": dict(
            sorted(Counter(row["reason"] for row in rows if row["reason"]).items())
        ),
        "recordings": rows,
    }
    return report


def _read_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Apply plan must be a JSON object")
    plan = dict(payload)
    _verify_plan(plan)
    return plan


def _write_report(report: Mapping[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or apply migration of recording-local placeholder subjects "
            "to identity-free count-only subject/setup authority."
        )
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--path-contains", action="append", default=[])
    parser.add_argument("--all-placeholders", action="store_true")
    parser.add_argument("--reviewer")
    parser.add_argument("--reason")
    parser.add_argument(
        "--apply-plan",
        type=Path,
        help="Apply an unchanged digest-bound JSON plan produced by this command.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        help="New SQLite backup path required with --apply-plan.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    if args.apply_plan is not None:
        if args.backup is None:
            raise ValueError("--apply-plan requires --backup")
        if args.backup.exists():
            raise FileExistsError(f"Backup already exists: {args.backup}")
        plan = _read_plan(args.apply_plan)
        planned_registry = Path(str(plan["registry_path"])).resolve(strict=False)
        if registry_path.expanduser().resolve(strict=False) != planned_registry:
            raise ValueError("--registry differs from the reviewed plan")
        create_backup(registry_path, args.backup)
        report = apply_plan(registry_path, plan)
        report["registry_backup"] = str(args.backup.resolve(strict=False))
        _write_report(report, args.output)
        return 1 if report["disposition_counts"].get("error", 0) else 0

    if not str(args.reviewer or "").strip() or not str(args.reason or "").strip():
        raise ValueError("Dry-run planning requires --reviewer and --reason")
    targets = select_targets(
        registry_path,
        recording_ids=args.recording_id,
        path_contains=args.path_contains,
        all_placeholders=bool(args.all_placeholders),
    )
    plan = build_plan(
        registry_path,
        targets,
        reviewer=str(args.reviewer),
        reason=str(args.reason),
    )
    _write_report(plan, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSERTION_KIND",
    "CountOnlyTarget",
    "LEGACY_IDENTITY_SCOPE",
    "LEGACY_SOURCE",
    "MANUAL_AUTHORITY_SOURCE",
    "REPORT_SCHEMA_ID",
    "apply_plan",
    "build_plan",
    "plan_target",
    "select_targets",
]
