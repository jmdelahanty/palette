"""Read-only registry evaluation for typed cohort specifications."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping, Sequence

from fisheye.cohorts.spec import CohortSpec, canonical_json_bytes, canonical_sha256


PLAN_SCHEMA_ID = "palette.cohort_selection_plan"
PLAN_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_ID = "palette.frozen_cohort_manifest"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_CANONICALIZATION = "json_sorted_keys_no_manifest_sha256_v1"


class CohortSelectionError(RuntimeError):
    """Raised when a cohort cannot be selected or frozen safely."""


def _connect_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"registry not found: {resolved}")
    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("BEGIN")
    return conn


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _json_values(raw: Any, *, integer: bool = False) -> list[Any]:
    if raw is None:
        return []
    try:
        values = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    normalized: list[Any] = []
    for value in values:
        if value is None:
            continue
        if integer:
            try:
                item: Any = int(value)
            except (TypeError, ValueError):
                continue
        else:
            item = str(value).strip()
            if not item:
                continue
        if item not in normalized:
            normalized.append(item)
    return sorted(normalized)


def _placeholders(values: Sequence[Any]) -> str:
    return ",".join("?" for _ in values)


def _candidate_rows(conn: sqlite3.Connection, spec: CohortSpec) -> list[dict[str, Any]]:
    dataset = spec.dataset
    sql = f"""
        SELECT
            dcc.dataset_id,
            dcc.recording_id,
            dcc.session_uuid,
            dcc.zarr_path,
            dcc.zarr_origin,
            dcc.zarr_use,
            dcc.dataset_status,
            dcc.recording_name,
            dcc.recording_started_utc,
            dcc.protocol_name,
            dcc.protocol_hash,
            dcc.rig_id,
            dcc.arena_id,
            dcc.camera_id,
            dcc.subject_context_source,
            dcc.subject_count_effective,
            dcc.legacy_cross_id,
            dcc.legacy_genotype,
            dcc.legacy_line_strain,
            dcc.legacy_dpf_at_acquisition,
            (
                SELECT COUNT(DISTINCT rso.subject_id)
                FROM recording_subject_overview rso
                WHERE rso.recording_id = dcc.recording_id
                  AND rso.dpf_at_acquisition IS NOT NULL
            ) AS dpf_subject_count,
            (
                SELECT COUNT(DISTINCT rso.subject_id)
                FROM recording_subject_overview rso
                WHERE rso.recording_id = dcc.recording_id
                  AND NULLIF(TRIM(rso.line_strain), '') IS NOT NULL
            ) AS line_strain_subject_count,
            (
                SELECT COUNT(DISTINCT rso.subject_id)
                FROM recording_subject_overview rso
                WHERE rso.recording_id = dcc.recording_id
                  AND NULLIF(TRIM(rso.genotype), '') IS NOT NULL
            ) AS genotype_subject_count,
            (
                SELECT COUNT(DISTINCT rso.subject_id)
                FROM recording_subject_overview rso
                WHERE rso.recording_id = dcc.recording_id
                  AND NULLIF(TRIM(rso.cross_id), '') IS NOT NULL
            ) AS cross_id_subject_count,
            dcc.cross_ids_json,
            dcc.genotypes_json,
            dcc.line_strains_json,
            dcc.dpf_values_json
        FROM dataset_context_current dcc
        WHERE dcc.dataset_status IN ({_placeholders(dataset.statuses)})
          AND dcc.zarr_use IN ({_placeholders(dataset.zarr_uses)})
          AND dcc.zarr_origin IN ({_placeholders(dataset.zarr_origins)})
        ORDER BY
            COALESCE(dcc.recording_started_utc, ''),
            COALESCE(dcc.recording_id, ''),
            COALESCE(dcc.arena_id, ''),
            dcc.dataset_id
    """
    params = [*dataset.statuses, *dataset.zarr_uses, *dataset.zarr_origins]
    rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    return rows


def _stimulus_context(
    conn: sqlite3.Connection, dataset_ids: Sequence[str]
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "stimulus_run_ids": set(),
            "stimulus_modes": set(),
            "protocol_hashes": set(),
            "protocol_names": set(),
        }
    )
    if not dataset_ids:
        return {}
    batch_size = 400
    for start in range(0, len(dataset_ids), batch_size):
        batch = dataset_ids[start : start + batch_size]
        sql = f"""
            SELECT dataset_id, stimulus_run_id, stimulus_mode, protocol_hash, protocol_name
            FROM recording_stimulus_mode_counts
            WHERE is_latest = 1
              AND dataset_id IN ({_placeholders(batch)})
            ORDER BY dataset_id, stimulus_mode
        """
        for row in conn.execute(sql, list(batch)).fetchall():
            dataset_id = str(row["dataset_id"])
            run_id = str(row["stimulus_run_id"] or "").strip()
            if run_id:
                result[dataset_id]["stimulus_run_ids"].add(run_id)
            for target, column, transform in (
                ("stimulus_modes", "stimulus_mode", str.upper),
                ("protocol_hashes", "protocol_hash", str.lower),
                ("protocol_names", "protocol_name", lambda value: value),
            ):
                raw = row[column]
                if raw is not None and str(raw).strip():
                    result[dataset_id][target].add(transform(str(raw).strip()))
    return {
        dataset_id: {key: sorted(values) for key, values in context.items()}
        for dataset_id, context in result.items()
    }


def _step_context(
    conn: sqlite3.Connection, dataset_ids: Sequence[str], steps: Sequence[str]
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    if not dataset_ids or not steps:
        return {}
    batch_size = 400
    for start in range(0, len(dataset_ids), batch_size):
        batch = dataset_ids[start : start + batch_size]
        sql = f"""
            SELECT dataset_id, step_name, status
            FROM recording_step_status_latest
            WHERE dataset_id IN ({_placeholders(batch)})
              AND step_name IN ({_placeholders(steps)})
            ORDER BY dataset_id, step_name
        """
        for row in conn.execute(sql, [*batch, *steps]).fetchall():
            result[str(row["dataset_id"])][str(row["step_name"])] = str(row["status"])
    return dict(result)


def _subject_match(
    values: Sequence[Any],
    predicate: Callable[[Any], bool],
    *,
    policy: str,
    total_subjects: int,
    known_subjects: int,
) -> tuple[str, str | None]:
    if not values:
        return "missing", None
    if policy == "unambiguous_recording":
        if total_subjects > known_subjects:
            return (
                "incomplete",
                f"{known_subjects} of {total_subjects} subjects have this value",
            )
        if len(values) != 1:
            return "ambiguous", f"recording has {len(values)} distinct subject values"
        return ("match", None) if predicate(values[0]) else ("mismatch", None)
    if policy == "any_subject":
        return (
            ("match", None)
            if any(predicate(value) for value in values)
            else ("mismatch", None)
        )
    if policy == "all_subjects":
        if total_subjects > known_subjects:
            return (
                "incomplete",
                f"{known_subjects} of {total_subjects} subjects have this value",
            )
        return (
            ("match", None)
            if all(predicate(value) for value in values)
            else ("mismatch", None)
        )
    raise AssertionError(f"unhandled subject match policy: {policy}")


def _record_decision(
    row: Mapping[str, Any],
    *,
    stimulus: Mapping[str, Sequence[str]],
    step_statuses: Mapping[str, str],
    spec: CohortSpec,
    duplicate_dataset_ids: Sequence[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    exclusions: list[str] = []
    warnings: list[str] = []
    protocol = spec.protocol

    if not row.get("recording_id"):
        blockers.append("missing_recording_id")
    if not row.get("zarr_path"):
        blockers.append("missing_zarr_path")
    if len(duplicate_dataset_ids) > 1:
        blockers.append("multiple_candidate_datasets_for_recording")

    stimulus_run_ids = list(stimulus.get("stimulus_run_ids", []))
    if len(stimulus_run_ids) > 1:
        blockers.append("multiple_latest_stimulus_runs")
    modes = list(stimulus.get("stimulus_modes", []))
    hashes = list(stimulus.get("protocol_hashes", []))
    names = list(stimulus.get("protocol_names", []))
    fallback_hash = str(row.get("protocol_hash") or "").strip().lower()
    fallback_name = str(row.get("protocol_name") or "").strip()
    if not hashes and fallback_hash:
        hashes = [fallback_hash]
        warnings.append("protocol_hash_from_dataset_context_fallback")
    if not names and fallback_name:
        names = [fallback_name]
        warnings.append("protocol_name_from_dataset_context_fallback")

    def evaluate_protocol(
        values: Sequence[str], expected: Sequence[str], label: str
    ) -> None:
        if not expected:
            return
        if not values:
            # Exact protocol selectors define membership only over normalized
            # values actually present in the registry. A null protocol value
            # is not a match; it remains visible as an exclusion reason.
            exclusions.append(f"missing_{label}")
        elif not set(values).intersection(expected):
            exclusions.append(f"{label}_mismatch")

    evaluate_protocol(modes, protocol.stimulus_modes_any, "stimulus_mode")
    evaluate_protocol(hashes, protocol.protocol_hashes_any, "protocol_hash")
    evaluate_protocol(names, protocol.protocol_names_any, "protocol_name")

    subject = spec.subjects
    total_subjects = int(row.get("subject_count_effective") or 0)
    subject_fields = (
        (
            "dpf",
            _json_values(row.get("dpf_values_json"), integer=True),
            subject.dpf.active,
            lambda value: (
                (not subject.dpf.values or int(value) in subject.dpf.values)
                and (subject.dpf.minimum is None or int(value) >= subject.dpf.minimum)
                and (subject.dpf.maximum is None or int(value) <= subject.dpf.maximum)
            ),
            int(row.get("dpf_subject_count") or 0),
        ),
        (
            "line_strain",
            _json_values(row.get("line_strains_json")),
            bool(subject.line_strains_any),
            lambda value: str(value) in subject.line_strains_any,
            int(row.get("line_strain_subject_count") or 0),
        ),
        (
            "genotype",
            _json_values(row.get("genotypes_json")),
            bool(subject.genotypes_any),
            lambda value: str(value) in subject.genotypes_any,
            int(row.get("genotype_subject_count") or 0),
        ),
        (
            "cross_id",
            _json_values(row.get("cross_ids_json")),
            bool(subject.cross_ids_any),
            lambda value: str(value) in subject.cross_ids_any,
            int(row.get("cross_id_subject_count") or 0),
        ),
    )
    subject_values: dict[str, list[Any]] = {}
    subject_value_counts: dict[str, dict[str, int]] = {}
    for label, values, active, predicate, known_subjects in subject_fields:
        subject_values[label] = values
        subject_value_counts[label] = {
            "known_subject_count": known_subjects,
            "total_subject_count": total_subjects,
        }
        if not active:
            continue
        outcome, detail = _subject_match(
            values,
            predicate,
            policy=subject.match_policy,
            total_subjects=total_subjects,
            known_subjects=known_subjects,
        )
        if outcome == "match":
            continue
        if outcome == "mismatch":
            exclusions.append(f"{label}_mismatch")
        elif outcome == "missing":
            target = (
                blockers if spec.missing_selected_metadata == "error" else exclusions
            )
            target.append(f"missing_{label}_metadata")
        elif outcome == "ambiguous":
            blockers.append(f"ambiguous_{label}_metadata")
            if detail:
                warnings.append(f"{label}: {detail}")
        elif outcome == "incomplete":
            target = (
                blockers if spec.missing_selected_metadata == "error" else exclusions
            )
            target.append(f"incomplete_{label}_metadata")
            if detail:
                warnings.append(f"{label}: {detail}")

    legacy_subject_values = {
        "dpf": (
            [int(row["legacy_dpf_at_acquisition"])]
            if row.get("legacy_dpf_at_acquisition") is not None
            else []
        ),
        "line_strain": (
            [str(row["legacy_line_strain"]).strip()]
            if str(row.get("legacy_line_strain") or "").strip()
            else []
        ),
        "genotype": (
            [str(row["legacy_genotype"]).strip()]
            if str(row.get("legacy_genotype") or "").strip()
            else []
        ),
        "cross_id": (
            [str(row["legacy_cross_id"]).strip()]
            if str(row.get("legacy_cross_id") or "").strip()
            else []
        ),
    }

    for step in spec.prerequisites.required_steps_ok:
        status = step_statuses.get(step)
        if status != "ok":
            blockers.append(f"required_step_not_ok:{step}:{status or 'missing'}")

    # A true predicate mismatch means the row is outside the cohort.  Blockers
    # are only release-stopping for rows that otherwise match the definition.
    if exclusions:
        decision = "excluded"
    elif blockers:
        decision = "blocked"
    else:
        decision = "included"

    return {
        "decision": decision,
        "dataset_id": str(row.get("dataset_id") or ""),
        "recording_id": str(row.get("recording_id") or ""),
        "recording_name": row.get("recording_name"),
        "recording_started_utc": row.get("recording_started_utc"),
        "zarr_path": str(row.get("zarr_path") or ""),
        "zarr_origin": row.get("zarr_origin"),
        "zarr_use": row.get("zarr_use"),
        "dataset_status": row.get("dataset_status"),
        "rig_id": row.get("rig_id"),
        "arena_id": row.get("arena_id"),
        "camera_id": row.get("camera_id"),
        "subject_context_source": row.get("subject_context_source"),
        "subject_count_effective": row.get("subject_count_effective"),
        "subject_values": subject_values,
        "subject_value_counts": subject_value_counts,
        "legacy_subject_values": legacy_subject_values,
        "stimulus_run_ids": stimulus_run_ids,
        "stimulus_modes": modes,
        "protocol_hashes": hashes,
        "protocol_names": names,
        "required_step_statuses": dict(sorted(step_statuses.items())),
        "duplicate_candidate_dataset_ids": list(duplicate_dataset_ids),
        "blockers": blockers,
        "exclusions": exclusions,
        "warnings": warnings,
    }


def build_cohort_plan(registry_path: str | Path, spec: CohortSpec) -> dict[str, Any]:
    registry = Path(registry_path).expanduser().resolve()
    conn = _connect_readonly(registry)
    try:
        candidates = _candidate_rows(conn, spec)
        ids = [str(row["dataset_id"]) for row in candidates]
        stimuli = _stimulus_context(conn, ids)
        steps = _step_context(conn, ids, spec.prerequisites.required_steps_ok)
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        schema_version_row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        schema_version = schema_version_row[0] if schema_version_row else None
    finally:
        conn.close()

    by_recording: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        by_recording[str(row.get("recording_id") or "")].append(str(row["dataset_id"]))

    decisions = [
        _record_decision(
            row,
            stimulus=stimuli.get(str(row["dataset_id"]), {}),
            step_statuses=steps.get(str(row["dataset_id"]), {}),
            spec=spec,
            duplicate_dataset_ids=by_recording[str(row.get("recording_id") or "")],
        )
        for row in candidates
    ]
    counts = Counter(row["decision"] for row in decisions)
    blocker_reasons = Counter(
        reason
        for row in decisions
        if row["decision"] == "blocked"
        for reason in row["blockers"]
    )
    exclusion_reasons = Counter(
        reason
        for row in decisions
        if row["decision"] == "excluded"
        for reason in row["exclusions"]
    )
    snapshot_payload = {
        "candidate_rows": candidates,
        "stimulus_context": stimuli,
        "step_context": steps,
    }
    return {
        "schema_id": PLAN_SCHEMA_ID,
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "cohort_id": spec.cohort_id,
        "cohort_name": spec.cohort_name,
        "purpose": spec.purpose,
        "cohort_query": spec.to_mapping(),
        "cohort_query_sha256": spec.sha256,
        "registry": {
            "path": str(registry),
            "sqlite_user_version": user_version,
            "schema_version": schema_version,
            "query_snapshot_sha256": canonical_sha256(snapshot_payload),
            "access_mode": "read_only",
        },
        "selection_policy": {
            "include_every_match": True,
            "limit": None,
            "ordering": [
                "recording_started_utc",
                "recording_id",
                "arena_id",
                "dataset_id",
            ],
            "duplicate_recording_dataset_policy": "error",
            "missing_selected_metadata": spec.missing_selected_metadata,
            "subject_match_policy": spec.subjects.match_policy,
        },
        "summary": {
            "candidate_dataset_count": len(decisions),
            "candidate_recording_count": len(by_recording),
            "included_count": counts["included"],
            "excluded_count": counts["excluded"],
            "blocked_count": counts["blocked"],
            "blocker_reasons": dict(sorted(blocker_reasons.items())),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        },
        "records": decisions,
    }


def _manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return payload


def compute_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_manifest_hash_payload(manifest))
    ).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_frozen_cohort(
    manifest: Mapping[str, Any], *, check_hash: bool = True
) -> list[str]:
    """Return validation errors for an immutable frozen-cohort manifest."""

    errors: list[str] = []
    if manifest.get("schema_id") != MANIFEST_SCHEMA_ID:
        errors.append(f"schema_id must be {MANIFEST_SCHEMA_ID!r}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    if manifest.get("manifest_canonicalization") != MANIFEST_CANONICALIZATION:
        errors.append(
            f"manifest_canonicalization must be {MANIFEST_CANONICALIZATION!r}"
        )
    for key in ("created_utc", "cohort_id", "cohort_name", "cohort_query_sha256"):
        if not str(manifest.get(key) or "").strip():
            errors.append(f"{key} is required")
    query = manifest.get("cohort_query")
    if not isinstance(query, Mapping):
        errors.append("cohort_query must be a mapping")
    elif manifest.get("cohort_query_sha256") != canonical_sha256(query):
        errors.append("cohort_query_sha256 mismatch")
    registry = manifest.get("registry")
    if not isinstance(registry, Mapping):
        errors.append("registry must be a mapping")
    else:
        snapshot = str(registry.get("query_snapshot_sha256") or "")
        if not _is_sha256_hex(snapshot):
            errors.append("registry.query_snapshot_sha256 must be a SHA-256 hex value")
        if registry.get("access_mode") != "read_only":
            errors.append("registry.access_mode must be 'read_only'")
    policy = manifest.get("selection_policy")
    if not isinstance(policy, Mapping):
        errors.append("selection_policy must be a mapping")
    else:
        if policy.get("include_every_match") is not True:
            errors.append("selection_policy.include_every_match must be true")
        if policy.get("limit") is not None:
            errors.append("selection_policy.limit must be null")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        errors.append("members must be a non-empty list")
        members = []
    if manifest.get("member_count") != len(members):
        errors.append("member_count does not match members")
    seen_dataset_ids: set[str] = set()
    seen_recording_ids: set[str] = set()
    seen_zarr_paths: set[str] = set()
    for index, member in enumerate(members):
        if not isinstance(member, Mapping):
            errors.append(f"members[{index}] must be a mapping")
            continue
        for key, seen in (
            ("dataset_id", seen_dataset_ids),
            ("recording_id", seen_recording_ids),
            ("zarr_path", seen_zarr_paths),
        ):
            value = str(member.get(key) or "").strip()
            if not value:
                errors.append(f"members[{index}].{key} is required")
            elif value in seen:
                errors.append(f"members[{index}].{key} is duplicated: {value}")
            else:
                seen.add(value)
        for key in ("zarr_origin", "zarr_use", "dataset_status"):
            if not str(member.get(key) or "").strip():
                errors.append(f"members[{index}].{key} is required")
    summary = manifest.get("selection_summary")
    if not isinstance(summary, Mapping):
        errors.append("selection_summary must be a mapping")
    else:
        if int(summary.get("included_count") or 0) != len(members):
            errors.append("selection_summary.included_count does not match members")
        if int(summary.get("blocked_count") or 0) != 0:
            errors.append("selection_summary.blocked_count must be zero")
    if check_hash and manifest.get("manifest_sha256") != compute_manifest_sha256(
        manifest
    ):
        errors.append("manifest_sha256 mismatch")
    return errors


def freeze_cohort(plan: Mapping[str, Any]) -> dict[str, Any]:
    summary = plan.get("summary")
    if not isinstance(summary, Mapping):
        raise CohortSelectionError("selection plan is missing its summary")
    blocked = int(summary.get("blocked_count") or 0)
    included = int(summary.get("included_count") or 0)
    if blocked:
        raise CohortSelectionError(
            f"cohort freeze refused: {blocked} otherwise-matching dataset(s) are blocked"
        )
    if included == 0:
        raise CohortSelectionError(
            "cohort freeze refused: query selected zero datasets"
        )
    records = plan.get("records")
    if not isinstance(records, list):
        raise CohortSelectionError("selection plan is missing records")
    members = [
        {
            key: row.get(key)
            for key in (
                "dataset_id",
                "recording_id",
                "recording_name",
                "recording_started_utc",
                "zarr_path",
                "zarr_origin",
                "zarr_use",
                "dataset_status",
                "rig_id",
                "arena_id",
                "camera_id",
                "subject_context_source",
                "subject_count_effective",
                "subject_values",
                "subject_value_counts",
                "stimulus_run_ids",
                "stimulus_modes",
                "protocol_hashes",
                "protocol_names",
                "required_step_statuses",
            )
        }
        for row in records
        if isinstance(row, Mapping) and row.get("decision") == "included"
    ]
    manifest: dict[str, Any] = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_canonicalization": MANIFEST_CANONICALIZATION,
        "created_utc": _utc_now(),
        "cohort_id": plan.get("cohort_id"),
        "cohort_name": plan.get("cohort_name"),
        "purpose": plan.get("purpose"),
        "cohort_query": plan.get("cohort_query"),
        "cohort_query_sha256": plan.get("cohort_query_sha256"),
        "registry": plan.get("registry"),
        "selection_policy": plan.get("selection_policy"),
        "member_count": len(members),
        "members": members,
        "selection_summary": summary,
    }
    manifest["manifest_sha256"] = compute_manifest_sha256(manifest)
    errors = validate_frozen_cohort(manifest, check_hash=True)
    if errors:
        raise CohortSelectionError(
            "internal frozen cohort validation failed: " + "; ".join(errors)
        )
    return manifest


def coverage_report(
    registry_path: str | Path,
    spec: CohortSpec,
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if plan is None:
        plan = build_cohort_plan(registry_path, spec)
    elif plan.get("cohort_query_sha256") != spec.sha256:
        raise CohortSelectionError("coverage plan does not match the cohort query")
    base_records = plan["records"]
    protocol_exclusion_prefixes = (
        "stimulus_mode_",
        "protocol_hash_",
        "protocol_name_",
        "missing_stimulus_mode",
        "missing_protocol_hash",
        "missing_protocol_name",
    )
    records = [
        row
        for row in base_records
        if not any(
            reason.startswith(protocol_exclusion_prefixes)
            for reason in row["exclusions"]
        )
    ]
    fields = ("dpf", "line_strain", "genotype", "cross_id")
    coverage: dict[str, Any] = {}
    for field in fields:
        present = [row for row in records if row["subject_values"][field]]
        ambiguous = [row for row in present if len(row["subject_values"][field]) > 1]
        legacy_candidates = [
            row
            for row in records
            if not row["subject_values"][field] and row["legacy_subject_values"][field]
        ]
        values = Counter(
            str(value) for row in records for value in row["subject_values"][field]
        )
        legacy_values = Counter(
            str(value)
            for row in legacy_candidates
            for value in row["legacy_subject_values"][field]
        )
        coverage[field] = {
            "candidate_count": len(records),
            "present_count": len(present),
            "missing_count": len(records) - len(present),
            "ambiguous_count": len(ambiguous),
            "legacy_provenance_candidate_count": len(legacy_candidates),
            "unavailable_in_registry_count": (
                len(records) - len(present) - len(legacy_candidates)
            ),
            "values": dict(sorted(values.items())),
            "legacy_provenance_values": dict(sorted(legacy_values.items())),
        }
    source_counts = Counter(
        str(row["subject_context_source"] or "missing") for row in records
    )
    return {
        "schema_id": "palette.cohort_metadata_coverage",
        "schema_version": 1,
        "created_utc": _utc_now(),
        "cohort_id": spec.cohort_id,
        "cohort_query_sha256": spec.sha256,
        "registry": plan["registry"],
        "base_candidate_dataset_count": len(base_records),
        "protocol_matched_dataset_count": len(records),
        "subject_context_source_counts": dict(sorted(source_counts.items())),
        "subject_metadata": coverage,
        "selection_summary": plan["summary"],
    }
