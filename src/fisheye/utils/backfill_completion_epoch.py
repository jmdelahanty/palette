"""Backfill strict Zarr run-completion epochs after verifying legacy runs."""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
from fisheye.shared.json_safety import write_jsonl_atomic
import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional

import zarr

from fisheye.shared.zarr.stage_arrays import STAGES, StageSpec, validate_run
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STAGE_ATTR,
    has_run_completion_contract,
    mark_run_complete,
)

FINALIZED_RUNS_PATH = "experiment_index/finalized_runs"
REPORT_SCHEMA_ID = "palette.backfill_completion_epoch_report.v1"
BLOCKED_PARENT_ROW_SCHEMA_ID = "palette.backfill_completion_epoch_blocked_parent.v1"
WRITE_FAILED_PARENT_ROW_SCHEMA_ID = "palette.backfill_completion_epoch_write_failed_parent.v1"
DEPRECATED_COMPLETION_BACKFILL_SCOPES = frozenset({"eye_masks", "refined_eye_masks"})


def _group_keys(group: Any) -> list[str]:
    try:
        if hasattr(group, "group_keys"):
            return sorted(str(name) for name in group.group_keys())
        return sorted(str(name) for name, value in group.items() if hasattr(value, "attrs"))
    except Exception:
        return []


def _is_group_like(value: Any) -> bool:
    return hasattr(value, "attrs") and hasattr(value, "__contains__") and hasattr(value, "__getitem__")


def _get_child_group(parent: Any, name: str) -> Any | None:
    parts = [part for part in str(name).split("/") if part]
    if len(parts) > 1:
        child = parent
        for part in parts:
            child = _get_child_group(child, part)
            if child is None:
                return None
        return child

    try:
        child = parent[parts[0] if parts else str(name)]
    except Exception:
        return None
    return child if _is_group_like(child) else None


def _stage_parent_prefix(spec: StageSpec) -> str:
    parts: list[str] = []
    for part in spec.zarr_group.strip("/").split("/"):
        if part.startswith("<") and part.endswith(">"):
            break
        parts.append(part)
    return "/".join(parts)


def _stage_parent_templates() -> dict[str, StageSpec]:
    templates: dict[str, StageSpec] = {}
    for spec in STAGES.values():
        if spec.stage_name == "detect_quality":
            continue
        prefix = _stage_parent_prefix(spec)
        if prefix:
            templates[prefix] = spec
    return templates


STAGE_PARENT_TEMPLATES = _stage_parent_templates()
STAGE_NAMES = frozenset(spec.stage_name for spec in STAGES.values())


def _spec_for_parent_path(parent_path: str) -> StageSpec | None:
    if parent_path in STAGE_PARENT_TEMPLATES:
        return STAGE_PARENT_TEMPLATES[parent_path]

    # Nested detect quality reports live under each detect run.
    parts = [part for part in parent_path.split("/") if part]
    if len(parts) == 3 and parts[0] == "detect_runs" and parts[2] == "quality_reports":
        return STAGES.get("detect_quality")
    return None


def _normalize_filter_values(values: Sequence[str] | None) -> frozenset[str]:
    if not values:
        return frozenset()
    return frozenset(str(value).strip("/") for value in values if str(value).strip("/"))


def _parent_matches_filters(
    parent_path: str,
    spec: StageSpec | None,
    *,
    selected_stages: frozenset[str] = frozenset(),
    selected_parent_paths: frozenset[str] = frozenset(),
) -> bool:
    if selected_parent_paths and parent_path.strip("/") not in selected_parent_paths:
        return False
    if selected_stages and (spec is None or spec.stage_name not in selected_stages):
        return False
    return True


def _iter_run_parents(group: Any, *, path: str = "", max_depth: int = 6) -> list[tuple[str, Any]]:
    if max_depth < 0:
        return []

    out: list[tuple[str, Any]] = []
    for name in _group_keys(group):
        child = _get_child_group(group, name)
        if child is None:
            continue
        child_path = f"{path}/{name}" if path else name
        if child_path == FINALIZED_RUNS_PATH:
            out.append((child_path, child))
            continue
        if name.endswith("_runs"):
            out.append((child_path, child))
        out.extend(_iter_run_parents(child, path=child_path, max_depth=max_depth - 1))
    return out


def _completion_status(group: Any) -> str | None:
    value = getattr(group, "attrs", {}).get(RUN_COMPLETION_STATUS_ATTR)
    return str(value).lower() if value is not None else None


def _run_name(report: Mapping[str, Any]) -> str | None:
    value = report.get("run_name")
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _protected_legacy_run_names(parent: Any) -> frozenset[str]:
    """Run names that must not be treated as ignorable legacy debris."""

    protected: set[str] = set()
    attrs = getattr(parent, "attrs", {})
    for attr_name in ("latest", "latest_complete"):
        value = attrs.get(attr_name)
        if value is None:
            continue
        text = str(value)
        if text:
            protected.add(text)
    return frozenset(protected)


def _is_ignorable_legacy_invalid_child(
    child_report: Mapping[str, Any],
    *,
    protected_run_names: frozenset[str],
) -> bool:
    if child_report.get("verification") != "invalid":
        return False
    run_name = _run_name(child_report)
    if run_name is None:
        return False
    if not protected_run_names:
        return False
    return run_name not in protected_run_names


def _is_blocking_child_report(child_report: Mapping[str, Any]) -> bool:
    if bool(child_report.get("ignored_for_parent_epoch")):
        return False
    return child_report.get("verification") in {"unverified", "invalid"}


def _parent_is_strict(parent: Any) -> bool:
    value = getattr(parent, "attrs", {}).get(COMPLETION_EPOCH_ATTR)
    try:
        return int(value) >= COMPLETION_EPOCH_STRICT
    except (TypeError, ValueError):
        return False


def _verify_child(
    child: Any,
    *,
    child_name: str,
    parent_path: str,
    spec: StageSpec | None,
) -> dict[str, Any]:
    if has_run_completion_contract(child):
        return {
            "run_name": child_name,
            "verification": "has_contract",
            "completion_status": _completion_status(child),
            "marked_complete": False,
        }

    if spec is None:
        return {
            "run_name": child_name,
            "verification": "unverified",
            "reason": "no_stage_array_spec",
            "marked_complete": False,
        }

    result = validate_run(child, spec)
    if not result.valid:
        return {
            "run_name": child_name,
            "verification": "invalid",
            "stage": spec.stage_name,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "marked_complete": False,
        }

    return {
        "run_name": child_name,
        "verification": "validated_legacy_complete",
        "stage": spec.stage_name,
        "warnings": list(result.warnings),
        "marked_complete": False,
        "contract_attr": RUN_COMPLETION_CONTRACT_ATTR,
    }


def _iter_verifiable_children(
    parent_path: str,
    parent: Any,
    spec: StageSpec | None,
) -> list[tuple[str, Any]]:
    if spec is not None and spec.stage_name == "track_kinematics":
        children: list[tuple[str, Any]] = []
        for scope_name in _group_keys(parent):
            scope = _get_child_group(parent, scope_name)
            if scope is None:
                continue
            for run_name in _group_keys(scope):
                child = _get_child_group(scope, run_name)
                if child is not None:
                    children.append((f"{scope_name}/{run_name}", child))
        return children

    return [
        (child_name, child)
        for child_name in _group_keys(parent)
        if (child := _get_child_group(parent, child_name)) is not None
    ]


def _summarize_parent(
    parent_path: str,
    parent: Any,
    *,
    apply: bool,
    timestamp_utc: str,
    selected_stages: frozenset[str] = frozenset(),
    selected_parent_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if parent_path == FINALIZED_RUNS_PATH:
        return {
            "parent_path": parent_path,
            "status": "skipped",
            "reason": "collection_index_not_stage_run_parent",
            "child_count": len(_group_keys(parent)),
        }

    spec = _spec_for_parent_path(parent_path)
    if not _parent_matches_filters(
        parent_path,
        spec,
        selected_stages=selected_stages,
        selected_parent_paths=selected_parent_paths,
    ):
        return {
            "parent_path": parent_path,
            "status": "filtered",
            "reason": "filter_mismatch",
            "stage": spec.stage_name if spec is not None else None,
            "stage_spec_available": spec is not None,
            "child_count": len(_group_keys(parent)),
            "filter": {
                "stages": sorted(selected_stages),
                "parent_paths": sorted(selected_parent_paths),
            },
        }

    children = _iter_verifiable_children(parent_path, parent, spec)
    child_reports = [
        _verify_child(
            child,
            child_name=child_name,
            parent_path=parent_path,
            spec=spec,
        )
        for child_name, child in children
    ]

    protected_run_names = _protected_legacy_run_names(parent)
    for child_report in child_reports:
        if _is_ignorable_legacy_invalid_child(
            child_report,
            protected_run_names=protected_run_names,
        ):
            child_report["ignored_for_parent_epoch"] = True
            child_report["ignore_reason"] = "non_latest_legacy_contract_mismatch"

    blocking_children = [
        report
        for report in child_reports
        if _is_blocking_child_report(report)
    ]
    ignored_legacy_children = [
        report
        for report in child_reports
        if bool(report.get("ignored_for_parent_epoch"))
    ]
    completion_epoch_before = getattr(parent, "attrs", {}).get(COMPLETION_EPOCH_ATTR)
    already_strict = _parent_is_strict(parent)
    can_stamp = not blocking_children
    would_mark_child_count = (
        sum(
            1
            for item in child_reports
            if item.get("verification") == "validated_legacy_complete"
        )
        if can_stamp
        else 0
    )
    stamped = False
    write_failure: dict[str, Any] | None = None
    if apply and can_stamp:
        for report in child_reports:
            if report.get("verification") != "validated_legacy_complete":
                continue
            child_name = str(report["run_name"])
            child = _get_child_group(parent, child_name)
            if child is None:
                continue
            try:
                mark_run_complete(
                    child,
                    run_name=child_name,
                    completed_at_utc=timestamp_utc,
                    allow_missing_run_provenance=True,
                    missing_run_provenance_reason=(
                        "legacy completion backfill re-mark of a pre-provenance run"
                    ),
                )
                report["marked_complete"] = True
            except Exception as exc:
                write_failure = {
                    "write_error_phase": "mark_child_complete",
                    "write_error_run_name": child_name,
                    "write_error": str(exc),
                    "write_error_type": type(exc).__name__,
                }
                break
            if report.get("stage"):
                try:
                    child.attrs[RUN_STAGE_ATTR] = str(report["stage"])
                except Exception as exc:
                    write_failure = {
                        "write_error_phase": "mark_child_stage",
                        "write_error_run_name": child_name,
                        "write_error": str(exc),
                        "write_error_type": type(exc).__name__,
                    }
                    break

    if apply and can_stamp and write_failure is None and not already_strict:
        try:
            parent.attrs[COMPLETION_EPOCH_ATTR] = COMPLETION_EPOCH_STRICT
            stamped = True
        except Exception as exc:
            write_failure = {
                "write_error_phase": "stamp_parent_epoch",
                "write_error": str(exc),
                "write_error_type": type(exc).__name__,
            }

    if write_failure is not None:
        status = "write_failed"
    elif already_strict:
        status = "already_strict"
    elif can_stamp:
        status = "stamped" if stamped else "would_stamp"
    else:
        status = "blocked"

    latest = getattr(parent, "attrs", {}).get("latest")
    latest_complete = getattr(parent, "attrs", {}).get("latest_complete")
    report: dict[str, Any] = {
        "parent_path": parent_path,
        "status": status,
        "stage": spec.stage_name if spec is not None else None,
        "stage_spec_available": spec is not None,
        "child_count": len(child_reports),
        "verified_child_count": (
            len(child_reports) - len(blocking_children) - len(ignored_legacy_children)
        ),
        "unverified_child_count": len(blocking_children),
        "ignored_legacy_child_count": len(ignored_legacy_children),
        "would_mark_child_count": would_mark_child_count,
        "marked_child_count": sum(1 for item in child_reports if item.get("marked_complete")),
        "completion_epoch_before": completion_epoch_before,
        "completion_epoch_after": getattr(parent, "attrs", {}).get(COMPLETION_EPOCH_ATTR),
        "latest": latest,
        "latest_complete": latest_complete,
        "children": child_reports,
    }
    if write_failure is not None:
        report.update(write_failure)
    return report


def _stage_or_parent_key(parent_report: Mapping[str, Any]) -> str:
    stage = parent_report.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    parent_path = parent_report.get("parent_path")
    return str(parent_path or "unknown")


def _blocked_child_reason(child_report: Mapping[str, Any]) -> str:
    reason = child_report.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    verification = child_report.get("verification")
    if isinstance(verification, str) and verification:
        return verification
    return "unknown"


def _first_error(child_report: Mapping[str, Any]) -> str | None:
    errors = child_report.get("errors")
    if isinstance(errors, list) and errors:
        return str(errors[0])
    return None


def _sorted_counter(counter: Counter[str], *, limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(key): int(value) for key, value in items}


def _ranked_counter(counter: Counter[str], *, limit: int | None = None) -> list[dict[str, Any]]:
    return [
        {"key": str(key), "count": int(value)}
        for key, value in counter.most_common(limit)
    ]


def _scope_filter_hint(scope_key: str) -> dict[str, str]:
    if scope_key in STAGE_NAMES:
        return {"stage": scope_key}
    return {"parent_path": scope_key}


def _build_backfill_scope_plan(
    *,
    blocked_by_stage_or_path: Counter[str],
    would_stamp_by_stage_or_path: Counter[str],
    stamped_by_stage_or_path: Counter[str],
    write_failed_by_stage_or_path: Counter[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(
        set(blocked_by_stage_or_path)
        | set(would_stamp_by_stage_or_path)
        | set(stamped_by_stage_or_path)
        | set(write_failed_by_stage_or_path)
    )
    for key in keys:
        blocked = int(blocked_by_stage_or_path.get(key, 0))
        would_stamp = int(would_stamp_by_stage_or_path.get(key, 0))
        stamped = int(stamped_by_stage_or_path.get(key, 0))
        write_failed = int(write_failed_by_stage_or_path.get(key, 0))
        if key in DEPRECATED_COMPLETION_BACKFILL_SCOPES:
            recommendation = "deprecated_scope_not_backfilled"
        elif write_failed > 0:
            recommendation = "write_failed_retry_after_fix"
        elif would_stamp > 0 and blocked == 0:
            recommendation = "ready_to_apply_if_approved"
        elif would_stamp > 0 and blocked > 0:
            recommendation = "partially_blocked"
        elif blocked > 0:
            recommendation = "blocked_triage_required"
        else:
            recommendation = "already_stamped_or_no_action"
        rows.append(
            {
                "key": key,
                "recommendation": recommendation,
                "blocked_parent_count": blocked,
                "would_stamp_parent_count": would_stamp,
                "stamped_parent_count": stamped,
                "write_failed_parent_count": write_failed,
                "filter": _scope_filter_hint(key),
            }
        )
    recommendation_priority = {
        "write_failed_retry_after_fix": 0,
        "ready_to_apply_if_approved": 1,
        "partially_blocked": 2,
        "blocked_triage_required": 3,
        "deprecated_scope_not_backfilled": 4,
        "already_stamped_or_no_action": 5,
    }
    return sorted(
        rows,
        key=lambda row: (
            recommendation_priority.get(str(row["recommendation"]), 99),
            -int(row["write_failed_parent_count"]),
            -int(row["would_stamp_parent_count"]),
            -int(row["blocked_parent_count"]),
            str(row["key"]),
        ),
    )


def _build_summary(stores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_status_counts: Counter[str] = Counter()
    blocked_by_stage_or_path: Counter[str] = Counter()
    would_stamp_by_stage_or_path: Counter[str] = Counter()
    stamped_by_stage_or_path: Counter[str] = Counter()
    write_failed_by_stage_or_path: Counter[str] = Counter()
    blocked_child_reason_counts: Counter[str] = Counter()
    blocked_child_first_error_counts: Counter[str] = Counter()
    ignored_legacy_child_reason_counts: Counter[str] = Counter()
    ignored_legacy_child_first_error_counts: Counter[str] = Counter()
    blocked_examples: list[dict[str, Any]] = []
    ignored_legacy_examples: list[dict[str, Any]] = []
    write_failed_examples: list[dict[str, Any]] = []

    for store in stores:
        zarr_path = str(store.get("zarr_path") or "")
        parents = store.get("parents")
        if not isinstance(parents, list):
            continue
        for parent in parents:
            if not isinstance(parent, Mapping):
                continue
            status = str(parent.get("status") or "unknown")
            key = _stage_or_parent_key(parent)
            parent_status_counts[status] += 1
            children = parent.get("children")
            if isinstance(children, list):
                ignored_children = [
                    child
                    for child in children
                    if isinstance(child, Mapping)
                    and bool(child.get("ignored_for_parent_epoch"))
                ]
                for child in ignored_children:
                    ignored_legacy_child_reason_counts[_blocked_child_reason(child)] += 1
                    first_error = _first_error(child)
                    if first_error:
                        ignored_legacy_child_first_error_counts[first_error] += 1
                if ignored_children and len(ignored_legacy_examples) < 25:
                    ignored_legacy_examples.append(
                        {
                            "zarr_path": zarr_path,
                            "parent_path": parent.get("parent_path"),
                            "stage": parent.get("stage"),
                            "latest": parent.get("latest"),
                            "latest_complete": parent.get("latest_complete"),
                            "ignored_legacy_child_count": len(ignored_children),
                        }
                    )
            if status == "blocked":
                blocked_by_stage_or_path[key] += 1
                if len(blocked_examples) < 25:
                    blocked_examples.append(
                        {
                            "zarr_path": zarr_path,
                            "parent_path": parent.get("parent_path"),
                            "stage": parent.get("stage"),
                            "child_count": parent.get("child_count"),
                            "unverified_child_count": parent.get("unverified_child_count"),
                        }
                    )
                if isinstance(children, list):
                    for child in children:
                        if not isinstance(child, Mapping):
                            continue
                        if not _is_blocking_child_report(child):
                            continue
                        blocked_child_reason_counts[_blocked_child_reason(child)] += 1
                        first_error = _first_error(child)
                        if first_error:
                            blocked_child_first_error_counts[first_error] += 1
            elif status == "would_stamp":
                would_stamp_by_stage_or_path[key] += 1
            elif status == "stamped":
                stamped_by_stage_or_path[key] += 1
            elif status == "write_failed":
                write_failed_by_stage_or_path[key] += 1
                if len(write_failed_examples) < 25:
                    write_failed_examples.append(
                        {
                            "zarr_path": zarr_path,
                            "parent_path": parent.get("parent_path"),
                            "stage": parent.get("stage"),
                            "child_count": parent.get("child_count"),
                            "marked_child_count": parent.get("marked_child_count"),
                            "write_error_phase": parent.get("write_error_phase"),
                            "write_error_run_name": parent.get("write_error_run_name"),
                            "write_error_type": parent.get("write_error_type"),
                        }
                    )

    return {
        "parent_status_counts": _sorted_counter(parent_status_counts),
        "blocked_parent_counts_by_stage_or_path": _sorted_counter(blocked_by_stage_or_path),
        "blocked_parent_ranked_by_stage_or_path": _ranked_counter(blocked_by_stage_or_path),
        "would_stamp_parent_counts_by_stage_or_path": _sorted_counter(would_stamp_by_stage_or_path),
        "would_stamp_parent_ranked_by_stage_or_path": _ranked_counter(would_stamp_by_stage_or_path),
        "stamped_parent_counts_by_stage_or_path": _sorted_counter(stamped_by_stage_or_path),
        "stamped_parent_ranked_by_stage_or_path": _ranked_counter(stamped_by_stage_or_path),
        "write_failed_parent_counts_by_stage_or_path": _sorted_counter(write_failed_by_stage_or_path),
        "write_failed_parent_ranked_by_stage_or_path": _ranked_counter(write_failed_by_stage_or_path),
        "blocked_child_reason_counts": _sorted_counter(blocked_child_reason_counts),
        "blocked_child_reason_ranked": _ranked_counter(blocked_child_reason_counts),
        "blocked_child_first_error_counts_top50": _sorted_counter(
            blocked_child_first_error_counts,
            limit=50,
        ),
        "blocked_child_first_error_ranked_top50": _ranked_counter(
            blocked_child_first_error_counts,
            limit=50,
        ),
        "ignored_legacy_child_reason_counts": _sorted_counter(ignored_legacy_child_reason_counts),
        "ignored_legacy_child_first_error_counts_top50": _sorted_counter(
            ignored_legacy_child_first_error_counts,
            limit=50,
        ),
        "backfill_scope_plan": _build_backfill_scope_plan(
            blocked_by_stage_or_path=blocked_by_stage_or_path,
            would_stamp_by_stage_or_path=would_stamp_by_stage_or_path,
            stamped_by_stage_or_path=stamped_by_stage_or_path,
            write_failed_by_stage_or_path=write_failed_by_stage_or_path,
        ),
        "blocked_parent_examples": blocked_examples,
        "ignored_legacy_parent_examples": ignored_legacy_examples,
        "write_failed_parent_examples": write_failed_examples,
    }


def _compact_child_blocker(child_report: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": child_report.get("run_name"),
        "verification": child_report.get("verification"),
    }
    reason = child_report.get("reason")
    if reason is not None:
        row["reason"] = reason
    errors = child_report.get("errors")
    if isinstance(errors, list) and errors:
        row["first_error"] = str(errors[0])
        row["error_count"] = len(errors)
    warnings = child_report.get("warnings")
    if isinstance(warnings, list) and warnings:
        row["warning_count"] = len(warnings)
    return row


def _blocked_parent_rows(report: Mapping[str, Any], *, max_children: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamp_utc = report.get("timestamp_utc")
    apply = report.get("apply")
    stores = report.get("stores")
    if not isinstance(stores, list):
        return rows

    for store in stores:
        if not isinstance(store, Mapping):
            continue
        zarr_path = store.get("zarr_path")
        parents = store.get("parents")
        if not isinstance(parents, list):
            continue
        for parent in parents:
            if not isinstance(parent, Mapping) or parent.get("status") != "blocked":
                continue
            children = parent.get("children")
            blocked_children: list[Mapping[str, Any]] = []
            if isinstance(children, list):
                blocked_children = [
                    child
                    for child in children
                    if isinstance(child, Mapping)
                    and _is_blocking_child_report(child)
                ]
            reason_counts: Counter[str] = Counter(_blocked_child_reason(child) for child in blocked_children)
            first_error_counts: Counter[str] = Counter(
                first_error for child in blocked_children if (first_error := _first_error(child))
            )
            rows.append(
                {
                    "schema_id": BLOCKED_PARENT_ROW_SCHEMA_ID,
                    "timestamp_utc": timestamp_utc,
                    "apply": apply,
                    "zarr_path": zarr_path,
                    "parent_path": parent.get("parent_path"),
                    "status": parent.get("status"),
                    "stage": parent.get("stage"),
                    "stage_spec_available": parent.get("stage_spec_available"),
                    "child_count": parent.get("child_count"),
                    "verified_child_count": parent.get("verified_child_count"),
                    "unverified_child_count": parent.get("unverified_child_count"),
                    "ignored_legacy_child_count": parent.get("ignored_legacy_child_count"),
                    "latest": parent.get("latest"),
                    "latest_complete": parent.get("latest_complete"),
                    "blocked_child_reason_counts": _sorted_counter(reason_counts),
                    "blocked_child_first_error_counts_top10": _sorted_counter(
                        first_error_counts,
                        limit=10,
                    ),
                    "blocked_child_examples": [
                        _compact_child_blocker(child)
                        for child in blocked_children[:max_children]
                    ],
                }
            )
    return rows


def _write_failed_parent_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamp_utc = report.get("timestamp_utc")
    apply = report.get("apply")
    stores = report.get("stores")
    if not isinstance(stores, list):
        return rows

    for store in stores:
        if not isinstance(store, Mapping):
            continue
        zarr_path = store.get("zarr_path")
        parents = store.get("parents")
        if not isinstance(parents, list):
            continue
        for parent in parents:
            if not isinstance(parent, Mapping) or parent.get("status") != "write_failed":
                continue
            rows.append(
                {
                    "schema_id": WRITE_FAILED_PARENT_ROW_SCHEMA_ID,
                    "timestamp_utc": timestamp_utc,
                    "apply": apply,
                    "zarr_path": zarr_path,
                    "parent_path": parent.get("parent_path"),
                    "status": parent.get("status"),
                    "stage": parent.get("stage"),
                    "stage_spec_available": parent.get("stage_spec_available"),
                    "child_count": parent.get("child_count"),
                    "marked_child_count": parent.get("marked_child_count"),
                    "completion_epoch_before": parent.get("completion_epoch_before"),
                    "completion_epoch_after": parent.get("completion_epoch_after"),
                    "latest": parent.get("latest"),
                    "latest_complete": parent.get("latest_complete"),
                    "write_error_phase": parent.get("write_error_phase"),
                    "write_error_run_name": parent.get("write_error_run_name"),
                    "write_error_type": parent.get("write_error_type"),
                    "write_error": parent.get("write_error"),
                }
            )
    return rows


def _write_report_jsonl_rows(rows: Sequence[Mapping[str, Any]], output_jsonl: str | Path | None) -> None:
    if not output_jsonl:
        return
    path = Path(output_jsonl).expanduser()
    write_jsonl_atomic(path, rows)


_write_jsonl_rows = _write_report_jsonl_rows


def backfill_completion_epoch(
    zarr_paths: Sequence[str | Path],
    *,
    apply: bool = False,
    stages: Sequence[str] | None = None,
    parent_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    timestamp_utc = _utc_now()
    selected_stages = _normalize_filter_values(stages)
    selected_parent_paths = _normalize_filter_values(parent_paths)
    stores: list[dict[str, Any]] = []
    for raw_path in zarr_paths:
        zarr_path = Path(raw_path).expanduser().resolve()
        if not zarr_path.exists():
            stores.append({"zarr_path": str(zarr_path), "status": "missing"})
            continue
        mode = "a" if apply else "r"
        try:
            root = zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)
        except Exception as exc:
            stores.append(
                {
                    "zarr_path": str(zarr_path),
                    "status": "open_failed",
                    "mode": mode,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            continue
        parents = _iter_run_parents(root)
        parent_reports = [
            _summarize_parent(
                parent_path,
                parent,
                apply=apply,
                timestamp_utc=timestamp_utc,
                selected_stages=selected_stages,
                selected_parent_paths=selected_parent_paths,
            )
            for parent_path, parent in parents
        ]
        stores.append(
            {
                "zarr_path": str(zarr_path),
                "status": "ok",
                "parent_count": len(parent_reports),
                "blocked_parent_count": sum(1 for item in parent_reports if item.get("status") == "blocked"),
                "filtered_parent_count": sum(1 for item in parent_reports if item.get("status") == "filtered"),
                "stamped_parent_count": sum(1 for item in parent_reports if item.get("status") == "stamped"),
                "would_stamp_parent_count": sum(1 for item in parent_reports if item.get("status") == "would_stamp"),
                "write_failed_parent_count": sum(
                    1 for item in parent_reports if item.get("status") == "write_failed"
                ),
                "would_mark_child_count": sum(
                    int(item.get("would_mark_child_count") or 0) for item in parent_reports
                ),
                "marked_child_count": sum(int(item.get("marked_child_count") or 0) for item in parent_reports),
                "ignored_legacy_child_count": sum(
                    int(item.get("ignored_legacy_child_count") or 0) for item in parent_reports
                ),
                "ignored_legacy_parent_count": sum(
                    1 for item in parent_reports if int(item.get("ignored_legacy_child_count") or 0) > 0
                ),
                "parents": parent_reports,
            }
        )

    summary = _build_summary(stores)
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "timestamp_utc": timestamp_utc,
        "apply": bool(apply),
        "filters": {
            "stages": sorted(selected_stages),
            "parent_paths": sorted(selected_parent_paths),
        },
        "store_count": len(stores),
        "ok_store_count": sum(1 for item in stores if item.get("status") == "ok"),
        "non_ok_store_count": sum(1 for item in stores if item.get("status") != "ok"),
        "blocked_parent_count": sum(int(item.get("blocked_parent_count") or 0) for item in stores),
        "filtered_parent_count": sum(int(item.get("filtered_parent_count") or 0) for item in stores),
        "stamped_parent_count": sum(int(item.get("stamped_parent_count") or 0) for item in stores),
        "would_stamp_parent_count": sum(int(item.get("would_stamp_parent_count") or 0) for item in stores),
        "write_failed_parent_count": sum(int(item.get("write_failed_parent_count") or 0) for item in stores),
        "would_mark_child_count": sum(int(item.get("would_mark_child_count") or 0) for item in stores),
        "marked_child_count": sum(int(item.get("marked_child_count") or 0) for item in stores),
        "ignored_legacy_child_count": sum(
            int(item.get("ignored_legacy_child_count") or 0) for item in stores
        ),
        "ignored_legacy_parent_count": sum(
            int(item.get("ignored_legacy_parent_count") or 0) for item in stores
        ),
        "summary": summary,
        "stores": stores,
    }


def _discover_zarrs(recordings_root: Path) -> list[Path]:
    return sorted(path for path in recordings_root.rglob("*.zarr") if path.is_dir())


def _write_report(
    report: Mapping[str, Any],
    output_json: str | Path | None,
    *,
    emit_stdout: bool = True,
) -> None:
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if output_json:
        Path(output_json).expanduser().write_text(text + "\n", encoding="utf-8")
    if emit_stdout:
        print(text)


def _summary_payload(report: Mapping[str, Any]) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema_id": f"{REPORT_SCHEMA_ID}.summary",
        "timestamp_utc": report["timestamp_utc"],
        "apply": report["apply"],
        "filters": report["filters"],
        "store_count": report["store_count"],
        "ok_store_count": report["ok_store_count"],
        "non_ok_store_count": report["non_ok_store_count"],
        "blocked_parent_count": report["blocked_parent_count"],
        "filtered_parent_count": report["filtered_parent_count"],
        "would_stamp_parent_count": report["would_stamp_parent_count"],
        "stamped_parent_count": report["stamped_parent_count"],
        "write_failed_parent_count": int(report.get("write_failed_parent_count") or 0),
        "would_mark_child_count": int(report.get("would_mark_child_count") or 0),
        "marked_child_count": report["marked_child_count"],
        "ignored_legacy_child_count": int(report.get("ignored_legacy_child_count") or 0),
        "ignored_legacy_parent_count": int(report.get("ignored_legacy_parent_count") or 0),
        **dict(report["summary"]),
    }
    if "apply_aborted" in report:
        payload["apply_aborted"] = report["apply_aborted"]
    if "apply_abort_reason" in report:
        payload["apply_abort_reason"] = report["apply_abort_reason"]
    if "apply_abort_message" in report:
        payload["apply_abort_message"] = report["apply_abort_message"]
    if "expected_counts" in report:
        payload["expected_counts"] = report["expected_counts"]
    if "preflight_counts" in report:
        payload["preflight_counts"] = report["preflight_counts"]
    if "expectation_errors" in report:
        payload["expectation_errors"] = report["expectation_errors"]
    if "post_apply_expected_counts" in report:
        payload["post_apply_expected_counts"] = report["post_apply_expected_counts"]
    if "post_apply_expectation_errors" in report:
        payload["post_apply_expectation_errors"] = report["post_apply_expectation_errors"]
    if "apply_failed" in report:
        payload["apply_failed"] = report["apply_failed"]
    if "apply_failure_reason" in report:
        payload["apply_failure_reason"] = report["apply_failure_reason"]
    if "apply_failure_message" in report:
        payload["apply_failure_message"] = report["apply_failure_message"]
    return payload


def _apply_filter_error(
    *,
    apply: bool,
    stages: Sequence[str],
    parent_paths: Sequence[str],
    allow_broad_apply: bool,
) -> str | None:
    if not apply or allow_broad_apply:
        return None
    if _normalize_filter_values(stages) or _normalize_filter_values(parent_paths):
        return None
    return (
        "--apply requires at least one --stage or --parent-path filter. "
        "Pass --allow-broad-apply to intentionally apply across every discovered runs-parent."
    )


def _post_apply_expectation_error(*, apply: bool, post_apply_expected_counts: Mapping[str, int]) -> str | None:
    if apply or not post_apply_expected_counts:
        return None
    flags = ", ".join(f"--expect-applied-{key.replace('_', '-')}" for key in sorted(post_apply_expected_counts))
    return f"{flags} can only be used with --apply."


def _expected_counts_from_namespace(args: argparse.Namespace) -> dict[str, int]:
    candidates = {
        "store_count": args.expect_store_count,
        "non_ok_store_count": args.expect_non_ok_store_count,
        "blocked_parent_count": args.expect_blocked_parent_count,
        "filtered_parent_count": args.expect_filtered_parent_count,
        "would_stamp_parent_count": args.expect_would_stamp_parent_count,
        "write_failed_parent_count": args.expect_write_failed_parent_count,
        "would_mark_child_count": args.expect_would_mark_child_count,
        "ignored_legacy_child_count": args.expect_ignored_legacy_child_count,
    }
    return {key: int(value) for key, value in candidates.items() if value is not None}


def _post_apply_expected_counts_from_namespace(args: argparse.Namespace) -> dict[str, int]:
    candidates = {
        "stamped_parent_count": args.expect_applied_stamped_parent_count,
        "marked_child_count": args.expect_applied_marked_child_count,
    }
    return {key: int(value) for key, value in candidates.items() if value is not None}


def _observed_count_payload(report: Mapping[str, Any]) -> dict[str, int]:
    keys = (
        "store_count",
        "non_ok_store_count",
        "blocked_parent_count",
        "filtered_parent_count",
        "would_stamp_parent_count",
        "stamped_parent_count",
        "write_failed_parent_count",
        "would_mark_child_count",
        "marked_child_count",
        "ignored_legacy_child_count",
        "ignored_legacy_parent_count",
    )
    return {key: int(report.get(key) or 0) for key in keys}


def _expected_count_errors(report: Mapping[str, Any], expected_counts: Mapping[str, int]) -> list[str]:
    errors: list[str] = []
    for key, expected in expected_counts.items():
        actual = int(report.get(key) or 0)
        if actual != int(expected):
            errors.append(f"{key}: expected {int(expected)}, observed {actual}")
    return errors


def _abort_report(
    report: Mapping[str, Any],
    *,
    reason: str,
    message: str,
    expected_counts: Mapping[str, int] | None = None,
    expectation_errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    out = dict(report)
    out["apply_aborted"] = True
    out["apply_abort_reason"] = reason
    out["apply_abort_message"] = message
    if expected_counts:
        out["expected_counts"] = dict(expected_counts)
    if expectation_errors:
        out["expectation_errors"] = list(expectation_errors)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify legacy Zarr run groups, mark verified legacy children complete, "
            "and stamp strict parent completion epochs."
        )
    )
    parser.add_argument("zarr_paths", nargs="*", help="Palette zarr stores to inspect/backfill.")
    parser.add_argument(
        "--recordings-root",
        type=Path,
        help="Optional root to scan recursively for *.zarr stores.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Only verify/apply parents whose resolved StageSpec stage matches this name. "
            "May be passed multiple times. Combined with --parent-path as an AND filter."
        ),
    )
    parser.add_argument(
        "--parent-path",
        action="append",
        default=[],
        help=(
            "Only verify/apply this exact runs-parent path, such as crop_runs. "
            "May be passed multiple times. Combined with --stage as an AND filter."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Write completion attrs and parent epochs.")
    parser.add_argument(
        "--allow-broad-apply",
        action="store_true",
        help="Permit --apply without --stage/--parent-path filters.",
    )
    parser.add_argument(
        "--allow-blocked-apply",
        action="store_true",
        help=(
            "Permit --apply when the selected scope still has blocked parents. "
            "Without this, apply runs a dry-run preflight and aborts before writes if any "
            "selected parent is blocked."
        ),
    )
    parser.add_argument(
        "--expect-store-count",
        type=int,
        help="Abort if the dry-run/preflight store_count differs from this value.",
    )
    parser.add_argument(
        "--expect-non-ok-store-count",
        type=int,
        help="Abort if the dry-run/preflight non_ok_store_count differs from this value.",
    )
    parser.add_argument(
        "--expect-blocked-parent-count",
        type=int,
        help="Abort if the dry-run/preflight blocked_parent_count differs from this value.",
    )
    parser.add_argument(
        "--expect-filtered-parent-count",
        type=int,
        help="Abort if the dry-run/preflight filtered_parent_count differs from this value.",
    )
    parser.add_argument(
        "--expect-would-stamp-parent-count",
        type=int,
        help="Abort if the dry-run/preflight would_stamp_parent_count differs from this value.",
    )
    parser.add_argument(
        "--expect-write-failed-parent-count",
        type=int,
        help="Abort if the dry-run/preflight write_failed_parent_count differs from this value.",
    )
    parser.add_argument(
        "--expect-would-mark-child-count",
        type=int,
        help="Abort if the dry-run/preflight would_mark_child_count differs from this value.",
    )
    parser.add_argument(
        "--expect-ignored-legacy-child-count",
        type=int,
        help=(
            "Abort if the dry-run/preflight ignored_legacy_child_count differs from this value. "
            "Ignored legacy children are invalid non-latest runs left unmarked before stamping "
            "the parent strict."
        ),
    )
    parser.add_argument(
        "--expect-applied-stamped-parent-count",
        type=int,
        help="After --apply, fail if stamped_parent_count differs from this value.",
    )
    parser.add_argument(
        "--expect-applied-marked-child-count",
        type=int,
        help="After --apply, fail if marked_child_count differs from this value.",
    )
    parser.add_argument("--output-json", help="Optional path to write the JSON report.")
    parser.add_argument(
        "--blocked-jsonl",
        help="Optional compact JSONL sidecar with one row per blocked parent for triage.",
    )
    parser.add_argument(
        "--write-failed-jsonl",
        help="Optional compact JSONL sidecar with one row per parent that failed during attr writes.",
    )
    parser.add_argument("--no-stdout", action="store_true", help="Write --output-json without printing full JSON.")
    parser.add_argument("--summary-only", action="store_true", help="Emit only aggregate summary counts.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    expected_counts = _expected_counts_from_namespace(args)
    post_apply_expected_counts = _post_apply_expected_counts_from_namespace(args)

    filter_error = _apply_filter_error(
        apply=bool(args.apply),
        stages=args.stage,
        parent_paths=args.parent_path,
        allow_broad_apply=bool(args.allow_broad_apply),
    )
    if filter_error is not None:
        raise SystemExit(filter_error)

    post_apply_usage_error = _post_apply_expectation_error(
        apply=bool(args.apply),
        post_apply_expected_counts=post_apply_expected_counts,
    )
    if post_apply_usage_error is not None:
        raise SystemExit(post_apply_usage_error)

    zarr_paths = [Path(value) for value in args.zarr_paths]
    if args.recordings_root is not None:
        zarr_paths.extend(_discover_zarrs(args.recordings_root.expanduser()))
    if not zarr_paths:
        raise SystemExit("Provide at least one zarr path or --recordings-root.")

    preflight_needed = bool(args.apply)
    preflight_counts: dict[str, int] | None = None
    if preflight_needed:
        preflight = backfill_completion_epoch(
            zarr_paths,
            apply=False,
            stages=args.stage,
            parent_paths=args.parent_path,
        )
        preflight_counts = _observed_count_payload(preflight)
        preflight["preflight_counts"] = preflight_counts
        expectation_errors = _expected_count_errors(preflight, expected_counts)
        if expectation_errors:
            preflight = _abort_report(
                preflight,
                reason="expectation_mismatch",
                message="Refusing --apply because preflight counts differ from expected values.",
                expected_counts=expected_counts,
                expectation_errors=expectation_errors,
            )
            payload = _summary_payload(preflight) if bool(args.summary_only) else preflight
            _write_report_jsonl_rows(_blocked_parent_rows(preflight), args.blocked_jsonl)
            _write_report_jsonl_rows(_write_failed_parent_rows(preflight), args.write_failed_jsonl)
            _write_report(payload, args.output_json, emit_stdout=not bool(args.no_stdout))
            raise SystemExit(str(preflight["apply_abort_message"]))

        if int(preflight["non_ok_store_count"]) > 0:
            preflight = _abort_report(
                preflight,
                reason="non_ok_stores_present",
                message=(
                    "Refusing --apply because one or more target zarr stores are missing "
                    "or failed to open during preflight."
                ),
                expected_counts=expected_counts,
            )
            payload = _summary_payload(preflight) if bool(args.summary_only) else preflight
            _write_report_jsonl_rows(_blocked_parent_rows(preflight), args.blocked_jsonl)
            _write_report_jsonl_rows(_write_failed_parent_rows(preflight), args.write_failed_jsonl)
            _write_report(payload, args.output_json, emit_stdout=not bool(args.no_stdout))
            raise SystemExit(str(preflight["apply_abort_message"]))

        if int(preflight["blocked_parent_count"]) > 0 and not bool(args.allow_blocked_apply):
            preflight = _abort_report(
                preflight,
                reason="blocked_parents_present",
                message=(
                    "Refusing --apply because selected scope contains blocked parents. "
                    "Inspect --blocked-jsonl output, narrow the filters, or pass "
                    "--allow-blocked-apply to stamp only unblocked parents."
                ),
                expected_counts=expected_counts,
            )
            payload: Mapping[str, Any] = _summary_payload(preflight) if bool(args.summary_only) else preflight
            _write_report_jsonl_rows(_blocked_parent_rows(preflight), args.blocked_jsonl)
            _write_report_jsonl_rows(_write_failed_parent_rows(preflight), args.write_failed_jsonl)
            _write_report(payload, args.output_json, emit_stdout=not bool(args.no_stdout))
            raise SystemExit(str(preflight["apply_abort_message"]))

    report = backfill_completion_epoch(
        zarr_paths,
        apply=bool(args.apply),
        stages=args.stage,
        parent_paths=args.parent_path,
    )
    if bool(args.apply) and preflight_counts is not None:
        report["preflight_counts"] = preflight_counts
    if bool(args.apply) and expected_counts:
        report["expected_counts"] = dict(expected_counts)
    if bool(args.apply) and post_apply_expected_counts:
        report["post_apply_expected_counts"] = dict(post_apply_expected_counts)
    expectation_errors = [] if bool(args.apply) else _expected_count_errors(report, expected_counts)
    if expectation_errors:
        report = _abort_report(
            report,
            reason="expectation_mismatch",
            message="Counts differ from expected values.",
            expected_counts=expected_counts,
            expectation_errors=expectation_errors,
        )
        payload = _summary_payload(report) if bool(args.summary_only) else report
        _write_report_jsonl_rows(_blocked_parent_rows(report), args.blocked_jsonl)
        _write_report_jsonl_rows(_write_failed_parent_rows(report), args.write_failed_jsonl)
        _write_report(payload, args.output_json, emit_stdout=not bool(args.no_stdout))
        raise SystemExit(str(report["apply_abort_message"]))

    if bool(args.apply) and int(report.get("write_failed_parent_count") or 0) > 0:
        report["apply_failed"] = True
        report["apply_failure_reason"] = "write_failed_parents_present"
        report["apply_failure_message"] = (
            "One or more parent groups failed during attr writes. The report is "
            "conservative: affected parents are not reported as stamped, and rerun "
            "is expected to be idempotent after fixing the write failure."
        )
        payload = _summary_payload(report) if bool(args.summary_only) else report
        _write_report_jsonl_rows(_blocked_parent_rows(report), args.blocked_jsonl)
        _write_report_jsonl_rows(_write_failed_parent_rows(report), args.write_failed_jsonl)
        _write_report(payload, args.output_json, emit_stdout=not bool(args.no_stdout))
        raise SystemExit(str(report["apply_failure_message"]))

    post_apply_expectation_errors = (
        _expected_count_errors(report, post_apply_expected_counts)
        if bool(args.apply)
        else []
    )
    if post_apply_expectation_errors:
        report["apply_failed"] = True
        report["apply_failure_reason"] = "post_apply_expectation_mismatch"
        report["apply_failure_message"] = (
            "Post-apply counts differed from expected values. Inspect the JSON "
            "report before rerunning; successful parent stamps are idempotent."
        )
        report["post_apply_expectation_errors"] = post_apply_expectation_errors
        payload = _summary_payload(report) if bool(args.summary_only) else report
        _write_report_jsonl_rows(_blocked_parent_rows(report), args.blocked_jsonl)
        _write_report_jsonl_rows(_write_failed_parent_rows(report), args.write_failed_jsonl)
        _write_report(payload, args.output_json, emit_stdout=not bool(args.no_stdout))
        raise SystemExit(str(report["apply_failure_message"]))

    payload: Mapping[str, Any] = _summary_payload(report) if bool(args.summary_only) else report
    _write_report_jsonl_rows(_blocked_parent_rows(report), args.blocked_jsonl)
    _write_report_jsonl_rows(_write_failed_parent_rows(report), args.write_failed_jsonl)
    _write_report(payload, args.output_json, emit_stdout=not bool(args.no_stdout))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
