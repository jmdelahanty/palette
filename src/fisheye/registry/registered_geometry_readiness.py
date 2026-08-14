"""Project immutable registered-geometry artifacts into registry stage rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class RegisteredGeometryStageProjection:
    step_name: str
    status: str
    run_name: str | None
    method: str | None
    review_status: Mapping[str, object] | None
    details: Mapping[str, object]
    run_group: object | None


def _decode(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _complete_groups(
    parent: object | None,
    *,
    predicate: Callable[[object], bool] | None = None,
) -> list[tuple[str, object]]:
    """Return complete modern runs without following mutable selectors."""

    if parent is None:
        return []
    try:
        names = list(parent.group_keys())  # type: ignore[attr-defined]
    except Exception:
        try:
            names = list(parent.keys())  # type: ignore[attr-defined]
        except Exception:
            return []
    groups: list[tuple[str, object]] = []
    for name in sorted(str(value) for value in names):
        try:
            group = parent[name]  # type: ignore[index]
            attrs = group.attrs  # type: ignore[attr-defined]
        except Exception:
            continue
        completion = _decode(attrs.get("palette_run_completion_status"))
        ordinary_status = _decode(attrs.get("status"))
        if completion != "complete" and ordinary_status not in {
            "complete",
            "completed",
        }:
            continue
        if predicate is not None and not predicate(group):
            continue
        groups.append((name, group))
    return groups


def _nested_group(root: object, path: str) -> object | None:
    group = root
    for part in str(path).strip("/").split("/"):
        if not part:
            continue
        try:
            group = group[part]  # type: ignore[index]
        except Exception:
            return None
    return group


def _status_from_presence(
    *,
    present: bool,
    prerequisite_statuses: Sequence[str],
) -> tuple[str, str]:
    if present:
        return "ok", "present"
    if not prerequisite_statuses or any(
        status == "ok" for status in prerequisite_statuses
    ):
        return "missing", "run_missing"
    if any(status == "error" for status in prerequisite_statuses):
        return "error", "upstream_error"
    if all(status == "na" for status in prerequisite_statuses):
        return "na", "upstream_na"
    return "absent", "upstream_missing"


def _single_run(groups: Sequence[tuple[str, object]]) -> str | None:
    return groups[0][0] if len(groups) == 1 else None


def _details(
    groups: Sequence[tuple[str, object]],
    *,
    common: Mapping[str, object],
    reason: str,
    upstream: Mapping[str, str],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        **common,
        "reason": reason,
        "complete_run_count": len(groups),
        "complete_run_ids": [name for name, _group in groups],
        "selection_mode": "immutable_explicit_runs_no_mutable_pointer",
        "legacy_dish_mask_inferred": False,
        "upstream": dict(upstream),
        **dict(extra or {}),
    }


def project_registered_geometry_stages(
    *,
    root: object,
    analysis_group: object | None,
    common_details: Mapping[str, object],
    raw_status: str,
    calibration_status: str,
    detect_status: str,
    detect_quality_status: str,
) -> tuple[RegisteredGeometryStageProjection, ...]:
    """Build six modern geometry-stage projections from exact immutable evidence."""

    candidate_parent = (
        analysis_group.get("arena_geometry_runs")  # type: ignore[attr-defined]
        if analysis_group is not None
        else None
    )
    imported = _complete_groups(
        candidate_parent,
        predicate=lambda group: _decode(  # type: ignore[attr-defined]
            group.attrs.get("candidate_kind")
        )
        == "acquisition_registered_dish",
    )
    reviewed_offline = _complete_groups(
        candidate_parent,
        predicate=lambda group: _decode(  # type: ignore[attr-defined]
            group.attrs.get("candidate_kind")
        )
        == "palette_recording_image_fit",
    )
    fit_review_parent = (
        analysis_group.get("arena_geometry_fit_runs")  # type: ignore[attr-defined]
        if analysis_group is not None
        else None
    )
    fit_reviews = _complete_groups(
        fit_review_parent,
        predicate=lambda group: (  # type: ignore[attr-defined]
            _decode(group.attrs.get("schema_id"))
            == "palette.arena_geometry_fit_review_run"
            and group.attrs.get("stage_selector_eligible") is False
        ),
    )
    # Once a reviewed Palette candidate exists it supersedes its immutable
    # pre-review package for readiness.  The package remains durable evidence.
    offline = reviewed_offline or fit_reviews
    pending_fit_review_runs = (
        [
            name
            for name, group in fit_reviews
            if _decode(group.attrs.get("review_status"))  # type: ignore[attr-defined]
            == "awaiting_explicit_human_review"
        ]
        if not reviewed_offline
        else []
    )
    comparison_parent = (
        analysis_group.get("arena_geometry_comparison_runs")  # type: ignore[attr-defined]
        if analysis_group is not None
        else None
    )
    comparisons = _complete_groups(
        comparison_parent,
        predicate=lambda group: _decode(  # type: ignore[attr-defined]
            group.attrs.get("schema_id")
        )
        == "palette.arena_geometry_comparison_run",
    )
    selection_parent = (
        analysis_group.get("arena_geometry_selection")  # type: ignore[attr-defined]
        if analysis_group is not None
        else None
    )

    def modern_selection(group: object) -> bool:
        record = group.attrs.get("selection_record")  # type: ignore[attr-defined]
        if not isinstance(record, Mapping) or record.get("schema_version") != 2:
            return False
        decision = record.get("decision")
        return isinstance(decision, Mapping) and isinstance(
            decision.get("comparison_binding"), Mapping
        )

    selections = _complete_groups(
        selection_parent,
        predicate=modern_selection,
    )
    gate_parent = (
        analysis_group.get("detection_gate_runs")  # type: ignore[attr-defined]
        if analysis_group is not None
        else None
    )
    gates = _complete_groups(
        gate_parent,
        predicate=lambda group: (
            _decode(group.attrs.get("schema_id"))  # type: ignore[attr-defined]
            == "palette.registered_detection_gate_run"
            and group.attrs.get("selection_record_schema_version") == 2  # type: ignore[attr-defined]
            and bool(_decode(group.attrs.get("comparison_run")))  # type: ignore[attr-defined]
        ),
    )

    import_status, import_reason = _status_from_presence(
        present=bool(imported),
        prerequisite_statuses=(raw_status, calibration_status),
    )
    offline_status, offline_reason = _status_from_presence(
        present=bool(offline), prerequisite_statuses=(raw_status,)
    )
    comparison_status, comparison_reason = _status_from_presence(
        present=bool(comparisons),
        prerequisite_statuses=(import_status, offline_status),
    )
    if not comparisons and pending_fit_review_runs:
        comparison_reason = "offline_fit_review_required"
    review_runs = [
        name
        for name, group in comparisons
        if group.attrs.get("review_required") is True  # type: ignore[attr-defined]
    ]
    if review_runs:
        comparison_reason = "review_required"
    selection_status, selection_reason = _status_from_presence(
        present=bool(selections), prerequisite_statuses=(comparison_status,)
    )
    if not selections and (review_runs or pending_fit_review_runs):
        selection_reason = (
            "comparison_review_required"
            if review_runs
            else "offline_fit_review_required"
        )

    selection_names = {name for name, _group in selections}
    current_gates: list[tuple[str, object]] = []
    stale_gate_reasons: dict[str, list[str]] = {}
    for name, group in gates:
        attrs = group.attrs  # type: ignore[attr-defined]
        reasons: list[str] = []
        selection_run = _decode(attrs.get("selection_run"))
        source_path = _decode(attrs.get("source_detection_group_path"))
        if not selection_run or selection_run not in selection_names:
            reasons.append("selection_missing_or_incomplete")
        if not source_path or _nested_group(root, source_path) is None:
            reasons.append("source_detection_missing")
        if reasons:
            stale_gate_reasons[name] = reasons
        else:
            current_gates.append((name, group))
    gate_status, gate_reason = _status_from_presence(
        present=bool(current_gates),
        prerequisite_statuses=(selection_status, detect_status),
    )
    if gates and not current_gates:
        gate_status = "error"
        gate_reason = "complete_gate_stale_or_incompatible"

    refined_parent = root.get("refined_detect_runs")  # type: ignore[attr-defined]
    consumers = _complete_groups(
        refined_parent,
        predicate=lambda group: (
            group.attrs.get("finalized_recording_authority") is True  # type: ignore[attr-defined]
            and _decode(  # type: ignore[attr-defined]
                group.attrs.get("registered_detection_gate_requirement")
            )
            in {"if_available", "required"}
        ),
    )
    current_gate_names = {name for name, _group in current_gates}
    valid_consumers: list[tuple[str, object]] = []
    unavailable_consumers: list[str] = []
    invalid_consumers: dict[str, str] = {}
    for name, group in consumers:
        evidence = group.attrs.get("registered_detection_gate")  # type: ignore[attr-defined]
        if not isinstance(evidence, Mapping):
            invalid_consumers[name] = "missing_gate_evidence"
            continue
        requirement = _decode(evidence.get("requirement"))
        applied = (
            evidence.get("applied") is True and evidence.get("status") == "applied"
        )
        consumed_gate = _decode(evidence.get("gate_run"))
        if applied and consumed_gate in current_gate_names:
            valid_consumers.append((name, group))
        elif requirement == "if_available" and not applied:
            unavailable_consumers.append(name)
        else:
            invalid_consumers[name] = "required_or_applied_gate_not_current"
    consumption_status, consumption_reason = _status_from_presence(
        present=bool(valid_consumers),
        prerequisite_statuses=(gate_status, detect_quality_status),
    )
    if invalid_consumers:
        consumption_status = "error"
        consumption_reason = "invalid_gate_consumption"
    elif unavailable_consumers and not valid_consumers:
        consumption_status = "missing"
        consumption_reason = "if_available_gate_not_applied"

    def projection(
        step_name: str,
        status: str,
        groups: Sequence[tuple[str, object]],
        *,
        reason: str,
        upstream: Mapping[str, str],
        method: str | None = None,
        review_status: Mapping[str, object] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> RegisteredGeometryStageProjection:
        return RegisteredGeometryStageProjection(
            step_name=step_name,
            status=status,
            run_name=_single_run(groups),
            method=method,
            review_status=review_status,
            details=_details(
                groups,
                common=common_details,
                reason=reason,
                upstream=upstream,
                extra=extra,
            ),
            run_group=groups[0][1] if len(groups) == 1 else None,
        )

    return (
        projection(
            "recording_geometry_import",
            import_status,
            imported,
            reason=import_reason,
            upstream={"raw": raw_status, "calibration": calibration_status},
            extra={"candidate_kind": "acquisition_registered_dish"},
        ),
        projection(
            "arena_geometry_offline_fit",
            offline_status,
            offline,
            reason=offline_reason,
            upstream={"raw": raw_status},
            review_status=(
                {
                    "state": "evidence_complete_review_pending",
                    "runs": pending_fit_review_runs,
                }
                if pending_fit_review_runs
                else None
            ),
            extra={
                "candidate_kind": "palette_recording_image_fit",
                "fit_review_run_ids": [name for name, _group in fit_reviews],
                "reviewed_candidate_run_ids": [
                    name for name, _group in reviewed_offline
                ],
                "pending_fit_review_run_ids": pending_fit_review_runs,
            },
        ),
        projection(
            "arena_geometry_comparison",
            comparison_status,
            comparisons,
            reason=comparison_reason,
            upstream={
                "recording_geometry_import": import_status,
                "arena_geometry_offline_fit": offline_status,
            },
            review_status=(
                {"state": "review_required", "runs": review_runs}
                if review_runs
                else (
                    {
                        "state": "review_required",
                        "reason": "offline_fit_review_required",
                        "fit_review_runs": pending_fit_review_runs,
                    }
                    if pending_fit_review_runs
                    else None
                )
            ),
            extra={
                "review_required_runs": review_runs,
                "pending_fit_review_run_ids": pending_fit_review_runs,
            },
        ),
        projection(
            "arena_geometry_selection",
            selection_status,
            selections,
            reason=selection_reason,
            upstream={"arena_geometry_comparison": comparison_status},
            review_status=(
                {"state": "review_required", "reason": selection_reason}
                if review_runs or pending_fit_review_runs
                else None
            ),
        ),
        projection(
            "registered_detection_gate",
            gate_status,
            current_gates,
            reason=gate_reason,
            upstream={
                "arena_geometry_selection": selection_status,
                "detect": detect_status,
            },
            method="native_camera_circle_signed_distance_inclusive_v1",
            extra={
                "complete_gate_run_ids": [name for name, _group in gates],
                "stale_gate_reasons": stale_gate_reasons,
            },
        ),
        projection(
            "registered_detection_gate_consumption",
            consumption_status,
            valid_consumers,
            reason=consumption_reason,
            upstream={
                "registered_detection_gate": gate_status,
                "detect_quality": detect_quality_status,
            },
            method="exact_instance_key_join_v1",
            extra={
                "unavailable_consumer_runs": unavailable_consumers,
                "invalid_consumer_runs": invalid_consumers,
            },
        ),
    )


__all__ = [
    "RegisteredGeometryStageProjection",
    "project_registered_geometry_stages",
]
