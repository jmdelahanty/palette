"""Read-only registry projection for registered-dish geometry review.

The registry is the queue/index authority.  This module intentionally does not
open Zarr stores and never writes the registry; the selected canonical archive
is inspected separately by the geometry-review application.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.status_page.query import (
    open_readonly_connection,
    resolve_registry_path,
    validate_registry_path,
)

from .status_ledger import RECORDING_STEP_STATUSES


REGISTERED_GEOMETRY_STAGES = (
    "recording_geometry_import",
    "arena_geometry_offline_fit",
    "arena_geometry_comparison",
    "arena_geometry_selection",
    "registered_detection_gate",
    "registered_detection_gate_consumption",
)
ACTIONABLE_REVIEW_STATES = frozenset(
    {"evidence_complete_review_pending", "review_required"}
)


class GeometryReviewRegistryError(ValueError):
    """The registry cannot be interpreted under the geometry-review contract."""


@dataclass(frozen=True)
class GeometryStageState:
    step_name: str
    status: str
    run_name: str | None
    review_status: Mapping[str, Any] | None
    details: Mapping[str, Any] | None
    source: str | None
    updated_utc: str | None

    @property
    def review_state(self) -> str | None:
        if not isinstance(self.review_status, Mapping):
            return None
        return _optional_text(self.review_status.get("state"))


@dataclass(frozen=True)
class GeometryReviewQueueItem:
    dataset_id: str
    recording_id: str
    zarr_path: Path
    camera_serial: str | None
    arena_id: str | None
    geometry_state: str
    actionable: bool
    stages: tuple[GeometryStageState, ...]

    def stage(self, step_name: str) -> GeometryStageState | None:
        return next(
            (stage for stage in self.stages if stage.step_name == step_name), None
        )


@dataclass(frozen=True)
class GeometryReviewTransition:
    event_key: str
    dataset_id: str
    recording_id: str
    zarr_path: Path
    stage: str
    semantic_state: str
    run_id: str
    digest: str


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _json_object(value: object, *, label: str) -> Mapping[str, Any] | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GeometryReviewRegistryError(f"{label} is not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise GeometryReviewRegistryError(f"{label} must contain one JSON object.")
    return parsed


def _walk_values(value: object) -> Iterable[tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_values(child)


def _named_text(payloads: Sequence[Mapping[str, Any] | None], names: set[str]) -> str | None:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key, value in _walk_values(payload):
            if key in names:
                text = _optional_text(value)
                if text is not None:
                    return text
    return None


def _run_ids(stage: GeometryStageState) -> tuple[str, ...]:
    found: list[str] = []
    payloads = (stage.review_status, stage.details)
    run_fields = {
        "run",
        "runs",
        "run_id",
        "run_ids",
        "fit_review_run",
        "fit_review_runs",
        "fit_review_run_ids",
        "pending_fit_review_run_ids",
        "review_required_runs",
        "comparison_run",
        "selection_run",
        "gate_run",
    }
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key, value in _walk_values(payload):
            if key not in run_fields:
                continue
            values = value if isinstance(value, (list, tuple)) else (value,)
            for item in values:
                text = _optional_text(item)
                if text and text not in found:
                    found.append(text)
    if stage.run_name and stage.run_name not in found:
        found.append(stage.run_name)
    return tuple(found) or ("unbound",)


def _digest(stage: GeometryStageState) -> str:
    digest = _named_text(
        (stage.review_status, stage.details),
        {
            "digest",
            "sha256",
            "review_record_sha256",
            "fit_review_record_sha256",
            "comparison_record_sha256",
            "selection_record_sha256",
            "gate_record_sha256",
        },
    )
    if digest:
        return digest
    state_payload = {
        "status": stage.status,
        "run_name": stage.run_name,
        "review_status": stage.review_status,
        "details": stage.details,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            state_payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()
    return f"registry-state-sha256:{fingerprint}"


def _semantic_state(stage: GeometryStageState) -> str:
    review_state = stage.review_state
    if review_state:
        semantic = _named_text(
            (stage.review_status, stage.details),
            {"evidence_outcome", "semantic_compatibility"},
        )
        if semantic:
            return semantic
        review_reason = (
            _optional_text(stage.review_status.get("reason"))
            if isinstance(stage.review_status, Mapping)
            else None
        )
        return review_reason or review_state
    if stage.status == "error":
        return _named_text(
            (stage.details,), {"evidence_outcome", "semantic_compatibility", "reason"}
        ) or "error"
    return stage.status


def _is_actionable(stage: GeometryStageState) -> bool:
    return stage.status == "error" or stage.review_state in ACTIONABLE_REVIEW_STATES


def _geometry_state(stages: Sequence[GeometryStageState]) -> str:
    actionable = [stage for stage in stages if _is_actionable(stage)]
    for stage in actionable:
        semantic = _semantic_state(stage).lower()
        if "coordinate" in semantic or "extent" in semantic:
            return "coordinate_or_extent_error"
        if "semantic" in semantic or "different_feature" in semantic:
            return "semantic_incompatibility"
        if "offline_fit_failed" in semantic:
            return "fit_failure"
    for stage in actionable:
        if stage.status == "error" and stage.step_name == "arena_geometry_offline_fit":
            return "fit_failure"
    if any(stage.status == "error" for stage in actionable):
        return "geometry_stage_error"
    if any(
        stage.step_name == "arena_geometry_offline_fit"
        and stage.review_state == "evidence_complete_review_pending"
        for stage in actionable
    ):
        return "fit_evidence_awaiting_review"
    if any(
        stage.step_name in {"arena_geometry_comparison", "arena_geometry_selection"}
        and stage.review_state == "review_required"
        for stage in actionable
    ):
        return "comparison_review_required"
    by_name = {stage.step_name: stage for stage in stages}
    if by_name.get("registered_detection_gate_consumption") and (
        by_name["registered_detection_gate_consumption"].status == "ok"
    ):
        return "gate_and_refinement_consumed"
    if by_name.get("registered_detection_gate") and (
        by_name["registered_detection_gate"].status == "ok"
    ):
        return "detection_gate_complete"
    if by_name.get("arena_geometry_selection") and (
        by_name["arena_geometry_selection"].status == "ok"
    ):
        return "geometry_selected"
    return "geometry_work_in_progress"


def load_geometry_review_queue(
    registry_path: str | Path | None,
    *,
    include_inactive: bool = False,
) -> list[GeometryReviewQueueItem]:
    """Build a geometry queue from valid ledger status plus review JSON only."""

    resolved = resolve_registry_path(
        Path(registry_path) if registry_path is not None else None,
        cwd=Path.cwd(),
    )
    validate_registry_path(resolved)
    placeholders = ",".join("?" for _ in REGISTERED_GEOMETRY_STAGES)
    sql = f"""
        SELECT
            d.dataset_id,
            COALESCE(NULLIF(trim(rss.recording_id), ''), d.recording_id, d.dataset_id)
                AS recording_id,
            d.zarr_path,
            r.camera_id,
            r.arena_id,
            rss.step_name,
            rss.status,
            rss.run_name,
            rss.review_status_json,
            rss.details_json,
            rss.source,
            rss.updated_utc
        FROM datasets AS d
        JOIN recording_step_status AS rss ON rss.dataset_id = d.dataset_id
        LEFT JOIN recordings AS r
          ON r.recording_id = COALESCE(NULLIF(trim(rss.recording_id), ''), d.recording_id)
        WHERE rss.step_name IN ({placeholders})
          AND (d.status IS NULL OR d.status != 'missing')
          AND (d.zarr_use = 'analysis' OR d.zarr_path LIKE '%_analysis.zarr')
        ORDER BY recording_id, d.dataset_id, rss.step_name
    """
    with open_readonly_connection(resolved) as conn:
        rows = conn.execute(sql, REGISTERED_GEOMETRY_STAGES).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_id = str(row["dataset_id"])
        status = str(row["status"] or "").strip()
        if status not in RECORDING_STEP_STATUSES:
            raise GeometryReviewRegistryError(
                f"recording_step_status has invalid status {status!r} for "
                f"dataset={dataset_id!r}, step={row['step_name']!r}; human review "
                "belongs in review_status_json."
            )
        review_status = _json_object(
            row["review_status_json"],
            label=f"{dataset_id}.{row['step_name']}.review_status_json",
        )
        details = _json_object(
            row["details_json"],
            label=f"{dataset_id}.{row['step_name']}.details_json",
        )
        item = grouped.setdefault(
            dataset_id,
            {
                "recording_id": str(row["recording_id"] or dataset_id),
                "zarr_path": Path(str(row["zarr_path"])).expanduser(),
                "camera_serial": _optional_text(row["camera_id"]),
                "arena_id": _optional_text(row["arena_id"]),
                "stages": [],
            },
        )
        if item["camera_serial"] is None:
            item["camera_serial"] = _named_text(
                (review_status, details), {"camera_serial", "camera_id"}
            )
        if item["arena_id"] is None:
            item["arena_id"] = _named_text(
                (review_status, details), {"arena_id", "arena_identity"}
            )
        item["stages"].append(
            GeometryStageState(
                step_name=str(row["step_name"]),
                status=status,
                run_name=_optional_text(row["run_name"]),
                review_status=review_status,
                details=details,
                source=_optional_text(row["source"]),
                updated_utc=_optional_text(row["updated_utc"]),
            )
        )

    queue: list[GeometryReviewQueueItem] = []
    order = {name: index for index, name in enumerate(REGISTERED_GEOMETRY_STAGES)}
    for dataset_id, raw in grouped.items():
        stages = tuple(sorted(raw["stages"], key=lambda stage: order[stage.step_name]))
        actionable = any(_is_actionable(stage) for stage in stages)
        if not include_inactive and not actionable:
            continue
        queue.append(
            GeometryReviewQueueItem(
                dataset_id=dataset_id,
                recording_id=raw["recording_id"],
                zarr_path=raw["zarr_path"],
                camera_serial=raw["camera_serial"],
                arena_id=raw["arena_id"],
                geometry_state=_geometry_state(stages),
                actionable=actionable,
                stages=stages,
            )
        )
    return sorted(queue, key=lambda item: (not item.actionable, item.recording_id, item.dataset_id))


def actionable_geometry_transitions(
    queue: Sequence[GeometryReviewQueueItem],
) -> list[GeometryReviewTransition]:
    """Return stable, exact transition identities for notification deduplication."""

    transitions: list[GeometryReviewTransition] = []
    for item in queue:
        for stage in item.stages:
            if not _is_actionable(stage):
                continue
            semantic_state = _semantic_state(stage)
            digest = _digest(stage)
            for run_id in _run_ids(stage):
                identity = {
                    "dataset_id": item.dataset_id,
                    "run_id": run_id,
                    "digest": digest,
                    "stage": stage.step_name,
                    "semantic_state": semantic_state,
                }
                event_key = hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()
                transitions.append(
                    GeometryReviewTransition(
                        event_key=event_key,
                        dataset_id=item.dataset_id,
                        recording_id=item.recording_id,
                        zarr_path=item.zarr_path,
                        stage=stage.step_name,
                        semantic_state=semantic_state,
                        run_id=run_id,
                        digest=digest,
                    )
                )
    return sorted(
        transitions,
        key=lambda item: (item.recording_id, item.stage, item.semantic_state, item.run_id),
    )


__all__ = [
    "ACTIONABLE_REVIEW_STATES",
    "REGISTERED_GEOMETRY_STAGES",
    "GeometryReviewQueueItem",
    "GeometryReviewRegistryError",
    "GeometryReviewTransition",
    "GeometryStageState",
    "actionable_geometry_transitions",
    "load_geometry_review_queue",
]
