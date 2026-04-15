"""Helpers for canonical subject-mask stale payloads and resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

MAX_STALE_INDEX_HISTORY = 2048

_COMPONENT_SOURCE_ATTR_NAMES = {
    "subject_body": "source_body_subject_mask_run",
    "eye_left": "source_eye_subject_mask_run",
    "eye_right": "source_eye_subject_mask_run",
    "eyes_union": "source_eye_subject_mask_run",
    "swim_bladder": "source_swim_subject_mask_run",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_int_list(value: object) -> List[int]:
    out: List[int] = []
    if value is None:
        return out
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    for item in items:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _trimmed_sorted_ints(values: Sequence[int]) -> List[int]:
    merged = sorted(set(int(v) for v in values))
    if len(merged) > MAX_STALE_INDEX_HISTORY:
        return merged[-MAX_STALE_INDEX_HISTORY:]
    return merged


def _iter_component_groups(run_group: zarr.Group) -> Iterable[Tuple[str, zarr.Group]]:
    components = run_group.get("components")
    if not isinstance(components, zarr.Group):
        return []
    if hasattr(components, "group_keys"):
        names = sorted(str(name) for name in components.group_keys())
    else:  # pragma: no cover - compatibility
        names = sorted(str(name) for name in components.keys())
    return [(name, components[name]) for name in names]


def _component_source_subject_mask_run(
    run_group: zarr.Group,
    *,
    component_name: str,
    component_group: zarr.Group,
) -> Optional[str]:
    provenance = component_group.get("provenance")
    if isinstance(provenance, zarr.Group):
        source_stage = str(provenance.attrs.get("source_stage") or "").strip()
        source_run = str(provenance.attrs.get("source_run") or "").strip()
        if source_stage == "subject_mask_runs" and source_run:
            return source_run

    source_runs = run_group.attrs.get("source_subject_mask_runs")
    if isinstance(source_runs, Mapping):
        value = source_runs.get(component_name)
        if value is not None and str(value).strip():
            return str(value).strip()

    attr_name = _COMPONENT_SOURCE_ATTR_NAMES.get(component_name)
    if attr_name is not None:
        value = run_group.attrs.get(attr_name)
        if value is not None and str(value).strip():
            return str(value).strip()

    fallback = run_group.attrs.get("source_subject_mask_run")
    if fallback is not None and str(fallback).strip():
        return str(fallback).strip()
    return None


def collect_active_subject_mask_stale(run_group: zarr.Group) -> Dict[str, Any]:
    components_payload: Dict[str, Dict[str, Any]] = {}
    all_rows: List[int] = []

    for component_name, component_group in _iter_component_groups(run_group):
        pending_rows = _coerce_int_list(component_group.attrs.get("source_update_pending_rows"))
        if not pending_rows:
            continue
        payload: Dict[str, Any] = {"roi_indices": _trimmed_sorted_ints(pending_rows)}
        source_run = _component_source_subject_mask_run(
            run_group,
            component_name=component_name,
            component_group=component_group,
        )
        if source_run:
            payload["source_subject_mask_run"] = source_run
        components_payload[str(component_name)] = payload
        all_rows.extend(pending_rows)

    return {
        "component_names": sorted(components_payload),
        "roi_indices": _trimmed_sorted_ints(all_rows),
        "components": components_payload,
    }


def sync_source_subject_mask_stale_payload(
    run_group: zarr.Group,
    *,
    reason: str = "source_subject_mask_rows_changed",
) -> Optional[Dict[str, Any]]:
    """Synchronize the run-level stale payload from component-level pending rows."""

    active = collect_active_subject_mask_stale(run_group)
    existing = run_group.attrs.get("source_subject_mask_stale")
    existing_payload = dict(existing) if isinstance(existing, Mapping) else None

    if not active["components"]:
        if existing_payload and str(existing_payload.get("state") or "").strip().lower() == "resolved":
            return existing_payload
        if "source_subject_mask_stale" in run_group.attrs:
            del run_group.attrs["source_subject_mask_stale"]
        return None

    payload: Dict[str, Any] = dict(existing_payload) if existing_payload is not None else {}
    for key in ("resolved_at_utc", "resolution", "resolved_by", "resolved_notes", "stale_timestamp_utc"):
        payload.pop(key, None)

    payload["state"] = "stale"
    payload["timestamp_utc"] = _utc_now()
    payload["reason"] = str(reason).strip() or "source_subject_mask_rows_changed"
    source_subject_mask_run = run_group.attrs.get("source_subject_mask_run")
    if source_subject_mask_run is not None and str(source_subject_mask_run).strip():
        payload["source_subject_mask_run"] = str(source_subject_mask_run).strip()
    payload["component_names"] = list(active["component_names"])
    payload["roi_indices"] = list(active["roi_indices"])
    payload["components"] = dict(active["components"])

    run_group.attrs["source_subject_mask_stale"] = payload
    return payload


def resolve_refined_subject_mask_run_stale(
    run_group: zarr.Group,
    *,
    resolution: str,
    reviewer: Optional[str] = None,
    notes: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Resolve active subject-mask stale markers on one refined subject run."""

    resolution_text = str(resolution or "").strip()
    if not resolution_text:
        return 0

    active = collect_active_subject_mask_stale(run_group)
    existing = run_group.attrs.get("source_subject_mask_stale")
    existing_payload = dict(existing) if isinstance(existing, Mapping) else {}
    state = str(existing_payload.get("state") or "").strip().lower()

    if not active["components"] and state != "stale":
        return 0

    if not existing_payload:
        source_subject_mask_run = run_group.attrs.get("source_subject_mask_run")
        if source_subject_mask_run is not None and str(source_subject_mask_run).strip():
            existing_payload["source_subject_mask_run"] = str(source_subject_mask_run).strip()
        existing_payload["reason"] = "source_subject_mask_rows_changed"
        existing_payload["component_names"] = list(active["component_names"])
        existing_payload["roi_indices"] = list(active["roi_indices"])
        existing_payload["components"] = dict(active["components"])
        existing_payload["timestamp_utc"] = _utc_now()

    stale_timestamp = (
        existing_payload.get("stale_timestamp_utc")
        or existing_payload.get("timestamp_utc")
        or existing_payload.get("timestamp")
        or existing_payload.get("stale_at_utc")
        or existing_payload.get("stale_at")
    )
    resolved_payload: Dict[str, Any] = dict(existing_payload)
    if stale_timestamp is not None:
        resolved_payload["stale_timestamp_utc"] = stale_timestamp
    resolved_payload["state"] = "resolved"
    resolved_payload["resolved_at_utc"] = _utc_now()
    resolved_payload["resolution"] = resolution_text
    if reviewer:
        resolved_payload["resolved_by"] = str(reviewer).strip()
    if notes:
        resolved_payload["resolved_notes"] = str(notes).strip()

    if dry_run:
        return 1

    for component_name, component_payload in active["components"].items():
        components = run_group.get("components")
        if not isinstance(components, zarr.Group) or str(component_name) not in components:
            continue
        component_group = components[str(component_name)]
        row_indices = _coerce_int_list(component_payload.get("roi_indices"))
        pending_rows = _coerce_int_list(component_group.attrs.get("source_update_pending_rows"))
        pending_set = set(row_indices)
        component_group.attrs["source_update_pending_rows"] = [idx for idx in pending_rows if idx not in pending_set]
        source_row_stale = component_group.get("source_row_stale")
        if source_row_stale is None:
            continue
        for roi_idx in row_indices:
            source_row_stale[int(roi_idx)] = False

    run_group.attrs["source_subject_mask_stale"] = resolved_payload
    return 1


def resolve_downstream_subject_mask_runs_stale(
    root: zarr.Group,
    *,
    refined_run: Optional[str] = None,
    source_subject_mask_run: Optional[str] = None,
    resolution: str,
    reviewer: Optional[str] = None,
    notes: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Resolve matching refined-subject stale markers in one archive."""

    parent = root.get("refined_subject_masks_runs")
    if not isinstance(parent, zarr.Group):
        return 0

    if refined_run:
        run_names = [str(refined_run)] if str(refined_run) in parent else []
    elif hasattr(parent, "group_keys"):
        run_names = sorted(str(name) for name in parent.group_keys())
    else:  # pragma: no cover - compatibility
        run_names = sorted(str(name) for name in parent.keys())

    touched = 0
    for run_name in run_names:
        run_group = parent[run_name]
        if source_subject_mask_run:
            candidate = run_group.attrs.get("source_subject_mask_run")
            if str(candidate or "").strip() != str(source_subject_mask_run).strip():
                continue
        touched += resolve_refined_subject_mask_run_stale(
            run_group,
            resolution=resolution,
            reviewer=reviewer,
            notes=notes,
            dry_run=dry_run,
        )

    return touched


__all__ = [
    "collect_active_subject_mask_stale",
    "resolve_downstream_subject_mask_runs_stale",
    "resolve_refined_subject_mask_run_stale",
    "sync_source_subject_mask_stale_payload",
]
