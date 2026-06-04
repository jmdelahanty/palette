"""Helpers for canonical subject-mask stale payloads and resolution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

from .provenance_attrs import resolve_source_keypoints_run

MAX_STALE_INDEX_HISTORY = 2048
_EYE_COMPONENT_NAMES = ("eye_left", "eye_right", "eyes_union")

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


def _merge_int_list(existing: object, new_values: Sequence[int]) -> List[int]:
    return _trimmed_sorted_ints(_coerce_int_list(existing) + [int(v) for v in new_values])


def _iter_group_names(parent: zarr.Group) -> List[str]:
    if hasattr(parent, "group_keys"):
        return sorted(str(name) for name in parent.group_keys())
    return sorted(str(name) for name in parent.keys())


def _iter_component_groups(run_group: zarr.Group) -> Iterable[Tuple[str, zarr.Group]]:
    components = run_group.get("components")
    if not isinstance(components, zarr.Group):
        return []
    if hasattr(components, "group_keys"):
        names = sorted(str(name) for name in components.group_keys())
    else:  # pragma: no cover - compatibility
        names = sorted(str(name) for name in components.keys())
    return [(name, components[name]) for name in names]


def _matches_source_keypoint_lineage(
    attrs: Mapping[str, Any],
    *,
    source_keypoint_group: str,
    source_keypoints_run: str,
) -> bool:
    run_source = resolve_source_keypoints_run(attrs)
    if str(run_source) != str(source_keypoints_run):
        return False

    if not source_keypoint_group:
        return True

    run_group_name = attrs.get("source_keypoint_group")
    if run_group_name not in (None, "", source_keypoint_group):
        return False
    return True


def _subject_stale_component_names(run_group: zarr.Group) -> List[str]:
    component_names = [name for name, _group in _iter_component_groups(run_group)]
    eye_names = [name for name in _EYE_COMPONENT_NAMES if name in set(component_names)]
    return eye_names or component_names


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


def mark_downstream_subject_mask_runs_stale(
    root: zarr.Group,
    *,
    source_keypoint_group: str,
    source_keypoints_run: str,
    roi_indices: Sequence[int],
    frame_indices: Sequence[int],
    reason: str,
    extra: Optional[Mapping[str, object]] = None,
) -> int:
    """Mark refined-subject mask runs that depend on a keypoint run as stale.

    This is the subject-mask successor to the legacy eye-mask
    ``source_keypoint_stale`` marker. It writes the canonical
    ``source_subject_mask_stale`` payload and component pending-row markers so
    existing subject-mask stale resolution tools can acknowledge the update.
    """

    if not source_keypoint_group or not source_keypoints_run:
        return 0

    roi_values = sorted(set(int(v) for v in roi_indices))
    frame_values = sorted(set(int(v) for v in frame_indices))
    if not roi_values and not frame_values:
        return 0

    parent = root.get("refined_subject_masks_runs")
    if not isinstance(parent, zarr.Group):
        return 0

    timestamp = _utc_now()
    touched = 0

    for run_name in _iter_group_names(parent):
        run_group = parent[run_name]
        attrs = dict(run_group.attrs)
        if not _matches_source_keypoint_lineage(
            attrs,
            source_keypoint_group=source_keypoint_group,
            source_keypoints_run=source_keypoints_run,
        ):
            continue

        existing = attrs.get("source_subject_mask_stale")
        payload: Dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
        for key in ("resolved_at_utc", "resolution", "resolved_by", "resolved_notes", "stale_timestamp_utc"):
            payload.pop(key, None)

        component_names = _subject_stale_component_names(run_group)
        existing_components = payload.get("components")
        components_payload: Dict[str, Dict[str, Any]] = (
            {str(key): dict(value) for key, value in existing_components.items() if isinstance(value, Mapping)}
            if isinstance(existing_components, Mapping)
            else {}
        )

        components_parent = run_group.get("components")
        for component_name in component_names:
            component_payload = components_payload.setdefault(str(component_name), {})
            component_payload["roi_indices"] = _merge_int_list(component_payload.get("roi_indices"), roi_values)
            component_payload["frame_indices"] = _merge_int_list(component_payload.get("frame_indices"), frame_values)
            if isinstance(components_parent, zarr.Group) and component_name in components_parent:
                component_group = components_parent[component_name]
                source_run = _component_source_subject_mask_run(
                    run_group,
                    component_name=component_name,
                    component_group=component_group,
                )
                if source_run:
                    component_payload["source_subject_mask_run"] = source_run
                component_group.attrs["source_update_pending_rows"] = _merge_int_list(
                    component_group.attrs.get("source_update_pending_rows"),
                    roi_values,
                )
                source_row_stale = component_group.get("source_row_stale")
                if source_row_stale is not None:
                    row_count = int(getattr(source_row_stale, "shape", (0,))[0])
                    for roi_idx in roi_values:
                        if 0 <= int(roi_idx) < row_count:
                            source_row_stale[int(roi_idx)] = True

        payload["state"] = "stale"
        payload["timestamp_utc"] = timestamp
        payload["reason"] = str(reason).strip() or "keypoint_source_rows_changed"
        payload["source_keypoint_group"] = source_keypoint_group
        payload["source_keypoints_run"] = source_keypoints_run
        payload["roi_indices"] = _merge_int_list(payload.get("roi_indices"), roi_values)
        payload["frame_indices"] = _merge_int_list(payload.get("frame_indices"), frame_values)
        payload["component_names"] = sorted(components_payload)
        payload["components"] = components_payload
        if extra:
            for key, value in extra.items():
                payload[str(key)] = value

        run_group.attrs["source_subject_mask_stale"] = payload
        touched += 1

    return touched


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
    source_keypoint_group: Optional[str] = None,
    source_keypoints_run: Optional[str] = None,
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
        if source_keypoints_run:
            if not _matches_source_keypoint_lineage(
                dict(run_group.attrs),
                source_keypoint_group=str(source_keypoint_group or ""),
                source_keypoints_run=str(source_keypoints_run),
            ):
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
    "mark_downstream_subject_mask_runs_stale",
    "resolve_downstream_subject_mask_runs_stale",
    "resolve_refined_subject_mask_run_stale",
    "sync_source_subject_mask_stale_payload",
]
