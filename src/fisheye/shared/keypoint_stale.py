"""Helpers for marking downstream eye-mask runs stale after keypoint edits."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Mapping, Optional, Sequence

import zarr

from .provenance_attrs import resolve_source_keypoints_run

MAX_STALE_INDEX_HISTORY = 2048


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


def _merge_int_list(existing: object, new_values: Sequence[int]) -> List[int]:
    merged = sorted(set(_coerce_int_list(existing) + [int(v) for v in new_values]))
    if len(merged) > MAX_STALE_INDEX_HISTORY:
        return merged[-MAX_STALE_INDEX_HISTORY:]
    return merged


def _iter_group_names(parent: zarr.Group) -> List[str]:
    if hasattr(parent, "group_keys"):
        return sorted(str(name) for name in parent.group_keys())
    return sorted(str(name) for name in parent.keys())


def mark_downstream_eye_mask_runs_stale(
    root: zarr.Group,
    *,
    source_keypoint_group: str,
    source_keypoints_run: str,
    roi_indices: Sequence[int],
    frame_indices: Sequence[int],
    reason: str,
    extra: Optional[Mapping[str, object]] = None,
) -> int:
    """Mark eye-mask runs that depend on a keypoint run as stale.

    Returns the number of run groups updated across eye_masks/refined_eye_masks.
    """

    if not source_keypoint_group or not source_keypoints_run:
        return 0

    roi_values = sorted(set(int(v) for v in roi_indices))
    frame_values = sorted(set(int(v) for v in frame_indices))
    if not roi_values and not frame_values:
        return 0

    timestamp = datetime.now(timezone.utc).isoformat()
    touched = 0

    for parent_name in ("eye_masks_runs", "refined_eye_masks_runs"):
        parent = root.get(parent_name)
        if not isinstance(parent, zarr.Group):
            continue
        for run_name in _iter_group_names(parent):
            run_group = parent[run_name]
            attrs = dict(run_group.attrs)
            run_source = resolve_source_keypoints_run(attrs)
            if str(run_source) != str(source_keypoints_run):
                continue

            run_group_name = attrs.get("source_keypoint_group")
            if run_group_name not in (None, "", source_keypoint_group):
                continue

            existing = attrs.get("source_keypoint_stale")
            payload = dict(existing) if isinstance(existing, dict) else {}
            payload["state"] = "stale"
            payload["timestamp"] = timestamp
            payload["reason"] = reason
            payload["source_keypoint_group"] = source_keypoint_group
            payload["source_keypoints_run"] = source_keypoints_run
            payload["roi_indices"] = _merge_int_list(payload.get("roi_indices"), roi_values)
            payload["frame_indices"] = _merge_int_list(payload.get("frame_indices"), frame_values)
            if extra:
                for key, value in extra.items():
                    payload[key] = value

            run_group.attrs["source_keypoint_stale"] = payload
            touched += 1

    return touched
