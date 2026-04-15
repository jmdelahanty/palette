"""Runtime registry status helpers for subject-mask stage families."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import zarr

from .registry_stage_complete import emit_stage_completion
from .type_conversions import clean_mapping, normalize_attr

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional at runtime
    Console = None  # type: ignore


def emit_subject_mask_stage_completion(
    root: zarr.Group,
    zarr_path: str | Path,
    *,
    run_group: zarr.Group,
    run_name: str,
    source: str,
    console: Optional[Console] = None,
    invalidate_on_ok: bool = True,
) -> bool:
    attrs = dict(run_group.attrs)
    summary = attrs.get("summary_statistics")
    summary_map = dict(summary) if isinstance(summary, Mapping) else {}
    review_status = attrs.get("subject_mask_review_status")
    review_payload = dict(review_status) if isinstance(review_status, Mapping) else None
    method = normalize_attr(attrs.get("method")) or "subject_masks"
    details = clean_mapping(
        {
            "reason": "present",
            "latest_selector": "runtime_subject_mask_write",
            "source_crop_run": normalize_attr(attrs.get("source_crop_run")),
            "source_keypoints_run": normalize_attr(
                attrs.get("source_keypoints_run") or attrs.get("source_keypoint_run")
            ),
            "source_keypoint_group": normalize_attr(attrs.get("source_keypoint_group")),
            "label_schema_id": normalize_attr(attrs.get("label_schema_id")),
            "run_semantics": normalize_attr(attrs.get("run_semantics")),
            "rows_total": summary_map.get("rows_total"),
            "rows_with_nonempty_masks": summary_map.get("rows_with_nonempty_masks"),
            "duration_seconds": summary_map.get("duration_seconds") or attrs.get("duration_seconds"),
        }
    )
    return emit_stage_completion(
        root,
        Path(zarr_path),
        step_name="subject_masks",
        status="ok",
        source=source,
        run_name=run_name,
        method=method,
        coverage_pct=None,
        review_status_json=review_payload,
        details_json=details,
        console=console,
        warning_label="subject_masks",
        invalidate_on_ok=invalidate_on_ok,
    )


def emit_refined_subject_mask_stage_completion(
    root: zarr.Group,
    zarr_path: str | Path,
    *,
    run_group: zarr.Group,
    run_name: str,
    source: str,
    console: Optional[Console] = None,
    invalidate_on_ok: bool = True,
) -> bool:
    attrs = dict(run_group.attrs)
    summary = attrs.get("summary_statistics")
    summary_map = dict(summary) if isinstance(summary, Mapping) else {}
    review_status = attrs.get("refined_subject_mask_review_status")
    review_payload = dict(review_status) if isinstance(review_status, Mapping) else None
    stale_payload = attrs.get("source_subject_mask_stale")
    stale_map = dict(stale_payload) if isinstance(stale_payload, Mapping) else {}
    method = normalize_attr(attrs.get("method")) or "refine_subject_masks"
    details = clean_mapping(
        {
            "reason": "present",
            "latest_selector": "runtime_refined_subject_mask_write",
            "source_subject_mask_run": normalize_attr(attrs.get("source_subject_mask_run")),
            "label_schema_id": normalize_attr(attrs.get("label_schema_id")),
            "component_names": attrs.get("mask_labels"),
            "stale_state": normalize_attr(stale_map.get("state")),
            "stale_reason": normalize_attr(stale_map.get("reason")),
            "duration_seconds": summary_map.get("duration_seconds") or attrs.get("duration_seconds"),
        }
    )
    return emit_stage_completion(
        root,
        Path(zarr_path),
        step_name="refined_subject_masks",
        status="ok",
        source=source,
        run_name=run_name,
        method=method,
        coverage_pct=None,
        review_status_json=review_payload,
        details_json=details,
        console=console,
        warning_label="refined_subject_masks",
        invalidate_on_ok=invalidate_on_ok,
    )


__all__ = [
    "emit_refined_subject_mask_stage_completion",
    "emit_subject_mask_stage_completion",
]
