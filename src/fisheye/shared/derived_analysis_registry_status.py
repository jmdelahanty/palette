"""Runtime registry status helpers for derived analysis publications."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import zarr

from ..registry.stage_complete import RegistryInput, emit_stage_completion
from .type_conversions import clean_mapping, normalize_attr

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional at runtime
    Console = None  # type: ignore


def emit_eye_angle_stage_completion(
    root: zarr.Group,
    zarr_path: str | Path,
    *,
    run_group: zarr.Group,
    run_name: str,
    source: str,
    registry: RegistryInput = None,
    console: Optional[Console] = None,
    invalidate_on_ok: bool = True,
) -> bool:
    """Project one activated canonical eye-angle run into the registry."""

    attrs = dict(run_group.attrs)
    details = clean_mapping(
        {
            "reason": "present",
            "latest_selector": "runtime_eye_angle_publication",
            "source_eye_geometry_stage": normalize_attr(
                attrs.get("source_eye_geometry_stage")
            ),
            "source_eye_geometry_run": normalize_attr(
                attrs.get("source_eye_geometry_run")
            ),
            "source_subject_shape_run": normalize_attr(
                attrs.get("source_subject_shape_run")
            ),
            "source_refined_subject_mask_run": normalize_attr(
                attrs.get("source_refined_subject_mask_run")
            ),
            "source_keypoints_run": normalize_attr(
                attrs.get("source_keypoints_run")
                or attrs.get("source_keypoint_run")
            ),
            "num_detections": attrs.get("num_detections"),
            "num_frames": attrs.get("num_frames"),
            "fps": attrs.get("fps"),
        }
    )
    return emit_stage_completion(
        root,
        Path(zarr_path),
        step_name="eye_angles",
        status="ok",
        source=source,
        run_name=run_name,
        method=normalize_attr(attrs.get("method"))
        or "ellipse_and_centroid_eye_angles",
        details_json=details,
        registry=registry,
        console=console,
        warning_label="eye_angles",
        invalidate_on_ok=invalidate_on_ok,
        require_complete_invalidation=True,
    )


def emit_track_kinematics_stage_completion(
    root: zarr.Group,
    zarr_path: str | Path,
    *,
    run_group: zarr.Group,
    run_name: str,
    run_type: str,
    source: str,
    registry: RegistryInput = None,
    console: Optional[Console] = None,
    invalidate_on_ok: bool = True,
) -> bool:
    """Project one activated canonical track-kinematics run into the registry."""

    normalized_type = str(run_type).strip().lower()
    if normalized_type not in {"online", "offline"}:
        raise ValueError("run_type must be 'online' or 'offline'")
    leaf_name = str(run_name).strip().strip("/")
    if not leaf_name or "/" in leaf_name or leaf_name in {".", ".."}:
        raise ValueError("run_name must be one nonempty child name")

    attrs = dict(run_group.attrs)
    qualified_name = f"{normalized_type}/{leaf_name}"
    details = clean_mapping(
        {
            "reason": "present",
            "latest_selector": "runtime_track_kinematics_publication",
            "run_type": normalized_type,
            "source_tracking_run": normalize_attr(
                attrs.get("source_tracking_run")
            ),
            "source_keypoints_run": normalize_attr(
                attrs.get("source_keypoints_run")
                or attrs.get("source_keypoint_run")
            ),
            "num_tracks": attrs.get("num_tracks"),
            "fps": attrs.get("fps"),
        }
    )
    return emit_stage_completion(
        root,
        Path(zarr_path),
        step_name="track_kinematics",
        status="ok",
        source=source,
        run_name=qualified_name,
        method=normalize_attr(attrs.get("method")) or "track_kinematics",
        details_json=details,
        registry=registry,
        console=console,
        warning_label="track_kinematics",
        invalidate_on_ok=invalidate_on_ok,
        require_complete_invalidation=True,
    )


__all__ = [
    "emit_eye_angle_stage_completion",
    "emit_track_kinematics_stage_completion",
]
