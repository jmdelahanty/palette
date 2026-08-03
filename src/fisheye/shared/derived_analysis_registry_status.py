"""Runtime registry status helpers for derived analysis publications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import zarr

from ..registry.stage_complete import RegistryInput, emit_stage_completion
from ..registry.stage_catalog import (
    DERIVED_ANALYSIS,
    canonical_stage_id,
    get_stage_spec,
)
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
    registry_publication_details: Mapping[str, Any] | None = None,
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
                attrs.get("source_keypoints_run") or attrs.get("source_keypoint_run")
            ),
            "num_detections": attrs.get("num_detections"),
            "num_frames": attrs.get("num_frames"),
            "fps": attrs.get("fps"),
            **dict(registry_publication_details or {}),
        }
    )
    return emit_stage_completion(
        root,
        Path(zarr_path),
        step_name="eye_angles",
        status="ok",
        source=source,
        run_name=run_name,
        method=normalize_attr(attrs.get("method")) or "ellipse_and_centroid_eye_angles",
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
    registry_publication_details: Mapping[str, Any] | None = None,
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
            "source_tracking_run": normalize_attr(attrs.get("source_tracking_run")),
            "source_keypoints_run": normalize_attr(
                attrs.get("source_keypoints_run") or attrs.get("source_keypoint_run")
            ),
            "num_tracks": attrs.get("num_tracks"),
            "fps": attrs.get("fps"),
            **dict(registry_publication_details or {}),
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


def emit_derived_analysis_stage_completion(
    root: zarr.Group,
    zarr_path: str | Path,
    *,
    stage_id: str,
    run_group: zarr.Group,
    run_name: str,
    source: str,
    registry: RegistryInput = None,
    console: Optional[Console] = None,
    invalidate_on_ok: bool = True,
    details: Mapping[str, Any] | None = None,
) -> bool:
    """Project one selected canonical derived-analysis run generically.

    Selection, completion, and eligibility are validated by the serial
    finalizer before this function is called. Stage-specific emitters remain
    available where their richer details are useful; this closed generic path
    prevents every maintained table family from inventing another registry
    transaction.
    """

    canonical = canonical_stage_id(stage_id)
    stage = get_stage_spec(canonical)
    if stage.category != DERIVED_ANALYSIS:
        raise ValueError(f"Stage {canonical!r} is not a derived-analysis stage.")
    leaf_name = str(run_name).strip().strip("/")
    if not leaf_name or "/" in leaf_name or leaf_name in {".", ".."}:
        raise ValueError("run_name must be one nonempty child name")
    attrs = dict(run_group.attrs)
    schema_version = attrs.get("schema_version")
    if isinstance(schema_version, bool):
        schema_version = None
    payload = clean_mapping(
        {
            "reason": "present",
            "latest_selector": "serialized_derived_analysis_finalizer",
            "artifact_path": (
                f"{stage.artifact_families[0]}/{leaf_name}"
                if stage.artifact_families
                else None
            ),
            "schema_id": normalize_attr(attrs.get("schema_id")),
            "schema_version": schema_version,
            "layout": normalize_attr(attrs.get("layout")),
            "source_refs": attrs.get("source_refs"),
            "parameters": attrs.get("parameters"),
            "array_schema_manifest_digest": normalize_attr(
                attrs.get("array_schema_manifest_sha256")
                or attrs.get("array_schema_manifest_digest")
                or attrs.get("payload_digest")
            ),
            **dict(details or {}),
        }
    )
    return emit_stage_completion(
        root,
        Path(zarr_path),
        step_name=canonical,
        status="ok",
        source=source,
        run_name=leaf_name,
        method=normalize_attr(attrs.get("method")) or canonical,
        details_json=payload,
        registry=registry,
        console=console,
        warning_label=canonical,
        invalidate_on_ok=invalidate_on_ok,
        require_complete_invalidation=True,
    )


__all__ = [
    "emit_eye_angle_stage_completion",
    "emit_derived_analysis_stage_completion",
    "emit_track_kinematics_stage_completion",
]
