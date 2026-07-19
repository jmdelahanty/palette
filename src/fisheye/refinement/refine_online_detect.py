#!/usr/bin/env python3
"""
Online Detection Refinement Pipeline

Refines online (H5-imported) target position data from stimulus runs by applying
smoothing, outlier removal, and gap interpolation to reduce tracking artifacts.

Workflow:
1. Load online target positions from stimulus run (chaser_states)
2. Resolve their persisted coordinate descriptor without changing coordinates
3. Smooth positions using Savitzky-Golay filter
4. Detect and remove outliers (large jumps, teleportation artifacts)
5. Interpolate small gaps
6. Save refined data in the same native space with descriptor provenance

This creates a new refined_online_runs group similar to refined_detect_runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console
from scipy.signal import savgol_filter

from ..analysis.chaser_metrics_loader import load_chaser_metrics
from ..shared.calibration import load_run_calibration
from ..shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    PIXEL_SPACE_IDS,
    CoordinateDescriptor,
    CoordinateDescriptorError,
    CoordinateIssue,
    CoordinateRecordRef,
    build_coordinate_descriptor,
    coordinate_descriptor_digest,
    load_coordinate_descriptor_attrs,
    parse_coordinate_descriptor,
    stamp_coordinate_descriptor,
)
from ..shared.system_metadata import get_environment_info, get_git_info

REFINED_ONLINE_GROUP = "refined_online_runs"

_POSITION_FIELDS = ("target_pos_x", "target_pos_y")
_LEGACY_TEXTURE_FRAMES = frozenset({"texture", "texture_px", "stimulus_texture_px"})


def _coordinate_error(code: str, path: str, message: str) -> CoordinateDescriptorError:
    return CoordinateDescriptorError((CoordinateIssue(code=code, path=path, message=message),))


def _required_source_text(attrs: Mapping[str, object], key: str) -> str:
    value = attrs.get(key)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, str) or not value.strip():
        raise _coordinate_error(
            "online_source_coordinate_attr_missing",
            f"$.source_attrs.{key}",
            f"Online position source must explicitly declare {key!r}.",
        )
    return value.strip()


def _position_field_names(value: object) -> tuple[str, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if isinstance(value, str):
        fields = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Mapping):
        fields = tuple(
            str(item).strip()
            for item in (*value.keys(), *value.values())
            if str(item).strip()
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        fields = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        fields = ()
    return fields


def _validate_legacy_xy_source_attrs(
    attrs: Mapping[str, object],
    *,
    expected_origins: frozenset[str],
) -> None:
    units_value = attrs.get("coordinate_units") or attrs.get("units") or ""
    if isinstance(units_value, bytes):
        units_value = units_value.decode("utf-8", errors="strict")
    units = str(units_value).strip().lower()
    if units not in {"px", "pixel", "pixels"}:
        raise _coordinate_error(
            "online_source_units_missing",
            "$.source_attrs.coordinate_units",
            "Legacy online coordinates require explicit pixel units.",
        )
    origin = _required_source_text(attrs, "coordinate_origin")
    if origin not in expected_origins:
        raise _coordinate_error(
            "online_source_origin_unsupported",
            "$.source_attrs.coordinate_origin",
            f"Unsupported online coordinate origin {origin!r}.",
        )
    x_direction = _required_source_text(attrs, "x_axis_direction")
    y_direction = _required_source_text(attrs, "y_axis_direction")
    if x_direction != "right" or y_direction != "down":
        raise _coordinate_error(
            "online_source_axes_unsupported",
            "$.source_attrs",
            "Legacy online coordinates must explicitly declare +X right and +Y down.",
        )
    fields = _position_field_names(attrs.get("position_fields"))
    if not set(_POSITION_FIELDS).issubset(fields):
        raise _coordinate_error(
            "online_source_position_fields_missing",
            "$.source_attrs.position_fields",
            "Online coordinate metadata must explicitly bind target_pos_x and target_pos_y.",
        )


def _positive_extent(value: object, *, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _coordinate_error(
            "online_reference_extent_invalid",
            path,
            "Reference dimension must be a positive finite number.",
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise _coordinate_error(
            "online_reference_extent_invalid",
            path,
            "Reference dimension must be a positive finite number.",
        )
    return int(numeric) if numeric.is_integer() else numeric


def _mapping_attr(value: object, *, path: str) -> Mapping[str, object]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _coordinate_error(
                "online_transform_record_invalid",
                path,
                f"Transform record is not valid JSON: {exc.msg}.",
            ) from exc
    if not isinstance(value, Mapping):
        raise _coordinate_error(
            "online_transform_record_invalid",
            path,
            "Transform evidence must be a mapping or JSON object.",
        )
    return value


def _record_sha256(value: Mapping[str, object]) -> str:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise _coordinate_error(
            "online_transform_record_invalid",
            "$.stimulus_run_attrs",
            "Transform evidence is not canonical JSON metadata.",
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_xy_descriptor(descriptor: CoordinateDescriptor) -> CoordinateDescriptor:
    if descriptor.geometry_type not in {"point_xy", "points_xy"}:
        raise _coordinate_error(
            "online_geometry_unsupported",
            "$.coordinate_descriptor.geometry_type",
            "Online target positions require point_xy or points_xy geometry.",
        )
    if descriptor.components != ("x", "y"):
        raise _coordinate_error(
            "online_components_unsupported",
            "$.coordinate_descriptor.components",
            "Online target positions require ordered x,y components.",
        )
    if descriptor.space_id not in PIXEL_SPACE_IDS or descriptor.component_units != (
        "px",
        "px",
    ):
        raise _coordinate_error(
            "online_pixel_space_required",
            "$.coordinate_descriptor",
            "refined_online positions_px requires a controlled pixel space with px,px components.",
        )
    return descriptor


def resolve_online_coordinate_descriptor(
    online_coordinate_metadata: Mapping[str, object],
    *,
    stimulus_run_path: str,
    stimulus_run_attrs: Mapping[str, object],
    arena_geometry_attrs: Mapping[str, object] | None,
) -> CoordinateDescriptor:
    """Resolve the native online-position space from persisted evidence only.

    A canonical descriptor is authoritative. Historical group attrs are accepted
    only through the two explicit compatibility rules below; absent dimensions,
    axes, units, origin, field binding, or transform evidence fail closed.
    """

    if not isinstance(online_coordinate_metadata, Mapping):
        raise _coordinate_error(
            "online_coordinate_metadata_missing",
            "$.online_coordinate_metadata",
            "Chaser metrics did not expose online coordinate metadata.",
        )
    source_path_value = online_coordinate_metadata.get("source_path")
    if not isinstance(source_path_value, str) or not source_path_value.strip():
        raise _coordinate_error(
            "online_coordinate_source_path_missing",
            "$.online_coordinate_metadata.source_path",
            "Exact chaser_states source path is required.",
        )
    source_path = source_path_value.strip()
    source_attrs_value = online_coordinate_metadata.get("source_attrs")
    if not isinstance(source_attrs_value, Mapping):
        raise _coordinate_error(
            "online_coordinate_source_attrs_missing",
            "$.online_coordinate_metadata.source_attrs",
            "Unmodified chaser_states source attrs are required.",
        )
    source_attrs = source_attrs_value

    if COORDINATE_DESCRIPTOR_ATTR in source_attrs:
        descriptor = load_coordinate_descriptor_attrs(source_attrs)
        return _validate_xy_descriptor(descriptor)

    coordinate_frame = _required_source_text(source_attrs, "coordinate_frame")
    lineage_ref = CoordinateRecordRef(ref=source_path)
    source_row_ref = f"{source_path}#rows"

    if coordinate_frame == "arena_relative_canvas_px":
        _validate_legacy_xy_source_attrs(
            source_attrs,
            expected_origins=frozenset({"top_left_of_active_arena"}),
        )
        if not isinstance(arena_geometry_attrs, Mapping):
            raise _coordinate_error(
                "online_arena_geometry_missing",
                "$.calibration.arena_geometry",
                "arena_relative_canvas_px requires selected-run arena_geometry metadata.",
            )
        width = _positive_extent(
            arena_geometry_attrs.get("arena_region_width_px"),
            path="$.calibration.arena_geometry.arena_region_width_px",
        )
        height = _positive_extent(
            arena_geometry_attrs.get("arena_region_height_px"),
            path="$.calibration.arena_geometry.arena_region_height_px",
        )
        authority = (
            f"{stimulus_run_path}/calibration/arena_geometry.attrs"
            "[arena_region_width_px,arena_region_height_px]"
        )
        return build_coordinate_descriptor(
            space_id="arena_relative_canvas_px",
            geometry_type="points_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            origin="arena_top_left",
            positive_x="right",
            positive_y="down",
            reference_width=width,
            reference_height=height,
            reference_units="px",
            reference_authority=authority,
            pixel_convention="continuous",
            row_identity_mode="sample_indices",
            row_identity_array_ref=source_row_ref,
            source_camera_overlay="requires_transform",
            lineage_refs=(lineage_ref,),
        )

    if coordinate_frame in _LEGACY_TEXTURE_FRAMES:
        _validate_legacy_xy_source_attrs(
            source_attrs,
            expected_origins=frozenset({"top_left", "top_left_of_texture"}),
        )
        transform_status = str(
            stimulus_run_attrs.get("coordinate_transform_status") or ""
        ).strip()
        if transform_status == "legacy_run_level_texture_to_camera":
            transform_attr = "coordinate_transform"
            expected_scope = "run_level_legacy_texture_space"
        elif transform_status == "suppressed_child_group_coordinate_metadata_authoritative":
            transform_attr = "legacy_texture_to_camera_transform"
            expected_scope = "legacy_texture_space_fallback"
        else:
            raise _coordinate_error(
                "online_legacy_texture_evidence_missing",
                "$.stimulus_run_attrs.coordinate_transform_status",
                "Explicit legacy texture coordinates require a recognized transform status.",
            )
        if transform_attr not in stimulus_run_attrs:
            raise _coordinate_error(
                "online_legacy_texture_evidence_missing",
                f"$.stimulus_run_attrs.{transform_attr}",
                "Exact legacy texture transform evidence is missing.",
            )
        transform = _mapping_attr(
            stimulus_run_attrs[transform_attr],
            path=f"$.stimulus_run_attrs.{transform_attr}",
        )
        if transform.get("scope") != expected_scope:
            raise _coordinate_error(
                "online_legacy_texture_scope_invalid",
                f"$.stimulus_run_attrs.{transform_attr}.scope",
                f"Expected legacy transform scope {expected_scope!r}.",
            )
        texture_dimensions = transform.get("texture_dimensions")
        camera_dimensions = transform.get("camera_dimensions")
        if not isinstance(texture_dimensions, Sequence) or isinstance(
            texture_dimensions, (str, bytes, bytearray)
        ) or len(texture_dimensions) != 2:
            raise _coordinate_error(
                "online_legacy_texture_dimensions_invalid",
                f"$.stimulus_run_attrs.{transform_attr}.texture_dimensions",
                "Exact texture width and height are required.",
            )
        if not isinstance(camera_dimensions, Sequence) or isinstance(
            camera_dimensions, (str, bytes, bytearray)
        ) or len(camera_dimensions) != 2:
            raise _coordinate_error(
                "online_legacy_camera_dimensions_invalid",
                f"$.stimulus_run_attrs.{transform_attr}.camera_dimensions",
                "Exact camera width and height are required.",
            )
        texture_width = _positive_extent(
            texture_dimensions[0],
            path=f"$.stimulus_run_attrs.{transform_attr}.texture_dimensions[0]",
        )
        texture_height = _positive_extent(
            texture_dimensions[1],
            path=f"$.stimulus_run_attrs.{transform_attr}.texture_dimensions[1]",
        )
        camera_width = _positive_extent(
            camera_dimensions[0],
            path=f"$.stimulus_run_attrs.{transform_attr}.camera_dimensions[0]",
        )
        camera_height = _positive_extent(
            camera_dimensions[1],
            path=f"$.stimulus_run_attrs.{transform_attr}.camera_dimensions[1]",
        )
        scale = _positive_extent(
            transform.get("texture_to_camera_scale"),
            path=f"$.stimulus_run_attrs.{transform_attr}.texture_to_camera_scale",
        )
        scale_value = float(scale)
        if not math.isclose(float(camera_width) / float(texture_width), scale_value) or not math.isclose(
            float(camera_height) / float(texture_height), scale_value
        ):
            raise _coordinate_error(
                "online_legacy_texture_scale_inconsistent",
                f"$.stimulus_run_attrs.{transform_attr}",
                "Legacy scale does not match both exact texture and camera dimensions.",
            )
        transform_ref_path = f"{stimulus_run_path}.attrs[{transform_attr}]"
        transform_ref = CoordinateRecordRef(
            ref=transform_ref_path,
            sha256=_record_sha256(transform),
        )
        return build_coordinate_descriptor(
            space_id="stimulus_texture_px",
            geometry_type="points_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            origin="top_left",
            positive_x="right",
            positive_y="down",
            reference_width=texture_width,
            reference_height=texture_height,
            reference_units="px",
            reference_authority=(
                f"{transform_ref_path}.texture_dimensions"
            ),
            pixel_convention="continuous",
            row_identity_mode="sample_indices",
            row_identity_array_ref=source_row_ref,
            source_camera_overlay="requires_transform",
            legacy_space_label="texture",
            lineage_refs=(lineage_ref,),
            transform_refs=(transform_ref,),
        )

    raise _coordinate_error(
        "online_coordinate_space_unsupported",
        "$.source_attrs.coordinate_frame",
        f"Unsupported or ambiguous online coordinate frame {coordinate_frame!r}.",
    )


def _bind_positions_descriptor(
    descriptor: CoordinateDescriptor,
    *,
    row_identity_array_ref: str,
    source_path: str,
) -> CoordinateDescriptor:
    """Bind native coordinate semantics to a concrete output row array."""

    payload = descriptor.to_dict()
    payload["row_identity"] = {
        "mode": "frame_indices",
        "array_ref": row_identity_array_ref,
    }
    refs = list(payload.get("lineage_refs") or [])
    if not any(isinstance(item, Mapping) and item.get("ref") == source_path for item in refs):
        refs.append({"ref": source_path})
    payload["lineage_refs"] = refs
    return parse_coordinate_descriptor(payload)


def load_online_positions(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    console: Optional[Console] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """Load online target positions without changing their persisted native space.

    Args:
        zarr_path: Path to zarr archive
        stimulus_run: Stimulus run name (defaults to latest)
        chaser_index: Chaser index to load
        console: Rich console for output

    Returns:
        Tuple of (camera_frames, positions_native, valid_mask, texture_to_camera_scale,
                  pixels_per_mm_projector, metadata)

    Note:
        The six-item tuple is retained for compatibility. ``metadata`` contains
        the authoritative ``coordinate_descriptor`` for ``positions_native``.
    """
    if console is None:
        console = Console()

    # Load chaser metrics bundle
    bundle = load_chaser_metrics(
        zarr_path,
        stimulus_run=stimulus_run,
        chaser_index=chaser_index,
    )

    # Extract online target positions in the source's native coordinate space.
    target_pos_x_raw = bundle.online.get("target_pos_x")
    target_pos_y_raw = bundle.online.get("target_pos_y")

    if target_pos_x_raw is None or target_pos_y_raw is None:
        raise ValueError("No target position data in online metrics")

    # Resolve coordinate semantics before any output run can be created.
    root = zarr.open(str(zarr_path), mode="r")
    stimulus_run_name = bundle.provenance.get("stimulus_run")

    if not isinstance(stimulus_run_name, str) or not stimulus_run_name:
        raise _coordinate_error(
            "online_stimulus_run_missing",
            "$.provenance.stimulus_run",
            "The selected stimulus run is required to resolve online coordinates.",
        )
    analysis_group = root.get("analysis")
    stimulus_parent = analysis_group.get("stimulus_runs") if analysis_group is not None else None
    stimulus_group = (
        stimulus_parent.get(stimulus_run_name)
        if stimulus_parent is not None
        else None
    )
    if stimulus_group is None:
        raise _coordinate_error(
            "online_stimulus_run_missing",
            "$.provenance.stimulus_run",
            f"Selected stimulus run {stimulus_run_name!r} is unavailable.",
        )
    calibration_group = stimulus_group.get("calibration")
    arena_geometry_group = (
        calibration_group.get("arena_geometry")
        if calibration_group is not None
        else None
    )
    source_descriptor = resolve_online_coordinate_descriptor(
        bundle.online_coordinate_metadata,
        stimulus_run_path=str(stimulus_group.path),
        stimulus_run_attrs=dict(stimulus_group.attrs),
        arena_geometry_attrs=(
            dict(arena_geometry_group.attrs)
            if arena_geometry_group is not None
            else None
        ),
    )
    source_path = str(bundle.online_coordinate_metadata.get("source_path") or "").strip()
    coordinate_descriptor = _bind_positions_descriptor(
        source_descriptor,
        row_identity_array_ref="camera_frame_ids",
        source_path=source_path,
    )

    # Keep legacy tuple calibration fields for callers; they do not transform positions.
    if stimulus_run_name:
        try:
            calibration = load_run_calibration(root, stimulus_run_name)
            texture_to_camera_scale = float(calibration.texture_to_camera_scale)
            pixels_per_mm_projector = calibration.pixels_per_mm_projector
            console.print(
                f"[cyan]Loaded calibration ({calibration.source}): texture_to_camera_scale = {texture_to_camera_scale:.6f}"
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Failed to load calibration: {exc}")
            texture_to_camera_scale = 1.0
            pixels_per_mm_projector = None
    else:
        texture_to_camera_scale = 1.0
        pixels_per_mm_projector = None

    target_pos_x = np.asarray(target_pos_x_raw, dtype=np.float64)
    target_pos_y = np.asarray(target_pos_y_raw, dtype=np.float64)
    camera_frames = bundle.camera_frame_ids

    # Create the native-space position array.
    positions = np.column_stack([target_pos_x, target_pos_y])

    # Valid mask (non-NaN positions)
    valid_mask = np.isfinite(positions[:, 0]) & np.isfinite(positions[:, 1])

    # Metadata
    metadata = {
        "stimulus_run": stimulus_run_name,
        "chaser_index": chaser_index,
        "total_frames": len(camera_frames),
        "valid_frames": int(valid_mask.sum()),
        "coverage_percent": float(valid_mask.sum() / len(camera_frames) * 100),
        "texture_to_camera_scale": texture_to_camera_scale,
        "pixels_per_mm_projector": pixels_per_mm_projector,
        "coordinate_space": coordinate_descriptor.space_id,
        "coordinate_descriptor": coordinate_descriptor.to_dict(),
        "coordinate_descriptor_sha256": coordinate_descriptor.digest(),
        "online_coordinate_source": dict(bundle.online_coordinate_metadata),
    }

    return camera_frames, positions, valid_mask, texture_to_camera_scale, pixels_per_mm_projector, metadata


def smooth_positions(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth positions using Savitzky-Golay filter.

    Args:
        positions: Position array (N, 2)
        valid_mask: Boolean mask of valid positions
        window_length: Filter window length (must be odd)
        polyorder: Polynomial order for fitting

    Returns:
        Tuple of (smoothed_positions, smoothed_mask)
    """
    smoothed = np.full_like(positions, np.nan)
    smoothed_mask = np.zeros(len(positions), dtype=bool)

    # Need at least window_length consecutive valid points to smooth
    if valid_mask.sum() < window_length:
        return positions.copy(), valid_mask.copy()

    # Find consecutive valid segments
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return positions.copy(), valid_mask.copy()

    # Group consecutive indices
    segments = []
    start_idx = valid_indices[0]
    for i in range(1, len(valid_indices)):
        if valid_indices[i] != valid_indices[i - 1] + 1:
            # Gap found, save segment
            segments.append((start_idx, valid_indices[i - 1]))
            start_idx = valid_indices[i]
    segments.append((start_idx, valid_indices[-1]))

    # Smooth each segment
    for start, end in segments:
        segment_length = end - start + 1

        if segment_length < window_length:
            # Too short to smooth, keep original
            smoothed[start : end + 1] = positions[start : end + 1]
            smoothed_mask[start : end + 1] = True
            continue

        # Apply Savitzky-Golay filter
        for axis in [0, 1]:
            smoothed[start : end + 1, axis] = savgol_filter(
                positions[start : end + 1, axis],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            )

        smoothed_mask[start : end + 1] = True

    return smoothed, smoothed_mask


def detect_outliers(
    positions: np.ndarray,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    displacement_threshold: float = 100.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect outliers based on displacement threshold.

    Args:
        positions: Position array (N, 2)
        frames: Frame indices
        valid_mask: Boolean mask of valid positions
        displacement_threshold: Maximum reasonable displacement in native coordinate units

    Returns:
        Tuple of (outlier_mask, outlier_stats)
    """
    outlier_mask = np.zeros(len(positions), dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return outlier_mask, {"outliers_detected": 0, "threshold": displacement_threshold}

    # Calculate frame-to-frame displacement
    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        # Only check consecutive frames
        if frames[idx2] - frames[idx1] != 1:
            continue

        displacement = np.linalg.norm(positions[idx2] - positions[idx1])

        if displacement > displacement_threshold:
            # Mark the second point as outlier (assumes first is correct)
            outlier_mask[idx2] = True

    outlier_stats = {
        "outliers_detected": int(outlier_mask.sum()),
        "threshold": float(displacement_threshold),
        "outlier_rate": float(outlier_mask.sum() / valid_mask.sum() * 100) if valid_mask.sum() > 0 else 0.0,
    }

    return outlier_mask, outlier_stats


def interpolate_gaps(
    positions: np.ndarray,
    frames: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Interpolate small gaps in position data.

    Args:
        positions: Position array (N, 2)
        frames: Frame indices
        valid_mask: Boolean mask of valid positions
        max_gap: Maximum gap size to interpolate (frames)

    Returns:
        Tuple of (interpolated_positions, interpolation_mask, interp_stats)
    """
    interpolated = positions.copy()
    interpolation_mask = np.zeros(len(positions), dtype=bool)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 2:
        return interpolated, interpolation_mask, {"gaps_filled": 0, "interpolated_frames": 0}

    # Find gaps
    gaps_filled = 0
    interpolated_frames = 0

    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        frame1 = frames[idx1]
        frame2 = frames[idx2]
        gap_size = frame2 - frame1 - 1

        if gap_size <= 0 or gap_size > max_gap:
            continue

        # Find indices to interpolate
        gap_indices = []
        for j in range(idx1 + 1, idx2):
            if frames[j] > frame1 and frames[j] < frame2:
                gap_indices.append(j)

        if len(gap_indices) == 0:
            continue

        # Linear interpolation
        t = (frames[gap_indices] - frame1) / (frame2 - frame1)
        for axis in [0, 1]:
            interpolated[gap_indices, axis] = positions[idx1, axis] + t * (
                positions[idx2, axis] - positions[idx1, axis]
            )

        interpolation_mask[gap_indices] = True
        gaps_filled += 1
        interpolated_frames += len(gap_indices)

    interp_stats = {
        "gaps_filled": gaps_filled,
        "interpolated_frames": interpolated_frames,
        "max_gap": max_gap,
    }

    return interpolated, interpolation_mask, interp_stats


def refine_online_positions(
    zarr_path: str,
    stimulus_run: Optional[str] = None,
    chaser_index: int = 0,
    window_length: int = 11,
    polyorder: int = 3,
    displacement_threshold: float = 100.0,
    max_gap: int = 20,
    console: Optional[Console] = None,
    created_at_utc: Optional[str] = None,
) -> str:
    """Refine online target positions with smoothing, outlier removal, and interpolation.

    Args:
        zarr_path: Path to zarr archive
        stimulus_run: Stimulus run name (defaults to latest)
        chaser_index: Chaser index to load
        window_length: Savitzky-Golay filter window length (must be odd)
        polyorder: Polynomial order for Savitzky-Golay filter
        displacement_threshold: Maximum reasonable displacement in native coordinate units
        max_gap: Maximum gap size to interpolate (frames)
        console: Rich console for output
        created_at_utc: Optional creation timestamp

    Returns:
        Name of created refined run
    """
    if console is None:
        console = Console()

    console.rule("[bold]Online Detection Refinement[/bold]")
    start_time = time.perf_counter()

    # Step 1: Load data
    console.print("[bold]Step 1: Loading Online Data[/bold]")
    frames, positions, valid_mask, scale, pixels_per_mm_projector, metadata = load_online_positions(
        zarr_path, stimulus_run, chaser_index, console
    )
    coordinate_descriptor = parse_coordinate_descriptor(
        metadata.get("coordinate_descriptor")
    )
    stored_descriptor_digest = metadata.get("coordinate_descriptor_sha256")
    if stored_descriptor_digest != coordinate_descriptor_digest(coordinate_descriptor):
        raise _coordinate_error(
            "online_descriptor_digest_mismatch",
            "$.metadata.coordinate_descriptor_sha256",
            "Online position descriptor digest is absent or inconsistent.",
        )
    source_metadata = metadata.get("online_coordinate_source")
    if not isinstance(source_metadata, Mapping):
        raise _coordinate_error(
            "online_coordinate_metadata_missing",
            "$.metadata.online_coordinate_source",
            "Exact chaser_states source metadata is required for refined output lineage.",
        )
    source_path = str(source_metadata.get("source_path") or "").strip()
    if not source_path:
        raise _coordinate_error(
            "online_coordinate_source_path_missing",
            "$.metadata.online_coordinate_source.source_path",
            "Exact chaser_states source path is required for refined output lineage.",
        )
    output_descriptor = _bind_positions_descriptor(
        coordinate_descriptor,
        row_identity_array_ref="camera_frame_ids",
        source_path=source_path,
    )

    console.print(f"  Stimulus run: [cyan]{metadata['stimulus_run']}[/cyan]")
    console.print(f"  Total frames: {metadata['total_frames']}")
    console.print(f"  Valid positions: {metadata['valid_frames']} ({metadata['coverage_percent']:.1f}%)")
    console.print(f"  Coordinate space: {metadata['coordinate_space']}")
    if pixels_per_mm_projector:
        console.print(f"  Projector calibration: {pixels_per_mm_projector:.6f} pixels/mm")

    # Step 2: Smooth positions
    console.print("\n[bold]Step 2: Smoothing Positions[/bold]")
    console.print(f"  Window length: {window_length}")
    console.print(f"  Polynomial order: {polyorder}")

    smoothed_positions, smoothed_mask = smooth_positions(
        positions, valid_mask, window_length, polyorder
    )

    console.print(f"  Smoothed frames: {smoothed_mask.sum()}")

    # Step 3: Detect outliers
    console.print("\n[bold]Step 3: Detecting Outliers[/bold]")
    native_units = (
        coordinate_descriptor.component_units[0]
        if len(set(coordinate_descriptor.component_units)) == 1
        else "native coordinate units"
    )
    console.print(f"  Displacement threshold: {displacement_threshold} {native_units}")

    outlier_mask, outlier_stats = detect_outliers(
        smoothed_positions, frames, smoothed_mask, displacement_threshold
    )

    console.print(f"  Outliers detected: {outlier_stats['outliers_detected']} ({outlier_stats['outlier_rate']:.2f}%)")

    # Remove outliers
    clean_mask = smoothed_mask & ~outlier_mask
    clean_positions = smoothed_positions.copy()
    clean_positions[~clean_mask] = np.nan

    # Step 4: Interpolate gaps
    console.print("\n[bold]Step 4: Interpolating Gaps[/bold]")
    console.print(f"  Max gap: {max_gap} frames")

    interpolated_positions, interpolation_mask, interp_stats = interpolate_gaps(
        clean_positions, frames, clean_mask, max_gap
    )

    console.print(f"  Gaps filled: {interp_stats['gaps_filled']}")
    console.print(f"  Interpolated frames: {interp_stats['interpolated_frames']}")

    # Final statistics
    final_valid = np.isfinite(interpolated_positions[:, 0]) & np.isfinite(interpolated_positions[:, 1])
    final_coverage = final_valid.sum() / len(frames) * 100

    console.print("\n[bold]Coverage Comparison:[/bold]")
    console.print(f"  Original: {metadata['valid_frames']} frames ({metadata['coverage_percent']:.1f}%)")
    console.print(f"  After smoothing: {smoothed_mask.sum()} frames ({smoothed_mask.sum()/len(frames)*100:.1f}%)")
    console.print(f"  After outlier removal: {clean_mask.sum()} frames ({clean_mask.sum()/len(frames)*100:.1f}%)")
    console.print(f"  After interpolation: {final_valid.sum()} frames ({final_coverage:.1f}%)")

    # Step 5: Save
    console.print("\n[bold]Step 5: Saving Refined Run[/bold]")

    root = zarr.open(zarr_path, mode="a")

    if REFINED_ONLINE_GROUP not in root:
        root.create_group(REFINED_ONLINE_GROUP)
    refined_runs = root[REFINED_ONLINE_GROUP]

    # Create timestamped run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"refined_online_{timestamp}"
    refined_group = refined_runs.create_group(run_name)
    refined_runs.attrs["latest"] = run_name

    # Save filtered data (after smoothing and outlier removal)
    filtered_grp = refined_group.create_group("filtered")
    filtered_grp.create_array("camera_frame_ids", data=frames, chunks=(10000,))
    filtered_positions_array = filtered_grp.create_array(
        "positions_px", data=clean_positions, chunks=(10000, 2)
    )
    stamp_coordinate_descriptor(filtered_positions_array, output_descriptor)
    filtered_grp.create_array("valid_mask", data=clean_mask, chunks=(10000,))

    filtered_grp.attrs["total_frames"] = len(frames)
    filtered_grp.attrs["valid_frames"] = int(clean_mask.sum())
    filtered_grp.attrs["coverage_percent"] = float(clean_mask.sum() / len(frames) * 100)
    filtered_grp.attrs["smoothing_applied"] = True
    filtered_grp.attrs["outliers_removed"] = outlier_stats["outliers_detected"]

    # Save interpolated data (final refined positions)
    interp_grp = refined_group.create_group("interpolated")
    interp_grp.create_array("camera_frame_ids", data=frames, chunks=(10000,))
    interpolated_positions_array = interp_grp.create_array(
        "positions_px", data=interpolated_positions, chunks=(10000, 2)
    )
    stamp_coordinate_descriptor(interpolated_positions_array, output_descriptor)
    interp_grp.create_array("valid_mask", data=final_valid, chunks=(10000,))
    interp_grp.create_array("interpolation_mask", data=interpolation_mask, chunks=(10000,))

    interp_grp.attrs["total_frames"] = len(frames)
    interp_grp.attrs["valid_frames"] = int(final_valid.sum())
    interp_grp.attrs["coverage_percent"] = float(final_coverage)
    interp_grp.attrs["gaps_filled"] = interp_stats["gaps_filled"]
    interp_grp.attrs["interpolated_frames"] = interp_stats["interpolated_frames"]

    # Store metadata arrays for tracking pipeline stages
    refined_group.create_array("camera_frame_ids", data=frames, chunks=(10000,))
    refined_group.create_array("original_valid_mask", data=valid_mask, chunks=(10000,))
    refined_group.create_array("smoothed_mask", data=smoothed_mask, chunks=(10000,))
    refined_group.create_array("outlier_mask", data=outlier_mask, chunks=(10000,))

    # Metadata
    duration = time.perf_counter() - start_time
    created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()

    parameters = {
        "window_length": window_length,
        "polyorder": polyorder,
        "displacement_threshold": displacement_threshold,
        "displacement_threshold_units": native_units,
        "max_gap": max_gap,
    }

    coverage_stats = {
        "original": {
            "valid_frames": metadata["valid_frames"],
            "coverage_percent": metadata["coverage_percent"],
        },
        "smoothed": {
            "valid_frames": int(smoothed_mask.sum()),
            "coverage_percent": float(smoothed_mask.sum() / len(frames) * 100),
        },
        "clean": {
            "valid_frames": int(clean_mask.sum()),
            "coverage_percent": float(clean_mask.sum() / len(frames) * 100),
            "outliers_removed": outlier_stats["outliers_detected"],
        },
        "final": {
            "valid_frames": int(final_valid.sum()),
            "coverage_percent": float(final_coverage),
            "interpolated_frames": interp_stats["interpolated_frames"],
        },
    }

    git_info = get_git_info()
    env_info = get_environment_info()
    environment_info = {
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "python_version": env_info["platform"].get("python_version", "unknown"),
        "system": env_info["platform"].get("system", "unknown"),
        "release": env_info["platform"].get("release", "unknown"),
    }

    provenance_record = {
        "stage": "refine_online_detect",
        "command": " ".join(sys.argv),
        "created_at_utc": created_timestamp,
        "version": git_info.get("short_hash") or git_info.get("commit_hash"),
        "git": {
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        "environment": environment_info,
        "parameters": parameters,
        "inputs": {
            "stimulus_run": metadata["stimulus_run"],
            "chaser_index": chaser_index,
        },
    }
    provenance_record = {k: v for k, v in provenance_record.items() if v is not None}

    refined_group.attrs["source_stimulus_run"] = metadata["stimulus_run"]
    refined_group.attrs["chaser_index"] = chaser_index
    refined_group.attrs["texture_to_camera_scale"] = scale
    refined_group.attrs["coordinate_space"] = output_descriptor.space_id
    refined_group.attrs["positions_coordinate_descriptor_refs"] = [
        "filtered/positions_px",
        "interpolated/positions_px",
    ]
    if output_descriptor.legacy_space_label is not None:
        refined_group.attrs["legacy_space_label"] = output_descriptor.legacy_space_label
    refined_group.attrs["pixels_per_mm_projector"] = pixels_per_mm_projector
    refined_group.attrs["refinement_timestamp"] = created_timestamp
    refined_group.attrs["processing_time_seconds"] = float(duration)
    refined_group.attrs["operations"] = ["smooth", "outlier_removal", "interpolate"]
    refined_group.attrs["parameters"] = parameters
    refined_group.attrs["coverage_stats"] = coverage_stats
    refined_group.attrs["outlier_stats"] = outlier_stats
    refined_group.attrs["interpolation_stats"] = interp_stats
    refined_group.attrs["provenance"] = provenance_record

    console.print(f"[green]✓[/green] Refined run saved: {refined_group.path}")
    console.print(f"[green]✓[/green] Processing completed in {duration:.2f} seconds")

    return run_name


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Refine online target positions from stimulus runs"
    )
    parser.add_argument("zarr_path", help="Path to Palette zarr archive")
    parser.add_argument("--stimulus-run", help="Stimulus run name (defaults to latest)")
    parser.add_argument(
        "--chaser-index",
        type=int,
        default=0,
        help="Chaser index to process (default: 0)",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=11,
        help="Savitzky-Golay filter window length (must be odd, default: 11)",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Polynomial order for Savitzky-Golay filter (default: 3)",
    )
    parser.add_argument(
        "--displacement-threshold",
        type=float,
        default=100.0,
        help="Maximum reasonable displacement in native coordinate units (default: 100)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=20,
        help="Maximum gap size to interpolate in frames (default: 20)",
    )

    args = parser.parse_args(argv)

    console = Console()

    try:
        refine_online_positions(
            zarr_path=args.zarr_path,
            stimulus_run=args.stimulus_run,
            chaser_index=args.chaser_index,
            window_length=args.window_length,
            polyorder=args.polyorder,
            displacement_threshold=args.displacement_threshold,
            max_gap=args.max_gap,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
