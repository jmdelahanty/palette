"""Compute near-field occupancy metrics for one or more chasers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_distance_runs import _bytes_array, _write_array
from fisheye.analysis.chaser_behavior import (
    BEHAVIOR_CLASS_LABELS,
    canonical_behavior_label,
)
from fisheye.analysis.chaser_quadrant_occupancy import (
    COMPONENT_PARENT_NAME as QUADRANT_OCCUPANCY_PARENT,
)
from fisheye.shared.arena_geometry import (
    ArenaGeometry,
    out_of_bounds_notes,
    resolve_arena_geometry as _resolve_shared_arena_geometry,
)
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.plot_artifacts import write_interactive_plot_spec_artifact, write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name
from fisheye.shared.system_metadata import get_git_info

SCHEMA_ID = "palette.chaser.near_field_occupancy.v1"
SCHEMA_VERSION = 1
METHOD = "chaser_near_field_occupancy"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "chaser_near_field_occupancy"
DEFAULT_COMPONENT_NAME = "chaser_relative_near_field_v1"
RADIAL_DENSITY_PNG_ARTIFACT_NAME = "chaser_near_field_occupancy_radial_density_png"
DISTANCE_CDF_PNG_ARTIFACT_NAME = "chaser_near_field_occupancy_distance_cdf_png"
SUMMARY_PNG_ARTIFACT_NAME = "chaser_near_field_occupancy_summary_png"
INTERACTIVE_ARTIFACT_NAME = "chaser_near_field_occupancy_interactive"
INTERACTIVE_RENDERER = "palette-chaser-near-field-occupancy-v1"
INTERACTIVE_SPEC_SCHEMA_ID = "palette.chaser.near_field_occupancy.interactive_spec.v1"
PHASE_LABELS = ("pre_static", "post_static")
DEFAULT_PERCENTILES = (5.0, 10.0)
DEFAULT_R_ZONE_MM = 5.0
DEFAULT_R_IN_MM = 5.0
DEFAULT_R_OUT_MM = 6.0
DEFAULT_CDF_THRESHOLDS_MM = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_PERIMETER_BAND_MM = 5.0
DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S = 1.0
DEFAULT_RECTANGLE_AREA_GRID_STEP_MM = 0.25


@dataclass(frozen=True)
class ChaserNearFieldIdentity:
    chaser_index: int
    behavior_class_id: int
    behavior_class: str
    raw_color_rgba: tuple[float, float, float, float]
    raw_color_hex: str


@dataclass(frozen=True)
class ChaserNearFieldPhase:
    phase_index: int
    phase_label: str
    effective_start_frame: int
    effective_end_frame: int
    total_frame_count: int


@dataclass(frozen=True)
class ChaserNearFieldOccupancyResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_quadrant_occupancy_component: str
    source_quadrant_occupancy_path: str
    source_stimulus_run: str | None
    source_stimulus_path: str | None
    source_stimulus_epoch_run: str | None
    source_stimulus_epoch_path: str | None
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    coordinate_frame: str
    coordinate_origin: str
    arena_width_px: float
    arena_height_px: float
    geometry_status: str
    arena_geometry_source: str | None
    arena_shape: str
    arena_center_x_px: float | None
    arena_center_y_px: float | None
    arena_radius_px: float | None
    r_zone_mm: float
    r_in_mm: float
    r_out_mm: float
    percentile_values: np.ndarray
    radial_bin_edges_mm: np.ndarray
    radial_bin_centers_mm: np.ndarray
    cdf_thresholds_mm: np.ndarray
    perimeter_band_mm: float
    immobility_speed_threshold_mm_s: float
    chasers: tuple[ChaserNearFieldIdentity, ...]
    phases: tuple[ChaserNearFieldPhase, ...]
    approach_percentile_mm: np.ndarray
    approach_percentile_cdf_fraction: np.ndarray
    chaser_x_px: np.ndarray
    chaser_y_px: np.ndarray
    chaser_x_mm: np.ndarray
    chaser_y_mm: np.ndarray
    chaser_distance_to_arena_center_mm: np.ndarray
    chaser_distance_to_wall_mm: np.ndarray
    chaser_displacement_from_pre_mm: np.ndarray
    near_zone_occupancy_fraction: np.ndarray
    near_zone_occupancy_fraction_of_epoch: np.ndarray
    near_zone_dwell_s: np.ndarray
    near_zone_density_per_mm2: np.ndarray
    near_zone_available_area_mm2: np.ndarray
    near_zone_entry_count: np.ndarray
    near_zone_entry_rate_per_min: np.ndarray
    near_zone_visit_median_dwell_s: np.ndarray
    near_zone_visit_total_dwell_s: np.ndarray
    valid_distance_count: np.ndarray
    missing_frame_count: np.ndarray
    tracking_dropout_fraction: np.ndarray
    radial_count: np.ndarray
    radial_fraction: np.ndarray
    radial_density_per_mm2: np.ndarray
    radial_available_area_mm2: np.ndarray
    radial_count_wall_excluded: np.ndarray
    radial_fraction_wall_excluded: np.ndarray
    radial_density_wall_excluded_per_mm2: np.ndarray
    radial_available_area_wall_excluded_mm2: np.ndarray
    radial_wall_excluded_valid_count: np.ndarray
    cdf_fraction: np.ndarray
    control_reference_labels: tuple[str, ...]
    control_reference_x_px: np.ndarray
    control_reference_y_px: np.ndarray
    control_reference_approach_percentile_mm: np.ndarray
    control_reference_radial_count: np.ndarray
    control_reference_radial_fraction: np.ndarray
    control_reference_radial_density_per_mm2: np.ndarray
    control_reference_radial_available_area_mm2: np.ndarray
    control_reference_cdf_fraction: np.ndarray
    thigmotaxis_fraction: np.ndarray
    thigmotaxis_dwell_s: np.ndarray
    mean_speed_mm_s: np.ndarray
    median_speed_mm_s: np.ndarray
    immobile_fraction: np.ndarray
    speed_sample_count: np.ndarray
    endpoint_status: str
    qc_warnings: tuple[str, ...]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _get_group_by_path(root: zarr.Group, path: str | None) -> zarr.Group | None:
    normalized = "/".join(part for part in str(path or "").strip("/").split("/") if part)
    if not normalized:
        return root
    current: Any = root
    for part in normalized.split("/"):
        try:
            current = current[part]
        except Exception:
            return None
        if not isinstance(current, zarr.Group):
            return None
    return current


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _optional_float(value: Any) -> float | None:
    out = _safe_float(value)
    return out if math.isfinite(out) else None


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row).strip() for row in values]
    return [decode_null_terminated_text(value).strip() for value in values.reshape(-1)]


def _resolve_chaser_distance_run(root: zarr.Group, run_name: str) -> tuple[zarr.Group, str, str]:
    parent = root.get("analysis/chaser_distance_runs")
    if parent is None:
        raise ValueError("Archive has no analysis/chaser_distance_runs group.")
    resolved = str(run_name).strip()
    if not resolved or resolved == "latest":
        resolved = resolve_authoritative_run_name(parent) or str(parent.attrs.get("latest") or "").strip()
    if not resolved or resolved not in parent:
        raise ValueError("No usable chaser-distance run found; pass --chaser-distance-run.")
    return parent[resolved], resolved, f"analysis/chaser_distance_runs/{resolved}"


def _resolve_quadrant_occupancy_component(
    run_group: zarr.Group,
    *,
    chaser_distance_run_path: str,
    component_name: str,
) -> tuple[zarr.Group, str, str]:
    parent = run_group.get(QUADRANT_OCCUPANCY_PARENT)
    if parent is None:
        raise ValueError(
            "Chaser near-field occupancy requires an existing chaser quadrant occupancy component under "
            f"{chaser_distance_run_path}/{QUADRANT_OCCUPANCY_PARENT}."
        )
    resolved = str(component_name).strip()
    if not resolved or resolved == "latest":
        resolved = resolve_authoritative_run_name(parent) or str(parent.attrs.get("latest") or "").strip()
    if not resolved or resolved not in parent:
        raise ValueError(
            "No usable chaser quadrant occupancy component found; "
            "pass --quadrant-occupancy-component."
        )
    component = parent[resolved]
    status = str(component.attrs.get("status") or "").strip().lower()
    if status and status != "computed":
        raise ValueError(
            f"chaser quadrant occupancy component is not computed: {resolved} status={status!r}."
        )
    return (
        component,
        resolved,
        f"{chaser_distance_run_path}/{QUADRANT_OCCUPANCY_PARENT}/{resolved}",
    )


def _read_chasers(component: zarr.Group) -> tuple[ChaserNearFieldIdentity, ...]:
    if "chasers" not in component:
        raise ValueError("chaser quadrant occupancy component lacks chasers group.")
    group = component["chasers"]
    required = (
        "chaser_index",
        "behavior_class_id",
        "raw_color_rgba",
        "raw_color_hex_bytes",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(
            f"chaser quadrant occupancy chasers group missing array(s): {missing}"
        )
    chaser_index = np.asarray(group["chaser_index"][:], dtype=np.int64).reshape(-1)
    behavior_class_id = np.asarray(
        group["behavior_class_id"][:], dtype=np.int16
    ).reshape(-1)
    if "behavior_class_label_bytes" not in group:
        raise ValueError(
            "chaser quadrant occupancy chasers group lacks authoritative "
            "behavior_class_label_bytes"
        )
    role_labels = _decode_text_column(
        np.asarray(group["behavior_class_label_bytes"][:])
    )
    color_rgba = np.asarray(group["raw_color_rgba"][:], dtype=np.float64)
    color_hex = _decode_text_column(np.asarray(group["raw_color_hex_bytes"][:]))
    n = min(
        chaser_index.shape[0],
        behavior_class_id.shape[0],
        color_rgba.shape[0],
        len(color_hex),
    )
    chasers: list[ChaserNearFieldIdentity] = []
    for idx in range(n):
        role = (
            canonical_behavior_label(role_labels[idx]) if idx < len(role_labels) else ""
        )
        expected_role = BEHAVIOR_CLASS_LABELS.get(int(behavior_class_id[idx]))
        if expected_role is None or role != expected_role:
            raise ValueError(
                "chaser behavior_class_id and behavior_class_label_bytes disagree: "
                f"index={int(chaser_index[idx])}, id={int(behavior_class_id[idx])}, "
                f"label={role!r}"
            )
        rgba_values = color_rgba[idx].reshape(-1)
        rgba = tuple(
            float(rgba_values[i]) if i < rgba_values.shape[0] else 1.0 for i in range(4)
        )
        chasers.append(
            ChaserNearFieldIdentity(
                chaser_index=int(chaser_index[idx]),
                behavior_class_id=int(behavior_class_id[idx]),
                behavior_class=role,
                raw_color_rgba=rgba,  # type: ignore[arg-type]
                raw_color_hex=str(color_hex[idx]).strip(),
            )
        )
    if not chasers:
        raise ValueError("chaser near-field occupancy requires at least one chaser")
    return tuple(chasers)


def _read_phases(
    component: zarr.Group, *, total_frames: int
) -> tuple[ChaserNearFieldPhase, ...]:
    if "phases" not in component:
        raise ValueError("chaser quadrant occupancy component lacks phases group.")
    group = component["phases"]
    required = ("phase_index", "phase_label_bytes", "effective_start_frame", "effective_end_frame")
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(
            f"chaser quadrant occupancy phases group missing array(s): {missing}"
        )
    indices = np.asarray(group["phase_index"][:], dtype=np.int64).reshape(-1)
    labels = _decode_text_column(np.asarray(group["phase_label_bytes"][:]))
    starts = np.asarray(group["effective_start_frame"][:], dtype=np.int64).reshape(-1)
    ends = np.asarray(group["effective_end_frame"][:], dtype=np.int64).reshape(-1)
    n = min(indices.shape[0], len(labels), starts.shape[0], ends.shape[0])
    by_label: dict[str, ChaserNearFieldPhase] = {}
    for idx in range(n):
        label = str(labels[idx]).strip()
        start = max(0, int(starts[idx]))
        end = min(int(total_frames) - 1, int(ends[idx]))
        count = max(0, end - start + 1)
        by_label[label] = ChaserNearFieldPhase(
            phase_index=int(indices[idx]),
            phase_label=label,
            effective_start_frame=start,
            effective_end_frame=end,
            total_frame_count=count,
        )
    missing_labels = [label for label in PHASE_LABELS if label not in by_label]
    if missing_labels:
        raise ValueError(
            f"chaser quadrant occupancy component missing phase(s): {missing_labels}"
        )
    return tuple(by_label[label] for label in PHASE_LABELS)


def _phase_slice(phase: ChaserNearFieldPhase, total_frames: int) -> slice:
    start = max(0, int(phase.effective_start_frame))
    end = min(int(total_frames) - 1, int(phase.effective_end_frame))
    if end < start:
        return slice(0, 0)
    return slice(start, end + 1)


def _format_percentile_key(value: float) -> str:
    number = float(value)
    if math.isclose(number, round(number)):
        return f"p{int(round(number)):02d}"
    text = f"{number:g}".replace(".", "_").replace("-", "m")
    return f"p{text}"


def _resolve_arena_geometry(
    root: zarr.Group,
    run_group: zarr.Group,
    quadrant_component: zarr.Group,
    *,
    pixels_per_mm: float,
) -> tuple[ArenaGeometry, list[str]]:
    """Prefer the fitted dish mask over the projector's nominal experimental_area circle.

    They are not the same circle: the nominal one sits ~3 mm off-centre and ~2.4 mm small, which
    put 56% of a post-period's frames "outside the arena" and silently inverted the thigmotaxis
    metric. See fisheye.shared.arena_geometry.
    """

    width_px = _safe_float(quadrant_component.attrs.get("quadrant_width_px"))
    height_px = _safe_float(quadrant_component.attrs.get("quadrant_height_px"))
    if (
        not math.isfinite(width_px)
        or width_px <= 0
        or not math.isfinite(height_px)
        or height_px <= 0
    ):
        raise ValueError(
            "chaser quadrant occupancy component lacks positive arena width/height attrs."
        )

    geometry, notes = _resolve_shared_arena_geometry(root, run_group, pixels_per_mm=float(pixels_per_mm))
    # The quadrant bounds are the component's own view of the arena box; keep them so the
    # rectangular fallback path is unchanged.
    return replace(geometry, width_px=float(width_px), height_px=float(height_px)), notes


def _normalize_float_array(values: Sequence[float] | np.ndarray, *, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if positive:
        array = array[array > 0]
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one finite value.")
    return np.unique(array.astype(np.float32))


def _default_radial_bin_edges(distance_mm: np.ndarray) -> np.ndarray:
    values = np.asarray(distance_mm, dtype=np.float64)
    finite = values[np.isfinite(values) & (values >= 0)]
    max_distance = float(np.nanmax(finite)) if finite.size else 20.0
    max_edge = max(20.0, float(math.ceil(max_distance / 5.0) * 5.0))
    near = np.arange(0.0, min(20.0, max_edge) + 2.0, 2.0, dtype=np.float64)
    if max_edge > 20.0:
        far = np.arange(25.0, max_edge + 5.0, 5.0, dtype=np.float64)
        edges = np.concatenate([near, far])
    else:
        edges = near
    if edges[-1] < max_distance:
        edges = np.append(edges, max_distance)
    return np.unique(edges.astype(np.float32))


def compute_hysteresis_visits(
    distance_mm: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    r_in_mm: float,
    r_out_mm: float,
) -> tuple[int, float, float, float]:
    """Count near-zone entries with enter/exit hysteresis."""

    if not float(r_out_mm) > float(r_in_mm):
        raise ValueError("r_out_mm must be greater than r_in_mm.")
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0
    distances = np.asarray(distance_mm, dtype=np.float64).reshape(-1)
    valid_arr = np.asarray(valid, dtype=bool).reshape(-1)
    n = min(distances.shape[0], valid_arr.shape[0])
    in_visit = False
    frames_in_visit = 0
    dwell_frames: list[int] = []
    for idx in range(n):
        is_valid = bool(valid_arr[idx]) and math.isfinite(float(distances[idx]))
        if not is_valid:
            if in_visit:
                dwell_frames.append(frames_in_visit)
                in_visit = False
                frames_in_visit = 0
            continue
        distance = float(distances[idx])
        if not in_visit:
            if distance < float(r_in_mm):
                in_visit = True
                frames_in_visit = 1
            continue
        if distance > float(r_out_mm):
            dwell_frames.append(frames_in_visit)
            in_visit = False
            frames_in_visit = 0
        else:
            frames_in_visit += 1
    if in_visit:
        dwell_frames.append(frames_in_visit)

    count = len(dwell_frames)
    dwell_s = np.asarray(dwell_frames, dtype=np.float64) / safe_fps if dwell_frames else np.asarray([], dtype=np.float64)
    total_duration_s = float(n) / safe_fps if n > 0 else math.nan
    rate = float(count) / (total_duration_s / 60.0) if count > 0 and math.isfinite(total_duration_s) and total_duration_s > 0 else 0.0
    median_dwell = float(np.nanmedian(dwell_s)) if dwell_s.size else math.nan
    total_dwell = float(np.nansum(dwell_s)) if dwell_s.size else 0.0
    return count, rate, median_dwell, total_dwell


def _rectangle_annulus_area_mm2(
    *,
    chaser_x_px: float,
    chaser_y_px: float,
    width_px: float,
    height_px: float,
    pixels_per_mm: float,
    bin_edges_mm: np.ndarray,
    grid_step_mm: float = DEFAULT_RECTANGLE_AREA_GRID_STEP_MM,
    exclude_perimeter_band_mm: float = 0.0,
) -> np.ndarray:
    edges = np.asarray(bin_edges_mm, dtype=np.float64).reshape(-1)
    if edges.shape[0] < 2:
        raise ValueError("bin_edges_mm must contain at least two values.")
    if not all(
        math.isfinite(v)
        for v in (chaser_x_px, chaser_y_px, width_px, height_px, pixels_per_mm)
    ):
        return np.full(edges.shape[0] - 1, np.nan, dtype=np.float32)
    if width_px <= 0 or height_px <= 0 or pixels_per_mm <= 0:
        return np.full(edges.shape[0] - 1, np.nan, dtype=np.float32)
    step = max(0.05, float(grid_step_mm))
    width_mm = float(width_px) / float(pixels_per_mm)
    height_mm = float(height_px) / float(pixels_per_mm)
    center_x_mm = float(chaser_x_px) / float(pixels_per_mm)
    center_y_mm = float(chaser_y_px) / float(pixels_per_mm)
    band_mm = max(0.0, float(exclude_perimeter_band_mm))
    x_min = min(max(0.0, band_mm), width_mm)
    x_max = max(x_min, width_mm - band_mm)
    y_min = min(max(0.0, band_mm), height_mm)
    y_max = max(y_min, height_mm - band_mm)
    x = np.arange(x_min + step / 2.0, x_max, step, dtype=np.float64)
    y = np.arange(y_min + step / 2.0, y_max, step, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return np.zeros(edges.shape[0] - 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    distance = np.sqrt((xx - center_x_mm) ** 2 + (yy - center_y_mm) ** 2)
    counts, _ = np.histogram(distance.reshape(-1), bins=edges)
    return (counts.astype(np.float64) * (step * step)).astype(np.float32)


def _available_annulus_area_mm2(
    *,
    geometry: ArenaGeometry,
    chaser_x_px: float,
    chaser_y_px: float,
    pixels_per_mm: float,
    bin_edges_mm: np.ndarray,
    grid_step_mm: float = DEFAULT_RECTANGLE_AREA_GRID_STEP_MM,
    exclude_perimeter_band_mm: float = 0.0,
) -> np.ndarray:
    if geometry.shape != "circle":
        return _rectangle_annulus_area_mm2(
            chaser_x_px=chaser_x_px,
            chaser_y_px=chaser_y_px,
            width_px=geometry.width_px,
            height_px=geometry.height_px,
            pixels_per_mm=pixels_per_mm,
            bin_edges_mm=bin_edges_mm,
            grid_step_mm=grid_step_mm,
            exclude_perimeter_band_mm=exclude_perimeter_band_mm,
        )
    edges = np.asarray(bin_edges_mm, dtype=np.float64).reshape(-1)
    if edges.shape[0] < 2:
        raise ValueError("bin_edges_mm must contain at least two values.")
    if (
        geometry.center_x_px is None
        or geometry.center_y_px is None
        or geometry.radius_px is None
        or not all(math.isfinite(v) for v in (chaser_x_px, chaser_y_px, pixels_per_mm))
        or pixels_per_mm <= 0
    ):
        return np.full(edges.shape[0] - 1, np.nan, dtype=np.float32)
    step = max(0.05, float(grid_step_mm))
    center_x_mm = float(geometry.center_x_px) / float(pixels_per_mm)
    center_y_mm = float(geometry.center_y_px) / float(pixels_per_mm)
    radius_mm = float(geometry.radius_px) / float(pixels_per_mm)
    chaser_x_mm = float(chaser_x_px) / float(pixels_per_mm)
    chaser_y_mm = float(chaser_y_px) / float(pixels_per_mm)
    min_x = center_x_mm - radius_mm
    max_x = center_x_mm + radius_mm
    min_y = center_y_mm - radius_mm
    max_y = center_y_mm + radius_mm
    x = np.arange(min_x + step / 2.0, max_x, step, dtype=np.float64)
    y = np.arange(min_y + step / 2.0, max_y, step, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return np.zeros(edges.shape[0] - 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    in_circle = (xx - center_x_mm) ** 2 + (yy - center_y_mm) ** 2 <= radius_mm**2
    band_mm = max(0.0, float(exclude_perimeter_band_mm))
    if band_mm > 0.0:
        core_radius_mm = max(0.0, radius_mm - band_mm)
        in_circle &= (xx - center_x_mm) ** 2 + (
            yy - center_y_mm
        ) ** 2 <= core_radius_mm**2
    chaser_distance = np.sqrt((xx - chaser_x_mm) ** 2 + (yy - chaser_y_mm) ** 2)
    counts, _ = np.histogram(chaser_distance[in_circle], bins=edges)
    return (counts.astype(np.float64) * (step * step)).astype(np.float32)


def _in_bounds_and_wall_mask(
    fish_xy: np.ndarray,
    fish_valid: np.ndarray,
    *,
    geometry: ArenaGeometry,
    perimeter_band_mm: float,
    pixels_per_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(fish_xy, dtype=np.float64)
    valid = np.asarray(fish_valid, dtype=bool).reshape(-1) & np.isfinite(xy).all(axis=1)
    in_bounds = np.zeros(valid.shape, dtype=bool)
    wall = np.zeros(valid.shape, dtype=bool)
    if not np.any(valid):
        return valid, in_bounds, wall
    band_px = max(0.0, float(perimeter_band_mm)) * float(pixels_per_mm)
    values = xy[valid]
    if geometry.shape == "circle" and geometry.center_x_px is not None and geometry.center_y_px is not None and geometry.radius_px is not None:
        radial = np.sqrt((values[:, 0] - float(geometry.center_x_px)) ** 2 + (values[:, 1] - float(geometry.center_y_px)) ** 2)
        in_bounds_valid = radial <= float(geometry.radius_px)
        wall_valid = radial >= max(0.0, float(geometry.radius_px) - band_px)
    else:
        in_bounds_valid = (
            (values[:, 0] >= 0.0)
            & (values[:, 1] >= 0.0)
            & (values[:, 0] < float(geometry.width_px))
            & (values[:, 1] < float(geometry.height_px))
        )
        wall_valid = (
            (values[:, 0] <= band_px)
            | (values[:, 0] >= float(geometry.width_px) - band_px)
            | (values[:, 1] <= band_px)
            | (values[:, 1] >= float(geometry.height_px) - band_px)
        )
    in_bounds[valid] = in_bounds_valid
    wall[valid] = wall_valid
    return valid, in_bounds, wall


def _thigmotaxis_for_phase(
    fish_xy: np.ndarray,
    fish_valid: np.ndarray,
    *,
    fps: float,
    geometry: ArenaGeometry,
    perimeter_band_mm: float,
    pixels_per_mm: float,
) -> tuple[float, float]:
    valid, in_bounds, wall = _in_bounds_and_wall_mask(
        fish_xy,
        fish_valid,
        geometry=geometry,
        perimeter_band_mm=perimeter_band_mm,
        pixels_per_mm=pixels_per_mm,
    )
    if not np.any(valid):
        return math.nan, math.nan
    # `wall` is unbounded above (radial >= radius - band), so a frame beyond the arena radius
    # still counts as wall -- which it physically is. The previous `in_bounds & wall` dropped
    # exactly those frames from the numerator while keeping them in the denominator, so the
    # harder the fish hugged the wall the MORE its wall-hugging was discarded. With a radius
    # 2.4 mm too small that turned a 0.37 -> 0.87 thigmotaxis increase into 0.354 -> 0.353.
    count = int(np.count_nonzero(wall))
    total = int(np.count_nonzero(valid))
    fraction = float(count) / float(total) if total > 0 else math.nan
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0
    return fraction, float(count) / safe_fps


def _speed_state_for_phase(
    fish_xy: np.ndarray,
    fish_valid: np.ndarray,
    *,
    fps: float,
    pixels_per_mm: float,
    immobility_speed_threshold_mm_s: float,
) -> tuple[float, float, float, int]:
    xy = np.asarray(fish_xy, dtype=np.float64)
    valid = np.asarray(fish_valid, dtype=bool).reshape(-1) & np.isfinite(xy).all(axis=1)
    if xy.shape[0] < 2 or valid.shape[0] < 2:
        return math.nan, math.nan, math.nan, 0
    pair_valid = valid[:-1] & valid[1:]
    if not np.any(pair_valid):
        return math.nan, math.nan, math.nan, 0
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0
    safe_ppm = float(pixels_per_mm) if math.isfinite(float(pixels_per_mm)) and float(pixels_per_mm) > 0 else math.nan
    if not math.isfinite(safe_ppm):
        return math.nan, math.nan, math.nan, 0
    delta_px = np.diff(xy, axis=0)
    speed = np.sqrt(np.sum(delta_px**2, axis=1)) / safe_ppm * safe_fps
    values = speed[pair_valid]
    if values.size == 0:
        return math.nan, math.nan, math.nan, 0
    threshold = max(0.0, float(immobility_speed_threshold_mm_s))
    return (
        float(np.nanmean(values)),
        float(np.nanmedian(values)),
        float(np.count_nonzero(values <= threshold) / values.size),
        int(values.size),
    )


def _chaser_distance_to_center_and_wall_mm(
    *,
    geometry: ArenaGeometry,
    chaser_x_px: float,
    chaser_y_px: float,
    pixels_per_mm: float,
) -> tuple[float, float]:
    if (
        not all(math.isfinite(v) for v in (chaser_x_px, chaser_y_px, pixels_per_mm))
        or pixels_per_mm <= 0
    ):
        return math.nan, math.nan
    if (
        geometry.shape == "circle"
        and geometry.center_x_px is not None
        and geometry.center_y_px is not None
        and geometry.radius_px is not None
    ):
        center_distance_px = math.hypot(
            float(chaser_x_px) - float(geometry.center_x_px),
            float(chaser_y_px) - float(geometry.center_y_px),
        )
        return center_distance_px / float(pixels_per_mm), (
            float(geometry.radius_px) - center_distance_px
        ) / float(pixels_per_mm)
    center_x_px = float(geometry.width_px) / 2.0
    center_y_px = float(geometry.height_px) / 2.0
    center_distance_px = math.hypot(
        float(chaser_x_px) - center_x_px, float(chaser_y_px) - center_y_px
    )
    wall_distance_px = min(
        float(chaser_x_px),
        float(chaser_y_px),
        float(geometry.width_px) - float(chaser_x_px),
        float(geometry.height_px) - float(chaser_y_px),
    )
    return center_distance_px / float(pixels_per_mm), wall_distance_px / float(pixels_per_mm)


def _control_reference_points(geometry: ArenaGeometry) -> tuple[tuple[str, float, float], ...]:
    if geometry.shape == "circle" and geometry.center_x_px is not None and geometry.center_y_px is not None:
        return (("dish_center", float(geometry.center_x_px), float(geometry.center_y_px)),)
    return (("dish_center", float(geometry.width_px) / 2.0, float(geometry.height_px) / 2.0),)


def _compute_near_field_arrays(
    *,
    chasers: Sequence[ChaserNearFieldIdentity],
    phases: Sequence[ChaserNearFieldPhase],
    chaser_indices: np.ndarray,
    fish_xy: np.ndarray,
    chaser_valid: np.ndarray,
    fish_valid: np.ndarray,
    distance_mm: np.ndarray,
    fps: float,
    pixels_per_mm: float,
    geometry: ArenaGeometry,
    chaser_phase_x_px: np.ndarray,
    chaser_phase_y_px: np.ndarray,
    percentile_values: np.ndarray,
    radial_bin_edges_mm: np.ndarray,
    cdf_thresholds_mm: np.ndarray,
    r_zone_mm: float,
    r_in_mm: float,
    r_out_mm: float,
    perimeter_band_mm: float,
    immobility_speed_threshold_mm_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[str, ...]]:
    n_phases = len(phases)
    n_chasers = len(chasers)
    n_percentiles = int(percentile_values.shape[0])
    n_radial_bins = int(radial_bin_edges_mm.shape[0] - 1)
    n_thresholds = int(cdf_thresholds_mm.shape[0])
    total_frames = int(fish_xy.shape[0])
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0
    by_chaser_index = {
        int(value): idx
        for idx, value in enumerate(np.asarray(chaser_indices).reshape(-1))
    }
    chaser_columns: list[int] = []
    for obj in chasers:
        if obj.chaser_index not in by_chaser_index:
            raise ValueError(
                f"Chaser index {obj.chaser_index} not present in chaser indices {list(by_chaser_index)}."
            )
        chaser_columns.append(int(by_chaser_index[obj.chaser_index]))

    shape = (n_phases, n_chasers)
    approach = np.full((n_phases, n_chasers, n_percentiles), np.nan, dtype=np.float32)
    approach_cdf_fraction = np.full(
        (n_phases, n_chasers, n_percentiles), np.nan, dtype=np.float32
    )
    chaser_x = np.asarray(chaser_phase_x_px[:n_phases, :n_chasers], dtype=np.float32)
    chaser_y = np.asarray(chaser_phase_y_px[:n_phases, :n_chasers], dtype=np.float32)
    chaser_x_mm = chaser_x / float(pixels_per_mm)
    chaser_y_mm = chaser_y / float(pixels_per_mm)
    chaser_center_distance = np.full(shape, np.nan, dtype=np.float32)
    chaser_wall_distance = np.full(shape, np.nan, dtype=np.float32)
    chaser_displacement_from_pre = np.full(shape, np.nan, dtype=np.float32)
    near_fraction = np.full(shape, np.nan, dtype=np.float32)
    near_fraction_epoch = np.full(shape, np.nan, dtype=np.float32)
    near_dwell_s = np.full(shape, np.nan, dtype=np.float32)
    near_density = np.full(shape, np.nan, dtype=np.float32)
    near_area = np.full(shape, np.nan, dtype=np.float32)
    entry_count = np.zeros(shape, dtype=np.int64)
    entry_rate = np.full(shape, np.nan, dtype=np.float32)
    visit_median_dwell = np.full(shape, np.nan, dtype=np.float32)
    visit_total_dwell = np.full(shape, np.nan, dtype=np.float32)
    valid_distance_count = np.zeros(shape, dtype=np.int64)
    missing_frame_count = np.zeros(shape, dtype=np.int64)
    dropout = np.full(shape, np.nan, dtype=np.float32)
    radial_count = np.zeros((n_phases, n_chasers, n_radial_bins), dtype=np.int64)
    radial_fraction = np.full(
        (n_phases, n_chasers, n_radial_bins), np.nan, dtype=np.float32
    )
    radial_area = np.full(
        (n_phases, n_chasers, n_radial_bins), np.nan, dtype=np.float32
    )
    radial_density = np.full(
        (n_phases, n_chasers, n_radial_bins), np.nan, dtype=np.float32
    )
    radial_count_wall_excluded = np.zeros(
        (n_phases, n_chasers, n_radial_bins), dtype=np.int64
    )
    radial_fraction_wall_excluded = np.full(
        (n_phases, n_chasers, n_radial_bins), np.nan, dtype=np.float32
    )
    radial_area_wall_excluded = np.full(
        (n_phases, n_chasers, n_radial_bins), np.nan, dtype=np.float32
    )
    radial_density_wall_excluded = np.full(
        (n_phases, n_chasers, n_radial_bins), np.nan, dtype=np.float32
    )
    radial_wall_excluded_valid_count = np.zeros(shape, dtype=np.int64)
    cdf_fraction = np.full(
        (n_phases, n_chasers, n_thresholds), np.nan, dtype=np.float32
    )
    reference_points = _control_reference_points(geometry)
    n_references = len(reference_points)
    reference_x = np.asarray([point[1] for point in reference_points], dtype=np.float32)
    reference_y = np.asarray([point[2] for point in reference_points], dtype=np.float32)
    reference_approach = np.full((n_phases, n_references, n_percentiles), np.nan, dtype=np.float32)
    reference_radial_count = np.zeros((n_phases, n_references, n_radial_bins), dtype=np.int64)
    reference_radial_fraction = np.full((n_phases, n_references, n_radial_bins), np.nan, dtype=np.float32)
    reference_radial_area = np.full((n_phases, n_references, n_radial_bins), np.nan, dtype=np.float32)
    reference_radial_density = np.full((n_phases, n_references, n_radial_bins), np.nan, dtype=np.float32)
    reference_cdf_fraction = np.full((n_phases, n_references, n_thresholds), np.nan, dtype=np.float32)
    thig_fraction = np.full(n_phases, np.nan, dtype=np.float32)
    thig_dwell = np.full(n_phases, np.nan, dtype=np.float32)
    mean_speed = np.full(n_phases, np.nan, dtype=np.float32)
    median_speed = np.full(n_phases, np.nan, dtype=np.float32)
    immobile_fraction = np.full(n_phases, np.nan, dtype=np.float32)
    speed_sample_count = np.zeros(n_phases, dtype=np.int64)
    warnings: list[str] = []
    # Warn on a *shape* fallback (a rectangle approximation), not on the status string --
    # "dish_mask" is the good case and must not read as a problem.
    if geometry.shape != "circle":
        warnings.append(f"arena_geometry_not_circular:{geometry.status}")

    for p_idx, phase in enumerate(phases):
        slc = _phase_slice(phase, total_frames)
        phase_len = max(0, int(slc.stop - slc.start))
        phase_fish_xy = np.asarray(fish_xy[slc], dtype=np.float64)
        phase_fish_valid = np.asarray(fish_valid[slc], dtype=bool) & np.isfinite(phase_fish_xy).all(axis=1)
        thig_fraction[p_idx], thig_dwell[p_idx] = _thigmotaxis_for_phase(
            phase_fish_xy,
            phase_fish_valid,
            fps=safe_fps,
            geometry=geometry,
            perimeter_band_mm=float(perimeter_band_mm),
            pixels_per_mm=float(pixels_per_mm),
        )
        speed_state = _speed_state_for_phase(
            phase_fish_xy,
            phase_fish_valid,
            fps=safe_fps,
            pixels_per_mm=float(pixels_per_mm),
            immobility_speed_threshold_mm_s=float(immobility_speed_threshold_mm_s),
        )
        mean_speed[p_idx] = float(speed_state[0]) if math.isfinite(float(speed_state[0])) else np.nan
        median_speed[p_idx] = float(speed_state[1]) if math.isfinite(float(speed_state[1])) else np.nan
        immobile_fraction[p_idx] = float(speed_state[2]) if math.isfinite(float(speed_state[2])) else np.nan
        speed_sample_count[p_idx] = int(speed_state[3])
        _phase_valid, phase_in_bounds, phase_wall = _in_bounds_and_wall_mask(
            phase_fish_xy,
            phase_fish_valid,
            geometry=geometry,
            perimeter_band_mm=float(perimeter_band_mm),
            pixels_per_mm=float(pixels_per_mm),
        )
        for reference_idx, (_label, ref_x, ref_y) in enumerate(reference_points):
            ref_distance = np.sqrt((phase_fish_xy[:, 0] - float(ref_x)) ** 2 + (phase_fish_xy[:, 1] - float(ref_y)) ** 2) / float(pixels_per_mm)
            ref_values = ref_distance[phase_fish_valid & np.isfinite(ref_distance)]
            if ref_values.size:
                reference_approach[p_idx, reference_idx, :] = np.percentile(ref_values, percentile_values).astype(np.float32)
                counts, _ = np.histogram(ref_values, bins=radial_bin_edges_mm.astype(np.float64))
                reference_radial_count[p_idx, reference_idx, :] = counts.astype(np.int64)
                reference_radial_fraction[p_idx, reference_idx, :] = (counts.astype(np.float64) / float(ref_values.shape[0])).astype(np.float32)
                for t_idx, threshold in enumerate(cdf_thresholds_mm):
                    reference_cdf_fraction[p_idx, reference_idx, t_idx] = float(
                        np.count_nonzero(ref_values <= float(threshold))
                    ) / float(ref_values.shape[0])
            reference_radial_area[p_idx, reference_idx, :] = (
                _available_annulus_area_mm2(
                    chaser_x_px=float(ref_x),
                    chaser_y_px=float(ref_y),
                    geometry=geometry,
                    pixels_per_mm=float(pixels_per_mm),
                    bin_edges_mm=radial_bin_edges_mm,
                )
            )
            finite_reference_area = np.isfinite(reference_radial_area[p_idx, reference_idx, :]) & (reference_radial_area[p_idx, reference_idx, :] > 0)
            reference_radial_density[p_idx, reference_idx, finite_reference_area] = (
                reference_radial_fraction[p_idx, reference_idx, finite_reference_area]
                / reference_radial_area[p_idx, reference_idx, finite_reference_area]
            )
        for o_idx, col_idx in enumerate(chaser_columns):
            center_distance, wall_distance = _chaser_distance_to_center_and_wall_mm(
                geometry=geometry,
                chaser_x_px=float(chaser_x[p_idx, o_idx]),
                chaser_y_px=float(chaser_y[p_idx, o_idx]),
                pixels_per_mm=float(pixels_per_mm),
            )
            chaser_center_distance[p_idx, o_idx] = (
                float(center_distance) if math.isfinite(center_distance) else np.nan
            )
            chaser_wall_distance[p_idx, o_idx] = (
                float(wall_distance) if math.isfinite(wall_distance) else np.nan
            )
            if p_idx == 0:
                chaser_displacement_from_pre[p_idx, o_idx] = 0.0
            else:
                chaser_displacement_from_pre[p_idx, o_idx] = math.hypot(
                    float(chaser_x[p_idx, o_idx]) - float(chaser_x[0, o_idx]),
                    float(chaser_y[p_idx, o_idx]) - float(chaser_y[0, o_idx]),
                ) / float(pixels_per_mm)
            missing_frame_count[p_idx, o_idx] = int(
                max(0, phase_len - np.count_nonzero(phase_fish_valid))
            )
            dropout[p_idx, o_idx] = (
                float(missing_frame_count[p_idx, o_idx]) / float(phase_len)
                if phase_len > 0
                else math.nan
            )
            phase_chaser_valid = np.asarray(chaser_valid[slc, col_idx], dtype=bool)
            phase_distance = np.asarray(distance_mm[slc, col_idx], dtype=np.float64)
            distance_valid = phase_fish_valid & phase_chaser_valid & np.isfinite(phase_distance)
            values = phase_distance[distance_valid]
            valid_distance_count[p_idx, o_idx] = int(values.shape[0])
            if values.size:
                approach[p_idx, o_idx, :] = np.percentile(values, percentile_values).astype(np.float32)
                for percentile_idx, approach_value in enumerate(approach[p_idx, o_idx, :]):
                    if math.isfinite(float(approach_value)):
                        approach_cdf_fraction[p_idx, o_idx, percentile_idx] = float(
                            np.count_nonzero(values <= float(approach_value)) / values.shape[0]
                        )
                near_count = int(np.count_nonzero(values <= float(r_zone_mm)))
                near_fraction[p_idx, o_idx] = float(near_count) / float(values.shape[0])
                near_fraction_epoch[p_idx, o_idx] = float(near_count) / float(phase_len) if phase_len > 0 else math.nan
                near_dwell_s[p_idx, o_idx] = float(near_count) / safe_fps
                counts, _ = np.histogram(values, bins=radial_bin_edges_mm.astype(np.float64))
                radial_count[p_idx, o_idx, :] = counts.astype(np.int64)
                radial_fraction[p_idx, o_idx, :] = (counts.astype(np.float64) / float(values.shape[0])).astype(np.float32)
                for t_idx, threshold in enumerate(cdf_thresholds_mm):
                    cdf_fraction[p_idx, o_idx, t_idx] = float(np.count_nonzero(values <= float(threshold))) / float(values.shape[0])

            wall_excluded_valid = distance_valid & phase_in_bounds & ~phase_wall
            wall_excluded_values = phase_distance[wall_excluded_valid]
            radial_wall_excluded_valid_count[p_idx, o_idx] = int(wall_excluded_values.shape[0])
            if wall_excluded_values.size:
                counts_wall_excluded, _ = np.histogram(wall_excluded_values, bins=radial_bin_edges_mm.astype(np.float64))
                radial_count_wall_excluded[p_idx, o_idx, :] = counts_wall_excluded.astype(np.int64)
                radial_fraction_wall_excluded[p_idx, o_idx, :] = (
                    counts_wall_excluded.astype(np.float64) / float(wall_excluded_values.shape[0])
                ).astype(np.float32)

            visit = compute_hysteresis_visits(
                phase_distance,
                distance_valid,
                fps=safe_fps,
                r_in_mm=float(r_in_mm),
                r_out_mm=float(r_out_mm),
            )
            entry_count[p_idx, o_idx] = int(visit[0])
            entry_rate[p_idx, o_idx] = float(visit[1])
            visit_median_dwell[p_idx, o_idx] = float(visit[2]) if math.isfinite(float(visit[2])) else np.nan
            visit_total_dwell[p_idx, o_idx] = float(visit[3])

            x = float(chaser_phase_x_px[p_idx, o_idx])
            y = float(chaser_phase_y_px[p_idx, o_idx])
            radial_area[p_idx, o_idx, :] = _available_annulus_area_mm2(
                chaser_x_px=x,
                chaser_y_px=y,
                geometry=geometry,
                pixels_per_mm=float(pixels_per_mm),
                bin_edges_mm=radial_bin_edges_mm,
            )
            radial_area_wall_excluded[p_idx, o_idx, :] = _available_annulus_area_mm2(
                chaser_x_px=x,
                chaser_y_px=y,
                geometry=geometry,
                pixels_per_mm=float(pixels_per_mm),
                bin_edges_mm=radial_bin_edges_mm,
                exclude_perimeter_band_mm=float(perimeter_band_mm),
            )
            zone_area = _available_annulus_area_mm2(
                chaser_x_px=x,
                chaser_y_px=y,
                geometry=geometry,
                pixels_per_mm=float(pixels_per_mm),
                bin_edges_mm=np.asarray([0.0, float(r_zone_mm)], dtype=np.float32),
            )
            near_area[p_idx, o_idx] = float(zone_area[0]) if zone_area.size else np.nan
            finite_area = np.isfinite(radial_area[p_idx, o_idx, :]) & (radial_area[p_idx, o_idx, :] > 0)
            radial_density[p_idx, o_idx, finite_area] = (
                radial_fraction[p_idx, o_idx, finite_area] / radial_area[p_idx, o_idx, finite_area]
            )
            finite_wall_excluded_area = np.isfinite(radial_area_wall_excluded[p_idx, o_idx, :]) & (
                radial_area_wall_excluded[p_idx, o_idx, :] > 0
            )
            radial_density_wall_excluded[p_idx, o_idx, finite_wall_excluded_area] = (
                radial_fraction_wall_excluded[p_idx, o_idx, finite_wall_excluded_area]
                / radial_area_wall_excluded[p_idx, o_idx, finite_wall_excluded_area]
            )
            if math.isfinite(float(near_area[p_idx, o_idx])) and float(near_area[p_idx, o_idx]) > 0:
                near_density[p_idx, o_idx] = near_fraction[p_idx, o_idx] / near_area[p_idx, o_idx]

    arrays = {
        "approach_percentile_mm": approach,
        "approach_percentile_cdf_fraction": approach_cdf_fraction,
        "chaser_x_px": chaser_x,
        "chaser_y_px": chaser_y,
        "chaser_x_mm": chaser_x_mm.astype(np.float32),
        "chaser_y_mm": chaser_y_mm.astype(np.float32),
        "chaser_distance_to_arena_center_mm": chaser_center_distance,
        "chaser_distance_to_wall_mm": chaser_wall_distance,
        "chaser_displacement_from_pre_mm": chaser_displacement_from_pre,
        "near_zone_occupancy_fraction": near_fraction,
        "near_zone_occupancy_fraction_of_epoch": near_fraction_epoch,
        "near_zone_dwell_s": near_dwell_s,
        "near_zone_density_per_mm2": near_density,
        "near_zone_available_area_mm2": near_area,
        "near_zone_entry_count": entry_count,
        "near_zone_entry_rate_per_min": entry_rate,
        "near_zone_visit_median_dwell_s": visit_median_dwell,
        "near_zone_visit_total_dwell_s": visit_total_dwell,
        "valid_distance_count": valid_distance_count,
        "missing_frame_count": missing_frame_count,
        "tracking_dropout_fraction": dropout,
        "radial_count": radial_count,
        "radial_fraction": radial_fraction,
        "radial_density_per_mm2": radial_density,
        "radial_available_area_mm2": radial_area,
        "radial_count_wall_excluded": radial_count_wall_excluded,
        "radial_fraction_wall_excluded": radial_fraction_wall_excluded,
        "radial_density_wall_excluded_per_mm2": radial_density_wall_excluded,
        "radial_available_area_wall_excluded_mm2": radial_area_wall_excluded,
        "radial_wall_excluded_valid_count": radial_wall_excluded_valid_count,
        "cdf_fraction": cdf_fraction,
        "control_reference_x_px": reference_x,
        "control_reference_y_px": reference_y,
        "control_reference_approach_percentile_mm": reference_approach,
        "control_reference_radial_count": reference_radial_count,
        "control_reference_radial_fraction": reference_radial_fraction,
        "control_reference_radial_density_per_mm2": reference_radial_density,
        "control_reference_radial_available_area_mm2": reference_radial_area,
        "control_reference_cdf_fraction": reference_cdf_fraction,
        "thigmotaxis_fraction": thig_fraction,
        "thigmotaxis_dwell_s": thig_dwell,
        "mean_speed_mm_s": mean_speed,
        "median_speed_mm_s": median_speed,
        "immobile_fraction": immobile_fraction,
        "speed_sample_count": speed_sample_count,
    }
    diagnostics = {
        "chaser_columns": chaser_columns,
        "control_reference_labels": [point[0] for point in reference_points],
        "geometry_status": geometry.status,
        "arena_geometry_source": geometry.source,
        "arena_shape": geometry.shape,
        "arena_center_x_px": geometry.center_x_px,
        "arena_center_y_px": geometry.center_y_px,
        "arena_radius_px": geometry.radius_px,
        "rectangle_area_grid_step_mm": DEFAULT_RECTANGLE_AREA_GRID_STEP_MM,
        "invalid_gap_policy": "close_active_visit_on_invalid_sample",
        "immobility_speed_threshold_mm_s": float(immobility_speed_threshold_mm_s),
        "wall_excluded_density_policy": "exclude fish samples and available area within perimeter_band_mm of arena wall",
    }
    return arrays, diagnostics, tuple(warnings)


def _value(array: np.ndarray, phase_index: int, chaser_index: int) -> float | None:
    value = float(array[int(phase_index), int(chaser_index)])
    return value if math.isfinite(value) else None


def _delta(pre: float | None, post: float | None) -> float | None:
    return None if pre is None or post is None else post - pre


def _build_summary(
    *,
    recording_id: str,
    chasers: Sequence[ChaserNearFieldIdentity],
    percentile_values: np.ndarray,
    approach_percentile_mm: np.ndarray,
    approach_percentile_cdf_fraction: np.ndarray,
    near_zone_occupancy_fraction: np.ndarray,
    near_zone_entry_rate_per_min: np.ndarray,
    tracking_dropout_fraction: np.ndarray,
    thigmotaxis_fraction: np.ndarray,
    mean_speed_mm_s: np.ndarray,
    immobile_fraction: np.ndarray,
    valid_distance_count: np.ndarray,
    phases: Sequence[ChaserNearFieldPhase],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "fish_id": "0",
        "recording_id": recording_id,
        "dpf": None,
        "chaser_count": len(chasers),
        "phase_labels": [phase.phase_label for phase in phases],
    }
    target_cdf = (np.asarray(percentile_values, dtype=np.float64).reshape(1, 1, -1) / 100.0).astype(np.float64)
    cdf_error = np.abs(np.asarray(approach_percentile_cdf_fraction, dtype=np.float64) - target_cdf)
    finite_cdf_error = cdf_error[np.isfinite(cdf_error)]
    summary["approach_percentile_cdf_max_abs_error"] = (
        float(np.nanmax(finite_cdf_error)) if finite_cdf_error.size else None
    )

    percentiles = np.asarray(percentile_values, dtype=np.float64).reshape(-1)
    per_chaser: list[dict[str, Any]] = []
    role_members: dict[str, list[int]] = {}
    for chaser_pos, chaser in enumerate(chasers):
        phase_values: list[dict[str, Any]] = []
        for phase_pos, phase in enumerate(phases):
            phase_values.append(
                {
                    "phase_label": phase.phase_label,
                    "valid_distance_count": int(
                        valid_distance_count[phase_pos, chaser_pos]
                    ),
                    "tracking_dropout_fraction": _value(
                        tracking_dropout_fraction, phase_pos, chaser_pos
                    ),
                    "near_zone_occupancy_fraction": _value(
                        near_zone_occupancy_fraction, phase_pos, chaser_pos
                    ),
                    "near_zone_entry_rate_per_min": _value(
                        near_zone_entry_rate_per_min, phase_pos, chaser_pos
                    ),
                    "approach_percentiles_mm": {
                        _format_percentile_key(float(percentile)): _value(
                            approach_percentile_mm[:, :, percentile_pos],
                            phase_pos,
                            chaser_pos,
                        )
                        for percentile_pos, percentile in enumerate(percentiles)
                    },
                }
            )
        per_chaser.append(
            {
                "chaser_index": int(chaser.chaser_index),
                "behavior_class_id": int(chaser.behavior_class_id),
                "behavior_class": chaser.behavior_class,
                "raw_color_hex": chaser.raw_color_hex,
                "phase_values": phase_values,
            }
        )
        role_members.setdefault(chaser.behavior_class, []).append(chaser_pos)

    def role_mean(
        source: np.ndarray, phase_pos: int, members: Sequence[int]
    ) -> float | None:
        values = np.asarray(
            [source[phase_pos, member] for member in members], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else None

    summary["per_chaser"] = per_chaser
    summary["per_role"] = [
        {
            "behavior_class": role,
            "chaser_count": len(members),
            "phase_values": [
                {
                    "phase_label": phase.phase_label,
                    "mean_near_zone_occupancy_fraction": role_mean(
                        near_zone_occupancy_fraction, phase_pos, members
                    ),
                    "mean_near_zone_entry_rate_per_min": role_mean(
                        near_zone_entry_rate_per_min, phase_pos, members
                    ),
                }
                for phase_pos, phase in enumerate(phases)
            ],
        }
        for role, members in sorted(role_members.items())
    ]
    summary["fish_phase_values"] = [
        {
            "phase_label": phase.phase_label,
            "thigmotaxis_fraction": (
                float(thigmotaxis_fraction[phase_pos])
                if math.isfinite(float(thigmotaxis_fraction[phase_pos]))
                else None
            ),
            "mean_speed_mm_s": (
                float(mean_speed_mm_s[phase_pos])
                if math.isfinite(float(mean_speed_mm_s[phase_pos]))
                else None
            ),
            "immobile_fraction": (
                float(immobile_fraction[phase_pos])
                if math.isfinite(float(immobile_fraction[phase_pos]))
                else None
            ),
        }
        for phase_pos, phase in enumerate(phases)
    ]
    summary["pairwise_role_contrast_policy"] = "not_computed_at_recording_level"
    return summary


def build_chaser_near_field_occupancy_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    quadrant_occupancy_component: str = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    r_zone_mm: float = DEFAULT_R_ZONE_MM,
    r_in_mm: float = DEFAULT_R_IN_MM,
    r_out_mm: float = DEFAULT_R_OUT_MM,
    percentile_values: Sequence[float] = DEFAULT_PERCENTILES,
    radial_bin_edges_mm: Sequence[float] | None = None,
    cdf_thresholds_mm: Sequence[float] = DEFAULT_CDF_THRESHOLDS_MM,
    perimeter_band_mm: float = DEFAULT_PERIMETER_BAND_MM,
    immobility_speed_threshold_mm_s: float = DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S,
) -> ChaserNearFieldOccupancyResult:
    if not float(r_out_mm) > float(r_in_mm):
        raise ValueError("r_out_mm must be greater than r_in_mm.")
    if float(r_zone_mm) <= 0:
        raise ValueError("r_zone_mm must be positive.")

    root = _open_root(zarr_path, mode="r")
    run_group, distance_run_name, distance_run_path = _resolve_chaser_distance_run(root, chaser_distance_run)
    coordinate_frame = str(run_group.attrs.get("coordinate_frame") or "")
    coordinate_origin = str(run_group.attrs.get("coordinate_origin") or "")
    if coordinate_frame != "arena_relative_canvas_px":
        raise ValueError(
            f"Chaser near-field occupancy requires coordinate_frame='arena_relative_canvas_px'; got {coordinate_frame!r}."
        )
    if coordinate_origin != "top_left_of_active_arena":
        raise ValueError(
            f"Chaser near-field occupancy requires coordinate_origin='top_left_of_active_arena'; got {coordinate_origin!r}."
        )
    pixels_per_mm = _safe_float(run_group.attrs.get("pixels_per_mm_projector"))
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError("Chaser-distance run lacks a positive pixels_per_mm_projector attr.")

    quadrant_component, quadrant_component_name, quadrant_component_path = (
        _resolve_quadrant_occupancy_component(
            run_group,
            chaser_distance_run_path=distance_run_path,
            component_name=quadrant_occupancy_component,
        )
    )
    if "positions" not in run_group or "distances" not in run_group or "chasers" not in run_group:
        raise ValueError("Chaser-distance run is missing positions, distances, or chasers group.")
    positions = run_group["positions"]
    distances = run_group["distances"]
    required_positions = ("fish_centroid_arena_xy", "fish_valid", "chaser_valid")
    missing = [name for name in required_positions if name not in positions]
    missing += [name for name in ("distance_mm",) if name not in distances]
    missing += [name for name in ("chaser_index",) if name not in run_group["chasers"]]
    if missing:
        raise ValueError(
            f"Chaser-distance run missing required Chaser near-field occupancy array(s): {missing}"
        )

    fish_xy = np.asarray(positions["fish_centroid_arena_xy"][:], dtype=np.float32)
    fish_valid = np.asarray(positions["fish_valid"][:], dtype=bool)
    chaser_valid = np.asarray(positions["chaser_valid"][:], dtype=bool)
    distance_mm = np.asarray(distances["distance_mm"][:], dtype=np.float32)
    chaser_indices = np.asarray(run_group["chasers"]["chaser_index"][:], dtype=np.int64)
    total_frames = int(fish_xy.shape[0])
    total_frames_attr = int(run_group.attrs.get("total_frames", 0) or 0)
    if total_frames_attr > 0 and total_frames_attr != total_frames:
        raise ValueError(
            "Chaser-distance run frame-axis mismatch: "
            f"attrs total_frames={total_frames_attr}, fish_centroid_arena_xy length={total_frames}."
        )
    expected = {
        "positions/fish_valid": fish_valid.shape[:1],
        "positions/chaser_valid": chaser_valid.shape[:1],
        "distances/distance_mm": distance_mm.shape[:1],
    }
    mismatched = {name: shape for name, shape in expected.items() if shape != (total_frames,)}
    if mismatched:
        raise ValueError(f"Chaser-distance run arrays disagree on camera-frame axis: {mismatched}")
    if fish_xy.ndim != 2 or fish_xy.shape[1] != 2:
        raise ValueError("positions/fish_centroid_arena_xy must have shape (frame, xy).")
    if distance_mm.ndim != 2:
        raise ValueError("distances/distance_mm must have shape (frame, chaser).")
    if chaser_valid.ndim != 2 or chaser_valid.shape != distance_mm.shape:
        raise ValueError("positions/chaser_valid shape must match distances/distance_mm.")
    if chaser_indices.shape[0] != distance_mm.shape[1]:
        raise ValueError("chasers/chaser_index length does not match distance columns.")

    chasers = _read_chasers(quadrant_component)
    phases = _read_phases(quadrant_component, total_frames=total_frames)
    if "chaser_phase" not in quadrant_component:
        raise ValueError(
            "chaser quadrant occupancy component lacks chaser_phase group."
        )
    chaser_phase = quadrant_component["chaser_phase"]
    if "chaser_x_px" not in chaser_phase or "chaser_y_px" not in chaser_phase:
        raise ValueError(
            "chaser quadrant occupancy chaser_phase group lacks chaser_x_px/chaser_y_px."
        )
    chaser_x_px = np.asarray(chaser_phase["chaser_x_px"][:], dtype=np.float32)
    chaser_y_px = np.asarray(chaser_phase["chaser_y_px"][:], dtype=np.float32)
    if chaser_x_px.shape[:2] != (len(phases), len(chasers)) or chaser_y_px.shape[
        :2
    ] != (len(phases), len(chasers)):
        raise ValueError(
            "chaser quadrant occupancy chaser_phase arrays do not match phase/chaser axes."
        )

    geometry, geometry_notes = _resolve_arena_geometry(
        root,
        run_group,
        quadrant_component,
        pixels_per_mm=float(pixels_per_mm),
    )
    # A fish "outside the arena" is a geometry error, not a fish. Say so instead of quietly
    # dropping those frames downstream.
    geometry_notes = geometry_notes + out_of_bounds_notes(fish_xy, fish_valid, geometry, label="fish")

    percentiles = _normalize_float_array(percentile_values, name="percentile_values")
    if np.any((percentiles < 0) | (percentiles > 100)):
        raise ValueError("percentile_values must be in [0, 100].")
    radial_edges = (
        _default_radial_bin_edges(distance_mm)
        if radial_bin_edges_mm is None
        else _normalize_float_array(radial_bin_edges_mm, name="radial_bin_edges_mm")
    )
    if radial_edges.shape[0] < 2 or not np.all(np.diff(radial_edges.astype(np.float64)) > 0):
        raise ValueError("radial_bin_edges_mm must contain at least two increasing values.")
    radial_centers = ((radial_edges[:-1] + radial_edges[1:]) / 2.0).astype(np.float32)
    cdf_thresholds = _normalize_float_array(cdf_thresholds_mm, name="cdf_thresholds_mm", positive=True)
    fps = _safe_float(run_group.attrs.get("fps"), 1.0)

    arrays, diagnostics, compute_warnings = _compute_near_field_arrays(
        chasers=chasers,
        phases=phases,
        chaser_indices=chaser_indices,
        fish_xy=fish_xy,
        fish_valid=fish_valid,
        chaser_valid=chaser_valid,
        distance_mm=distance_mm,
        fps=float(fps),
        pixels_per_mm=float(pixels_per_mm),
        geometry=geometry,
        chaser_phase_x_px=chaser_x_px,
        chaser_phase_y_px=chaser_y_px,
        percentile_values=percentiles,
        radial_bin_edges_mm=radial_edges,
        cdf_thresholds_mm=cdf_thresholds,
        r_zone_mm=float(r_zone_mm),
        r_in_mm=float(r_in_mm),
        r_out_mm=float(r_out_mm),
        perimeter_band_mm=float(perimeter_band_mm),
        immobility_speed_threshold_mm_s=float(immobility_speed_threshold_mm_s),
    )

    recording_id = str(run_group.attrs.get("recording_id") or root.attrs.get("recording_id") or Path(zarr_path).stem)
    summary = _build_summary(
        recording_id=recording_id,
        chasers=chasers,
        percentile_values=percentiles,
        approach_percentile_mm=arrays["approach_percentile_mm"],
        approach_percentile_cdf_fraction=arrays["approach_percentile_cdf_fraction"],
        near_zone_occupancy_fraction=arrays["near_zone_occupancy_fraction"],
        near_zone_entry_rate_per_min=arrays["near_zone_entry_rate_per_min"],
        tracking_dropout_fraction=arrays["tracking_dropout_fraction"],
        thigmotaxis_fraction=arrays["thigmotaxis_fraction"],
        mean_speed_mm_s=arrays["mean_speed_mm_s"],
        immobile_fraction=arrays["immobile_fraction"],
        valid_distance_count=arrays["valid_distance_count"],
        phases=phases,
    )
    compute_warnings = tuple(
        dict.fromkeys(list(compute_warnings) + list(geometry_notes))
    )
    source_refs = quadrant_component.attrs.get("source_refs")
    if not isinstance(source_refs, Mapping):
        source_refs = {}
    diagnostics.update(
        {
            "source_quadrant_occupancy_schema_id": quadrant_component.attrs.get(
                "schema_id"
            ),
            "qc_warning_count": len(compute_warnings),
        }
    )
    return ChaserNearFieldOccupancyResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        component_name=str(component_name),
        chaser_distance_run_name=distance_run_name,
        chaser_distance_run_path=distance_run_path,
        source_quadrant_occupancy_component=quadrant_component_name,
        source_quadrant_occupancy_path=quadrant_component_path,
        source_stimulus_run=source_refs.get("source_stimulus_run")
        or run_group.attrs.get("source_stimulus_run"),
        source_stimulus_path=source_refs.get("source_stimulus_path")
        or run_group.attrs.get("source_stimulus_path"),
        source_stimulus_epoch_run=source_refs.get("source_stimulus_epoch_run")
        or run_group.attrs.get("source_stimulus_epoch_run"),
        source_stimulus_epoch_path=source_refs.get("source_stimulus_epoch_path")
        or run_group.attrs.get("source_stimulus_epoch_path"),
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(pixels_per_mm),
        coordinate_frame=coordinate_frame,
        coordinate_origin=coordinate_origin,
        arena_width_px=float(geometry.width_px),
        arena_height_px=float(geometry.height_px),
        geometry_status=geometry.status,
        arena_geometry_source=geometry.source,
        arena_shape=geometry.shape,
        arena_center_x_px=geometry.center_x_px,
        arena_center_y_px=geometry.center_y_px,
        arena_radius_px=geometry.radius_px,
        r_zone_mm=float(r_zone_mm),
        r_in_mm=float(r_in_mm),
        r_out_mm=float(r_out_mm),
        percentile_values=percentiles.astype(np.float32),
        radial_bin_edges_mm=radial_edges.astype(np.float32),
        radial_bin_centers_mm=radial_centers.astype(np.float32),
        cdf_thresholds_mm=cdf_thresholds.astype(np.float32),
        perimeter_band_mm=float(perimeter_band_mm),
        immobility_speed_threshold_mm_s=float(immobility_speed_threshold_mm_s),
        chasers=chasers,
        phases=phases,
        approach_percentile_mm=arrays["approach_percentile_mm"],
        approach_percentile_cdf_fraction=arrays["approach_percentile_cdf_fraction"],
        chaser_x_px=arrays["chaser_x_px"],
        chaser_y_px=arrays["chaser_y_px"],
        chaser_x_mm=arrays["chaser_x_mm"],
        chaser_y_mm=arrays["chaser_y_mm"],
        chaser_distance_to_arena_center_mm=arrays["chaser_distance_to_arena_center_mm"],
        chaser_distance_to_wall_mm=arrays["chaser_distance_to_wall_mm"],
        chaser_displacement_from_pre_mm=arrays["chaser_displacement_from_pre_mm"],
        near_zone_occupancy_fraction=arrays["near_zone_occupancy_fraction"],
        near_zone_occupancy_fraction_of_epoch=arrays["near_zone_occupancy_fraction_of_epoch"],
        near_zone_dwell_s=arrays["near_zone_dwell_s"],
        near_zone_density_per_mm2=arrays["near_zone_density_per_mm2"],
        near_zone_available_area_mm2=arrays["near_zone_available_area_mm2"],
        near_zone_entry_count=arrays["near_zone_entry_count"],
        near_zone_entry_rate_per_min=arrays["near_zone_entry_rate_per_min"],
        near_zone_visit_median_dwell_s=arrays["near_zone_visit_median_dwell_s"],
        near_zone_visit_total_dwell_s=arrays["near_zone_visit_total_dwell_s"],
        valid_distance_count=arrays["valid_distance_count"],
        missing_frame_count=arrays["missing_frame_count"],
        tracking_dropout_fraction=arrays["tracking_dropout_fraction"],
        radial_count=arrays["radial_count"],
        radial_fraction=arrays["radial_fraction"],
        radial_density_per_mm2=arrays["radial_density_per_mm2"],
        radial_available_area_mm2=arrays["radial_available_area_mm2"],
        radial_count_wall_excluded=arrays["radial_count_wall_excluded"],
        radial_fraction_wall_excluded=arrays["radial_fraction_wall_excluded"],
        radial_density_wall_excluded_per_mm2=arrays["radial_density_wall_excluded_per_mm2"],
        radial_available_area_wall_excluded_mm2=arrays["radial_available_area_wall_excluded_mm2"],
        radial_wall_excluded_valid_count=arrays["radial_wall_excluded_valid_count"],
        cdf_fraction=arrays["cdf_fraction"],
        control_reference_labels=tuple(diagnostics.get("control_reference_labels") or ()),
        control_reference_x_px=arrays["control_reference_x_px"],
        control_reference_y_px=arrays["control_reference_y_px"],
        control_reference_approach_percentile_mm=arrays["control_reference_approach_percentile_mm"],
        control_reference_radial_count=arrays["control_reference_radial_count"],
        control_reference_radial_fraction=arrays["control_reference_radial_fraction"],
        control_reference_radial_density_per_mm2=arrays["control_reference_radial_density_per_mm2"],
        control_reference_radial_available_area_mm2=arrays["control_reference_radial_available_area_mm2"],
        control_reference_cdf_fraction=arrays["control_reference_cdf_fraction"],
        thigmotaxis_fraction=arrays["thigmotaxis_fraction"],
        thigmotaxis_dwell_s=arrays["thigmotaxis_dwell_s"],
        mean_speed_mm_s=arrays["mean_speed_mm_s"],
        median_speed_mm_s=arrays["median_speed_mm_s"],
        immobile_fraction=arrays["immobile_fraction"],
        speed_sample_count=arrays["speed_sample_count"],
        endpoint_status="computed",
        qc_warnings=tuple(compute_warnings),
        summary=summary,
        diagnostics=diagnostics,
    )


def render_chaser_near_field_radial_density_png(
    result: ChaserNearFieldOccupancyResult, *, dpi: int = 150
) -> bytes:
    fig, axes = plt.subplots(
        1,
        len(result.phases),
        figsize=(5.0 * len(result.phases), 4.4),
        sharey=True,
        constrained_layout=True,
    )
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for p_idx, (ax, phase) in enumerate(zip(axes_list, result.phases)):
        for o_idx, obj in enumerate(result.chasers):
            density = result.radial_density_per_mm2[p_idx, o_idx, :]
            if not np.isfinite(density).any():
                continue
            ax.plot(
                result.radial_bin_centers_mm,
                density,
                marker="o",
                markersize=3.0,
                linewidth=1.3,
                color=obj.raw_color_hex or None,
                label=f"{obj.behavior_class} ({obj.raw_color_hex})",
            )
        ax.axvline(result.r_zone_mm, color="#334155", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(phase.phase_label.replace("_", " "))
        ax.set_xlabel("distance to chaser (mm)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes_list[0].set_ylabel("occupancy density (fraction / mm2)")
    fig.suptitle(
        f"Chaser near-field occupancy radial density: {result.recording_id}",
        fontsize=12,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_chaser_near_field_distance_cdf_png(
    result: ChaserNearFieldOccupancyResult, *, dpi: int = 150
) -> bytes:
    fig, axes = plt.subplots(
        1,
        len(result.phases),
        figsize=(5.0 * len(result.phases), 4.4),
        sharey=True,
        constrained_layout=True,
    )
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for p_idx, (ax, phase) in enumerate(zip(axes_list, result.phases)):
        for o_idx, obj in enumerate(result.chasers):
            cdf = result.cdf_fraction[p_idx, o_idx, :]
            if not np.isfinite(cdf).any():
                continue
            ax.plot(
                result.cdf_thresholds_mm,
                cdf,
                marker="o",
                linewidth=1.4,
                color=obj.raw_color_hex or None,
                label=f"{obj.behavior_class} ({obj.raw_color_hex})",
            )
        ax.axvline(result.r_zone_mm, color="#334155", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(phase.phase_label.replace("_", " "))
        ax.set_xlabel("distance threshold (mm)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes_list[0].set_ylabel("P(distance <= threshold)")
    fig.suptitle(
        f"Chaser near-field occupancy distance CDF: {result.recording_id}", fontsize=12
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_chaser_near_field_summary_png(
    result: ChaserNearFieldOccupancyResult, *, dpi: int = 150
) -> bytes:
    phase_labels = [phase.phase_label.replace("_", " ") for phase in result.phases]
    x = np.arange(len(phase_labels), dtype=np.float64)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes_list = list(axes.ravel())
    for o_idx, obj in enumerate(result.chasers):
        offset = (o_idx - (len(result.chasers) - 1) / 2.0) * width
        axes_list[0].bar(
            x + offset,
            result.near_zone_occupancy_fraction[:, o_idx],
            width=width,
            color=obj.raw_color_hex or None,
            alpha=0.75,
            label=f"{obj.behavior_class} ({obj.raw_color_hex})",
        )
        axes_list[1].bar(
            x + offset,
            result.approach_percentile_mm[:, o_idx, 0],
            width=width,
            color=obj.raw_color_hex or None,
            alpha=0.75,
            label=f"{obj.behavior_class} ({obj.raw_color_hex})",
        )
    axes_list[0].set_title(f"Near-zone occupancy <= {result.r_zone_mm:g} mm")
    axes_list[0].set_ylabel("fraction of valid distance frames")
    axes_list[0].set_ylim(bottom=0.0)
    axes_list[1].set_title(f"Close approach {_format_percentile_key(float(result.percentile_values[0]))}")
    axes_list[1].set_ylabel("distance (mm)")
    axes_list[1].set_ylim(bottom=0.0)
    for ax in axes_list:
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Chaser near-field occupancy summary: {result.recording_id}", fontsize=12
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _interactive_spec(
    result: ChaserNearFieldOccupancyResult, component_path: str
) -> dict[str, Any]:
    return {
        "schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
        "schema_version": 1,
        "renderer": INTERACTIVE_RENDERER,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "component_path": component_path,
        "source_paths": {
            "component": component_path,
            "config": f"{component_path}/config",
            "chasers": f"{component_path}/chasers",
            "phases": f"{component_path}/phases",
            "per_chaser_phase": f"{component_path}/per_chaser_phase",
            "radial_density": f"{component_path}/radial_density",
            "distance_cdf": f"{component_path}/distance_cdf",
            "control_references": f"{component_path}/control_references",
            "thigmotaxis": f"{component_path}/thigmotaxis",
            "summary": f"{component_path}/summary",
        },
        "summary": result.summary,
        "parameters": {
            "r_zone_mm": result.r_zone_mm,
            "r_in_mm": result.r_in_mm,
            "r_out_mm": result.r_out_mm,
            "percentile_values": result.percentile_values,
            "radial_bin_edges_mm": result.radial_bin_edges_mm,
            "cdf_thresholds_mm": result.cdf_thresholds_mm,
            "perimeter_band_mm": result.perimeter_band_mm,
            "immobility_speed_threshold_mm_s": result.immobility_speed_threshold_mm_s,
            "geometry_status": result.geometry_status,
            "arena_geometry_source": result.arena_geometry_source,
            "arena_shape": result.arena_shape,
            "arena_center_x_px": result.arena_center_x_px,
            "arena_center_y_px": result.arena_center_y_px,
            "arena_radius_px": result.arena_radius_px,
        },
        "qc_warnings": list(result.qc_warnings),
    }


def _source_refs(result: ChaserNearFieldOccupancyResult) -> dict[str, Any]:
    return {
        "source_chaser_distance_run": result.chaser_distance_run_name,
        "source_chaser_distance_path": result.chaser_distance_run_path,
        "source_quadrant_occupancy_component": result.source_quadrant_occupancy_component,
        "source_quadrant_occupancy_path": result.source_quadrant_occupancy_path,
        "source_stimulus_run": result.source_stimulus_run,
        "source_stimulus_path": result.source_stimulus_path,
        "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
        "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
    }


def _parameters(result: ChaserNearFieldOccupancyResult) -> dict[str, Any]:
    return {
        "r_zone_mm": result.r_zone_mm,
        "r_in_mm": result.r_in_mm,
        "r_out_mm": result.r_out_mm,
        "percentile_values": result.percentile_values,
        "radial_bin_edges_mm": result.radial_bin_edges_mm,
        "cdf_thresholds_mm": result.cdf_thresholds_mm,
        "perimeter_band_mm": result.perimeter_band_mm,
        "immobility_speed_threshold_mm_s": result.immobility_speed_threshold_mm_s,
        "geometry_mode": result.geometry_status,
        "arena_geometry_source": result.arena_geometry_source,
        "arena_shape": result.arena_shape,
        "arena_center_x_px": result.arena_center_x_px,
        "arena_center_y_px": result.arena_center_y_px,
        "arena_radius_px": result.arena_radius_px,
        "invalid_gap_policy": result.diagnostics.get("invalid_gap_policy"),
    }


def write_chaser_near_field_occupancy_component(
    zarr_path: Path,
    result: ChaserNearFieldOccupancyResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
    write_interactive_spec: bool = True,
) -> str:
    root = _open_root(zarr_path, mode="a")
    run_group = root[result.chaser_distance_run_path]
    parent = run_group.require_group(COMPONENT_PARENT_NAME)
    component_name = result.component_name
    if component_name in parent:
        if not overwrite:
            raise ValueError(
                f"Chaser near-field occupancy component already exists: {result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"
            )
        del parent[component_name]
    component = parent.create_group(component_name)
    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"

    config = component.require_group("config")
    _write_array(config, "percentile_values", result.percentile_values)
    _write_array(config, "radial_bin_edges_mm", result.radial_bin_edges_mm)
    _write_array(config, "radial_bin_centers_mm", result.radial_bin_centers_mm)
    _write_array(config, "cdf_thresholds_mm", result.cdf_thresholds_mm)
    config.attrs.update(
        {
            "r_zone_mm": float(result.r_zone_mm),
            "r_in_mm": float(result.r_in_mm),
            "r_out_mm": float(result.r_out_mm),
            "perimeter_band_mm": float(result.perimeter_band_mm),
            "immobility_speed_threshold_mm_s": float(result.immobility_speed_threshold_mm_s),
            "geometry_status": result.geometry_status,
            "arena_geometry_source": result.arena_geometry_source,
            "arena_shape": result.arena_shape,
            "arena_center_x_px": result.arena_center_x_px,
            "arena_center_y_px": result.arena_center_y_px,
            "arena_radius_px": result.arena_radius_px,
        }
    )

    chasers = component.require_group("chasers")
    _write_array(
        chasers,
        "chaser_index",
        np.asarray([obj.chaser_index for obj in result.chasers], dtype=np.int16),
    )
    _write_array(
        chasers,
        "behavior_class_id",
        np.asarray([obj.behavior_class_id for obj in result.chasers], dtype=np.int8),
    )
    _write_array(
        chasers,
        "behavior_class_label_bytes",
        _bytes_array([obj.behavior_class for obj in result.chasers], width=32),
    )
    _write_array(
        chasers,
        "raw_color_rgba",
        np.asarray([obj.raw_color_rgba for obj in result.chasers], dtype=np.float32),
    )
    _write_array(
        chasers,
        "raw_color_hex_bytes",
        _bytes_array([obj.raw_color_hex for obj in result.chasers], width=16),
    )
    chasers.attrs.update(
        {
            "row_axis": "chasers",
            "behavior_class_vocabulary": {
                "0": "unknown",
                "1": "aggressive",
                "2": "random_non_chasing",
                "3": "inert",
            },
        }
    )

    phases = component.require_group("phases")
    _write_array(
        phases,
        "phase_index",
        np.asarray([phase.phase_index for phase in result.phases], dtype=np.int16),
    )
    _write_array(
        phases,
        "phase_label_bytes",
        _bytes_array([phase.phase_label for phase in result.phases], width=48),
    )
    _write_array(
        phases,
        "effective_start_frame",
        np.asarray(
            [phase.effective_start_frame for phase in result.phases], dtype=np.int64
        ),
    )
    _write_array(
        phases,
        "effective_end_frame",
        np.asarray(
            [phase.effective_end_frame for phase in result.phases], dtype=np.int64
        ),
    )
    _write_array(
        phases,
        "total_frame_count",
        np.asarray(
            [phase.total_frame_count for phase in result.phases], dtype=np.int64
        ),
    )
    phases.attrs.update({"row_axis": "chaser_near_field_occupancy_phases"})

    per_chaser = component.require_group("per_chaser_phase")
    _write_array(per_chaser, "approach_percentile_mm", result.approach_percentile_mm)
    _write_array(
        per_chaser,
        "approach_percentile_cdf_fraction",
        result.approach_percentile_cdf_fraction,
    )
    _write_array(per_chaser, "chaser_x_px", result.chaser_x_px)
    _write_array(per_chaser, "chaser_y_px", result.chaser_y_px)
    _write_array(per_chaser, "chaser_x_mm", result.chaser_x_mm)
    _write_array(per_chaser, "chaser_y_mm", result.chaser_y_mm)
    _write_array(
        per_chaser,
        "chaser_distance_to_arena_center_mm",
        result.chaser_distance_to_arena_center_mm,
    )
    _write_array(
        per_chaser, "chaser_distance_to_wall_mm", result.chaser_distance_to_wall_mm
    )
    _write_array(
        per_chaser,
        "chaser_displacement_from_pre_mm",
        result.chaser_displacement_from_pre_mm,
    )
    _write_array(
        per_chaser, "near_zone_occupancy_fraction", result.near_zone_occupancy_fraction
    )
    _write_array(
        per_chaser,
        "near_zone_occupancy_fraction_of_epoch",
        result.near_zone_occupancy_fraction_of_epoch,
    )
    _write_array(per_chaser, "near_zone_dwell_s", result.near_zone_dwell_s)
    _write_array(
        per_chaser, "near_zone_density_per_mm2", result.near_zone_density_per_mm2
    )
    _write_array(
        per_chaser, "near_zone_available_area_mm2", result.near_zone_available_area_mm2
    )
    _write_array(per_chaser, "near_zone_entry_count", result.near_zone_entry_count)
    _write_array(
        per_chaser, "near_zone_entry_rate_per_min", result.near_zone_entry_rate_per_min
    )
    _write_array(
        per_chaser,
        "near_zone_visit_median_dwell_s",
        result.near_zone_visit_median_dwell_s,
    )
    _write_array(
        per_chaser,
        "near_zone_visit_total_dwell_s",
        result.near_zone_visit_total_dwell_s,
    )
    _write_array(per_chaser, "valid_distance_count", result.valid_distance_count)
    _write_array(per_chaser, "missing_frame_count", result.missing_frame_count)
    _write_array(
        per_chaser, "tracking_dropout_fraction", result.tracking_dropout_fraction
    )
    per_chaser.attrs.update(
        {
            "axis_order": {
                "approach_percentile_mm": ["phase", "chaser", "percentile"],
                "approach_percentile_cdf_fraction": ["phase", "chaser", "percentile"],
                "chaser_geometry_arrays": ["phase", "chaser"],
                "near_zone_arrays": ["phase", "chaser"],
            },
            "approach_percentile_cdf_fraction_definition": "fraction of valid distance samples <= persisted approach percentile; should match percentile/100 within interpolation/counting tolerance",
            "near_zone_occupancy_denominator": "valid fish-to-chaser distance frames in effective phase",
            "near_zone_occupancy_of_epoch_denominator": "all frames in effective phase",
        }
    )

    radial = component.require_group("radial_density")
    _write_array(radial, "radial_count", result.radial_count)
    _write_array(radial, "radial_fraction", result.radial_fraction)
    _write_array(radial, "radial_density_per_mm2", result.radial_density_per_mm2)
    _write_array(radial, "radial_available_area_mm2", result.radial_available_area_mm2)
    _write_array(radial, "radial_count_wall_excluded", result.radial_count_wall_excluded)
    _write_array(radial, "radial_fraction_wall_excluded", result.radial_fraction_wall_excluded)
    _write_array(radial, "radial_density_wall_excluded_per_mm2", result.radial_density_wall_excluded_per_mm2)
    _write_array(radial, "radial_available_area_wall_excluded_mm2", result.radial_available_area_wall_excluded_mm2)
    _write_array(radial, "radial_wall_excluded_valid_count", result.radial_wall_excluded_valid_count)
    radial.attrs.update(
        {
            "axis_order": ["phase", "chaser", "radial_bin"],
            "geometry_status": result.geometry_status,
            "wall_excluded_policy": "exclude fish samples and available area within perimeter_band_mm of arena wall",
        }
    )

    cdf = component.require_group("distance_cdf")
    _write_array(cdf, "cdf_fraction", result.cdf_fraction)
    cdf.attrs.update(
        {"axis_order": ["phase", "chaser", "threshold"], "threshold_unit": "mm"}
    )

    controls = component.require_group("control_references")
    _write_array(controls, "reference_label_bytes", _bytes_array(result.control_reference_labels, width=48))
    _write_array(controls, "reference_x_px", result.control_reference_x_px)
    _write_array(controls, "reference_y_px", result.control_reference_y_px)
    _write_array(controls, "approach_percentile_mm", result.control_reference_approach_percentile_mm)
    _write_array(controls, "radial_count", result.control_reference_radial_count)
    _write_array(controls, "radial_fraction", result.control_reference_radial_fraction)
    _write_array(controls, "radial_density_per_mm2", result.control_reference_radial_density_per_mm2)
    _write_array(controls, "radial_available_area_mm2", result.control_reference_radial_available_area_mm2)
    _write_array(controls, "cdf_fraction", result.control_reference_cdf_fraction)
    controls.attrs.update(
        {
            "row_axis": "control_reference",
            "axis_order": {
                "reference_arrays": ["reference"],
                "approach_percentile_mm": ["phase", "reference", "percentile"],
                "radial_arrays": ["phase", "reference", "radial_bin"],
                "cdf_fraction": ["phase", "reference", "threshold"],
            },
            "purpose": "non-chaser reference diagnostics for global spatial shifts",
        }
    )

    thig = component.require_group("thigmotaxis")
    _write_array(thig, "thigmotaxis_fraction", result.thigmotaxis_fraction)
    _write_array(thig, "thigmotaxis_dwell_s", result.thigmotaxis_dwell_s)
    _write_array(thig, "mean_speed_mm_s", result.mean_speed_mm_s)
    _write_array(thig, "median_speed_mm_s", result.median_speed_mm_s)
    _write_array(thig, "immobile_fraction", result.immobile_fraction)
    _write_array(thig, "speed_sample_count", result.speed_sample_count)
    _write_array(thig, "geometry_status_bytes", _bytes_array([result.geometry_status for _ in result.phases], width=48))
    thig.attrs.update(
        {
            "row_axis": "phase",
            "geometry_status": result.geometry_status,
            "arena_geometry_source": result.arena_geometry_source,
            "arena_shape": result.arena_shape,
            "immobility_speed_threshold_mm_s": float(result.immobility_speed_threshold_mm_s),
        }
    )

    summary_group = component.require_group("summary")
    for key, value in result.summary.items():
        if isinstance(value, str) or value is None:
            _write_array(
                summary_group,
                f"{key}_bytes",
                _bytes_array(["" if value is None else str(value)], width=128),
            )
        elif isinstance(value, (list, dict)):
            _write_array(
                summary_group,
                f"{key}_json_bytes",
                _bytes_array(
                    [json.dumps(json_attr_safe(value), sort_keys=True)], width=65536
                ),
            )
        elif isinstance(value, int):
            _write_array(summary_group, key, np.asarray([value], dtype=np.int64))
        else:
            _write_array(summary_group, key, np.asarray([np.nan if value is None else float(value)], dtype=np.float32))
    _write_array(summary_group, "endpoint_status_bytes", _bytes_array([result.endpoint_status], width=48))
    _write_array(summary_group, "diagnostics_json_bytes", _bytes_array([json.dumps(json_attr_safe(result.diagnostics), sort_keys=True)], width=4096))
    _write_array(summary_group, "qc_warnings_json_bytes", _bytes_array([json.dumps(list(result.qc_warnings), sort_keys=True)], width=4096))
    summary_group.attrs.update({"row_axis": "fish_recording", "summary": json_attr_safe(result.summary)})

    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = _source_refs(result)
    parameters = _parameters(result)
    attrs = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_name": component_name,
        "recording_id": result.recording_id,
        "row_axis": "fish_recording",
        "status": result.endpoint_status,
        "source_refs": source_refs,
        "parameters": parameters,
        "summary": result.summary,
        "qc_warnings": list(result.qc_warnings),
        "diagnostics": result.diagnostics,
        "coordinate_frame": result.coordinate_frame,
        "coordinate_origin": result.coordinate_origin,
        "x_axis_direction": "right",
        "y_axis_direction": "down",
        "pixels_per_mm_projector": result.pixels_per_mm_projector,
        "arena_width_px": result.arena_width_px,
        "arena_height_px": result.arena_height_px,
        "geometry_status": result.geometry_status,
        "arena_geometry_source": result.arena_geometry_source,
        "arena_shape": result.arena_shape,
        "arena_center_x_px": result.arena_center_x_px,
        "arena_center_y_px": result.arena_center_y_px,
        "arena_radius_px": result.arena_radius_px,
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("is_dirty"),
        "provenance": {
            "stage": "chaser_near_field_occupancy",
            "created_by": "fisheye.analysis.chaser_near_field_occupancy",
            "inputs": source_refs,
            "parameters": parameters,
        },
    }
    component.attrs.update(json_attr_safe(attrs))
    lineage_payload = build_run_lineage_payload(
        run_family=f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}",
        analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "row_axis": "fish_recording"},
        method=METHOD,
        method_version=METHOD_VERSION,
        source_refs=source_refs,
        parameters=parameters,
        code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
    )
    write_run_lineage_attrs(component, lineage_payload, fingerprint_status="best_effort", overwrite=True)
    parent.attrs["latest"] = component_name
    parent.attrs["latest_complete"] = component_name

    if write_png:
        source_paths = {
            **source_refs,
            "component": component_path,
            "per_chaser_phase": f"{component_path}/per_chaser_phase",
            "radial_density": f"{component_path}/radial_density",
            "distance_cdf": f"{component_path}/distance_cdf",
            "control_references": f"{component_path}/control_references",
        }
        for artifact_name, renderer, description in (
            (
                RADIAL_DENSITY_PNG_ARTIFACT_NAME,
                render_chaser_near_field_radial_density_png,
                "chaser Chaser near-field occupancy radial occupancy density.",
            ),
            (
                DISTANCE_CDF_PNG_ARTIFACT_NAME,
                render_chaser_near_field_distance_cdf_png,
                "chaser Chaser near-field occupancy distance CDF.",
            ),
            (
                SUMMARY_PNG_ARTIFACT_NAME,
                render_chaser_near_field_summary_png,
                "chaser Chaser near-field occupancy summary.",
            ),
        ):
            write_png_visualization_artifact(
                component,
                artifact_name,
                renderer(result),
                description=description,
                created_by="fisheye.analysis.chaser_near_field_occupancy",
                role="analysis_summary",
                source_paths=source_paths,
                source_runs={"chaser_distance_run": result.chaser_distance_run_name},
                parameters=parameters,
                extra_attrs={
                    "chaser_near_field_occupancy_schema_id": SCHEMA_ID,
                    "component_path": component_path,
                    "canonical_artifact": True,
                },
                overwrite=True,
            )

    if write_interactive_spec:
        spec = _interactive_spec(result, component_path)
        write_interactive_plot_spec_artifact(
            component,
            INTERACTIVE_ARTIFACT_NAME,
            spec,
            description="chaser Chaser near-field occupancy interactive plot spec.",
            created_by="fisheye.analysis.chaser_near_field_occupancy",
            renderer=INTERACTIVE_RENDERER,
            artifact_signature=None,
            snapshot_artifact=SUMMARY_PNG_ARTIFACT_NAME,
            source_paths=spec["source_paths"],
            source_runs={"chaser_distance_run": result.chaser_distance_run_name},
            parameters=parameters,
            extra_attrs={
                "plot_schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
                "component_path": component_path,
                "summary": json_attr_safe(result.summary),
                "canonical_artifact": True,
            },
            overwrite=True,
        )
    return component_path


def _result_payload(
    result: ChaserNearFieldOccupancyResult, *, applied_path: str | None
) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "applied_path": applied_path,
        "chaser_distance_run": result.chaser_distance_run_name,
        "source_quadrant_occupancy_component": result.source_quadrant_occupancy_component,
        "endpoint_status": result.endpoint_status,
        "qc_warnings": list(result.qc_warnings),
        "summary": result.summary,
    }


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--quadrant-occupancy-component", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--r-zone-mm", type=float, default=DEFAULT_R_ZONE_MM)
    parser.add_argument("--r-in-mm", type=float, default=DEFAULT_R_IN_MM)
    parser.add_argument("--r-out-mm", type=float, default=DEFAULT_R_OUT_MM)
    parser.add_argument("--percentiles", default=",".join(f"{value:g}" for value in DEFAULT_PERCENTILES))
    parser.add_argument("--radial-bin-edges-mm", default=None, help="Comma-separated edges. Defaults to data-driven bins.")
    parser.add_argument("--cdf-thresholds-mm", default=",".join(f"{value:g}" for value in DEFAULT_CDF_THRESHOLDS_MM))
    parser.add_argument("--perimeter-band-mm", type=float, default=DEFAULT_PERIMETER_BAND_MM)
    parser.add_argument("--immobility-speed-threshold-mm-s", type=float, default=DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S)
    parser.add_argument("--apply", action="store_true", help="Write the near-field component.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing component.")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG artifacts.")
    parser.add_argument("--no-interactive-spec", action="store_true", help="Skip interactive plot spec artifact.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_near_field_occupancy_result(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        quadrant_occupancy_component=str(args.quadrant_occupancy_component),
        component_name=str(args.component_name),
        r_zone_mm=float(args.r_zone_mm),
        r_in_mm=float(args.r_in_mm),
        r_out_mm=float(args.r_out_mm),
        percentile_values=_parse_float_list(str(args.percentiles)),
        radial_bin_edges_mm=None if args.radial_bin_edges_mm is None else _parse_float_list(str(args.radial_bin_edges_mm)),
        cdf_thresholds_mm=_parse_float_list(str(args.cdf_thresholds_mm)),
        perimeter_band_mm=float(args.perimeter_band_mm),
        immobility_speed_threshold_mm_s=float(args.immobility_speed_threshold_mm_s),
    )
    applied_path = None
    if args.apply:
        applied_path = write_chaser_near_field_occupancy_component(
            Path(args.zarr_path),
            result,
            overwrite=bool(args.overwrite),
            write_png=not bool(args.no_png),
            write_interactive_spec=not bool(args.no_interactive_spec),
        )
    payload = _result_payload(result, applied_path=applied_path)
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"recording_id\t{result.recording_id}")
        print(f"component_name\t{result.component_name}")
        print(f"chaser_distance_run\t{result.chaser_distance_run_name}")
        print(
            f"quadrant_occupancy_component\t{result.source_quadrant_occupancy_component}"
        )
        print(f"endpoint_status\t{result.endpoint_status}")
        print(f"qc_warning_count\t{len(result.qc_warnings)}")
        if applied_path:
            print(f"applied_path\t{applied_path}")
        else:
            print("dry_run\ttrue")
            print("pass --apply to write the near-field component")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
