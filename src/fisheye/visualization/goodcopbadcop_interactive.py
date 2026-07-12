"""Interactive visualization adapters for chaser-protocol analyses.

The first persisted dashboard artifacts used GoodCopBadCop-specific schema,
renderer, and artifact names. New writers use protocol-neutral names below; the
legacy constants remain as read aliases so existing zarrs stay viewable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl
import zarr

from fisheye.analysis.chaser_behavior import canonical_behavior_label
from fisheye.analysis.chaser_state_interpolator import load_structured_dataset
from fisheye.shared.coordinate_transform import load_calibration_transform, projector_to_camera_px
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.shared.zarr_io import open_zarr_root


CHASER_PROTOCOL_DASHBOARD_SPEC_SCHEMA_ID = "palette.plot_spec.chaser_protocol_dashboard.v1"
CHASER_PROTOCOL_DASHBOARD_RENDERER = "palette-chaser-protocol-dashboard-v1"
DEFAULT_CHASER_PROTOCOL_DASHBOARD_INTERACTIVE_ARTIFACT = "chaser_protocol_dashboard_interactive"
LEGACY_CHASER_DASHBOARD_SPEC_SCHEMA_ID = "palette.plot_spec.chaser_dashboard.v1"
LEGACY_CHASER_DASHBOARD_RENDERER = "palette-chaser-dashboard-v1"
LEGACY_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT = "chaser_dashboard_interactive"
LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID = "palette.plot_spec.goodcopbadcop_chaser_dashboard.v1"
LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER = "palette-goodcopbadcop-chaser-dashboard-v1"
LEGACY_GOODCOPBADCOP_INTERACTIVE_ARTIFACT = "goodcopbadcop_chaser_dashboard_interactive"
CHASER_PROTOCOL_DASHBOARD_SPEC_SCHEMA_IDS = (
    CHASER_PROTOCOL_DASHBOARD_SPEC_SCHEMA_ID,
    LEGACY_CHASER_DASHBOARD_SPEC_SCHEMA_ID,
    LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID,
)
CHASER_PROTOCOL_DASHBOARD_RENDERERS = (
    CHASER_PROTOCOL_DASHBOARD_RENDERER,
    LEGACY_CHASER_DASHBOARD_RENDERER,
    LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
)
CHASER_PROTOCOL_DASHBOARD_INTERACTIVE_ARTIFACTS = (
    DEFAULT_CHASER_PROTOCOL_DASHBOARD_INTERACTIVE_ARTIFACT,
    LEGACY_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
    LEGACY_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
)

# Shorter protocol-neutral aliases retained for code readability.
CHASER_DASHBOARD_SPEC_SCHEMA_ID = CHASER_PROTOCOL_DASHBOARD_SPEC_SCHEMA_ID
CHASER_DASHBOARD_RENDERER = CHASER_PROTOCOL_DASHBOARD_RENDERER
DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT = DEFAULT_CHASER_PROTOCOL_DASHBOARD_INTERACTIVE_ARTIFACT
CHASER_DASHBOARD_SPEC_SCHEMA_IDS = CHASER_PROTOCOL_DASHBOARD_SPEC_SCHEMA_IDS
CHASER_DASHBOARD_RENDERERS = CHASER_PROTOCOL_DASHBOARD_RENDERERS
CHASER_DASHBOARD_INTERACTIVE_ARTIFACTS = CHASER_PROTOCOL_DASHBOARD_INTERACTIVE_ARTIFACTS

# Backward-compatible public aliases for existing callers and tests.
GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID = LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID
GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER = LEGACY_GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER
DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT = LEGACY_GOODCOPBADCOP_INTERACTIVE_ARTIFACT
GOODCOPBADCOP_CRA_COMPONENT_PARENT = "cra_primary_endpoint"
GOODCOPBADCOP_CRA_SCHEMA_ID = "palette.goodcopbadcop.cra_primary_endpoint.v1"
GOODCOPBADCOP_CRA_NEAR_FIELD_COMPONENT_PARENT = "cra_near_field"
GOODCOPBADCOP_CRA_NEAR_FIELD_SCHEMA_ID = "palette.goodcopbadcop.cra_near_field.v1"
GOODCOPBADCOP_EPOCH_BEHAVIOR_COMPONENT_PARENT = "epoch_behavior_summary"
GOODCOPBADCOP_EPOCH_BEHAVIOR_SCHEMA_ID = "palette.goodcopbadcop.epoch_behavior_summary.v1"
GOODCOPBADCOP_ESCAPE_FREEZE_COMPONENT_PARENT = "chaser_escape_freeze"
GOODCOPBADCOP_ESCAPE_FREEZE_SCHEMA_ID = "palette.goodcopbadcop.chaser_escape_freeze_canary.v1"
GOODCOPBADCOP_ESCAPE_FREEZE_PER_TRIAL_PNG = "escape_freeze_per_trial_diagnostic_png"
GOODCOPBADCOP_ESCAPE_FREEZE_FISH_CENTERED_PNG = "escape_freeze_fish_centered_diagnostic_png"
GOODCOPBADCOP_ESCAPE_FREEZE_SCATTER_PNG = "escape_freeze_speed_displacement_scatter_png"
GOODCOPBADCOP_ESCAPE_FREEZE_RESPONSE_CLASS_BAR_PNG = "escape_freeze_response_class_bar_png"
GOODCOPBADCOP_ESCAPE_FREEZE_TRIAL_OUTCOME_TIMELINE_PNG = "escape_freeze_trial_outcome_timeline_png"
GOODCOPBADCOP_ESCAPE_FREEZE_FISH_CENTERED_POLAR_APPROACH_PNG = "escape_freeze_fish_centered_polar_approach_png"
GOODCOPBADCOP_ESCAPE_FREEZE_FISH_CENTERED_POLAR_DENSITY_PNG = "escape_freeze_fish_centered_polar_density_png"
GOODCOPBADCOP_CRA_QUADRANT_LABELS = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass(frozen=True)
class GoodCopBadCopWindow:
    window_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float


@dataclass(frozen=True)
class GoodCopBadCopSpatialOccupancyZoneSet:
    zone_set_id: str
    zone_set_source: str
    coordinate_frame: str
    coordinate_origin: str
    x_axis_direction: str
    y_axis_direction: str
    zone_id: tuple[str, ...]
    zone_label: tuple[str, ...]
    display_order: np.ndarray
    bounds_xyxy: np.ndarray
    frame_count: np.ndarray
    time_s: np.ndarray
    fraction_of_epoch: np.ndarray
    fraction_of_detected: np.ndarray
    detected_frame_count: np.ndarray
    missing_frame_count: np.ndarray
    total_span_frames: np.ndarray
    coverage_pct: np.ndarray


@dataclass(frozen=True)
class GoodCopBadCopRunOption:
    run_name: str
    run_path: str
    artifact_name: str
    label: str
    is_latest: bool
    attrs: Mapping[str, Any]
    spec: Mapping[str, Any]


@dataclass(frozen=True)
class GoodCopBadCopInteractiveData:
    zarr_path: Path
    run_name: str
    run_path: str
    artifact_name: str
    spec: Mapping[str, Any]
    attrs: Mapping[str, Any]
    source_paths: Mapping[str, str]
    fps: float
    total_frames: int
    camera_frame_id: np.ndarray
    time_seconds: np.ndarray
    stimulus_epoch_window_id: Optional[np.ndarray]
    windows: tuple[GoodCopBadCopWindow, ...]
    chaser_indices: np.ndarray
    chaser_color_hex: Mapping[int, str]
    fish_centroid_arena_xy: np.ndarray
    fish_valid: np.ndarray
    chaser_arena_xy: Optional[np.ndarray]
    chaser_source_img_xy: Optional[np.ndarray]
    chaser_valid: Optional[np.ndarray]
    distance_mm: np.ndarray
    nearest_distance_mm: Optional[np.ndarray]
    nearest_chaser_index: Optional[np.ndarray]
    occupancy_normalized: Optional[np.ndarray]
    occupancy_counts: Optional[np.ndarray]
    occupancy_x_edges: Optional[np.ndarray]
    occupancy_y_edges: Optional[np.ndarray]
    spatial_occupancy: tuple[GoodCopBadCopSpatialOccupancyZoneSet, ...]
    egocentric_component_name: Optional[str]
    egocentric_component_path: Optional[str]
    egocentric_fish_heading_deg: Optional[np.ndarray]
    egocentric_fish_heading_valid: Optional[np.ndarray]
    egocentric_bearing_deg: Optional[np.ndarray]
    egocentric_alignment_cos: Optional[np.ndarray]
    egocentric_lateral_sin: Optional[np.ndarray]
    egocentric_valid: Optional[np.ndarray]
    egocentric_distance_bin_edges_mm: Optional[np.ndarray]
    egocentric_distance_bin_centers_mm: Optional[np.ndarray]
    egocentric_bearing_bin_edges_deg: Optional[np.ndarray]
    egocentric_bearing_bin_centers_deg: Optional[np.ndarray]
    egocentric_hist_counts: Optional[np.ndarray]
    egocentric_hist_probability: Optional[np.ndarray]


@dataclass(frozen=True)
class GoodCopBadCopCRAEndpointData:
    zarr_path: Path
    run_path: str
    component_name: str
    component_path: str
    attrs: Mapping[str, Any]
    summary: Mapping[str, Any]
    qc_warnings: tuple[str, ...]
    fps: float
    quadrant_width_px: float
    quadrant_height_px: float
    objects_df: pl.DataFrame
    phases_df: pl.DataFrame
    object_phase_df: pl.DataFrame
    per_object_phase_df: pl.DataFrame


@dataclass(frozen=True)
class GoodCopBadCopCRANearFieldData:
    zarr_path: Path
    run_path: str
    component_name: str
    component_path: str
    attrs: Mapping[str, Any]
    parameters: Mapping[str, Any]
    summary: Mapping[str, Any]
    qc_warnings: tuple[str, ...]
    fps: float
    geometry_status: str
    arena_shape: str
    arena_geometry_source: Optional[str]
    objects_df: pl.DataFrame
    phases_df: pl.DataFrame
    per_object_phase_df: pl.DataFrame
    radial_density_df: pl.DataFrame
    cdf_df: pl.DataFrame
    control_reference_radial_density_df: pl.DataFrame
    control_reference_cdf_df: pl.DataFrame
    control_reference_phase_df: pl.DataFrame
    thigmotaxis_df: pl.DataFrame


@dataclass(frozen=True)
class GoodCopBadCopEpochBehaviorData:
    zarr_path: Path
    run_path: str
    component_name: str
    component_path: str
    attrs: Mapping[str, Any]
    source_refs: Mapping[str, Any]
    parameters: Mapping[str, Any]
    per_epoch_fish_df: pl.DataFrame
    per_epoch_chaser_df: pl.DataFrame
    per_epoch_bouts_df: pl.DataFrame
    per_epoch_bout_histograms_df: pl.DataFrame
    per_epoch_inter_bout_interval_histograms_df: pl.DataFrame
    center_distance_histogram_df: pl.DataFrame


@dataclass(frozen=True)
class GoodCopBadCopEscapeFreezeData:
    zarr_path: Path
    run_path: str
    component_name: str
    component_path: str
    attrs: Mapping[str, Any]
    parameters: Mapping[str, Any]
    summary: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...]
    trials_df: pl.DataFrame
    trial_metrics_df: pl.DataFrame
    trial_trajectories_df: pl.DataFrame
    per_trial_png_path: Optional[str]
    per_trial_png_bytes: bytes
    per_trial_png_error: Optional[str]
    fish_centered_png_path: Optional[str]
    fish_centered_png_bytes: bytes
    fish_centered_png_error: Optional[str]
    response_class_bar_png_path: Optional[str]
    response_class_bar_png_bytes: bytes
    response_class_bar_png_error: Optional[str]
    trial_outcome_timeline_png_path: Optional[str]
    trial_outcome_timeline_png_bytes: bytes
    trial_outcome_timeline_png_error: Optional[str]
    fish_centered_polar_approach_png_path: Optional[str]
    fish_centered_polar_approach_png_bytes: bytes
    fish_centered_polar_approach_png_error: Optional[str]
    fish_centered_polar_density_png_path: Optional[str]
    fish_centered_polar_density_png_bytes: bytes
    fish_centered_polar_density_png_error: Optional[str]
    scatter_png_path: Optional[str]
    scatter_png_bytes: bytes
    scatter_png_error: Optional[str]


def _normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip("/").split("/") if part)


def _join_path(*parts: str) -> str:
    return "/".join(_normalize_path(part) for part in parts if _normalize_path(part))


def _group_keys(group: object) -> list[str]:
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(key) for key in keys_fn())
        except Exception:
            return []
    return []


def _array_keys(group: object) -> list[str]:
    keys_fn = getattr(group, "array_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(key) for key in keys_fn())
        except Exception:
            return []
    return []


def _node_exists(root: zarr.Group, path: str) -> bool:
    try:
        root[_normalize_path(path)]
        return True
    except Exception:
        return False


def _json_from_uint8_array(array: zarr.Array) -> Mapping[str, Any]:
    payload = np.asarray(array[:], dtype=np.uint8).tobytes().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, Mapping):
        raise ValueError("interactive spec payload must be a JSON object")
    return parsed


def _as_str_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _unit_color_to_hex(red: object, green: object, blue: object) -> Optional[str]:
    try:
        channels = [float(red), float(green), float(blue)]
    except Exception:
        return None
    if not all(np.isfinite(value) for value in channels):
        return None
    values = [int(round(max(0.0, min(1.0, value)) * 255.0)) for value in channels]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _chaser_colors_from_protocol_payload(payload: Mapping[str, Any]) -> dict[int, str]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return {}
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        parameters = step.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        chasers = parameters.get("chasers")
        if not isinstance(chasers, list):
            continue
        colors: dict[int, str] = {}
        for index, chaser in enumerate(chasers):
            if not isinstance(chaser, Mapping):
                continue
            color = _unit_color_to_hex(
                chaser.get("color_r"),
                chaser.get("color_g"),
                chaser.get("color_b"),
            )
            if color:
                colors[int(index)] = color
        if colors:
            return colors
    return {}


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row).strip() for row in values]
    return [decode_null_terminated_text(value).strip() for value in values.reshape(-1)]


def _resolve_interactive_artifact(
    root: zarr.Group,
    *,
    run_path: str,
    artifact_name: str,
) -> zarr.Group:
    artifact_path = _join_path(run_path, "visualizations", artifact_name)
    try:
        artifact = root[artifact_path]
    except Exception as exc:
        raise ValueError(f"Interactive artifact not found: {artifact_path}") from exc
    if not isinstance(artifact, zarr.Group) and not hasattr(artifact, "group_keys"):
        raise ValueError(f"Interactive artifact is not a group: {artifact_path}")
    if "spec_json" not in artifact:
        raise ValueError(f"Interactive artifact missing spec_json: {artifact_path}")
    return artifact


def _try_resolve_interactive_artifact(
    root: zarr.Group,
    *,
    run_path: str,
    artifact_name: str,
) -> Optional[zarr.Group]:
    try:
        return _resolve_interactive_artifact(root, run_path=run_path, artifact_name=artifact_name)
    except ValueError:
        return None


def _load_array(
    root: zarr.Group,
    source_paths: Mapping[str, str],
    key: str,
    *,
    required: bool = False,
) -> Optional[np.ndarray]:
    path = source_paths.get(key)
    if not path:
        if required:
            raise ValueError(f"Interactive spec does not define source path {key!r}")
        return None
    try:
        return np.asarray(root[_normalize_path(path)][:])
    except Exception as exc:
        if required:
            raise ValueError(f"Interactive source array not found for {key!r}: {path}") from exc
        return None


def _load_array_at_path(root: zarr.Group, path: str) -> Optional[np.ndarray]:
    try:
        return np.asarray(root[_normalize_path(path)][:])
    except Exception:
        return None


def _derive_detection_spatial_occupancy_path(source_paths: Mapping[str, str]) -> Optional[str]:
    explicit = source_paths.get("detection_spatial_occupancy")
    if explicit:
        return _normalize_path(explicit)
    for key in (
        "detection_occupancy_heatmap_normalized",
        "detection_occupancy_heatmap_counts",
        "detection_occupancy_windows_label_bytes",
    ):
        path = source_paths.get(key)
        if not path:
            continue
        normalized = _normalize_path(path)
        marker = "/heatmaps/"
        if marker in normalized:
            return normalized.split(marker, 1)[0] + "/spatial_occupancy"
        marker = "/windows/"
        if marker in normalized:
            return normalized.split(marker, 1)[0] + "/spatial_occupancy"
    return None


def _load_optional_summary_array(
    group: zarr.Group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: Any,
) -> np.ndarray:
    if name in group:
        data = np.asarray(group[name][:], dtype=dtype)
        if data.shape == shape:
            return data
    return np.zeros(shape, dtype=dtype)


def _load_spatial_occupancy_zone_sets(
    root: zarr.Group,
    source_paths: Mapping[str, str],
) -> tuple[GoodCopBadCopSpatialOccupancyZoneSet, ...]:
    spatial_path = _derive_detection_spatial_occupancy_path(source_paths)
    if not spatial_path:
        return ()
    try:
        spatial_parent = root[_normalize_path(spatial_path)]
    except Exception:
        return ()

    zone_sets: list[GoodCopBadCopSpatialOccupancyZoneSet] = []
    for zone_set_id in _group_keys(spatial_parent):
        try:
            zone_group = spatial_parent[zone_set_id]
            zone_spec = zone_group["zone_spec"]
            summary = zone_group["summary"]
        except Exception:
            continue
        frame_count = _load_array_at_path(root, _join_path(spatial_path, zone_set_id, "summary/frame_count"))
        if frame_count is None:
            continue
        frame_count = np.asarray(frame_count, dtype=np.int64)
        if frame_count.ndim != 2:
            continue
        n_windows, n_zones = frame_count.shape

        zone_ids_raw = (
            np.asarray(zone_spec["zone_id"][:])
            if "zone_id" in zone_spec
            else np.zeros((n_zones, 1), dtype=np.uint8)
        )
        zone_ids = tuple(_decode_text_column(zone_ids_raw)[:n_zones])
        if len(zone_ids) < n_zones:
            zone_ids = tuple([*zone_ids, *(f"zone_{idx}" for idx in range(len(zone_ids), n_zones))])

        if "label_bytes" in zone_spec:
            labels = tuple(_decode_text_column(np.asarray(zone_spec["label_bytes"][:]))[:n_zones])
        else:
            labels = zone_ids
        if len(labels) < n_zones:
            labels = tuple([*labels, *zone_ids[len(labels) :]])

        display_order = (
            np.asarray(zone_spec["display_order"][:], dtype=np.int16).reshape(-1)
            if "display_order" in zone_spec
            else np.arange(n_zones, dtype=np.int16)
        )
        if display_order.shape[0] != n_zones:
            display_order = np.arange(n_zones, dtype=np.int16)
        bounds_xyxy = (
            np.asarray(zone_spec["bounds_xyxy"][:], dtype=np.float64)
            if "bounds_xyxy" in zone_spec
            else np.zeros((n_zones, 4), dtype=np.float64)
        )
        if bounds_xyxy.shape != (n_zones, 4):
            bounds_xyxy = np.zeros((n_zones, 4), dtype=np.float64)

        zone_sets.append(
            GoodCopBadCopSpatialOccupancyZoneSet(
                zone_set_id=str(zone_set_id),
                zone_set_source=str(zone_group.attrs.get("zone_set_source") or ""),
                coordinate_frame=str(zone_group.attrs.get("coordinate_frame") or ""),
                coordinate_origin=str(zone_group.attrs.get("coordinate_origin") or ""),
                x_axis_direction=str(zone_group.attrs.get("x_axis_direction") or ""),
                y_axis_direction=str(zone_group.attrs.get("y_axis_direction") or ""),
                zone_id=zone_ids,
                zone_label=labels,
                display_order=display_order,
                bounds_xyxy=bounds_xyxy,
                frame_count=frame_count,
                time_s=_load_optional_summary_array(summary, "time_s", shape=(n_windows, n_zones), dtype=np.float64),
                fraction_of_epoch=_load_optional_summary_array(
                    summary,
                    "fraction_of_epoch",
                    shape=(n_windows, n_zones),
                    dtype=np.float64,
                ),
                fraction_of_detected=_load_optional_summary_array(
                    summary,
                    "fraction_of_detected",
                    shape=(n_windows, n_zones),
                    dtype=np.float64,
                ),
                detected_frame_count=_load_optional_summary_array(
                    summary,
                    "detected_frame_count",
                    shape=(n_windows,),
                    dtype=np.int64,
                ),
                missing_frame_count=_load_optional_summary_array(
                    summary,
                    "missing_frame_count",
                    shape=(n_windows,),
                    dtype=np.int64,
                ),
                total_span_frames=_load_optional_summary_array(
                    summary,
                    "total_span_frames",
                    shape=(n_windows,),
                    dtype=np.int64,
                ),
                coverage_pct=_load_optional_summary_array(
                    summary,
                    "coverage_pct",
                    shape=(n_windows,),
                    dtype=np.float64,
                ),
            )
        )
    return tuple(zone_sets)


def _resolve_source_stimulus(
    root: zarr.Group,
    *,
    run_path: str,
    spec: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    stimulus_run: Optional[str] = None
    stimulus_path: Optional[str] = None
    try:
        run_group = root[_normalize_path(run_path)]
        raw_run = getattr(run_group, "attrs", {}).get("source_stimulus_run")
        if raw_run:
            stimulus_run = str(raw_run).strip() or None
        raw_path = getattr(run_group, "attrs", {}).get("source_stimulus_path")
        if raw_path:
            stimulus_path = _normalize_path(str(raw_path))
    except Exception:
        stimulus_run = None
        stimulus_path = None

    source_runs = spec.get("source_runs")
    if isinstance(source_runs, Mapping) and source_runs.get("stimulus"):
        stimulus_run = stimulus_run or str(source_runs["stimulus"]).strip() or None
        stimulus_path = stimulus_path or _join_path("analysis/stimulus_runs", str(source_runs["stimulus"]))

    if not stimulus_run and stimulus_path:
        parts = _normalize_path(stimulus_path).split("/")
        if len(parts) >= 3 and parts[-3:-1] == ["analysis", "stimulus_runs"]:
            stimulus_run = parts[-1]

    return stimulus_run, stimulus_path


def _load_chaser_color_hex(
    root: zarr.Group,
    *,
    run_path: str,
    spec: Mapping[str, Any],
    chaser_indices: np.ndarray,
) -> dict[int, str]:
    _stimulus_run, stimulus_path = _resolve_source_stimulus(root, run_path=run_path, spec=spec)

    if not stimulus_path:
        return {}
    try:
        stimulus_group = root[stimulus_path]
    except Exception:
        return {}
    protocol_json = getattr(stimulus_group, "attrs", {}).get("protocol_json")
    if not protocol_json:
        return {}
    try:
        payload = json.loads(str(protocol_json))
    except Exception:
        return {}

    by_protocol_index = _chaser_colors_from_protocol_payload(payload)
    if not by_protocol_index:
        return {}
    out: dict[int, str] = {}
    for chaser_index in np.asarray(chaser_indices, dtype=np.int64).reshape(-1).tolist():
        color = by_protocol_index.get(int(chaser_index))
        if color:
            out[int(chaser_index)] = color
    return out


def _xy_pair_from_attr(value: object) -> Optional[tuple[float, float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    x = _safe_float(value[0])
    y = _safe_float(value[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(x), float(y)


def _arena_origin_from_run(
    root: zarr.Group,
    *,
    run_path: str,
) -> Optional[tuple[float, float]]:
    try:
        run_group = root[_normalize_path(run_path)]
    except Exception:
        return None
    attrs = getattr(run_group, "attrs", {})
    origin = _xy_pair_from_attr(attrs.get("arena_origin_in_canvas_xy"))
    if origin is not None:
        return origin
    x = _safe_float(attrs.get("arena_origin_in_canvas_x_px"))
    y = _safe_float(attrs.get("arena_origin_in_canvas_y_px"))
    if np.isfinite(x) and np.isfinite(y):
        return float(x), float(y)
    return None


def _arena_origin_from_stimulus(
    root: zarr.Group,
    *,
    stimulus_path: Optional[str],
) -> Optional[tuple[float, float]]:
    if not stimulus_path:
        return None
    try:
        stimulus_group = root[_normalize_path(stimulus_path)]
        calibration = stimulus_group.get("calibration")
        arena_geometry = calibration.get("arena_geometry") if calibration is not None else None
    except Exception:
        return None
    if arena_geometry is None:
        return None
    attrs = getattr(arena_geometry, "attrs", {})
    x = _safe_float(attrs.get("arena_origin_in_canvas_x_px"))
    y = _safe_float(attrs.get("arena_origin_in_canvas_y_px"))
    if np.isfinite(x) and np.isfinite(y):
        return float(x), float(y)
    return None


def _load_chaser_source_img_xy(
    root: zarr.Group,
    *,
    run_path: str,
    spec: Mapping[str, Any],
    chaser_arena_xy: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if chaser_arena_xy is None:
        return None

    stimulus_run, stimulus_path = _resolve_source_stimulus(root, run_path=run_path, spec=spec)
    origin = _arena_origin_from_run(root, run_path=run_path) or _arena_origin_from_stimulus(
        root,
        stimulus_path=stimulus_path,
    )
    if origin is None:
        return None

    calibration = load_calibration_transform(root, stimulus_run=stimulus_run)
    homography = calibration.get("homography")
    if homography is None:
        return None
    homography = np.asarray(homography, dtype=np.float64)
    if homography.shape != (3, 3):
        return None
    try:
        canvas_to_source = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None

    source_xy = np.full_like(np.asarray(chaser_arena_xy, dtype=np.float64), np.nan, dtype=np.float64)
    finite = np.isfinite(chaser_arena_xy).all(axis=2)
    if not np.any(finite):
        return source_xy
    origin_xy = np.asarray(origin, dtype=np.float64).reshape(1, 2)
    canvas_xy = np.asarray(chaser_arena_xy, dtype=np.float64)[finite] + origin_xy
    source_xy[finite] = projector_to_camera_px(canvas_xy, canvas_to_source)
    return source_xy


def _derive_component_path_from_source(source_paths: Mapping[str, str], key: str, marker: str) -> Optional[str]:
    path = source_paths.get(key)
    if not path:
        return None
    normalized = _normalize_path(path)
    if marker not in normalized:
        return None
    return normalized.split(marker, 1)[0]


def resolve_related_detection_occupancy_run_path(
    root: zarr.Group,
    chaser_run_group: zarr.Group,
) -> Optional[str]:
    """Find the detection-occupancy run that matches a chaser-distance run."""

    parent = root.get("analysis/detection_occupancy_runs")
    if parent is None:
        return None

    chaser_attrs = dict(getattr(chaser_run_group, "attrs", {}))
    wanted_epoch = str(chaser_attrs.get("source_stimulus_epoch_run") or "").strip()
    wanted_detection = str(chaser_attrs.get("source_detection_path") or "").strip()
    latest_complete = str(parent.attrs.get("latest_complete") or parent.attrs.get("latest") or "").strip()

    matches: list[tuple[int, str]] = []
    for run_name in _group_keys(parent):
        try:
            run_group = parent[run_name]
        except Exception:
            continue
        attrs = dict(getattr(run_group, "attrs", {}))
        if wanted_epoch and str(attrs.get("source_stimulus_epoch_run") or "").strip() != wanted_epoch:
            continue
        if wanted_detection and str(attrs.get("source_detection_path") or "").strip() != wanted_detection:
            continue
        rank = 0 if run_name == latest_complete else 1
        matches.append((rank, run_name))

    if not matches:
        return None
    _rank, selected = sorted(matches)[0]
    return _join_path("analysis/detection_occupancy_runs", selected)


def resolve_latest_egocentric_bearing_component_path(
    root: zarr.Group,
    *,
    run_path: str,
) -> Optional[str]:
    """Return the latest complete egocentric-bearing component for a chaser run."""

    parent_path = _join_path(run_path, "egocentric_bearing")
    try:
        parent = root[parent_path]
    except Exception:
        return None

    for attr_name in ("latest_complete", "latest"):
        candidate = str(getattr(parent, "attrs", {}).get(attr_name) or "").strip()
        if candidate and candidate in _group_keys(parent):
            return _join_path(parent_path, candidate)

    complete_candidates: list[str] = []
    for name in _group_keys(parent):
        try:
            group = parent[name]
        except Exception:
            continue
        attrs = getattr(group, "attrs", {})
        if str(attrs.get("status") or "").strip() == "complete":
            complete_candidates.append(name)
    if complete_candidates:
        return _join_path(parent_path, sorted(complete_candidates)[-1])
    return None


def resolve_latest_cra_primary_endpoint_component_path(
    root: zarr.Group,
    *,
    run_path: str,
) -> Optional[str]:
    """Return the latest complete CRA primary endpoint component for a chaser run."""

    parent_path = _join_path(run_path, GOODCOPBADCOP_CRA_COMPONENT_PARENT)
    try:
        parent = root[parent_path]
    except Exception:
        return None

    keys = set(_group_keys(parent))
    for attr_name in ("latest_complete", "latest"):
        candidate = str(getattr(parent, "attrs", {}).get(attr_name) or "").strip()
        if candidate and candidate in keys:
            return _join_path(parent_path, candidate)

    complete_candidates: list[str] = []
    for name in sorted(keys):
        try:
            group = parent[name]
        except Exception:
            continue
        attrs = getattr(group, "attrs", {})
        if str(attrs.get("status") or "").strip() in {"computed", "complete"}:
            complete_candidates.append(name)
    if complete_candidates:
        return _join_path(parent_path, complete_candidates[-1])
    if keys:
        return _join_path(parent_path, sorted(keys)[-1])
    return None


def resolve_latest_cra_near_field_component_path(
    root: zarr.Group,
    *,
    run_path: str,
) -> Optional[str]:
    """Return the latest complete CRA near-field component for a chaser run."""

    parent_path = _join_path(run_path, GOODCOPBADCOP_CRA_NEAR_FIELD_COMPONENT_PARENT)
    try:
        parent = root[parent_path]
    except Exception:
        return None

    keys = set(_group_keys(parent))
    for attr_name in ("latest_complete", "latest"):
        candidate = str(getattr(parent, "attrs", {}).get(attr_name) or "").strip()
        if candidate and candidate in keys:
            return _join_path(parent_path, candidate)

    complete_candidates: list[str] = []
    for name in sorted(keys):
        try:
            group = parent[name]
        except Exception:
            continue
        attrs = getattr(group, "attrs", {})
        if str(attrs.get("status") or "").strip() in {"computed", "complete"}:
            complete_candidates.append(name)
    if complete_candidates:
        return _join_path(parent_path, complete_candidates[-1])
    if keys:
        return _join_path(parent_path, sorted(keys)[-1])
    return None


def resolve_latest_epoch_behavior_summary_component_path(
    root: zarr.Group,
    *,
    run_path: str,
) -> Optional[str]:
    """Return the latest complete epoch-behavior component for a chaser run."""

    parent_path = _join_path(run_path, GOODCOPBADCOP_EPOCH_BEHAVIOR_COMPONENT_PARENT)
    try:
        parent = root[parent_path]
    except Exception:
        return None

    keys = set(_group_keys(parent))
    for attr_name in ("latest_complete", "latest"):
        candidate = str(getattr(parent, "attrs", {}).get(attr_name) or "").strip()
        if candidate and candidate in keys:
            return _join_path(parent_path, candidate)

    complete_candidates: list[str] = []
    for name in sorted(keys):
        try:
            group = parent[name]
        except Exception:
            continue
        attrs = getattr(group, "attrs", {})
        if str(attrs.get("status") or "").strip() == "complete":
            complete_candidates.append(name)
    if complete_candidates:
        return _join_path(parent_path, complete_candidates[-1])
    return None


def resolve_latest_escape_freeze_component_path(
    root: zarr.Group,
    *,
    run_path: str,
) -> Optional[str]:
    """Return the latest escape/freeze canary component for a chaser run."""

    parent_path = _join_path(run_path, GOODCOPBADCOP_ESCAPE_FREEZE_COMPONENT_PARENT)
    try:
        parent = root[parent_path]
    except Exception:
        return None

    keys = set(_group_keys(parent))
    for attr_name in ("latest_complete", "latest"):
        candidate = str(getattr(parent, "attrs", {}).get(attr_name) or "").strip()
        if candidate and candidate in keys:
            return _join_path(parent_path, candidate)

    canary_candidates: list[str] = []
    for name in sorted(keys):
        try:
            group = parent[name]
        except Exception:
            continue
        attrs = getattr(group, "attrs", {})
        if str(attrs.get("status") or "").strip() in {"diagnostic_canary", "computed", "complete"}:
            canary_candidates.append(name)
    if canary_candidates:
        return _join_path(parent_path, canary_candidates[-1])
    if keys:
        return _join_path(parent_path, sorted(keys)[-1])
    return None


def _structured_records_to_polars(records: np.ndarray) -> pl.DataFrame:
    if records.size == 0 or records.dtype.names is None:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for name in records.dtype.names:
            value = record[name]
            dtype = records.dtype.fields[name][0]
            if dtype.kind == "S":
                row[name] = decode_null_terminated_text(value).strip()
            elif isinstance(value, np.generic):
                row[name] = value.item()
            else:
                row[name] = value
        rows.append(row)
    return pl.DataFrame(rows)


def _columnar_group_to_polars(group: zarr.Group) -> pl.DataFrame:
    array_names = _array_keys(group)
    if not array_names:
        return pl.DataFrame()
    lengths: list[int] = []
    raw_columns: dict[str, np.ndarray] = {}
    for name in array_names:
        try:
            values = np.asarray(group[name][:])
        except Exception:
            continue
        if values.ndim == 0:
            continue
        raw_columns[name] = values
        lengths.append(int(values.shape[0]))
    if not raw_columns or not lengths:
        return pl.DataFrame()
    n = min(lengths)
    if n <= 0:
        return pl.DataFrame()
    columns: dict[str, Any] = {}
    for name, values in raw_columns.items():
        trimmed = values[:n]
        if trimmed.dtype.kind == "S":
            columns[name] = [decode_null_terminated_text(value).strip() for value in trimmed.reshape(-1)]
        else:
            columns[name] = trimmed.tolist()
    return pl.DataFrame(columns)


def _first_scalar_from_array(group: zarr.Group, name: str) -> Any:
    if name not in group:
        return None
    data = np.asarray(group[name][:])
    if data.size == 0:
        return None
    value = data.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _summary_from_cra_component(component: zarr.Group) -> dict[str, Any]:
    attrs_summary = getattr(component, "attrs", {}).get("summary")
    if isinstance(attrs_summary, Mapping):
        return dict(attrs_summary)

    try:
        summary_group = component["summary"]
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for key in _array_keys(summary_group):
        if key.endswith("_bytes"):
            decoded = _decode_text_column(np.asarray(summary_group[key][:]))
            out[key[:-6]] = decoded[0] if decoded else ""
        else:
            out[key] = _first_scalar_from_array(summary_group, key)
    return out


def _qc_warnings_from_cra_component(component: zarr.Group) -> tuple[str, ...]:
    attrs = getattr(component, "attrs", {})
    raw = attrs.get("qc_warnings")
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    try:
        summary_group = component["summary"]
        decoded = _decode_text_column(np.asarray(summary_group["qc_warnings_json_bytes"][:]))
        parsed = json.loads(decoded[0]) if decoded else []
    except Exception:
        parsed = []
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    return ()


def _load_cra_objects_dataframe(component: zarr.Group) -> pl.DataFrame:
    try:
        group = component["objects"]
    except Exception:
        return pl.DataFrame()
    object_index = np.asarray(group["object_index"][:], dtype=np.int64) if "object_index" in group else np.asarray([], dtype=np.int64)
    n = int(object_index.shape[0])
    if "behavior_class_label_bytes" in group:
        roles = _decode_text_column(np.asarray(group["behavior_class_label_bytes"][:]))[:n]
    elif "object_role_label_bytes" in group:
        roles = _decode_text_column(np.asarray(group["object_role_label_bytes"][:]))[:n]
    else:
        roles = []
    colors = _decode_text_column(np.asarray(group["raw_color_hex_bytes"][:]))[:n] if "raw_color_hex_bytes" in group else []
    start_presets = (
        _decode_text_column(np.asarray(group["start_position_preset_bytes"][:]))[:n]
        if "start_position_preset_bytes" in group
        else []
    )
    end_presets = (
        _decode_text_column(np.asarray(group["end_position_preset_bytes"][:]))[:n]
        if "end_position_preset_bytes" in group
        else []
    )
    enable_chase = np.asarray(group["enable_chase"][:], dtype=bool) if "enable_chase" in group else np.zeros(n, dtype=bool)
    behavior_mode = (
        np.asarray(group["behavior_mode"][:], dtype=np.int64)
        if "behavior_mode" in group
        else np.full(n, -1, dtype=np.int64)
    )
    raw_rgba = (
        np.asarray(group["raw_color_rgba"][:], dtype=np.float64)
        if "raw_color_rgba" in group
        else np.full((n, 4), np.nan, dtype=np.float64)
    )
    rows = []
    for idx in range(n):
        rows.append(
            {
                "object_index": int(object_index[idx]),
                "object_axis_index": int(idx),
                "object_role": canonical_behavior_label(roles[idx]) if idx < len(roles) else "unknown",
                "behavior_class": canonical_behavior_label(roles[idx]) if idx < len(roles) else "unknown",
                "raw_color_hex": colors[idx] if idx < len(colors) else "",
                "enable_chase": bool(enable_chase[idx]) if idx < enable_chase.shape[0] else False,
                "behavior_mode": int(behavior_mode[idx]) if idx < behavior_mode.shape[0] else -1,
                "start_position_preset": start_presets[idx] if idx < len(start_presets) else "",
                "end_position_preset": end_presets[idx] if idx < len(end_presets) else "",
                "raw_color_r": float(raw_rgba[idx, 0]) if raw_rgba.shape[0] > idx and raw_rgba.shape[1] > 0 else np.nan,
                "raw_color_g": float(raw_rgba[idx, 1]) if raw_rgba.shape[0] > idx and raw_rgba.shape[1] > 1 else np.nan,
                "raw_color_b": float(raw_rgba[idx, 2]) if raw_rgba.shape[0] > idx and raw_rgba.shape[1] > 2 else np.nan,
                "raw_color_a": float(raw_rgba[idx, 3]) if raw_rgba.shape[0] > idx and raw_rgba.shape[1] > 3 else np.nan,
            }
        )
    return pl.DataFrame(rows)


def _load_cra_phases_dataframe(component: zarr.Group, *, fps: float) -> pl.DataFrame:
    try:
        group = component["phases"]
    except Exception:
        return pl.DataFrame()
    phase_index = np.asarray(group["phase_index"][:], dtype=np.int64) if "phase_index" in group else np.asarray([], dtype=np.int64)
    n = int(phase_index.shape[0])
    labels = _decode_text_column(np.asarray(group["phase_label_bytes"][:]))[:n] if "phase_label_bytes" in group else []
    source_labels = (
        _decode_text_column(np.asarray(group["source_window_label_bytes"][:]))[:n]
        if "source_window_label_bytes" in group
        else []
    )

    def ints(name: str) -> np.ndarray:
        return np.asarray(group[name][:], dtype=np.int64) if name in group else np.zeros(n, dtype=np.int64)

    source_start = ints("source_start_frame")
    source_end = ints("source_end_frame")
    effective_start = ints("effective_start_frame")
    effective_end = ints("effective_end_frame")
    settle_excluded = ints("settle_excluded_frame_count")
    safe_fps = float(fps) if np.isfinite(fps) and fps > 0 else 1.0
    rows = []
    for idx in range(n):
        effective_frame_count = max(0, int(effective_end[idx]) - int(effective_start[idx]) + 1)
        rows.append(
            {
                "phase_index": int(phase_index[idx]),
                "phase_label": labels[idx] if idx < len(labels) else f"phase_{idx}",
                "source_window_label": source_labels[idx] if idx < len(source_labels) else "",
                "source_start_frame": int(source_start[idx]),
                "source_end_frame": int(source_end[idx]),
                "effective_start_frame": int(effective_start[idx]),
                "effective_end_frame": int(effective_end[idx]),
                "settle_excluded_frame_count": int(settle_excluded[idx]),
                "effective_frame_count": int(effective_frame_count),
                "effective_duration_s": float(effective_frame_count) / safe_fps,
                "settle_excluded_duration_s": float(settle_excluded[idx]) / safe_fps,
            }
        )
    return pl.DataFrame(rows)


def _quadrant_label(code: int, labels: Sequence[str] = GOODCOPBADCOP_CRA_QUADRANT_LABELS) -> str:
    return str(labels[int(code)]) if 0 <= int(code) < len(labels) else ""


def _load_cra_object_phase_dataframe(
    component: zarr.Group,
    *,
    objects_df: pl.DataFrame,
    phases_df: pl.DataFrame,
) -> pl.DataFrame:
    try:
        group = component["object_phase"]
    except Exception:
        return pl.DataFrame()
    if objects_df.is_empty() or phases_df.is_empty() or "object_x_px" not in group:
        return pl.DataFrame()

    object_rows = objects_df.to_dicts()
    phase_rows = phases_df.to_dicts()
    shape = (len(phase_rows), len(object_rows))

    def array(name: str, dtype: Any, default: float | int = np.nan) -> np.ndarray:
        if name not in group:
            return np.full(shape, default, dtype=dtype)
        data = np.asarray(group[name][:], dtype=dtype)
        return data if data.shape == shape else np.full(shape, default, dtype=dtype)

    labels = (
        _decode_text_column(np.asarray(group["object_quadrant_label_bytes"][:]))
        if "object_quadrant_label_bytes" in group
        else list(GOODCOPBADCOP_CRA_QUADRANT_LABELS)
    )
    x_px = array("object_x_px", np.float64)
    y_px = array("object_y_px", np.float64)
    x_mm = array("object_x_mm", np.float64)
    y_mm = array("object_y_mm", np.float64)
    q_codes = array("object_quadrant_code", np.int64, default=-1)
    sample_count = array("object_position_sample_count", np.int64, default=0)
    max_drift = array("object_max_drift_mm", np.float64)
    median_drift = array("object_median_drift_mm", np.float64)

    rows = []
    for phase_idx, phase in enumerate(phase_rows):
        for object_idx, obj in enumerate(object_rows):
            q_code = int(q_codes[phase_idx, object_idx])
            rows.append(
                {
                    **phase,
                    "object_index": int(obj.get("object_index", object_idx)),
                    "object_axis_index": int(obj.get("object_axis_index", object_idx)),
                    "object_role": str(obj.get("object_role") or ""),
                    "raw_color_hex": str(obj.get("raw_color_hex") or ""),
                    "enable_chase": bool(obj.get("enable_chase")),
                    "start_position_preset": str(obj.get("start_position_preset") or ""),
                    "end_position_preset": str(obj.get("end_position_preset") or ""),
                    "object_x_px": float(x_px[phase_idx, object_idx]),
                    "object_y_px": float(y_px[phase_idx, object_idx]),
                    "object_x_mm": float(x_mm[phase_idx, object_idx]),
                    "object_y_mm": float(y_mm[phase_idx, object_idx]),
                    "object_quadrant_code": q_code,
                    "object_quadrant": _quadrant_label(q_code, labels),
                    "object_position_sample_count": int(sample_count[phase_idx, object_idx]),
                    "object_max_drift_mm": float(max_drift[phase_idx, object_idx]),
                    "object_median_drift_mm": float(median_drift[phase_idx, object_idx]),
                }
            )
    return pl.DataFrame(rows)


def _load_cra_per_object_phase_dataframe(
    component: zarr.Group,
    *,
    object_phase_df: pl.DataFrame,
) -> pl.DataFrame:
    try:
        group = component["per_object_phase"]
    except Exception:
        return pl.DataFrame()
    if object_phase_df.is_empty():
        return pl.DataFrame()
    phase_indices = object_phase_df["phase_index"].to_numpy()
    object_axis_indices = object_phase_df["object_axis_index"].to_numpy()
    n_phase = int(max(phase_indices)) + 1 if phase_indices.size else 0
    n_object = int(max(object_axis_indices)) + 1 if object_axis_indices.size else 0
    shape = (n_phase, n_object)

    def array(name: str, dtype: Any, default: float | int = np.nan) -> np.ndarray:
        if name not in group:
            return np.full(shape, default, dtype=dtype)
        data = np.asarray(group[name][:], dtype=dtype)
        return data if data.shape == shape else np.full(shape, default, dtype=dtype)

    metric_arrays = {
        "median_distance_mm": array("median_distance_mm", np.float64),
        "mean_distance_mm": array("mean_distance_mm", np.float64),
        "occupancy_fraction": array("occupancy_fraction", np.float64),
        "occupancy_fraction_of_epoch": array("occupancy_fraction_of_epoch", np.float64),
        "valid_frame_count": array("valid_frame_count", np.int64, default=0),
        "distance_valid_frame_count": array("distance_valid_frame_count", np.int64, default=0),
        "total_frame_count": array("total_frame_count", np.int64, default=0),
        "missing_frame_count": array("missing_frame_count", np.int64, default=0),
        "tracking_dropout_fraction": array("tracking_dropout_fraction", np.float64),
    }
    rows = []
    for row in object_phase_df.to_dicts():
        phase_idx = int(row["phase_index"])
        object_idx = int(row["object_axis_index"])
        metrics = {
            key: (
                int(values[phase_idx, object_idx])
                if key.endswith("_count")
                else float(values[phase_idx, object_idx])
            )
            for key, values in metric_arrays.items()
        }
        rows.append({**row, **metrics})
    return pl.DataFrame(rows)


def _percentile_column_key(value: float) -> str:
    number = float(value)
    if np.isclose(number, round(number)):
        return f"p{int(round(number)):02d}"
    text = f"{number:g}".replace(".", "_").replace("-", "m")
    return f"p{text}"


def _array_or_full(
    group: zarr.Group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    default: float | int = np.nan,
) -> np.ndarray:
    if name not in group:
        return np.full(shape, default, dtype=dtype)
    data = np.asarray(group[name][:], dtype=dtype)
    return data if data.shape == shape else np.full(shape, default, dtype=dtype)


def _load_config_float_array(component: zarr.Group, name: str) -> np.ndarray:
    try:
        group = component["config"]
    except Exception:
        return np.asarray([], dtype=np.float32)
    if name not in group:
        return np.asarray([], dtype=np.float32)
    return np.asarray(group[name][:], dtype=np.float32).reshape(-1)


def _load_cra_near_field_per_object_phase_dataframe(
    component: zarr.Group,
    *,
    objects_df: pl.DataFrame,
    phases_df: pl.DataFrame,
) -> pl.DataFrame:
    try:
        group = component["per_object_phase"]
    except Exception:
        return pl.DataFrame()
    if objects_df.is_empty() or phases_df.is_empty():
        return pl.DataFrame()

    phase_rows = phases_df.sort("phase_index").to_dicts()
    object_rows = objects_df.sort("object_axis_index").to_dicts()
    n_phase = len(phase_rows)
    n_object = len(object_rows)
    shape = (n_phase, n_object)
    percentiles = _load_config_float_array(component, "percentile_values")
    approach = (
        np.asarray(group["approach_percentile_mm"][:], dtype=np.float64)
        if "approach_percentile_mm" in group
        else np.full((n_phase, n_object, 0), np.nan, dtype=np.float64)
    )
    if approach.ndim != 3 or approach.shape[:2] != shape:
        approach = np.full((n_phase, n_object, int(percentiles.shape[0])), np.nan, dtype=np.float64)
    if percentiles.shape[0] != approach.shape[2]:
        percentiles = np.asarray([float(index) for index in range(approach.shape[2])], dtype=np.float32)
    approach_cdf = _array_or_full(
        group,
        "approach_percentile_cdf_fraction",
        shape=(n_phase, n_object, int(percentiles.shape[0])),
        dtype=np.float64,
    )

    metric_arrays = {
        "object_x_px": _array_or_full(group, "object_x_px", shape=shape, dtype=np.float64),
        "object_y_px": _array_or_full(group, "object_y_px", shape=shape, dtype=np.float64),
        "object_x_mm": _array_or_full(group, "object_x_mm", shape=shape, dtype=np.float64),
        "object_y_mm": _array_or_full(group, "object_y_mm", shape=shape, dtype=np.float64),
        "object_distance_to_arena_center_mm": _array_or_full(
            group,
            "object_distance_to_arena_center_mm",
            shape=shape,
            dtype=np.float64,
        ),
        "object_distance_to_wall_mm": _array_or_full(
            group,
            "object_distance_to_wall_mm",
            shape=shape,
            dtype=np.float64,
        ),
        "object_displacement_from_pre_mm": _array_or_full(
            group,
            "object_displacement_from_pre_mm",
            shape=shape,
            dtype=np.float64,
        ),
        "near_zone_occupancy_fraction": _array_or_full(group, "near_zone_occupancy_fraction", shape=shape, dtype=np.float64),
        "near_zone_occupancy_fraction_of_epoch": _array_or_full(
            group,
            "near_zone_occupancy_fraction_of_epoch",
            shape=shape,
            dtype=np.float64,
        ),
        "near_zone_dwell_s": _array_or_full(group, "near_zone_dwell_s", shape=shape, dtype=np.float64),
        "near_zone_density_per_mm2": _array_or_full(group, "near_zone_density_per_mm2", shape=shape, dtype=np.float64),
        "near_zone_available_area_mm2": _array_or_full(
            group,
            "near_zone_available_area_mm2",
            shape=shape,
            dtype=np.float64,
        ),
        "near_zone_entry_count": _array_or_full(group, "near_zone_entry_count", shape=shape, dtype=np.int64, default=0),
        "near_zone_entry_rate_per_min": _array_or_full(
            group,
            "near_zone_entry_rate_per_min",
            shape=shape,
            dtype=np.float64,
        ),
        "near_zone_visit_median_dwell_s": _array_or_full(
            group,
            "near_zone_visit_median_dwell_s",
            shape=shape,
            dtype=np.float64,
        ),
        "near_zone_visit_total_dwell_s": _array_or_full(
            group,
            "near_zone_visit_total_dwell_s",
            shape=shape,
            dtype=np.float64,
        ),
        "valid_distance_count": _array_or_full(group, "valid_distance_count", shape=shape, dtype=np.int64, default=0),
        "missing_frame_count": _array_or_full(group, "missing_frame_count", shape=shape, dtype=np.int64, default=0),
        "tracking_dropout_fraction": _array_or_full(group, "tracking_dropout_fraction", shape=shape, dtype=np.float64),
    }
    rows = []
    for phase_axis, phase in enumerate(phase_rows):
        for object_axis, obj in enumerate(object_rows):
            metrics: dict[str, Any] = {}
            for key, values in metric_arrays.items():
                value = values[phase_axis, object_axis]
                metrics[key] = int(value) if key.endswith("_count") else float(value)
            for percentile_axis, percentile in enumerate(percentiles):
                key = _percentile_column_key(float(percentile))
                metrics[f"approach_{key}_mm"] = float(approach[phase_axis, object_axis, percentile_axis])
            for percentile_axis, percentile in enumerate(percentiles):
                key = _percentile_column_key(float(percentile))
                metrics[f"approach_{key}_cdf_fraction"] = float(
                    approach_cdf[phase_axis, object_axis, percentile_axis]
                )
            rows.append({**phase, **obj, **metrics})
    return pl.DataFrame(rows)


def _load_cra_near_field_radial_density_dataframe(
    component: zarr.Group,
    *,
    objects_df: pl.DataFrame,
    phases_df: pl.DataFrame,
) -> pl.DataFrame:
    try:
        group = component["radial_density"]
    except Exception:
        return pl.DataFrame()
    if objects_df.is_empty() or phases_df.is_empty():
        return pl.DataFrame()

    phase_rows = phases_df.sort("phase_index").to_dicts()
    object_rows = objects_df.sort("object_axis_index").to_dicts()
    edges = _load_config_float_array(component, "radial_bin_edges_mm")
    centers = _load_config_float_array(component, "radial_bin_centers_mm")
    n_phase = len(phase_rows)
    n_object = len(object_rows)
    n_bin = int(max(0, edges.shape[0] - 1))
    if centers.shape[0] != n_bin and n_bin > 0:
        centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    shape = (n_phase, n_object, n_bin)

    count = _array_or_full(group, "radial_count", shape=shape, dtype=np.int64, default=0)
    fraction = _array_or_full(group, "radial_fraction", shape=shape, dtype=np.float64)
    density = _array_or_full(group, "radial_density_per_mm2", shape=shape, dtype=np.float64)
    area = _array_or_full(group, "radial_available_area_mm2", shape=shape, dtype=np.float64)
    count_wall_excluded = _array_or_full(
        group,
        "radial_count_wall_excluded",
        shape=shape,
        dtype=np.int64,
        default=0,
    )
    fraction_wall_excluded = _array_or_full(
        group,
        "radial_fraction_wall_excluded",
        shape=shape,
        dtype=np.float64,
    )
    density_wall_excluded = _array_or_full(
        group,
        "radial_density_wall_excluded_per_mm2",
        shape=shape,
        dtype=np.float64,
    )
    area_wall_excluded = _array_or_full(
        group,
        "radial_available_area_wall_excluded_mm2",
        shape=shape,
        dtype=np.float64,
    )
    wall_excluded_valid = _array_or_full(
        group,
        "radial_wall_excluded_valid_count",
        shape=(n_phase, n_object),
        dtype=np.int64,
        default=0,
    )

    rows = []
    for phase_axis, phase in enumerate(phase_rows):
        for object_axis, obj in enumerate(object_rows):
            for bin_axis in range(n_bin):
                rows.append(
                    {
                        **phase,
                        **obj,
                        "radial_bin_index": int(bin_axis),
                        "radial_bin_start_mm": float(edges[bin_axis]),
                        "radial_bin_end_mm": float(edges[bin_axis + 1]),
                        "radial_bin_center_mm": float(centers[bin_axis]),
                        "radial_count": int(count[phase_axis, object_axis, bin_axis]),
                        "radial_fraction": float(fraction[phase_axis, object_axis, bin_axis]),
                        "radial_density_per_mm2": float(density[phase_axis, object_axis, bin_axis]),
                        "radial_available_area_mm2": float(area[phase_axis, object_axis, bin_axis]),
                        "radial_count_wall_excluded": int(count_wall_excluded[phase_axis, object_axis, bin_axis]),
                        "radial_fraction_wall_excluded": float(
                            fraction_wall_excluded[phase_axis, object_axis, bin_axis]
                        ),
                        "radial_density_wall_excluded_per_mm2": float(
                            density_wall_excluded[phase_axis, object_axis, bin_axis]
                        ),
                        "radial_available_area_wall_excluded_mm2": float(
                            area_wall_excluded[phase_axis, object_axis, bin_axis]
                        ),
                        "radial_wall_excluded_valid_count": int(wall_excluded_valid[phase_axis, object_axis]),
                    }
                )
    return pl.DataFrame(rows)


def _load_cra_near_field_cdf_dataframe(
    component: zarr.Group,
    *,
    objects_df: pl.DataFrame,
    phases_df: pl.DataFrame,
) -> pl.DataFrame:
    try:
        group = component["distance_cdf"]
    except Exception:
        return pl.DataFrame()
    if objects_df.is_empty() or phases_df.is_empty():
        return pl.DataFrame()

    phase_rows = phases_df.sort("phase_index").to_dicts()
    object_rows = objects_df.sort("object_axis_index").to_dicts()
    thresholds = _load_config_float_array(component, "cdf_thresholds_mm")
    shape = (len(phase_rows), len(object_rows), int(thresholds.shape[0]))
    values = _array_or_full(group, "cdf_fraction", shape=shape, dtype=np.float64)

    rows = []
    for phase_axis, phase in enumerate(phase_rows):
        for object_axis, obj in enumerate(object_rows):
            for threshold_axis, threshold in enumerate(thresholds):
                rows.append(
                    {
                        **phase,
                        **obj,
                        "threshold_index": int(threshold_axis),
                        "threshold_mm": float(threshold),
                        "cdf_fraction": float(values[phase_axis, object_axis, threshold_axis]),
                    }
                )
    return pl.DataFrame(rows)


def _load_cra_near_field_control_reference_dataframes(
    component: zarr.Group,
    *,
    phases_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    try:
        group = component["control_references"]
    except Exception:
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    if phases_df.is_empty():
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    phase_rows = phases_df.sort("phase_index").to_dicts()
    labels = (
        _decode_text_column(np.asarray(group["reference_label_bytes"][:]))
        if "reference_label_bytes" in group
        else []
    )
    ref_x = np.asarray(group["reference_x_px"][:], dtype=np.float64).reshape(-1) if "reference_x_px" in group else np.asarray([], dtype=np.float64)
    ref_y = np.asarray(group["reference_y_px"][:], dtype=np.float64).reshape(-1) if "reference_y_px" in group else np.asarray([], dtype=np.float64)
    n_reference = int(max(len(labels), ref_x.shape[0], ref_y.shape[0]))
    if n_reference == 0:
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    reference_rows = [
        {
            "reference_axis_index": int(index),
            "reference_label": labels[index] if index < len(labels) else f"reference_{index}",
            "reference_x_px": float(ref_x[index]) if index < ref_x.shape[0] else np.nan,
            "reference_y_px": float(ref_y[index]) if index < ref_y.shape[0] else np.nan,
        }
        for index in range(n_reference)
    ]

    percentiles = _load_config_float_array(component, "percentile_values")
    thresholds = _load_config_float_array(component, "cdf_thresholds_mm")
    edges = _load_config_float_array(component, "radial_bin_edges_mm")
    centers = _load_config_float_array(component, "radial_bin_centers_mm")
    n_phase = len(phase_rows)
    n_bin = int(max(0, edges.shape[0] - 1))
    if centers.shape[0] != n_bin and n_bin > 0:
        centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)

    approach = _array_or_full(
        group,
        "approach_percentile_mm",
        shape=(n_phase, n_reference, int(percentiles.shape[0])),
        dtype=np.float64,
    )
    phase_metric_rows = []
    for phase_axis, phase in enumerate(phase_rows):
        for reference_axis, reference in enumerate(reference_rows):
            metrics = {}
            for percentile_axis, percentile in enumerate(percentiles):
                key = _percentile_column_key(float(percentile))
                metrics[f"reference_approach_{key}_mm"] = float(
                    approach[phase_axis, reference_axis, percentile_axis]
                )
            phase_metric_rows.append({**phase, **reference, **metrics})

    radial_shape = (n_phase, n_reference, n_bin)
    radial_count = _array_or_full(group, "radial_count", shape=radial_shape, dtype=np.int64, default=0)
    radial_fraction = _array_or_full(group, "radial_fraction", shape=radial_shape, dtype=np.float64)
    radial_density = _array_or_full(group, "radial_density_per_mm2", shape=radial_shape, dtype=np.float64)
    radial_area = _array_or_full(group, "radial_available_area_mm2", shape=radial_shape, dtype=np.float64)
    radial_rows = []
    for phase_axis, phase in enumerate(phase_rows):
        for reference_axis, reference in enumerate(reference_rows):
            for bin_axis in range(n_bin):
                radial_rows.append(
                    {
                        **phase,
                        **reference,
                        "radial_bin_index": int(bin_axis),
                        "radial_bin_start_mm": float(edges[bin_axis]),
                        "radial_bin_end_mm": float(edges[bin_axis + 1]),
                        "radial_bin_center_mm": float(centers[bin_axis]),
                        "radial_count": int(radial_count[phase_axis, reference_axis, bin_axis]),
                        "radial_fraction": float(radial_fraction[phase_axis, reference_axis, bin_axis]),
                        "radial_density_per_mm2": float(radial_density[phase_axis, reference_axis, bin_axis]),
                        "radial_available_area_mm2": float(radial_area[phase_axis, reference_axis, bin_axis]),
                    }
                )

    cdf_shape = (n_phase, n_reference, int(thresholds.shape[0]))
    cdf_values = _array_or_full(group, "cdf_fraction", shape=cdf_shape, dtype=np.float64)
    cdf_rows = []
    for phase_axis, phase in enumerate(phase_rows):
        for reference_axis, reference in enumerate(reference_rows):
            for threshold_axis, threshold in enumerate(thresholds):
                cdf_rows.append(
                    {
                        **phase,
                        **reference,
                        "threshold_index": int(threshold_axis),
                        "threshold_mm": float(threshold),
                        "cdf_fraction": float(cdf_values[phase_axis, reference_axis, threshold_axis]),
                    }
                )
    return pl.DataFrame(radial_rows), pl.DataFrame(cdf_rows), pl.DataFrame(phase_metric_rows)


def _load_cra_near_field_thigmotaxis_dataframe(
    component: zarr.Group,
    *,
    phases_df: pl.DataFrame,
) -> pl.DataFrame:
    try:
        group = component["thigmotaxis"]
    except Exception:
        return pl.DataFrame()
    if phases_df.is_empty():
        return pl.DataFrame()

    phase_rows = phases_df.sort("phase_index").to_dicts()
    n_phase = len(phase_rows)
    fraction = np.asarray(group["thigmotaxis_fraction"][:], dtype=np.float64).reshape(-1) if "thigmotaxis_fraction" in group else np.full(n_phase, np.nan)
    dwell_s = np.asarray(group["thigmotaxis_dwell_s"][:], dtype=np.float64).reshape(-1) if "thigmotaxis_dwell_s" in group else np.full(n_phase, np.nan)
    mean_speed = np.asarray(group["mean_speed_mm_s"][:], dtype=np.float64).reshape(-1) if "mean_speed_mm_s" in group else np.full(n_phase, np.nan)
    median_speed = np.asarray(group["median_speed_mm_s"][:], dtype=np.float64).reshape(-1) if "median_speed_mm_s" in group else np.full(n_phase, np.nan)
    immobile = np.asarray(group["immobile_fraction"][:], dtype=np.float64).reshape(-1) if "immobile_fraction" in group else np.full(n_phase, np.nan)
    speed_count = np.asarray(group["speed_sample_count"][:], dtype=np.int64).reshape(-1) if "speed_sample_count" in group else np.zeros(n_phase, dtype=np.int64)
    status = (
        _decode_text_column(np.asarray(group["geometry_status_bytes"][:]))[:n_phase]
        if "geometry_status_bytes" in group
        else [str(getattr(group, "attrs", {}).get("geometry_status") or "") for _ in range(n_phase)]
    )
    rows = []
    for phase_axis, phase in enumerate(phase_rows):
        rows.append(
            {
                **phase,
                "thigmotaxis_fraction": float(fraction[phase_axis]) if phase_axis < fraction.shape[0] else np.nan,
                "thigmotaxis_dwell_s": float(dwell_s[phase_axis]) if phase_axis < dwell_s.shape[0] else np.nan,
                "mean_speed_mm_s": float(mean_speed[phase_axis]) if phase_axis < mean_speed.shape[0] else np.nan,
                "median_speed_mm_s": float(median_speed[phase_axis]) if phase_axis < median_speed.shape[0] else np.nan,
                "immobile_fraction": float(immobile[phase_axis]) if phase_axis < immobile.shape[0] else np.nan,
                "speed_sample_count": int(speed_count[phase_axis]) if phase_axis < speed_count.shape[0] else 0,
                "geometry_status": status[phase_axis] if phase_axis < len(status) else "",
            }
        )
    return pl.DataFrame(rows)


def load_goodcopbadcop_cra_primary_endpoint_data(
    zarr_path: Path | str,
    *,
    run_path: str,
    component_name: Optional[str] = None,
) -> Optional[GoodCopBadCopCRAEndpointData]:
    """Load the persisted CRA primary endpoint component for a chaser run."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    normalized_run_path = _normalize_path(run_path)
    if component_name and str(component_name).strip() not in {"latest", ""}:
        component_path = _join_path(normalized_run_path, GOODCOPBADCOP_CRA_COMPONENT_PARENT, str(component_name))
    else:
        component_path = resolve_latest_cra_primary_endpoint_component_path(root, run_path=normalized_run_path)
    if not component_path:
        return None
    try:
        component = root[component_path]
    except Exception:
        return None

    attrs = dict(getattr(component, "attrs", {}))
    schema_id = str(attrs.get("schema_id") or "")
    if schema_id and schema_id != GOODCOPBADCOP_CRA_SCHEMA_ID:
        return None

    run_group = root[normalized_run_path]
    fps = _safe_float(getattr(run_group, "attrs", {}).get("fps"), default=1.0)
    objects_df = _load_cra_objects_dataframe(component)
    phases_df = _load_cra_phases_dataframe(component, fps=fps)
    object_phase_df = _load_cra_object_phase_dataframe(
        component,
        objects_df=objects_df,
        phases_df=phases_df,
    )
    per_object_phase_df = _load_cra_per_object_phase_dataframe(
        component,
        object_phase_df=object_phase_df,
    )
    return GoodCopBadCopCRAEndpointData(
        zarr_path=archive,
        run_path=normalized_run_path,
        component_name=_normalize_path(component_path).split("/")[-1],
        component_path=_normalize_path(component_path),
        attrs=attrs,
        summary=_summary_from_cra_component(component),
        qc_warnings=_qc_warnings_from_cra_component(component),
        fps=fps,
        quadrant_width_px=_safe_float(attrs.get("quadrant_width_px"), default=np.nan),
        quadrant_height_px=_safe_float(attrs.get("quadrant_height_px"), default=np.nan),
        objects_df=objects_df,
        phases_df=phases_df,
        object_phase_df=object_phase_df,
        per_object_phase_df=per_object_phase_df,
    )


def load_goodcopbadcop_cra_near_field_data(
    zarr_path: Path | str,
    *,
    run_path: str,
    component_name: Optional[str] = None,
) -> Optional[GoodCopBadCopCRANearFieldData]:
    """Load the persisted CRA near-field component for a chaser run."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    normalized_run_path = _normalize_path(run_path)
    if component_name and str(component_name).strip() not in {"latest", ""}:
        component_path = _join_path(
            normalized_run_path,
            GOODCOPBADCOP_CRA_NEAR_FIELD_COMPONENT_PARENT,
            str(component_name),
        )
    else:
        component_path = resolve_latest_cra_near_field_component_path(root, run_path=normalized_run_path)
    if not component_path:
        return None
    try:
        component = root[component_path]
    except Exception:
        return None

    attrs = dict(getattr(component, "attrs", {}))
    schema_id = str(attrs.get("schema_id") or "")
    if schema_id and schema_id != GOODCOPBADCOP_CRA_NEAR_FIELD_SCHEMA_ID:
        return None

    run_group = root[normalized_run_path]
    fps = _safe_float(getattr(run_group, "attrs", {}).get("fps"), default=1.0)
    objects_df = _load_cra_objects_dataframe(component)
    phases_df = _load_cra_phases_dataframe(component, fps=fps)
    per_object_phase_df = _load_cra_near_field_per_object_phase_dataframe(
        component,
        objects_df=objects_df,
        phases_df=phases_df,
    )
    radial_density_df = _load_cra_near_field_radial_density_dataframe(
        component,
        objects_df=objects_df,
        phases_df=phases_df,
    )
    cdf_df = _load_cra_near_field_cdf_dataframe(
        component,
        objects_df=objects_df,
        phases_df=phases_df,
    )
    control_reference_radial_density_df, control_reference_cdf_df, control_reference_phase_df = (
        _load_cra_near_field_control_reference_dataframes(
            component,
            phases_df=phases_df,
        )
    )
    thigmotaxis_df = _load_cra_near_field_thigmotaxis_dataframe(
        component,
        phases_df=phases_df,
    )
    parameters = attrs.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    return GoodCopBadCopCRANearFieldData(
        zarr_path=archive,
        run_path=normalized_run_path,
        component_name=_normalize_path(component_path).split("/")[-1],
        component_path=_normalize_path(component_path),
        attrs=attrs,
        parameters=dict(parameters),
        summary=_summary_from_cra_component(component),
        qc_warnings=_qc_warnings_from_cra_component(component),
        fps=fps,
        geometry_status=str(attrs.get("geometry_status") or ""),
        arena_shape=str(attrs.get("arena_shape") or ""),
        arena_geometry_source=str(attrs.get("arena_geometry_source") or "") or None,
        objects_df=objects_df,
        phases_df=phases_df,
        per_object_phase_df=per_object_phase_df,
        radial_density_df=radial_density_df,
        cdf_df=cdf_df,
        control_reference_radial_density_df=control_reference_radial_density_df,
        control_reference_cdf_df=control_reference_cdf_df,
        control_reference_phase_df=control_reference_phase_df,
        thigmotaxis_df=thigmotaxis_df,
    )


def load_goodcopbadcop_epoch_behavior_data(
    zarr_path: Path | str,
    *,
    run_path: str,
    component_name: Optional[str] = None,
) -> Optional[GoodCopBadCopEpochBehaviorData]:
    """Load the persisted GoodCopBadCop epoch behavior summary component."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    normalized_run_path = _normalize_path(run_path)
    if component_name and str(component_name).strip() not in {"latest", ""}:
        component_path = _join_path(
            normalized_run_path,
            GOODCOPBADCOP_EPOCH_BEHAVIOR_COMPONENT_PARENT,
            str(component_name),
        )
    else:
        component_path = resolve_latest_epoch_behavior_summary_component_path(
            root,
            run_path=normalized_run_path,
        )
    if not component_path:
        return None
    try:
        component = root[component_path]
    except Exception:
        return None

    attrs = dict(getattr(component, "attrs", {}))
    schema_id = str(attrs.get("schema_id") or "")
    if schema_id and schema_id != GOODCOPBADCOP_EPOCH_BEHAVIOR_SCHEMA_ID:
        return None

    try:
        fish_records, _fish_attrs = load_structured_dataset(component, "per_epoch_fish")
        chaser_records, _chaser_attrs = load_structured_dataset(component, "per_epoch_chaser")
    except Exception:
        return None
    try:
        center_hist_records, _center_hist_attrs = load_structured_dataset(component, "center_distance_histogram")
    except Exception:
        center_hist_records = np.zeros(0, dtype=[])
    try:
        per_epoch_bout_records, _per_epoch_bout_attrs = load_structured_dataset(component, "per_epoch_bouts")
    except Exception:
        per_epoch_bout_records = np.zeros(0, dtype=[])
    try:
        bout_hist_records, _bout_hist_attrs = load_structured_dataset(component, "per_epoch_bout_histograms")
    except Exception:
        bout_hist_records = np.zeros(0, dtype=[])
    try:
        ibi_hist_records, _ibi_hist_attrs = load_structured_dataset(
            component,
            "per_epoch_inter_bout_interval_histograms",
        )
    except Exception:
        ibi_hist_records = np.zeros(0, dtype=[])

    source_refs = attrs.get("source_refs")
    if not isinstance(source_refs, Mapping):
        source_refs = {}
    parameters = attrs.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    return GoodCopBadCopEpochBehaviorData(
        zarr_path=archive,
        run_path=normalized_run_path,
        component_name=_normalize_path(component_path).split("/")[-1],
        component_path=_normalize_path(component_path),
        attrs=attrs,
        source_refs=dict(source_refs),
        parameters=dict(parameters),
        per_epoch_fish_df=_structured_records_to_polars(fish_records),
        per_epoch_chaser_df=_structured_records_to_polars(chaser_records),
        per_epoch_bouts_df=_structured_records_to_polars(per_epoch_bout_records),
        per_epoch_bout_histograms_df=_structured_records_to_polars(bout_hist_records),
        per_epoch_inter_bout_interval_histograms_df=_structured_records_to_polars(ibi_hist_records),
        center_distance_histogram_df=_structured_records_to_polars(center_hist_records),
    )


def _load_optional_png(root: zarr.Group, artifact_path: str) -> tuple[Optional[str], bytes, Optional[str]]:
    try:
        resolved, payload = load_png_artifact_bytes(root, artifact_path)
        return resolved, payload, None
    except Exception as exc:
        return artifact_path, b"", str(exc)


def load_goodcopbadcop_escape_freeze_data(
    zarr_path: Path | str,
    *,
    run_path: str,
    component_name: Optional[str] = None,
) -> Optional[GoodCopBadCopEscapeFreezeData]:
    """Load the persisted GoodCopBadCop escape/freeze canary component."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    normalized_run_path = _normalize_path(run_path)
    if component_name and str(component_name).strip() not in {"latest", ""}:
        component_path = _join_path(
            normalized_run_path,
            GOODCOPBADCOP_ESCAPE_FREEZE_COMPONENT_PARENT,
            str(component_name),
        )
    else:
        component_path = resolve_latest_escape_freeze_component_path(root, run_path=normalized_run_path)
    if not component_path:
        return None
    try:
        component = root[component_path]
    except Exception:
        return None

    attrs = dict(getattr(component, "attrs", {}))
    schema_id = str(attrs.get("schema_id") or "")
    if schema_id and schema_id != GOODCOPBADCOP_ESCAPE_FREEZE_SCHEMA_ID:
        return None

    parameters = attrs.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    summary = attrs.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    diagnostics = attrs.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    warnings_value = attrs.get("warnings")
    warnings = tuple(str(item) for item in warnings_value) if isinstance(warnings_value, Sequence) and not isinstance(warnings_value, str) else ()

    trials_df = _columnar_group_to_polars(component["trials"]) if "trials" in component else pl.DataFrame()
    metrics_df = _columnar_group_to_polars(component["trial_metrics"]) if "trial_metrics" in component else pl.DataFrame()
    trajectories_df = (
        _columnar_group_to_polars(component["trial_trajectories"])
        if "trial_trajectories" in component
        else pl.DataFrame()
    )
    per_trial_path, per_trial_bytes, per_trial_error = _load_optional_png(
        root,
        _join_path(component_path, "visualizations", GOODCOPBADCOP_ESCAPE_FREEZE_PER_TRIAL_PNG),
    )
    fish_centered_path, fish_centered_bytes, fish_centered_error = _load_optional_png(
        root,
        _join_path(component_path, "visualizations", GOODCOPBADCOP_ESCAPE_FREEZE_FISH_CENTERED_PNG),
    )
    response_class_bar_path, response_class_bar_bytes, response_class_bar_error = _load_optional_png(
        root,
        _join_path(component_path, "visualizations", GOODCOPBADCOP_ESCAPE_FREEZE_RESPONSE_CLASS_BAR_PNG),
    )
    trial_outcome_timeline_path, trial_outcome_timeline_bytes, trial_outcome_timeline_error = _load_optional_png(
        root,
        _join_path(component_path, "visualizations", GOODCOPBADCOP_ESCAPE_FREEZE_TRIAL_OUTCOME_TIMELINE_PNG),
    )
    polar_approach_path, polar_approach_bytes, polar_approach_error = _load_optional_png(
        root,
        _join_path(component_path, "visualizations", GOODCOPBADCOP_ESCAPE_FREEZE_FISH_CENTERED_POLAR_APPROACH_PNG),
    )
    polar_density_path, polar_density_bytes, polar_density_error = _load_optional_png(
        root,
        _join_path(component_path, "visualizations", GOODCOPBADCOP_ESCAPE_FREEZE_FISH_CENTERED_POLAR_DENSITY_PNG),
    )
    scatter_path, scatter_bytes, scatter_error = _load_optional_png(
        root,
        _join_path(component_path, "visualizations", GOODCOPBADCOP_ESCAPE_FREEZE_SCATTER_PNG),
    )
    return GoodCopBadCopEscapeFreezeData(
        zarr_path=archive,
        run_path=normalized_run_path,
        component_name=_normalize_path(component_path).split("/")[-1],
        component_path=_normalize_path(component_path),
        attrs=attrs,
        parameters=dict(parameters),
        summary=dict(summary),
        diagnostics=dict(diagnostics),
        warnings=warnings,
        trials_df=trials_df,
        trial_metrics_df=metrics_df,
        trial_trajectories_df=trajectories_df,
        per_trial_png_path=per_trial_path,
        per_trial_png_bytes=per_trial_bytes,
        per_trial_png_error=per_trial_error,
        fish_centered_png_path=fish_centered_path,
        fish_centered_png_bytes=fish_centered_bytes,
        fish_centered_png_error=fish_centered_error,
        response_class_bar_png_path=response_class_bar_path,
        response_class_bar_png_bytes=response_class_bar_bytes,
        response_class_bar_png_error=response_class_bar_error,
        trial_outcome_timeline_png_path=trial_outcome_timeline_path,
        trial_outcome_timeline_png_bytes=trial_outcome_timeline_bytes,
        trial_outcome_timeline_png_error=trial_outcome_timeline_error,
        fish_centered_polar_approach_png_path=polar_approach_path,
        fish_centered_polar_approach_png_bytes=polar_approach_bytes,
        fish_centered_polar_approach_png_error=polar_approach_error,
        fish_centered_polar_density_png_path=polar_density_path,
        fish_centered_polar_density_png_bytes=polar_density_bytes,
        fish_centered_polar_density_png_error=polar_density_error,
        scatter_png_path=scatter_path,
        scatter_png_bytes=scatter_bytes,
        scatter_png_error=scatter_error,
    )


def _source_paths_for_chaser_dashboard(
    root: zarr.Group,
    *,
    run_path: str,
    detection_occupancy_run_path: Optional[str],
    egocentric_bearing_component_path: Optional[str],
    cra_near_field_component_path: Optional[str] = None,
) -> dict[str, str]:
    source_paths: dict[str, str] = {}

    def add_run_path(key: str, relative_path: str) -> None:
        path = _join_path(run_path, relative_path)
        if _node_exists(root, path):
            source_paths[key] = path

    def add_path(key: str, path: Optional[str]) -> None:
        if path and _node_exists(root, path):
            source_paths[key] = _normalize_path(path)

    add_run_path("camera_frame_id", "frames/camera_frame_id")
    add_run_path("stimulus_frame_num", "frames/stimulus_frame_num")
    add_run_path("timestamp_ns", "frames/timestamp_ns")
    add_run_path("stimulus_epoch_window_id", "frames/stimulus_epoch_window_id")
    add_run_path("chaser_index", "chasers/chaser_index")
    add_run_path("fish_centroid_arena_xy", "positions/fish_centroid_arena_xy")
    add_run_path("fish_centroid_img_xy", "positions/fish_centroid_img_xy")
    add_run_path("fish_valid", "positions/fish_valid")
    add_run_path("chaser_arena_xy", "positions/chaser_arena_xy")
    add_run_path("chaser_valid", "positions/chaser_valid")
    add_run_path("distance_mm", "distances/distance_mm")
    add_run_path("nearest_distance_mm", "distances/nearest_distance_mm")
    add_run_path("nearest_chaser_index", "distances/nearest_chaser_index")
    add_run_path("epoch_window_id", "epoch_summary/window_id")
    add_run_path("epoch_label_bytes", "epoch_summary/label_bytes")
    add_run_path("epoch_start_frame", "epoch_summary/start_frame")
    add_run_path("epoch_end_frame", "epoch_summary/end_frame")
    add_run_path("epoch_p50_distance_mm", "epoch_summary/p50_distance_mm")
    add_run_path("epoch_mean_distance_mm", "epoch_summary/mean_distance_mm")
    add_run_path("epoch_hist_bin_centers_mm", "epoch_distributions/bin_centers_mm")
    add_run_path("epoch_hist_counts", "epoch_distributions/hist_counts")
    add_run_path("epoch_hist_density", "epoch_distributions/hist_density")

    if egocentric_bearing_component_path:
        add_path("egocentric_fish_heading_deg", _join_path(egocentric_bearing_component_path, "frames/fish_heading_deg"))
        add_path(
            "egocentric_fish_heading_valid",
            _join_path(egocentric_bearing_component_path, "frames/fish_heading_valid"),
        )
        add_path("egocentric_chaser_index", _join_path(egocentric_bearing_component_path, "per_chaser/chaser_index"))
        add_path("egocentric_bearing_deg", _join_path(egocentric_bearing_component_path, "per_chaser/bearing_deg"))
        add_path(
            "egocentric_alignment_cos",
            _join_path(egocentric_bearing_component_path, "per_chaser/alignment_cos"),
        )
        add_path("egocentric_lateral_sin", _join_path(egocentric_bearing_component_path, "per_chaser/lateral_sin"))
        add_path("egocentric_valid", _join_path(egocentric_bearing_component_path, "per_chaser/valid"))
        add_path(
            "egocentric_distance_bin_edges_mm",
            _join_path(egocentric_bearing_component_path, "distance_bearing_histogram/distance_bin_edges_mm"),
        )
        add_path(
            "egocentric_distance_bin_centers_mm",
            _join_path(egocentric_bearing_component_path, "distance_bearing_histogram/distance_bin_centers_mm"),
        )
        add_path(
            "egocentric_bearing_bin_edges_deg",
            _join_path(egocentric_bearing_component_path, "distance_bearing_histogram/bearing_bin_edges_deg"),
        )
        add_path(
            "egocentric_bearing_bin_centers_deg",
            _join_path(egocentric_bearing_component_path, "distance_bearing_histogram/bearing_bin_centers_deg"),
        )
        add_path(
            "egocentric_hist_counts",
            _join_path(egocentric_bearing_component_path, "distance_bearing_histogram/hist_counts"),
        )
        add_path(
            "egocentric_hist_probability",
            _join_path(egocentric_bearing_component_path, "distance_bearing_histogram/hist_probability"),
        )

    if cra_near_field_component_path:
        add_path("cra_near_field", cra_near_field_component_path)
        add_path("cra_near_field_config", _join_path(cra_near_field_component_path, "config"))
        add_path("cra_near_field_per_object_phase", _join_path(cra_near_field_component_path, "per_object_phase"))
        add_path("cra_near_field_radial_density", _join_path(cra_near_field_component_path, "radial_density"))
        add_path("cra_near_field_distance_cdf", _join_path(cra_near_field_component_path, "distance_cdf"))
        add_path("cra_near_field_thigmotaxis", _join_path(cra_near_field_component_path, "thigmotaxis"))

    if detection_occupancy_run_path:
        add_path(
            "detection_occupancy_windows_label_bytes",
            _join_path(detection_occupancy_run_path, "windows/label_bytes"),
        )
        add_path(
            "detection_occupancy_windows_start_frame",
            _join_path(detection_occupancy_run_path, "windows/start_frame"),
        )
        add_path("detection_occupancy_windows_end_frame", _join_path(detection_occupancy_run_path, "windows/end_frame"))
        add_path(
            "detection_occupancy_windows_start_time_s",
            _join_path(detection_occupancy_run_path, "windows/start_time_s"),
        )
        add_path(
            "detection_occupancy_windows_end_time_s",
            _join_path(detection_occupancy_run_path, "windows/end_time_s"),
        )
        add_path("detection_occupancy_heatmap_counts", _join_path(detection_occupancy_run_path, "heatmaps/counts"))
        add_path(
            "detection_occupancy_heatmap_normalized",
            _join_path(detection_occupancy_run_path, "heatmaps/normalized"),
        )
        add_path("detection_occupancy_heatmap_x_edges", _join_path(detection_occupancy_run_path, "heatmaps/x_edges"))
        add_path("detection_occupancy_heatmap_y_edges", _join_path(detection_occupancy_run_path, "heatmaps/y_edges"))
        add_path("detection_spatial_occupancy", _join_path(detection_occupancy_run_path, "spatial_occupancy"))

    return source_paths


def _dashboard_artifact_candidates(artifact_name: str | None) -> tuple[str, ...]:
    if artifact_name is None or str(artifact_name) == DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT:
        return CHASER_DASHBOARD_INTERACTIVE_ARTIFACTS
    return (str(artifact_name),)


def build_chaser_protocol_dashboard_spec(
    root: zarr.Group,
    *,
    run_name: str,
    run_path: str,
    run_group: zarr.Group,
    detection_occupancy_run_path: Optional[str] = None,
) -> Mapping[str, Any]:
    """Build a renderer-neutral chaser-protocol dashboard spec."""

    attrs = dict(getattr(run_group, "attrs", {}))
    resolved_occupancy_path = detection_occupancy_run_path or resolve_related_detection_occupancy_run_path(
        root,
        run_group,
    )
    egocentric_component_path = resolve_latest_egocentric_bearing_component_path(root, run_path=run_path)
    egocentric_component_name = (
        _normalize_path(egocentric_component_path).split("/")[-1]
        if egocentric_component_path
        else None
    )
    cra_near_field_component_path = resolve_latest_cra_near_field_component_path(root, run_path=run_path)
    cra_near_field_component_name = (
        _normalize_path(cra_near_field_component_path).split("/")[-1]
        if cra_near_field_component_path
        else None
    )
    source_paths = _source_paths_for_chaser_dashboard(
        root,
        run_path=run_path,
        detection_occupancy_run_path=resolved_occupancy_path,
        egocentric_bearing_component_path=egocentric_component_path,
        cra_near_field_component_path=cra_near_field_component_path,
    )
    source_runs = {
        "chaser_distance": run_name,
        "stimulus": attrs.get("source_stimulus_run"),
        "stimulus_epoch": attrs.get("source_stimulus_epoch_run"),
        "detection_occupancy": _normalize_path(resolved_occupancy_path).split("/")[-1]
        if resolved_occupancy_path
        else None,
        "egocentric_bearing": egocentric_component_name,
        "cra_near_field": cra_near_field_component_name,
    }

    return {
        "schema_id": CHASER_DASHBOARD_SPEC_SCHEMA_ID,
        "renderer": CHASER_DASHBOARD_RENDERER,
        "artifact_family": "chaser_protocol_dashboard",
        "protocol_family": "chaser",
        "title": f"Chaser protocol dashboard - {attrs.get('recording_id', run_name)}",
        "run_name": run_name,
        "run_path": _normalize_path(run_path),
        "recording_id": attrs.get("recording_id"),
        "fps": _safe_float(attrs.get("fps"), default=1.0),
        "total_frames": _safe_int(attrs.get("total_frames"), default=0),
        "coordinate_frame": attrs.get("coordinate_frame"),
        "coordinate_origin": attrs.get("coordinate_origin"),
        "source_paths": source_paths,
        "source_runs": source_runs,
        "static_artifacts": {
            "chaser_distance_timeseries": "visualizations/chaser_distance_timeseries_png",
            "chaser_distance_epoch_median": "visualizations/chaser_distance_epoch_median_png",
            "chaser_distance_epoch_distribution": "visualizations/chaser_distance_epoch_distribution_png",
            "egocentric_bearing_pre_post_polar": (
                _join_path(
                    egocentric_component_path,
                    "visualizations/egocentric_bearing_pre_post_polar_png",
                )
                if egocentric_component_path
                else None
            ),
            "egocentric_bearing_pre_post_polar_point_cloud": (
                _join_path(
                    egocentric_component_path,
                    "visualizations/egocentric_bearing_pre_post_polar_point_cloud_png",
                )
                if egocentric_component_path
                else None
            ),
            "detection_occupancy": (
                _join_path(resolved_occupancy_path, "visualizations/detection_occupancy_overview_png")
                if resolved_occupancy_path
                else None
            ),
            "cra_near_field_radial_density": (
                _join_path(cra_near_field_component_path, "visualizations/cra_near_field_radial_density_png")
                if cra_near_field_component_path
                else None
            ),
            "cra_near_field_distance_cdf": (
                _join_path(cra_near_field_component_path, "visualizations/cra_near_field_distance_cdf_png")
                if cra_near_field_component_path
                else None
            ),
            "cra_near_field_summary": (
                _join_path(cra_near_field_component_path, "visualizations/cra_near_field_summary_png")
                if cra_near_field_component_path
                else None
            ),
        },
        "panels": [
            {
                "id": "distance_timeseries",
                "kind": "timeseries",
                "x_path_key": "camera_frame_id",
                "y_path_key": "distance_mm",
                "series_axis": "chaser",
                "interval_overlay": "stimulus_epoch_windows",
            },
            {
                "id": "selected_window_arena_occupancy",
                "kind": "hist2d",
                "position_path_key": "fish_centroid_arena_xy",
                "valid_path_key": "fish_valid",
                "overlay_path_key": "chaser_arena_xy",
                "unit": "arena_relative_canvas_px",
            },
            {
                "id": "epoch_distance_distribution",
                "kind": "line_distribution",
                "x_path_key": "epoch_hist_bin_centers_mm",
                "y_path_key": "epoch_hist_density",
                "series_axes": ["window", "chaser"],
            },
            {
                "id": "detection_epoch_heatmaps",
                "kind": "heatmap_cube",
                "heatmap_path_key": "detection_occupancy_heatmap_normalized",
                "x_edges_path_key": "detection_occupancy_heatmap_x_edges",
                "y_edges_path_key": "detection_occupancy_heatmap_y_edges",
                "enabled": bool(resolved_occupancy_path),
            },
            {
                "id": "detection_spatial_occupancy",
                "kind": "zone_summary",
                "spatial_occupancy_path_key": "detection_spatial_occupancy",
                "enabled": bool(
                    resolved_occupancy_path
                    and _node_exists(root, _join_path(resolved_occupancy_path, "spatial_occupancy"))
                ),
            },
            {
                "id": "egocentric_chaser_bearing",
                "kind": "polar_scatter",
                "bearing_path_key": "egocentric_bearing_deg",
                "distance_path_key": "distance_mm",
                "valid_path_key": "egocentric_valid",
                "series_axis": "chaser",
                "enabled": bool(egocentric_component_path),
            },
            {
                "id": "egocentric_alignment_by_distance",
                "kind": "distance_binned_line",
                "distance_path_key": "distance_mm",
                "alignment_path_key": "egocentric_alignment_cos",
                "valid_path_key": "egocentric_valid",
                "series_axis": "chaser",
                "enabled": bool(egocentric_component_path),
            },
        ],
    }


def build_chaser_dashboard_spec(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Compatibility wrapper for the shorter pre-protocol-neutral builder name."""

    return build_chaser_protocol_dashboard_spec(*args, **kwargs)


def build_goodcopbadcop_chaser_dashboard_spec(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Compatibility wrapper for the original GoodCopBadCop-specific builder name."""

    return build_chaser_protocol_dashboard_spec(*args, **kwargs)


def discover_chaser_dashboard_options(
    zarr_path: Path | str,
    *,
    artifact_name: str | None = DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
) -> list[GoodCopBadCopRunOption]:
    """Return chaser-distance runs with a persisted chaser dashboard spec."""

    root = open_zarr_root(Path(zarr_path), mode="r")
    parent = root.get("analysis/chaser_distance_runs")
    if parent is None:
        return []
    latest = str(parent.attrs.get("latest_complete") or parent.attrs.get("latest") or "").strip()
    artifact_candidates = _dashboard_artifact_candidates(artifact_name)
    options: list[GoodCopBadCopRunOption] = []
    for run_name in _group_keys(parent):
        run_path = _join_path("analysis/chaser_distance_runs", run_name)
        resolved_artifact_name = None
        artifact = None
        for candidate in artifact_candidates:
            artifact = _try_resolve_interactive_artifact(root, run_path=run_path, artifact_name=candidate)
            if artifact is not None:
                resolved_artifact_name = candidate
                break
        if artifact is None:
            continue
        spec = _json_from_uint8_array(artifact["spec_json"])
        if spec.get("schema_id") not in CHASER_DASHBOARD_SPEC_SCHEMA_IDS:
            continue
        run_group = parent[run_name]
        attrs = dict(getattr(run_group, "attrs", {}))
        summary = attrs.get("summary", {}) if isinstance(attrs.get("summary"), Mapping) else {}
        chaser_count = len(summary.get("chaser_indices", [])) if isinstance(summary, Mapping) else 0
        frame_count = _safe_int(attrs.get("total_frames"), default=0)
        is_latest = bool(latest and latest == run_name)
        suffix = " | latest" if is_latest else ""
        options.append(
            GoodCopBadCopRunOption(
                run_name=run_name,
                run_path=run_path,
                artifact_name=str(resolved_artifact_name or artifact_name or ""),
                label=f"{run_name} | {frame_count:,} frames | {chaser_count} chasers{suffix}",
                is_latest=is_latest,
                attrs=attrs,
                spec=spec,
            )
        )
    return sorted(options, key=lambda item: (not item.is_latest, item.run_name))


def _load_windows(root: zarr.Group, source_paths: Mapping[str, str], fps: float) -> tuple[GoodCopBadCopWindow, ...]:
    ids = _load_array(root, source_paths, "epoch_window_id")
    labels_raw = _load_array(root, source_paths, "epoch_label_bytes")
    starts = _load_array(root, source_paths, "epoch_start_frame")
    ends = _load_array(root, source_paths, "epoch_end_frame")
    if ids is None or labels_raw is None or starts is None or ends is None:
        labels_raw = _load_array(root, source_paths, "detection_occupancy_windows_label_bytes")
        starts = _load_array(root, source_paths, "detection_occupancy_windows_start_frame")
        ends = _load_array(root, source_paths, "detection_occupancy_windows_end_frame")
        if labels_raw is None or starts is None or ends is None:
            return ()
        ids = np.arange(np.asarray(starts).reshape(-1).shape[0], dtype=np.int32)
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    labels = _decode_text_column(np.asarray(labels_raw))
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    ends = np.asarray(ends, dtype=np.int64).reshape(-1)
    n = min(ids.shape[0], len(labels), starts.shape[0], ends.shape[0])
    if n == 0:
        return ()
    start_times = _load_array(root, source_paths, "detection_occupancy_windows_start_time_s")
    end_times = _load_array(root, source_paths, "detection_occupancy_windows_end_time_s")
    if start_times is not None and end_times is not None and len(start_times) >= n and len(end_times) >= n:
        start_s = np.asarray(start_times, dtype=np.float64).reshape(-1)[:n]
        end_s = np.asarray(end_times, dtype=np.float64).reshape(-1)[:n]
    else:
        safe_fps = fps if fps > 0 else 1.0
        start_s = starts[:n].astype(np.float64) / safe_fps
        end_s = (ends[:n].astype(np.float64) + 1.0) / safe_fps
    return tuple(
        GoodCopBadCopWindow(
            window_id=int(ids[i]),
            label=str(labels[i]),
            start_frame=int(starts[i]),
            end_frame=int(ends[i]),
            start_time_s=float(start_s[i]),
            end_time_s=float(end_s[i]),
            duration_s=max(0.0, float(end_s[i]) - float(start_s[i])),
        )
        for i in range(n)
    )


def discover_goodcopbadcop_chaser_dashboard_options(
    zarr_path: Path | str,
    *,
    artifact_name: str | None = DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
) -> list[GoodCopBadCopRunOption]:
    """Compatibility wrapper for the original GoodCopBadCop-specific discovery name."""

    return discover_chaser_dashboard_options(zarr_path, artifact_name=artifact_name)


def load_chaser_dashboard_data(
    zarr_path: Path | str,
    *,
    run_path: str,
    artifact_name: str | None = DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
) -> GoodCopBadCopInteractiveData:
    """Load a persisted chaser dashboard spec and source arrays."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    artifact = None
    for candidate in _dashboard_artifact_candidates(artifact_name):
        artifact = _try_resolve_interactive_artifact(root, run_path=run_path, artifact_name=candidate)
        if artifact is not None:
            break
    if artifact is None:
        wanted = ", ".join(_dashboard_artifact_candidates(artifact_name))
        raise KeyError(f"No chaser dashboard artifact found under {run_path!r}; tried {wanted}")
    spec = _json_from_uint8_array(artifact["spec_json"])
    schema_id = spec.get("schema_id")
    if schema_id not in CHASER_DASHBOARD_SPEC_SCHEMA_IDS:
        raise ValueError(
            f"Unsupported interactive spec schema: {schema_id!r}; "
            f"expected one of {CHASER_DASHBOARD_SPEC_SCHEMA_IDS!r}"
        )
    source_paths = _as_str_mapping(spec.get("source_paths"))
    fps = _safe_float(spec.get("fps"), default=1.0)
    total_frames = _safe_int(spec.get("total_frames"), default=0)

    camera_frame_id = _load_array(root, source_paths, "camera_frame_id", required=True).astype(np.int64, copy=False)
    time_seconds = camera_frame_id.astype(np.float64, copy=False) / (fps if fps > 0 else 1.0)
    stimulus_epoch_window_id = _load_array(root, source_paths, "stimulus_epoch_window_id")
    if stimulus_epoch_window_id is not None:
        stimulus_epoch_window_id = stimulus_epoch_window_id.astype(np.int32, copy=False)

    fish_xy = _load_array(root, source_paths, "fish_centroid_arena_xy", required=True).astype(np.float64, copy=False)
    fish_valid = _load_array(root, source_paths, "fish_valid")
    if fish_valid is None:
        fish_valid = np.isfinite(fish_xy).all(axis=1)
    fish_valid = fish_valid.astype(bool, copy=False).reshape(-1)

    chaser_xy = _load_array(root, source_paths, "chaser_arena_xy")
    if chaser_xy is not None:
        chaser_xy = chaser_xy.astype(np.float64, copy=False)
    chaser_valid = _load_array(root, source_paths, "chaser_valid")
    if chaser_valid is not None:
        chaser_valid = chaser_valid.astype(bool, copy=False)
    chaser_source_img_xy = _load_chaser_source_img_xy(
        root,
        run_path=run_path,
        spec=spec,
        chaser_arena_xy=chaser_xy,
    )

    distance_mm = _load_array(root, source_paths, "distance_mm", required=True).astype(np.float64, copy=False)
    if distance_mm.ndim == 1:
        distance_mm = distance_mm.reshape(-1, 1)
    chaser_indices = _load_array(root, source_paths, "chaser_index")
    if chaser_indices is None:
        chaser_indices = np.arange(distance_mm.shape[1], dtype=np.int32)
    chaser_indices = chaser_indices.astype(np.int32, copy=False).reshape(-1)
    chaser_color_hex = _load_chaser_color_hex(
        root,
        run_path=run_path,
        spec=spec,
        chaser_indices=chaser_indices,
    )

    nearest_distance_mm = _load_array(root, source_paths, "nearest_distance_mm")
    if nearest_distance_mm is not None:
        nearest_distance_mm = nearest_distance_mm.astype(np.float64, copy=False).reshape(-1)
    nearest_chaser_index = _load_array(root, source_paths, "nearest_chaser_index")
    if nearest_chaser_index is not None:
        nearest_chaser_index = nearest_chaser_index.astype(np.int32, copy=False).reshape(-1)

    windows = _load_windows(root, source_paths, fps)
    occupancy_normalized = _load_array(root, source_paths, "detection_occupancy_heatmap_normalized")
    occupancy_counts = _load_array(root, source_paths, "detection_occupancy_heatmap_counts")
    occupancy_x_edges = _load_array(root, source_paths, "detection_occupancy_heatmap_x_edges")
    occupancy_y_edges = _load_array(root, source_paths, "detection_occupancy_heatmap_y_edges")
    spatial_occupancy = _load_spatial_occupancy_zone_sets(root, source_paths)
    egocentric_component_path = _derive_component_path_from_source(
        source_paths,
        "egocentric_bearing_deg",
        "/per_chaser/",
    )
    egocentric_component_name = (
        _normalize_path(egocentric_component_path).split("/")[-1]
        if egocentric_component_path
        else None
    )
    egocentric_fish_heading_deg = _load_array(root, source_paths, "egocentric_fish_heading_deg")
    if egocentric_fish_heading_deg is not None:
        egocentric_fish_heading_deg = egocentric_fish_heading_deg.astype(np.float64, copy=False).reshape(-1)
    egocentric_fish_heading_valid = _load_array(root, source_paths, "egocentric_fish_heading_valid")
    if egocentric_fish_heading_valid is not None:
        egocentric_fish_heading_valid = egocentric_fish_heading_valid.astype(bool, copy=False).reshape(-1)
    egocentric_bearing_deg = _load_array(root, source_paths, "egocentric_bearing_deg")
    if egocentric_bearing_deg is not None:
        egocentric_bearing_deg = egocentric_bearing_deg.astype(np.float64, copy=False)
    egocentric_alignment_cos = _load_array(root, source_paths, "egocentric_alignment_cos")
    if egocentric_alignment_cos is not None:
        egocentric_alignment_cos = egocentric_alignment_cos.astype(np.float64, copy=False)
    egocentric_lateral_sin = _load_array(root, source_paths, "egocentric_lateral_sin")
    if egocentric_lateral_sin is not None:
        egocentric_lateral_sin = egocentric_lateral_sin.astype(np.float64, copy=False)
    egocentric_valid = _load_array(root, source_paths, "egocentric_valid")
    if egocentric_valid is not None:
        egocentric_valid = egocentric_valid.astype(bool, copy=False)
    egocentric_distance_bin_edges_mm = _load_array(root, source_paths, "egocentric_distance_bin_edges_mm")
    if egocentric_distance_bin_edges_mm is not None:
        egocentric_distance_bin_edges_mm = egocentric_distance_bin_edges_mm.astype(np.float64, copy=False).reshape(-1)
    egocentric_distance_bin_centers_mm = _load_array(root, source_paths, "egocentric_distance_bin_centers_mm")
    if egocentric_distance_bin_centers_mm is not None:
        egocentric_distance_bin_centers_mm = egocentric_distance_bin_centers_mm.astype(np.float64, copy=False).reshape(-1)
    egocentric_bearing_bin_edges_deg = _load_array(root, source_paths, "egocentric_bearing_bin_edges_deg")
    if egocentric_bearing_bin_edges_deg is not None:
        egocentric_bearing_bin_edges_deg = egocentric_bearing_bin_edges_deg.astype(np.float64, copy=False).reshape(-1)
    egocentric_bearing_bin_centers_deg = _load_array(root, source_paths, "egocentric_bearing_bin_centers_deg")
    if egocentric_bearing_bin_centers_deg is not None:
        egocentric_bearing_bin_centers_deg = egocentric_bearing_bin_centers_deg.astype(np.float64, copy=False).reshape(-1)
    egocentric_hist_counts = _load_array(root, source_paths, "egocentric_hist_counts")
    if egocentric_hist_counts is not None:
        egocentric_hist_counts = egocentric_hist_counts.astype(np.uint32, copy=False)
    egocentric_hist_probability = _load_array(root, source_paths, "egocentric_hist_probability")
    if egocentric_hist_probability is not None:
        egocentric_hist_probability = egocentric_hist_probability.astype(np.float64, copy=False)

    return GoodCopBadCopInteractiveData(
        zarr_path=archive,
        run_name=str(spec.get("run_name") or _normalize_path(run_path).split("/")[-1]),
        run_path=_normalize_path(run_path),
        artifact_name=artifact_name,
        spec=spec,
        attrs=dict(artifact.attrs),
        source_paths=source_paths,
        fps=fps,
        total_frames=total_frames,
        camera_frame_id=camera_frame_id,
        time_seconds=time_seconds,
        stimulus_epoch_window_id=stimulus_epoch_window_id,
        windows=windows,
        chaser_indices=chaser_indices,
        chaser_color_hex=chaser_color_hex,
        fish_centroid_arena_xy=fish_xy,
        fish_valid=fish_valid,
        chaser_arena_xy=chaser_xy,
        chaser_source_img_xy=chaser_source_img_xy,
        chaser_valid=chaser_valid,
        distance_mm=distance_mm,
        nearest_distance_mm=nearest_distance_mm,
        nearest_chaser_index=nearest_chaser_index,
        occupancy_normalized=occupancy_normalized,
        occupancy_counts=occupancy_counts,
        occupancy_x_edges=occupancy_x_edges,
        occupancy_y_edges=occupancy_y_edges,
        spatial_occupancy=spatial_occupancy,
        egocentric_component_name=egocentric_component_name,
        egocentric_component_path=egocentric_component_path,
        egocentric_fish_heading_deg=egocentric_fish_heading_deg,
        egocentric_fish_heading_valid=egocentric_fish_heading_valid,
        egocentric_bearing_deg=egocentric_bearing_deg,
        egocentric_alignment_cos=egocentric_alignment_cos,
        egocentric_lateral_sin=egocentric_lateral_sin,
        egocentric_valid=egocentric_valid,
        egocentric_distance_bin_edges_mm=egocentric_distance_bin_edges_mm,
        egocentric_distance_bin_centers_mm=egocentric_distance_bin_centers_mm,
        egocentric_bearing_bin_edges_deg=egocentric_bearing_bin_edges_deg,
        egocentric_bearing_bin_centers_deg=egocentric_bearing_bin_centers_deg,
        egocentric_hist_counts=egocentric_hist_counts,
        egocentric_hist_probability=egocentric_hist_probability,
    )


def load_goodcopbadcop_interactive_data(*args: Any, **kwargs: Any) -> GoodCopBadCopInteractiveData:
    """Compatibility wrapper for the original GoodCopBadCop-specific loader name."""

    return load_chaser_dashboard_data(*args, **kwargs)


def to_window_dataframe(data: GoodCopBadCopInteractiveData) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": window.window_id,
                "label": window.label,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "start_time_s": window.start_time_s,
                "end_time_s": window.end_time_s,
                "duration_s": window.duration_s,
            }
            for window in data.windows
        ]
    )


def to_distance_timeseries_dataframe(data: GoodCopBadCopInteractiveData) -> pd.DataFrame:
    n = min(data.time_seconds.shape[0], data.distance_mm.shape[0])
    frame: dict[str, Any] = {
        "time_s": data.time_seconds[:n],
        "frame_index": data.camera_frame_id[:n],
    }
    if data.stimulus_epoch_window_id is not None and data.stimulus_epoch_window_id.shape[0] >= n:
        frame["stimulus_epoch_window_id"] = data.stimulus_epoch_window_id[:n]
    if data.nearest_distance_mm is not None and data.nearest_distance_mm.shape[0] >= n:
        frame["nearest_distance_mm"] = data.nearest_distance_mm[:n]
    if data.nearest_chaser_index is not None and data.nearest_chaser_index.shape[0] >= n:
        frame["nearest_chaser_index"] = data.nearest_chaser_index[:n]
    for col_idx, chaser_index in enumerate(data.chaser_indices.tolist()):
        if col_idx >= data.distance_mm.shape[1]:
            continue
        frame[f"distance_mm_chaser_{int(chaser_index)}"] = data.distance_mm[:n, col_idx]
    return pd.DataFrame(frame)


def to_position_dataframe(data: GoodCopBadCopInteractiveData) -> pd.DataFrame:
    n = min(data.time_seconds.shape[0], data.fish_centroid_arena_xy.shape[0], data.fish_valid.shape[0])
    frame: dict[str, Any] = {
        "time_s": data.time_seconds[:n],
        "frame_index": data.camera_frame_id[:n],
        "x": data.fish_centroid_arena_xy[:n, 0],
        "y": data.fish_centroid_arena_xy[:n, 1],
        "fish_valid": data.fish_valid[:n],
        "unit": "arena_relative_canvas_px",
    }
    if data.nearest_distance_mm is not None and data.nearest_distance_mm.shape[0] >= n:
        frame["nearest_distance_mm"] = data.nearest_distance_mm[:n]
    if data.stimulus_epoch_window_id is not None and data.stimulus_epoch_window_id.shape[0] >= n:
        frame["stimulus_epoch_window_id"] = data.stimulus_epoch_window_id[:n]
    return pd.DataFrame(frame)


def to_chaser_position_dataframe(
    data: GoodCopBadCopInteractiveData,
    *,
    sample_step: int = 1,
) -> pd.DataFrame:
    if data.chaser_arena_xy is None:
        return pd.DataFrame(columns=["time_s", "frame_index", "chaser_index", "x", "y", "chaser_valid"])
    step = max(1, int(sample_step))
    n = min(data.time_seconds.shape[0], data.chaser_arena_xy.shape[0])
    row_indices = np.arange(0, n, step, dtype=np.int64)
    frames: list[pd.DataFrame] = []
    for col_idx, chaser_index in enumerate(data.chaser_indices.tolist()):
        if col_idx >= data.chaser_arena_xy.shape[1]:
            continue
        valid = (
            data.chaser_valid[row_indices, col_idx]
            if data.chaser_valid is not None and data.chaser_valid.shape[0] >= n
            else np.isfinite(data.chaser_arena_xy[row_indices, col_idx, :]).all(axis=1)
        )
        frames.append(
            pd.DataFrame(
                {
                    "time_s": data.time_seconds[row_indices],
                    "frame_index": data.camera_frame_id[row_indices],
                    "chaser_index": int(chaser_index),
                    "x": data.chaser_arena_xy[row_indices, col_idx, 0],
                    "y": data.chaser_arena_xy[row_indices, col_idx, 1],
                    "chaser_valid": valid,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["time_s", "frame_index", "chaser_index", "x", "y", "chaser_valid"])
    return pd.concat(frames, ignore_index=True)


def to_spatial_occupancy_dataframe(data: GoodCopBadCopInteractiveData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zone_set in data.spatial_occupancy:
        n_windows, n_zones = zone_set.frame_count.shape
        for window_idx in range(n_windows):
            window = data.windows[window_idx] if window_idx < len(data.windows) else None
            window_id = window.window_id if window is not None else int(window_idx)
            window_label = window.label if window is not None else f"window_{window_idx}"
            for zone_idx in range(n_zones):
                rows.append(
                    {
                        "zone_set_id": zone_set.zone_set_id,
                        "zone_set_source": zone_set.zone_set_source,
                        "coordinate_frame": zone_set.coordinate_frame,
                        "coordinate_origin": zone_set.coordinate_origin,
                        "x_axis_direction": zone_set.x_axis_direction,
                        "y_axis_direction": zone_set.y_axis_direction,
                        "window_id": int(window_id),
                        "window_label": str(window_label),
                        "window_index": int(window_idx),
                        "zone_index": int(zone_idx),
                        "display_order": int(zone_set.display_order[zone_idx])
                        if zone_idx < zone_set.display_order.shape[0]
                        else int(zone_idx),
                        "zone_id": (
                            str(zone_set.zone_id[zone_idx])
                            if zone_idx < len(zone_set.zone_id)
                            else f"zone_{zone_idx}"
                        ),
                        "zone_label": str(zone_set.zone_label[zone_idx])
                        if zone_idx < len(zone_set.zone_label)
                        else f"zone {zone_idx}",
                        "x_min": float(zone_set.bounds_xyxy[zone_idx, 0]),
                        "y_min": float(zone_set.bounds_xyxy[zone_idx, 1]),
                        "x_max": float(zone_set.bounds_xyxy[zone_idx, 2]),
                        "y_max": float(zone_set.bounds_xyxy[zone_idx, 3]),
                        "frame_count": int(zone_set.frame_count[window_idx, zone_idx]),
                        "time_s": float(zone_set.time_s[window_idx, zone_idx]),
                        "fraction_of_epoch": float(zone_set.fraction_of_epoch[window_idx, zone_idx]),
                        "fraction_of_detected": float(zone_set.fraction_of_detected[window_idx, zone_idx]),
                        "detected_frame_count": int(zone_set.detected_frame_count[window_idx])
                        if window_idx < zone_set.detected_frame_count.shape[0]
                        else 0,
                        "missing_frame_count": int(zone_set.missing_frame_count[window_idx])
                        if window_idx < zone_set.missing_frame_count.shape[0]
                        else 0,
                        "total_span_frames": int(zone_set.total_span_frames[window_idx])
                        if window_idx < zone_set.total_span_frames.shape[0]
                        else 0,
                        "coverage_pct": float(zone_set.coverage_pct[window_idx])
                        if window_idx < zone_set.coverage_pct.shape[0]
                        else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _empty_egocentric_bearing_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "time_s": pl.Float64,
            "frame_index": pl.Int64,
            "stimulus_epoch_window_id": pl.Int32,
            "window_label": pl.Utf8,
            "chaser_index": pl.Int32,
            "distance_mm": pl.Float64,
            "bearing_deg": pl.Float64,
            "alignment_cos": pl.Float64,
            "lateral_sin": pl.Float64,
            "valid": pl.Boolean,
        }
    )


def _empty_egocentric_heading_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "time_s": pl.Float64,
            "frame_index": pl.Int64,
            "stimulus_epoch_window_id": pl.Int32,
            "window_label": pl.Utf8,
            "fish_heading_deg": pl.Float64,
            "fish_heading_valid": pl.Boolean,
        }
    )


def _window_label_by_id(data: GoodCopBadCopInteractiveData) -> dict[int, str]:
    return {int(window.window_id): str(window.label) for window in data.windows}


def to_egocentric_heading_dataframe(
    data: GoodCopBadCopInteractiveData,
    *,
    sample_step: int = 1,
    valid_only: bool = True,
) -> pl.DataFrame:
    """Return fish heading rows from the egocentric component as a Polars DataFrame."""

    if data.egocentric_fish_heading_deg is None or data.egocentric_fish_heading_valid is None:
        return _empty_egocentric_heading_frame()

    step = max(1, int(sample_step))
    n = min(
        data.time_seconds.shape[0],
        data.camera_frame_id.shape[0],
        data.egocentric_fish_heading_deg.shape[0],
        data.egocentric_fish_heading_valid.shape[0],
    )
    if n == 0:
        return _empty_egocentric_heading_frame()

    row_indices = np.arange(0, n, step, dtype=np.int64)
    if data.stimulus_epoch_window_id is not None and data.stimulus_epoch_window_id.shape[0] >= n:
        epoch_ids = data.stimulus_epoch_window_id[row_indices].astype(np.int32, copy=False)
    else:
        epoch_ids = np.full(row_indices.shape[0], -1, dtype=np.int32)
    label_by_id = _window_label_by_id(data)
    epoch_labels = np.asarray([label_by_id.get(int(value), "unassigned") for value in epoch_ids], dtype=object)

    frame = pl.DataFrame(
        {
            "time_s": data.time_seconds[row_indices],
            "frame_index": data.camera_frame_id[row_indices],
            "stimulus_epoch_window_id": epoch_ids,
            "window_label": epoch_labels,
            "fish_heading_deg": data.egocentric_fish_heading_deg[row_indices],
            "fish_heading_valid": data.egocentric_fish_heading_valid[row_indices],
        }
    )
    if valid_only:
        frame = frame.filter(pl.col("fish_heading_valid") & pl.col("fish_heading_deg").is_finite())
    return frame


def to_egocentric_bearing_dataframe(
    data: GoodCopBadCopInteractiveData,
    *,
    sample_step: int = 1,
    valid_only: bool = True,
) -> pl.DataFrame:
    """Return long-form egocentric chaser bearing rows as a Polars DataFrame."""

    if (
        data.egocentric_bearing_deg is None
        or data.egocentric_alignment_cos is None
        or data.egocentric_lateral_sin is None
        or data.egocentric_valid is None
    ):
        return _empty_egocentric_bearing_frame()

    step = max(1, int(sample_step))
    n = min(
        data.time_seconds.shape[0],
        data.camera_frame_id.shape[0],
        data.distance_mm.shape[0],
        data.egocentric_bearing_deg.shape[0],
        data.egocentric_alignment_cos.shape[0],
        data.egocentric_lateral_sin.shape[0],
        data.egocentric_valid.shape[0],
    )
    if n == 0:
        return _empty_egocentric_bearing_frame()

    row_indices = np.arange(0, n, step, dtype=np.int64)
    if data.stimulus_epoch_window_id is not None and data.stimulus_epoch_window_id.shape[0] >= n:
        epoch_ids = data.stimulus_epoch_window_id[row_indices].astype(np.int32, copy=False)
    else:
        epoch_ids = np.full(row_indices.shape[0], -1, dtype=np.int32)
    label_by_id = _window_label_by_id(data)
    epoch_labels = np.asarray([label_by_id.get(int(value), "unassigned") for value in epoch_ids], dtype=object)

    frames: list[pl.DataFrame] = []
    for col_idx, chaser_index in enumerate(data.chaser_indices.tolist()):
        if (
            col_idx >= data.distance_mm.shape[1]
            or col_idx >= data.egocentric_bearing_deg.shape[1]
            or col_idx >= data.egocentric_alignment_cos.shape[1]
            or col_idx >= data.egocentric_lateral_sin.shape[1]
            or col_idx >= data.egocentric_valid.shape[1]
        ):
            continue
        frame = pl.DataFrame(
            {
                "time_s": data.time_seconds[row_indices],
                "frame_index": data.camera_frame_id[row_indices],
                "stimulus_epoch_window_id": epoch_ids,
                "window_label": epoch_labels,
                "chaser_index": np.full(row_indices.shape[0], int(chaser_index), dtype=np.int32),
                "distance_mm": data.distance_mm[row_indices, col_idx],
                "bearing_deg": data.egocentric_bearing_deg[row_indices, col_idx],
                "alignment_cos": data.egocentric_alignment_cos[row_indices, col_idx],
                "lateral_sin": data.egocentric_lateral_sin[row_indices, col_idx],
                "valid": data.egocentric_valid[row_indices, col_idx],
            }
        )
        frames.append(frame)

    if not frames:
        return _empty_egocentric_bearing_frame()
    out = pl.concat(frames, how="vertical")
    if valid_only:
        out = out.filter(
            pl.col("valid")
            & pl.col("distance_mm").is_finite()
            & pl.col("bearing_deg").is_finite()
            & pl.col("alignment_cos").is_finite()
        )
    return out


def to_egocentric_distance_alignment_dataframe(
    data: GoodCopBadCopInteractiveData,
    *,
    distance_bin_width_mm: Optional[float] = None,
) -> pl.DataFrame:
    """Return distance-binned egocentric alignment summaries as Polars rows."""

    frame = to_egocentric_bearing_dataframe(data, valid_only=True)
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "stimulus_epoch_window_id": pl.Int32,
                "window_label": pl.Utf8,
                "chaser_index": pl.Int32,
                "distance_bin_start_mm": pl.Float64,
                "distance_bin_center_mm": pl.Float64,
                "n": pl.UInt32,
                "mean_alignment_cos": pl.Float64,
                "mean_abs_bearing_deg": pl.Float64,
            }
        )

    width = distance_bin_width_mm
    if width is None and data.egocentric_distance_bin_edges_mm is not None and data.egocentric_distance_bin_edges_mm.size > 1:
        width = float(data.egocentric_distance_bin_edges_mm[1] - data.egocentric_distance_bin_edges_mm[0])
    width = float(width if width is not None and np.isfinite(width) and width > 0 else 2.0)

    return (
        frame.with_columns(
            ((pl.col("distance_mm") / width).floor() * width).alias("distance_bin_start_mm")
        )
        .group_by(["stimulus_epoch_window_id", "window_label", "chaser_index", "distance_bin_start_mm"])
        .agg(
            pl.len().cast(pl.UInt32).alias("n"),
            pl.col("alignment_cos").mean().alias("mean_alignment_cos"),
            pl.col("bearing_deg").abs().mean().alias("mean_abs_bearing_deg"),
        )
        .with_columns((pl.col("distance_bin_start_mm") + (width / 2.0)).alias("distance_bin_center_mm"))
        .sort(["stimulus_epoch_window_id", "chaser_index", "distance_bin_start_mm"])
    )
