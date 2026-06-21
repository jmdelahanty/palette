"""Interactive visualization adapters for GoodCopBadCop chaser analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import zarr

from fisheye.shared.coordinate_transform import load_calibration_transform, projector_to_camera_px
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.utils.zarr_io import open_zarr_root


GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID = "palette.plot_spec.goodcopbadcop_chaser_dashboard.v1"
GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER = "palette-goodcopbadcop-chaser-dashboard-v1"
DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT = "goodcopbadcop_chaser_dashboard_interactive"


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


def _source_paths_for_chaser_dashboard(
    root: zarr.Group,
    *,
    run_path: str,
    detection_occupancy_run_path: Optional[str],
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


def build_goodcopbadcop_chaser_dashboard_spec(
    root: zarr.Group,
    *,
    run_name: str,
    run_path: str,
    run_group: zarr.Group,
    detection_occupancy_run_path: Optional[str] = None,
) -> Mapping[str, Any]:
    """Build a renderer-neutral GoodCopBadCop chaser dashboard spec."""

    attrs = dict(getattr(run_group, "attrs", {}))
    resolved_occupancy_path = detection_occupancy_run_path or resolve_related_detection_occupancy_run_path(
        root,
        run_group,
    )
    source_paths = _source_paths_for_chaser_dashboard(
        root,
        run_path=run_path,
        detection_occupancy_run_path=resolved_occupancy_path,
    )
    source_runs = {
        "chaser_distance": run_name,
        "stimulus": attrs.get("source_stimulus_run"),
        "stimulus_epoch": attrs.get("source_stimulus_epoch_run"),
        "detection_occupancy": _normalize_path(resolved_occupancy_path).split("/")[-1]
        if resolved_occupancy_path
        else None,
    }

    return {
        "schema_id": GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID,
        "renderer": GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
        "artifact_family": "goodcopbadcop_chaser_dashboard",
        "title": f"GoodCopBadCop chaser dashboard - {attrs.get('recording_id', run_name)}",
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
            "detection_occupancy": (
                _join_path(resolved_occupancy_path, "visualizations/detection_occupancy_overview_png")
                if resolved_occupancy_path
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
        ],
    }


def discover_goodcopbadcop_chaser_dashboard_options(
    zarr_path: Path | str,
    *,
    artifact_name: str = DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
) -> list[GoodCopBadCopRunOption]:
    """Return chaser-distance runs with a GoodCopBadCop interactive spec."""

    root = open_zarr_root(Path(zarr_path), mode="r")
    parent = root.get("analysis/chaser_distance_runs")
    if parent is None:
        return []
    latest = str(parent.attrs.get("latest_complete") or parent.attrs.get("latest") or "").strip()
    options: list[GoodCopBadCopRunOption] = []
    for run_name in _group_keys(parent):
        run_path = _join_path("analysis/chaser_distance_runs", run_name)
        artifact = _try_resolve_interactive_artifact(root, run_path=run_path, artifact_name=artifact_name)
        if artifact is None:
            continue
        spec = _json_from_uint8_array(artifact["spec_json"])
        if spec.get("schema_id") != GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID:
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
                artifact_name=artifact_name,
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


def load_goodcopbadcop_interactive_data(
    zarr_path: Path | str,
    *,
    run_path: str,
    artifact_name: str = DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
) -> GoodCopBadCopInteractiveData:
    """Load a persisted GoodCopBadCop chaser dashboard spec and source arrays."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    artifact = _resolve_interactive_artifact(root, run_path=run_path, artifact_name=artifact_name)
    spec = _json_from_uint8_array(artifact["spec_json"])
    schema_id = spec.get("schema_id")
    if schema_id != GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID:
        raise ValueError(
            f"Unsupported interactive spec schema: {schema_id!r}; "
            f"expected {GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID!r}"
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
    )


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
