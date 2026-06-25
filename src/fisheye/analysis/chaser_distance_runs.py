"""Materialize offline fish-to-chaser distances inside analysis zarrs."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_state_interpolator import load_structured_dataset
from fisheye.analysis.detection_occupancy_runs import _read_epoch_windows, _resolve_epoch_run
from fisheye.shared.coordinate_transform import load_calibration_transform, projector_to_camera_px
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.plot_artifacts import write_interactive_plot_spec_artifact, write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import (
    build_run_lineage_payload,
    write_run_lineage_attrs,
)
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_failed,
    mark_run_pending,
    mark_run_started,
    require_runs_parent,
    resolve_latest_complete_run_name,
)
from fisheye.utils.system import get_git_info
from fisheye.visualization.plot_detection_epoch_heatmaps import (
    _read_detection_centers,
    _read_scores,
    _resolve_detection_group,
    _resolve_dimensions,
)
from fisheye.visualization.goodcopbadcop_interactive import (
    DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
    GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
    GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID,
    build_goodcopbadcop_chaser_dashboard_spec,
)


SCHEMA_ID = "palette.chaser_distance.v1"
SCHEMA_VERSION = 1
METHOD = "offline_detection_to_chaser_distance"
METHOD_VERSION = "1"
PARENT_NAME = "chaser_distance_runs"
TIMESERIES_PNG_ARTIFACT_NAME = "chaser_distance_timeseries_png"
EPOCH_MEDIAN_PNG_ARTIFACT_NAME = "chaser_distance_epoch_median_png"
EPOCH_DISTRIBUTION_PNG_ARTIFACT_NAME = "chaser_distance_epoch_distribution_png"
PNG_ARTIFACT_NAME = TIMESERIES_PNG_ARTIFACT_NAME


def _artifact_signature(payload: Mapping[str, Any]) -> str:
    safe_payload = json_attr_safe(payload)
    encoded = json.dumps(safe_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_epoch_label(label: str) -> str:
    text = str(label).replace("_event", "").replace("_", " ").strip()
    return text or str(label)


@dataclass(frozen=True)
class ChaserDistanceWindow:
    window_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float


@dataclass(frozen=True)
class ChaserDistanceResult:
    zarr_path: str
    recording_id: str
    run_name: str
    source_detection_path: str
    source_detection_kind: str
    source_stimulus_run: str
    source_stimulus_path: str
    source_stimulus_epoch_run: Optional[str]
    source_stimulus_epoch_path: Optional[str]
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    coordinate_frame: str
    coordinate_origin: str
    arena_origin_in_canvas_xy: tuple[float, float]
    chaser_indices: np.ndarray
    camera_frame_id: np.ndarray
    stimulus_frame_num: np.ndarray
    timestamp_ns: np.ndarray
    stimulus_epoch_window_id: np.ndarray
    fish_centroid_img_xy: np.ndarray
    fish_centroid_arena_xy: np.ndarray
    chaser_arena_xy: np.ndarray
    fish_valid: np.ndarray
    chaser_valid: np.ndarray
    distance_px: np.ndarray
    distance_mm: np.ndarray
    nearest_chaser_index: np.ndarray
    nearest_distance_mm: np.ndarray
    windows: tuple[ChaserDistanceWindow, ...]
    epoch_valid_frame_count: np.ndarray
    epoch_mean_distance_mm: np.ndarray
    epoch_min_distance_mm: np.ndarray
    epoch_p05_distance_mm: np.ndarray
    epoch_p50_distance_mm: np.ndarray
    epoch_p95_distance_mm: np.ndarray
    epoch_fraction_within_threshold: np.ndarray
    threshold_mm: float
    distribution_bin_width_mm: float
    histogram_bin_edges_mm: np.ndarray
    histogram_bin_centers_mm: np.ndarray
    histogram_counts: np.ndarray
    histogram_density: np.ndarray


def utc_run_name(prefix: str = "chaser_distance") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _attr_text(attrs: Any, *keys: str) -> Optional[str]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _attr_float(attrs: Any, *keys: str) -> Optional[float]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _resolve_stimulus_run(root: zarr.Group, stimulus_run: Optional[str]) -> tuple[zarr.Group, str, str]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise ValueError("Archive has no analysis/stimulus_runs group.")
    parent = analysis["stimulus_runs"]
    resolved = str(stimulus_run).strip() if stimulus_run else None
    if not resolved:
        resolved = resolve_latest_complete_run_name(parent)
    if not resolved:
        latest = parent.attrs.get("latest")
        resolved = str(latest).strip() if latest else None
    if not resolved or resolved not in parent:
        raise ValueError("No usable stimulus run found; pass --stimulus-run.")
    return parent[resolved], resolved, f"analysis/stimulus_runs/{resolved}"


def _field_name(array: np.ndarray, *candidates: str, required: bool = True) -> Optional[str]:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    if required:
        raise ValueError(f"Structured array missing expected field: {candidates}")
    return None


def _load_frame_alignment(stim_group: zarr.Group, total_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    meta_group = stim_group.get("video_metadata")
    if meta_group is None:
        raise ValueError("Stimulus run has no video_metadata group.")

    dataset_name = "camera_aligned_frame_metadata" if "camera_aligned_frame_metadata" in meta_group else "frame_metadata"
    metadata, _ = load_structured_dataset(meta_group, dataset_name)
    stim_field = _field_name(metadata, "stimulus_frame_num", "frame_number", "stim_frame_num")
    camera_field = _field_name(metadata, "triggering_camera_frame_id", "camera_frame_id")
    timestamp_field = _field_name(
        metadata,
        "timestamp_ns",
        "timestamp_ns_session",
        "relative_timestamp_ns",
        "timestamp_ns_epoch",
        required=False,
    )

    frame_axis = np.arange(int(total_frames), dtype=np.int64)
    stimulus_frame_num = np.full(int(total_frames), -1, dtype=np.int64)
    timestamp_ns = np.full(int(total_frames), -1, dtype=np.int64)
    stim_to_camera: dict[int, int] = {}

    stim_values = np.asarray(metadata[stim_field], dtype=np.int64)
    camera_values = np.asarray(metadata[camera_field], dtype=np.int64)
    timestamp_values = (
        np.asarray(metadata[timestamp_field], dtype=np.int64)
        if timestamp_field is not None
        else np.full(stim_values.shape, -1, dtype=np.int64)
    )

    for stim, camera, timestamp in zip(stim_values, camera_values, timestamp_values):
        camera_i = int(camera)
        stim_i = int(stim)
        stim_to_camera.setdefault(stim_i, camera_i)
        if 0 <= camera_i < total_frames:
            stimulus_frame_num[camera_i] = stim_i
            timestamp_ns[camera_i] = int(timestamp)
    return frame_axis, stimulus_frame_num, timestamp_ns, stim_to_camera


def _arena_geometry_attrs(stim_group: zarr.Group) -> Mapping[str, Any]:
    calibration = stim_group.get("calibration")
    if calibration is not None and "arena_geometry" in calibration:
        return calibration["arena_geometry"].attrs
    return {}


def _resolve_arena_origin(stim_group: zarr.Group) -> tuple[float, float]:
    attrs = _arena_geometry_attrs(stim_group)
    x = _attr_float(attrs, "arena_origin_in_canvas_x_px")
    y = _attr_float(attrs, "arena_origin_in_canvas_y_px")
    if x is None or y is None:
        raise ValueError("Stimulus run calibration/arena_geometry lacks arena_origin_in_canvas_{x,y}_px.")
    return float(x), float(y)


def _resolve_pixels_per_mm_projector(root: zarr.Group, chaser_states: np.ndarray) -> float:
    analysis = root.get("analysis")
    calibration = analysis.get("calibration") if analysis is not None else None
    if calibration is not None:
        value = _attr_float(calibration.attrs, "pixels_per_mm_projector")
        if value is not None and value > 0:
            return float(value)
    if chaser_states.dtype.names and "pixels_per_mm" in chaser_states.dtype.names:
        values = np.asarray(chaser_states["pixels_per_mm"], dtype=np.float64)
        finite = values[np.isfinite(values) & (values > 0)]
        if finite.size:
            return float(np.median(finite))
    raise ValueError("Unable to resolve pixels_per_mm_projector for chaser-distance conversion.")


def _camera_to_arena_xy(
    points_img_xy: np.ndarray,
    *,
    homography_camera_to_canvas: np.ndarray,
    arena_origin_in_canvas_xy: tuple[float, float],
) -> np.ndarray:
    # Current Citrus calibration snapshots store the matrix that maps camera
    # pixels into projector/canvas pixels. The helper applies an arbitrary
    # 3x3 homogeneous transform despite its historical projector->camera name.
    canvas_xy = projector_to_camera_px(points_img_xy, np.asarray(homography_camera_to_canvas, dtype=np.float64))
    origin = np.asarray(arena_origin_in_canvas_xy, dtype=np.float64).reshape(1, 2)
    return canvas_xy - origin


def _dense_fish_positions(
    frames: np.ndarray,
    centers_img_xy: np.ndarray,
    scores: Optional[np.ndarray],
    *,
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    fish = np.full((int(total_frames), 2), np.nan, dtype=np.float32)
    valid = np.zeros(int(total_frames), dtype=bool)
    best_score = np.full(int(total_frames), -np.inf, dtype=np.float64)

    score_values = (
        np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores is not None
        else np.zeros(np.asarray(frames).reshape(-1).shape[0], dtype=np.float64)
    )
    for frame, center, score in zip(np.asarray(frames, dtype=np.int64), centers_img_xy, score_values):
        frame_i = int(frame)
        if frame_i < 0 or frame_i >= total_frames:
            continue
        if not np.isfinite(center).all():
            continue
        if not valid[frame_i] or float(score) >= best_score[frame_i]:
            fish[frame_i] = np.asarray(center, dtype=np.float32)
            valid[frame_i] = True
            best_score[frame_i] = float(score)
    return fish, valid


def _dense_chaser_positions(
    chaser_states: np.ndarray,
    *,
    stim_to_camera: Mapping[int, int],
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    dtype_names = chaser_states.dtype.names or ()
    stim_field = _field_name(chaser_states, "stimulus_frame_num", "frame_number", "stim_frame_num", required=False)
    camera_field = _field_name(chaser_states, "camera_frame_id", "triggering_camera_frame_id", required=False)
    if camera_field is None and stim_field is None:
        raise ValueError("chaser_states must include camera_frame_id or stimulus_frame_num.")

    x_field = _field_name(chaser_states, "chaser_pos_x", "pos_x_px", "target_clamped_pos_x")
    y_field = _field_name(chaser_states, "chaser_pos_y", "pos_y_px", "target_clamped_pos_y")
    index_field = _field_name(chaser_states, "chaser_index", required=False)

    if index_field is None:
        chaser_indices = np.asarray([0], dtype=np.uint8)
        row_indices = np.zeros(chaser_states.shape[0], dtype=np.int64)
    else:
        raw_indices = np.asarray(chaser_states[index_field], dtype=np.int64).reshape(-1)
        chaser_indices = np.asarray(sorted(int(v) for v in np.unique(raw_indices)), dtype=np.uint8)
        index_to_col = {int(value): idx for idx, value in enumerate(chaser_indices.tolist())}
        row_indices = np.asarray([index_to_col[int(value)] for value in raw_indices], dtype=np.int64)

    chaser_xy = np.full((int(total_frames), int(chaser_indices.shape[0]), 2), np.nan, dtype=np.float32)
    valid = np.zeros((int(total_frames), int(chaser_indices.shape[0])), dtype=bool)

    if camera_field is not None:
        camera_frames = np.asarray(chaser_states[camera_field], dtype=np.int64)
    else:
        camera_frames = np.asarray(
            [stim_to_camera.get(int(stim), -1) for stim in np.asarray(chaser_states[stim_field], dtype=np.int64)],
            dtype=np.int64,
        )

    x = np.asarray(chaser_states[x_field], dtype=np.float64)
    y = np.asarray(chaser_states[y_field], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    for frame, chaser_col, x_value, y_value, is_finite in zip(camera_frames, row_indices, x, y, finite):
        if not is_finite:
            continue
        frame_i = int(frame)
        if frame_i < 0 or frame_i >= total_frames:
            continue
        col_i = int(chaser_col)
        chaser_xy[frame_i, col_i, 0] = float(x_value)
        chaser_xy[frame_i, col_i, 1] = float(y_value)
        valid[frame_i, col_i] = True

    attrs = getattr(chaser_states, "attrs", {})
    coordinate_frame = str(getattr(attrs, "get", lambda _k, _d=None: _d)("coordinate_frame", "unknown"))
    coordinate_origin = str(getattr(attrs, "get", lambda _k, _d=None: _d)("coordinate_origin", "unknown"))
    return chaser_indices, chaser_xy, valid, coordinate_frame, coordinate_origin


def _dense_chaser_positions_from_group(
    chaser_group: zarr.Group,
    *,
    chaser_states: np.ndarray,
    stim_to_camera: Mapping[int, int],
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    chaser_indices, chaser_xy, valid, _frame, _origin = _dense_chaser_positions(
        chaser_states,
        stim_to_camera=stim_to_camera,
        total_frames=total_frames,
    )
    coordinate_frame = _attr_text(chaser_group.attrs, "coordinate_frame") or "unknown"
    coordinate_origin = _attr_text(chaser_group.attrs, "coordinate_origin") or "unknown"
    return chaser_indices, chaser_xy, valid, coordinate_frame, coordinate_origin


def _assign_epoch_ids(total_frames: int, windows: Sequence[ChaserDistanceWindow]) -> np.ndarray:
    out = np.full(int(total_frames), -1, dtype=np.int32)
    for window in windows:
        start = max(0, int(window.start_frame))
        end = min(int(total_frames) - 1, int(window.end_frame))
        if end >= start:
            out[start : end + 1] = int(window.window_id)
    return out


def _summarize_epochs(
    distance_mm: np.ndarray,
    fish_valid: np.ndarray,
    chaser_valid: np.ndarray,
    *,
    windows: Sequence[ChaserDistanceWindow],
    threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_windows = len(windows)
    n_chasers = int(distance_mm.shape[1])
    counts = np.zeros((n_windows, n_chasers), dtype=np.int64)
    mean = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    min_v = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    p05 = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    p50 = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    p95 = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)
    frac = np.full((n_windows, n_chasers), np.nan, dtype=np.float32)

    for w_idx, window in enumerate(windows):
        start = max(0, int(window.start_frame))
        end = min(distance_mm.shape[0] - 1, int(window.end_frame))
        if end < start:
            continue
        for c_idx in range(n_chasers):
            mask = (
                fish_valid[start : end + 1]
                & chaser_valid[start : end + 1, c_idx]
                & np.isfinite(distance_mm[start : end + 1, c_idx])
            )
            values = distance_mm[start : end + 1, c_idx][mask]
            if values.size == 0:
                continue
            counts[w_idx, c_idx] = int(values.size)
            mean[w_idx, c_idx] = float(np.mean(values))
            min_v[w_idx, c_idx] = float(np.min(values))
            p05[w_idx, c_idx] = float(np.percentile(values, 5))
            p50[w_idx, c_idx] = float(np.percentile(values, 50))
            p95[w_idx, c_idx] = float(np.percentile(values, 95))
            frac[w_idx, c_idx] = float(np.mean(values <= float(threshold_mm)))
    return counts, mean, min_v, p05, p50, p95, frac


def _compute_epoch_distributions(
    distance_mm: np.ndarray,
    fish_valid: np.ndarray,
    chaser_valid: np.ndarray,
    *,
    windows: Sequence[ChaserDistanceWindow],
    bin_width_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_width = float(bin_width_mm)
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("distribution_bin_width_mm must be positive.")
    finite = distance_mm[np.isfinite(distance_mm)]
    max_distance = float(np.max(finite)) if finite.size else bin_width
    max_edge = max(bin_width, float(np.ceil(max_distance / bin_width) * bin_width))
    bin_edges = np.arange(0.0, max_edge + bin_width * 0.5, bin_width, dtype=np.float32)
    if bin_edges.shape[0] < 2:
        bin_edges = np.asarray([0.0, bin_width], dtype=np.float32)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).astype(np.float32)
    n_windows = len(windows)
    n_chasers = int(distance_mm.shape[1])
    n_bins = int(bin_centers.shape[0])
    counts = np.zeros((n_windows, n_chasers, n_bins), dtype=np.uint32)
    density = np.zeros((n_windows, n_chasers, n_bins), dtype=np.float32)

    for w_idx, window in enumerate(windows):
        start = max(0, int(window.start_frame))
        end = min(distance_mm.shape[0] - 1, int(window.end_frame))
        if end < start:
            continue
        for c_idx in range(n_chasers):
            mask = (
                fish_valid[start : end + 1]
                & chaser_valid[start : end + 1, c_idx]
                & np.isfinite(distance_mm[start : end + 1, c_idx])
            )
            values = distance_mm[start : end + 1, c_idx][mask]
            if values.size == 0:
                continue
            hist, _edges = np.histogram(values, bins=bin_edges)
            counts[w_idx, c_idx, :] = hist.astype(np.uint32, copy=False)
            density[w_idx, c_idx, :] = hist.astype(np.float32) / (float(values.size) * bin_width)
    return bin_edges, bin_centers, counts, density


def _windows_from_epoch_run(root: zarr.Group, epoch_run: Optional[str]) -> tuple[Optional[str], Optional[str], tuple[ChaserDistanceWindow, ...]]:
    try:
        epoch_group, resolved, path = _resolve_epoch_run(root, epoch_run)
    except ValueError:
        return None, None, ()
    windows = tuple(
        ChaserDistanceWindow(
            window_id=w.window_id,
            label=w.label,
            start_frame=w.start_frame,
            end_frame=w.end_frame,
            start_time_s=w.start_time_s,
            end_time_s=w.end_time_s,
            duration_s=w.duration_s,
        )
        for w in _read_epoch_windows(epoch_group)
    )
    return resolved, path, windows


def build_chaser_distance_result(
    zarr_path: Path,
    *,
    run_name: str,
    stimulus_run: Optional[str] = None,
    stimulus_epoch_run: Optional[str] = None,
    source: str = "active",
    detect_run: Optional[str] = None,
    detection_path: Optional[str] = None,
    threshold_mm: float = 20.0,
    distribution_bin_width_mm: float = 2.0,
) -> ChaserDistanceResult:
    root = _open_root(zarr_path, mode="r")
    stim_group, stim_name, stim_path = _resolve_stimulus_run(root, stimulus_run)
    detect_group, source_detection_path, source_detection_kind = _resolve_detection_group(
        root,
        source=source,
        detect_run=detect_run,
        detection_path=detection_path,
    )
    width, height, fps, total_frames_raw = _resolve_dimensions(root, detect_group)
    total_frames = int(total_frames_raw) if int(total_frames_raw) > 0 else int(max(np.asarray(detect_group["frame_indices"][:])) + 1)

    frames, centers_img = _read_detection_centers(detect_group, width=width, height=height)
    scores = _read_scores(detect_group)
    fish_img, fish_valid = _dense_fish_positions(frames, centers_img, scores, total_frames=total_frames)

    camera_frame_id, stimulus_frame_num, timestamp_ns, stim_to_camera = _load_frame_alignment(stim_group, total_frames)
    chaser_group = stim_group["tracking_data"]["chaser_states"]
    chaser_states, _ = load_structured_dataset(stim_group["tracking_data"], "chaser_states")
    chaser_indices, chaser_xy, chaser_valid, coordinate_frame, coordinate_origin = _dense_chaser_positions_from_group(
        chaser_group,
        chaser_states=chaser_states,
        stim_to_camera=stim_to_camera,
        total_frames=total_frames,
    )

    if coordinate_frame != "arena_relative_canvas_px":
        raise ValueError(
            "Only coordinate_frame='arena_relative_canvas_px' is currently supported for chaser_distance_runs; "
            f"found {coordinate_frame!r}."
        )

    arena_origin = _resolve_arena_origin(stim_group)
    calibration = load_calibration_transform(root, stimulus_run=stim_name)
    homography = calibration.get("homography")
    if homography is None:
        raise ValueError("Calibration homography_matrix is required to transform detection centers into arena coordinates.")
    ppm_projector = _resolve_pixels_per_mm_projector(root, chaser_states)

    fish_arena = np.full_like(fish_img, np.nan, dtype=np.float32)
    valid_rows = fish_valid & np.isfinite(fish_img).all(axis=1)
    if np.any(valid_rows):
        fish_arena[valid_rows] = _camera_to_arena_xy(
            fish_img[valid_rows].astype(np.float64),
            homography_camera_to_canvas=np.asarray(homography, dtype=np.float64),
            arena_origin_in_canvas_xy=arena_origin,
        ).astype(np.float32)

    distance_px = np.full((total_frames, int(chaser_indices.shape[0])), np.nan, dtype=np.float32)
    for c_idx in range(int(chaser_indices.shape[0])):
        valid = fish_valid & chaser_valid[:, c_idx] & np.isfinite(fish_arena).all(axis=1)
        delta = chaser_xy[:, c_idx, :].astype(np.float32) - fish_arena
        distance_px[valid, c_idx] = np.linalg.norm(delta[valid], axis=1).astype(np.float32)
    distance_mm = (distance_px / float(ppm_projector)).astype(np.float32)

    nearest_chaser_index = np.full(total_frames, -1, dtype=np.int16)
    nearest_distance_mm = np.full(total_frames, np.nan, dtype=np.float32)
    finite_any = np.isfinite(distance_mm).any(axis=1)
    if np.any(finite_any):
        filled = np.where(np.isfinite(distance_mm), distance_mm, np.inf)
        nearest_col = np.argmin(filled[finite_any], axis=1)
        nearest_chaser_index[finite_any] = chaser_indices[nearest_col].astype(np.int16)
        nearest_distance_mm[finite_any] = filled[finite_any, nearest_col].astype(np.float32)

    epoch_run_name, epoch_path, windows = _windows_from_epoch_run(root, stimulus_epoch_run)
    stimulus_epoch_window_id = _assign_epoch_ids(total_frames, windows) if windows else np.full(total_frames, -1, dtype=np.int32)
    summaries = _summarize_epochs(
        distance_mm,
        fish_valid,
        chaser_valid,
        windows=windows,
        threshold_mm=threshold_mm,
    )
    distributions = _compute_epoch_distributions(
        distance_mm,
        fish_valid,
        chaser_valid,
        windows=windows,
        bin_width_mm=float(distribution_bin_width_mm),
    )
    recording_id = _attr_text(root.attrs, "recording_id", "recording_name") or Path(zarr_path).stem

    return ChaserDistanceResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        run_name=run_name,
        source_detection_path=source_detection_path,
        source_detection_kind=source_detection_kind,
        source_stimulus_run=stim_name,
        source_stimulus_path=stim_path,
        source_stimulus_epoch_run=epoch_run_name,
        source_stimulus_epoch_path=epoch_path,
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(ppm_projector),
        coordinate_frame=coordinate_frame,
        coordinate_origin=coordinate_origin,
        arena_origin_in_canvas_xy=arena_origin,
        chaser_indices=chaser_indices.astype(np.uint8),
        camera_frame_id=camera_frame_id,
        stimulus_frame_num=stimulus_frame_num,
        timestamp_ns=timestamp_ns,
        stimulus_epoch_window_id=stimulus_epoch_window_id,
        fish_centroid_img_xy=fish_img.astype(np.float32),
        fish_centroid_arena_xy=fish_arena.astype(np.float32),
        chaser_arena_xy=chaser_xy.astype(np.float32),
        fish_valid=fish_valid,
        chaser_valid=chaser_valid,
        distance_px=distance_px,
        distance_mm=distance_mm,
        nearest_chaser_index=nearest_chaser_index,
        nearest_distance_mm=nearest_distance_mm,
        windows=windows,
        epoch_valid_frame_count=summaries[0],
        epoch_mean_distance_mm=summaries[1],
        epoch_min_distance_mm=summaries[2],
        epoch_p05_distance_mm=summaries[3],
        epoch_p50_distance_mm=summaries[4],
        epoch_p95_distance_mm=summaries[5],
        epoch_fraction_within_threshold=summaries[6],
        threshold_mm=float(threshold_mm),
        distribution_bin_width_mm=float(distribution_bin_width_mm),
        histogram_bin_edges_mm=distributions[0],
        histogram_bin_centers_mm=distributions[1],
        histogram_counts=distributions[2],
        histogram_density=distributions[3],
    )


def _bytes_array(values: Sequence[str], *, width: int = 96) -> np.ndarray:
    out = np.zeros((len(values), int(width)), dtype=np.uint8)
    for row_idx, value in enumerate(values):
        payload = str(value).encode("utf-8", "ignore")[: max(0, int(width) - 1)]
        if payload:
            out[row_idx, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return out


def _chunks_for(data: np.ndarray) -> tuple[int, ...]:
    shape = tuple(int(v) for v in data.shape)
    if not shape:
        return (1,)
    if len(shape) == 1:
        return (max(1, min(shape[0], 8192)),)
    return (max(1, min(shape[0], 2048)), *shape[1:])


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    arr = np.asarray(data)
    group.create_array(name, data=arr, chunks=_chunks_for(arr), overwrite=True)


def render_chaser_distance_timeseries_png(
    result: ChaserDistanceResult,
    *,
    dpi: int = 150,
    max_points: int = 5000,
) -> bytes:
    n_chasers = int(result.chaser_indices.shape[0])
    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    time_min = result.camera_frame_id.astype(np.float64) / float(result.fps) / 60.0
    step = max(1, int(np.ceil(result.total_frames / max(1, int(max_points)))))
    idx = np.arange(0, result.total_frames, step, dtype=np.int64)

    for window in result.windows:
        start_min = window.start_frame / float(result.fps) / 60.0
        end_min = (window.end_frame + 1) / float(result.fps) / 60.0
        ax.axvspan(start_min, end_min, alpha=0.08)

    for c_idx, chaser_index in enumerate(result.chaser_indices.tolist()):
        ax.plot(
            time_min[idx],
            result.distance_mm[idx, c_idx],
            linewidth=0.8,
            alpha=0.85,
            label=f"chaser {int(chaser_index)}",
        )
    ax.set_ylabel("distance (mm)")
    ax.set_xlabel("time (min)")
    ax.set_title(f"Offline refined detection to chaser distance: {result.recording_id}")
    ax.legend(loc="upper right", fontsize=8, ncol=max(1, min(n_chasers, 4)))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_chaser_distance_epoch_median_png(result: ChaserDistanceResult, *, dpi: int = 150) -> bytes:
    n_chasers = int(result.chaser_indices.shape[0])
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    labels = [_display_epoch_label(w.label) for w in result.windows]
    x = np.arange(len(labels))
    width = 0.8 / max(1, n_chasers)
    for c_idx, chaser_index in enumerate(result.chaser_indices.tolist()):
        offset = (c_idx - (n_chasers - 1) / 2.0) * width
        ax.bar(
            x + offset,
            result.epoch_p50_distance_mm[:, c_idx],
            width=width,
            label=f"chaser {int(chaser_index)}",
        )
    ax.set_ylabel("median distance (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("stimulus epoch")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Epoch median fish-to-chaser distance\n{result.recording_id}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=max(1, min(n_chasers, 4)))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_chaser_distance_epoch_distribution_png(result: ChaserDistanceResult, *, dpi: int = 150) -> bytes:
    n_chasers = int(result.chaser_indices.shape[0])
    labels = [_display_epoch_label(w.label) for w in result.windows]
    fig, axes = plt.subplots(
        1,
        max(1, len(result.windows)),
        figsize=(4.2 * max(1, len(result.windows)), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    centers = result.histogram_bin_centers_mm.astype(np.float32)
    for w_idx, (ax, label) in enumerate(zip(axes_list, labels)):
        for c_idx, chaser_index in enumerate(result.chaser_indices.tolist()):
            density = result.histogram_density[w_idx, c_idx, :]
            sample_count = int(np.sum(result.histogram_counts[w_idx, c_idx, :]))
            color = colors[c_idx % len(colors)] if colors else f"C{c_idx}"
            if sample_count == 0:
                continue
            ax.plot(
                centers,
                density,
                color=color,
                linewidth=1.5,
                label=f"chaser {int(chaser_index)} n={sample_count:,}",
            )
            ax.fill_between(centers, density, color=color, alpha=0.18)
            median = float(result.epoch_p50_distance_mm[w_idx, c_idx])
            if np.isfinite(median):
                ax.axvline(median, color=color, linestyle="--", linewidth=0.9, alpha=0.75)

        if not np.any(result.histogram_counts[w_idx, :, :]):
            ax.text(0.5, 0.5, "no valid distances", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(label)
        ax.set_xlabel("distance (mm)")
        ax.set_xlim(left=0, right=float(result.histogram_bin_edges_mm[-1]))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes_list[0].set_ylabel("density")
    fig.suptitle(f"Fish-to-chaser distance distributions by epoch\n{result.recording_id}", fontsize=12)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def write_goodcopbadcop_chaser_dashboard_spec_artifact(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    run_name: str,
    run_path: str,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    summary: Mapping[str, Any],
    overwrite: bool = True,
) -> None:
    """Write the GoodCopBadCop interactive dashboard spec for a chaser run."""

    spec = build_goodcopbadcop_chaser_dashboard_spec(
        root,
        run_name=run_name,
        run_path=run_path,
        run_group=run_group,
    )
    source_paths = spec.get("source_paths", {})
    source_runs = spec.get("source_runs", {})
    signature = _artifact_signature(
        {
            "schema_id": GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID,
            "run_name": run_name,
            "run_path": run_path,
            "source_paths": source_paths,
            "source_runs": source_runs,
            "source_refs": source_refs,
            "parameters": parameters,
        }
    )
    write_interactive_plot_spec_artifact(
        run_group,
        DEFAULT_GOODCOPBADCOP_INTERACTIVE_ARTIFACT,
        spec,
        description="GoodCopBadCop chaser dashboard interactive plot spec.",
        created_by="fisheye.analysis.chaser_distance_runs",
        renderer=GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
        artifact_signature=signature,
        snapshot_artifact=TIMESERIES_PNG_ARTIFACT_NAME,
        source_paths=source_paths if isinstance(source_paths, Mapping) else None,
        source_runs=source_runs if isinstance(source_runs, Mapping) else None,
        parameters=parameters,
        extra_attrs={
            "plot_schema_id": GOODCOPBADCOP_CHASER_DASHBOARD_SPEC_SCHEMA_ID,
            "run_name": run_name,
            "source_refs": json_attr_safe(source_refs),
            "summary": json_attr_safe(summary),
        },
        overwrite=overwrite,
    )


def write_chaser_distance_run(
    zarr_path: Path,
    result: ChaserDistanceResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
    write_interactive_spec: bool = True,
) -> str:
    root = _open_root(zarr_path, mode="a")
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, PARENT_NAME)
    run_name = result.run_name
    if run_name in parent:
        if not overwrite:
            raise ValueError(f"Chaser distance run already exists: analysis/{PARENT_NAME}/{run_name}")
        del parent[run_name]
    run = parent.create_group(run_name)
    mark_run_pending(parent, run_name)
    mark_run_started(run, run_name=run_name, stage="chaser_distance")
    try:
        frames = run.require_group("frames")
        _write_array(frames, "camera_frame_id", result.camera_frame_id)
        _write_array(frames, "stimulus_frame_num", result.stimulus_frame_num)
        _write_array(frames, "timestamp_ns", result.timestamp_ns)
        _write_array(frames, "stimulus_epoch_window_id", result.stimulus_epoch_window_id)
        frames.attrs.update({"row_axis": "camera_frames"})

        chasers = run.require_group("chasers")
        _write_array(chasers, "chaser_index", result.chaser_indices)

        positions = run.require_group("positions")
        _write_array(positions, "fish_centroid_img_xy", result.fish_centroid_img_xy)
        _write_array(positions, "fish_centroid_arena_xy", result.fish_centroid_arena_xy)
        _write_array(positions, "chaser_arena_xy", result.chaser_arena_xy)
        _write_array(positions, "fish_valid", result.fish_valid)
        _write_array(positions, "chaser_valid", result.chaser_valid)
        positions.attrs.update(
            {
                "fish_centroid_img_xy_coordinate_frame": "source_image_px",
                "fish_centroid_arena_xy_coordinate_frame": "arena_relative_canvas_px",
                "fish_centroid_arena_xy_coordinate_origin": "top_left_of_active_arena",
                "chaser_arena_xy_coordinate_frame": result.coordinate_frame,
                "chaser_arena_xy_coordinate_origin": result.coordinate_origin,
                "coordinate_frame": "arena_relative_canvas_px",
                "coordinate_origin": "top_left_of_active_arena",
                "x_axis_direction": "right",
                "y_axis_direction": "down",
                "axis_order": {
                    "fish_centroid_img_xy": ["camera_frame", "xy"],
                    "fish_centroid_arena_xy": ["camera_frame", "xy"],
                    "chaser_arena_xy": ["camera_frame", "chaser", "xy"],
                    "chaser_valid": ["camera_frame", "chaser"],
                },
            }
        )

        distances = run.require_group("distances")
        _write_array(distances, "distance_px", result.distance_px)
        _write_array(distances, "distance_mm", result.distance_mm)
        _write_array(distances, "nearest_chaser_index", result.nearest_chaser_index)
        _write_array(distances, "nearest_distance_mm", result.nearest_distance_mm)
        distances.attrs.update(
            {
                "distance_px_coordinate_frame": "arena_relative_canvas_px",
                "distance_mm_conversion": "distance_px / pixels_per_mm_projector",
                "axis_order": {
                    "distance_px": ["camera_frame", "chaser"],
                    "distance_mm": ["camera_frame", "chaser"],
                    "nearest_distance_mm": ["camera_frame"],
                },
            }
        )

        epoch_summary = run.require_group("epoch_summary")
        _write_array(epoch_summary, "window_id", np.asarray([w.window_id for w in result.windows], dtype=np.int32))
        _write_array(epoch_summary, "label_bytes", _bytes_array([w.label for w in result.windows]))
        _write_array(epoch_summary, "start_frame", np.asarray([w.start_frame for w in result.windows], dtype=np.int64))
        _write_array(epoch_summary, "end_frame", np.asarray([w.end_frame for w in result.windows], dtype=np.int64))
        _write_array(epoch_summary, "valid_frame_count", result.epoch_valid_frame_count)
        _write_array(epoch_summary, "mean_distance_mm", result.epoch_mean_distance_mm)
        _write_array(epoch_summary, "min_distance_mm", result.epoch_min_distance_mm)
        _write_array(epoch_summary, "p05_distance_mm", result.epoch_p05_distance_mm)
        _write_array(epoch_summary, "p50_distance_mm", result.epoch_p50_distance_mm)
        _write_array(epoch_summary, "p95_distance_mm", result.epoch_p95_distance_mm)
        _write_array(epoch_summary, "fraction_within_threshold", result.epoch_fraction_within_threshold)
        epoch_summary.attrs.update(
            {
                "row_axis": "stimulus_epoch_windows",
                "column_axis": "chasers",
                "threshold_mm": float(result.threshold_mm),
                "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
            }
        )

        epoch_distributions = run.require_group("epoch_distributions")
        _write_array(epoch_distributions, "window_id", np.asarray([w.window_id for w in result.windows], dtype=np.int32))
        _write_array(epoch_distributions, "chaser_index", result.chaser_indices)
        _write_array(epoch_distributions, "bin_edges_mm", result.histogram_bin_edges_mm)
        _write_array(epoch_distributions, "bin_centers_mm", result.histogram_bin_centers_mm)
        _write_array(epoch_distributions, "hist_counts", result.histogram_counts)
        _write_array(epoch_distributions, "hist_density", result.histogram_density)
        _write_array(epoch_distributions, "valid_sample_count", result.epoch_valid_frame_count)
        epoch_distributions.attrs.update(
            {
                "row_axis": "stimulus_epoch_windows",
                "column_axis": "chasers",
                "bin_axis": "distance_mm_bins",
                "bin_width_mm": float(result.distribution_bin_width_mm),
                "hist_counts_axis_order": ["window", "chaser", "distance_bin"],
                "hist_density_axis_order": ["window", "chaser", "distance_bin"],
                "density_normalization": "sum(hist_density * bin_width_mm) == 1 for non-empty window/chaser pairs",
            }
        )

        git = get_git_info(Path(__file__).resolve().parents[3])
        source_refs = {
            "source_detection_path": result.source_detection_path,
            "source_detection_kind": result.source_detection_kind,
            "source_stimulus_run": result.source_stimulus_run,
            "source_stimulus_path": result.source_stimulus_path,
            "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
            "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
        }
        parameters = {
            "threshold_mm": float(result.threshold_mm),
            "distribution_bin_width_mm": float(result.distribution_bin_width_mm),
            "coordinate_transform": "source_image_px_to_arena_relative_canvas_px_via_stored_homography",
        }
        summary = {
            "chaser_indices": result.chaser_indices.astype(int).tolist(),
            "fish_valid_frame_count": int(np.sum(result.fish_valid)),
            "chaser_valid_frame_count": np.sum(result.chaser_valid, axis=0).astype(int).tolist(),
            "distance_valid_frame_count": np.sum(np.isfinite(result.distance_mm), axis=0).astype(int).tolist(),
            "epoch_labels": [w.label for w in result.windows],
            "epoch_p50_distance_mm": result.epoch_p50_distance_mm.tolist(),
        }
        attrs = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "recording_id": result.recording_id,
            "row_axis": "camera_frames",
            "source_detection_path": result.source_detection_path,
            "source_detection_kind": result.source_detection_kind,
            "source_stimulus_run": result.source_stimulus_run,
            "source_stimulus_path": result.source_stimulus_path,
            "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
            "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
            "fps": float(result.fps),
            "total_frames": int(result.total_frames),
            "pixels_per_mm_projector": float(result.pixels_per_mm_projector),
            "coordinate_frame": "arena_relative_canvas_px",
            "coordinate_origin": "top_left_of_active_arena",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
            "arena_origin_in_canvas_xy": list(result.arena_origin_in_canvas_xy),
            "source_refs": source_refs,
            "parameters": parameters,
            "summary": summary,
            "git_commit": git.get("commit_hash"),
            "git_branch": git.get("branch"),
            "git_dirty": git.get("is_dirty"),
            "provenance": {
                "stage": "chaser_distance",
                "created_by": "fisheye.analysis.chaser_distance_runs",
                "inputs": source_refs,
                "parameters": parameters,
            },
        }
        run.attrs.update(json_attr_safe(attrs))
        lineage_payload = build_run_lineage_payload(
            run_family="analysis/chaser_distance_runs",
            analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "row_axis": "camera_frames"},
            method=METHOD,
            method_version=METHOD_VERSION,
            source_refs=source_refs,
            parameters=parameters,
            code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
        )
        write_run_lineage_attrs(run, lineage_payload, fingerprint_status="best_effort", overwrite=True)

        if write_png:
            timeseries_png = render_chaser_distance_timeseries_png(result)
            write_png_visualization_artifact(
                run,
                TIMESERIES_PNG_ARTIFACT_NAME,
                timeseries_png,
                description="Offline refined-detection fish centroid distance to each chaser over time.",
                created_by="fisheye.analysis.chaser_distance_runs",
                role="analysis_overview",
                source_paths={
                    "detection": result.source_detection_path,
                    "stimulus": result.source_stimulus_path,
                    "stimulus_epoch": result.source_stimulus_epoch_path,
                },
                source_runs={
                    "stimulus_run": result.source_stimulus_run,
                    "stimulus_epoch_run": result.source_stimulus_epoch_run,
                },
                parameters=parameters,
                extra_attrs={"chaser_distance_schema_id": SCHEMA_ID, "summary": json_attr_safe(summary)},
                overwrite=True,
            )
            epoch_median_png = render_chaser_distance_epoch_median_png(result)
            write_png_visualization_artifact(
                run,
                EPOCH_MEDIAN_PNG_ARTIFACT_NAME,
                epoch_median_png,
                description="Epoch-median offline refined-detection fish-to-chaser distance barplot.",
                created_by="fisheye.analysis.chaser_distance_runs",
                role="analysis_summary",
                source_paths={
                    "detection": result.source_detection_path,
                    "stimulus": result.source_stimulus_path,
                    "stimulus_epoch": result.source_stimulus_epoch_path,
                },
                source_runs={
                    "stimulus_run": result.source_stimulus_run,
                    "stimulus_epoch_run": result.source_stimulus_epoch_run,
                },
                parameters=parameters,
                extra_attrs={"chaser_distance_schema_id": SCHEMA_ID, "summary": json_attr_safe(summary)},
                overwrite=True,
            )
            epoch_distribution_png = render_chaser_distance_epoch_distribution_png(result)
            write_png_visualization_artifact(
                run,
                EPOCH_DISTRIBUTION_PNG_ARTIFACT_NAME,
                epoch_distribution_png,
                description="Per-epoch distribution of offline refined-detection fish-to-chaser distances.",
                created_by="fisheye.analysis.chaser_distance_runs",
                role="analysis_distribution",
                source_paths={
                    "detection": result.source_detection_path,
                    "stimulus": result.source_stimulus_path,
                    "stimulus_epoch": result.source_stimulus_epoch_path,
                },
                source_runs={
                    "stimulus_run": result.source_stimulus_run,
                    "stimulus_epoch_run": result.source_stimulus_epoch_run,
                },
                parameters=parameters,
                extra_attrs={"chaser_distance_schema_id": SCHEMA_ID, "summary": json_attr_safe(summary)},
                overwrite=True,
            )
            if write_interactive_spec:
                write_goodcopbadcop_chaser_dashboard_spec_artifact(
                    root,
                    run,
                    run_name=run_name,
                    run_path=f"analysis/{PARENT_NAME}/{run_name}",
                    source_refs=source_refs,
                    parameters=parameters,
                    summary=summary,
                    overwrite=True,
                )
        mark_run_complete(run, parent_group=parent, run_name=run_name)
    except Exception as exc:
        mark_run_failed(run, error=str(exc))
        raise
    return f"analysis/{PARENT_NAME}/{run_name}"


def _result_payload(result: ChaserDistanceResult) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "run_name": result.run_name,
        "source_detection_path": result.source_detection_path,
        "source_stimulus_run": result.source_stimulus_run,
        "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
        "total_frames": result.total_frames,
        "chaser_indices": result.chaser_indices.astype(int).tolist(),
        "fish_valid_frame_count": int(np.sum(result.fish_valid)),
        "distance_valid_frame_count": np.sum(np.isfinite(result.distance_mm), axis=0).astype(int).tolist(),
        "windows": [
            {
                "window_id": window.window_id,
                "label": window.label,
                "start_frame": window.start_frame,
                "end_frame": window.end_frame,
                "p50_distance_mm": result.epoch_p50_distance_mm[i].tolist(),
                "mean_distance_mm": result.epoch_mean_distance_mm[i].tolist(),
            }
            for i, window in enumerate(result.windows)
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--run-name", default=utc_run_name(), help="Chaser distance run name.")
    parser.add_argument("--stimulus-run", help="Source stimulus run name. Defaults to latest complete.")
    parser.add_argument("--stimulus-epoch-run", help="Source stimulus epoch run name. Defaults to latest complete.")
    parser.add_argument("--source", choices=("active", "raw"), default="active")
    parser.add_argument("--detect-run", help="Raw detect run name when --source raw is used.")
    parser.add_argument("--detection-path", help="Explicit detection group path.")
    parser.add_argument("--threshold-mm", type=float, default=20.0)
    parser.add_argument("--distribution-bin-width-mm", type=float, default=2.0)
    parser.add_argument("--apply", action="store_true", help="Write analysis/chaser_distance_runs/<run>.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run with the same name.")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG overview artifact.")
    parser.add_argument(
        "--no-interactive-spec",
        action="store_true",
        help="Skip the GoodCopBadCop interactive plot spec artifact.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_distance_result(
        Path(args.zarr_path),
        run_name=str(args.run_name),
        stimulus_run=args.stimulus_run,
        stimulus_epoch_run=args.stimulus_epoch_run,
        source=str(args.source),
        detect_run=args.detect_run,
        detection_path=args.detection_path,
        threshold_mm=float(args.threshold_mm),
        distribution_bin_width_mm=float(args.distribution_bin_width_mm),
    )
    path = None
    if args.apply:
        path = write_chaser_distance_run(
            Path(args.zarr_path),
            result,
            overwrite=bool(args.overwrite),
            write_png=not bool(args.no_png),
            write_interactive_spec=not bool(args.no_interactive_spec),
        )
    payload = _result_payload(result)
    payload["applied_path"] = path
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"recording_id: {result.recording_id}")
        print(f"detection_source: {result.source_detection_path}")
        print(f"stimulus_run: {result.source_stimulus_run}")
        print(f"chaser_indices: {result.chaser_indices.astype(int).tolist()}")
        for i, window in enumerate(result.windows):
            print(
                f"  {window.label}: p50_distance_mm="
                f"{[round(float(v), 3) if np.isfinite(v) else None for v in result.epoch_p50_distance_mm[i]]}"
            )
        if path:
            print(f"wrote: {path}")
        else:
            print("dry_run: pass --apply to write analysis/chaser_distance_runs/<run>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
