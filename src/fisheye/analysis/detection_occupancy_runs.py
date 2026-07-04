"""Materialize detection occupancy heatmaps inside analysis zarrs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.plot_artifacts import write_png_visualization_artifact
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
    resolve_authoritative_run_name,
)
from fisheye.shared.system_metadata import get_git_info
from fisheye.visualization.plot_detection_epoch_heatmaps import (
    EpochWindow,
    _read_detection_centers,
    _read_scores,
    _resolve_detection_group,
    _resolve_dimensions,
    compute_heatmap,
)


SCHEMA_ID = "palette.detection_occupancy.v1"
SCHEMA_VERSION = 1
METHOD = "detection_centroid_epoch_occupancy"
METHOD_VERSION = "1"
PARENT_NAME = "detection_occupancy_runs"
PNG_ARTIFACT_NAME = "detection_occupancy_overview_png"
SPATIAL_OCCUPANCY_SCHEMA_ID = "palette.spatial_occupancy_zones.v1"
SPATIAL_OCCUPANCY_SCHEMA_VERSION = 1
IMAGE_QUADRANTS_ZONE_SET_ID = "image_quadrants_v1"


@dataclass(frozen=True)
class OccupancyWindow:
    window_id: int
    label: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float


@dataclass(frozen=True)
class SpatialOccupancyZoneSet:
    zone_set_id: str
    zone_set_source: str
    zone_set_source_ref: str
    coordinate_frame: str
    coordinate_origin: str
    x_axis_direction: str
    y_axis_direction: str
    detection_selection_policy: str
    zone_overlap_policy: str
    time_basis: str
    zone_id: tuple[str, ...]
    label: tuple[str, ...]
    geometry_type: tuple[str, ...]
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
class DetectionOccupancyResult:
    zarr_path: str
    recording_id: str
    run_name: str
    source_detection_path: str
    source_detection_kind: str
    source_stimulus_epoch_run: str
    source_stimulus_epoch_path: str
    width: int
    height: int
    fps: float
    total_frames: int
    bin_size: int
    smooth_sigma: float
    min_score: Optional[float]
    windows: tuple[OccupancyWindow, ...]
    counts: np.ndarray
    normalized: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    detection_count: np.ndarray
    covered_frame_count: np.ndarray
    total_span_frames: np.ndarray
    coverage_pct: np.ndarray
    spatial_occupancy: tuple[SpatialOccupancyZoneSet, ...]


def utc_run_name(prefix: str = "detection_occupancy") -> str:
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


def _resolve_epoch_run(root: zarr.Group, epoch_run: Optional[str]) -> tuple[zarr.Group, str, str]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_epoch_runs" not in analysis:
        raise ValueError("Archive has no analysis/stimulus_epoch_runs group.")
    parent = analysis["stimulus_epoch_runs"]
    resolved = str(epoch_run).strip() if epoch_run else None
    if not resolved:
        resolved = resolve_authoritative_run_name(parent)
    if not resolved:
        latest = parent.attrs.get("latest")
        resolved = str(latest).strip() if latest else None
    if not resolved or resolved not in parent:
        raise ValueError("No usable stimulus epoch run found; pass --stimulus-epoch-run.")
    return parent[resolved], resolved, f"analysis/stimulus_epoch_runs/{resolved}"


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row) for row in values]
    return [decode_null_terminated_text(value) for value in values.reshape(-1)]


def _read_epoch_windows(epoch_group: zarr.Group) -> tuple[OccupancyWindow, ...]:
    windows = epoch_group.get("windows")
    if windows is None:
        raise ValueError("Stimulus epoch run has no windows group.")
    required = ("window_id", "label_bytes", "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s")
    missing = [name for name in required if name not in windows]
    if missing:
        raise ValueError(f"Stimulus epoch windows missing arrays: {missing}")
    ids = np.asarray(windows["window_id"][:], dtype=np.int32).reshape(-1)
    labels = _decode_text_column(np.asarray(windows["label_bytes"][:]))
    start_frame = np.asarray(windows["start_frame"][:], dtype=np.int64).reshape(-1)
    end_frame = np.asarray(windows["end_frame"][:], dtype=np.int64).reshape(-1)
    start_time = np.asarray(windows["start_time_s"][:], dtype=np.float64).reshape(-1)
    end_time = np.asarray(windows["end_time_s"][:], dtype=np.float64).reshape(-1)
    duration = np.asarray(windows["duration_s"][:], dtype=np.float64).reshape(-1)
    n = ids.shape[0]
    arrays = (labels, start_frame, end_frame, start_time, end_time, duration)
    if any(len(item) != n for item in arrays):
        raise ValueError("Stimulus epoch window arrays disagree on length.")
    return tuple(
        OccupancyWindow(
            window_id=int(ids[i]),
            label=str(labels[i]),
            start_frame=int(start_frame[i]),
            end_frame=int(end_frame[i]),
            start_time_s=float(start_time[i]),
            end_time_s=float(end_time[i]),
            duration_s=float(duration[i]),
        )
        for i in range(n)
    )


def _window_frame_bounds(window: OccupancyWindow, *, fps: float, total_frames: int) -> tuple[int, int, int]:
    start_frame = max(0, int(window.start_frame))
    requested_end = int(window.end_frame)
    if total_frames > 0:
        end_frame = min(max(0, int(total_frames) - 1), requested_end)
    else:
        end_frame = requested_end
    if end_frame < start_frame:
        end_frame = start_frame
    span = max(0, end_frame - start_frame + 1)
    return start_frame, end_frame, span


def _select_one_detection_per_frame(
    frames: np.ndarray,
    centers: np.ndarray,
    *,
    scores: Optional[np.ndarray],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_values = np.asarray(frames, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64)
    if center_values.ndim != 2 or center_values.shape[1] != 2:
        raise ValueError("Detection centers must have shape (N, 2).")
    if frame_values.shape[0] != center_values.shape[0]:
        raise ValueError("Detection frames and centers disagree on row count.")

    score_values: Optional[np.ndarray] = None
    if scores is not None:
        score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_values.shape[0] != frame_values.shape[0]:
            raise ValueError("Score array length does not match frame_indices.")

    valid = np.isfinite(center_values).all(axis=1)
    valid &= center_values[:, 0] >= 0.0
    valid &= center_values[:, 0] < float(width)
    valid &= center_values[:, 1] >= 0.0
    valid &= center_values[:, 1] < float(height)

    selected_by_frame: dict[int, int] = {}
    for row_idx in np.flatnonzero(valid):
        frame = int(frame_values[row_idx])
        if frame not in selected_by_frame:
            selected_by_frame[frame] = int(row_idx)
            continue
        if score_values is None:
            continue
        current_idx = selected_by_frame[frame]
        new_score = float(score_values[row_idx]) if np.isfinite(score_values[row_idx]) else float("-inf")
        current_score = (
            float(score_values[current_idx]) if np.isfinite(score_values[current_idx]) else float("-inf")
        )
        if new_score > current_score:
            selected_by_frame[frame] = int(row_idx)

    if not selected_by_frame:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float64)

    selected_indices = np.asarray(
        [selected_by_frame[frame] for frame in sorted(selected_by_frame)],
        dtype=np.int64,
    )
    return frame_values[selected_indices], center_values[selected_indices]


def _build_image_quadrants_zone_set(
    *,
    windows: Sequence[OccupancyWindow],
    frames: np.ndarray,
    centers: np.ndarray,
    scores: Optional[np.ndarray],
    width: int,
    height: int,
    fps: float,
    total_frames: int,
) -> SpatialOccupancyZoneSet:
    selected_frames, selected_centers = _select_one_detection_per_frame(
        frames,
        centers,
        scores=scores,
        width=width,
        height=height,
    )
    zone_ids = ("top_left", "top_right", "bottom_left", "bottom_right")
    labels = ("Top left", "Top right", "Bottom left", "Bottom right")
    geometry_types = ("bounds_xyxy",) * 4
    mid_x = float(width) / 2.0
    mid_y = float(height) / 2.0
    bounds_xyxy = np.asarray(
        [
            [0.0, 0.0, mid_x, mid_y],
            [mid_x, 0.0, float(width), mid_y],
            [0.0, mid_y, mid_x, float(height)],
            [mid_x, mid_y, float(width), float(height)],
        ],
        dtype=np.float32,
    )
    n_windows = len(windows)
    n_zones = len(zone_ids)
    frame_count = np.zeros((n_windows, n_zones), dtype=np.int64)
    detected_frame_count = np.zeros(n_windows, dtype=np.int64)
    missing_frame_count = np.zeros(n_windows, dtype=np.int64)
    total_span_frames = np.zeros(n_windows, dtype=np.int64)
    coverage_pct = np.zeros(n_windows, dtype=np.float32)

    if selected_centers.size:
        zone_index = ((selected_centers[:, 1] >= mid_y).astype(np.int64) * 2) + (
            selected_centers[:, 0] >= mid_x
        ).astype(np.int64)
    else:
        zone_index = np.zeros(0, dtype=np.int64)

    for window_idx, window in enumerate(windows):
        start_frame, end_frame, span = _window_frame_bounds(window, fps=fps, total_frames=total_frames)
        in_window = (selected_frames >= start_frame) & (selected_frames <= end_frame)
        selected_zones = zone_index[in_window]
        if selected_zones.size:
            frame_count[window_idx] = np.bincount(selected_zones, minlength=n_zones)[:n_zones]
        detected = int(selected_zones.shape[0])
        detected_frame_count[window_idx] = detected
        total_span_frames[window_idx] = int(span)
        missing_frame_count[window_idx] = max(0, int(span) - detected)
        coverage_pct[window_idx] = float(detected / span * 100.0) if span else 0.0

    if fps > 0 and np.isfinite(fps):
        time_s = frame_count.astype(np.float64) / float(fps)
        time_basis = "frame_count / frame_rate_hz"
    else:
        time_s = np.zeros_like(frame_count, dtype=np.float64)
        time_basis = "frame_count; fps unavailable"

    fraction_of_epoch = np.divide(
        frame_count,
        total_span_frames[:, None],
        out=np.zeros_like(frame_count, dtype=np.float64),
        where=total_span_frames[:, None] > 0,
    )
    fraction_of_detected = np.divide(
        frame_count,
        detected_frame_count[:, None],
        out=np.zeros_like(frame_count, dtype=np.float64),
        where=detected_frame_count[:, None] > 0,
    )

    return SpatialOccupancyZoneSet(
        zone_set_id=IMAGE_QUADRANTS_ZONE_SET_ID,
        zone_set_source="predefined_spec:quadrants.v1",
        zone_set_source_ref="palette:predefined_spatial_occupancy/quadrants.v1",
        coordinate_frame="source_image_px",
        coordinate_origin="top_left",
        x_axis_direction="right",
        y_axis_direction="down",
        detection_selection_policy=(
            "one_valid_detection_per_frame; finite source-image centroid; "
            "highest score if available; first stable source row breaks ties"
        ),
        zone_overlap_policy="exclusive_bounds_xyxy; midlines_to_right_or_bottom",
        time_basis=time_basis,
        zone_id=zone_ids,
        label=labels,
        geometry_type=geometry_types,
        display_order=np.arange(n_zones, dtype=np.int16),
        bounds_xyxy=bounds_xyxy,
        frame_count=frame_count,
        time_s=time_s.astype(np.float32),
        fraction_of_epoch=fraction_of_epoch.astype(np.float32),
        fraction_of_detected=fraction_of_detected.astype(np.float32),
        detected_frame_count=detected_frame_count,
        missing_frame_count=missing_frame_count,
        total_span_frames=total_span_frames,
        coverage_pct=coverage_pct,
    )


def build_detection_occupancy_result(
    zarr_path: Path,
    *,
    run_name: str,
    stimulus_epoch_run: Optional[str] = None,
    epoch_windows: Optional[Sequence[OccupancyWindow]] = None,
    source: str = "active",
    detect_run: Optional[str] = None,
    detection_path: Optional[str] = None,
    bin_size: int = 128,
    smooth_sigma: float = 1.0,
    min_score: Optional[float] = None,
) -> DetectionOccupancyResult:
    root = _open_root(zarr_path, mode="r")
    if epoch_windows is None:
        epoch_group, epoch_run_name, epoch_path = _resolve_epoch_run(root, stimulus_epoch_run)
        windows = _read_epoch_windows(epoch_group)
    else:
        if not stimulus_epoch_run:
            raise ValueError("stimulus_epoch_run is required when epoch_windows are supplied.")
        epoch_run_name = str(stimulus_epoch_run)
        epoch_path = f"analysis/stimulus_epoch_runs/{epoch_run_name}"
        windows = tuple(epoch_windows)
    detect_group, source_detection_path, source_detection_kind = _resolve_detection_group(
        root,
        source=source,
        detect_run=detect_run,
        detection_path=detection_path,
    )
    width, height, fps, total_frames = _resolve_dimensions(root, detect_group)
    frames, centers = _read_detection_centers(detect_group, width=width, height=height)
    scores = _read_scores(detect_group)
    if min_score is not None and scores is not None:
        if scores.shape[0] != frames.shape[0]:
            raise ValueError("Score array length does not match frame_indices.")
        keep = scores >= float(min_score)
        frames = frames[keep]
        centers = centers[keep]
        scores = scores[keep]

    counts_list: list[np.ndarray] = []
    normalized_list: list[np.ndarray] = []
    detection_counts: list[int] = []
    covered_counts: list[int] = []
    span_counts: list[int] = []
    coverage_values: list[float] = []
    x_edges = np.linspace(0.0, float(width), max(1, int(np.ceil(float(width) / float(bin_size)))) + 1)
    y_edges = np.linspace(0.0, float(height), max(1, int(np.ceil(float(height) / float(bin_size)))) + 1)

    for item in windows:
        window = EpochWindow(
            label=item.label,
            start_s=item.start_time_s,
            end_s=item.end_time_s,
            start_frame=item.start_frame,
            end_frame=item.end_frame,
            source="stimulus_epoch_run",
        )
        counts, _start, _end, span, detections, covered, coverage = compute_heatmap(
            frames=frames,
            centers=centers,
            window=window,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            bin_size=bin_size,
            smooth_sigma=0.0,
            normalize="count",
        )
        normalized, *_ = compute_heatmap(
            frames=frames,
            centers=centers,
            window=window,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            bin_size=bin_size,
            smooth_sigma=smooth_sigma,
            normalize="max",
        )
        counts_list.append(counts.astype(np.uint32, copy=False))
        normalized_list.append(normalized.astype(np.float32, copy=False))
        detection_counts.append(int(detections))
        covered_counts.append(int(covered))
        span_counts.append(int(span))
        coverage_values.append(float(coverage))

    recording_id = _attr_text(root.attrs, "recording_id", "recording_name") or Path(zarr_path).stem
    spatial_occupancy = (
        _build_image_quadrants_zone_set(
            windows=windows,
            frames=frames,
            centers=centers,
            scores=scores,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
        ),
    )
    return DetectionOccupancyResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        run_name=run_name,
        source_detection_path=source_detection_path,
        source_detection_kind=source_detection_kind,
        source_stimulus_epoch_run=epoch_run_name,
        source_stimulus_epoch_path=epoch_path,
        width=int(width),
        height=int(height),
        fps=float(fps),
        total_frames=int(total_frames),
        bin_size=int(bin_size),
        smooth_sigma=float(smooth_sigma),
        min_score=min_score,
        windows=windows,
        counts=np.stack(counts_list, axis=0),
        normalized=np.stack(normalized_list, axis=0),
        x_edges=x_edges.astype(np.float32),
        y_edges=y_edges.astype(np.float32),
        detection_count=np.asarray(detection_counts, dtype=np.int64),
        covered_frame_count=np.asarray(covered_counts, dtype=np.int64),
        total_span_frames=np.asarray(span_counts, dtype=np.int64),
        coverage_pct=np.asarray(coverage_values, dtype=np.float32),
        spatial_occupancy=spatial_occupancy,
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
    return (max(1, min(shape[0], 16)), *shape[1:])


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    arr = np.asarray(data)
    group.create_array(name, data=arr, chunks=_chunks_for(arr), overwrite=True)


def _write_spatial_occupancy(
    run: zarr.Group,
    result: DetectionOccupancyResult,
) -> None:
    if not result.spatial_occupancy:
        return
    spatial_parent = run.require_group("spatial_occupancy")
    spatial_parent.attrs.update(
        json_attr_safe(
            {
                "description": "Detection-derived spatial occupancy summaries by stimulus epoch window.",
                "schema_id": SPATIAL_OCCUPANCY_SCHEMA_ID,
                "schema_version": SPATIAL_OCCUPANCY_SCHEMA_VERSION,
                "row_axis": "stimulus_epoch_windows",
                "column_axis": "occupancy_zones",
            }
        )
    )
    for zone_set in result.spatial_occupancy:
        if zone_set.zone_set_id in spatial_parent:
            del spatial_parent[zone_set.zone_set_id]
        group = spatial_parent.create_group(zone_set.zone_set_id)
        group.attrs.update(
            json_attr_safe(
                {
                    "schema_id": SPATIAL_OCCUPANCY_SCHEMA_ID,
                    "schema_version": SPATIAL_OCCUPANCY_SCHEMA_VERSION,
                    "zone_set_id": zone_set.zone_set_id,
                    "zone_set_source": zone_set.zone_set_source,
                    "zone_set_source_ref": zone_set.zone_set_source_ref,
                    "coordinate_frame": zone_set.coordinate_frame,
                    "coordinate_origin": zone_set.coordinate_origin,
                    "x_axis_direction": zone_set.x_axis_direction,
                    "y_axis_direction": zone_set.y_axis_direction,
                    "source_detection_path": result.source_detection_path,
                    "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
                    "detection_selection_policy": zone_set.detection_selection_policy,
                    "zone_overlap_policy": zone_set.zone_overlap_policy,
                    "time_basis": zone_set.time_basis,
                    "width": int(result.width),
                    "height": int(result.height),
                    "fps": float(result.fps),
                }
            )
        )

        zone_spec = group.require_group("zone_spec")
        _write_array(zone_spec, "zone_id", _bytes_array(zone_set.zone_id))
        _write_array(zone_spec, "label_bytes", _bytes_array(zone_set.label))
        _write_array(zone_spec, "geometry_type", _bytes_array(zone_set.geometry_type))
        _write_array(zone_spec, "display_order", zone_set.display_order)
        _write_array(zone_spec, "bounds_xyxy", zone_set.bounds_xyxy)
        zone_spec.attrs.update(
            json_attr_safe(
                {
                    "row_axis": "occupancy_zones",
                    "text_encoding": "null_terminated_utf8_uint8_matrix",
                    "bounds_xyxy_description": "[x_min, y_min, x_max, y_max]; max bounds are exclusive.",
                }
            )
        )

        summary = group.require_group("summary")
        _write_array(summary, "frame_count", zone_set.frame_count)
        _write_array(summary, "time_s", zone_set.time_s)
        _write_array(summary, "fraction_of_epoch", zone_set.fraction_of_epoch)
        _write_array(summary, "fraction_of_detected", zone_set.fraction_of_detected)
        _write_array(summary, "detected_frame_count", zone_set.detected_frame_count)
        _write_array(summary, "missing_frame_count", zone_set.missing_frame_count)
        _write_array(summary, "total_span_frames", zone_set.total_span_frames)
        _write_array(summary, "coverage_pct", zone_set.coverage_pct)
        summary.attrs.update(
            json_attr_safe(
                {
                    "axis_order": ["window", "zone"],
                    "frame_count_description": "Selected valid detection frames per epoch window and zone.",
                    "fraction_of_epoch_description": "frame_count divided by total epoch span frames.",
                    "fraction_of_detected_description": "frame_count divided by selected detected frames.",
                }
            )
        )


def render_occupancy_png(result: DetectionOccupancyResult, *, dpi: int = 150) -> bytes:
    n = len(result.windows)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False, constrained_layout=True)
    image = None
    for idx, window in enumerate(result.windows):
        ax = axes[0, idx]
        image = ax.imshow(
            result.normalized[idx],
            cmap="inferno",
            origin="upper",
            extent=[0, result.width, result.height, 0],
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(
            f"{window.label.replace('_', ' ')}\n"
            f"{window.start_time_s / 60.0:.2f}-{window.end_time_s / 60.0:.2f} min\n"
            f"frames {window.start_frame}-{window.end_frame}",
            fontsize=10,
        )
        ax.set_xlabel("x px")
        stats = (
            f"n={int(result.detection_count[idx]):,}\n"
            f"frames={int(result.covered_frame_count[idx]):,}/{int(result.total_span_frames[idx]):,}\n"
            f"cov={float(result.coverage_pct[idx]):.1f}%"
        )
        ax.text(
            0.02,
            0.02,
            stats,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="white",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.55, "linewidth": 0},
        )
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Max-normalized occupancy")
    fig.suptitle(f"Detection occupancy: {result.recording_id}", fontsize=13)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def write_detection_occupancy_run(
    zarr_path: Path,
    result: DetectionOccupancyResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
) -> str:
    root = _open_root(zarr_path, mode="a")
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, PARENT_NAME)
    run_name = result.run_name
    if run_name in parent:
        if not overwrite:
            raise ValueError(f"Detection occupancy run already exists: analysis/{PARENT_NAME}/{run_name}")
        del parent[run_name]
    run = parent.create_group(run_name)
    mark_run_pending(parent, run_name)
    mark_run_started(run, run_name=run_name, stage="detection_occupancy")
    try:
        windows = run.require_group("windows")
        _write_array(windows, "window_id", np.asarray([w.window_id for w in result.windows], dtype=np.int32))
        _write_array(windows, "label_bytes", _bytes_array([w.label for w in result.windows]))
        _write_array(windows, "start_frame", np.asarray([w.start_frame for w in result.windows], dtype=np.int64))
        _write_array(windows, "end_frame", np.asarray([w.end_frame for w in result.windows], dtype=np.int64))
        _write_array(windows, "start_time_s", np.asarray([w.start_time_s for w in result.windows], dtype=np.float64))
        _write_array(windows, "end_time_s", np.asarray([w.end_time_s for w in result.windows], dtype=np.float64))
        _write_array(windows, "duration_s", np.asarray([w.duration_s for w in result.windows], dtype=np.float64))
        _write_array(
            windows,
            "source_stimulus_epoch_window_id",
            np.asarray([w.window_id for w in result.windows], dtype=np.int32),
        )
        windows.attrs.update({"storage_layout": "columnar"})

        coverage = run.require_group("coverage")
        _write_array(coverage, "detection_count", result.detection_count)
        _write_array(coverage, "covered_frame_count", result.covered_frame_count)
        _write_array(coverage, "total_span_frames", result.total_span_frames)
        _write_array(coverage, "coverage_pct", result.coverage_pct)
        coverage.attrs.update({"row_axis": "stimulus_epoch_windows"})

        heatmaps = run.require_group("heatmaps")
        _write_array(heatmaps, "counts", result.counts)
        _write_array(heatmaps, "normalized", result.normalized)
        _write_array(heatmaps, "x_edges", result.x_edges)
        _write_array(heatmaps, "y_edges", result.y_edges)
        heatmaps.attrs.update(
            {
                "counts_description": "Raw detection-centroid counts per stimulus epoch window.",
                "normalized_description": "Max-normalized smoothed occupancy per stimulus epoch window.",
                "axis_order": ["window", "y_bin", "x_bin"],
                "bin_size_px": int(result.bin_size),
                "smooth_sigma_bins": float(result.smooth_sigma),
            }
        )

        _write_spatial_occupancy(run, result)

        git = get_git_info(Path(__file__).resolve().parents[3])
        source_refs = {
            "source_detection_path": result.source_detection_path,
            "source_detection_kind": result.source_detection_kind,
            "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
            "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
        }
        parameters = {
            "bin_size": int(result.bin_size),
            "smooth_sigma": float(result.smooth_sigma),
            "min_score": result.min_score,
            "spatial_occupancy_zone_sets": [zone_set.zone_set_id for zone_set in result.spatial_occupancy],
        }
        summary = {
            "window_labels": [w.label for w in result.windows],
            "detection_count": result.detection_count.tolist(),
            "covered_frame_count": result.covered_frame_count.tolist(),
            "total_span_frames": result.total_span_frames.tolist(),
            "coverage_pct": result.coverage_pct.tolist(),
            "spatial_occupancy_zone_sets": {
                zone_set.zone_set_id: {
                    "zone_id": list(zone_set.zone_id),
                    "frame_count": zone_set.frame_count.tolist(),
                    "time_s": zone_set.time_s.tolist(),
                    "fraction_of_epoch": zone_set.fraction_of_epoch.tolist(),
                    "fraction_of_detected": zone_set.fraction_of_detected.tolist(),
                }
                for zone_set in result.spatial_occupancy
            },
        }
        attrs = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_axis": "stimulus_epoch_windows",
            "run_name": run_name,
            "recording_id": result.recording_id,
            "source_detection_path": result.source_detection_path,
            "source_detection_kind": result.source_detection_kind,
            "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
            "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
            "coordinate_space": "source_image_pixels",
            "width": int(result.width),
            "height": int(result.height),
            "fps": float(result.fps),
            "total_frames": int(result.total_frames),
            "source_refs": source_refs,
            "parameters": parameters,
            "summary": summary,
            "git_commit": git.get("commit_hash"),
            "git_branch": git.get("branch"),
            "git_dirty": git.get("is_dirty"),
            "provenance": {
                "stage": "detection_occupancy",
                "created_by": "fisheye.analysis.detection_occupancy_runs",
                "inputs": source_refs,
                "parameters": parameters,
            },
        }
        run.attrs.update(json_attr_safe(attrs))
        lineage_payload = build_run_lineage_payload(
            run_family="analysis/detection_occupancy_runs",
            analysis_schema={
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "row_axis": "stimulus_epoch_windows",
            },
            method=METHOD,
            method_version=METHOD_VERSION,
            source_refs=source_refs,
            parameters=parameters,
            code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
        )
        write_run_lineage_attrs(run, lineage_payload, fingerprint_status="best_effort", overwrite=True)
        if write_png:
            png = render_occupancy_png(result)
            write_png_visualization_artifact(
                run,
                PNG_ARTIFACT_NAME,
                png,
                description="Detection centroid occupancy by stimulus epoch window.",
                created_by="fisheye.analysis.detection_occupancy_runs",
                role="analysis_overview",
                source_paths={
                    "detection": result.source_detection_path,
                    "stimulus_epoch": result.source_stimulus_epoch_path,
                },
                source_runs={"stimulus_epoch_run": result.source_stimulus_epoch_run},
                parameters=parameters,
                extra_attrs={"occupancy_schema_id": SCHEMA_ID, "summary": json_attr_safe(summary)},
                overwrite=True,
            )
        mark_run_complete(run, parent_group=parent, run_name=run_name)
    except Exception as exc:
        mark_run_failed(run, error=str(exc))
        raise
    return f"analysis/{PARENT_NAME}/{run_name}"


def _result_payload(result: DetectionOccupancyResult) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "run_name": result.run_name,
        "source_detection_path": result.source_detection_path,
        "source_detection_kind": result.source_detection_kind,
        "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
        "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
        "width": result.width,
        "height": result.height,
        "fps": result.fps,
        "total_frames": result.total_frames,
        "bin_size": result.bin_size,
        "smooth_sigma": result.smooth_sigma,
        "windows": [
            {
                "window_id": w.window_id,
                "label": w.label,
                "start_frame": w.start_frame,
                "end_frame": w.end_frame,
                "duration_s": w.duration_s,
                "detection_count": int(result.detection_count[i]),
                "covered_frame_count": int(result.covered_frame_count[i]),
                "total_span_frames": int(result.total_span_frames[i]),
                "coverage_pct": float(result.coverage_pct[i]),
            }
            for i, w in enumerate(result.windows)
        ],
        "spatial_occupancy": [
            {
                "zone_set_id": zone_set.zone_set_id,
                "zone_set_source": zone_set.zone_set_source,
                "zone_id": list(zone_set.zone_id),
                "frame_count": zone_set.frame_count.tolist(),
                "time_s": zone_set.time_s.tolist(),
                "fraction_of_epoch": zone_set.fraction_of_epoch.tolist(),
                "fraction_of_detected": zone_set.fraction_of_detected.tolist(),
                "detected_frame_count": zone_set.detected_frame_count.tolist(),
                "missing_frame_count": zone_set.missing_frame_count.tolist(),
                "coverage_pct": zone_set.coverage_pct.tolist(),
            }
            for zone_set in result.spatial_occupancy
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--run-name", default=utc_run_name(), help="Detection occupancy run name.")
    parser.add_argument("--stimulus-epoch-run", help="Source stimulus epoch run name. Defaults to latest complete.")
    parser.add_argument("--source", choices=("active", "raw"), default="active")
    parser.add_argument("--detect-run", help="Raw detect run name when --source raw is used.")
    parser.add_argument("--detection-path", help="Explicit detection group path.")
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--apply", action="store_true", help="Write analysis/detection_occupancy_runs/<run>.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing run with the same name.")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG overview artifact.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_detection_occupancy_result(
        Path(args.zarr_path),
        run_name=str(args.run_name),
        stimulus_epoch_run=args.stimulus_epoch_run,
        source=str(args.source),
        detect_run=args.detect_run,
        detection_path=args.detection_path,
        bin_size=int(args.bin_size),
        smooth_sigma=float(args.smooth_sigma),
        min_score=args.min_score,
    )
    path = None
    if args.apply:
        path = write_detection_occupancy_run(
            Path(args.zarr_path),
            result,
            overwrite=bool(args.overwrite),
            write_png=not bool(args.no_png),
        )
    payload = _result_payload(result)
    payload["applied_path"] = path
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"recording_id: {result.recording_id}")
        print(f"detection_source: {result.source_detection_path}")
        print(f"stimulus_epoch_run: {result.source_stimulus_epoch_run}")
        for i, window in enumerate(result.windows):
            print(
                f"  {window.label}: frames={window.start_frame}-{window.end_frame} "
                f"detections={int(result.detection_count[i])} "
                f"coverage={float(result.coverage_pct[i]):.1f}%"
            )
        for zone_set in result.spatial_occupancy:
            print(f"spatial_occupancy/{zone_set.zone_set_id}: zones={', '.join(zone_set.zone_id)}")
            for i, window in enumerate(result.windows):
                counts = ", ".join(
                    f"{zone}={int(zone_set.frame_count[i, zone_idx])}"
                    for zone_idx, zone in enumerate(zone_set.zone_id)
                )
                print(f"  {window.label}: {counts}")
        if path:
            print(f"wrote: {path}")
        else:
            print("dry_run: pass --apply to write analysis/detection_occupancy_runs/<run>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
