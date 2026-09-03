#!/usr/bin/env python3
"""Plot detection-centroid heatmaps for behavior epochs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from fisheye.shared.citrus_enums import load_event_types
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.refined_detect_resolution import resolve_detection_read_source
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name


@dataclass(frozen=True)
class EpochWindow:
    label: str
    start_s: float
    end_s: float
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    source: str = "elapsed_time"

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))


@dataclass(frozen=True)
class HeatmapResult:
    recording_id: str
    arena_id: Optional[str]
    camera_id: Optional[str]
    zarr_path: str
    source_path: str
    source_kind: str
    width: int
    height: int
    fps: float
    window: EpochWindow
    start_frame: int
    end_frame: int
    total_span_frames: int
    detection_count: int
    covered_frame_count: int
    coverage_pct: float
    heatmap: np.ndarray


def default_windows(
    *,
    first_minutes: float = 10.0,
    training_minutes: float = 2.5,
    post_minutes: float = 10.0,
    start_second: float = 0.0,
) -> list[EpochWindow]:
    start = float(start_second)
    first_end = start + float(first_minutes) * 60.0
    train_end = first_end + float(training_minutes) * 60.0
    post_end = train_end + float(post_minutes) * 60.0
    return [
        EpochWindow(f"first_{_format_minutes(first_minutes)}min", start, first_end),
        EpochWindow(
            f"training_{_format_minutes(training_minutes)}min", first_end, train_end
        ),
        EpochWindow(f"post_{_format_minutes(post_minutes)}min", train_end, post_end),
    ]


def _format_minutes(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p")
    return text


def parse_window_specs(raw_specs: Sequence[str]) -> list[EpochWindow]:
    windows: list[EpochWindow] = []
    for spec in raw_specs:
        parts = str(spec).split(":")
        if len(parts) != 3:
            raise ValueError(f"Window spec must be label:start_s:end_s, got {spec!r}")
        label, start_raw, end_raw = parts
        label = label.strip()
        if not label:
            raise ValueError(f"Window label must be non-empty in {spec!r}")
        start_s = float(start_raw)
        end_s = float(end_raw)
        if end_s <= start_s:
            raise ValueError(f"Window end must be after start in {spec!r}")
        windows.append(EpochWindow(label, start_s, end_s))
    if not windows:
        raise ValueError("At least one window is required.")
    return windows


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_structured_group(node: Any) -> dict[str, np.ndarray]:
    if hasattr(node, "array_keys"):
        field_names = node.attrs.get("field_names")
        if not field_names:
            field_names = list(node.array_keys())
        return {
            str(name): np.asarray(node[str(name)][:])
            for name in field_names
            if str(name) in node
        }

    array = np.asarray(node[:])
    if array.dtype.names:
        return {name: np.asarray(array[name]) for name in array.dtype.names}
    raise ValueError(
        "Expected a column group or structured array with named event fields."
    )


def _decode_text_value(value: Any) -> str:
    return decode_null_terminated_text(value, errors="ignore").strip()


def _event_names_from_columns(
    root: zarr.Group, columns: dict[str, np.ndarray]
) -> np.ndarray:
    for name_field in ("event_name", "event_type_name", "name"):
        if name_field in columns:
            return np.array(
                [_decode_text_value(value) for value in columns[name_field]],
                dtype=object,
            )

    event_type_ids = None
    for id_field in ("event_type_id", "event_type", "event_id"):
        if id_field in columns:
            event_type_ids = np.asarray(columns[id_field]).reshape(-1)
            break
    if event_type_ids is None:
        raise ValueError("Stimulus events lack event_name or event_type_id columns.")

    event_type_map = load_event_types(root)
    return np.array(
        [
            event_type_map.get(int(value), f"UNKNOWN_{int(value)}")
            for value in event_type_ids
        ],
        dtype=object,
    )


def _first_column(columns: dict[str, np.ndarray], *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in columns:
            return np.asarray(columns[name]).reshape(-1)
    return None


def _first_event_frame(
    event_frames: dict[str, int], names: Sequence[str]
) -> Optional[int]:
    for name in names:
        if name in event_frames:
            return int(event_frames[name])
    return None


def build_chaser_event_windows(
    event_frames: dict[str, int],
    *,
    fps: float,
    total_frames: int,
) -> list[EpochWindow]:
    """Build pre/training/post windows from chaser/protocol camera-frame events."""
    pre_start = _first_event_frame(
        event_frames, ("CHASER_PRE_PERIOD_START", "PROTOCOL_START")
    )
    training_start = _first_event_frame(event_frames, ("CHASER_TRAINING_START",))
    post_start = _first_event_frame(event_frames, ("CHASER_POST_PERIOD_START",))
    finish = _first_event_frame(
        event_frames,
        ("PROTOCOL_FINISH", "PROTOCOL_STOP", "STEP_END", "CHASER_PRESENTATION_END"),
    )

    if training_start is None:
        raise ValueError("Stimulus events do not include CHASER_TRAINING_START.")
    if post_start is None:
        raise ValueError("Stimulus events do not include CHASER_POST_PERIOD_START.")
    if pre_start is None:
        pre_start = 0
    if finish is None:
        finish = int(total_frames) if total_frames > 0 else post_start

    max_frame = (
        int(total_frames) - 1 if total_frames > 0 else max(finish - 1, post_start)
    )

    def window(label: str, start_frame: int, end_frame: int) -> EpochWindow:
        start = max(0, int(start_frame))
        end = min(max_frame, max(start, int(end_frame)))
        return EpochWindow(
            label=label,
            start_s=float(start) / float(fps),
            end_s=float(end + 1) / float(fps),
            start_frame=start,
            end_frame=end,
            source="stimulus_events",
        )

    return [
        window("pre_event", pre_start, training_start - 1),
        window("training_event", training_start, post_start - 1),
        window("post_event", post_start, finish - 1),
    ]


def _resolve_stimulus_run(
    root: zarr.Group,
    stimulus_run: Optional[str],
) -> tuple[str, zarr.Group]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise ValueError("Archive has no analysis/stimulus_runs group.")
    runs = analysis["stimulus_runs"]
    resolved = str(stimulus_run).strip() if stimulus_run else None
    if not resolved:
        resolved = resolve_authoritative_run_name(runs)
    if not resolved:
        run_names = sorted(str(name) for name in runs.group_keys())
        resolved = run_names[-1] if run_names else None
    if not resolved or resolved not in runs:
        raise ValueError("No usable stimulus run found; pass --stimulus-run.")
    return resolved, runs[resolved]


def resolve_stimulus_event_windows(
    root: zarr.Group,
    *,
    fps: float,
    total_frames: int,
    stimulus_run: Optional[str],
) -> list[EpochWindow]:
    resolved, stim_run = _resolve_stimulus_run(root, stimulus_run)
    events_node = stim_run.get("events")
    if events_node is None:
        raise ValueError(f"Stimulus run {resolved!r} has no events group.")
    columns = _load_structured_group(events_node)
    event_names = _event_names_from_columns(root, columns)
    camera_frames = _first_column(
        columns, "camera_frame_id", "camera_frame_num", "triggering_camera_frame_id"
    )
    if camera_frames is None:
        raise ValueError(f"Stimulus run {resolved!r} events lack camera frame column.")
    if len(event_names) != len(camera_frames):
        raise ValueError(
            "Stimulus event name and camera frame columns disagree on length."
        )

    event_frames: dict[str, int] = {}
    for event_name, frame_value in zip(event_names, camera_frames):
        name = str(event_name).strip()
        if not name:
            continue
        frame = int(_to_python_scalar(frame_value))
        if frame < 0:
            continue
        event_frames.setdefault(name, frame)

    return build_chaser_event_windows(event_frames, fps=fps, total_frames=total_frames)


def _open_root(zarr_path: Path) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)


def _attr_text(attrs: Any, *keys: str) -> Optional[str]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _attr_int(attrs: Any, *keys: str) -> Optional[int]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
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


def _resolve_raw_detect_group(
    root: zarr.Group, run_name: Optional[str]
) -> tuple[zarr.Group, str]:
    parent = root.get("detect_runs")
    if parent is None:
        raise ValueError("Archive has no detect_runs group.")
    resolved = str(run_name).strip() if run_name else None
    if not resolved:
        resolved = resolve_authoritative_run_name(parent)
    if not resolved:
        latest = parent.attrs.get("latest")
        resolved = str(latest).strip() if latest else None
    if not resolved or resolved not in parent:
        raise ValueError("No usable detect run found; pass --detect-run.")
    return parent[resolved], f"detect_runs/{resolved}"


def _resolve_detection_payload_group(
    group: zarr.Group,
    path: str,
) -> tuple[zarr.Group, str]:
    if "frame_indices" in group:
        return group, path
    instances = group.get("instances")
    if (
        instances is not None
        and "frame_indices" in instances
        and ("bbox_img_xyxy" in instances or "bbox_norm_coords" in instances)
    ):
        return instances, f"{path.rstrip('/')}/instances"
    return group, path


def _resolve_detection_group(
    root: zarr.Group,
    *,
    source: str,
    detect_run: Optional[str],
    detection_path: Optional[str],
) -> tuple[zarr.Group, str, str]:
    if detection_path:
        path = str(detection_path).strip().strip("/")
        if not path:
            raise ValueError("--detection-path must be non-empty.")
        group, payload_path = _resolve_detection_payload_group(root[path], path)
        return group, payload_path, "explicit"

    if source == "raw":
        group, path = _resolve_raw_detect_group(root, detect_run)
        group, payload_path = _resolve_detection_payload_group(group, path)
        return group, payload_path, "raw"

    resolution = resolve_detection_read_source(
        root, prefer_curated=True, allow_sparse_fallback=True
    )
    if not resolution.detection_path:
        raise ValueError("No active detection source resolved.")
    group, payload_path = _resolve_detection_payload_group(
        root[resolution.detection_path], resolution.detection_path
    )
    return (
        group,
        payload_path,
        resolution.detection_kind or "active",
    )


def _resolve_dimensions(
    root: zarr.Group, group: zarr.Group
) -> tuple[int, int, float, int]:
    width = (
        _attr_int(
            group.attrs,
            "source_full_width",
            "source_video_width",
            "width",
            "video_width",
        )
        or _attr_int(
            root.attrs,
            "width",
            "video_width",
            "source_video_width",
            "palette_video_width",
        )
        or 4512
    )
    height = (
        _attr_int(
            group.attrs,
            "source_full_height",
            "source_video_height",
            "height",
            "video_height",
        )
        or _attr_int(
            root.attrs,
            "height",
            "video_height",
            "source_video_height",
            "palette_video_height",
        )
        or 4512
    )
    fps = (
        _attr_float(root.attrs, "fps", "video_fps")
        or _attr_float(group.attrs, "fps", "video_fps")
        or 30.0
    )
    total_frames = (
        _attr_int(root.attrs, "total_frames", "n_frames", "source_video_total_frames")
        or _attr_int(group.attrs, "total_frames", "n_frames")
        or 0
    )
    return int(width), int(height), float(fps), int(total_frames)


def _read_detection_centers(
    group: zarr.Group, *, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    if "frame_indices" not in group:
        raise ValueError("Detection group missing frame_indices.")
    frame_indices = np.asarray(group["frame_indices"][:], dtype=np.int64).reshape(-1)
    if "bbox_img_xyxy" in group:
        bbox = np.asarray(group["bbox_img_xyxy"][:], dtype=np.float64)
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise ValueError("bbox_img_xyxy must have shape (N, 4).")
        centers = np.column_stack(
            [(bbox[:, 0] + bbox[:, 2]) * 0.5, (bbox[:, 1] + bbox[:, 3]) * 0.5]
        )
    elif "bbox_norm_coords" in group:
        bbox = np.asarray(group["bbox_norm_coords"][:], dtype=np.float64)
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise ValueError("bbox_norm_coords must have shape (N, 4).")
        centers = np.column_stack(
            [bbox[:, 0] * float(width), bbox[:, 1] * float(height)]
        )
    else:
        raise ValueError(
            "Detection group has neither bbox_img_xyxy nor bbox_norm_coords."
        )
    if frame_indices.shape[0] != centers.shape[0]:
        raise ValueError("frame_indices and bbox arrays disagree on row count.")
    return frame_indices, centers


def _read_scores(group: zarr.Group) -> Optional[np.ndarray]:
    for name in ("confidence_scores", "scores"):
        if name in group:
            return np.asarray(group[name][:], dtype=np.float64).reshape(-1)
    return None


def _make_edges(size: int, bin_size: int) -> np.ndarray:
    bins = max(1, int(np.ceil(float(size) / float(bin_size))))
    return np.linspace(0.0, float(size), bins + 1)


def compute_heatmap(
    *,
    frames: np.ndarray,
    centers: np.ndarray,
    window: EpochWindow,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    bin_size: int,
    smooth_sigma: float,
    normalize: str,
) -> tuple[np.ndarray, int, int, int, int, int, float]:
    if window.start_frame is not None:
        start_frame = max(0, int(window.start_frame))
    else:
        start_frame = max(0, int(np.floor(window.start_s * fps)))
    if window.end_frame is not None:
        requested_end = int(window.end_frame)
    else:
        requested_end = int(np.ceil(window.end_s * fps)) - 1
    if total_frames > 0:
        end_frame = min(max(0, total_frames - 1), requested_end)
    else:
        end_frame = requested_end
    if end_frame < start_frame:
        end_frame = start_frame

    mask = (frames >= start_frame) & (frames <= end_frame)
    selected_frames = frames[mask]
    selected_centers = centers[mask]
    valid = (
        np.isfinite(selected_centers).all(axis=1)
        if selected_centers.size
        else np.zeros(0, dtype=bool)
    )
    selected_frames = selected_frames[valid]
    selected_centers = selected_centers[valid]
    in_bounds = (
        (selected_centers[:, 0] >= 0)
        & (selected_centers[:, 0] <= width)
        & (selected_centers[:, 1] >= 0)
        & (selected_centers[:, 1] <= height)
    )
    selected_frames = selected_frames[in_bounds]
    selected_centers = selected_centers[in_bounds]

    x_edges = _make_edges(width, bin_size)
    y_edges = _make_edges(height, bin_size)
    heatmap, _, _ = np.histogram2d(
        selected_centers[:, 0], selected_centers[:, 1], bins=[x_edges, y_edges]
    )
    heatmap = heatmap.T
    if smooth_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=float(smooth_sigma))

    if normalize == "max":
        max_value = float(np.max(heatmap)) if heatmap.size else 0.0
        if max_value > 0:
            heatmap = heatmap / max_value
    elif normalize == "density":
        total = float(np.sum(heatmap))
        if total > 0:
            heatmap = heatmap / total
    elif normalize != "count":
        raise ValueError(f"Unknown normalization mode: {normalize}")

    span = max(0, end_frame - start_frame + 1)
    covered = int(np.unique(selected_frames).shape[0]) if selected_frames.size else 0
    coverage = float(covered / span * 100.0) if span else 0.0
    return (
        heatmap,
        start_frame,
        end_frame,
        span,
        int(selected_centers.shape[0]),
        covered,
        coverage,
    )


def build_heatmaps_for_archive(
    zarr_path: Path,
    *,
    windows: Optional[Sequence[EpochWindow]],
    windows_from_stimulus: bool,
    stimulus_run: Optional[str],
    source: str,
    detect_run: Optional[str],
    detection_path: Optional[str],
    bin_size: int,
    smooth_sigma: float,
    normalize: str,
    min_score: Optional[float],
) -> list[HeatmapResult]:
    root = _open_root(zarr_path)
    group, source_path, source_kind = _resolve_detection_group(
        root,
        source=source,
        detect_run=detect_run,
        detection_path=detection_path,
    )
    width, height, fps, total_frames = _resolve_dimensions(root, group)
    resolved_windows = (
        resolve_stimulus_event_windows(
            root, fps=fps, total_frames=total_frames, stimulus_run=stimulus_run
        )
        if windows_from_stimulus
        else list(windows or [])
    )
    if not resolved_windows:
        raise ValueError("No heatmap windows resolved.")
    frames, centers = _read_detection_centers(group, width=width, height=height)
    scores = _read_scores(group)
    if min_score is not None and scores is not None:
        if scores.shape[0] != frames.shape[0]:
            raise ValueError("Score array length does not match frame_indices.")
        keep = scores >= float(min_score)
        frames = frames[keep]
        centers = centers[keep]

    recording_id = (
        _attr_text(root.attrs, "recording_id", "recording_name") or zarr_path.stem
    )
    arena_id = _attr_text(root.attrs, "arena_id")
    camera_id = _attr_text(root.attrs, "camera_id", "camera_serial")

    results: list[HeatmapResult] = []
    for window in resolved_windows:
        heatmap, start_frame, end_frame, span, detections, covered, coverage = (
            compute_heatmap(
                frames=frames,
                centers=centers,
                window=window,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                bin_size=bin_size,
                smooth_sigma=smooth_sigma,
                normalize=normalize,
            )
        )
        results.append(
            HeatmapResult(
                recording_id=recording_id,
                arena_id=arena_id,
                camera_id=camera_id,
                zarr_path=str(zarr_path),
                source_path=source_path,
                source_kind=source_kind,
                width=width,
                height=height,
                fps=fps,
                window=window,
                start_frame=start_frame,
                end_frame=end_frame,
                total_span_frames=span,
                detection_count=detections,
                covered_frame_count=covered,
                coverage_pct=coverage,
                heatmap=heatmap,
            )
        )
    return results


def _recording_label(result: HeatmapResult) -> str:
    parts = [result.recording_id]
    extra = " ".join(part for part in (result.arena_id, result.camera_id) if part)
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _window_title(window: EpochWindow) -> str:
    frame_text = ""
    if window.start_frame is not None and window.end_frame is not None:
        frame_text = f"\nframes {window.start_frame}-{window.end_frame}"
    return f"{window.label.replace('_', ' ')}\n{window.start_s / 60:.2f}-{window.end_s / 60:.2f} min{frame_text}"


def render_heatmap_panel(
    results_by_recording: Sequence[Sequence[HeatmapResult]],
    *,
    output: Path,
    title: str,
    origin: str,
    cmap: str,
    normalize: str,
) -> None:
    if not results_by_recording:
        raise ValueError("No heatmap results to plot.")
    n_rows = len(results_by_recording)
    n_cols = max(len(row) for row in results_by_recording)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    all_maps = [
        item.heatmap
        for row in results_by_recording
        for item in row
        if item.heatmap.size
    ]
    if normalize == "count":
        finite_values = (
            np.concatenate([arr[np.isfinite(arr)].reshape(-1) for arr in all_maps])
            if all_maps
            else np.array([])
        )
        vmax = float(np.percentile(finite_values, 99.0)) if finite_values.size else 1.0
        if vmax <= 0:
            vmax = 1.0
    elif normalize == "density":
        finite_values = (
            np.concatenate(
                [arr[np.isfinite(arr) & (arr > 0)].reshape(-1) for arr in all_maps]
            )
            if all_maps
            else np.array([])
        )
        vmax = float(np.percentile(finite_values, 99.5)) if finite_values.size else 1.0
        if vmax <= 0:
            vmax = 1.0
    else:
        vmax = 1.0

    image = None
    for row_idx, row in enumerate(results_by_recording):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(row):
                ax.axis("off")
                continue
            item = row[col_idx]
            image = ax.imshow(
                item.heatmap,
                cmap=cmap,
                origin=origin,
                extent=(
                    [0, item.width, item.height, 0]
                    if origin == "upper"
                    else [0, item.width, 0, item.height]
                ),
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            if row_idx == 0:
                ax.set_title(_window_title(item.window), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(_recording_label(item), fontsize=8)
            ax.set_xlabel("x px")
            stats = (
                f"n={item.detection_count:,}\n"
                f"frames={item.covered_frame_count:,}/{item.total_span_frames:,}\n"
                f"cov={item.coverage_pct:.1f}%"
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
                bbox={
                    "boxstyle": "round",
                    "facecolor": "black",
                    "alpha": 0.55,
                    "linewidth": 0,
                },
            )
    if image is not None:
        label = {
            "max": "Max-normalized occupancy",
            "density": "Detection density",
            "count": "Detection count",
        }[normalize]
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label=label)
    fig.suptitle(title, fontsize=13, y=1.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    results_by_recording: Sequence[Sequence[HeatmapResult]], path: Path
) -> None:
    payload: list[dict[str, Any]] = []
    for row in results_by_recording:
        for item in row:
            payload.append(
                {
                    "recording_id": item.recording_id,
                    "arena_id": item.arena_id,
                    "camera_id": item.camera_id,
                    "zarr_path": item.zarr_path,
                    "source_path": item.source_path,
                    "source_kind": item.source_kind,
                    "fps": item.fps,
                    "width": item.width,
                    "height": item.height,
                    "window": {
                        "label": item.window.label,
                        "source": item.window.source,
                        "start_s": item.window.start_s,
                        "end_s": item.window.end_s,
                        "start_frame": item.window.start_frame,
                        "end_frame": item.window.end_frame,
                        "duration_s": item.window.duration_s,
                    },
                    "start_frame": item.start_frame,
                    "end_frame": item.end_frame,
                    "total_span_frames": item.total_span_frames,
                    "detection_count": item.detection_count,
                    "covered_frame_count": item.covered_frame_count,
                    "coverage_pct": item.coverage_pct,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr", type=Path, nargs="+", help="Analysis Zarr archive(s).")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the heatmap panel.",
    )
    parser.add_argument(
        "--summary-json", type=Path, help="Optional JSON summary output path."
    )
    parser.add_argument(
        "--source",
        choices=("active", "raw"),
        default="active",
        help="Detection source. 'active' prefers curated/refined surfaces; 'raw' uses detect_runs latest.",
    )
    parser.add_argument(
        "--detect-run", help="Raw detect run name when --source raw is used."
    )
    parser.add_argument(
        "--detection-path", help="Explicit detection group path inside the archive."
    )
    parser.add_argument("--first-minutes", type=float, default=10.0)
    parser.add_argument("--training-minutes", type=float, default=2.5)
    parser.add_argument("--post-minutes", type=float, default=10.0)
    parser.add_argument("--start-second", type=float, default=0.0)
    parser.add_argument(
        "--windows-from-stimulus",
        action="store_true",
        help=(
            "Use CHASER_PRE_PERIOD_START, CHASER_TRAINING_START, "
            "CHASER_POST_PERIOD_START, and protocol-finish event camera frames."
        ),
    )
    parser.add_argument(
        "--stimulus-run", help="Stimulus run name for --windows-from-stimulus."
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Custom window as label:start_s:end_s. Repeatable. Overrides duration defaults.",
    )
    parser.add_argument(
        "--bin-size", type=int, default=64, help="Heatmap bin size in pixels."
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in heatmap bins.",
    )
    parser.add_argument(
        "--min-score", type=float, help="Optional minimum detection confidence."
    )
    parser.add_argument(
        "--normalize", choices=("max", "density", "count"), default="max"
    )
    parser.add_argument("--origin", choices=("upper", "lower"), default="upper")
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--title", default="Detection centroid heatmaps")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.windows_from_stimulus and args.window:
        raise ValueError("--window cannot be combined with --windows-from-stimulus.")
    windows = None
    if not args.windows_from_stimulus:
        windows = (
            parse_window_specs(args.window)
            if args.window
            else default_windows(
                first_minutes=float(args.first_minutes),
                training_minutes=float(args.training_minutes),
                post_minutes=float(args.post_minutes),
                start_second=float(args.start_second),
            )
        )
    results_by_recording = [
        build_heatmaps_for_archive(
            zarr_path=path,
            windows=windows,
            windows_from_stimulus=bool(args.windows_from_stimulus),
            stimulus_run=args.stimulus_run,
            source=str(args.source),
            detect_run=args.detect_run,
            detection_path=args.detection_path,
            bin_size=int(args.bin_size),
            smooth_sigma=float(args.smooth_sigma),
            normalize=str(args.normalize),
            min_score=args.min_score,
        )
        for path in args.zarr
    ]
    render_heatmap_panel(
        results_by_recording,
        output=args.output,
        title=str(args.title),
        origin=str(args.origin),
        cmap=str(args.cmap),
        normalize=str(args.normalize),
    )
    if args.summary_json:
        write_summary(results_by_recording, args.summary_json)
    for row in results_by_recording:
        if not row:
            continue
        first = row[0]
        print(f"{first.recording_id}: source={first.source_path} fps={first.fps:g}")
        for item in row:
            print(
                f"  {item.window.label}: frames={item.start_frame}-{item.end_frame} "
                f"detections={item.detection_count} coverage={item.coverage_pct:.1f}%"
            )
    print(f"saved: {args.output}")
    if args.summary_json:
        print(f"summary_json: {args.summary_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
