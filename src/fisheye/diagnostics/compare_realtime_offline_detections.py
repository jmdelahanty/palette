"""Compare acquisition-time realtime boxes against offline/refined detections.

The diagnostic writes optional run-local arrays under:

``analysis/detection_comparison_runs/<run>/``

and stores a compact PNG summary under that run's ``visualizations/`` group.
Numeric arrays are the source of truth; the PNG is a QC snapshot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
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

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.plot_artifacts import write_png_visualization_artifact
from fisheye.shared.refined_detect_resolution import resolve_detection_read_source
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_failed,
    mark_run_pending,
    mark_run_started,
    require_runs_parent,
    resolve_authoritative_run_name,
)
from fisheye.visualization.plot_detection_epoch_heatmaps import EpochWindow
from fisheye.visualization.plot_detection_epoch_heatmaps import resolve_stimulus_event_windows


SCHEMA_ID = "palette.detection_realtime_offline_comparison.v1"
PNG_ARTIFACT_NAME = "realtime_offline_detection_comparison_png"
DEFAULT_PARENT_PATH = "analysis/detection_comparison_runs"
DEFAULT_SCATTER_LIMIT = 50_000
DEFAULT_BAD_EXAMPLE_LIMIT = 25
CROP_REASON_LABELS = {
    0: "unassigned",
    1: "offline_absent",
    2: "inside_crop",
    3: "missing_crop_row",
    4: "blank_frame",
    5: "no_realtime_detection",
    6: "crop_elsewhere",
}


@dataclass(frozen=True)
class DetectionRows:
    source_path: str
    source_kind: str
    run_name: Optional[str]
    frame_indices: np.ndarray
    bbox_img_xyxy: np.ndarray
    centers_xy: np.ndarray
    confidence: np.ndarray
    row_indices: np.ndarray


@dataclass(frozen=True)
class CropMetaRows:
    source_path: str
    frame_indices: np.ndarray
    crop_xywh: np.ndarray
    has_detection: np.ndarray
    blank_frame: np.ndarray
    row_indices: np.ndarray


@dataclass(frozen=True)
class ComparisonArrays:
    frame_indices: np.ndarray
    offline_present: np.ndarray
    realtime_present: np.ndarray
    offline_row_index: np.ndarray
    realtime_row_index: np.ndarray
    offline_center_xy: np.ndarray
    realtime_center_xy: np.ndarray
    offline_bbox_img_xyxy: np.ndarray
    realtime_bbox_img_xyxy: np.ndarray
    offline_confidence: np.ndarray
    realtime_confidence: np.ndarray
    centroid_delta_px: np.ndarray
    bbox_iou: np.ndarray
    epoch_label_code: np.ndarray
    realtime_crop_xywh: Optional[np.ndarray] = None
    realtime_crop_has_detection: Optional[np.ndarray] = None
    realtime_crop_blank_frame: Optional[np.ndarray] = None
    offline_center_inside_realtime_crop: Optional[np.ndarray] = None
    offline_bbox_inside_realtime_crop: Optional[np.ndarray] = None
    offline_crop_edge_margins: Optional[np.ndarray] = None
    crop_sufficiency_reason_code: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ComparisonResult:
    zarr_path: str
    recording_id: str
    offline_source_path: str
    offline_source_kind: str
    offline_run_name: Optional[str]
    realtime_source_path: str
    realtime_source_kind: str
    stimulus_run_name: str
    run_name: str
    width: int
    height: int
    fps: float
    total_frames: int
    realtime_frame_offset: int
    arrays: ComparisonArrays
    summary: dict[str, Any]
    epoch_windows: tuple[EpochWindow, ...]


def utc_run_name(prefix: str = "detection_comparison") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{stamp}"


def _open_root(zarr_path: Path, *, mode: str = "r") -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def infer_recording_dir_from_zarr(zarr_path: Path) -> Path:
    zarr_path = Path(zarr_path)
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    for parent in zarr_path.parents:
        if (parent / "recording_manifest.json").exists():
            return parent
    return zarr_path.parent


def _manifest_crop_meta_path(recording_dir: Path) -> Optional[Path]:
    manifest_path = recording_dir / "recording_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    streams = manifest.get("video_streams")
    if not isinstance(streams, dict):
        return None
    stream_entries = streams.get("streams")
    if not isinstance(stream_entries, dict):
        return None
    crop_stream = stream_entries.get("crop")
    if not isinstance(crop_stream, dict):
        return None
    metadata = crop_stream.get("metadata")
    if not metadata:
        return None
    path = Path(str(metadata))
    if not path.is_absolute():
        path = recording_dir / path
    return path


def resolve_crop_meta_path(
    zarr_path: Path,
    *,
    recording_dir: Optional[Path] = None,
    crop_meta_path: Optional[Path] = None,
    required: bool = False,
) -> Optional[Path]:
    if crop_meta_path is not None:
        path = Path(crop_meta_path)
        if not path.exists():
            raise ValueError(f"Explicit crop metadata path does not exist: {path}")
        return path

    root_dir = Path(recording_dir) if recording_dir is not None else infer_recording_dir_from_zarr(zarr_path)
    manifest_path = _manifest_crop_meta_path(root_dir)
    if manifest_path is not None and manifest_path.exists():
        return manifest_path

    candidates = sorted((root_dir / "derived" / "external_crop_recorder").glob("*_crop_meta.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(
            f"Multiple crop metadata files found under {root_dir}; pass --crop-meta explicitly."
        )
    if required:
        raise ValueError(f"No external crop recorder metadata found for recording dir: {root_dir}")
    return None


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


def _resolve_dimensions(root: zarr.Group, groups: Sequence[zarr.Group]) -> tuple[int, int, float, int]:
    width = _attr_int(root.attrs, "width", "video_width", "source_video_width", "palette_video_width")
    height = _attr_int(root.attrs, "height", "video_height", "source_video_height", "palette_video_height")
    fps = _attr_float(root.attrs, "fps", "video_fps")
    total_frames = _attr_int(root.attrs, "total_frames", "n_frames", "source_video_total_frames")
    for group in groups:
        width = width or _attr_int(group.attrs, "source_full_width", "source_video_width", "width", "video_width")
        height = height or _attr_int(group.attrs, "source_full_height", "source_video_height", "height", "video_height")
        fps = fps or _attr_float(group.attrs, "fps", "video_fps")
        total_frames = total_frames or _attr_int(group.attrs, "total_frames", "n_frames")
    return int(width or 4512), int(height or 4512), float(fps or 30.0), int(total_frames or 0)


def _resolve_raw_detect_group(root: zarr.Group, run_name: Optional[str]) -> tuple[zarr.Group, str, str]:
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
    return parent[resolved], f"detect_runs/{resolved}", resolved


def _resolve_refined_detect_group(root: zarr.Group, run_name: Optional[str]) -> tuple[zarr.Group, str, str]:
    parent = root.get("refined_detect_runs")
    if parent is None:
        raise ValueError("Archive has no refined_detect_runs group.")
    resolved = str(run_name).strip() if run_name else None
    if not resolved:
        resolved = resolve_authoritative_run_name(parent)
    if not resolved:
        latest = parent.attrs.get("latest")
        resolved = str(latest).strip() if latest else None
    if not resolved or resolved not in parent:
        raise ValueError("No usable refined detect run found; pass --refined-run.")
    group = parent[resolved]
    if "instances" in group:
        return group["instances"], f"refined_detect_runs/{resolved}/instances", resolved
    return group, f"refined_detect_runs/{resolved}", resolved


def resolve_offline_detection_group(
    root: zarr.Group,
    *,
    offline_source: str,
    detection_path: Optional[str],
    detect_run: Optional[str],
    refined_run: Optional[str],
) -> tuple[zarr.Group, str, str, Optional[str]]:
    if detection_path:
        path = str(detection_path).strip().strip("/")
        if not path:
            raise ValueError("--detection-path must be non-empty.")
        run_name = path.split("/")[1] if "/" in path else None
        return root[path], path, "explicit", run_name
    if offline_source == "raw":
        group, path, run = _resolve_raw_detect_group(root, detect_run)
        return group, path, "raw", run
    if offline_source == "refined":
        group, path, run = _resolve_refined_detect_group(root, refined_run)
        return group, path, "refined", run

    resolution = resolve_detection_read_source(root, prefer_curated=True, allow_sparse_fallback=True)
    if not resolution.detection_path:
        raise ValueError("No active offline detection source resolved.")
    run_name = None
    parts = resolution.detection_path.split("/")
    if len(parts) >= 2:
        run_name = parts[1]
    return root[resolution.detection_path], resolution.detection_path, resolution.detection_kind or "active", run_name


def _read_scores(group: zarr.Group, row_count: int) -> np.ndarray:
    for name in ("confidence_scores", "scores", "confidence"):
        if name in group:
            values = np.asarray(group[name][:], dtype=np.float64).reshape(-1)
            if values.shape[0] == row_count:
                return values
    return np.full((row_count,), np.nan, dtype=np.float64)


def _bbox_from_detection_group(group: zarr.Group, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if "bbox_img_xyxy" in group:
        bbox = np.asarray(group["bbox_img_xyxy"][:], dtype=np.float64)
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise ValueError("bbox_img_xyxy must have shape (N, 4).")
        centers = np.column_stack([(bbox[:, 0] + bbox[:, 2]) * 0.5, (bbox[:, 1] + bbox[:, 3]) * 0.5])
        return bbox, centers
    if "bbox_norm_coords" not in group:
        raise ValueError("Offline detection group has neither bbox_img_xyxy nor bbox_norm_coords.")
    norm = np.asarray(group["bbox_norm_coords"][:], dtype=np.float64)
    if norm.ndim != 2 or norm.shape[1] != 4:
        raise ValueError("bbox_norm_coords must have shape (N, 4).")
    cx = norm[:, 0] * float(width)
    cy = norm[:, 1] * float(height)
    bw = norm[:, 2] * float(width)
    bh = norm[:, 3] * float(height)
    bbox = np.column_stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5])
    centers = np.column_stack([cx, cy])
    return bbox, centers


def _select_top_one_per_frame(
    frames: np.ndarray,
    *,
    confidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    if frames.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if confidence is None:
        confidence = np.full(frames.shape, np.nan, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    safe_conf = np.where(np.isfinite(confidence), confidence, -np.inf)
    order = np.lexsort((-safe_conf, frames))
    sorted_frames = frames[order]
    keep = np.r_[True, sorted_frames[1:] != sorted_frames[:-1]]
    return order[keep].astype(np.int64, copy=False)


def load_offline_detection_rows(
    root: zarr.Group,
    *,
    offline_source: str,
    detection_path: Optional[str],
    detect_run: Optional[str],
    refined_run: Optional[str],
) -> DetectionRows:
    group, source_path, source_kind, run_name = resolve_offline_detection_group(
        root,
        offline_source=offline_source,
        detection_path=detection_path,
        detect_run=detect_run,
        refined_run=refined_run,
    )
    width, height, _fps, _total = _resolve_dimensions(root, [group])
    if "frame_indices" not in group:
        raise ValueError(f"Offline detection group missing frame_indices: {source_path}")
    frames = np.asarray(group["frame_indices"][:], dtype=np.int64).reshape(-1)
    bbox, centers = _bbox_from_detection_group(group, width=width, height=height)
    if frames.shape[0] != bbox.shape[0]:
        raise ValueError("Offline frame_indices and bbox arrays disagree on row count.")
    confidence = _read_scores(group, frames.shape[0])
    valid = np.isfinite(centers).all(axis=1) & np.isfinite(bbox).all(axis=1) & (frames >= 0)
    rows = np.arange(frames.shape[0], dtype=np.int64)
    frames = frames[valid]
    bbox = bbox[valid]
    centers = centers[valid]
    confidence = confidence[valid]
    rows = rows[valid]
    selected = _select_top_one_per_frame(frames, confidence=confidence)
    return DetectionRows(
        source_path=source_path,
        source_kind=source_kind,
        run_name=run_name,
        frame_indices=frames[selected],
        bbox_img_xyxy=bbox[selected],
        centers_xy=centers[selected],
        confidence=confidence[selected],
        row_indices=rows[selected],
    )


def resolve_stimulus_run(root: zarr.Group, stimulus_run: Optional[str]) -> tuple[zarr.Group, str, str]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        raise ValueError("Archive has no analysis/stimulus_runs group.")
    parent = analysis["stimulus_runs"]
    resolved = str(stimulus_run).strip() if stimulus_run else None
    if not resolved:
        resolved = resolve_authoritative_run_name(parent)
    if not resolved:
        names = sorted(str(name) for name in parent.group_keys())
        resolved = names[-1] if names else None
    if not resolved or resolved not in parent:
        raise ValueError("No usable stimulus run found; pass --stimulus-run.")
    return parent[resolved], f"analysis/stimulus_runs/{resolved}", resolved


def load_realtime_detection_rows(
    root: zarr.Group,
    *,
    stimulus_run: Optional[str],
    frame_offset: int,
) -> DetectionRows:
    return load_stimulus_h5_realtime_detection_rows(
        root,
        stimulus_run=stimulus_run,
        frame_offset=frame_offset,
    )


def load_stimulus_h5_realtime_detection_rows(
    root: zarr.Group,
    *,
    stimulus_run: Optional[str],
    frame_offset: int,
) -> DetectionRows:
    stim_group, stim_path, resolved = resolve_stimulus_run(root, stimulus_run)
    bbox_group = stim_group.get("tracking_data/bounding_boxes")
    if bbox_group is None:
        raise ValueError(f"Stimulus run {resolved!r} has no tracking_data/bounding_boxes group.")
    required = ("payload_frame_id", "x_min", "y_min", "width", "height")
    missing = [name for name in required if name not in bbox_group]
    if missing:
        raise ValueError(f"Realtime bounding_boxes missing required arrays: {missing}")
    frames = np.asarray(bbox_group["payload_frame_id"][:], dtype=np.int64).reshape(-1) + int(frame_offset)
    x = np.asarray(bbox_group["x_min"][:], dtype=np.float64).reshape(-1)
    y = np.asarray(bbox_group["y_min"][:], dtype=np.float64).reshape(-1)
    w = np.asarray(bbox_group["width"][:], dtype=np.float64).reshape(-1)
    h = np.asarray(bbox_group["height"][:], dtype=np.float64).reshape(-1)
    bbox = np.column_stack([x, y, x + w, y + h])
    if "centroid_x" in bbox_group and "centroid_y" in bbox_group:
        centers = np.column_stack(
            [
                np.asarray(bbox_group["centroid_x"][:], dtype=np.float64).reshape(-1),
                np.asarray(bbox_group["centroid_y"][:], dtype=np.float64).reshape(-1),
            ]
        )
    else:
        centers = np.column_stack([x + w * 0.5, y + h * 0.5])
    confidence = _read_scores(bbox_group, frames.shape[0])
    valid = (
        np.isfinite(centers).all(axis=1)
        & np.isfinite(bbox).all(axis=1)
        & (frames >= 0)
        & (w >= 0)
        & (h >= 0)
    )
    rows = np.arange(frames.shape[0], dtype=np.int64)
    frames = frames[valid]
    bbox = bbox[valid]
    centers = centers[valid]
    confidence = confidence[valid]
    rows = rows[valid]
    selected = _select_top_one_per_frame(frames, confidence=confidence)
    return DetectionRows(
        source_path=f"{stim_path}/tracking_data/bounding_boxes",
        source_kind="realtime_tracking",
        run_name=resolved,
        frame_indices=frames[selected],
        bbox_img_xyxy=bbox[selected],
        centers_xy=centers[selected],
        confidence=confidence[selected],
        row_indices=rows[selected],
    )


def _csv_float(row: dict[str, str], name: str) -> float:
    value = row.get(name)
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _csv_int(row: dict[str, str], name: str, *, default: int = 0) -> int:
    value = row.get(name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _dedupe_first_by_frame(
    frames: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    if frames.size == 0:
        return (frames, *arrays)
    order = np.argsort(frames, kind="stable")
    sorted_frames = frames[order]
    keep = np.r_[True, sorted_frames[1:] != sorted_frames[:-1]]
    selected = order[keep]
    return (frames[selected], *(np.asarray(array)[selected] for array in arrays))


def load_crop_meta_realtime_detection_rows(
    crop_meta_path: Path,
    *,
    frame_offset: int = 0,
) -> tuple[DetectionRows, CropMetaRows]:
    frames_all: list[int] = []
    crop_xywh_all: list[tuple[float, float, float, float]] = []
    has_detection_all: list[bool] = []
    blank_frame_all: list[bool] = []
    row_indices_all: list[int] = []

    frames_det: list[int] = []
    bbox_det: list[tuple[float, float, float, float]] = []
    centers_det: list[tuple[float, float]] = []
    confidence_det: list[float] = []
    row_indices_det: list[int] = []

    with Path(crop_meta_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "recording_frame_id",
            "has_detection",
            "blank_frame",
            "detection_confidence",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
            "detection_x",
            "detection_y",
            "detection_w",
            "detection_h",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Crop metadata missing required columns: {missing}")
        for row_index, row in enumerate(reader):
            frame = _csv_int(row, "recording_frame_id", default=0) - 1 + int(frame_offset)
            has_detection = bool(_csv_int(row, "has_detection", default=0))
            blank_frame = bool(_csv_int(row, "blank_frame", default=0))
            crop_x = _csv_float(row, "crop_x")
            crop_y = _csv_float(row, "crop_y")
            crop_w = _csv_float(row, "crop_w")
            crop_h = _csv_float(row, "crop_h")

            frames_all.append(frame)
            crop_xywh_all.append((crop_x, crop_y, crop_w, crop_h))
            has_detection_all.append(has_detection)
            blank_frame_all.append(blank_frame)
            row_indices_all.append(row_index)

            det_x = _csv_float(row, "detection_x")
            det_y = _csv_float(row, "detection_y")
            det_w = _csv_float(row, "detection_w")
            det_h = _csv_float(row, "detection_h")
            if (
                frame >= 0
                and has_detection
                and not blank_frame
                and np.isfinite([det_x, det_y, det_w, det_h]).all()
                and det_w >= 0.0
                and det_h >= 0.0
            ):
                frames_det.append(frame)
                bbox_det.append((det_x, det_y, det_x + det_w, det_y + det_h))
                centers_det.append((det_x + det_w * 0.5, det_y + det_h * 0.5))
                confidence_det.append(_csv_float(row, "detection_confidence"))
                row_indices_det.append(row_index)

    frames_all_arr = np.asarray(frames_all, dtype=np.int64)
    crop_xywh_arr = np.asarray(crop_xywh_all, dtype=np.float64).reshape(-1, 4)
    has_detection_arr = np.asarray(has_detection_all, dtype=bool)
    blank_frame_arr = np.asarray(blank_frame_all, dtype=bool)
    row_indices_all_arr = np.asarray(row_indices_all, dtype=np.int64)
    (
        frames_all_arr,
        crop_xywh_arr,
        has_detection_arr,
        blank_frame_arr,
        row_indices_all_arr,
    ) = _dedupe_first_by_frame(
        frames_all_arr,
        crop_xywh_arr,
        has_detection_arr,
        blank_frame_arr,
        row_indices_all_arr,
    )

    frames_det_arr = np.asarray(frames_det, dtype=np.int64)
    bbox_det_arr = np.asarray(bbox_det, dtype=np.float64).reshape(-1, 4)
    centers_det_arr = np.asarray(centers_det, dtype=np.float64).reshape(-1, 2)
    confidence_det_arr = np.asarray(confidence_det, dtype=np.float64)
    row_indices_det_arr = np.asarray(row_indices_det, dtype=np.int64)
    selected = _select_top_one_per_frame(frames_det_arr, confidence=confidence_det_arr)

    source_path = str(Path(crop_meta_path))
    return (
        DetectionRows(
            source_path=source_path,
            source_kind="external_crop_recorder_crop_meta",
            run_name=None,
            frame_indices=frames_det_arr[selected],
            bbox_img_xyxy=bbox_det_arr[selected],
            centers_xy=centers_det_arr[selected],
            confidence=confidence_det_arr[selected],
            row_indices=row_indices_det_arr[selected],
        ),
        CropMetaRows(
            source_path=source_path,
            frame_indices=frames_all_arr,
            crop_xywh=crop_xywh_arr,
            has_detection=has_detection_arr,
            blank_frame=blank_frame_arr,
            row_indices=row_indices_all_arr,
        ),
    )


def compute_bbox_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("bbox arrays must both have shape (N, 4).")
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    valid = union > 0
    out[valid] = inter[valid] / union[valid]
    return out


def _nan_bbox(row_count: int) -> np.ndarray:
    return np.full((row_count, 4), np.nan, dtype=np.float32)


def _nan_centers(row_count: int) -> np.ndarray:
    return np.full((row_count, 2), np.nan, dtype=np.float32)


def compare_detection_rows(
    offline: DetectionRows,
    realtime: DetectionRows,
    *,
    epoch_windows: Sequence[EpochWindow] = (),
    crop_meta: Optional[CropMetaRows] = None,
) -> ComparisonArrays:
    frames = np.union1d(offline.frame_indices, realtime.frame_indices).astype(np.int64, copy=False)
    n = int(frames.shape[0])
    offline_present = np.isin(frames, offline.frame_indices, assume_unique=True)
    realtime_present = np.isin(frames, realtime.frame_indices, assume_unique=True)

    offline_pos = np.searchsorted(offline.frame_indices, frames)
    realtime_pos = np.searchsorted(realtime.frame_indices, frames)
    offline_match = offline_present & (offline_pos < offline.frame_indices.shape[0])
    realtime_match = realtime_present & (realtime_pos < realtime.frame_indices.shape[0])

    offline_centers = _nan_centers(n)
    realtime_centers = _nan_centers(n)
    offline_bbox = _nan_bbox(n)
    realtime_bbox = _nan_bbox(n)
    offline_conf = np.full((n,), np.nan, dtype=np.float32)
    realtime_conf = np.full((n,), np.nan, dtype=np.float32)
    offline_row = np.full((n,), -1, dtype=np.int64)
    realtime_row = np.full((n,), -1, dtype=np.int64)

    offline_centers[offline_match] = offline.centers_xy[offline_pos[offline_match]].astype(np.float32, copy=False)
    realtime_centers[realtime_match] = realtime.centers_xy[realtime_pos[realtime_match]].astype(np.float32, copy=False)
    offline_bbox[offline_match] = offline.bbox_img_xyxy[offline_pos[offline_match]].astype(np.float32, copy=False)
    realtime_bbox[realtime_match] = realtime.bbox_img_xyxy[realtime_pos[realtime_match]].astype(np.float32, copy=False)
    offline_conf[offline_match] = offline.confidence[offline_pos[offline_match]].astype(np.float32, copy=False)
    realtime_conf[realtime_match] = realtime.confidence[realtime_pos[realtime_match]].astype(np.float32, copy=False)
    offline_row[offline_match] = offline.row_indices[offline_pos[offline_match]]
    realtime_row[realtime_match] = realtime.row_indices[realtime_pos[realtime_match]]

    both = offline_present & realtime_present
    delta = np.full((n,), np.nan, dtype=np.float32)
    delta[both] = np.linalg.norm(offline_centers[both] - realtime_centers[both], axis=1).astype(np.float32)
    iou = np.full((n,), np.nan, dtype=np.float32)
    if np.any(both):
        iou[both] = compute_bbox_iou(offline_bbox[both], realtime_bbox[both]).astype(np.float32, copy=False)

    epoch_codes = np.zeros((n,), dtype=np.int16)
    for code, window in enumerate(epoch_windows, start=1):
        if window.start_frame is None or window.end_frame is None:
            continue
        mask = (frames >= int(window.start_frame)) & (frames <= int(window.end_frame))
        epoch_codes[mask] = code

    realtime_crop_xywh = None
    realtime_crop_has_detection = None
    realtime_crop_blank_frame = None
    offline_center_inside_realtime_crop = None
    offline_bbox_inside_realtime_crop = None
    offline_crop_edge_margins = None
    crop_sufficiency_reason_code = None
    if crop_meta is not None:
        crop_pos = np.searchsorted(crop_meta.frame_indices, frames)
        crop_match = np.zeros(n, dtype=bool)
        crop_in_bounds = crop_pos < crop_meta.frame_indices.shape[0]
        crop_match[crop_in_bounds] = crop_meta.frame_indices[crop_pos[crop_in_bounds]] == frames[crop_in_bounds]

        realtime_crop_xywh = np.full((n, 4), np.nan, dtype=np.float32)
        realtime_crop_has_detection = np.zeros((n,), dtype=bool)
        realtime_crop_blank_frame = np.zeros((n,), dtype=bool)
        offline_center_inside_realtime_crop = np.zeros((n,), dtype=bool)
        offline_bbox_inside_realtime_crop = np.zeros((n,), dtype=bool)
        offline_crop_edge_margins = np.full((n, 4), np.nan, dtype=np.float32)
        crop_sufficiency_reason_code = np.zeros((n,), dtype=np.int8)
        crop_sufficiency_reason_code[~offline_present] = 1

        if np.any(crop_match):
            matched_crop_pos = crop_pos[crop_match]
            realtime_crop_xywh[crop_match] = crop_meta.crop_xywh[matched_crop_pos].astype(np.float32, copy=False)
            realtime_crop_has_detection[crop_match] = crop_meta.has_detection[matched_crop_pos]
            realtime_crop_blank_frame[crop_match] = crop_meta.blank_frame[matched_crop_pos]

        offline_crop = offline_present & crop_match
        missing_crop = offline_present & ~crop_match
        blank_crop = offline_crop & realtime_crop_blank_frame
        no_detection_crop = offline_crop & ~realtime_crop_blank_frame & ~realtime_crop_has_detection
        crop_sufficiency_reason_code[missing_crop] = 3
        crop_sufficiency_reason_code[blank_crop] = 4
        crop_sufficiency_reason_code[no_detection_crop] = 5

        crop_valid = offline_crop & np.isfinite(realtime_crop_xywh).all(axis=1)
        if np.any(crop_valid):
            crop_x = realtime_crop_xywh[:, 0].astype(np.float64, copy=False)
            crop_y = realtime_crop_xywh[:, 1].astype(np.float64, copy=False)
            crop_w = realtime_crop_xywh[:, 2].astype(np.float64, copy=False)
            crop_h = realtime_crop_xywh[:, 3].astype(np.float64, copy=False)
            crop_x2 = crop_x + crop_w
            crop_y2 = crop_y + crop_h
            offline_x1 = offline_bbox[:, 0].astype(np.float64, copy=False)
            offline_y1 = offline_bbox[:, 1].astype(np.float64, copy=False)
            offline_x2 = offline_bbox[:, 2].astype(np.float64, copy=False)
            offline_y2 = offline_bbox[:, 3].astype(np.float64, copy=False)
            margins = np.column_stack(
                [
                    offline_x1 - crop_x,
                    offline_y1 - crop_y,
                    crop_x2 - offline_x2,
                    crop_y2 - offline_y2,
                ]
            )
            offline_crop_edge_margins[crop_valid] = margins[crop_valid].astype(np.float32, copy=False)
            usable_crop = crop_valid & realtime_crop_has_detection & ~realtime_crop_blank_frame
            center_x = offline_centers[:, 0].astype(np.float64, copy=False)
            center_y = offline_centers[:, 1].astype(np.float64, copy=False)
            offline_center_inside_realtime_crop[usable_crop] = (
                (center_x[usable_crop] >= crop_x[usable_crop])
                & (center_x[usable_crop] <= crop_x2[usable_crop])
                & (center_y[usable_crop] >= crop_y[usable_crop])
                & (center_y[usable_crop] <= crop_y2[usable_crop])
            )
            offline_bbox_inside_realtime_crop[usable_crop] = np.all(margins[usable_crop] >= 0.0, axis=1)

        eligible = offline_crop & ~realtime_crop_blank_frame & realtime_crop_has_detection
        crop_sufficiency_reason_code[eligible & offline_bbox_inside_realtime_crop] = 2
        crop_sufficiency_reason_code[eligible & ~offline_bbox_inside_realtime_crop] = 6

    return ComparisonArrays(
        frame_indices=frames,
        offline_present=offline_present.astype(bool, copy=False),
        realtime_present=realtime_present.astype(bool, copy=False),
        offline_row_index=offline_row,
        realtime_row_index=realtime_row,
        offline_center_xy=offline_centers,
        realtime_center_xy=realtime_centers,
        offline_bbox_img_xyxy=offline_bbox,
        realtime_bbox_img_xyxy=realtime_bbox,
        offline_confidence=offline_conf,
        realtime_confidence=realtime_conf,
        centroid_delta_px=delta,
        bbox_iou=iou,
        epoch_label_code=epoch_codes,
        realtime_crop_xywh=realtime_crop_xywh,
        realtime_crop_has_detection=realtime_crop_has_detection,
        realtime_crop_blank_frame=realtime_crop_blank_frame,
        offline_center_inside_realtime_crop=offline_center_inside_realtime_crop,
        offline_bbox_inside_realtime_crop=offline_bbox_inside_realtime_crop,
        offline_crop_edge_margins=offline_crop_edge_margins,
        crop_sufficiency_reason_code=crop_sufficiency_reason_code,
    )


def _percentiles(values: np.ndarray, points: Sequence[float] = (50, 90, 95, 99, 100)) -> dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(point)}": None for point in points}
    vals = np.percentile(arr, points)
    return {f"p{int(point)}": float(value) for point, value in zip(points, vals)}


def summarize_comparison(
    arrays: ComparisonArrays,
    *,
    total_frames: int,
    epoch_windows: Sequence[EpochWindow],
    crop_meta: Optional[CropMetaRows] = None,
) -> dict[str, Any]:
    both = arrays.offline_present & arrays.realtime_present
    offline_only = arrays.offline_present & ~arrays.realtime_present
    realtime_only = arrays.realtime_present & ~arrays.offline_present
    neither_count = None
    if total_frames > 0:
        union_count = int(np.count_nonzero(arrays.offline_present | arrays.realtime_present))
        neither_count = max(0, int(total_frames) - union_count)
    summary: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "total_frames": int(total_frames),
        "comparison_frame_count": int(arrays.frame_indices.shape[0]),
        "offline_present_count": int(np.count_nonzero(arrays.offline_present)),
        "realtime_present_count": int(np.count_nonzero(arrays.realtime_present)),
        "both_present_count": int(np.count_nonzero(both)),
        "offline_only_count": int(np.count_nonzero(offline_only)),
        "realtime_only_count": int(np.count_nonzero(realtime_only)),
        "neither_present_count": neither_count,
        "centroid_delta_px": _percentiles(arrays.centroid_delta_px),
        "bbox_iou": _percentiles(arrays.bbox_iou, points=(0, 1, 5, 10, 50)),
    }
    if total_frames > 0:
        summary["offline_coverage_pct"] = float(np.count_nonzero(arrays.offline_present) / total_frames * 100.0)
        summary["realtime_coverage_pct"] = float(np.count_nonzero(arrays.realtime_present) / total_frames * 100.0)
        summary["both_coverage_pct"] = float(np.count_nonzero(both) / total_frames * 100.0)

    if arrays.crop_sufficiency_reason_code is not None:
        offline_count = int(np.count_nonzero(arrays.offline_present))
        center_inside_mask = np.asarray(arrays.offline_center_inside_realtime_crop, dtype=bool)
        bbox_inside_mask = np.asarray(arrays.offline_bbox_inside_realtime_crop, dtype=bool)
        center_inside = int(np.count_nonzero(arrays.offline_present & center_inside_mask))
        bbox_inside = int(np.count_nonzero(arrays.offline_present & bbox_inside_mask))
        reason_counts = {
            CROP_REASON_LABELS[int(code)]: int(count)
            for code, count in zip(
                *np.unique(arrays.crop_sufficiency_reason_code.astype(np.int16, copy=False), return_counts=True)
            )
        }
        summary.update(
            {
                "crop_sufficiency_available": True,
                "crop_sufficiency_reason_labels": CROP_REASON_LABELS,
                "crop_sufficiency_reason_counts": reason_counts,
                "offline_center_inside_crop_count": center_inside,
                "offline_center_inside_crop_pct": (
                    float(center_inside / offline_count * 100.0) if offline_count else None
                ),
                "offline_full_bbox_inside_crop_count": bbox_inside,
                "offline_full_bbox_inside_crop_pct": (
                    float(bbox_inside / offline_count * 100.0) if offline_count else None
                ),
                "blank_crop_rows_for_offline_count": int(
                    np.count_nonzero(arrays.crop_sufficiency_reason_code == 4)
                ),
                "no_detection_crop_rows_for_offline_count": int(
                    np.count_nonzero(arrays.crop_sufficiency_reason_code == 5)
                ),
                "missing_crop_rows_for_offline_count": int(
                    np.count_nonzero(arrays.crop_sufficiency_reason_code == 3)
                ),
                "crop_elsewhere_rows_for_offline_count": int(
                    np.count_nonzero(arrays.crop_sufficiency_reason_code == 6)
                ),
            }
        )
        if crop_meta is not None:
            summary.update(
                {
                    "crop_meta_source_path": crop_meta.source_path,
                    "crop_meta_row_count": int(crop_meta.frame_indices.shape[0]),
                    "crop_meta_detection_row_count": int(np.count_nonzero(crop_meta.has_detection)),
                    "crop_meta_blank_row_count": int(np.count_nonzero(crop_meta.blank_frame)),
                }
            )
    else:
        summary["crop_sufficiency_available"] = False

    epoch_summaries: dict[str, Any] = {}
    for code, window in enumerate(epoch_windows, start=1):
        mask = arrays.epoch_label_code == code
        if not np.any(mask):
            continue
        span = None
        if window.start_frame is not None and window.end_frame is not None:
            span = max(0, int(window.end_frame) - int(window.start_frame) + 1)
        epoch_summaries[window.label] = {
            "start_frame": window.start_frame,
            "end_frame": window.end_frame,
            "span_frames": span,
            "offline_present_count": int(np.count_nonzero(arrays.offline_present[mask])),
            "realtime_present_count": int(np.count_nonzero(arrays.realtime_present[mask])),
            "both_present_count": int(np.count_nonzero(both[mask])),
            "offline_only_count": int(np.count_nonzero(offline_only[mask])),
            "realtime_only_count": int(np.count_nonzero(realtime_only[mask])),
            "centroid_delta_px": _percentiles(arrays.centroid_delta_px[mask]),
            "bbox_iou": _percentiles(arrays.bbox_iou[mask], points=(0, 1, 5, 10, 50)),
        }
    summary["epochs"] = epoch_summaries
    return summary


def load_realtime_source(
    root: zarr.Group,
    zarr_path: Path,
    *,
    realtime_source: str,
    stimulus_run: Optional[str],
    frame_offset: int,
    recording_dir: Optional[Path],
    crop_meta_path: Optional[Path],
) -> tuple[DetectionRows, Optional[CropMetaRows]]:
    if realtime_source not in {"auto", "crop-meta", "stimulus-h5"}:
        raise ValueError(f"Unsupported realtime source: {realtime_source}")

    if realtime_source in {"auto", "crop-meta"}:
        resolved_crop_meta = resolve_crop_meta_path(
            zarr_path,
            recording_dir=recording_dir,
            crop_meta_path=crop_meta_path,
            required=realtime_source == "crop-meta",
        )
        if resolved_crop_meta is not None:
            return load_crop_meta_realtime_detection_rows(resolved_crop_meta, frame_offset=frame_offset)
        if realtime_source == "crop-meta":
            raise ValueError("No crop metadata resolved for --realtime-source crop-meta.")

    realtime = load_stimulus_h5_realtime_detection_rows(
        root,
        stimulus_run=stimulus_run,
        frame_offset=frame_offset,
    )
    return realtime, None


def build_comparison_result(
    zarr_path: Path,
    *,
    offline_source: str,
    detection_path: Optional[str],
    detect_run: Optional[str],
    refined_run: Optional[str],
    realtime_source: str,
    stimulus_run: Optional[str],
    realtime_frame_offset: int,
    recording_dir: Optional[Path],
    crop_meta_path: Optional[Path],
    run_name: Optional[str],
) -> ComparisonResult:
    root = _open_root(zarr_path, mode="r")
    offline = load_offline_detection_rows(
        root,
        offline_source=offline_source,
        detection_path=detection_path,
        detect_run=detect_run,
        refined_run=refined_run,
    )
    realtime, crop_meta = load_realtime_source(
        root,
        zarr_path,
        realtime_source=realtime_source,
        stimulus_run=stimulus_run,
        frame_offset=realtime_frame_offset,
        recording_dir=recording_dir,
        crop_meta_path=crop_meta_path,
    )
    width, height, fps, total_frames = _resolve_dimensions(root, [])
    try:
        epoch_windows = tuple(
            resolve_stimulus_event_windows(root, fps=fps, total_frames=total_frames, stimulus_run=stimulus_run)
        )
    except Exception:
        epoch_windows = ()
    arrays = compare_detection_rows(offline, realtime, epoch_windows=epoch_windows, crop_meta=crop_meta)
    summary = summarize_comparison(arrays, total_frames=total_frames, epoch_windows=epoch_windows, crop_meta=crop_meta)
    recording_id = _attr_text(root.attrs, "recording_id", "recording_name") or zarr_path.stem
    resolved_run_name = run_name or utc_run_name()
    return ComparisonResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        offline_source_path=offline.source_path,
        offline_source_kind=offline.source_kind,
        offline_run_name=offline.run_name,
        realtime_source_path=realtime.source_path,
        realtime_source_kind=realtime.source_kind,
        stimulus_run_name=str(realtime.run_name or ""),
        run_name=resolved_run_name,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        realtime_frame_offset=int(realtime_frame_offset),
        arrays=arrays,
        summary=summary,
        epoch_windows=epoch_windows,
    )


def _downsample_indices(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, int(limit), dtype=np.int64)


def render_comparison_png(
    result: ComparisonResult,
    *,
    dpi: int = 150,
    scatter_limit: int = DEFAULT_SCATTER_LIMIT,
) -> bytes:
    arrays = result.arrays
    both = arrays.offline_present & arrays.realtime_present
    offline_only = arrays.offline_present & ~arrays.realtime_present
    realtime_only = arrays.realtime_present & ~arrays.offline_present

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    off_idx = _downsample_indices(int(np.count_nonzero(arrays.offline_present)), scatter_limit)
    rt_idx = _downsample_indices(int(np.count_nonzero(arrays.realtime_present)), scatter_limit)
    off_centers = arrays.offline_center_xy[arrays.offline_present]
    rt_centers = arrays.realtime_center_xy[arrays.realtime_present]
    if off_centers.size:
        ax.scatter(off_centers[off_idx, 0], off_centers[off_idx, 1], s=2, alpha=0.18, label="offline/refined")
    if rt_centers.size:
        ax.scatter(rt_centers[rt_idx, 0], rt_centers[rt_idx, 1], s=2, alpha=0.18, label="realtime")
    ax.set_xlim(0, result.width)
    ax.set_ylim(result.height, 0)
    ax.set_title("Centroid overlay")
    ax.set_xlabel("x px")
    ax.set_ylabel("y px")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    valid_delta = np.isfinite(arrays.centroid_delta_px)
    if np.any(valid_delta):
        ax.plot(arrays.frame_indices[valid_delta], arrays.centroid_delta_px[valid_delta], linewidth=0.4, alpha=0.7)
        p99 = result.summary["centroid_delta_px"].get("p99")
        if p99 is not None:
            ax.axhline(float(p99), color="tab:red", linestyle="--", linewidth=1, label=f"p99={float(p99):.1f}px")
            ax.legend(fontsize=8)
    for window in result.epoch_windows:
        if window.start_frame is not None:
            ax.axvline(window.start_frame, color="black", linewidth=0.6, alpha=0.35)
    ax.set_title("Centroid delta over time")
    ax.set_xlabel("frame")
    ax.set_ylabel("offline - realtime px")
    ax.set_ylim(bottom=0)

    ax = axes[1, 0]
    codes = np.zeros(arrays.frame_indices.shape[0], dtype=np.int8)
    codes[both] = 1
    codes[offline_only] = 2
    codes[realtime_only] = 3
    ax.imshow(codes[np.newaxis, :], aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=3)
    ax.set_title("Presence barcode: both / offline-only / realtime-only")
    ax.set_yticks([])
    ax.set_xlabel("comparison row sorted by frame")

    ax = axes[1, 1]
    delta = arrays.centroid_delta_px[np.isfinite(arrays.centroid_delta_px)]
    if delta.size:
        ax.hist(delta, bins=80, alpha=0.75, label="centroid delta px")
    iou = arrays.bbox_iou[np.isfinite(arrays.bbox_iou)]
    ax2 = ax.twinx()
    if iou.size:
        ax2.hist(iou, bins=np.linspace(0, 1, 51), alpha=0.35, color="tab:orange", label="bbox IoU")
    ax.set_title("Agreement distributions")
    ax.set_xlabel("delta px / IoU")
    ax.set_ylabel("delta count")
    ax2.set_ylabel("IoU count")

    summary = result.summary
    crop_suffix = ""
    if summary.get("crop_sufficiency_available"):
        crop_pct = summary.get("offline_full_bbox_inside_crop_pct")
        crop_suffix = (
            f"\ncrop full-bbox inside={crop_pct:.2f}% "
            f"blank_offline={summary.get('blank_crop_rows_for_offline_count'):,} "
            f"elsewhere={summary.get('crop_elsewhere_rows_for_offline_count'):,}"
            if crop_pct is not None
            else ""
        )
    text = (
        f"{result.recording_id}\n"
        f"offline: {result.offline_source_path}\n"
        f"realtime: {result.realtime_source_path}\n"
        f"both={summary['both_present_count']:,} "
        f"offline_only={summary['offline_only_count']:,} "
        f"realtime_only={summary['realtime_only_count']:,}\n"
        f"delta p50={summary['centroid_delta_px'].get('p50'):.2f}px "
        f"p99={summary['centroid_delta_px'].get('p99'):.2f}px"
        f"{crop_suffix}"
    )
    fig.suptitle("Realtime vs offline detection comparison", fontsize=14)
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, family="monospace")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _chunks_for(data: np.ndarray) -> tuple[int, ...]:
    shape = tuple(int(v) for v in data.shape)
    if not shape:
        return (1,)
    if len(shape) == 1:
        return (max(1, min(shape[0], 65_536)),)
    return (max(1, min(shape[0], 8192)), *shape[1:])


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_array(name, data=data, chunks=_chunks_for(np.asarray(data)), overwrite=True)


def write_comparison_run(
    zarr_path: Path,
    result: ComparisonResult,
    *,
    png_bytes: Optional[bytes],
    overwrite: bool,
) -> str:
    root = _open_root(zarr_path, mode="a")
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "detection_comparison_runs")
    run_name = result.run_name
    if run_name in parent:
        if not overwrite:
            raise ValueError(f"Comparison run already exists: {DEFAULT_PARENT_PATH}/{run_name}")
        del parent[run_name]
    run = parent.create_group(run_name)
    mark_run_pending(parent, run_name)
    mark_run_started(run, run_name=run_name, stage="detection_comparison")
    try:
        arrays = result.arrays
        arrays_to_write = {
            "frame_indices": arrays.frame_indices.astype(np.int64, copy=False),
            "offline_present": arrays.offline_present.astype(bool, copy=False),
            "realtime_present": arrays.realtime_present.astype(bool, copy=False),
            "offline_row_index": arrays.offline_row_index.astype(np.int64, copy=False),
            "realtime_row_index": arrays.realtime_row_index.astype(np.int64, copy=False),
            "offline_center_xy": arrays.offline_center_xy.astype(np.float32, copy=False),
            "realtime_center_xy": arrays.realtime_center_xy.astype(np.float32, copy=False),
            "offline_bbox_img_xyxy": arrays.offline_bbox_img_xyxy.astype(np.float32, copy=False),
            "realtime_bbox_img_xyxy": arrays.realtime_bbox_img_xyxy.astype(np.float32, copy=False),
            "offline_confidence": arrays.offline_confidence.astype(np.float32, copy=False),
            "realtime_confidence": arrays.realtime_confidence.astype(np.float32, copy=False),
            "centroid_delta_px": arrays.centroid_delta_px.astype(np.float32, copy=False),
            "bbox_iou": arrays.bbox_iou.astype(np.float32, copy=False),
            "epoch_label_code": arrays.epoch_label_code.astype(np.int16, copy=False),
        }
        if arrays.realtime_crop_xywh is not None:
            arrays_to_write.update(
                {
                    "realtime_crop_xywh": arrays.realtime_crop_xywh.astype(np.float32, copy=False),
                    "realtime_crop_has_detection": arrays.realtime_crop_has_detection.astype(bool, copy=False),
                    "realtime_crop_blank_frame": arrays.realtime_crop_blank_frame.astype(bool, copy=False),
                    "offline_center_inside_realtime_crop": (
                        arrays.offline_center_inside_realtime_crop.astype(bool, copy=False)
                    ),
                    "offline_bbox_inside_realtime_crop": (
                        arrays.offline_bbox_inside_realtime_crop.astype(bool, copy=False)
                    ),
                    "offline_crop_edge_margins": arrays.offline_crop_edge_margins.astype(np.float32, copy=False),
                    "crop_sufficiency_reason_code": (
                        arrays.crop_sufficiency_reason_code.astype(np.int8, copy=False)
                    ),
                }
            )
        for name, data in arrays_to_write.items():
            _write_array(run, name, np.asarray(data))

        epoch_labels = {0: "unassigned"}
        for code, window in enumerate(result.epoch_windows, start=1):
            epoch_labels[code] = window.label
        attrs = {
            "schema_id": SCHEMA_ID,
            "run_name": run_name,
            "recording_id": result.recording_id,
            "offline_source_path": result.offline_source_path,
            "offline_source_kind": result.offline_source_kind,
            "offline_run_name": result.offline_run_name,
            "realtime_source_path": result.realtime_source_path,
            "realtime_source_kind": result.realtime_source_kind,
            "stimulus_run_name": result.stimulus_run_name,
            "realtime_frame_offset": int(result.realtime_frame_offset),
            "coordinate_space": "source_image_pixels_xyxy",
            "width": int(result.width),
            "height": int(result.height),
            "fps": float(result.fps),
            "total_frames": int(result.total_frames),
            "summary": json_attr_safe(result.summary),
            "epoch_label_codes": json_attr_safe(epoch_labels),
            "epoch_windows": json_attr_safe([asdict(window) for window in result.epoch_windows]),
            "crop_sufficiency_reason_codes": json_attr_safe(CROP_REASON_LABELS),
        }
        run.attrs.update(attrs)
        if png_bytes:
            write_png_visualization_artifact(
                run,
                PNG_ARTIFACT_NAME,
                png_bytes,
                description="Realtime acquisition boxes compared against offline/refined detections.",
                created_by="fisheye.diagnostics.compare_realtime_offline_detections",
                role="diagnostic_dashboard",
                source_paths={
                    "offline": result.offline_source_path,
                    "realtime": result.realtime_source_path,
                },
                source_runs={
                    "offline_run": result.offline_run_name,
                    "stimulus_run": result.stimulus_run_name,
                },
                parameters={
                    "realtime_frame_offset": int(result.realtime_frame_offset),
                    "offline_source_kind": result.offline_source_kind,
                },
                extra_attrs={
                    "comparison_schema_id": SCHEMA_ID,
                    "summary": json_attr_safe(result.summary),
                },
                overwrite=True,
            )
        mark_run_complete(
            run,
            parent_group=parent,
            run_name=run_name,
            run_provenance=build_writer_run_provenance(
                command="fisheye.diagnostics.compare_realtime_offline_detections",
                params={
                    "realtime_frame_offset": int(result.realtime_frame_offset),
                    "offline_source_kind": result.offline_source_kind,
                },
                input_run_ids={
                    "offline_source_path": result.offline_source_path,
                    "realtime_source_path": result.realtime_source_path,
                    "offline_run": result.offline_run_name,
                    "stimulus_run": result.stimulus_run_name,
                },
            ),
        )
    except Exception as exc:
        mark_run_failed(run, error=str(exc))
        raise
    return f"{DEFAULT_PARENT_PATH}/{run_name}"


def write_summary_json(result: ComparisonResult, path: Path) -> None:
    payload = {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "run_name": result.run_name,
        "offline_source_path": result.offline_source_path,
        "offline_source_kind": result.offline_source_kind,
        "offline_run_name": result.offline_run_name,
        "realtime_source_path": result.realtime_source_path,
        "realtime_source_kind": result.realtime_source_kind,
        "stimulus_run_name": result.stimulus_run_name,
        "summary": result.summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def print_report(result: ComparisonResult) -> None:
    summary = result.summary
    print(f"recording_id: {result.recording_id}")
    print(f"offline: {result.offline_source_path}")
    print(f"realtime: {result.realtime_source_path} ({result.realtime_source_kind})")
    print(f"run_name: {result.run_name}")
    print(f"frame_count: {summary['comparison_frame_count']}")
    print(
        "presence: "
        f"offline={summary['offline_present_count']} "
        f"realtime={summary['realtime_present_count']} "
        f"both={summary['both_present_count']} "
        f"offline_only={summary['offline_only_count']} "
        f"realtime_only={summary['realtime_only_count']}"
    )
    delta = summary["centroid_delta_px"]
    iou = summary["bbox_iou"]
    print(
        "centroid_delta_px: "
        f"p50={delta.get('p50')} p90={delta.get('p90')} p99={delta.get('p99')} max={delta.get('p100')}"
    )
    print(f"bbox_iou: p50={iou.get('p50')} p10={iou.get('p10')} min={iou.get('p0')}")
    if summary.get("crop_sufficiency_available"):
        print(
            "crop_sufficiency: "
            f"bbox_inside={summary.get('offline_full_bbox_inside_crop_count')}/"
            f"{summary.get('offline_present_count')} "
            f"({summary.get('offline_full_bbox_inside_crop_pct')}%) "
            f"blank={summary.get('blank_crop_rows_for_offline_count')} "
            f"no_detection={summary.get('no_detection_crop_rows_for_offline_count')} "
            f"missing={summary.get('missing_crop_rows_for_offline_count')} "
            f"elsewhere={summary.get('crop_elsewhere_rows_for_offline_count')}"
        )
    if summary.get("epochs"):
        print("epochs:")
        for label, payload in summary["epochs"].items():
            d = payload["centroid_delta_px"]
            print(
                f"  {label}: frames={payload['start_frame']}-{payload['end_frame']} "
                f"both={payload['both_present_count']} "
                f"offline_only={payload['offline_only_count']} "
                f"realtime_only={payload['realtime_only_count']} "
                f"delta_p99={d.get('p99')}"
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--apply", action="store_true", help="Write comparison run into the zarr store.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing comparison run with the same name.")
    parser.add_argument("--run-name", help="Comparison run name (default: timestamped detection_comparison_<utc>).")
    parser.add_argument("--offline-source", choices=("active", "refined", "raw"), default="active")
    parser.add_argument("--detection-path", help="Explicit offline detection group path.")
    parser.add_argument("--detect-run", help="Raw detect run when --offline-source raw is used.")
    parser.add_argument("--refined-run", help="Refined detect run when --offline-source refined is used.")
    parser.add_argument(
        "--realtime-source",
        choices=("auto", "crop-meta", "stimulus-h5"),
        default="auto",
        help="Realtime source. auto prefers external crop-recorder metadata when available, then H5.",
    )
    parser.add_argument("--crop-meta", type=Path, help="Explicit external crop-recorder *_crop_meta.csv path.")
    parser.add_argument("--recording-dir", type=Path, help="Recording root used to resolve recording_manifest.json.")
    parser.add_argument("--stimulus-run", help="Realtime stimulus run (default: latest).")
    parser.add_argument("--realtime-frame-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional external PNG output.")
    parser.add_argument("--summary-json", type=Path, help="Optional external JSON summary output.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--scatter-limit", type=int, default=DEFAULT_SCATTER_LIMIT)
    parser.add_argument("--no-zarr-png", action="store_true", help="When --apply is set, skip persisted zarr PNG artifact.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary instead of text.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = build_comparison_result(
        args.zarr_path,
        offline_source=str(args.offline_source),
        detection_path=args.detection_path,
        detect_run=args.detect_run,
        refined_run=args.refined_run,
        realtime_source=str(args.realtime_source),
        stimulus_run=args.stimulus_run,
        realtime_frame_offset=int(args.realtime_frame_offset),
        recording_dir=args.recording_dir,
        crop_meta_path=args.crop_meta,
        run_name=args.run_name,
    )
    png_bytes = render_comparison_png(result, dpi=int(args.dpi), scatter_limit=int(args.scatter_limit))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(png_bytes)
    if args.summary_json:
        write_summary_json(result, args.summary_json)
    zarr_run_path = None
    if args.apply:
        zarr_run_path = write_comparison_run(
            args.zarr_path,
            result,
            png_bytes=None if args.no_zarr_png else png_bytes,
            overwrite=bool(args.overwrite),
        )
    if args.json:
        payload = {
            "schema_id": SCHEMA_ID,
            "zarr_path": result.zarr_path,
            "recording_id": result.recording_id,
            "run_name": result.run_name,
            "zarr_run_path": zarr_run_path,
            "realtime_source_kind": result.realtime_source_kind,
            "summary": result.summary,
            "output": str(args.output) if args.output else None,
            "summary_json": str(args.summary_json) if args.summary_json else None,
        }
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print_report(result)
        if zarr_run_path:
            print(f"zarr_run: {zarr_run_path}")
        if args.output:
            print(f"output: {args.output}")
        if args.summary_json:
            print(f"summary_json: {args.summary_json}")
        if not args.apply:
            print("dry_run: pass --apply to write analysis/detection_comparison_runs/<run>")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
