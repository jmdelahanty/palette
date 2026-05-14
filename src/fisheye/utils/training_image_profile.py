#!/usr/bin/env python3
"""Compute image-domain profiles for sampled training Zarr archives."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import json_attr_safe, strict_json_dumps
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.utils.detection_profile import (
    DetectionProfileError,
    DetectionSourceError,
    _extract_composition,
    _load_detection_arrays,
    _normalize_text,
    infer_zarr_use,
    resolve_detection_source,
)


SCHEMA_NAME = "training_image_profile"
SCHEMA_VERSION = "v1"
SOURCE_CONTENT_FINGERPRINT_SCHEMA_ID = "palette.training_image_profile.source_content_fingerprint"
SOURCE_CONTENT_FINGERPRINT_SCHEMA_VERSION = 1
SOURCE_CONTENT_FINGERPRINT_CANONICALIZATION = "sha256_profiled_training_frames_v1"

IMAGE_PERCENTILES = (1, 5, 50, 95, 99)
AGGREGATE_METRIC_FIELDS = (
    "intensity_mean",
    "intensity_std",
    "contrast_p99_p01",
    "contrast_p95_p05",
    "sharpness_laplacian_var",
    "sharpness_gradient_mean",
    "clip_dark_fraction",
    "clip_bright_fraction",
    "illumination_center_edge_delta",
    "illumination_slope_x",
    "illumination_slope_y",
)
COMPOSITION_FIELDS = (
    "rig_id",
    "camera_id",
    "arena_id",
    "dish_design",
    "canvas_name",
    "protocol_name",
    "genotype",
    "dpf_at_acquisition",
)

DEFAULT_PROFILE_CONFIG: dict[str, Any] = {
    "clip_dark_threshold": 1.0,
    "clip_bright_threshold": 254.0,
    "histogram_bins": 256,
    "label_context_scale": 1.75,
    "illumination_edge_fraction": 0.15,
    "illumination_center_fraction": 0.50,
}


class TrainingImageProfileError(RuntimeError):
    """Raised when a training image profile cannot be computed or written."""


@dataclass(frozen=True)
class ResolvedFrameArray:
    path: str
    array: Any


@dataclass(frozen=True)
class TrainingImageProfileWriteResult:
    run_name: str
    source_frame_array: str
    source_frame_content_hash: str
    profile_summary: dict[str, Any]


_utc_now = utc_now


def _default_profile_run_name(created_at_utc: str) -> str:
    try:
        stamp = datetime.fromisoformat(str(created_at_utc))
    except Exception:
        stamp = datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    return f"training_image_profile_{stamp.strftime('%Y-%m-%d_%H-%M-%S')}"


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _coerce_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        raw = value.decode("utf-8", "ignore")
    elif isinstance(value, str):
        raw = value
    else:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _merge_profile_config(profile_config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_PROFILE_CONFIG)
    if not profile_config:
        return merged
    for key in DEFAULT_PROFILE_CONFIG:
        if key in profile_config and profile_config[key] is not None:
            merged[key] = float(profile_config[key]) if key != "histogram_bins" else int(profile_config[key])
    merged["histogram_bins"] = max(1, int(merged["histogram_bins"]))
    merged["label_context_scale"] = max(1.0, float(merged["label_context_scale"]))
    merged["illumination_edge_fraction"] = min(0.45, max(0.01, float(merged["illumination_edge_fraction"])))
    merged["illumination_center_fraction"] = min(0.95, max(0.05, float(merged["illumination_center_fraction"])))
    return merged


def _metric_stats(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    payload: dict[str, Any] = {"count": int(arr.size)}
    if arr.size == 0:
        payload.update({"min": None, "max": None, "mean": None, "std": None})
        for pct in IMAGE_PERCENTILES:
            payload[f"p{pct:02d}"] = None
        return payload
    payload.update(
        {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    )
    for pct in IMAGE_PERCENTILES:
        payload[f"p{pct:02d}"] = float(np.percentile(arr, pct))
    return payload


def _resolve_frame_array(root: zarr.Group, frame_source: str = "auto") -> ResolvedFrameArray:
    raw_video = root.get("raw_video")
    if raw_video is None:
        raise TrainingImageProfileError("raw_video group missing")
    requested = str(frame_source or "auto").strip()
    if requested != "auto":
        key = requested
        if key.startswith("raw_video/"):
            key = key.split("/", 1)[1]
        if key not in raw_video:
            raise TrainingImageProfileError(f"raw_video/{key} frame array missing")
        return ResolvedFrameArray(path=f"raw_video/{key}", array=raw_video[key])
    for key in ("images_ds_rgb", "images_ds", "images_full"):
        if key in raw_video:
            return ResolvedFrameArray(path=f"raw_video/{key}", array=raw_video[key])
    raise TrainingImageProfileError("raw_video has no supported image arrays")


def _sample_indices(n_frames: int, max_frames: Optional[int]) -> np.ndarray:
    n_frames = max(0, int(n_frames))
    if max_frames is None or max_frames >= n_frames:
        return np.arange(n_frames, dtype=np.int64)
    count = max(1, int(max_frames))
    return np.unique(np.linspace(0, n_frames - 1, count, dtype=np.int64))


def _frame_to_gray(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(np.float64, copy=False)
        return (0.2126 * rgb[..., 0]) + (0.7152 * rgb[..., 1]) + (0.0722 * rgb[..., 2])
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float64, copy=False)
    raise TrainingImageProfileError(f"unsupported frame shape for image profile: {arr.shape}")


def _sharpness_metrics(gray: np.ndarray) -> tuple[float, float]:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0, 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - (4.0 * center)
    )
    dy = np.diff(gray, axis=0)
    dx = np.diff(gray, axis=1)
    grad_y = dy[:, :-1]
    grad_x = dx[:-1, :]
    grad = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    return float(np.var(lap)), float(np.mean(grad)) if grad.size else 0.0


def _illumination_metrics(gray: np.ndarray, *, center_fraction: float, edge_fraction: float) -> tuple[float, float, float]:
    h, w = gray.shape
    center_h = max(1, int(round(h * float(center_fraction))))
    center_w = max(1, int(round(w * float(center_fraction))))
    y0 = max(0, (h - center_h) // 2)
    x0 = max(0, (w - center_w) // 2)
    center = gray[y0 : y0 + center_h, x0 : x0 + center_w]

    edge_y = max(1, int(round(h * float(edge_fraction))))
    edge_x = max(1, int(round(w * float(edge_fraction))))
    edge_mask = np.zeros(gray.shape, dtype=bool)
    edge_mask[:edge_y, :] = True
    edge_mask[-edge_y:, :] = True
    edge_mask[:, :edge_x] = True
    edge_mask[:, -edge_x:] = True
    edge = gray[edge_mask]
    center_edge_delta = float(np.mean(center) - np.mean(edge)) if edge.size else 0.0

    y_coords = np.linspace(-0.5, 0.5, h, dtype=np.float64)
    x_coords = np.linspace(-0.5, 0.5, w, dtype=np.float64)
    col_mean = np.mean(gray, axis=0)
    row_mean = np.mean(gray, axis=1)
    slope_x = float(np.polyfit(x_coords, col_mean, 1)[0]) if w > 1 else 0.0
    slope_y = float(np.polyfit(y_coords, row_mean, 1)[0]) if h > 1 else 0.0
    return center_edge_delta, slope_x, slope_y


def _frame_metrics(gray: np.ndarray, config: Mapping[str, Any]) -> dict[str, float]:
    p01, p05, p50, p95, p99 = np.percentile(gray, [1, 5, 50, 95, 99])
    lap_var, grad_mean = _sharpness_metrics(gray)
    center_edge, slope_x, slope_y = _illumination_metrics(
        gray,
        center_fraction=float(config["illumination_center_fraction"]),
        edge_fraction=float(config["illumination_edge_fraction"]),
    )
    return {
        "intensity_mean": float(np.mean(gray)),
        "intensity_std": float(np.std(gray)),
        "intensity_min": float(np.min(gray)),
        "intensity_max": float(np.max(gray)),
        "intensity_p01": float(p01),
        "intensity_p05": float(p05),
        "intensity_p50": float(p50),
        "intensity_p95": float(p95),
        "intensity_p99": float(p99),
        "contrast_p99_p01": float(p99 - p01),
        "contrast_p95_p05": float(p95 - p05),
        "clip_dark_fraction": float(np.mean(gray <= float(config["clip_dark_threshold"]))),
        "clip_bright_fraction": float(np.mean(gray >= float(config["clip_bright_threshold"]))),
        "sharpness_laplacian_var": lap_var,
        "sharpness_gradient_mean": grad_mean,
        "illumination_center_edge_delta": center_edge,
        "illumination_slope_x": slope_x,
        "illumination_slope_y": slope_y,
    }


def _histogram_edges(bins: int) -> np.ndarray:
    return np.linspace(0.0, 256.0, int(bins) + 1, dtype=np.float64)


def _update_histogram(hist_counts: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> None:
    values = np.clip(gray.reshape(-1), edges[0], np.nextafter(edges[-1], edges[0]))
    counts, _ = np.histogram(values, bins=edges)
    hist_counts += counts.astype(np.int64, copy=False)


def _source_frame_hash(
    frame_array: Any,
    *,
    source_path: str,
    selected_indices: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(source_path.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(str(getattr(frame_array, "dtype", "")).encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(json.dumps([int(v) for v in getattr(frame_array, "shape", ())]).encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(np.ascontiguousarray(selected_indices, dtype=np.int64).tobytes())
    hasher.update(b"\x00")
    for idx in selected_indices:
        frame = np.asarray(frame_array[int(idx)])
        hasher.update(str(frame.dtype).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(json.dumps([int(v) for v in frame.shape]).encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(np.ascontiguousarray(frame).tobytes())
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _map_detection_frames_to_rows(root: zarr.Group, frame_indices: np.ndarray, n_rows: int) -> np.ndarray:
    mapped = np.full(frame_indices.shape, -1, dtype=np.int64)
    direct = (frame_indices >= 0) & (frame_indices < int(n_rows))
    mapped[direct] = frame_indices[direct]
    if np.all(mapped >= 0):
        return mapped

    raw_video = root.get("raw_video")
    if raw_video is None or "original_frame_indices" not in raw_video:
        return mapped
    original = np.asarray(raw_video["original_frame_indices"][:], dtype=np.int64)
    lookup = {int(frame): idx for idx, frame in enumerate(original.tolist())}
    for idx, frame in enumerate(frame_indices):
        if mapped[idx] >= 0:
            continue
        mapped[idx] = lookup.get(int(frame), -1)
    return mapped


def _bbox_pixels(bbox_norm: np.ndarray, *, width: int, height: int, scale: float = 1.0) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = [float(v) for v in bbox_norm]
    bw *= float(scale)
    bh *= float(scale)
    x0 = int(np.floor((cx - (bw * 0.5)) * width))
    x1 = int(np.ceil((cx + (bw * 0.5)) * width))
    y0 = int(np.floor((cy - (bh * 0.5)) * height))
    y1 = int(np.ceil((cy + (bh * 0.5)) * height))
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return x0, y0, x1, y1


def _label_conditioned_metrics_for_frame(
    gray: np.ndarray,
    bboxes: np.ndarray,
    *,
    context_scale: float,
) -> list[dict[str, float]]:
    h, w = gray.shape
    rows: list[dict[str, float]] = []
    for bbox in bboxes:
        x0, y0, x1, y1 = _bbox_pixels(bbox, width=w, height=h, scale=1.0)
        if x1 <= x0 or y1 <= y0:
            continue
        cx0, cy0, cx1, cy1 = _bbox_pixels(bbox, width=w, height=h, scale=context_scale)
        fish = gray[y0:y1, x0:x1]
        context = gray[cy0:cy1, cx0:cx1]
        if context.size == 0 or fish.size == 0:
            continue
        bg_mask = np.ones(context.shape, dtype=bool)
        inner_x0 = max(0, x0 - cx0)
        inner_x1 = max(0, x1 - cx0)
        inner_y0 = max(0, y0 - cy0)
        inner_y1 = max(0, y1 - cy0)
        bg_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
        background = context[bg_mask]
        if background.size == 0:
            continue
        fish_mean = float(np.mean(fish))
        background_mean = float(np.mean(background))
        rows.append(
            {
                "fish_mean": fish_mean,
                "background_mean": background_mean,
                "fish_background_delta": fish_mean - background_mean,
                "fish_background_abs_delta": abs(fish_mean - background_mean),
                "fish_std": float(np.std(fish)),
                "background_std": float(np.std(background)),
            }
        )
    return rows


def _resolve_detection_rows(root: zarr.Group, *, n_frame_rows: int) -> tuple[Optional[dict[int, np.ndarray]], Optional[dict[str, Any]]]:
    try:
        source = resolve_detection_source(root)
        bbox, frame_indices = _load_detection_arrays(root, source)
    except (DetectionSourceError, DetectionProfileError, KeyError, ValueError) as exc:
        return None, {"status": "missing", "reason": str(exc)}

    mapped_rows = _map_detection_frames_to_rows(root, frame_indices, n_frame_rows)
    valid = mapped_rows >= 0
    if not np.any(valid):
        return None, {
            "status": "unmapped",
            "detection_path": source.detection_path,
            "detection_type": source.detection_type,
            "rows_total": int(frame_indices.size),
            "rows_mapped": 0,
        }

    by_row: dict[int, list[np.ndarray]] = {}
    for row_idx, box in zip(mapped_rows[valid], bbox[valid], strict=False):
        by_row.setdefault(int(row_idx), []).append(np.asarray(box, dtype=np.float64))
    packed = {row_idx: np.vstack(values) for row_idx, values in by_row.items()}
    return packed, {
        "status": "available",
        "detection_path": source.detection_path,
        "detection_type": source.detection_type,
        "detect_run": source.detect_run,
        "refined_run": source.refined_run,
        "manual_group": source.manual_group,
        "review_state": source.review_state,
        "review_intended_use": source.review_intended_use,
        "rows_total": int(frame_indices.size),
        "rows_mapped": int(np.sum(valid)),
    }


def build_training_image_profile_summary(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    frame_source: str = "auto",
    max_frames: Optional[int] = None,
    profile_config: Optional[Mapping[str, Any]] = None,
    allow_analysis: bool = False,
) -> dict[str, Any]:
    config = _merge_profile_config(profile_config)
    created_at_utc = created_at_utc or _utc_now()
    resolved_zarr_use = _normalize_text(zarr_use) or infer_zarr_use(root, zarr_path)
    if resolved_zarr_use == "analysis" and not allow_analysis:
        raise TrainingImageProfileError(
            "training image profiles default to training Zarrs; pass allow_analysis=True for diagnostics"
        )

    frame_resolution = _resolve_frame_array(root, frame_source)
    frame_array = frame_resolution.array
    n_frames = int(frame_array.shape[0])
    selected = _sample_indices(n_frames, max_frames)
    hist_edges = _histogram_edges(int(config["histogram_bins"]))
    hist_counts = np.zeros(hist_edges.size - 1, dtype=np.int64)

    per_frame: dict[str, list[float]] = {name: [] for name in set(AGGREGATE_METRIC_FIELDS) | {
        "intensity_min",
        "intensity_max",
        "intensity_p01",
        "intensity_p05",
        "intensity_p50",
        "intensity_p95",
        "intensity_p99",
    }}
    label_rows, label_source = _resolve_detection_rows(root, n_frame_rows=n_frames)
    label_metrics: dict[str, list[float]] = {
        "fish_mean": [],
        "background_mean": [],
        "fish_background_delta": [],
        "fish_background_abs_delta": [],
        "fish_std": [],
        "background_std": [],
    }
    label_frame_count = 0
    label_detection_count = 0

    for frame_idx in selected:
        gray = _frame_to_gray(frame_array[int(frame_idx)])
        metrics = _frame_metrics(gray, config)
        for key, value in metrics.items():
            per_frame.setdefault(key, []).append(float(value))
        _update_histogram(hist_counts, gray, hist_edges)

        if label_rows is not None and int(frame_idx) in label_rows:
            rows = _label_conditioned_metrics_for_frame(
                gray,
                label_rows[int(frame_idx)],
                context_scale=float(config["label_context_scale"]),
            )
            if rows:
                label_frame_count += 1
                label_detection_count += len(rows)
            for row in rows:
                for key, value in row.items():
                    label_metrics[key].append(float(value))

    source_hash = _source_frame_hash(frame_array, source_path=frame_resolution.path, selected_indices=selected)
    dataset_id = _normalize_text(dataset_id) or _normalize_text(root.attrs.get("dataset_id"))
    recording_id = (
        _normalize_text(recording_id)
        or _normalize_text(root.attrs.get("recording_id"))
        or _normalize_text(root.attrs.get("session_uuid"))
    )

    summary: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "dataset": {
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "zarr_use": resolved_zarr_use,
            "zarr_path": str(zarr_path) if zarr_path is not None else None,
        },
        "source": {
            "frame_array": frame_resolution.path,
            "frame_shape": [int(v) for v in frame_array.shape],
            "frame_dtype": str(frame_array.dtype),
            "frames_total": int(n_frames),
            "frames_profiled": int(selected.size),
            "sample_policy": "all" if selected.size == n_frames else "evenly_spaced",
            "selected_frame_indices": [int(v) for v in selected.tolist()],
            "content_fingerprint_schema_id": SOURCE_CONTENT_FINGERPRINT_SCHEMA_ID,
            "content_fingerprint_schema_version": SOURCE_CONTENT_FINGERPRINT_SCHEMA_VERSION,
            "content_fingerprint_canonicalization": SOURCE_CONTENT_FINGERPRINT_CANONICALIZATION,
            "content_hash": source_hash,
        },
        "image_metrics": {
            key: _metric_stats(per_frame.get(key, []))
            for key in sorted(per_frame)
        },
        "intensity_histogram": {
            "bin_edges": [float(v) for v in hist_edges.tolist()],
            "counts": [int(v) for v in hist_counts.tolist()],
            "source_pixel_count": int(np.sum(hist_counts)),
        },
        "label_conditioned": {
            "source": label_source or {"status": "missing"},
            "profiled_frames_with_detections": int(label_frame_count),
            "profiled_detection_count": int(label_detection_count),
            "metrics": {
                key: _metric_stats(values)
                for key, values in sorted(label_metrics.items())
            },
        },
        "profile_config": config,
    }
    composition = _extract_composition(root)
    if composition:
        summary["composition"] = composition
    return json_attr_safe(summary)


def _get_or_create_group(parent: zarr.Group, name: str) -> zarr.Group:
    existing = parent.get(name)
    if existing is None:
        try:
            existing = parent[name]
        except Exception:
            existing = None
    if existing is not None:
        return existing
    try:
        return parent.create_group(name)
    except Exception:
        existing = parent.get(name)
        if existing is None:
            try:
                existing = parent[name]
            except Exception:
                existing = None
        if existing is None:
            raise
        return existing


def _create_or_replace_array(group: zarr.Group, name: str, data: Any) -> None:
    if name in group:
        del group[name]
    arr = np.asarray(data)
    try:
        group.create_array(name, data=arr, chunks=arr.shape)
    except TypeError:
        group.create_dataset(name, data=arr, chunks=arr.shape)


def write_training_image_profile(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    frame_source: str = "auto",
    max_frames: Optional[int] = None,
    profile_config: Optional[Mapping[str, Any]] = None,
    allow_analysis: bool = False,
) -> TrainingImageProfileWriteResult:
    config = _merge_profile_config(profile_config)
    summary = build_training_image_profile_summary(
        root,
        zarr_path=zarr_path,
        dataset_id=dataset_id,
        recording_id=recording_id,
        zarr_use=zarr_use,
        created_at_utc=created_at_utc,
        frame_source=frame_source,
        max_frames=max_frames,
        profile_config=config,
        allow_analysis=allow_analysis,
    )
    created = _normalize_text(summary.get("created_at_utc")) or _utc_now()
    run_name = _normalize_text(run_name) or _default_profile_run_name(created)

    analysis_group = _get_or_create_group(root, "analysis")
    runs_parent = _get_or_create_group(analysis_group, "training_image_profile_runs")
    if run_name in runs_parent:
        if not overwrite:
            raise FileExistsError(f"analysis/training_image_profile_runs/{run_name} already exists")
        del runs_parent[run_name]
    run_group = runs_parent.create_group(run_name)

    dataset = summary.get("dataset", {})
    source = summary.get("source", {})
    histogram = summary.get("intensity_histogram", {})
    run_group.attrs.update(
        json_attr_safe(
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": created,
                "source_dataset_id": dataset.get("dataset_id"),
                "source_recording_id": dataset.get("recording_id"),
                "source_zarr_use": dataset.get("zarr_use"),
                "source_frame_array": source.get("frame_array"),
                "source_frame_count": source.get("frames_total"),
                "profiled_frame_count": source.get("frames_profiled"),
                "sample_policy": source.get("sample_policy"),
                "source_frame_content_hash": source.get("content_hash"),
                "source_frame_content_fingerprint_schema_id": SOURCE_CONTENT_FINGERPRINT_SCHEMA_ID,
                "source_frame_content_fingerprint_schema_version": SOURCE_CONTENT_FINGERPRINT_SCHEMA_VERSION,
                "source_frame_content_fingerprint_canonicalization": SOURCE_CONTENT_FINGERPRINT_CANONICALIZATION,
                "profile_config": config,
                "profile_summary": summary,
            }
        )
    )
    _create_or_replace_array(run_group, "intensity_histogram_counts", np.asarray(histogram.get("counts", []), dtype=np.int64))
    _create_or_replace_array(run_group, "intensity_histogram_bin_edges", np.asarray(histogram.get("bin_edges", []), dtype=np.float64))

    lineage_payload = build_run_lineage_payload(
        run_family="training_image_profile_run",
        analysis_schema={"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION},
        method="training_image_profile",
        method_version=SCHEMA_VERSION,
        source_refs={
            "source_frame_array": source.get("frame_array"),
            "source_dataset_id": dataset.get("dataset_id"),
            "source_recording_id": dataset.get("recording_id"),
            "source_zarr_use": dataset.get("zarr_use"),
        },
        source_fingerprints={
            "source_frame_content_hash": source.get("content_hash"),
            "source_frame_content_fingerprint_schema_id": SOURCE_CONTENT_FINGERPRINT_SCHEMA_ID,
            "source_frame_content_fingerprint_schema_version": SOURCE_CONTENT_FINGERPRINT_SCHEMA_VERSION,
        },
        parameters=config | {"frame_source": frame_source, "max_frames": max_frames},
    )
    write_run_lineage_attrs(run_group, lineage_payload, fingerprint_status="complete", overwrite=True)
    runs_parent.attrs["latest"] = run_name

    return TrainingImageProfileWriteResult(
        run_name=run_name,
        source_frame_array=str(source.get("frame_array") or ""),
        source_frame_content_hash=str(source.get("content_hash") or ""),
        profile_summary=summary,
    )


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)
    except TypeError:
        try:
            return zarr.open_group(str(zarr_path), mode=mode, consolidated=False)
        except TypeError:
            return zarr.open_group(str(zarr_path), mode=mode)


def _select_latest_profile_run(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = sorted(str(name) for name in parent.group_keys())
    except Exception:
        names = sorted(str(name) for name in parent.keys())
    return names[-1] if names else None


def _profile_summary(root: zarr.Group, profile_run: Optional[str]) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[str]]:
    analysis = root.get("analysis")
    if analysis is None:
        return profile_run, None, "analysis group missing"
    runs_parent = analysis.get("training_image_profile_runs")
    if runs_parent is None:
        return profile_run, None, "analysis/training_image_profile_runs missing"
    selected = _normalize_text(profile_run) or _select_latest_profile_run(runs_parent)
    if selected is None:
        return None, None, "analysis/training_image_profile_runs has no runs"
    if selected not in runs_parent:
        return selected, None, f"profile run missing: {selected}"
    run_group = runs_parent[selected]
    summary = _coerce_mapping(run_group.attrs.get("profile_summary"))
    if summary is None:
        return selected, None, f"profile_summary missing or invalid on run: {selected}"
    return selected, summary, None


def _dataset_row_for_zarr(registry: Registry, zarr_path: Path, *, dataset_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    if dataset_id:
        row = registry.conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ? LIMIT 1;",
            (str(dataset_id),),
        ).fetchone()
        return dict(row) if row is not None else None
    candidates = [str(zarr_path)]
    try:
        resolved = str(zarr_path.resolve())
    except OSError:
        resolved = None
    if resolved and resolved not in candidates:
        candidates.append(resolved)
    for candidate in candidates:
        row = registry.conn.execute(
            "SELECT * FROM datasets WHERE zarr_path = ? LIMIT 1;",
            (candidate,),
        ).fetchone()
        if row is not None:
            return dict(row)
    return None


def _summary_metric_p50(summary: Mapping[str, Any], key: str) -> Optional[float]:
    metrics = summary.get("image_metrics")
    if not isinstance(metrics, Mapping):
        return None
    item = metrics.get(key)
    if not isinstance(item, Mapping):
        return None
    return _as_float(item.get("p50"))


def _summary_metric_mean(summary: Mapping[str, Any], key: str) -> Optional[float]:
    metrics = summary.get("image_metrics")
    if not isinstance(metrics, Mapping):
        return None
    item = metrics.get(key)
    if not isinstance(item, Mapping):
        return None
    return _as_float(item.get("mean"))


def _label_metric_p50(summary: Mapping[str, Any], key: str) -> Optional[float]:
    label = summary.get("label_conditioned")
    if not isinstance(label, Mapping):
        return None
    metrics = label.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    item = metrics.get(key)
    if not isinstance(item, Mapping):
        return None
    return _as_float(item.get("p50"))


def _build_registry_payload(
    *,
    dataset_id: str,
    fallback_recording_id: Optional[str],
    fallback_zarr_use: Optional[str],
    fallback_genotype: Optional[str],
    fallback_dpf_at_acquisition: Optional[int],
    profile_run: str,
    summary: Mapping[str, Any],
    zarr_path: Path,
) -> dict[str, Any]:
    dataset = summary.get("dataset")
    source = summary.get("source")
    composition = summary.get("composition")
    dataset_map = dict(dataset) if isinstance(dataset, Mapping) else {}
    source_map = dict(source) if isinstance(source, Mapping) else {}
    composition_map = dict(composition) if isinstance(composition, Mapping) else {}
    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except OSError:
        zarr_mtime_ns = None
    dpf_at_acquisition = _as_int(composition_map.get("dpf_at_acquisition"))
    if dpf_at_acquisition is None:
        dpf_at_acquisition = fallback_dpf_at_acquisition
    return {
        "dataset_id": dataset_id,
        "profile_run": profile_run,
        "recording_id": _normalize_text(dataset_map.get("recording_id")) or fallback_recording_id,
        "zarr_use": _normalize_text(dataset_map.get("zarr_use")) or fallback_zarr_use,
        "source_frame_array": _normalize_text(source_map.get("frame_array")),
        "profile_created_utc": _normalize_text(summary.get("created_at_utc")),
        "zarr_mtime_ns": zarr_mtime_ns,
        "frames_total": _as_int(source_map.get("frames_total")),
        "frames_profiled": _as_int(source_map.get("frames_profiled")),
        "mean_intensity_p50": _summary_metric_p50(summary, "intensity_mean"),
        "contrast_p50": _summary_metric_p50(summary, "contrast_p99_p01"),
        "sharpness_p50": _summary_metric_p50(summary, "sharpness_laplacian_var"),
        "clip_dark_rate_mean": _summary_metric_mean(summary, "clip_dark_fraction"),
        "clip_bright_rate_mean": _summary_metric_mean(summary, "clip_bright_fraction"),
        "illumination_center_edge_p50": _summary_metric_p50(summary, "illumination_center_edge_delta"),
        "illumination_slope_x_p50": _summary_metric_p50(summary, "illumination_slope_x"),
        "illumination_slope_y_p50": _summary_metric_p50(summary, "illumination_slope_y"),
        "fish_bg_contrast_p50": _label_metric_p50(summary, "fish_background_abs_delta"),
        "rig_id": _normalize_text(composition_map.get("rig_id")),
        "camera_id": _normalize_text(composition_map.get("camera_id")),
        "arena_id": _normalize_text(composition_map.get("arena_id")),
        "dish_design": _normalize_text(composition_map.get("dish_design")),
        "canvas_name": _normalize_text(composition_map.get("canvas_name")),
        "protocol_name": _normalize_text(composition_map.get("protocol_name")),
        "genotype": _normalize_text(composition_map.get("genotype")) or fallback_genotype,
        "dpf_at_acquisition": dpf_at_acquisition,
        "profile_json": strict_json_dumps(summary),
    }


def sync_latest_training_image_profile_for_zarr(
    registry: Registry,
    zarr_path: Path,
    *,
    root: Optional[zarr.Group] = None,
    dataset_id: Optional[str] = None,
    profile_run: Optional[str] = None,
    apply: bool = True,
) -> dict[str, Any]:
    row = _dataset_row_for_zarr(registry, zarr_path, dataset_id=dataset_id)
    if row is None:
        return {
            "status": "missing_dataset",
            "zarr_path": str(zarr_path),
            "profile_run": profile_run,
            "reason": "zarr_path is not registered",
        }
    resolved_dataset_id = _normalize_text(row.get("dataset_id"))
    if not resolved_dataset_id:
        return {
            "status": "error",
            "zarr_path": str(zarr_path),
            "profile_run": profile_run,
            "reason": "registry dataset row has no dataset_id",
        }
    opened_root = root if root is not None else _open_root(zarr_path, mode="r")
    selected_profile_run, summary, summary_error = _profile_summary(opened_root, profile_run)
    if summary is None:
        return {
            "status": "missing_profile",
            "dataset_id": resolved_dataset_id,
            "zarr_path": str(zarr_path),
            "profile_run": selected_profile_run,
            "reason": summary_error,
        }
    payload = _build_registry_payload(
        dataset_id=resolved_dataset_id,
        fallback_recording_id=_normalize_text(row.get("recording_id")),
        fallback_zarr_use=_normalize_text(row.get("zarr_use")),
        fallback_genotype=_normalize_text(row.get("genotype")),
        fallback_dpf_at_acquisition=_as_int(row.get("dpf_at_acquisition")),
        profile_run=str(selected_profile_run),
        summary=summary,
        zarr_path=zarr_path,
    )
    if apply:
        registry.upsert_training_image_profile(**payload)
        status = "updated"
    else:
        status = "would_upsert"
    return {
        "status": status,
        "dataset_id": resolved_dataset_id,
        "zarr_path": str(zarr_path),
        "profile_run": selected_profile_run,
        "payload": payload,
    }


def _parse_max_frames(value: str) -> Optional[int]:
    text = str(value).strip().lower()
    if text == "all":
        return None
    number = int(text)
    if number <= 0:
        raise argparse.ArgumentTypeError("--max-frames must be 'all' or a positive integer")
    return number


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Training Zarr archive to profile.")
    parser.add_argument("--apply", action="store_true", help="Write profile run to Zarr (default: dry-run).")
    parser.add_argument("--run-name", type=str, help="Profile run name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing profile run of the same name.")
    parser.add_argument(
        "--zarr-use",
        choices=("training", "analysis", "any"),
        default="training",
        help="Expected Zarr use guard (default: training).",
    )
    parser.add_argument("--allow-analysis", action="store_true", help="Permit profiling analysis Zarrs for diagnostics.")
    parser.add_argument(
        "--frame-source",
        choices=("auto", "images_ds_rgb", "images_ds", "images_full", "raw_video/images_ds_rgb", "raw_video/images_ds", "raw_video/images_full"),
        default="auto",
        help="Frame array source (default: auto).",
    )
    parser.add_argument("--max-frames", type=_parse_max_frames, default=None, help="'all' or an integer sample count.")
    parser.add_argument("--dataset-id", type=str, help="Optional dataset_id override.")
    parser.add_argument("--recording-id", type=str, help="Optional recording_id override.")
    parser.add_argument("--registry", type=Path, help="Registry SQLite path for optional sync.")
    parser.add_argument("--sync-registry", action="store_true", help="Sync the written/latest profile into registry.")
    args = parser.parse_args(argv)

    zarr_path = Path(args.zarr_path)
    mode = "a" if args.apply else "r"
    root = _open_root(zarr_path, mode=mode)
    inferred_use = infer_zarr_use(root, zarr_path)
    if args.zarr_use != "any" and inferred_use is not None and inferred_use != args.zarr_use:
        parser.error(f"expected zarr_use={args.zarr_use}, found {inferred_use}; pass --zarr-use any to override")
    allow_analysis = bool(args.allow_analysis or args.zarr_use == "analysis" or args.zarr_use == "any")

    if args.apply:
        result = write_training_image_profile(
            root,
            zarr_path=zarr_path,
            run_name=args.run_name,
            overwrite=bool(args.overwrite),
            dataset_id=args.dataset_id,
            recording_id=args.recording_id,
            zarr_use=inferred_use if inferred_use is not None else (None if args.zarr_use == "any" else args.zarr_use),
            frame_source=args.frame_source,
            max_frames=args.max_frames,
            allow_analysis=allow_analysis,
        )
        summary = result.profile_summary
        status = "updated"
        profile_run = result.run_name
    else:
        summary = build_training_image_profile_summary(
            root,
            zarr_path=zarr_path,
            dataset_id=args.dataset_id,
            recording_id=args.recording_id,
            zarr_use=inferred_use if inferred_use is not None else (None if args.zarr_use == "any" else args.zarr_use),
            frame_source=args.frame_source,
            max_frames=args.max_frames,
            allow_analysis=allow_analysis,
        )
        status = "would_write"
        profile_run = args.run_name or _default_profile_run_name(str(summary.get("created_at_utc") or _utc_now()))

    registry_sync: Optional[dict[str, Any]] = None
    if args.sync_registry:
        if not args.apply:
            registry_sync = {
                "status": "not_synced",
                "reason": "--sync-registry is only applied when --apply writes a profile run",
            }
        else:
            registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
            registry = Registry(registry_path)
            try:
                registry_sync = sync_latest_training_image_profile_for_zarr(
                    registry,
                    zarr_path,
                    root=root,
                    dataset_id=args.dataset_id,
                    profile_run=profile_run,
                    apply=True,
                )
            finally:
                registry.close()

    print(
        strict_json_dumps(
            {
                "status": status,
                "zarr_path": str(zarr_path),
                "profile_run": profile_run,
                "source_frame_array": summary.get("source", {}).get("frame_array"),
                "frames_profiled": summary.get("source", {}).get("frames_profiled"),
                "registry_sync": registry_sync,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "DEFAULT_PROFILE_CONFIG",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TrainingImageProfileError",
    "TrainingImageProfileWriteResult",
    "build_training_image_profile_summary",
    "sync_latest_training_image_profile_for_zarr",
    "write_training_image_profile",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
