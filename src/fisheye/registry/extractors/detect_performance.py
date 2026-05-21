"""Detection-performance row extractors for registry scans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _decode_attr


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _coerce_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return None


def _detect_run_names(detect_parent: zarr.Group) -> List[str]:
    try:
        names = list(detect_parent.group_keys())
    except Exception:
        names = [name for name in detect_parent.keys() if isinstance(name, str)]
    return sorted(str(name) for name in names)


def _extract_detect_coverage_summary(detect_group: zarr.Group) -> Dict[str, Any]:
    frame_counts = detect_group.get("frame_counts") or detect_group.get("n_detections")
    if frame_counts is None:
        return {}
    try:
        counts = np.asarray(frame_counts[:], dtype=np.int64).reshape(-1)
    except Exception:
        return {}
    total_frames = int(counts.shape[0])
    if total_frames <= 0:
        return {}
    frames_with_detections = int(np.sum(counts > 0))
    frames_zero_detections = int(total_frames - frames_with_detections)
    coverage_percent = float(frames_with_detections) / float(total_frames) * 100.0
    return {
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "frames_zero_detections": frames_zero_detections,
        "coverage_percent": coverage_percent,
    }


def _extract_detect_performance_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    detect_parent = root.get("detect_runs")
    if detect_parent is None:
        return []

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = utc_now()

    rows: List[Dict[str, Any]] = []
    for detect_run in _detect_run_names(detect_parent):
        if detect_run not in detect_parent:
            continue
        detect_group = detect_parent[detect_run]
        summary = _coerce_mapping(detect_group.attrs.get("summary_statistics")) or {}
        parameters = _coerce_mapping(detect_group.attrs.get("parameters")) or {}
        provenance = _coerce_mapping(detect_group.attrs.get("provenance")) or {}
        model_resolution = _coerce_mapping(provenance.get("model_resolution")) or {}
        model_resolution_selected = _coerce_mapping(model_resolution.get("selected")) or {}

        detect_created_utc = (
            _decode_attr(detect_group.attrs.get("created_at_utc"))
            or _decode_attr(detect_group.attrs.get("detect_timestamp_utc"))
            or _decode_attr(detect_group.attrs.get("created_utc"))
            or _decode_attr(detect_group.attrs.get("timestamp_utc"))
            or _decode_attr(provenance.get("created_at_utc"))
        )
        detect_method = (
            _decode_attr(detect_group.attrs.get("detection_method"))
            or _decode_attr(detect_group.attrs.get("method"))
            or _decode_attr(provenance.get("method"))
        )
        model_run_id = (
            _decode_attr(detect_group.attrs.get("model_resolution_selected_run_id"))
            or _decode_attr(model_resolution_selected.get("run_id"))
        )
        model_set_id = (
            _decode_attr(detect_group.attrs.get("model_resolution_selected_set_id"))
            or _decode_attr(model_resolution_selected.get("set_id"))
        )
        model_path = (
            _decode_attr(detect_group.attrs.get("model_path"))
            or _decode_attr(detect_group.attrs.get("model_resolution_selected_model_path"))
            or _decode_attr(model_resolution_selected.get("model_path"))
        )
        model_name = _decode_attr(detect_group.attrs.get("model_name"))
        if model_name is None and model_path:
            model_name = Path(model_path).name

        coverage_percent = _as_float(summary.get("percent_frames_with_detections"))
        frames_with_detections = _as_int(summary.get("frames_with_detections"))
        frames_zero_detections = _as_int(summary.get("frames_with_zero_detections"))
        total_frames = _as_int(summary.get("total_frames"))
        coverage_fallback = _extract_detect_coverage_summary(detect_group)
        if coverage_percent is None:
            coverage_percent = _as_float(coverage_fallback.get("coverage_percent"))
        if frames_with_detections is None:
            frames_with_detections = _as_int(coverage_fallback.get("frames_with_detections"))
        if frames_zero_detections is None:
            frames_zero_detections = _as_int(coverage_fallback.get("frames_zero_detections"))
        if total_frames is None:
            total_frames = _as_int(coverage_fallback.get("total_frames"))

        rows.append(
            {
                "detect_run": str(detect_run),
                "detect_created_utc": detect_created_utc,
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "detection_method": detect_method,
                "model_run_id": model_run_id,
                "model_set_id": model_set_id,
                "model_path": model_path,
                "model_name": model_name,
                "coverage_percent": coverage_percent,
                "frames_with_detections": frames_with_detections,
                "frames_zero_detections": frames_zero_detections,
                "total_frames": total_frames,
                "mean_confidence": _as_float(summary.get("mean_confidence")),
                "min_confidence": _as_float(summary.get("min_confidence")),
                "max_confidence": _as_float(summary.get("max_confidence")),
                "inference_duration_seconds": _as_float(detect_group.attrs.get("inference_duration_seconds")),
                "inference_average_fps": _as_float(detect_group.attrs.get("inference_average_fps")),
                "inference_avg_batch_ms": _as_float(detect_group.attrs.get("inference_avg_batch_ms")),
                "inference_avg_read_ms": _as_float(detect_group.attrs.get("inference_avg_read_ms")),
                "conf_threshold": _as_float(parameters.get("conf_threshold")),
                "iou_threshold": _as_float(parameters.get("iou_threshold")),
                "batch_size": _as_int(parameters.get("batch_size")),
                "inference_width": _as_int(detect_group.attrs.get("inference_width")),
                "inference_height": _as_int(detect_group.attrs.get("inference_height")),
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc,
            }
        )

    return rows


__all__ = ["_extract_detect_performance_rows"]
