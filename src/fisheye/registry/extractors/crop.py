"""Crop-quality row extractors for registry scans."""

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


def _normalize_path_text(value: Any) -> Optional[str]:
    text = _decode_attr(value)
    if not text:
        return None
    return str(text).strip().strip("/") or None


def _crop_run_names(crop_parent: zarr.Group) -> List[str]:
    try:
        names = list(crop_parent.group_keys())
    except Exception:
        names = [name for name in crop_parent.keys() if isinstance(name, str)]
    return sorted(str(name) for name in names)


def _extract_crop_quality_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return []

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = utc_now()

    rows: List[Dict[str, Any]] = []
    for crop_run in _crop_run_names(crop_parent):
        if crop_run not in crop_parent:
            continue
        crop_group = crop_parent[crop_run]
        summary = _coerce_mapping(crop_group.attrs.get("summary_statistics")) or {}
        review_status = _coerce_mapping(crop_group.attrs.get("crop_review_status")) or {}

        crop_created_utc = (
            _decode_attr(crop_group.attrs.get("created_at_utc"))
            or _decode_attr(crop_group.attrs.get("started_at_utc"))
            or _decode_attr(crop_group.attrs.get("created_utc"))
            or _decode_attr(crop_group.attrs.get("timestamp_utc"))
        )
        source_detect_run = _decode_attr(crop_group.attrs.get("source_detect_run"))
        source_refined_run = _decode_attr(crop_group.attrs.get("source_refined_run"))
        detection_source_type = _decode_attr(crop_group.attrs.get("detection_source_type"))
        detection_source_path = _normalize_path_text(crop_group.attrs.get("detection_source_path"))

        total_rois = _as_int(summary.get("total_rois_cropped"))
        if total_rois is None and "roi_images" in crop_group:
            total_rois = int(crop_group["roi_images"].shape[0])
        if total_rois is None and "bbox_norm_coords" in crop_group:
            total_rois = int(crop_group["bbox_norm_coords"].shape[0])

        total_frames = _as_int(summary.get("total_frames"))
        if total_frames is None and "frame_counts" in crop_group:
            total_frames = int(crop_group["frame_counts"].shape[0])

        frames_with_crops = _as_int(summary.get("frames_with_crops"))
        if frames_with_crops is None and "frame_counts" in crop_group:
            try:
                frame_counts = np.asarray(crop_group["frame_counts"][:], dtype=np.int64).reshape(-1)
                frames_with_crops = int(np.sum(frame_counts > 0))
                if total_frames is None:
                    total_frames = int(frame_counts.shape[0])
            except Exception:
                frames_with_crops = None

        percent_frames_with_crops = _as_float(summary.get("percent_frames_with_crops"))
        if (
            percent_frames_with_crops is None
            and frames_with_crops is not None
            and total_frames is not None
            and int(total_frames) > 0
        ):
            percent_frames_with_crops = float(frames_with_crops) / float(total_frames) * 100.0

        n_real = _as_int(crop_group.attrs.get("n_real_detections"))
        n_interpolated = _as_int(crop_group.attrs.get("n_interpolated_detections"))
        if (n_real is None or n_interpolated is None) and "detection_source" in crop_group:
            try:
                source_codes = np.asarray(crop_group["detection_source"][:], dtype=np.int64).reshape(-1)
                if n_real is None:
                    n_real = int(np.sum(source_codes == 0))
                if n_interpolated is None:
                    n_interpolated = int(np.sum(source_codes != 0))
                if total_rois is None:
                    total_rois = int(source_codes.shape[0])
            except Exception:
                pass
        if n_interpolated is None:
            n_interpolated = 0
        if n_real is None and total_rois is not None:
            n_real = max(int(total_rois) - int(n_interpolated), 0)

        includes_interpolated_attr = crop_group.attrs.get("includes_interpolated")
        if includes_interpolated_attr is None:
            includes_interpolated = 1 if int(n_interpolated or 0) > 0 else 0
        else:
            includes_interpolated = 1 if bool(includes_interpolated_attr) else 0

        review_timestamp_utc = (
            _decode_attr(review_status.get("timestamp_utc"))
            or _decode_attr(review_status.get("timestamp"))
            or _decode_attr(review_status.get("reviewed_at_utc"))
            or _decode_attr(review_status.get("reviewed_at"))
        )

        rows.append(
            {
                "crop_run": str(crop_run),
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "crop_created_utc": crop_created_utc,
                "source_detect_run": source_detect_run,
                "source_refined_run": source_refined_run,
                "detection_source_type": detection_source_type,
                "detection_source_path": detection_source_path,
                "total_rois": total_rois,
                "frames_with_crops": frames_with_crops,
                "total_frames": total_frames,
                "percent_frames_with_crops": percent_frames_with_crops,
                "includes_interpolated": includes_interpolated,
                "n_real_detections": n_real,
                "n_interpolated_detections": n_interpolated,
                "review_state": _decode_attr(review_status.get("state")),
                "review_method": _decode_attr(review_status.get("method")),
                "review_intended_use": _decode_attr(review_status.get("intended_use")),
                "review_reviewer": _decode_attr(review_status.get("reviewer")),
                "review_timestamp_utc": review_timestamp_utc,
                "review_notes": _decode_attr(review_status.get("notes")),
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc,
            }
        )

    return rows


__all__ = ["_extract_crop_quality_rows"]
