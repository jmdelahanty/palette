"""Quality-row extractors for registry scans."""

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


def _format_ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _extract_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is not None:
        return refined_parent
    return root.get("keypoints_refined_runs")


def _extract_refined_detect_parent(root: zarr.Group) -> Optional[zarr.Group]:
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is not None:
        return refined_parent
    return root.get("refined_runs")


def _extract_keypoint_quality_rows(root: zarr.Group, *, zarr_path: Path) -> List[Dict[str, Any]]:
    keypoints_parent = root.get("keypoints_runs")
    refined_parent = _extract_refined_parent(root)
    if keypoints_parent is None or refined_parent is None:
        return []

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    quality_updated_utc = utc_now()

    rows: List[Dict[str, Any]] = []
    for refined_run in refined_parent.group_keys():
        refined_group = refined_parent[refined_run]
        source_keypoint_run = _decode_attr(refined_group.attrs.get("source_keypoints_run"))
        if not source_keypoint_run or source_keypoint_run not in keypoints_parent:
            continue

        source_group = keypoints_parent[source_keypoint_run]
        keypoint_method = _decode_attr(source_group.attrs.get("method"))
        review_status = _coerce_mapping(refined_group.attrs.get("keypoint_review_status"))
        review_state = _decode_attr(review_status.get("state")) if review_status else None
        review_method = _decode_attr(review_status.get("method")) if review_status else None
        review_intended_use = _decode_attr(review_status.get("intended_use")) if review_status else None
        review_reviewer = _decode_attr(review_status.get("reviewer")) if review_status else None
        review_notes = _decode_attr(review_status.get("notes")) if review_status else None
        auto_review = _coerce_mapping(review_status.get("auto_review")) if review_status else None
        review_policy_id = _decode_attr(auto_review.get("policy_id")) if auto_review else None
        review_policy_version = _as_int(auto_review.get("policy_version")) if auto_review else None
        review_timestamp_utc = (
            _decode_attr(review_status.get("timestamp_utc"))
            or _decode_attr(review_status.get("timestamp"))
            or _decode_attr(review_status.get("reviewed_at_utc"))
            or _decode_attr(review_status.get("reviewed_at"))
            if review_status
            else None
        )

        total_keypoints: Optional[int] = None
        usable_keypoints: Optional[int] = None
        usable_keypoints_rate: Optional[float] = None
        if "usable_keypoints" in refined_group:
            usable_arr = refined_group["usable_keypoints"]
            total_keypoints = int(usable_arr.shape[0])
            usable_keypoints = int(np.asarray(usable_arr[:]).sum())
            usable_keypoints_rate = _format_ratio(usable_keypoints, total_keypoints)

        summary_stats = refined_group.attrs.get("summary_statistics")
        if isinstance(summary_stats, Mapping):
            postprocess = summary_stats.get("postprocess")
            for candidate in (postprocess, summary_stats):
                if not isinstance(candidate, Mapping):
                    continue
                if usable_keypoints is None:
                    usable_keypoints = _as_int(candidate.get("usable_keypoints"))
                if total_keypoints is None:
                    total_keypoints = _as_int(candidate.get("total_rois"))
                if usable_keypoints_rate is None:
                    usable_keypoints_rate = _format_ratio(usable_keypoints, total_keypoints)

        keypoint_rows = int(source_group["keypoints_roi"].shape[0]) if "keypoints_roi" in source_group else None
        raw_keypoints_success_rate = _as_float(source_group.attrs.get("success_rate"))
        raw_keypoints_successful: Optional[int] = None
        if raw_keypoints_success_rate is not None and keypoint_rows is not None:
            raw_keypoints_successful = int(round(raw_keypoints_success_rate * float(keypoint_rows)))
        elif "detection_success" in source_group:
            success_arr = source_group["detection_success"]
            raw_keypoints_successful = int(np.asarray(success_arr[:]).sum())
            raw_keypoints_success_rate = _format_ratio(raw_keypoints_successful, int(success_arr.shape[0]))

        rows.append(
            {
                "refined_run": str(refined_run),
                "refined_created_utc": _decode_attr(
                    refined_group.attrs.get("created_at_utc")
                    or refined_group.attrs.get("refinement_timestamp")
                    or refined_group.attrs.get("created_utc")
                    or refined_group.attrs.get("timestamp_utc")
                ),
                "source_keypoint_run": source_keypoint_run,
                "keypoint_method": keypoint_method,
                "review_state": review_state,
                "review_method": review_method,
                "review_intended_use": review_intended_use,
                "review_reviewer": review_reviewer,
                "review_notes": review_notes,
                "review_policy_id": review_policy_id,
                "review_policy_version": review_policy_version,
                "review_timestamp_utc": review_timestamp_utc,
                "usable_keypoints": usable_keypoints,
                "total_keypoints": total_keypoints,
                "usable_keypoints_rate": usable_keypoints_rate,
                "raw_keypoints_success_rate": raw_keypoints_success_rate,
                "raw_keypoints_successful": raw_keypoints_successful,
                "quality_updated_utc": quality_updated_utc,
                "zarr_mtime_ns": zarr_mtime_ns,
            }
        )
    return rows


def _extract_detect_quality_rows(root: zarr.Group, *, zarr_path: Path) -> List[Dict[str, Any]]:
    refined_parent = _extract_refined_detect_parent(root)
    if refined_parent is None:
        return []

    detect_runs_parent = root.get("detect_runs")
    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    quality_updated_utc = utc_now()

    try:
        refined_run_names = list(refined_parent.group_keys())
    except Exception:
        refined_run_names = [name for name in refined_parent.keys() if isinstance(name, str)]

    rows: List[Dict[str, Any]] = []
    for refined_run in refined_run_names:
        if refined_run not in refined_parent:
            continue
        refined_group = refined_parent[refined_run]
        source_detect_run = _decode_attr(refined_group.attrs.get("source_detect_run"))
        if not source_detect_run:
            continue

        detect_method = None
        if detect_runs_parent is not None and source_detect_run in detect_runs_parent:
            source_group = detect_runs_parent[source_detect_run]
            detect_method = _decode_attr(
                source_group.attrs.get("method")
                or source_group.attrs.get("detection_method")
            )

        review_status = _coerce_mapping(refined_group.attrs.get("detect_review_status"))
        review_state = _decode_attr(review_status.get("state")) if review_status else None
        review_method = _decode_attr(review_status.get("method")) if review_status else None
        review_intended_use = _decode_attr(review_status.get("intended_use")) if review_status else None
        review_reviewer = _decode_attr(review_status.get("reviewer")) if review_status else None
        review_notes = _decode_attr(review_status.get("notes")) if review_status else None
        review_timestamp_utc = (
            _decode_attr(review_status.get("timestamp_utc"))
            or _decode_attr(review_status.get("timestamp"))
            or _decode_attr(review_status.get("reviewed_at_utc"))
            or _decode_attr(review_status.get("reviewed_at"))
            if review_status
            else None
        )

        resolved_group = _decode_attr(review_status.get("resolved_group")) if review_status else None
        resolved = None
        instances = refined_group.get("instances")
        if instances is not None:
            resolved_group = "refined"
            resolved = instances
        elif resolved_group == "refined":
            resolved = refined_group.get("instances")
            if resolved is None:
                active_sparse_group = _decode_attr(refined_group.attrs.get("active_sparse_group"))
                if active_sparse_group and active_sparse_group in refined_group:
                    resolved = refined_group.get(active_sparse_group)
        elif resolved_group:
            resolved = refined_group.get(resolved_group)

        if resolved is None and not resolved_group:
            manual_latest = _decode_attr(refined_group.attrs.get("manual_review_latest"))
            if manual_latest and manual_latest in refined_group:
                resolved_group = manual_latest
                resolved = refined_group.get(manual_latest)
            elif "interpolated" in refined_group:
                resolved_group = "interpolated"
                resolved = refined_group.get("interpolated")
            elif "filtered" in refined_group:
                resolved_group = "filtered"
                resolved = refined_group.get("filtered")
            else:
                resolved_group = "raw"
                resolved = refined_group.get("raw")

        total_detections: Optional[int] = None
        real_detections: Optional[int] = None
        interpolated_detections: Optional[int] = None
        interpolated_detections_rate: Optional[float] = None

        if resolved is not None and "bbox_norm_coords" in resolved:
            total_detections = int(resolved["bbox_norm_coords"].shape[0])

        if resolved is not None and "detection_source" in resolved:
            source_arr = np.asarray(resolved["detection_source"][:], dtype=np.int64)
            real_detections = int(np.sum(source_arr == 0))
            interpolated_detections = int(np.sum(source_arr != 0))
            total_detections = int(source_arr.shape[0])
        elif resolved is not None and "source_kind_codes" in resolved:
            source_kind_arr = np.asarray(resolved["source_kind_codes"][:], dtype=np.int64)
            interpolated_detections = int(np.sum(source_kind_arr == 2))
            total_detections = int(source_kind_arr.shape[0])
            real_detections = int(total_detections - interpolated_detections)
        elif resolved is not None:
            if total_detections is None:
                total_detections = _as_int(resolved.attrs.get("total_detections"))
            interpolated_detections = _as_int(resolved.attrs.get("interpolated_detections"))
            if interpolated_detections is None and _decode_attr(resolved_group) == "filtered":
                interpolated_detections = 0
            real_detections = _as_int(resolved.attrs.get("original_detections"))
            if real_detections is None and total_detections is not None and interpolated_detections is not None:
                real_detections = int(total_detections) - int(interpolated_detections)

        if total_detections is None and real_detections is not None and interpolated_detections is not None:
            total_detections = int(real_detections) + int(interpolated_detections)
        if total_detections is not None and interpolated_detections is not None and int(total_detections) > 0:
            interpolated_detections_rate = float(interpolated_detections) / float(total_detections)

        rows.append(
            {
                "refined_run": str(refined_run),
                "refined_created_utc": _decode_attr(
                    refined_group.attrs.get("created_at_utc")
                    or refined_group.attrs.get("refinement_timestamp")
                    or refined_group.attrs.get("created_utc")
                    or refined_group.attrs.get("timestamp_utc")
                ),
                "source_detect_run": source_detect_run,
                "detect_method": detect_method,
                "review_state": review_state,
                "review_method": review_method,
                "review_intended_use": review_intended_use,
                "review_reviewer": review_reviewer,
                "review_notes": review_notes,
                "review_timestamp_utc": review_timestamp_utc,
                "review_resolved_group": resolved_group,
                "total_detections": total_detections,
                "real_detections": real_detections,
                "interpolated_detections": interpolated_detections,
                "interpolated_detections_rate": interpolated_detections_rate,
                "quality_updated_utc": quality_updated_utc,
                "zarr_mtime_ns": zarr_mtime_ns,
            }
        )
    return rows


__all__ = [
    "_extract_detect_quality_rows",
    "_extract_keypoint_quality_rows",
]
