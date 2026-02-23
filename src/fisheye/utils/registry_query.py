#!/usr/bin/env python3
"""Query the registry for datasets matching acquisition/provenance filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fisheye.registry.db import Registry, RegistryPaths


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _query_detect_performance_map(
    registry: Registry,
    *,
    dataset_ids: list[str],
    model_only: bool,
) -> dict[str, dict[str, Any]]:
    if not dataset_ids:
        return {}
    placeholders = ",".join("?" for _ in dataset_ids)
    view_name = "detect_model_performance_latest" if model_only else "detect_performance_latest"
    rows = registry.conn.execute(
        f"""
        SELECT
            dataset_id,
            recording_id,
            detect_run,
            detect_created_utc,
            detection_method,
            model_run_id,
            model_set_id,
            model_path,
            model_name,
            coverage_percent,
            inference_average_fps,
            inference_avg_read_ms
        FROM {view_name}
        WHERE dataset_id IN ({placeholders});
        """,
        dataset_ids,
    ).fetchall()
    return {str(row["dataset_id"]): dict(row) for row in rows if row["dataset_id"] is not None}


def _query_crop_quality_map(
    registry: Registry,
    *,
    dataset_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not dataset_ids:
        return {}
    placeholders = ",".join("?" for _ in dataset_ids)
    rows = registry.conn.execute(
        f"""
        SELECT
            dataset_id,
            recording_id,
            crop_run,
            crop_created_utc,
            source_detect_run,
            source_refined_run,
            detection_source_type,
            detection_source_path,
            total_rois,
            frames_with_crops,
            total_frames,
            percent_frames_with_crops,
            includes_interpolated,
            n_real_detections,
            n_interpolated_detections,
            review_state,
            review_method,
            review_intended_use,
            review_reviewer,
            review_timestamp_utc,
            review_notes
        FROM crop_quality_current
        WHERE dataset_id IN ({placeholders});
        """,
        dataset_ids,
    ).fetchall()
    return {str(row["dataset_id"]): dict(row) for row in rows if row["dataset_id"] is not None}


def _query_keypoint_quality_map(
    registry: Registry,
    *,
    dataset_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not dataset_ids:
        return {}
    placeholders = ",".join("?" for _ in dataset_ids)
    rows = registry.conn.execute(
        f"""
        SELECT
            dataset_id,
            refined_run,
            refined_created_utc,
            source_keypoint_run,
            keypoint_method,
            review_state,
            review_intended_use,
            review_reviewer,
            review_timestamp_utc,
            usable_keypoints,
            total_keypoints,
            usable_keypoints_rate,
            raw_keypoints_success_rate,
            raw_keypoints_successful,
            quality_updated_utc
        FROM keypoint_quality_current
        WHERE dataset_id IN ({placeholders});
        """,
        dataset_ids,
    ).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset_id = row["dataset_id"]
        if dataset_id is None:
            continue
        out.setdefault(str(dataset_id), []).append(dict(row))
    return out


def _query_keypoint_performance_map(
    registry: Registry,
    *,
    dataset_ids: list[str],
    model_only: bool,
) -> dict[str, dict[str, Any]]:
    if not dataset_ids:
        return {}
    placeholders = ",".join("?" for _ in dataset_ids)
    where_model_only = (
        "AND (trim(COALESCE(model_path, '')) <> '' OR trim(COALESCE(model_name, '')) <> '')"
        if model_only
        else ""
    )
    rows = registry.conn.execute(
        f"""
        SELECT
            dataset_id,
            keypoint_run,
            keypoint_created_utc,
            recording_id,
            zarr_use,
            keypoint_method,
            model_run_id,
            model_set_id,
            model_path,
            model_name,
            source_crop_run,
            source_detect_run,
            source_refined_run,
            total_rois,
            successful_detections,
            failed_detections,
            success_rate_percent,
            frames_with_keypoints,
            mean_confidence,
            duration_seconds,
            inference_duration_seconds,
            keypoints_per_second,
            inference_average_fps,
            batch_size,
            imgsz,
            conf_threshold,
            iou_threshold,
            summary_statistics_json
        FROM keypoint_performance_latest
        WHERE dataset_id IN ({placeholders})
        {where_model_only};
        """,
        dataset_ids,
    ).fetchall()
    return {str(row["dataset_id"]): dict(row) for row in rows if row["dataset_id"] is not None}


def _query_eye_mask_performance_map(
    registry: Registry,
    *,
    dataset_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not dataset_ids:
        return {}
    placeholders = ",".join("?" for _ in dataset_ids)
    rows = registry.conn.execute(
        f"""
        SELECT
            dataset_id,
            stage_group,
            run_name,
            run_created_utc,
            recording_id,
            zarr_use,
            method,
            source_crop_run,
            source_keypoint_group,
            source_keypoints_run,
            source_eye_masks_run,
            source_eye_masks_method,
            total_rois,
            successful_eyes,
            successful_roi_pairs,
            successful_roi_pair_rate,
            duration_seconds,
            rois_per_second,
            inference_duration_seconds,
            inference_average_fps,
            review_state,
            review_method,
            review_intended_use,
            review_reviewer,
            review_timestamp_utc,
            source_keypoint_stale_state,
            source_keypoint_stale_reason,
            source_keypoint_stale_timestamp_utc,
            source_keypoint_stale_json,
            lifecycle_state,
            lifecycle_reason
        FROM eye_mask_performance_latest
        WHERE dataset_id IN ({placeholders});
        """,
        dataset_ids,
    ).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset_id = row["dataset_id"]
        if dataset_id is None:
            continue
        out.setdefault(str(dataset_id), []).append(dict(row))
    return out


def _query_recording_step_status_map(
    registry: Registry,
    *,
    dataset_ids: list[str],
    step_name: Optional[str],
    status: Optional[str],
) -> dict[str, list[dict[str, Any]]]:
    if not dataset_ids:
        return {}
    placeholders = ",".join("?" for _ in dataset_ids)
    sql = [
        f"""
        SELECT
            dataset_id,
            recording_id,
            step_name,
            status,
            run_name,
            method,
            coverage_pct,
            source,
            updated_utc
        FROM recording_step_status_latest
        WHERE dataset_id IN ({placeholders})
        """.strip()
    ]
    params: list[object] = list(dataset_ids)
    if step_name is not None:
        sql.append("AND lower(step_name) = ?")
        params.append(str(step_name).strip().lower())
    if status is not None:
        status_norm = str(status).strip().lower()
        if status_norm == "non-ok":
            sql.append("AND status != 'ok'")
        else:
            sql.append("AND status = ?")
            params.append(status_norm)
    sql.append("ORDER BY COALESCE(updated_utc, '') DESC, step_name, run_name")
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset_id = row["dataset_id"]
        if dataset_id is None:
            continue
        out.setdefault(str(dataset_id), []).append(dict(row))
    return out


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _matches_optional_text_filter(value: object, text_filter: Optional[str]) -> bool:
    if text_filter is None:
        return True
    filter_norm = str(text_filter).strip().lower()
    value_norm = _normalize_text(value)
    if filter_norm == "missing":
        return value_norm is None
    if value_norm is None:
        return False
    return value_norm.lower() == filter_norm


def _pick_eye_mask_candidate(
    candidates: list[dict[str, Any]],
    *,
    stage_filter: Optional[str],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Expected at least one eye-mask candidate.")
    if stage_filter is not None:
        ranked = candidates
    else:
        refined = [row for row in candidates if str(row.get("stage_group") or "") == "refined_eye_masks_runs"]
        ranked = refined if refined else candidates
    ranked = sorted(
        ranked,
        key=lambda row: (
            str(row.get("run_created_utc") or ""),
            str(row.get("run_name") or ""),
        ),
    )
    return ranked[-1]


def _pick_keypoint_quality_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Expected at least one keypoint-quality candidate.")
    ranked = sorted(
        candidates,
        key=lambda row: (
            str(row.get("review_timestamp_utc") or row.get("refined_created_utc") or ""),
            str(row.get("refined_run") or ""),
            str(row.get("keypoint_method") or ""),
        ),
    )
    return ranked[-1]


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (float(pct) / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _metric_summary(prefix: str, values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {
            f"{prefix}_avg": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_p10": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
        }
    return {
        f"{prefix}_avg": float(sum(values) / len(values)),
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
        f"{prefix}_p10": _percentile(values, 10.0),
        f"{prefix}_p50": _percentile(values, 50.0),
        f"{prefix}_p90": _percentile(values, 90.0),
    }


def _group_descriptor(row: dict[str, Any], *, group_by: str) -> tuple[tuple[str, ...], dict[str, Optional[str]]]:
    if group_by == "model":
        model_name = str(row.get("detect_model_name") or "").strip()
        model_path = str(row.get("detect_model_path") or "").strip()
        model_run_id = str(row.get("detect_model_run_id") or "").strip()
        model_set_id = str(row.get("detect_model_set_id") or "").strip()
        key = (model_name, model_path, model_run_id, model_set_id)
        fields = {
            "model_name": model_name or None,
            "model_path": model_path or None,
            "model_run_id": model_run_id or None,
            "model_set_id": model_set_id or None,
        }
        return key, fields
    field_map = {
        "rig": "rig_id",
        "camera": "camera_id",
        "arena": "arena_id",
        "dish": "dish_design",
    }
    field = field_map[group_by]
    value = str(row.get(field) or "").strip()
    key = (value,)
    fields = {
        "group_by": group_by,
        field: value or None,
        "group_value": value or None,
    }
    return key, fields


def _rows_to_group_summary(rows: list[dict[str, Any]], *, group_by: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key, descriptor = _group_descriptor(row, group_by=group_by)
        if all(v in (None, "") for v in descriptor.values() if v is not None):
            continue
        rec_id = str(row.get("detect_recording_id") or row.get("recording_id") or "").strip() or None
        bucket = grouped.setdefault(
            key,
            {
                **descriptor,
                "datasets": 0,
                "recording_ids": set(),
                "coverage_values": [],
                "fps_values": [],
                "read_ms_values": [],
            },
        )
        bucket["datasets"] += 1
        if rec_id:
            bucket["recording_ids"].add(rec_id)
        cov = _as_float(row.get("detect_coverage_percent"))
        if cov is not None:
            bucket["coverage_values"].append(cov)
        fps = _as_float(row.get("detect_inference_average_fps"))
        if fps is not None:
            bucket["fps_values"].append(fps)
        read_ms = _as_float(row.get("detect_inference_avg_read_ms"))
        if read_ms is not None:
            bucket["read_ms_values"].append(read_ms)

    summary: list[dict[str, Any]] = []
    for bucket in grouped.values():
        coverage_values = bucket["coverage_values"]
        fps_values = bucket["fps_values"]
        read_values = bucket["read_ms_values"]
        row: dict[str, Any] = {
            key: val
            for key, val in bucket.items()
            if key not in {"recording_ids", "coverage_values", "fps_values", "read_ms_values"}
        }
        row["datasets"] = int(bucket["datasets"])
        row["recordings"] = int(len(bucket["recording_ids"]))
        row.update(_metric_summary("coverage", coverage_values))
        row.update(_metric_summary("fps", fps_values))
        row.update(_metric_summary("read_ms", read_values))
        summary.append(row)
    summary.sort(
        key=lambda item: (
            -int(item["recordings"]),
            -int(item["datasets"]),
            str(item.get("group_value") or item.get("model_name") or item.get("model_path") or item.get("model_run_id") or ""),
        )
    )
    return summary


def _keypoint_group_descriptor(row: dict[str, Any], *, group_by: str) -> tuple[tuple[str, ...], dict[str, Optional[str]]]:
    if group_by == "model":
        model_name = str(row.get("keypoint_model_name") or "").strip()
        model_path = str(row.get("keypoint_model_path") or "").strip()
        model_run_id = str(row.get("keypoint_model_run_id") or "").strip()
        model_set_id = str(row.get("keypoint_model_set_id") or "").strip()
        key = (model_name, model_path, model_run_id, model_set_id)
        fields = {
            "model_name": model_name or None,
            "model_path": model_path or None,
            "model_run_id": model_run_id or None,
            "model_set_id": model_set_id or None,
        }
        return key, fields
    if group_by == "method":
        method = str(row.get("keypoint_method") or "").strip()
        key = (method,)
        fields = {
            "group_by": "method",
            "keypoint_method": method or None,
            "group_value": method or None,
        }
        return key, fields
    field_map = {
        "rig": "rig_id",
        "camera": "camera_id",
        "arena": "arena_id",
        "dish": "dish_design",
    }
    field = field_map[group_by]
    value = str(row.get(field) or "").strip()
    key = (value,)
    fields = {
        "group_by": group_by,
        field: value or None,
        "group_value": value or None,
    }
    return key, fields


def _rows_to_keypoint_group_summary(rows: list[dict[str, Any]], *, group_by: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key, descriptor = _keypoint_group_descriptor(row, group_by=group_by)
        if all(v in (None, "") for v in descriptor.values() if v is not None):
            continue
        rec_id = str(row.get("keypoint_recording_id") or row.get("recording_id") or "").strip() or None
        bucket = grouped.setdefault(
            key,
            {
                **descriptor,
                "datasets": 0,
                "recording_ids": set(),
                "success_rate_values": [],
                "kps_values": [],
                "duration_values": [],
            },
        )
        bucket["datasets"] += 1
        if rec_id:
            bucket["recording_ids"].add(rec_id)
        success_rate = _as_float(row.get("keypoint_success_rate_percent"))
        if success_rate is not None:
            bucket["success_rate_values"].append(success_rate)
        keypoints_per_second = _as_float(row.get("keypoint_keypoints_per_second"))
        if keypoints_per_second is not None:
            bucket["kps_values"].append(keypoints_per_second)
        duration_seconds = _as_float(row.get("keypoint_duration_seconds"))
        if duration_seconds is not None:
            bucket["duration_values"].append(duration_seconds)

    summary: list[dict[str, Any]] = []
    for bucket in grouped.values():
        success_rate_values = bucket["success_rate_values"]
        kps_values = bucket["kps_values"]
        duration_values = bucket["duration_values"]
        row: dict[str, Any] = {
            key: val
            for key, val in bucket.items()
            if key not in {"recording_ids", "success_rate_values", "kps_values", "duration_values"}
        }
        row["datasets"] = int(bucket["datasets"])
        row["recordings"] = int(len(bucket["recording_ids"]))
        row.update(_metric_summary("success_rate", success_rate_values))
        row.update(_metric_summary("kps", kps_values))
        row.update(_metric_summary("duration", duration_values))
        summary.append(row)
    summary.sort(
        key=lambda item: (
            -int(item["recordings"]),
            -int(item["datasets"]),
            str(item.get("group_value") or item.get("keypoint_method") or item.get("model_name") or item.get("model_path") or ""),
        )
    )
    return summary


def _query_dataset_ids_by_subject_lineage(
    registry: Registry,
    *,
    cross_id: Optional[str],
    genotype: Optional[str],
    dpf: Optional[int],
    dpf_min: Optional[int],
    dpf_max: Optional[int],
) -> set[str]:
    sql = [
        "SELECT DISTINCT d.dataset_id",
        "FROM datasets d",
        "JOIN recording_subject_overview rso ON rso.recording_id = d.recording_id",
        "WHERE 1=1",
    ]
    params: list[object] = []
    if cross_id is not None:
        sql.append("AND rso.cross_id = ?")
        params.append(str(cross_id))
    if genotype is not None:
        sql.append("AND rso.genotype = ?")
        params.append(str(genotype))
    if dpf is not None:
        sql.append("AND rso.dpf_at_acquisition = ?")
        params.append(int(dpf))
    if dpf_min is not None:
        sql.append("AND rso.dpf_at_acquisition >= ?")
        params.append(int(dpf_min))
    if dpf_max is not None:
        sql.append("AND rso.dpf_at_acquisition <= ?")
        params.append(int(dpf_max))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return {
        str(row["dataset_id"])
        for row in rows
        if row["dataset_id"] is not None
    }


def _query_detect_model_summary_rows(
    registry: Registry,
    *,
    scope: str,
    model_like: Optional[str],
    model_run_id: Optional[str],
    model_set_id: Optional[str],
    limit: Optional[int],
) -> list[dict[str, Any]]:
    view_name = (
        "recording_detect_model_performance_summary"
        if scope == "recording"
        else "detect_model_performance_summary"
    )
    sql = [f"SELECT * FROM {view_name} WHERE 1=1"]
    params: list[object] = []
    if model_run_id:
        sql.append("AND model_run_id = ?")
        params.append(str(model_run_id))
    if model_set_id:
        sql.append("AND model_set_id = ?")
        params.append(str(model_set_id))
    if model_like:
        sql.append(
            "AND lower(COALESCE(model_run_id, '') || ' ' || COALESCE(model_set_id, '') || ' ' || "
            "COALESCE(model_name, '') || ' ' || COALESCE(model_path, '')) LIKE ?"
        )
        params.append(f"%{str(model_like).strip().lower()}%")
    sql.append("ORDER BY recording_count DESC, dataset_count DESC, model_set_id, model_run_id, model_name, model_path")
    if limit is not None:
        sql.append("LIMIT ?")
        params.append(int(limit))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return [dict(row) for row in rows]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--dish-design", type=str, help="Exact dish design match.")
    parser.add_argument("--dish-design-like", type=str, help="Substring match for dish design.")
    parser.add_argument("--fish-id", type=str, help="Exact fish_id match.")
    parser.add_argument("--subject-count-min", type=int)
    parser.add_argument("--subject-count-max", type=int)
    parser.add_argument("--zarr-use", type=str, help="Exact zarr use match (training/analysis/inference/export/archive).")
    parser.add_argument("--fps-min", type=float)
    parser.add_argument("--fps-max", type=float)
    parser.add_argument("--exposure-min", type=float)
    parser.add_argument("--exposure-max", type=float)
    parser.add_argument("--frame-rate-min", type=float)
    parser.add_argument("--frame-rate-max", type=float)
    parser.add_argument("--gain-min", type=float)
    parser.add_argument("--gain-max", type=float)
    parser.add_argument("--video-codec", type=str)
    parser.add_argument("--video-pix-fmt", type=str)
    parser.add_argument("--format-encoder", type=str, help="Exact match on container encoder tag.")
    parser.add_argument("--format-title", type=str, help="Exact match on container title tag.")
    parser.add_argument("--format-comment", type=str, help="Exact match on container comment tag.")
    parser.add_argument("--encoder-name", type=str, help="Exact match on encoder name in comment.")
    parser.add_argument("--encoder-codec", type=str, help="Exact match on encoder codec in comment.")
    parser.add_argument("--encoder-preset", type=str, help="Exact match on encoder preset in comment.")
    parser.add_argument("--encoder-tuning", type=str, help="Exact match on encoder tuning in comment.")
    parser.add_argument("--encoder-rc", type=str, help="Exact match on encoder rate control in comment.")
    parser.add_argument("--compression", type=str, help="Compression name (e.g., lz4, zstd).")
    parser.add_argument("--camera-model", type=str)
    parser.add_argument("--camera-serial", type=str)
    parser.add_argument("--camera-id", type=str)
    parser.add_argument("--rig-id", type=str)
    parser.add_argument("--arena-id", type=str)
    parser.add_argument("--cross-id", type=str, help="Exact cross_id match via recording subject lineage.")
    parser.add_argument("--genotype", type=str, help="Exact genotype match via recording subject lineage.")
    parser.add_argument("--dpf", type=int, help="Exact dpf_at_acquisition match via recording subject lineage.")
    parser.add_argument("--dpf-min", type=int, help="Minimum dpf_at_acquisition via recording subject lineage.")
    parser.add_argument("--dpf-max", type=int, help="Maximum dpf_at_acquisition via recording subject lineage.")
    parser.add_argument(
        "--step-name",
        type=str,
        help="Recording step name filter from recording_step_status_latest (e.g. detect, keypoints).",
    )
    parser.add_argument(
        "--step-status",
        type=str,
        help="Recording step status filter: ok, missing, absent, na, error, or non-ok.",
    )
    parser.add_argument("--detect-coverage-min", type=float, help="Minimum detect coverage percent from detect performance view.")
    parser.add_argument("--detect-fps-min", type=float, help="Minimum detect inference average FPS from detect performance view.")
    parser.add_argument("--detect-read-ms-max", type=float, help="Maximum detect average read ms from detect performance view.")
    parser.add_argument("--detect-method", type=str, help="Exact detect method match from detect performance view.")
    parser.add_argument("--detect-model-like", type=str, help="Substring match on detect model_name/model_path.")
    parser.add_argument("--keypoint-method", type=str, help="Keypoint method filter from keypoint quality/performance views (or 'missing').")
    parser.add_argument("--keypoint-review-state", type=str, help="Keypoint review state filter (or 'missing').")
    parser.add_argument(
        "--keypoint-review-intended-use",
        type=str,
        help="Keypoint review intended-use filter (or 'missing').",
    )
    parser.add_argument(
        "--keypoint-usable-rate-min",
        type=float,
        help="Minimum keypoint usable_keypoints_rate from keypoint quality view.",
    )
    parser.add_argument(
        "--keypoint-usable-rate-max",
        type=float,
        help="Maximum keypoint usable_keypoints_rate from keypoint quality view.",
    )
    parser.add_argument(
        "--keypoint-success-rate-min",
        type=float,
        help="Minimum keypoint success_rate_percent from keypoint performance view.",
    )
    parser.add_argument(
        "--keypoint-kps-min",
        type=float,
        help="Minimum keypoint keypoints_per_second from keypoint performance view.",
    )
    parser.add_argument(
        "--keypoint-duration-max",
        type=float,
        help="Maximum keypoint duration_seconds from keypoint performance view.",
    )
    parser.add_argument(
        "--keypoint-fps-min",
        type=float,
        help="Minimum keypoint inference_average_fps from keypoint performance view.",
    )
    parser.add_argument(
        "--keypoint-model-like",
        type=str,
        help="Substring match on keypoint model_name/model_path/model_run_id/model_set_id.",
    )
    parser.add_argument(
        "--keypoint-model-only",
        action="store_true",
        help="Restrict keypoint performance matching to model-backed keypoint runs.",
    )
    parser.add_argument("--crop-review-state", type=str, help="Crop review state filter (or 'missing').")
    parser.add_argument(
        "--crop-review-intended-use",
        type=str,
        help="Crop review intended-use filter (or 'missing').",
    )
    parser.add_argument(
        "--crop-source-type",
        type=str,
        help="Crop detection source type filter (or 'missing').",
    )
    parser.add_argument(
        "--crop-percent-frames-min",
        type=float,
        help="Minimum crop percent_frames_with_crops from crop quality view.",
    )
    parser.add_argument(
        "--crop-percent-frames-max",
        type=float,
        help="Maximum crop percent_frames_with_crops from crop quality view.",
    )
    parser.add_argument(
        "--eye-mask-stage",
        choices=["eye_masks_runs", "refined_eye_masks_runs", "any"],
        default="any",
        help="Filter eye-mask performance rows by stage group (default: any).",
    )
    parser.add_argument(
        "--eye-mask-method",
        type=str,
        help="Eye-mask method filter from eye-mask performance view (or 'missing').",
    )
    parser.add_argument(
        "--eye-mask-review-state",
        type=str,
        help="Eye-mask review state filter (or 'missing').",
    )
    parser.add_argument(
        "--eye-mask-review-intended-use",
        type=str,
        help="Eye-mask review intended-use filter (or 'missing').",
    )
    parser.add_argument(
        "--eye-mask-reviewer",
        type=str,
        help="Eye-mask reviewer filter (or 'missing').",
    )
    parser.add_argument(
        "--eye-mask-stale-state",
        type=str,
        help="Eye-mask source_keypoint_stale.state filter (or 'missing').",
    )
    parser.add_argument(
        "--eye-mask-lifecycle-state",
        type=str,
        help="Eye-mask lifecycle state filter (approved/rejected/in_progress/stale or 'missing').",
    )
    parser.add_argument(
        "--eye-mask-source-keypoints-run",
        type=str,
        help="Eye-mask source_keypoints_run filter (or 'missing').",
    )
    parser.add_argument(
        "--eye-mask-success-rate-min",
        type=float,
        help="Minimum eye-mask successful_roi_pair_rate from eye-mask performance view.",
    )
    parser.add_argument(
        "--eye-mask-rois-per-second-min",
        type=float,
        help="Minimum eye-mask rois_per_second from eye-mask performance view.",
    )
    parser.add_argument(
        "--eye-mask-duration-max",
        type=float,
        help="Maximum eye-mask duration_seconds from eye-mask performance view.",
    )
    parser.add_argument(
        "--detect-model-only",
        action="store_true",
        help="Restrict detect performance matching to model-backed detect runs.",
    )
    parser.add_argument(
        "--group-by-model",
        action="store_true",
        help="Emit model-level summary from matched datasets using detect model performance rows.",
    )
    parser.add_argument(
        "--group-by",
        choices=["model", "rig", "camera", "arena", "dish"],
        help="Emit detect-performance summary grouped by this dimension.",
    )
    parser.add_argument(
        "--keypoint-group-by",
        choices=["model", "method", "rig", "camera", "arena", "dish"],
        help="Emit keypoint-performance summary grouped by this dimension.",
    )
    parser.add_argument(
        "--detect-model-summary",
        action="store_true",
        help="Query precomputed model summary views (no raw SQL needed).",
    )
    parser.add_argument(
        "--detect-model-summary-scope",
        choices=["recording", "dataset"],
        default="recording",
        help="Summary source view: recording-latest or dataset-latest model-backed detect rows.",
    )
    parser.add_argument("--detect-model-run-id", type=str, help="Filter detect-model summary by exact model run id.")
    parser.add_argument("--detect-model-set-id", type=str, help="Filter detect-model summary by exact model set id.")
    parser.add_argument(
        "--model-input",
        choices=["gray", "rgb"],
        help="Filter datasets by available downsample modality required for training.",
    )
    parser.add_argument("--path-contains", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--output-file-list", type=Path, help="Write matched zarr paths to file.")

    args = parser.parse_args(argv)
    if (
        args.dpf_min is not None
        and args.dpf_max is not None
        and int(args.dpf_min) > int(args.dpf_max)
    ):
        raise SystemExit("--dpf-min must be <= --dpf-max.")
    if args.group_by_model and args.group_by and args.group_by != "model":
        raise SystemExit("--group-by-model cannot be combined with --group-by non-model values.")
    if args.group_by and args.keypoint_group_by:
        raise SystemExit("--group-by cannot be combined with --keypoint-group-by.")
    step_status_filter = str(args.step_status).strip().lower() if args.step_status is not None else None
    if step_status_filter is not None:
        allowed_step_status = {"ok", "missing", "absent", "na", "error", "non-ok"}
        if step_status_filter not in allowed_step_status:
            raise SystemExit(
                "--step-status must be one of: ok, missing, absent, na, error, non-ok."
            )

    group_by = args.group_by or ("model" if args.group_by_model else None)
    keypoint_group_by = args.keypoint_group_by
    if args.detect_model_summary and args.output_file_list:
        raise SystemExit("--output-file-list is only supported for dataset-row query mode.")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    if args.detect_model_summary:
        try:
            summary_rows = _query_detect_model_summary_rows(
                registry,
                scope=str(args.detect_model_summary_scope),
                model_like=args.detect_model_like,
                model_run_id=args.detect_model_run_id,
                model_set_id=args.detect_model_set_id,
                limit=args.limit,
            )
        finally:
            registry.close()
        if args.json:
            print(json.dumps(summary_rows, indent=2))
        else:
            for row in summary_rows:
                print(
                    f"set_id={row.get('model_set_id') or '-'}\t"
                    f"run_id={row.get('model_run_id') or '-'}\t"
                    f"model={row.get('model_name') or '-'}\t"
                    f"recordings={row.get('recording_count')}\t"
                    f"datasets={row.get('dataset_count')}\t"
                    f"coverage_p50={row.get('coverage_p50') if row.get('coverage_p50') is not None else '-'}\t"
                    f"fps_p50={row.get('fps_p50') if row.get('fps_p50') is not None else '-'}\t"
                    f"read_ms_p50={row.get('read_ms_p50') if row.get('read_ms_p50') is not None else '-'}"
                )
        return 0

    use_subject_filters = any(
        value is not None
        for value in (args.cross_id, args.genotype, args.dpf, args.dpf_min, args.dpf_max)
    )
    use_step_status_filters = any(
        value is not None
        for value in (args.step_name, step_status_filter)
    )
    use_detect_filters = any(
        value is not None
        for value in (
            args.detect_coverage_min,
            args.detect_fps_min,
            args.detect_read_ms_max,
            args.detect_method,
            args.detect_model_like,
        )
    ) or bool(args.detect_model_only) or bool(group_by)
    use_crop_filters = any(
        value is not None
        for value in (
            args.crop_review_state,
            args.crop_review_intended_use,
            args.crop_source_type,
            args.crop_percent_frames_min,
            args.crop_percent_frames_max,
        )
    )
    use_eye_mask_filters = any(
        value is not None
        for value in (
            args.eye_mask_method,
            args.eye_mask_review_state,
            args.eye_mask_review_intended_use,
            args.eye_mask_reviewer,
            args.eye_mask_stale_state,
            args.eye_mask_lifecycle_state,
            args.eye_mask_source_keypoints_run,
            args.eye_mask_success_rate_min,
            args.eye_mask_rois_per_second_min,
            args.eye_mask_duration_max,
        )
    ) or str(args.eye_mask_stage) != "any"
    use_keypoint_filters = any(
        value is not None
        for value in (
            args.keypoint_method,
            args.keypoint_review_state,
            args.keypoint_review_intended_use,
            args.keypoint_usable_rate_min,
            args.keypoint_usable_rate_max,
            args.keypoint_success_rate_min,
            args.keypoint_kps_min,
            args.keypoint_duration_max,
            args.keypoint_fps_min,
            args.keypoint_model_like,
        )
    ) or bool(args.keypoint_model_only) or bool(keypoint_group_by)

    try:
        rows = registry.query_datasets(
            dish_design=args.dish_design,
            dish_design_like=args.dish_design_like,
            fish_id=args.fish_id,
            subject_count_min=args.subject_count_min,
            subject_count_max=args.subject_count_max,
            zarr_use=args.zarr_use,
            fps_min=args.fps_min,
            fps_max=args.fps_max,
            exposure_min=args.exposure_min,
            exposure_max=args.exposure_max,
            frame_rate_min=args.frame_rate_min,
            frame_rate_max=args.frame_rate_max,
            gain_min=args.gain_min,
            gain_max=args.gain_max,
            video_codec=args.video_codec,
            video_pix_fmt=args.video_pix_fmt,
            format_encoder=args.format_encoder,
            format_title=args.format_title,
            format_comment=args.format_comment,
            encoder_name=args.encoder_name,
            encoder_codec=args.encoder_codec,
            encoder_preset=args.encoder_preset,
            encoder_tuning=args.encoder_tuning,
            encoder_rc=args.encoder_rc,
            compression_name=args.compression,
            camera_model=args.camera_model,
            camera_serial=args.camera_serial,
            camera_id=args.camera_id,
            rig_id=args.rig_id,
            arena_id=args.arena_id,
            model_input=args.model_input,
            path_contains=args.path_contains,
            # Apply row limit after lineage/stage filtering.
            limit=(
                None
                if (
                    use_subject_filters
                    or use_step_status_filters
                    or use_detect_filters
                    or use_crop_filters
                    or use_eye_mask_filters
                    or use_keypoint_filters
                )
                else args.limit
            ),
        )
        if use_subject_filters:
            try:
                allowed_dataset_ids = _query_dataset_ids_by_subject_lineage(
                    registry,
                    cross_id=args.cross_id,
                    genotype=args.genotype,
                    dpf=args.dpf,
                    dpf_min=args.dpf_min,
                    dpf_max=args.dpf_max,
                )
            except Exception as exc:
                raise SystemExit(
                    "Subject-lineage filters (--cross-id/--genotype/--dpf/--dpf-min/--dpf-max) require "
                    f"`recording_subject_overview` to be queryable: {exc}"
                ) from exc
            rows = [
                row
                for row in rows
                if str(row["dataset_id"]) in allowed_dataset_ids
            ]

        result_rows: list[dict[str, Any]] = [dict(row) for row in rows]
        if use_step_status_filters:
            dataset_ids = [str(row["dataset_id"]) for row in result_rows if row.get("dataset_id") is not None]
            step_map = _query_recording_step_status_map(
                registry,
                dataset_ids=dataset_ids,
                step_name=args.step_name,
                status=step_status_filter,
            )
            filtered: list[dict[str, Any]] = []
            for row in result_rows:
                dataset_id = str(row.get("dataset_id") or "")
                candidates = step_map.get(dataset_id, [])
                if not candidates:
                    continue
                selected = sorted(
                    candidates,
                    key=lambda item: (
                        str(item.get("updated_utc") or ""),
                        str(item.get("step_name") or ""),
                        str(item.get("run_name") or ""),
                    ),
                )[-1]
                row["recording_step_name"] = selected.get("step_name")
                row["recording_step_status"] = selected.get("status")
                row["recording_step_run_name"] = selected.get("run_name")
                row["recording_step_method"] = selected.get("method")
                row["recording_step_coverage_pct"] = selected.get("coverage_pct")
                row["recording_step_source"] = selected.get("source")
                row["recording_step_updated_utc"] = selected.get("updated_utc")
                filtered.append(row)
            result_rows = filtered

        if use_detect_filters:
            dataset_ids = [str(row["dataset_id"]) for row in rows if row["dataset_id"] is not None]
            model_only_view = bool(args.detect_model_only or group_by == "model" or args.detect_model_like)
            detect_map = _query_detect_performance_map(
                registry,
                dataset_ids=dataset_ids,
                model_only=model_only_view,
            )
            method_filter = (str(args.detect_method).strip().lower() if args.detect_method else None)
            model_like_filter = (str(args.detect_model_like).strip().lower() if args.detect_model_like else None)
            filtered: list[dict[str, Any]] = []
            for row in result_rows:
                dataset_id = str(row.get("dataset_id") or "")
                detect = detect_map.get(dataset_id)
                if group_by is not None and detect is None:
                    continue
                if bool(args.detect_model_only) and detect is None:
                    continue
                if any(
                    value is not None
                    for value in (
                        args.detect_coverage_min,
                        args.detect_fps_min,
                        args.detect_read_ms_max,
                        method_filter,
                        model_like_filter,
                    )
                ) and detect is None:
                    continue
                if detect is not None:
                    detect_method = str(detect.get("detection_method") or "").strip().lower()
                    model_name = str(detect.get("model_name") or "").strip()
                    model_path = str(detect.get("model_path") or "").strip()
                    model_run_id = str(detect.get("model_run_id") or "").strip()
                    model_set_id = str(detect.get("model_set_id") or "").strip()
                    coverage = _as_float(detect.get("coverage_percent"))
                    fps = _as_float(detect.get("inference_average_fps"))
                    read_ms = _as_float(detect.get("inference_avg_read_ms"))
                    if args.detect_coverage_min is not None and (coverage is None or coverage < float(args.detect_coverage_min)):
                        continue
                    if args.detect_fps_min is not None and (fps is None or fps < float(args.detect_fps_min)):
                        continue
                    if args.detect_read_ms_max is not None and (read_ms is None or read_ms > float(args.detect_read_ms_max)):
                        continue
                    if method_filter is not None and detect_method != method_filter:
                        continue
                    if model_like_filter is not None:
                        haystack = f"{model_name} {model_path} {model_run_id} {model_set_id}".lower()
                        if model_like_filter not in haystack:
                            continue
                    row["detect_run"] = detect.get("detect_run")
                    row["detect_created_utc"] = detect.get("detect_created_utc")
                    row["detect_recording_id"] = detect.get("recording_id")
                    row["detect_method"] = detect.get("detection_method")
                    row["detect_model_run_id"] = detect.get("model_run_id")
                    row["detect_model_set_id"] = detect.get("model_set_id")
                    row["detect_model_name"] = detect.get("model_name")
                    row["detect_model_path"] = detect.get("model_path")
                    row["detect_coverage_percent"] = detect.get("coverage_percent")
                    row["detect_inference_average_fps"] = detect.get("inference_average_fps")
                    row["detect_inference_avg_read_ms"] = detect.get("inference_avg_read_ms")
                filtered.append(row)
            result_rows = filtered

        if use_crop_filters:
            dataset_ids = [str(row["dataset_id"]) for row in result_rows if row.get("dataset_id") is not None]
            crop_map = _query_crop_quality_map(registry, dataset_ids=dataset_ids)
            filtered = []
            for row in result_rows:
                dataset_id = str(row.get("dataset_id") or "")
                crop = crop_map.get(dataset_id)
                if crop is None:
                    continue

                review_state = crop.get("review_state")
                review_intended_use = crop.get("review_intended_use")
                source_type = crop.get("detection_source_type")
                percent_frames = _as_float(crop.get("percent_frames_with_crops"))

                if not _matches_optional_text_filter(review_state, args.crop_review_state):
                    continue
                if not _matches_optional_text_filter(review_intended_use, args.crop_review_intended_use):
                    continue
                if not _matches_optional_text_filter(source_type, args.crop_source_type):
                    continue
                if args.crop_percent_frames_min is not None and (
                    percent_frames is None or percent_frames < float(args.crop_percent_frames_min)
                ):
                    continue
                if args.crop_percent_frames_max is not None and (
                    percent_frames is None or percent_frames > float(args.crop_percent_frames_max)
                ):
                    continue

                row["crop_run"] = crop.get("crop_run")
                row["crop_created_utc"] = crop.get("crop_created_utc")
                row["crop_recording_id"] = crop.get("recording_id")
                row["crop_source_detect_run"] = crop.get("source_detect_run")
                row["crop_source_refined_run"] = crop.get("source_refined_run")
                row["crop_source_type"] = crop.get("detection_source_type")
                row["crop_source_path"] = crop.get("detection_source_path")
                row["crop_total_rois"] = crop.get("total_rois")
                row["crop_frames_with_crops"] = crop.get("frames_with_crops")
                row["crop_total_frames"] = crop.get("total_frames")
                row["crop_percent_frames_with_crops"] = crop.get("percent_frames_with_crops")
                row["crop_includes_interpolated"] = crop.get("includes_interpolated")
                row["crop_n_real_detections"] = crop.get("n_real_detections")
                row["crop_n_interpolated_detections"] = crop.get("n_interpolated_detections")
                row["crop_review_state"] = crop.get("review_state")
                row["crop_review_method"] = crop.get("review_method")
                row["crop_review_intended_use"] = crop.get("review_intended_use")
                row["crop_review_reviewer"] = crop.get("review_reviewer")
                row["crop_review_timestamp_utc"] = crop.get("review_timestamp_utc")
                row["crop_review_notes"] = crop.get("review_notes")
                filtered.append(row)
            result_rows = filtered

        if use_eye_mask_filters:
            dataset_ids = [str(row["dataset_id"]) for row in result_rows if row.get("dataset_id") is not None]
            eye_mask_map = _query_eye_mask_performance_map(registry, dataset_ids=dataset_ids)
            filtered = []
            stage_filter = None if str(args.eye_mask_stage) == "any" else str(args.eye_mask_stage)
            method_filter = str(args.eye_mask_method).strip().lower() if args.eye_mask_method else None
            for row in result_rows:
                dataset_id = str(row.get("dataset_id") or "")
                candidates = eye_mask_map.get(dataset_id, [])
                if stage_filter is not None:
                    candidates = [
                        candidate
                        for candidate in candidates
                        if str(candidate.get("stage_group") or "") == stage_filter
                    ]
                elif any(
                    value is not None
                    for value in (
                        args.eye_mask_review_state,
                        args.eye_mask_review_intended_use,
                        args.eye_mask_reviewer,
                        args.eye_mask_stale_state,
                        args.eye_mask_lifecycle_state,
                    )
                ):
                    refined_candidates = [
                        candidate
                        for candidate in candidates
                        if str(candidate.get("stage_group") or "") == "refined_eye_masks_runs"
                    ]
                    if refined_candidates:
                        candidates = refined_candidates
                if not candidates:
                    continue

                matching_candidates: list[dict[str, Any]] = []
                for candidate in candidates:
                    if method_filter is not None:
                        method_value = str(candidate.get("method") or "").strip().lower()
                        if method_filter == "missing":
                            if method_value:
                                continue
                        elif method_value != method_filter:
                            continue
                    if not _matches_optional_text_filter(candidate.get("review_state"), args.eye_mask_review_state):
                        continue
                    if not _matches_optional_text_filter(
                        candidate.get("review_intended_use"),
                        args.eye_mask_review_intended_use,
                    ):
                        continue
                    if not _matches_optional_text_filter(candidate.get("review_reviewer"), args.eye_mask_reviewer):
                        continue
                    if not _matches_optional_text_filter(
                        candidate.get("source_keypoint_stale_state"),
                        args.eye_mask_stale_state,
                    ):
                        continue
                    if not _matches_optional_text_filter(candidate.get("lifecycle_state"), args.eye_mask_lifecycle_state):
                        continue
                    if not _matches_optional_text_filter(
                        candidate.get("source_keypoints_run"),
                        args.eye_mask_source_keypoints_run,
                    ):
                        continue

                    success_rate = _as_float(candidate.get("successful_roi_pair_rate"))
                    if args.eye_mask_success_rate_min is not None and (
                        success_rate is None or success_rate < float(args.eye_mask_success_rate_min)
                    ):
                        continue

                    rois_per_second = _as_float(candidate.get("rois_per_second"))
                    if args.eye_mask_rois_per_second_min is not None and (
                        rois_per_second is None or rois_per_second < float(args.eye_mask_rois_per_second_min)
                    ):
                        continue

                    duration_seconds = _as_float(candidate.get("duration_seconds"))
                    if args.eye_mask_duration_max is not None and (
                        duration_seconds is None or duration_seconds > float(args.eye_mask_duration_max)
                    ):
                        continue

                    matching_candidates.append(candidate)

                if not matching_candidates:
                    continue

                selected = _pick_eye_mask_candidate(matching_candidates, stage_filter=stage_filter)
                row["eye_mask_stage_group"] = selected.get("stage_group")
                row["eye_mask_run"] = selected.get("run_name")
                row["eye_mask_created_utc"] = selected.get("run_created_utc")
                row["eye_mask_recording_id"] = selected.get("recording_id")
                row["eye_mask_method"] = selected.get("method")
                row["eye_mask_source_crop_run"] = selected.get("source_crop_run")
                row["eye_mask_source_keypoint_group"] = selected.get("source_keypoint_group")
                row["eye_mask_source_keypoints_run"] = selected.get("source_keypoints_run")
                row["eye_mask_source_eye_masks_run"] = selected.get("source_eye_masks_run")
                row["eye_mask_source_eye_masks_method"] = selected.get("source_eye_masks_method")
                row["eye_mask_total_rois"] = selected.get("total_rois")
                row["eye_mask_successful_eyes"] = selected.get("successful_eyes")
                row["eye_mask_successful_roi_pairs"] = selected.get("successful_roi_pairs")
                row["eye_mask_successful_roi_pair_rate"] = selected.get("successful_roi_pair_rate")
                row["eye_mask_duration_seconds"] = selected.get("duration_seconds")
                row["eye_mask_rois_per_second"] = selected.get("rois_per_second")
                row["eye_mask_inference_duration_seconds"] = selected.get("inference_duration_seconds")
                row["eye_mask_inference_average_fps"] = selected.get("inference_average_fps")
                row["eye_mask_review_state"] = selected.get("review_state")
                row["eye_mask_review_method"] = selected.get("review_method")
                row["eye_mask_review_intended_use"] = selected.get("review_intended_use")
                row["eye_mask_review_reviewer"] = selected.get("review_reviewer")
                row["eye_mask_review_timestamp_utc"] = selected.get("review_timestamp_utc")
                row["eye_mask_source_keypoint_stale_state"] = selected.get("source_keypoint_stale_state")
                row["eye_mask_source_keypoint_stale_reason"] = selected.get("source_keypoint_stale_reason")
                row["eye_mask_source_keypoint_stale_timestamp_utc"] = selected.get(
                    "source_keypoint_stale_timestamp_utc"
                )
                row["eye_mask_lifecycle_state"] = selected.get("lifecycle_state")
                row["eye_mask_lifecycle_reason"] = selected.get("lifecycle_reason")
                filtered.append(row)
            result_rows = filtered

        if use_keypoint_filters:
            dataset_ids = [str(row["dataset_id"]) for row in result_rows if row.get("dataset_id") is not None]
            keypoint_quality_map = _query_keypoint_quality_map(registry, dataset_ids=dataset_ids)
            model_only_view = bool(args.keypoint_model_only or keypoint_group_by == "model" or args.keypoint_model_like)
            keypoint_perf_map = _query_keypoint_performance_map(
                registry,
                dataset_ids=dataset_ids,
                model_only=model_only_view,
            )
            quality_filters_active = any(
                value is not None
                for value in (
                    args.keypoint_review_state,
                    args.keypoint_review_intended_use,
                    args.keypoint_usable_rate_min,
                    args.keypoint_usable_rate_max,
                )
            )
            model_like_filter = str(args.keypoint_model_like).strip().lower() if args.keypoint_model_like else None
            performance_filters_active = any(
                value is not None
                for value in (
                    args.keypoint_success_rate_min,
                    args.keypoint_kps_min,
                    args.keypoint_duration_max,
                    args.keypoint_fps_min,
                    model_like_filter,
                )
            ) or bool(args.keypoint_model_only) or bool(keypoint_group_by)
            filtered = []
            for row in result_rows:
                dataset_id = str(row.get("dataset_id") or "")
                quality_candidates = keypoint_quality_map.get(dataset_id, [])
                if args.keypoint_method is not None:
                    quality_candidates = [
                        candidate
                        for candidate in quality_candidates
                        if _matches_optional_text_filter(candidate.get("keypoint_method"), args.keypoint_method)
                    ]

                matching_quality_candidates: list[dict[str, Any]] = []
                for candidate in quality_candidates:
                    if not _matches_optional_text_filter(candidate.get("review_state"), args.keypoint_review_state):
                        continue
                    if not _matches_optional_text_filter(
                        candidate.get("review_intended_use"),
                        args.keypoint_review_intended_use,
                    ):
                        continue
                    usable_rate = _as_float(candidate.get("usable_keypoints_rate"))
                    if args.keypoint_usable_rate_min is not None and (
                        usable_rate is None or usable_rate < float(args.keypoint_usable_rate_min)
                    ):
                        continue
                    if args.keypoint_usable_rate_max is not None and (
                        usable_rate is None or usable_rate > float(args.keypoint_usable_rate_max)
                    ):
                        continue
                    matching_quality_candidates.append(candidate)

                perf = keypoint_perf_map.get(dataset_id)
                if perf is not None and args.keypoint_method is not None:
                    if not _matches_optional_text_filter(perf.get("keypoint_method"), args.keypoint_method):
                        perf = None
                selected_perf = None
                if perf is not None:
                    success_rate = _as_float(perf.get("success_rate_percent"))
                    keypoints_per_second = _as_float(perf.get("keypoints_per_second"))
                    duration_seconds = _as_float(perf.get("duration_seconds"))
                    inference_fps = _as_float(perf.get("inference_average_fps"))
                    if args.keypoint_success_rate_min is not None and (
                        success_rate is None or success_rate < float(args.keypoint_success_rate_min)
                    ):
                        perf = None
                    if perf is not None and args.keypoint_kps_min is not None and (
                        keypoints_per_second is None or keypoints_per_second < float(args.keypoint_kps_min)
                    ):
                        perf = None
                    if perf is not None and args.keypoint_duration_max is not None and (
                        duration_seconds is None or duration_seconds > float(args.keypoint_duration_max)
                    ):
                        perf = None
                    if perf is not None and args.keypoint_fps_min is not None and (
                        inference_fps is None or inference_fps < float(args.keypoint_fps_min)
                    ):
                        perf = None
                    if perf is not None and model_like_filter is not None:
                        model_name = str(perf.get("model_name") or "").strip()
                        model_path = str(perf.get("model_path") or "").strip()
                        model_run_id = str(perf.get("model_run_id") or "").strip()
                        model_set_id = str(perf.get("model_set_id") or "").strip()
                        haystack = f"{model_name} {model_path} {model_run_id} {model_set_id}".lower()
                        if model_like_filter not in haystack:
                            perf = None
                    selected_perf = perf

                if quality_filters_active and not matching_quality_candidates:
                    continue
                if performance_filters_active and selected_perf is None:
                    continue
                if not quality_filters_active and not performance_filters_active and args.keypoint_method is not None:
                    if not matching_quality_candidates and selected_perf is None:
                        continue

                selected_quality = None
                if matching_quality_candidates:
                    if selected_perf is not None and args.keypoint_method is None:
                        perf_method = str(selected_perf.get("keypoint_method") or "").strip().lower()
                        aligned = [
                            candidate
                            for candidate in matching_quality_candidates
                            if str(candidate.get("keypoint_method") or "").strip().lower() == perf_method
                        ]
                        if aligned:
                            matching_quality_candidates = aligned
                    selected_quality = _pick_keypoint_quality_candidate(matching_quality_candidates)

                if selected_quality is not None:
                    row["keypoint_quality_refined_run"] = selected_quality.get("refined_run")
                    row["keypoint_quality_refined_created_utc"] = selected_quality.get("refined_created_utc")
                    row["keypoint_quality_source_keypoint_run"] = selected_quality.get("source_keypoint_run")
                    row["keypoint_quality_method"] = selected_quality.get("keypoint_method")
                    row["keypoint_review_state"] = selected_quality.get("review_state")
                    row["keypoint_review_intended_use"] = selected_quality.get("review_intended_use")
                    row["keypoint_review_reviewer"] = selected_quality.get("review_reviewer")
                    row["keypoint_review_timestamp_utc"] = selected_quality.get("review_timestamp_utc")
                    row["keypoint_usable_keypoints"] = selected_quality.get("usable_keypoints")
                    row["keypoint_total_keypoints"] = selected_quality.get("total_keypoints")
                    row["keypoint_usable_keypoints_rate"] = selected_quality.get("usable_keypoints_rate")
                    row["keypoint_raw_success_rate"] = selected_quality.get("raw_keypoints_success_rate")
                    row["keypoint_raw_successful"] = selected_quality.get("raw_keypoints_successful")
                    row["keypoint_quality_updated_utc"] = selected_quality.get("quality_updated_utc")

                if selected_perf is not None:
                    row["keypoint_run"] = selected_perf.get("keypoint_run")
                    row["keypoint_created_utc"] = selected_perf.get("keypoint_created_utc")
                    row["keypoint_recording_id"] = selected_perf.get("recording_id")
                    row["keypoint_performance_method"] = selected_perf.get("keypoint_method")
                    row["keypoint_model_run_id"] = selected_perf.get("model_run_id")
                    row["keypoint_model_set_id"] = selected_perf.get("model_set_id")
                    row["keypoint_model_name"] = selected_perf.get("model_name")
                    row["keypoint_model_path"] = selected_perf.get("model_path")
                    row["keypoint_source_crop_run"] = selected_perf.get("source_crop_run")
                    row["keypoint_source_detect_run"] = selected_perf.get("source_detect_run")
                    row["keypoint_source_refined_run"] = selected_perf.get("source_refined_run")
                    row["keypoint_total_rois"] = selected_perf.get("total_rois")
                    row["keypoint_successful_detections"] = selected_perf.get("successful_detections")
                    row["keypoint_failed_detections"] = selected_perf.get("failed_detections")
                    row["keypoint_success_rate_percent"] = selected_perf.get("success_rate_percent")
                    row["keypoint_frames_with_keypoints"] = selected_perf.get("frames_with_keypoints")
                    row["keypoint_mean_confidence"] = selected_perf.get("mean_confidence")
                    row["keypoint_duration_seconds"] = selected_perf.get("duration_seconds")
                    row["keypoint_inference_duration_seconds"] = selected_perf.get("inference_duration_seconds")
                    row["keypoint_keypoints_per_second"] = selected_perf.get("keypoints_per_second")
                    row["keypoint_inference_average_fps"] = selected_perf.get("inference_average_fps")
                    row["keypoint_batch_size"] = selected_perf.get("batch_size")
                    row["keypoint_imgsz"] = selected_perf.get("imgsz")
                    row["keypoint_conf_threshold"] = selected_perf.get("conf_threshold")
                    row["keypoint_iou_threshold"] = selected_perf.get("iou_threshold")

                row["keypoint_method"] = (
                    (selected_perf.get("keypoint_method") if selected_perf is not None else None)
                    or (selected_quality.get("keypoint_method") if selected_quality is not None else None)
                )
                filtered.append(row)
            result_rows = filtered

        if args.limit is not None:
            result_rows = result_rows[: int(args.limit)]
    finally:
        registry.close()

    if keypoint_group_by is not None:
        keypoint_summary = _rows_to_keypoint_group_summary(result_rows, group_by=keypoint_group_by)
        if args.json:
            print(json.dumps(keypoint_summary, indent=2))
        else:
            for row in keypoint_summary:
                if keypoint_group_by == "model":
                    group_label = f"model={row.get('model_name') or '-'}"
                elif keypoint_group_by == "method":
                    group_label = f"method={row.get('keypoint_method') or '-'}"
                else:
                    group_label = f"{keypoint_group_by}={row.get('group_value') or '-'}"
                print(
                    f"{group_label}\t"
                    f"run_id={row.get('model_run_id') or '-'}\t"
                    f"set_id={row.get('model_set_id') or '-'}\t"
                    f"path={row.get('model_path') or '-'}\t"
                    f"recordings={row.get('recordings')}\t"
                    f"datasets={row.get('datasets')}\t"
                    f"success_rate_avg={row.get('success_rate_avg') if row.get('success_rate_avg') is not None else '-'}\t"
                    f"success_rate_p50={row.get('success_rate_p50') if row.get('success_rate_p50') is not None else '-'}\t"
                    f"kps_avg={row.get('kps_avg') if row.get('kps_avg') is not None else '-'}\t"
                    f"kps_p50={row.get('kps_p50') if row.get('kps_p50') is not None else '-'}\t"
                    f"duration_avg={row.get('duration_avg') if row.get('duration_avg') is not None else '-'}\t"
                    f"duration_p50={row.get('duration_p50') if row.get('duration_p50') is not None else '-'}"
                )
        return 0

    if group_by is not None:
        model_summary = _rows_to_group_summary(result_rows, group_by=group_by)
        if args.json:
            print(json.dumps(model_summary, indent=2))
        else:
            for row in model_summary:
                group_label = (
                    f"{group_by}={row.get('group_value') or '-'}"
                    if group_by != "model"
                    else f"model={row.get('model_name') or '-'}"
                )
                print(
                    f"{group_label}\t"
                    f"run_id={row.get('model_run_id') or '-'}\t"
                    f"set_id={row.get('model_set_id') or '-'}\t"
                    f"path={row.get('model_path') or '-'}\t"
                    f"recordings={row.get('recordings')}\t"
                    f"datasets={row.get('datasets')}\t"
                    f"coverage_avg={row.get('coverage_avg') if row.get('coverage_avg') is not None else '-'}\t"
                    f"coverage_p50={row.get('coverage_p50') if row.get('coverage_p50') is not None else '-'}\t"
                    f"fps_avg={row.get('fps_avg') if row.get('fps_avg') is not None else '-'}\t"
                    f"fps_p50={row.get('fps_p50') if row.get('fps_p50') is not None else '-'}\t"
                    f"read_ms_avg={row.get('read_ms_avg') if row.get('read_ms_avg') is not None else '-'}\t"
                    f"read_ms_p50={row.get('read_ms_p50') if row.get('read_ms_p50') is not None else '-'}"
                )
        return 0

    if args.json:
        payload = result_rows
        print(json.dumps(payload, indent=2))
    else:
        for row in result_rows:
            encoder_name = row["encoder_name"] if "encoder_name" in row.keys() else None
            format_encoder = row["format_encoder"] if "format_encoder" in row.keys() else None
            print(
                f"{row['zarr_path']}\t"
                f"dish={row['dish_design'] or '-'}\t"
                f"fps={row['fps'] or '-'}\t"
                f"exposure_us={row['exposure'] or '-'}\t"
                f"codec={row['video_codec'] or '-'}\t"
                f"pixfmt={row['video_pix_fmt'] or '-'}\t"
                f"encoder={encoder_name or format_encoder or '-'}"
            )

    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text(
            "\n".join([str(row["zarr_path"]) for row in result_rows]) + ("\n" if result_rows else ""),
            encoding="utf-8",
        )
        print(f"Wrote {len(result_rows)} paths to {args.output_file_list}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
