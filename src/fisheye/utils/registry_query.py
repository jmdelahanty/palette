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
    parser.add_argument("--detect-coverage-min", type=float, help="Minimum detect coverage percent from detect performance view.")
    parser.add_argument("--detect-fps-min", type=float, help="Minimum detect inference average FPS from detect performance view.")
    parser.add_argument("--detect-read-ms-max", type=float, help="Maximum detect average read ms from detect performance view.")
    parser.add_argument("--detect-method", type=str, help="Exact detect method match from detect performance view.")
    parser.add_argument("--detect-model-like", type=str, help="Substring match on detect model_name/model_path.")
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

    group_by = args.group_by or ("model" if args.group_by_model else None)
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
            # Apply row limit after lineage/detect/crop filtering.
            limit=(None if (use_subject_filters or use_detect_filters or use_crop_filters or use_eye_mask_filters) else args.limit),
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

        if args.limit is not None:
            result_rows = result_rows[: int(args.limit)]
    finally:
        registry.close()

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
