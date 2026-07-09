"""Keypoint performance/profile row extractors for registry scans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _decode_attr

_ABSENT_SOURCE_RUN_SENTINELS = {"unknown", "none", "null", "n/a", "na"}


def _decode_source_run(value: Any) -> Optional[str]:
    text = _decode_attr(value)
    if text is None:
        return None
    if text.strip().lower() in _ABSENT_SOURCE_RUN_SENTINELS:
        return None
    return text


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


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


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


def _lookup_attr(attrs: Mapping[str, Any], fallback: Mapping[str, Any], key: str) -> Any:
    value = attrs.get(key)
    if value is not None:
        return value
    return fallback.get(key)


def _as_bool_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return 1
        if text in {"0", "false", "no", "n"}:
            return 0
    coerced = _as_int(value)
    return int(coerced) if coerced is not None else None


def _resolve_source_roi_pixel_contract(
    attrs: Mapping[str, Any],
    provenance_inputs: Mapping[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    contract = (
        _coerce_mapping(_lookup_attr(attrs, provenance_inputs, "source_roi_pixel_contract"))
        or _coerce_mapping(_lookup_attr(attrs, provenance_inputs, "roi_pixel_contract"))
    )
    contract_name = (
        _decode_attr(_lookup_attr(attrs, provenance_inputs, "source_roi_pixel_contract_name"))
        or _decode_attr(_lookup_attr(attrs, provenance_inputs, "roi_pixel_contract_name"))
    )
    if contract and contract_name is None:
        contract_name = _decode_attr(contract.get("name"))
    return contract, contract_name


def _open_child_group(parent: Any, key: str) -> Any:
    store = getattr(parent, "store", None)
    if store is None:
        return None
    parent_path = _decode_attr(getattr(parent, "path", None))
    child_path = f"{parent_path}/{key}" if parent_path else key
    try:
        return zarr.open_group(store=store, path=child_path, mode="r")
    except TypeError:
        try:
            return zarr.open_group(store, mode="r", path=child_path)
        except Exception:
            return None
    except Exception:
        return None


def _get_group(parent: Any, key: str) -> Any:
    child = None
    getter = getattr(parent, "get", None)
    if callable(getter):
        try:
            child = getter(key)
        except Exception:
            child = None
    if child is not None:
        return child
    try:
        child = parent[key]
    except Exception:
        child = None
    if child is not None:
        return child
    return _open_child_group(parent, key)


def _keypoint_run_names(parent: zarr.Group) -> List[str]:
    names: List[str] = []
    try:
        names = list(parent.group_keys())
    except Exception:
        names = [name for name in parent.keys() if isinstance(name, str)]
    store_root = getattr(getattr(parent, "store", None), "root", None)
    parent_path = _decode_attr(getattr(parent, "path", None))
    if store_root is not None and parent_path:
        try:
            base_path = Path(store_root) / parent_path
            names.extend(
                child.name
                for child in base_path.iterdir()
                if child.is_dir() and (child / "zarr.json").is_file()
            )
        except Exception:
            pass
    return sorted({str(name) for name in names})


def _extract_stat_value(metric_payload: Mapping[str, Any], key: str) -> Optional[float]:
    if not isinstance(metric_payload, Mapping):
        return None
    stats = metric_payload.get("stats")
    if isinstance(stats, Mapping):
        return _as_float(stats.get(key))
    return _as_float(metric_payload.get(key))


def _normalize_kpt_shape_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        try:
            return _canonical_json_text([int(v) for v in value])
        except Exception:
            try:
                return _canonical_json_text(list(value))
            except Exception:
                return None
    return _decode_attr(value)


def _normalize_json_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
        return text or None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        return _canonical_json_text(dict(value))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        try:
            return _canonical_json_text(list(value))
        except Exception:
            return None
    try:
        return _canonical_json_text(value)
    except Exception:
        return _decode_attr(value)


def _extract_keypoint_performance_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    keypoints_parent = root.get("keypoints_runs")
    if keypoints_parent is None:
        return []

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = utc_now()

    rows: List[Dict[str, Any]] = []
    for keypoint_run in _keypoint_run_names(keypoints_parent):
        keypoint_group = _get_group(keypoints_parent, keypoint_run)
        if keypoint_group is None:
            continue
        attrs = dict(keypoint_group.attrs)
        summary = _coerce_mapping(attrs.get("summary_statistics")) or {}
        parameters = _coerce_mapping(attrs.get("parameters")) or {}
        provenance = _coerce_mapping(attrs.get("provenance")) or {}
        provenance_inputs = _coerce_mapping(provenance.get("inputs")) or {}
        model_resolution = _coerce_mapping(provenance.get("model_resolution")) or {}
        model_resolution_selected = _coerce_mapping(model_resolution.get("selected")) or {}
        source_roi_pixel_contract, source_roi_pixel_contract_name = _resolve_source_roi_pixel_contract(
            attrs,
            provenance_inputs,
        )

        keypoint_created_utc = (
            _decode_attr(attrs.get("created_at_utc"))
            or _decode_attr(attrs.get("keypoints_timestamp_utc"))
            or _decode_attr(attrs.get("created_utc"))
            or _decode_attr(attrs.get("timestamp_utc"))
            or _decode_attr(provenance.get("created_at_utc"))
        )
        keypoint_method = (
            _decode_attr(attrs.get("method"))
            or _decode_attr(provenance.get("method"))
        )
        model_run_id = (
            _decode_attr(attrs.get("model_resolution_selected_run_id"))
            or _decode_attr(model_resolution_selected.get("run_id"))
        )
        model_set_id = (
            _decode_attr(attrs.get("model_resolution_selected_set_id"))
            or _decode_attr(model_resolution_selected.get("set_id"))
        )
        model_path = (
            _decode_attr(attrs.get("model_path"))
            or _decode_attr(attrs.get("model_resolution_selected_model_path"))
            or _decode_attr(model_resolution_selected.get("model_path"))
        )
        model_name = _decode_attr(attrs.get("model_name"))
        if model_name is None and model_path:
            model_name = Path(model_path).name

        total_rois = _as_int(summary.get("total_rois"))
        if total_rois is None and "keypoints_roi" in keypoint_group:
            try:
                total_rois = int(keypoint_group["keypoints_roi"].shape[0])
            except Exception:
                total_rois = None
        successful_detections = _as_int(summary.get("successful_detections"))
        if successful_detections is None:
            successful_detections = _as_int(summary.get("successful_keypoint_detections"))
        failed_detections = _as_int(summary.get("failed_detections"))
        if failed_detections is None:
            failed_detections = _as_int(summary.get("failed_keypoint_detections"))
        success_rate_percent = _as_float(summary.get("success_rate_percent"))
        if success_rate_percent is None:
            success_rate_percent = _as_float(attrs.get("success_rate"))

        duration_seconds = _as_float(attrs.get("duration_seconds"))
        inference_duration_seconds = _as_float(attrs.get("inference_duration_seconds"))
        if duration_seconds is None:
            duration_seconds = inference_duration_seconds
        keypoints_processed = _as_int(attrs.get("keypoints_processed"))
        if keypoints_processed is None:
            keypoints_processed = total_rois

        inference_average_fps = _as_float(attrs.get("inference_average_fps"))
        keypoints_per_second = _as_float(attrs.get("inference_poses_per_second"))
        if keypoints_per_second is None and inference_average_fps is not None:
            keypoints_per_second = inference_average_fps
        if keypoints_per_second is None and duration_seconds and duration_seconds > 0 and keypoints_processed is not None:
            keypoints_per_second = float(keypoints_processed) / float(duration_seconds)

        conf_threshold = _as_float(parameters.get("confidence_threshold"))
        if conf_threshold is None:
            conf_threshold = _as_float(parameters.get("conf_threshold"))
        iou_threshold = _as_float(parameters.get("iou_threshold"))

        rows.append(
            {
                "keypoint_run": str(keypoint_run),
                "keypoint_created_utc": keypoint_created_utc,
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "keypoint_method": keypoint_method,
                "model_run_id": model_run_id,
                "model_set_id": model_set_id,
                "model_path": model_path,
                "model_name": model_name,
                "source_crop_run": _decode_source_run(attrs.get("source_crop_run")),
                "source_detect_run": _decode_source_run(attrs.get("source_detect_run")),
                "source_refined_run": _decode_source_run(attrs.get("source_refined_run")),
                "source_crop_storage_mode": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "source_crop_storage_mode")
                ),
                "source_crop_signature": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "source_crop_signature")
                ),
                "source_crop_revision": _as_int(
                    _lookup_attr(attrs, provenance_inputs, "source_crop_revision")
                ),
                "source_roi_image_representation": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "source_roi_image_representation")
                ),
                "source_roi_pixel_contract_name": source_roi_pixel_contract_name,
                "source_roi_pixel_contract_json": (
                    _canonical_json_text(source_roi_pixel_contract)
                    if source_roi_pixel_contract
                    else None
                ),
                "source_roi_read_mode": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "source_roi_read_mode")
                ),
                "roi_cache_policy": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "roi_cache_policy")
                ),
                "source_roi_cache_used": _as_bool_int(
                    _lookup_attr(attrs, provenance_inputs, "source_roi_cache_used")
                    if _lookup_attr(attrs, provenance_inputs, "source_roi_cache_used") is not None
                    else _lookup_attr(attrs, provenance_inputs, "roi_cache_used")
                ),
                "source_roi_cache_backend": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "source_roi_cache_backend")
                    or _lookup_attr(attrs, provenance_inputs, "roi_cache_backend")
                ),
                "source_roi_live_acceleration_effective": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "source_roi_live_acceleration_effective")
                    or _lookup_attr(attrs, provenance_inputs, "roi_live_acceleration_effective")
                ),
                "source_roi_live_gpu_chunk_frames": _as_int(
                    _lookup_attr(attrs, provenance_inputs, "source_roi_live_gpu_chunk_frames")
                    or _lookup_attr(attrs, provenance_inputs, "roi_live_gpu_chunk_frames")
                ),
                "input_mode_requested": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "input_mode_requested")
                ),
                "input_mode_effective": _decode_attr(
                    _lookup_attr(attrs, provenance_inputs, "input_mode_effective")
                ),
                "total_rois": total_rois,
                "successful_detections": successful_detections,
                "failed_detections": failed_detections,
                "success_rate_percent": success_rate_percent,
                "frames_with_keypoints": _as_int(summary.get("frames_with_keypoints")),
                "mean_confidence": _as_float(summary.get("mean_confidence")),
                "duration_seconds": duration_seconds,
                "inference_duration_seconds": inference_duration_seconds,
                "keypoints_per_second": keypoints_per_second,
                "inference_average_fps": inference_average_fps,
                "batch_size": _as_int(parameters.get("batch_size")),
                "imgsz": _decode_attr(parameters.get("imgsz")),
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "summary_statistics_json": _json_dumps(summary) if summary else None,
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc,
            }
        )

    return rows


def _extract_keypoint_profile_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    dataset_id: str,
    recording_id: Optional[str],
    zarr_use: Optional[str],
    genotype: Optional[str],
    dpf_at_acquisition: Optional[int],
) -> List[Dict[str, Any]]:
    analysis = _get_group(root, "analysis")
    if analysis is None:
        return []
    runs_parent = _get_group(analysis, "keypoint_profile_runs")
    if runs_parent is None:
        return []

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = utc_now()

    rows: List[Dict[str, Any]] = []
    for profile_run in _keypoint_run_names(runs_parent):
        run_group = _get_group(runs_parent, profile_run)
        if run_group is None:
            continue
        summary = _coerce_mapping(run_group.attrs.get("profile_summary"))
        if not summary:
            continue

        dataset_map = _coerce_mapping(summary.get("dataset")) or {}
        source_map = _coerce_mapping(summary.get("source")) or {}
        quality_map = _coerce_mapping(summary.get("quality")) or {}
        geometry_map = _coerce_mapping(summary.get("geometry")) or {}
        composition_map = _coerce_mapping(summary.get("composition")) or {}

        triangle_stats = _coerce_mapping(geometry_map.get("triangle_area")) or {}
        min_angle_stats = _coerce_mapping(geometry_map.get("min_angle")) or {}
        heading_stats = _coerce_mapping(geometry_map.get("heading")) or {}

        row_recording_id = (
            _decode_attr(run_group.attrs.get("source_recording_id"))
            or _decode_attr(dataset_map.get("recording_id"))
            or recording_id
        )
        row_zarr_use = (
            _decode_attr(run_group.attrs.get("source_zarr_use"))
            or _decode_attr(dataset_map.get("zarr_use"))
            or zarr_use
        )

        row_genotype = _decode_attr(composition_map.get("genotype")) or genotype
        row_dpf = _as_int(composition_map.get("dpf_at_acquisition"))
        if row_dpf is None:
            row_dpf = dpf_at_acquisition

        rows.append(
            {
                "dataset_id": str(dataset_id),
                "profile_run": str(profile_run),
                "recording_id": row_recording_id,
                "zarr_use": row_zarr_use,
                "keypoint_method": (
                    _decode_attr(run_group.attrs.get("source_keypoint_method"))
                    or _decode_attr(source_map.get("keypoint_method"))
                ),
                "source_keypoint_path": (
                    _decode_attr(run_group.attrs.get("source_keypoint_path"))
                    or _decode_attr(source_map.get("keypoint_path"))
                ),
                "source_keypoint_run": (
                    _decode_attr(run_group.attrs.get("source_keypoint_run"))
                    or _decode_attr(source_map.get("keypoint_run"))
                ),
                "skeleton_id": (
                    _decode_attr(run_group.attrs.get("source_skeleton_id"))
                    or _decode_attr(source_map.get("skeleton_id"))
                ),
                "kpt_shape": _normalize_kpt_shape_text(
                    run_group.attrs.get("source_kpt_shape") or source_map.get("kpt_shape")
                ),
                "pose_schema_name": (
                    _decode_attr(run_group.attrs.get("source_pose_schema_name"))
                    or _decode_attr(source_map.get("pose_schema_name"))
                    or _decode_attr(
                        (_coerce_mapping(run_group.attrs.get("source_pose_schema")) or {}).get("name")
                    )
                    or _decode_attr((_coerce_mapping(source_map.get("pose_schema")) or {}).get("name"))
                ),
                "pose_schema_json": _normalize_json_text(
                    run_group.attrs.get("source_pose_schema")
                    if run_group.attrs.get("source_pose_schema") is not None
                    else source_map.get("pose_schema")
                ),
                "heading_computation_source": (
                    _decode_attr(run_group.attrs.get("source_heading_computation_source"))
                    or _decode_attr(source_map.get("heading_computation_source"))
                ),
                "heading_computation_json": _normalize_json_text(
                    run_group.attrs.get("source_heading_computation")
                    if run_group.attrs.get("source_heading_computation") is not None
                    else source_map.get("heading_computation")
                ),
                "profile_created_utc": (
                    _decode_attr(run_group.attrs.get("created_at_utc"))
                    or _decode_attr(summary.get("created_at_utc"))
                ),
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc,
                "rows_total": _as_int(quality_map.get("rows_total")),
                "rows_usable": _as_int(
                    quality_map.get("rows_usable")
                    if quality_map.get("rows_usable") is not None
                    else quality_map.get("usable_keypoints_total")
                ),
                "usable_keypoints_total": _as_int(quality_map.get("usable_keypoints_total")),
                "usable_rate": _as_float(
                    quality_map.get("usable_rate")
                    if quality_map.get("usable_rate") is not None
                    else quality_map.get("usable_keypoints_rate_overall")
                ),
                "confidence_valid_rate": _as_float(quality_map.get("confidence_valid_rate")),
                "geometry_valid_rate": _as_float(quality_map.get("geometry_valid_rate")),
                "triangle_area_p10": _extract_stat_value(triangle_stats, "p10"),
                "triangle_area_p50": _extract_stat_value(triangle_stats, "p50"),
                "triangle_area_p90": _extract_stat_value(triangle_stats, "p90"),
                "min_angle_p10": _extract_stat_value(min_angle_stats, "p10"),
                "min_angle_p50": _extract_stat_value(min_angle_stats, "p50"),
                "min_angle_p90": _extract_stat_value(min_angle_stats, "p90"),
                "heading_p10": _extract_stat_value(heading_stats, "p10"),
                "heading_p50": _extract_stat_value(heading_stats, "p50"),
                "heading_p90": _extract_stat_value(heading_stats, "p90"),
                "rig_id": _decode_attr(composition_map.get("rig_id")),
                "camera_id": _decode_attr(composition_map.get("camera_id")),
                "arena_id": _decode_attr(composition_map.get("arena_id")),
                "dish_design": _decode_attr(composition_map.get("dish_design")),
                "canvas_name": _decode_attr(composition_map.get("canvas_name")),
                "protocol_name": _decode_attr(composition_map.get("protocol_name")),
                "genotype": row_genotype,
                "dpf_at_acquisition": row_dpf,
                "profile_json": _canonical_json_text(summary),
            }
        )

    return rows


__all__ = [
    "_extract_keypoint_performance_rows",
    "_extract_keypoint_profile_rows",
]
