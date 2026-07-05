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


def _profile_run_names(parent: zarr.Group) -> List[str]:
    try:
        names = list(parent.group_keys())
    except Exception:
        names = [name for name in parent.keys() if isinstance(name, str)]
    return sorted(str(name) for name in names)


def _profile_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _geometry_percentile(
    geometry: Mapping[str, Any],
    *,
    axis: str,
    percentile_key: str,
) -> Optional[float]:
    value = geometry.get(axis)
    if not isinstance(value, Mapping):
        return None
    return _as_float(value.get(percentile_key))


def _build_detection_data_profile_row(
    *,
    dataset_id: str,
    profile_run: str,
    summary: Mapping[str, Any],
    run_attrs: Optional[Mapping[str, Any]],
    fallback_recording_id: Optional[str],
    fallback_zarr_use: Optional[str],
    fallback_genotype: Optional[str],
    fallback_dpf_at_acquisition: Optional[int],
    zarr_mtime_ns: Optional[int],
) -> Dict[str, Any]:
    """Build one ``detection_data_profile`` row from a profile summary.

    This is the row shape historically produced by
    ``utils/sync_detection_profile_registry.py::_build_profile_payload`` and is
    kept byte-for-byte compatible so the reconcile orchestrator produces rows
    identical to the retired sync script.
    """

    dataset_map = _coerce_mapping(summary.get("dataset")) or {}
    source_map = _coerce_mapping(summary.get("source")) or {}
    coverage_map = _coerce_mapping(summary.get("coverage")) or {}
    counts_map = _coerce_mapping(summary.get("counts")) or {}
    geometry_map = _coerce_mapping(summary.get("geometry_norm")) or {}
    spatial_map = _coerce_mapping(summary.get("spatial")) or {}
    composition_map = _coerce_mapping(summary.get("composition")) or {}
    run_attrs_map = dict(run_attrs) if isinstance(run_attrs, Mapping) else {}
    detections_per_frame_map = _coerce_mapping(counts_map.get("detections_per_frame")) or {}

    dpf_at_acquisition = _as_int(composition_map.get("dpf_at_acquisition"))
    if dpf_at_acquisition is None:
        dpf_at_acquisition = fallback_dpf_at_acquisition

    return {
        "dataset_id": str(dataset_id),
        "profile_run": str(profile_run),
        "recording_id": (
            _decode_attr(run_attrs_map.get("source_recording_id"))
            or _decode_attr(dataset_map.get("recording_id"))
            or fallback_recording_id
        ),
        "zarr_use": (
            _decode_attr(run_attrs_map.get("source_zarr_use"))
            or _decode_attr(dataset_map.get("zarr_use"))
            or fallback_zarr_use
        ),
        "detection_type": (
            _decode_attr(run_attrs_map.get("source_detection_type"))
            or _decode_attr(source_map.get("detection_type"))
        ),
        "detection_path": (
            _decode_attr(run_attrs_map.get("source_detection_path"))
            or _decode_attr(source_map.get("detection_path"))
        ),
        "profile_created_utc": (
            _decode_attr(run_attrs_map.get("created_at_utc"))
            or _decode_attr(summary.get("created_at_utc"))
        ),
        "frames_total": _as_int(coverage_map.get("frames_total")),
        "frames_with_detections": _as_int(coverage_map.get("frames_with_detections")),
        "coverage_percent": _as_float(coverage_map.get("coverage_percent")),
        "detections_total": _as_int(counts_map.get("detections_total")),
        "detections_per_frame_p50": _as_float(detections_per_frame_map.get("p50")),
        "detections_per_frame_p90": _as_float(detections_per_frame_map.get("p90")),
        "w_p10": _geometry_percentile(geometry_map, axis="w", percentile_key="p10"),
        "w_p50": _geometry_percentile(geometry_map, axis="w", percentile_key="p50"),
        "w_p90": _geometry_percentile(geometry_map, axis="w", percentile_key="p90"),
        "h_p10": _geometry_percentile(geometry_map, axis="h", percentile_key="p10"),
        "h_p50": _geometry_percentile(geometry_map, axis="h", percentile_key="p50"),
        "h_p90": _geometry_percentile(geometry_map, axis="h", percentile_key="p90"),
        "area_p10": _geometry_percentile(geometry_map, axis="area", percentile_key="p10"),
        "area_p50": _geometry_percentile(geometry_map, axis="area", percentile_key="p50"),
        "area_p90": _geometry_percentile(geometry_map, axis="area", percentile_key="p90"),
        "aspect_ratio_p10": _geometry_percentile(geometry_map, axis="aspect_ratio", percentile_key="p10"),
        "aspect_ratio_p50": _geometry_percentile(geometry_map, axis="aspect_ratio", percentile_key="p50"),
        "aspect_ratio_p90": _geometry_percentile(geometry_map, axis="aspect_ratio", percentile_key="p90"),
        "edge_proximity_rate": _as_float(spatial_map.get("edge_proximity_rate")),
        "rig_id": _decode_attr(composition_map.get("rig_id")),
        "camera_id": _decode_attr(composition_map.get("camera_id")),
        "arena_id": _decode_attr(composition_map.get("arena_id")),
        "dish_design": _decode_attr(composition_map.get("dish_design")),
        "canvas_name": _decode_attr(composition_map.get("canvas_name")),
        "protocol_name": _decode_attr(composition_map.get("protocol_name")),
        "genotype": _decode_attr(composition_map.get("genotype")) or fallback_genotype,
        "dpf_at_acquisition": dpf_at_acquisition,
        "profile_json": _profile_json_text(summary),
        "zarr_mtime_ns": zarr_mtime_ns,
    }


def _extract_detection_data_profile_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    dataset_id: str,
    recording_id: Optional[str],
    zarr_use: Optional[str],
    genotype: Optional[str],
    dpf_at_acquisition: Optional[int],
) -> List[Dict[str, Any]]:
    """Extract ``detection_data_profile`` rows from ``analysis/detection_profile_runs``.

    One row per profile run (mirrors ``_extract_keypoint_profile_rows``). Rows are
    inserted via ``Registry.replace_detection_data_profile`` under the reconcile
    orchestrator, replacing the standalone ``sync_detection_profile_registry.py``.
    """

    analysis = _get_group(root, "analysis")
    if analysis is None:
        return []
    runs_parent = _get_group(analysis, "detection_profile_runs")
    if runs_parent is None:
        return []

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = utc_now()

    rows: List[Dict[str, Any]] = []
    for profile_run in _profile_run_names(runs_parent):
        run_group = _get_group(runs_parent, profile_run)
        if run_group is None:
            continue
        summary = _coerce_mapping(run_group.attrs.get("profile_summary"))
        if not summary:
            continue
        row = _build_detection_data_profile_row(
            dataset_id=dataset_id,
            profile_run=profile_run,
            summary=summary,
            run_attrs=dict(run_group.attrs),
            fallback_recording_id=recording_id,
            fallback_zarr_use=zarr_use,
            fallback_genotype=genotype,
            fallback_dpf_at_acquisition=dpf_at_acquisition,
            zarr_mtime_ns=zarr_mtime_ns,
        )
        row["updated_utc"] = updated_utc
        rows.append(row)

    return rows


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


__all__ = [
    "_extract_detect_performance_rows",
    "_extract_detection_data_profile_rows",
    "_build_detection_data_profile_row",
]
