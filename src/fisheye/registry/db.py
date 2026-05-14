"""SQLite-backed registry for datasets, provenance, and training runs."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _shared_decode_attr
from .stage_catalog import recording_status_stage_ids, recording_tuning_stage_ids


def _require_sql_identifier(value: str) -> str:
    if not value or not all(ch.isalnum() or ch == "_" for ch in value):
        raise ValueError(f"Unsafe SQL identifier fragment: {value!r}")
    return value


def _recording_step_status_pivot_columns() -> str:
    lines = []
    for step_name in recording_status_stage_ids():
        step = _require_sql_identifier(step_name)
        lines.append(
            f"MAX(CASE WHEN step_name = '{step}' THEN status END) AS {step}_status"
        )
    return ",\n                    ".join(lines)


def _recording_tuning_ok_count_sql(alias: str) -> str:
    table_alias = _require_sql_identifier(alias)
    terms = []
    for step_name in recording_tuning_stage_ids():
        step = _require_sql_identifier(step_name)
        terms.append(f"CASE WHEN {table_alias}.{step}_status = 'ok' THEN 1 ELSE 0 END")
    return "\n                        + ".join(terms) if terms else "0"


def _recording_step_status_display_sql(status_expr: str, details_expr: str) -> str:
    return f"""
                    CASE
                        WHEN {status_expr} = 'ok' THEN 'OK'
                        WHEN {status_expr} = 'na' THEN 'N/A'
                        WHEN {status_expr} = 'error' THEN 'ERR'
                        WHEN json_extract({details_expr}, '$.source_freshness_state') = 'stale' THEN 'STALE'
                        WHEN json_extract({details_expr}, '$.source_freshness_state') IN (
                            'missing_source_attrs',
                            'upstream_source_unavailable'
                        ) THEN 'UNVER'
                        ELSE 'MISS'
                    END
    """.strip()


@dataclass(frozen=True)
class RegistryPaths:
    path: Path

    @staticmethod
    def from_env(default_root: Path) -> "RegistryPaths":
        env_path = os.environ.get("PALETTE_REGISTRY_PATH")
        if env_path:
            return RegistryPaths(path=Path(env_path))
        config_path = _load_registry_path(default_root)
        if config_path:
            return RegistryPaths(path=config_path)
        return RegistryPaths(path=default_root / "runs" / "registry" / "palette_registry.sqlite")


def _load_registry_path(default_root: Path) -> Optional[Path]:
    config_path = default_root / "configs" / "fisheye" / "registry.yaml"
    if not config_path.exists():
        return None
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    path_value = None
    if "registry_path" in data:
        path_value = data.get("registry_path")
    elif isinstance(data.get("registry"), dict):
        path_value = data["registry"].get("path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _import_zarr():
    """Lazy import to keep SQL-only registry commands independent of zarr."""
    try:
        import zarr  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise ModuleNotFoundError(
            "zarr is required for scan/register operations. Install zarr to read Zarr archives."
        ) from exc
    return zarr


def _open_zarr_group_non_consolidated(zarr_path: Path, *, mode: str = "r"):
    """Open a zarr root without trusting possibly stale consolidated metadata."""
    zarr = _import_zarr()
    try:
        return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)
    except TypeError:
        try:
            return zarr.open_group(str(zarr_path), mode=mode, consolidated=False)
        except TypeError:
            return zarr.open_group(str(zarr_path), mode=mode)


_utc_now = utc_now


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_loads(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


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


def _as_bool_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(bool(value))
    decoded = _shared_decode_attr(value)
    if decoded is None:
        return None
    if isinstance(decoded, (bool, np.bool_)):
        return int(decoded)
    text = str(decoded)
    norm = text.strip().lower()
    if norm in {"1", "true", "yes", "y", "available"}:
        return 1
    if norm in {"0", "false", "no", "n", "absent", "none", "missing"}:
        return 0
    return None


_decode_attr = _shared_decode_attr


def _normalize_task_type(value: Any) -> Optional[str]:
    text = _decode_attr(value)
    if not text:
        return None
    norm = text.lower()
    alias = {
        "detect": "detect",
        "detection": "detect",
        "pose": "pose",
        "keypoint": "pose",
        "keypoints": "pose",
        "eye_masks": "eye_masks",
        "eyemasks": "eye_masks",
        "subject_masks": "subject_masks",
        "subjectmasks": "subject_masks",
        "segmentation": "subject_masks",
    }
    return alias.get(norm)


def _infer_task_type_from_text(value: Any) -> Optional[str]:
    text = _decode_attr(value)
    if not text:
        return None
    norm = text.lower()
    if norm.startswith("detect_") or "_detect_" in norm or "/detect/" in norm:
        return "detect"
    if norm.startswith("pose_") or "_pose_" in norm or "/pose/" in norm:
        return "pose"
    if norm.startswith("keypoint_") or norm.startswith("keypoints_") or "keypoint" in norm:
        return "pose"
    if norm.startswith("eye_mask_") or "eye_mask" in norm or "eyemask" in norm:
        return "eye_masks"
    if norm.startswith("subject_mask_") or "subject_mask" in norm or "subjectmask" in norm:
        return "subject_masks"
    return None


def _infer_task_type(
    *,
    explicit: Any = None,
    set_id: Any = None,
    run_id: Any = None,
    config_path: Any = None,
    manifest_path: Any = None,
    model_path: Any = None,
    invocation: Optional[Mapping[str, Any]] = None,
    query_filter: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    direct = _normalize_task_type(explicit)
    if direct:
        return direct

    for candidate in (set_id, run_id, config_path, manifest_path, model_path):
        inferred = _infer_task_type_from_text(candidate)
        if inferred:
            return inferred

    for payload in (invocation, query_filter):
        if not isinstance(payload, Mapping):
            continue
        for key in ("task_type", "task"):
            inferred = _normalize_task_type(payload.get(key))
            if inferred:
                return inferred
        args_payload = payload.get("args")
        if isinstance(args_payload, Mapping):
            for key in ("task_type", "task"):
                inferred = _normalize_task_type(args_payload.get(key))
                if inferred:
                    return inferred

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


def _coerce_text_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return None
    items = [str(item) for item in value if item is not None]
    return items


def _subject_mask_component_groups_from_labels(labels: Sequence[str]) -> List[str]:
    groups: List[str] = []
    for label in labels:
        if label == "subject_body":
            group = "body"
        elif label in {"eyes_union", "eye_left", "eye_right"}:
            group = "eyes"
        elif label == "swim_bladder":
            group = "swim_bladder"
        else:
            continue
        if group not in groups:
            groups.append(group)
    preferred_order = {"body": 0, "eyes": 1, "swim_bladder": 2}
    return sorted(groups, key=lambda item: preferred_order.get(item, len(preferred_order)))


def _subject_mask_coverage_class_from_groups(groups: Sequence[str]) -> Optional[str]:
    group_set = set(groups)
    if not group_set:
        return None
    if group_set == {"eyes"}:
        return "eyes_only"
    if {"body", "eyes", "swim_bladder"}.issubset(group_set):
        return "dense_all_components"
    return "partial_subject_masks"


def _training_model_discovery_metadata(
    *,
    task_type: Optional[str],
    final_metrics: Optional[Mapping[str, Any]],
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build stable model-discovery metadata for training_models rows."""

    payload: Dict[str, Any] = dict(metadata or {})
    payload.setdefault("source", "training_runs")

    final = dict(final_metrics or {})
    summary = _coerce_mapping(final.get("subject_mask_model_summary")) or {}
    resolved_task_type = (
        _normalize_task_type(payload.get("task_type"))
        or _normalize_task_type(task_type)
        or ("subject_masks" if summary else None)
    )
    if resolved_task_type:
        payload["task_type"] = resolved_task_type

    for key in ("best_val_dice", "best_epoch", "epochs", "train_samples", "val_samples"):
        if key in final and final[key] is not None:
            payload[key] = final[key]

    if resolved_task_type == "subject_masks" or summary:
        source = summary or final
        for key in (
            "label_schema_id",
            "mask_labels",
            "coverage_class",
            "component_groups",
            "component_coverage_key",
            "contains_only_eye_masks",
            "available_labels",
            "missing_labels",
            "supervised_row_counts",
            "positive_row_counts",
            "negative_row_counts",
            "unsupervised_row_counts",
            "source_artifact_count",
            "summarized_artifact_count",
        ):
            value = source.get(key)
            if value is not None:
                payload[key] = value
        if summary:
            payload["subject_mask_model_summary"] = dict(summary)
        labels = _coerce_text_list(payload.get("mask_labels")) or []
        if labels:
            payload.setdefault("available_labels", labels)
            payload.setdefault("missing_labels", [])
            groups = _coerce_text_list(payload.get("component_groups"))
            if not groups:
                groups = _subject_mask_component_groups_from_labels(labels)
                if groups:
                    payload["component_groups"] = groups
            if groups:
                payload.setdefault("component_coverage_key", "+".join(groups))
                coverage_class = _subject_mask_coverage_class_from_groups(groups)
                if coverage_class is not None:
                    payload.setdefault("coverage_class", coverage_class)

    return payload


def _training_model_discovery_index_fields(
    *,
    task_type: Optional[str],
    final_metrics: Optional[Mapping[str, Any]],
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = _training_model_discovery_metadata(
        task_type=task_type,
        final_metrics=final_metrics,
        metadata=metadata,
    )
    resolved_task_type = _normalize_task_type(payload.get("task_type")) or _normalize_task_type(task_type)
    mask_labels = _coerce_text_list(payload.get("mask_labels"))
    component_groups = _coerce_text_list(payload.get("component_groups"))
    best_val_dice = _as_float(payload.get("best_val_dice"))
    return {
        "task_type": resolved_task_type,
        "label_schema_id": _decode_attr(payload.get("label_schema_id")),
        "coverage_class": _decode_attr(payload.get("coverage_class")),
        "component_coverage_key": _decode_attr(payload.get("component_coverage_key")),
        "mask_labels_json": _json_dumps(mask_labels),
        "component_groups_json": _json_dumps(component_groups),
        "best_metric_name": "best_val_dice" if best_val_dice is not None else None,
        "best_metric_value": best_val_dice,
        "best_epoch": _as_int(payload.get("best_epoch")),
    }


def _open_child_group(parent: Any, key: str) -> Any:
    store = getattr(parent, "store", None)
    if store is None:
        return None
    parent_path = _decode_attr(getattr(parent, "path", None))
    child_path = f"{parent_path}/{key}" if parent_path else key
    zarr = _import_zarr()
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
    quality_updated_utc = _utc_now()

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
    quality_updated_utc = _utc_now()

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
    updated_utc = _utc_now()

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
    updated_utc = _utc_now()

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


def _keypoint_run_names(parent: zarr.Group) -> List[str]:
    try:
        names = list(parent.group_keys())
    except Exception:
        names = [name for name in parent.keys() if isinstance(name, str)]
    return sorted(str(name) for name in names)


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
    updated_utc = _utc_now()

    rows: List[Dict[str, Any]] = []
    for keypoint_run in _keypoint_run_names(keypoints_parent):
        if keypoint_run not in keypoints_parent:
            continue
        keypoint_group = keypoints_parent[keypoint_run]
        attrs = dict(keypoint_group.attrs)
        summary = _coerce_mapping(attrs.get("summary_statistics")) or {}
        parameters = _coerce_mapping(attrs.get("parameters")) or {}
        provenance = _coerce_mapping(attrs.get("provenance")) or {}
        model_resolution = _coerce_mapping(provenance.get("model_resolution")) or {}
        model_resolution_selected = _coerce_mapping(model_resolution.get("selected")) or {}

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
                "source_crop_run": _decode_attr(attrs.get("source_crop_run")),
                "source_detect_run": _decode_attr(attrs.get("source_detect_run")),
                "source_refined_run": _decode_attr(attrs.get("source_refined_run")),
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
    updated_utc = _utc_now()

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


def _eye_mask_run_names(parent: zarr.Group) -> List[str]:
    try:
        names = list(parent.group_keys())
    except Exception:
        names = [name for name in parent.keys() if isinstance(name, str)]
    return sorted(str(name) for name in names)


def _resolve_source_keypoints_run(attrs: Mapping[str, Any]) -> Optional[str]:
    value = _decode_attr(attrs.get("source_keypoints_run"))
    if value:
        return value
    return _decode_attr(attrs.get("source_keypoint_run"))


def _coerce_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray, str)):
        text = _decode_attr(value)
        return [text] if text else []
    if isinstance(value, Sequence):
        values: List[str] = []
        for item in value:
            text = _decode_attr(item)
            if text:
                values.append(text)
        return values
    return []


def _coerce_bool_list(value: Any) -> List[bool]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        try:
            value = value.tolist()
        except Exception:
            return []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [bool(item) for item in value]
    return []


def _extract_review_fields(
    review_status: Optional[Mapping[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not review_status:
        return None, None, None, None, None
    return (
        _decode_attr(review_status.get("state")) or _decode_attr(review_status.get("review_state")),
        _decode_attr(review_status.get("method")) or _decode_attr(review_status.get("review_method")),
        _decode_attr(review_status.get("intended_use")) or _decode_attr(review_status.get("review_intended_use")),
        _decode_attr(review_status.get("reviewer")) or _decode_attr(review_status.get("review_reviewer")),
        _decode_attr(review_status.get("timestamp_utc"))
        or _decode_attr(review_status.get("timestamp"))
        or _decode_attr(review_status.get("review_timestamp_utc"))
        or _decode_attr(review_status.get("reviewed_at_utc"))
        or _decode_attr(review_status.get("reviewed_at")),
    )


def _derive_review_lifecycle(
    *,
    review_state: Optional[str],
    stale_state: Optional[str] = None,
    stale_reason: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    stale_state_norm = str(stale_state).strip().lower() if stale_state else None
    if stale_state_norm == "stale":
        return "stale", stale_reason or "source_subject_mask_stale"
    review_state_norm = str(review_state).strip().lower() if review_state else None
    if review_state_norm in {"pending", "needs_review", "review"}:
        return "in_progress", review_state
    if review_state_norm in {"approved", "rejected"}:
        return review_state_norm, review_state
    return None, None


def _subject_component_family(component_name: Optional[str]) -> Optional[str]:
    name = _decode_attr(component_name)
    if not name:
        return None
    if name in {"eye_left", "eye_right", "eyes_union"}:
        return "eyes"
    return name


def _extract_subject_mask_total_rois(run_group: zarr.Group, attrs: Mapping[str, Any]) -> Optional[int]:
    total_rois = _as_int(attrs.get("total_rois"))
    if total_rois is not None:
        return total_rois
    for key in ("masks_roi", "frame_indices", "detection_indices"):
        arr = _get_group(run_group, key)
        shape = getattr(arr, "shape", None)
        if shape:
            try:
                return int(shape[0])
            except Exception:
                pass
    frame_counts = _get_group(run_group, "frame_counts")
    if frame_counts is not None:
        try:
            values = np.asarray(frame_counts[:])
            return int(values.sum())
        except Exception:
            return None
    return None


def _extract_subject_mask_presence(
    run_group: zarr.Group,
    *,
    mask_labels: Sequence[str],
    total_rois: Optional[int],
) -> Tuple[Optional[int], Dict[str, int]]:
    counts_by_label = {str(label): 0 for label in mask_labels}
    any_present_rows: Optional[int] = None

    metrics = _get_group(run_group, "metrics")
    mask_present = _get_group(metrics, "mask_present") if metrics is not None else None
    if mask_present is not None:
        try:
            values = np.asarray(mask_present[:], dtype=bool)
            if values.ndim == 2 and values.shape[1] >= len(mask_labels):
                any_present_rows = int(values[:, : len(mask_labels)].any(axis=1).sum())
                for index, label in enumerate(mask_labels):
                    counts_by_label[str(label)] = int(values[:, index].sum())
                return any_present_rows, counts_by_label
        except Exception:
            pass

    masks_roi = _get_group(run_group, "masks_roi")
    if masks_roi is not None:
        try:
            values = np.asarray(masks_roi[:])
            if values.ndim == 4 and values.shape[1] >= len(mask_labels):
                present = values[:, : len(mask_labels), :, :].reshape(values.shape[0], len(mask_labels), -1).any(axis=2)
                any_present_rows = int(present.any(axis=1).sum())
                for index, label in enumerate(mask_labels):
                    counts_by_label[str(label)] = int(present[:, index].sum())
                return any_present_rows, counts_by_label
        except Exception:
            pass

    if total_rois is not None and total_rois >= 0:
        return 0, counts_by_label
    return None, counts_by_label


def _extract_subject_mask_performance_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = _utc_now()
    latest_subject_mask_run = _resolve_latest_group_name(root.get("subject_mask_runs"))

    rows: List[Dict[str, Any]] = []
    for stage_group in ("subject_mask_runs", "refined_subject_masks_runs"):
        parent = root.get(stage_group)
        if parent is None:
            continue
        for run_name in _eye_mask_run_names(parent):
            if run_name not in parent:
                continue
            run_group = parent[run_name]
            attrs = dict(run_group.attrs)
            provenance = _coerce_mapping(attrs.get("provenance")) or {}
            summary_statistics = _coerce_mapping(attrs.get("summary_statistics"))
            reason_counts = _coerce_mapping(attrs.get("reason_counts"))
            review_status = _coerce_mapping(
                attrs.get("refined_subject_mask_review_status")
                if stage_group == "refined_subject_masks_runs"
                else attrs.get("subject_mask_review_status")
            ) or _coerce_mapping(attrs.get("subject_mask_review_status"))
            component_review_statuses = _coerce_mapping(attrs.get("component_review_statuses")) or {}

            provenance_parameters = _coerce_mapping(provenance.get("parameters")) or {}
            provenance_inputs = _coerce_mapping(provenance.get("inputs")) or {}
            run_created_utc = (
                _decode_attr(attrs.get("created_at_utc"))
                or _decode_attr(attrs.get("created_utc"))
                or _decode_attr(attrs.get("timestamp_utc"))
                or _decode_attr(provenance.get("created_at_utc"))
            )
            method = (
                _decode_attr(attrs.get("method"))
                or _decode_attr(provenance_parameters.get("method"))
                or _decode_attr(provenance.get("method"))
            )
            if not method and stage_group == "refined_subject_masks_runs":
                method = "refine_subject_masks"

            review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc = (
                _extract_review_fields(review_status)
            )

            source_subject_mask_run = _decode_attr(attrs.get("source_subject_mask_run")) or _decode_attr(
                provenance_inputs.get("source_subject_mask_run")
            )
            source_subject_mask_method = _decode_attr(attrs.get("source_subject_mask_method")) or _decode_attr(
                provenance_inputs.get("source_subject_mask_method")
            )
            source_subject_mask_stale = _coerce_mapping(attrs.get("source_subject_mask_stale"))
            source_subject_mask_stale_state = (
                _decode_attr(source_subject_mask_stale.get("state")) if source_subject_mask_stale else None
            )
            source_subject_mask_stale_reason = (
                _decode_attr(source_subject_mask_stale.get("reason")) if source_subject_mask_stale else None
            )
            source_subject_mask_stale_timestamp_utc = (
                _decode_attr(source_subject_mask_stale.get("timestamp_utc"))
                or _decode_attr(source_subject_mask_stale.get("timestamp"))
                or _decode_attr(source_subject_mask_stale.get("stale_at_utc"))
                or _decode_attr(source_subject_mask_stale.get("stale_at"))
                if source_subject_mask_stale
                else None
            )
            run_semantics = _decode_attr(attrs.get("run_semantics")) or _decode_attr(
                provenance_parameters.get("run_semantics")
            )
            probability_semantics = _decode_attr(attrs.get("probability_semantics")) or _decode_attr(
                provenance_parameters.get("probability_semantics")
            )
            tuning_source = _decode_attr(attrs.get("tuning_source")) or _decode_attr(
                provenance_parameters.get("tuning_source")
            )
            tuning_timestamp = _decode_attr(attrs.get("tuning_timestamp")) or _decode_attr(
                provenance_parameters.get("tuning_timestamp")
            )
            source_background_run = _decode_attr(attrs.get("source_background_run")) or _decode_attr(
                provenance_inputs.get("source_background_run")
            )
            source_background_array = _decode_attr(attrs.get("source_background_array")) or _decode_attr(
                provenance_inputs.get("source_background_array")
            )
            source_dish_mask_array = _decode_attr(attrs.get("source_dish_mask_array")) or _decode_attr(
                provenance_inputs.get("source_dish_mask_array")
            )
            if (
                source_subject_mask_stale_state is None
                and stage_group == "refined_subject_masks_runs"
                and latest_subject_mask_run
                and source_subject_mask_run
                and source_subject_mask_run != latest_subject_mask_run
            ):
                source_subject_mask_stale_state = "stale"
                source_subject_mask_stale_reason = "latest_subject_mask_run_mismatch"

            lifecycle_state, lifecycle_reason = _derive_review_lifecycle(
                review_state=review_state,
                stale_state=source_subject_mask_stale_state,
                stale_reason=source_subject_mask_stale_reason,
            )

            mask_labels = _coerce_text_list(attrs.get("mask_labels"))
            available_flags: List[bool] = []
            available_channels = _get_group(run_group, "available_channels")
            if available_channels is not None:
                try:
                    available_flags = _coerce_bool_list(available_channels[:])
                except Exception:
                    available_flags = []
            if mask_labels:
                if not available_flags:
                    available_flags = [True] * len(mask_labels)
                if len(available_flags) < len(mask_labels):
                    available_flags.extend([False] * (len(mask_labels) - len(available_flags)))
                available_flags = available_flags[: len(mask_labels)]
            available_components = [label for label, flag in zip(mask_labels, available_flags) if flag]
            unavailable_components = [label for label, flag in zip(mask_labels, available_flags) if not flag]
            component_review_states = {
                str(label): state
                for label, payload in component_review_statuses.items()
                if (state := (_decode_attr((_coerce_mapping(payload) or {}).get("state")) or _decode_attr((_coerce_mapping(payload) or {}).get("review_state"))))
            }

            eye_component_mode: Optional[str] = None
            if "eye_left" in mask_labels or "eye_right" in mask_labels:
                eye_component_mode = "lr"
            elif "eyes_union" in mask_labels:
                eye_component_mode = "union"

            total_rois = _extract_subject_mask_total_rois(run_group, attrs)
            rows_with_any_mask, _ = _extract_subject_mask_presence(
                run_group,
                mask_labels=mask_labels,
                total_rois=total_rois,
            )
            coverage_percent = None
            if rows_with_any_mask is not None and total_rois is not None:
                ratio = _format_ratio(rows_with_any_mask, total_rois)
                coverage_percent = float(ratio) * 100.0 if ratio is not None else None

            duration_seconds = _as_float(attrs.get("duration_seconds"))
            rois_per_second = None
            if duration_seconds is not None and duration_seconds > 0 and total_rois is not None:
                rois_per_second = float(total_rois) / float(duration_seconds)

            rows.append(
                {
                    "stage_group": stage_group,
                    "run_name": str(run_name),
                    "run_created_utc": run_created_utc,
                    "recording_id": recording_id,
                    "zarr_use": zarr_use,
                    "subject_mask_method": method,
                    "label_schema_id": _decode_attr(attrs.get("label_schema_id")),
                    "source_crop_run": _decode_attr(attrs.get("source_crop_run")) or _decode_attr(
                        provenance_inputs.get("source_crop_run")
                    ),
                    "source_keypoint_group": _decode_attr(attrs.get("source_keypoint_group")) or _decode_attr(
                        provenance_inputs.get("source_keypoint_group")
                    ),
                    "source_keypoints_run": _resolve_source_keypoints_run(attrs)
                    or _decode_attr(provenance_inputs.get("source_keypoints_run"))
                    or _decode_attr(provenance_inputs.get("source_keypoint_run")),
                    "source_subject_mask_run": source_subject_mask_run,
                    "source_subject_mask_method": source_subject_mask_method,
                    "run_semantics": run_semantics,
                    "probability_semantics": probability_semantics,
                    "source_background_run": source_background_run,
                    "source_background_array": source_background_array,
                    "source_dish_mask_array": source_dish_mask_array,
                    "tuning_source": tuning_source,
                    "tuning_timestamp": tuning_timestamp,
                    "total_rois": total_rois,
                    "rows_with_any_mask": rows_with_any_mask,
                    "coverage_percent": coverage_percent,
                    "duration_seconds": duration_seconds,
                    "rois_per_second": rois_per_second,
                    "available_component_count": len(available_components),
                    "available_components_json": _json_dumps(available_components),
                    "unavailable_components_json": _json_dumps(unavailable_components),
                    "component_review_states_json": _json_dumps(component_review_states),
                    "eye_component_mode": eye_component_mode,
                    "reason_counts_json": _json_dumps(reason_counts),
                    "summary_statistics_json": _json_dumps(summary_statistics),
                    "review_state": review_state,
                    "review_method": review_method,
                    "review_intended_use": review_intended_use,
                    "review_reviewer": review_reviewer,
                    "review_timestamp_utc": review_timestamp_utc,
                    "source_subject_mask_stale_state": source_subject_mask_stale_state,
                    "source_subject_mask_stale_reason": source_subject_mask_stale_reason,
                    "source_subject_mask_stale_timestamp_utc": source_subject_mask_stale_timestamp_utc,
                    "source_subject_mask_stale_json": _json_dumps(source_subject_mask_stale),
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_reason": lifecycle_reason,
                    "zarr_mtime_ns": zarr_mtime_ns,
                    "updated_utc": updated_utc,
                }
            )

    return rows


def _extract_subject_mask_component_quality_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    performance_rows = _extract_subject_mask_performance_rows(
        root,
        zarr_path=zarr_path,
        recording_id=recording_id,
        zarr_use=zarr_use,
    )
    rows: List[Dict[str, Any]] = []
    for performance in performance_rows:
        stage_group = str(performance.get("stage_group") or "")
        run_name = str(performance.get("run_name") or "")
        parent = root.get(stage_group)
        if parent is None or run_name not in parent:
            continue
        run_group = parent[run_name]
        attrs = dict(run_group.attrs)
        mask_labels = _coerce_text_list(attrs.get("mask_labels"))
        if not mask_labels:
            continue

        available_flags: List[bool] = []
        available_channels = _get_group(run_group, "available_channels")
        if available_channels is not None:
            try:
                available_flags = _coerce_bool_list(available_channels[:])
            except Exception:
                available_flags = []
        if not available_flags:
            available_flags = [True] * len(mask_labels)
        if len(available_flags) < len(mask_labels):
            available_flags.extend([False] * (len(mask_labels) - len(available_flags)))
        available_flags = available_flags[: len(mask_labels)]

        total_rois = _as_int(performance.get("total_rois"))
        _, counts_by_label = _extract_subject_mask_presence(
            run_group,
            mask_labels=mask_labels,
            total_rois=total_rois,
        )
        component_review_statuses = _coerce_mapping(attrs.get("component_review_statuses")) or {}
        stale_state = _decode_attr(performance.get("source_subject_mask_stale_state"))
        stale_reason = _decode_attr(performance.get("source_subject_mask_stale_reason"))

        for index, component_name in enumerate(mask_labels):
            available = bool(available_flags[index]) if index < len(available_flags) else False
            review_payload = _coerce_mapping(component_review_statuses.get(component_name))
            review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc = (
                _extract_review_fields(review_payload)
            )
            rows_with_component_mask = counts_by_label.get(component_name)
            rows_with_component_mask_rate = _format_ratio(rows_with_component_mask, total_rois)
            lifecycle_state: Optional[str]
            lifecycle_reason: Optional[str]
            if not available:
                lifecycle_state, lifecycle_reason = "na", "component_unavailable"
            else:
                lifecycle_state, lifecycle_reason = _derive_review_lifecycle(
                    review_state=review_state,
                    stale_state=stale_state,
                    stale_reason=stale_reason,
                )

            rows.append(
                {
                    "stage_group": stage_group,
                    "run_name": run_name,
                    "component_name": component_name,
                    "component_family": _subject_component_family(component_name),
                    "run_created_utc": performance.get("run_created_utc"),
                    "recording_id": performance.get("recording_id"),
                    "zarr_use": performance.get("zarr_use"),
                    "subject_mask_method": performance.get("subject_mask_method"),
                    "label_schema_id": performance.get("label_schema_id"),
                    "eye_component_mode": performance.get("eye_component_mode"),
                    "source_subject_mask_run": performance.get("source_subject_mask_run"),
                    "available": int(available),
                    "review_state": review_state,
                    "review_method": review_method,
                    "review_intended_use": review_intended_use,
                    "review_reviewer": review_reviewer,
                    "review_timestamp_utc": review_timestamp_utc,
                    "total_rois": total_rois,
                    "rows_with_component_mask": rows_with_component_mask,
                    "rows_with_component_mask_rate": rows_with_component_mask_rate,
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_reason": lifecycle_reason,
                    "quality_updated_utc": performance.get("updated_utc") or _utc_now(),
                    "zarr_mtime_ns": performance.get("zarr_mtime_ns"),
                }
            )
    return rows


def _extract_eye_mask_performance_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = _utc_now()

    rows: List[Dict[str, Any]] = []
    for stage_group in ("eye_masks_runs", "refined_eye_masks_runs"):
        parent = root.get(stage_group)
        if parent is None:
            continue
        for run_name in _eye_mask_run_names(parent):
            if run_name not in parent:
                continue
            run_group = parent[run_name]
            attrs = dict(run_group.attrs)
            provenance = _coerce_mapping(attrs.get("provenance")) or {}
            summary_statistics = _coerce_mapping(attrs.get("summary_statistics"))
            reason_counts = _coerce_mapping(attrs.get("reason_counts"))
            review_status = _coerce_mapping(attrs.get("eye_mask_review_status"))
            source_keypoint_stale = _coerce_mapping(attrs.get("source_keypoint_stale"))

            run_created_utc = (
                _decode_attr(attrs.get("created_at_utc"))
                or _decode_attr(attrs.get("created_utc"))
                or _decode_attr(attrs.get("timestamp_utc"))
                or _decode_attr(provenance.get("created_at_utc"))
            )
            method = _decode_attr(attrs.get("method")) or _decode_attr(provenance.get("method"))
            review_state = _decode_attr(review_status.get("state")) if review_status else None
            review_method = _decode_attr(review_status.get("method")) if review_status else None
            review_intended_use = _decode_attr(review_status.get("intended_use")) if review_status else None
            review_reviewer = _decode_attr(review_status.get("reviewer")) if review_status else None
            review_timestamp_utc = (
                _decode_attr(review_status.get("timestamp_utc"))
                or _decode_attr(review_status.get("timestamp"))
                or _decode_attr(review_status.get("reviewed_at_utc"))
                or _decode_attr(review_status.get("reviewed_at"))
                if review_status
                else None
            )
            source_keypoint_stale_state = (
                _decode_attr(source_keypoint_stale.get("state")) if source_keypoint_stale else None
            )
            source_keypoint_stale_reason = (
                _decode_attr(source_keypoint_stale.get("reason")) if source_keypoint_stale else None
            )
            source_keypoint_stale_timestamp_utc = (
                _decode_attr(source_keypoint_stale.get("timestamp_utc"))
                or _decode_attr(source_keypoint_stale.get("timestamp"))
                or _decode_attr(source_keypoint_stale.get("stale_at_utc"))
                or _decode_attr(source_keypoint_stale.get("stale_at"))
                if source_keypoint_stale
                else None
            )
            review_state_norm = str(review_state).strip().lower() if review_state else None
            source_stale_state_norm = (
                str(source_keypoint_stale_state).strip().lower() if source_keypoint_stale_state else None
            )
            lifecycle_state: Optional[str] = None
            lifecycle_reason: Optional[str] = None
            if source_stale_state_norm == "stale":
                lifecycle_state = "stale"
                lifecycle_reason = source_keypoint_stale_reason or "source_keypoint_stale"
            elif review_state_norm in {"pending", "needs_review", "review"}:
                lifecycle_state = "in_progress"
                lifecycle_reason = review_state
            elif review_state_norm in {"approved", "rejected"}:
                lifecycle_state = review_state_norm
                lifecycle_reason = review_state

            total_rois = _as_int(attrs.get("total_rois"))
            successful_eyes = _as_int(attrs.get("successful_eyes"))
            successful_roi_pairs = _as_int(attrs.get("successful_roi_pairs"))
            successful_roi_pair_rate = _as_float(attrs.get("successful_roi_pair_rate"))
            if successful_roi_pair_rate is None:
                successful_roi_pair_rate = _format_ratio(successful_roi_pairs, total_rois)
            duration_seconds = _as_float(attrs.get("duration_seconds"))
            rois_per_second = None
            if duration_seconds is not None and duration_seconds > 0 and total_rois is not None:
                rois_per_second = float(total_rois) / float(duration_seconds)
            inference_duration_seconds = _as_float(attrs.get("inference_duration_seconds"))
            inference_average_fps = _as_float(attrs.get("inference_average_fps"))
            if inference_average_fps is None and rois_per_second is not None:
                inference_average_fps = rois_per_second

            rows.append(
                {
                    "stage_group": stage_group,
                    "run_name": str(run_name),
                    "run_created_utc": run_created_utc,
                    "recording_id": recording_id,
                    "zarr_use": zarr_use,
                    "method": method,
                    "source_crop_run": _decode_attr(attrs.get("source_crop_run")),
                    "source_keypoint_group": _decode_attr(attrs.get("source_keypoint_group")),
                    "source_keypoints_run": _resolve_source_keypoints_run(attrs),
                    "source_eye_masks_run": _decode_attr(attrs.get("source_eye_masks_run")),
                    "source_eye_masks_method": _decode_attr(attrs.get("source_eye_masks_method")),
                    "total_rois": total_rois,
                    "successful_eyes": successful_eyes,
                    "successful_roi_pairs": successful_roi_pairs,
                    "successful_roi_pair_rate": successful_roi_pair_rate,
                    "duration_seconds": duration_seconds,
                    "rois_per_second": rois_per_second,
                    "inference_duration_seconds": inference_duration_seconds,
                    "inference_average_fps": inference_average_fps,
                    "reason_counts_json": _json_dumps(reason_counts),
                    "summary_statistics_json": _json_dumps(summary_statistics),
                    "review_state": review_state,
                    "review_method": review_method,
                    "review_intended_use": review_intended_use,
                    "review_reviewer": review_reviewer,
                    "review_timestamp_utc": review_timestamp_utc,
                    "source_keypoint_stale_state": source_keypoint_stale_state,
                    "source_keypoint_stale_reason": source_keypoint_stale_reason,
                    "source_keypoint_stale_timestamp_utc": source_keypoint_stale_timestamp_utc,
                    "source_keypoint_stale_json": _json_dumps(source_keypoint_stale),
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_reason": lifecycle_reason,
                    "zarr_mtime_ns": zarr_mtime_ns,
                    "updated_utc": updated_utc,
                }
            )

    return rows


def _extract_eye_mask_quality_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    performance_rows = _extract_eye_mask_performance_rows(
        root,
        zarr_path=zarr_path,
        recording_id=recording_id,
        zarr_use=zarr_use,
    )
    rows: List[Dict[str, Any]] = []
    for performance in performance_rows:
        # Quality parity tracks reviewed refined runs, mirroring detect/keypoint quality scope.
        if str(performance.get("stage_group") or "") != "refined_eye_masks_runs":
            continue
        rows.append(
            {
                "stage_group": str(performance.get("stage_group") or "refined_eye_masks_runs"),
                "run_name": str(performance.get("run_name") or ""),
                "run_created_utc": performance.get("run_created_utc"),
                "recording_id": performance.get("recording_id"),
                "zarr_use": performance.get("zarr_use"),
                "eye_mask_method": performance.get("method"),
                "source_crop_run": performance.get("source_crop_run"),
                "source_keypoint_group": performance.get("source_keypoint_group"),
                "source_keypoints_run": performance.get("source_keypoints_run"),
                "source_eye_masks_run": performance.get("source_eye_masks_run"),
                "source_eye_masks_method": performance.get("source_eye_masks_method"),
                "review_state": performance.get("review_state"),
                "review_method": performance.get("review_method"),
                "review_intended_use": performance.get("review_intended_use"),
                "review_reviewer": performance.get("review_reviewer"),
                "review_timestamp_utc": performance.get("review_timestamp_utc"),
                "total_rois": performance.get("total_rois"),
                "successful_eyes": performance.get("successful_eyes"),
                "successful_roi_pairs": performance.get("successful_roi_pairs"),
                "successful_roi_pair_rate": performance.get("successful_roi_pair_rate"),
                "source_keypoint_stale_state": performance.get("source_keypoint_stale_state"),
                "source_keypoint_stale_reason": performance.get("source_keypoint_stale_reason"),
                "source_keypoint_stale_timestamp_utc": performance.get("source_keypoint_stale_timestamp_utc"),
                "source_keypoint_stale_json": performance.get("source_keypoint_stale_json"),
                "lifecycle_state": performance.get("lifecycle_state"),
                "lifecycle_reason": performance.get("lifecycle_reason"),
                "quality_updated_utc": performance.get("updated_utc") or _utc_now(),
                "zarr_mtime_ns": performance.get("zarr_mtime_ns"),
            }
        )
    return rows


def _first_value(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if value is not None:
                return value
    return None


def _normalize_parents(value: Any) -> List[Dict[str, Optional[str]]]:
    if value is None:
        return []
    if isinstance(value, list):
        parents: List[Dict[str, Optional[str]]] = []
        for item in value:
            if isinstance(item, dict):
                parents.append(
                    {
                        "identifier": item.get("identifier"),
                        "sex": item.get("sex"),
                    }
                )
            elif isinstance(item, str):
                parents.append({"identifier": item, "sex": None})
        return parents
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        parsed = _json_loads(value)
        if isinstance(parsed, list):
            return _normalize_parents(parsed)
        parents = []
        for part in value.split(";"):
            ident = part.strip()
            if ident:
                parents.append({"identifier": ident, "sex": None})
        return parents
    return []


def _compute_path_hash(path: Path) -> str:
    return sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _extract_session_uuid(root: zarr.Group) -> Optional[str]:
    for key in ("session_uuid", "session_id"):
        value = root.attrs.get(key)
        if value:
            return str(value)
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        value = analysis.attrs.get("session_uuid")
        if value:
            return str(value)
    return None


def resolve_dataset_id(root: zarr.Group, zarr_path: Path) -> Tuple[str, Optional[str]]:
    session_uuid = _extract_session_uuid(root)
    dataset_id = session_uuid or f"path-{_compute_path_hash(zarr_path)[:12]}"
    return dataset_id, session_uuid


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    session_uuid: Optional[str]
    recording_id: Optional[str]
    zarr_use: Optional[str]
    zarr_purpose: Optional[str]


def extract_dataset_metadata(
    root: zarr.Group,
    zarr_path: Path,
    *,
    resolve_dataset_id_fn: Callable[[zarr.Group, Path], Tuple[str, Optional[str]]] = resolve_dataset_id,
) -> DatasetMetadata:
    resolved_path = Path(zarr_path).expanduser().resolve()
    dataset_id, session_uuid = resolve_dataset_id_fn(root, resolved_path)
    return DatasetMetadata(
        dataset_id=dataset_id,
        session_uuid=session_uuid,
        recording_id=_decode_attr(root.attrs.get("recording_id")) or _decode_attr(session_uuid),
        zarr_use=_decode_attr(root.attrs.get("zarr_use")),
        zarr_purpose=_decode_attr(root.attrs.get("zarr_purpose")),
    )


def _infer_dataset_artifact_kind(
    *,
    zarr_path: Path,
    dataset_id: str,
    session_uuid: Optional[str],
) -> str:
    normalized = str(zarr_path).replace("\\", "/").lower()
    is_recording_path = "/recordings/" in normalized
    if is_recording_path and session_uuid:
        return "source_recording"
    if "/training/datasets/" in normalized or dataset_id.endswith("_merged"):
        return "derived_training_merge"
    if is_recording_path:
        return "source_recording"
    return "derived_analysis"


def _extract_protocol(root: zarr.Group) -> Tuple[Optional[str], Optional[str]]:
    stim_parent = None
    if "analysis" in root and "stimulus_runs" in root["analysis"]:
        stim_parent = root["analysis"]["stimulus_runs"]
    if stim_parent is None:
        return None, None
    latest = stim_parent.attrs.get("latest")
    if not latest or latest not in stim_parent:
        return None, None
    stim_group = stim_parent[latest]
    raw = stim_group.attrs.get("protocol_json")
    payload = _json_loads(raw)
    if not payload:
        return None, None
    name = payload.get("protocol_name")
    proto_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return str(name) if name else None, proto_hash


def _extract_snapshot(root: zarr.Group) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        for key in ("zebrobot_snapshot", "subject_metadata"):
            raw = analysis.attrs.get(key)
            payload = _json_loads(raw)
            if payload:
                return payload, key
    return None, None


def _extract_session_context(root: zarr.Group) -> Dict[str, Any]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return {}
    raw = analysis.attrs.get("session_context")
    payload = _json_loads(raw)
    return payload if isinstance(payload, dict) else {}


def _extract_recording_context(
    root: zarr.Group,
    zarr_path: Path,
    metadata: DatasetMetadata,
    *,
    context: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    protocol_name: Optional[str],
) -> Dict[str, Any]:
    """Extract canonical recording context embedded in a Zarr root.

    The registry can still be populated from recording manifests, but recording-only
    and video-only archives need direct scan support. Only emit a row when the root
    carries explicit recording context; this avoids creating sparse recording rows
    for older archives that only have a session UUID.
    """

    attrs = root.attrs

    def first_text(*keys: str) -> Optional[str]:
        for key in keys:
            value = _decode_attr(attrs.get(key))
            if value:
                return str(value)
            value = _decode_attr(context.get(key))
            if value:
                return str(value)
        return None

    recording_id = metadata.recording_id or first_text("recording_id")
    if not recording_id:
        return {}

    explicit_keys = (
        "recording_name",
        "recording_type",
        "recording_subtype",
        "behavior_mode",
        "artifact_schema_id",
        "experiment_context_status",
        "experiment_context_source",
    )
    has_explicit_recording_context = any(first_text(key) for key in explicit_keys)
    if not has_explicit_recording_context:
        return {}

    if zarr_path.parent.name == "zarr":
        recording_path = zarr_path.parent.parent
    else:
        recording_path = zarr_path.parent

    stimulus_runs_available = attrs.get("stimulus_runs_available")
    if stimulus_runs_available is None:
        stimulus_runs_available = context.get("stimulus_runs_available")

    return {
        "recording_id": str(recording_id),
        "session_uuid": metadata.session_uuid or first_text("session_uuid", "session_id"),
        "recording_name": first_text("recording_name") or Path(recording_path).name,
        "recording_path": first_text("recording_path") or str(recording_path),
        "started_utc": first_text("session_start_iso8601_utc", "started_utc"),
        "recording_type": first_text("recording_type"),
        "recording_subtype": first_text("recording_subtype"),
        "behavior_mode": first_text("behavior_mode"),
        "artifact_schema_id": first_text("artifact_schema_id"),
        "experiment_context_status": first_text("experiment_context_status"),
        "experiment_context_source": first_text("experiment_context_source"),
        "experiment_context_status_detail": first_text("experiment_context_status_detail"),
        "stimulus_runs_available": _as_bool_int(stimulus_runs_available),
        "rig_id": first_text("rig_id"),
        "arena_id": first_text("arena_id"),
        "camera_id": first_text("camera_id"),
        "canvas_name": first_text("canvas_name"),
        "protocol_name": (
            first_text("protocol_name", "protocol_name_from_definition")
            or protocol_name
        ),
        "dish_design": first_text("dish_design") or _decode_attr(acquisition.get("dish_design")),
    }


def _extract_arena_config(root: zarr.Group) -> Dict[str, Any]:
    analysis = root.get("analysis")
    if analysis is None or "stimulus_runs" not in analysis:
        return {}
    stim_parent = analysis["stimulus_runs"]
    latest = stim_parent.attrs.get("latest")
    if not latest or latest not in stim_parent:
        return {}
    run_group = stim_parent[latest]
    raw = run_group.attrs.get("arena_config_json")
    payload = _json_loads(raw)
    return payload if isinstance(payload, dict) else {}


def _extract_dish_design(root: zarr.Group) -> Optional[str]:
    value = root.attrs.get("dish_design")
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str) and value.strip():
        return value.strip()
    arena_config = _extract_arena_config(root)
    dish_name = arena_config.get("selected_dish_type_name")
    if isinstance(dish_name, (bytes, bytearray)):
        dish_name = dish_name.decode("utf-8", "ignore")
    if isinstance(dish_name, str) and dish_name.strip():
        return dish_name.strip()
    return None


def _extract_camera_metadata(root: zarr.Group) -> Optional[Dict[str, Any]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    raw = analysis.attrs.get("camera_metadata")
    payload = _json_loads(raw)
    return payload if isinstance(payload, dict) else None


def _normalize_downsample_format(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"gray", "grey", "grayscale"}:
        return "gray"
    if text in {"rgb", "color", "colour"}:
        return "rgb"
    return None


def _extract_acquisition(root: zarr.Group) -> Dict[str, Any]:
    raw_video = root.get("raw_video")
    video_codec = None
    video_pix_fmt = None
    fps = None
    source_video = None
    format_title = None
    format_comment = None
    format_encoder = None
    encoder_name = None
    encoder_codec = None
    encoder_preset = None
    encoder_tuning = None
    encoder_rc = None
    encoder_bpp = None
    encoder_target_bps = None
    encoder_res = None
    encoder_res_width = None
    encoder_res_height = None
    encoder_fps = None
    encoder_color = None
    encoder_params = None
    compression_name = None
    compression_level = None
    has_images_ds = None
    has_images_ds_rgb = None
    downsample_formats: List[str] = []
    if raw_video is not None:
        fps = _as_float(raw_video.attrs.get("fps") or raw_video.attrs.get("frames_per_second"))
        video_codec = raw_video.attrs.get("video_codec") or raw_video.attrs.get("codec")
        video_pix_fmt = raw_video.attrs.get("video_pix_fmt") or raw_video.attrs.get("pix_fmt")
        source_video = raw_video.attrs.get("source_video")
        format_title = raw_video.attrs.get("format_title")
        format_comment = raw_video.attrs.get("format_comment")
        format_encoder = raw_video.attrs.get("format_encoder")
        encoder_name = raw_video.attrs.get("encoder_name")
        encoder_codec = raw_video.attrs.get("encoder_codec")
        encoder_preset = raw_video.attrs.get("encoder_preset")
        encoder_tuning = raw_video.attrs.get("encoder_tuning")
        encoder_rc = raw_video.attrs.get("encoder_rc")
        encoder_bpp = raw_video.attrs.get("encoder_bpp")
        encoder_target_bps = raw_video.attrs.get("encoder_target_bps")
        encoder_res = raw_video.attrs.get("encoder_res")
        encoder_res_width = raw_video.attrs.get("encoder_res_width")
        encoder_res_height = raw_video.attrs.get("encoder_res_height")
        encoder_fps = raw_video.attrs.get("encoder_fps")
        encoder_color = raw_video.attrs.get("encoder_color")
        encoder_params = raw_video.attrs.get("encoder_params")
        has_images_ds = "images_ds" in raw_video
        has_images_ds_rgb = "images_ds_rgb" in raw_video
        raw_formats = raw_video.attrs.get("downsample_formats")
        if isinstance(raw_formats, (list, tuple)):
            for item in raw_formats:
                normalized = _normalize_downsample_format(item)
                if normalized and normalized not in downsample_formats:
                    downsample_formats.append(normalized)
        compressor = raw_video.attrs.get("compressor")
        if isinstance(compressor, dict):
            compression_name = compressor.get("name")
            compression_level = _as_int(compressor.get("clevel"))
    if has_images_ds and "gray" not in downsample_formats:
        downsample_formats.append("gray")
    if has_images_ds_rgb and "rgb" not in downsample_formats:
        downsample_formats.append("rgb")
    if format_title is None:
        format_title = root.attrs.get("format_title")
    if format_comment is None:
        format_comment = root.attrs.get("format_comment")
    if format_encoder is None:
        format_encoder = root.attrs.get("format_encoder")
    if encoder_name is None:
        encoder_name = root.attrs.get("encoder_name")
    if encoder_codec is None:
        encoder_codec = root.attrs.get("encoder_codec")
    if encoder_preset is None:
        encoder_preset = root.attrs.get("encoder_preset")
    if encoder_tuning is None:
        encoder_tuning = root.attrs.get("encoder_tuning")
    if encoder_rc is None:
        encoder_rc = root.attrs.get("encoder_rc")
    if encoder_bpp is None:
        encoder_bpp = root.attrs.get("encoder_bpp")
    if encoder_target_bps is None:
        encoder_target_bps = root.attrs.get("encoder_target_bps")
    if encoder_res is None:
        encoder_res = root.attrs.get("encoder_res")
    if encoder_res_width is None:
        encoder_res_width = root.attrs.get("encoder_res_width")
    if encoder_res_height is None:
        encoder_res_height = root.attrs.get("encoder_res_height")
    if encoder_fps is None:
        encoder_fps = root.attrs.get("encoder_fps")
    if encoder_color is None:
        encoder_color = root.attrs.get("encoder_color")
    if encoder_params is None:
        encoder_params = root.attrs.get("encoder_params")

    camera_meta = _extract_camera_metadata(root) or {}
    exposure = _as_float(_first_value(camera_meta, ("exposure", "exposure_ms", "exposure_us")))
    gain = _as_float(_first_value(camera_meta, ("gain", "camera_gain")))
    frame_rate = _as_float(_first_value(camera_meta, ("frame_rate", "fps", "framerate")))
    pixel_format = _first_value(camera_meta, ("pixel_format", "pixelFormat"))
    binning = _first_value(camera_meta, ("bin", "binning"))
    adc = _first_value(camera_meta, ("adc", "bit_depth"))
    camera_model = _first_value(camera_meta, ("device_model_name", "camera_model"))
    camera_serial = _first_value(camera_meta, ("device_serial_number", "serial_number", "camera_id"))

    return {
        "dish_design": _extract_dish_design(root),
        "fps": fps,
        "video_codec": str(video_codec) if video_codec is not None else None,
        "video_pix_fmt": str(video_pix_fmt) if video_pix_fmt is not None else None,
        "source_video": str(source_video) if source_video is not None else None,
        "format_title": _as_text(format_title),
        "format_comment": _as_text(format_comment),
        "format_encoder": _as_text(format_encoder),
        "encoder_name": _as_text(encoder_name),
        "encoder_codec": _as_text(encoder_codec),
        "encoder_preset": _as_text(encoder_preset),
        "encoder_tuning": _as_text(encoder_tuning),
        "encoder_rc": _as_text(encoder_rc),
        "encoder_bpp": _as_float(encoder_bpp),
        "encoder_target_bps": _as_int(encoder_target_bps),
        "encoder_res": _as_text(encoder_res),
        "encoder_res_width": _as_int(encoder_res_width),
        "encoder_res_height": _as_int(encoder_res_height),
        "encoder_fps": _as_float(encoder_fps),
        "encoder_color": _as_int(encoder_color),
        "encoder_params_json": _json_dumps(encoder_params) if encoder_params else None,
        "compression_name": str(compression_name) if compression_name is not None else None,
        "compression_level": compression_level,
        "exposure": exposure,
        "exposure_unit": "us" if exposure is not None else None,
        "gain": gain,
        "frame_rate": frame_rate,
        "pixel_format": str(pixel_format) if pixel_format is not None else None,
        "binning": str(binning) if binning is not None else None,
        "adc": str(adc) if adc is not None else None,
        "camera_model": str(camera_model) if camera_model is not None else None,
        "camera_serial": str(camera_serial) if camera_serial is not None else None,
        "camera_metadata_json": _json_dumps(camera_meta) if camera_meta else None,
        "has_images_ds": bool(has_images_ds) if has_images_ds is not None else None,
        "has_images_ds_rgb": bool(has_images_ds_rgb) if has_images_ds_rgb is not None else None,
        "downsample_formats_json": _json_dumps(downsample_formats) if downsample_formats else None,
    }


def _extract_provenance(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not snapshot:
        return {}
    dish = snapshot.get("dish") or snapshot
    cross = snapshot.get("cross") or {}
    return {
        "fish_id": snapshot.get("fish_id"),
        "subject_count": snapshot.get("subject_count"),
        "dish_id": snapshot.get("dish_id") or dish.get("dish_id"),
        "cross_id": dish.get("cross_id") or cross.get("cross_id"),
        "line_strain": cross.get("line_strain") or dish.get("line_strain"),
        "genotype": dish.get("genotype"),
        "parents": _normalize_parents(cross.get("parents") or dish.get("parents")),
        "species": dish.get("species"),
        "sex": dish.get("sex"),
        # Canonical source field in current subject metadata payloads.
        "dpf_at_acquisition": _as_int(snapshot.get("days_post_fertilization")),
        "snapshot_status": snapshot.get("status"),
        "snapshot_missing": snapshot.get("missing"),
    }


def _extract_zarr_purpose(root: zarr.Group) -> Optional[str]:
    value = root.attrs.get("zarr_purpose")
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_zarr_origin(value: Any) -> Optional[str]:
    text = _as_text(value)
    if text is None:
        return None
    norm = text.lower()
    if norm in {"source", "derived", "imported"}:
        return norm
    return None


def _normalize_zarr_use(value: Any) -> Optional[str]:
    text = _as_text(value)
    if text is None:
        return None
    norm = text.lower()
    if norm in {"training", "analysis", "inference", "export", "archive"}:
        return norm
    if norm in {"production"}:
        return "analysis"
    if norm in {"source_training", "training_merged", "derived_training"}:
        return "training"
    if norm in {"source_analysis", "derived_analysis"}:
        return "analysis"
    if norm in {"inference_output"}:
        return "inference"
    if norm in {"model_input_export"}:
        return "export"
    return None


def _infer_zarr_origin_use(
    *,
    artifact_kind: Optional[str],
    zarr_purpose: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    kind = _as_text(artifact_kind)
    kind_norm = kind.lower() if kind else None
    purpose_origin = None
    purpose_text = _as_text(zarr_purpose)
    if purpose_text:
        purpose_norm = purpose_text.lower()
        if purpose_norm.startswith("source_"):
            purpose_origin = "source"
        elif purpose_norm.startswith("derived_"):
            purpose_origin = "derived"
    if kind_norm == "source_recording":
        kind_origin = "source"
    elif kind_norm in {"derived_analysis", "derived_training_merge", "model_input_export"}:
        kind_origin = "derived"
    else:
        kind_origin = None
    zarr_origin = _normalize_zarr_origin(kind_origin or purpose_origin)

    if kind_norm == "derived_training_merge":
        kind_use = "training"
    elif kind_norm == "model_input_export":
        kind_use = "export"
    elif kind_norm == "source_recording":
        kind_use = None
    elif kind_norm == "derived_analysis":
        kind_use = "analysis"
    else:
        kind_use = None
    zarr_use = _normalize_zarr_use(zarr_purpose) or _normalize_zarr_use(kind_use)
    return zarr_origin, zarr_use


_as_text = _shared_decode_attr


def _normalize_path_text(value: Any) -> Optional[str]:
    text = _as_text(value)
    if text is None:
        return None
    normalized = text.strip("/")
    return normalized or None


def _canonical_run_path(path: Optional[str]) -> Optional[str]:
    normalized = _normalize_path_text(path)
    if normalized is None:
        return None
    parts = normalized.split("/")
    if len(parts) >= 2 and parts[0] in {"refined_detect_runs", "refined_runs", "detect_runs"}:
        return "/".join(parts[:2])
    return normalized


def _infer_detection_source_type(path: Optional[str], fallback: Optional[Any]) -> str:
    fallback_text = _as_text(fallback)
    fallback_norm = fallback_text.lower() if fallback_text else None
    normalized_path = _normalize_path_text(path)
    if normalized_path:
        parts = normalized_path.split("/")
        tail = parts[-1].lower()
        if tail in {"detect", "filtered", "interpolated", "manual", "retune"}:
            return tail
        if parts[0] == "detect_runs":
            return "detect"
    if fallback_norm:
        return fallback_norm
    return "detect"


def _resolve_latest_group_name(parent: Optional[zarr.Group]) -> Optional[str]:
    if parent is None:
        return None
    latest = _as_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    if hasattr(parent, "group_keys"):
        names = sorted(
            name
            for name in parent.group_keys()
            if isinstance(name, str)
        )
    else:
        names = sorted(
            name
            for name in parent.keys()
            if isinstance(name, str)
        )
    if not names:
        return None
    return names[-1]


def _build_detection_source_records(root: zarr.Group) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    crop_parent = root.get("crop_runs")
    crop_run_name = _resolve_latest_group_name(crop_parent)
    if crop_parent is not None and crop_run_name and crop_run_name in crop_parent:
        crop_group = crop_parent[crop_run_name]
        source_path = _normalize_path_text(crop_group.attrs.get("detection_source_path"))
        source_type = _infer_detection_source_type(source_path, crop_group.attrs.get("detection_source_type"))

        refined_ref = _canonical_run_path(crop_group.attrs.get("detect_review_status_ref"))
        if refined_ref is None:
            refined_ref = _canonical_run_path(source_path)
        if refined_ref is None:
            source_refined = _as_text(crop_group.attrs.get("source_refined_run"))
            if source_refined:
                preferred = f"refined_detect_runs/{source_refined}"
                legacy = f"refined_runs/{source_refined}"
                if preferred in root:
                    refined_ref = preferred
                elif legacy in root:
                    refined_ref = legacy
                else:
                    refined_ref = preferred
        if refined_ref is None:
            refined_ref = "unknown"

        total_detections = int(crop_group["bbox_norm_coords"].shape[0]) if "bbox_norm_coords" in crop_group else 0
        source_code_counts: Dict[str, int] = {}
        if "detection_source" in crop_group:
            raw_source = np.asarray(crop_group["detection_source"][:], dtype=np.int64)
            if raw_source.size > 0:
                unique, counts = np.unique(raw_source, return_counts=True)
                source_code_counts = {
                    str(int(code)): int(count)
                    for code, count in zip(unique.tolist(), counts.tolist())
                }

        n_real_attr = _as_int(crop_group.attrs.get("n_real_detections"))
        n_interp_attr = _as_int(crop_group.attrs.get("n_interpolated_detections"))
        n_real = source_code_counts.get("0", n_real_attr if n_real_attr is not None else total_detections)
        n_interpolated = source_code_counts.get("1", n_interp_attr if n_interp_attr is not None else 0)
        includes_interpolated = bool(
            crop_group.attrs.get("includes_interpolated", n_interpolated > 0)
        )

        counts_payload = {
            "crop_run": crop_run_name,
            "detection_source_path": source_path,
            "total_detections": int(total_detections),
            "n_real_detections": int(max(n_real, 0)),
            "n_interpolated_detections": int(max(n_interpolated, 0)),
            "includes_interpolated": includes_interpolated,
        }
        if source_code_counts:
            counts_payload["detection_source_codes"] = source_code_counts

        records.append(
            {
                "refined_run": refined_ref,
                "source_type": source_type,
                "counts": counts_payload,
            }
        )
        return records

    detect_parent = root.get("detect_runs")
    detect_run_name = _resolve_latest_group_name(detect_parent)
    if detect_parent is None or detect_run_name is None or detect_run_name not in detect_parent:
        return records

    detect_group = detect_parent[detect_run_name]
    total_detections = int(detect_group["bbox_norm_coords"].shape[0]) if "bbox_norm_coords" in detect_group else 0
    detect_path = f"detect_runs/{detect_run_name}"
    records.append(
        {
            "refined_run": detect_path,
            "source_type": "detect",
            "counts": {
                "detect_run": detect_run_name,
                "detection_source_path": detect_path,
                "total_detections": total_detections,
                "n_real_detections": total_detections,
                "n_interpolated_detections": 0,
                "includes_interpolated": False,
            },
        }
    )
    return records


class Registry:
    def __init__(self, path: Path):
        self.path = path
        _ensure_parent(self.path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._ensure_schema_version_table()
        self._apply_schema_migrations()

    def _schema_migrations(self) -> List[Tuple[int, str, Callable[[], None]]]:
        return [
            (1, "initial_registry_schema", self._migration_001_initial_schema),
            # Reserved template migration to make the ordered migration pattern explicit.
            # Future schema changes should append new versions rather than modifying old ones.
            (2, "reserved_noop_template", self._migration_002_reserved_noop),
            (3, "recording_columns_reconcile", self._migration_003_recording_columns_reconcile),
            (4, "recording_overview_refresh", self._migration_004_recording_overview_refresh),
            (5, "drop_provenance_zarr_purpose", self._migration_005_drop_provenance_zarr_purpose),
            (6, "subject_dish_cross_entities", self._migration_006_subject_dish_cross_entities),
            (7, "subjects_entities_and_query_indexes", self._migration_007_subjects_entities_and_query_indexes),
            (8, "recording_subject_overview_view", self._migration_008_recording_subject_overview_view),
            (9, "training_task_type_columns", self._migration_009_training_task_type_columns),
            (10, "detect_performance_registry", self._migration_010_detect_performance_registry),
            (11, "detect_model_performance_views", self._migration_011_detect_model_performance_views),
            (12, "detect_performance_model_identity", self._migration_012_detect_performance_model_identity),
            (13, "detect_model_performance_summary_views", self._migration_013_detect_model_performance_summary_views),
            (14, "crop_quality_registry", self._migration_014_crop_quality_registry),
            (15, "eye_mask_performance_registry", self._migration_015_eye_mask_performance_registry),
            (16, "model_export_nms_threshold_columns", self._migration_016_model_export_nms_threshold_columns),
            (
                17,
                "eye_mask_performance_review_stale_columns",
                self._migration_017_eye_mask_performance_review_stale_columns,
            ),
            (18, "keypoint_performance_registry", self._migration_018_keypoint_performance_registry),
            (19, "recording_step_status_registry", self._migration_019_recording_step_status_registry),
            (20, "recording_step_status_wide_view", self._migration_020_recording_step_status_wide_view),
            (
                21,
                "detect_keypoint_quality_review_columns",
                self._migration_021_detect_keypoint_quality_review_columns,
            ),
            (
                22,
                "detection_data_profile_registry",
                self._migration_022_detection_data_profile_registry,
            ),
            (
                23,
                "detection_data_profile_lineage_projection",
                self._migration_023_detection_data_profile_lineage_projection,
            ),
            (
                24,
                "keypoint_data_profile_registry",
                self._migration_024_keypoint_data_profile_registry,
            ),
            (
                25,
                "eye_mask_data_profile_registry",
                self._migration_025_eye_mask_data_profile_registry,
            ),
            (
                26,
                "eye_mask_quality_registry",
                self._migration_026_eye_mask_quality_registry,
            ),
            (
                27,
                "detect_quality_wide_view_columns",
                self._migration_027_detect_quality_wide_view_columns,
            ),
            (
                28,
                "keypoint_auto_review_policy_columns",
                self._migration_028_keypoint_auto_review_policy_columns,
            ),
            (
                29,
                "keypoint_quality_current_latest_source_preference",
                self._migration_029_keypoint_quality_current_latest_source_preference,
            ),
            (
                30,
                "tracking_unassigned_warning_wide_view",
                self._migration_030_tracking_unassigned_warning_wide_view,
            ),
            (
                31,
                "tracking_qc_state_wide_view",
                self._migration_031_tracking_qc_state_wide_view,
            ),
            (
                32,
                "subject_mask_registry",
                self._migration_032_subject_mask_registry,
            ),
            (
                33,
                "subject_mask_registry_semantics_columns",
                self._migration_033_subject_mask_registry_semantics_columns,
            ),
            (
                34,
                "dataset_context_current_view",
                self._migration_034_dataset_context_current_view,
            ),
            (
                35,
                "recording_step_status_latest_dataset_context_current",
                self._migration_035_recording_step_status_latest_dataset_context_current,
            ),
            (
                36,
                "subject_mask_component_latest_views",
                self._migration_036_subject_mask_component_latest_views,
            ),
            (
                37,
                "subject_mask_component_eye_compat_latest_views",
                self._migration_037_subject_mask_component_eye_compat_latest_views,
            ),
            (
                38,
                "subject_mask_component_partial_run_preference",
                self._migration_038_subject_mask_component_partial_run_preference,
            ),
            (
                39,
                "subject_mask_component_source_stale_views",
                self._migration_039_subject_mask_component_source_stale_views,
            ),
            (
                40,
                "subject_mask_training_model_discovery",
                self._migration_040_subject_mask_training_model_discovery,
            ),
            (
                41,
                "analytics_manifest_registry",
                self._migration_041_analytics_manifest_registry,
            ),
            (
                42,
                "recording_experiment_context_columns",
                self._migration_042_recording_experiment_context_columns,
            ),
            (
                43,
                "stage_catalog_recording_step_status_wide_view",
                self._migration_043_stage_catalog_recording_step_status_wide_view,
            ),
            (
                44,
                "derived_analysis_recording_step_status_wide_view",
                self._migration_044_derived_analysis_recording_step_status_wide_view,
            ),
            (
                45,
                "tail_behavior_recording_step_status_wide_view",
                self._migration_045_tail_behavior_recording_step_status_wide_view,
            ),
            (
                46,
                "source_freshness_recording_step_status_wide_view",
                self._migration_046_source_freshness_recording_step_status_wide_view,
            ),
            (
                47,
                "bout_stimulus_source_freshness_recording_step_status_wide_view",
                self._migration_047_bout_stimulus_source_freshness_recording_step_status_wide_view,
            ),
            (
                48,
                "eye_shape_source_freshness_recording_step_status_wide_view",
                self._migration_048_eye_shape_source_freshness_recording_step_status_wide_view,
            ),
            (
                49,
                "model_input_shape_registry",
                self._migration_049_model_input_shape_registry,
            ),
            (
                50,
                "detect_quality_current_reviewed_preference",
                self._migration_050_detect_quality_current_reviewed_preference,
            ),
            (
                51,
                "training_image_profile_registry",
                self._migration_051_training_image_profile_registry,
            ),
        ]

    def _ensure_schema_version_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_utc TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _current_schema_version(self) -> Optional[int]:
        row = self.conn.execute("SELECT MAX(version) AS version FROM schema_version;").fetchone()
        if row is None:
            return None
        value = row["version"]
        if value is None:
            return None
        return int(value)

    def _has_legacy_schema(self) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='datasets'
            LIMIT 1;
            """
        ).fetchone()
        return row is not None

    def _table_exists(self, table_name: str) -> bool:
        row = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name=?
            LIMIT 1;
            """,
            (str(table_name),),
        ).fetchone()
        return row is not None

    def _sqlite_object_exists(self, object_name: str, *, object_types: Sequence[str] = ("table", "view")) -> bool:
        allowed_types = {str(item) for item in object_types if item}
        if not allowed_types:
            return False
        placeholders = ", ".join("?" for _ in allowed_types)
        row = self.conn.execute(
            f"""
            SELECT 1
            FROM sqlite_master
            WHERE type IN ({placeholders}) AND name=?
            LIMIT 1;
            """,
            (*sorted(allowed_types), str(object_name)),
        ).fetchone()
        return row is not None

    def _record_schema_version(self, *, version: int, name: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO schema_version (version, name, applied_utc)
            VALUES (?, ?, ?);
            """,
            (int(version), str(name), _utc_now()),
        )

    def _apply_schema_migrations(self) -> None:
        migrations = sorted(self._schema_migrations(), key=lambda item: item[0])
        if not migrations:
            return
        latest = migrations[-1][0]
        current = self._current_schema_version()
        if current is None:
            if self._has_legacy_schema():
                # Existing registry predates schema_version tracking.
                self.conn.execute("BEGIN IMMEDIATE;")
                try:
                    self._record_schema_version(version=latest, name="legacy_bootstrap")
                    self.conn.execute(f"PRAGMA user_version = {int(latest)};")
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise
                return
            current = 0

        for version, name, apply_fn in migrations:
            if version <= current:
                continue
            self.conn.execute("BEGIN IMMEDIATE;")
            try:
                apply_fn()
                self._record_schema_version(version=version, name=name)
                self.conn.execute(f"PRAGMA user_version = {int(version)};")
                self.conn.commit()
                current = version
            except Exception:
                self.conn.rollback()
                raise

    def _migration_001_initial_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                zarr_path TEXT NOT NULL,
                recording_id TEXT,
                artifact_kind TEXT,
                zarr_origin TEXT,
                zarr_use TEXT,
                path_hash TEXT,
                created_utc TEXT,
                last_seen_utc TEXT,
                status TEXT
            );
            """
        )
        # Existing registries may predate these columns; add them before creating
        # any index/view that references the new fields.
        self._ensure_columns(
            "datasets",
            {
                "recording_id": "TEXT",
                "artifact_kind": "TEXT",
                "zarr_origin": "TEXT",
                "zarr_use": "TEXT",
            },
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                recording_name TEXT,
                recording_path TEXT,
                started_utc TEXT,
                recording_type TEXT,
                recording_subtype TEXT,
                behavior_mode TEXT,
                artifact_schema_id TEXT,
                experiment_context_status TEXT,
                experiment_context_source TEXT,
                experiment_context_status_detail TEXT,
                stimulus_runs_available INTEGER,
                rig_id TEXT,
                arena_id TEXT,
                camera_id TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                dish_design TEXT,
                created_utc TEXT,
                updated_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_type_vocab (
                recording_type TEXT PRIMARY KEY,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_subtype_vocab (
                recording_type TEXT NOT NULL,
                recording_subtype TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT,
                PRIMARY KEY (recording_type, recording_subtype),
                FOREIGN KEY(recording_type) REFERENCES recording_type_vocab(recording_type) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS zarr_origin_vocab (
                zarr_origin TEXT PRIMARY KEY,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS zarr_use_vocab (
                zarr_use TEXT PRIMARY KEY,
                active INTEGER NOT NULL DEFAULT 1,
                description TEXT
            );
            """
        )
        # Existing registries may predate recording_subtype.
        # Add it before creating indexes that reference this column.
        self._ensure_columns(
            "recordings",
            {
                "recording_subtype": "TEXT",
                "behavior_mode": "TEXT",
                "dish_design": "TEXT",
                "experiment_context_status": "TEXT",
                "experiment_context_source": "TEXT",
                "experiment_context_status_detail": "TEXT",
                "stimulus_runs_available": "INTEGER",
            },
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO recording_type_vocab (recording_type, active, description)
            VALUES (?, 1, ?);
            """,
            [
                ("behavior", "Behavior recordings"),
                ("microscopy", "Microscopy recordings"),
                ("histology", "Histology recordings"),
            ],
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO recording_subtype_vocab (
                recording_type, recording_subtype, active, description
            )
            VALUES (?, ?, 1, ?);
            """,
            [
                ("behavior", "free", "Freely swimming behavior"),
                ("behavior", "embedded", "Embedded behavior"),
                ("microscopy", "lightsheet", "Light-sheet microscopy"),
                ("microscopy", "confocal", "Confocal microscopy"),
                ("microscopy", "2p", "Two-photon microscopy"),
                ("histology", "section", "Section histology"),
                ("histology", "wholemount", "Whole-mount histology"),
            ],
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO zarr_origin_vocab (zarr_origin, active, description)
            VALUES (?, 1, ?);
            """,
            [
                ("source", "Source recording artifact"),
                ("derived", "Derived artifact produced from other artifacts"),
                ("imported", "Imported external artifact"),
            ],
        )
        cur.executemany(
            """
            INSERT OR IGNORE INTO zarr_use_vocab (zarr_use, active, description)
            VALUES (?, 1, ?);
            """,
            [
                ("training", "Used for model training"),
                ("analysis", "Used for analysis"),
                ("inference", "Inference outputs"),
                ("export", "Exported model/input artifact"),
                ("archive", "Archived/cold artifact"),
            ],
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_artifacts (
                artifact_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                artifact_group TEXT,
                relpath TEXT,
                path TEXT NOT NULL,
                file_ext TEXT,
                status TEXT,
                size_bytes INTEGER,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                FOREIGN KEY(recording_id) REFERENCES recordings(recording_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance (
                dataset_id TEXT PRIMARY KEY,
                fish_id TEXT,
                subject_count INTEGER,
                dish_id TEXT,
                dish_design TEXT,
                cross_id TEXT,
                line_strain TEXT,
                genotype TEXT,
                parents_json TEXT,
                species TEXT,
                sex TEXT,
                dpf_at_acquisition INTEGER,
                rig_id TEXT,
                arena_id TEXT,
                camera_id TEXT,
                canvas_name TEXT,
                fps REAL,
                video_codec TEXT,
                video_pix_fmt TEXT,
                format_title TEXT,
                format_comment TEXT,
                format_encoder TEXT,
                encoder_name TEXT,
                encoder_codec TEXT,
                encoder_preset TEXT,
                encoder_tuning TEXT,
                encoder_rc TEXT,
                encoder_bpp REAL,
                encoder_target_bps INTEGER,
                encoder_res TEXT,
                encoder_res_width INTEGER,
                encoder_res_height INTEGER,
                encoder_fps REAL,
                encoder_color INTEGER,
                encoder_params_json TEXT,
                source_video TEXT,
                compression_name TEXT,
                compression_level INTEGER,
                exposure REAL,
                exposure_unit TEXT,
                gain REAL,
                frame_rate REAL,
                pixel_format TEXT,
                binning TEXT,
                adc TEXT,
                camera_model TEXT,
                camera_serial TEXT,
                camera_metadata_json TEXT,
                has_images_ds INTEGER,
                has_images_ds_rgb INTEGER,
                downsample_formats_json TEXT,
                protocol_name TEXT,
                protocol_hash TEXT,
                snapshot_status TEXT,
                snapshot_missing_json TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_sources (
                dataset_id TEXT NOT NULL,
                refined_run TEXT,
                source_type TEXT,
                counts_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (dataset_id, refined_run, source_type),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_lineage (
                child_dataset_id TEXT NOT NULL,
                parent_dataset_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                source_set_id TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                PRIMARY KEY (child_dataset_id, parent_dataset_id, relationship_type),
                FOREIGN KEY(child_dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
                FOREIGN KEY(parent_dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_dataset_lineage_no_self_insert
            BEFORE INSERT ON dataset_lineage
            FOR EACH ROW
            WHEN NEW.child_dataset_id = NEW.parent_dataset_id
            BEGIN
                SELECT RAISE(ABORT, 'dataset_lineage self-edge is not allowed');
            END;
            """
        )
        cur.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_dataset_lineage_no_self_update
            BEFORE UPDATE ON dataset_lineage
            FOR EACH ROW
            WHEN NEW.child_dataset_id = NEW.parent_dataset_id
            BEGIN
                SELECT RAISE(ABORT, 'dataset_lineage self-edge is not allowed');
            END;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keypoint_quality (
                dataset_id TEXT NOT NULL,
                refined_run TEXT NOT NULL,
                refined_created_utc TEXT,
                source_keypoint_run TEXT NOT NULL,
                keypoint_method TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_notes TEXT,
                review_policy_id TEXT,
                review_policy_version INTEGER,
                review_timestamp_utc TEXT,
                usable_keypoints INTEGER,
                total_keypoints INTEGER,
                usable_keypoints_rate REAL,
                raw_keypoints_success_rate REAL,
                raw_keypoints_successful INTEGER,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, refined_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detect_quality (
                dataset_id TEXT NOT NULL,
                refined_run TEXT NOT NULL,
                refined_created_utc TEXT,
                source_detect_run TEXT NOT NULL,
                detect_method TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_notes TEXT,
                review_timestamp_utc TEXT,
                review_resolved_group TEXT,
                total_detections INTEGER,
                real_detections INTEGER,
                interpolated_detections INTEGER,
                interpolated_detections_rate REAL,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, refined_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pose_skeleton_specs (
                skeleton_id TEXT PRIMARY KEY,
                spec_sha256 TEXT NOT NULL UNIQUE,
                name TEXT,
                kpt_shape_json TEXT,
                keypoint_labels_json TEXT,
                edges_json TEXT,
                spec_json TEXT,
                created_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_sets (
                set_id TEXT PRIMARY KEY,
                name TEXT,
                task_type TEXT,
                query_filter TEXT,
                dataset_ids_json TEXT,
                skeleton_id TEXT,
                invocation_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                task_type TEXT,
                config_path TEXT,
                manifest_path TEXT,
                skeleton_id TEXT,
                model_path TEXT,
                metrics_path TEXT,
                config_sha256 TEXT,
                manifest_sha256 TEXT,
                model_sha256 TEXT,
                metrics_sha256 TEXT,
                status TEXT,
                final_metrics_json TEXT,
                invocation_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_exports (
                run_id TEXT NOT NULL,
                export_type TEXT NOT NULL,
                path TEXT,
                manifest_path TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (run_id, export_type),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_models (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                model_path TEXT,
                model_sha256 TEXT,
                metrics_path TEXT,
                metrics_sha256 TEXT,
                status TEXT,
                task_type TEXT,
                label_schema_id TEXT,
                coverage_class TEXT,
                component_coverage_key TEXT,
                mask_labels_json TEXT,
                component_groups_json TEXT,
                best_metric_name TEXT,
                best_metric_value REAL,
                best_epoch INTEGER,
                input_shape TEXT,
                input_layout TEXT,
                input_channels INTEGER,
                img_h INTEGER,
                img_w INTEGER,
                max_batch INTEGER,
                dynamic_shapes INTEGER,
                input_dtype TEXT,
                input_color_space TEXT,
                input_shape_source TEXT,
                input_shape_status TEXT,
                final_metrics_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS onnx_models (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                skeleton_id TEXT,
                detection_model_run_id TEXT,
                path TEXT,
                sha256 TEXT,
                manifest_path TEXT,
                manifest_sha256 TEXT,
                opset INTEGER,
                nms_conf REAL,
                nms_iou REAL,
                nms_topk INTEGER,
                input_shape TEXT,
                img_h INTEGER,
                img_w INTEGER,
                max_batch INTEGER,
                dynamic_shapes INTEGER,
                file_size_bytes INTEGER,
                exporter_torch_version TEXT,
                exporter_cuda_version TEXT,
                exporter_hostname TEXT,
                requires_plugins INTEGER,
                plugin_ops_json TEXT,
                plugin_versions_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tensorrt_models (
                run_id TEXT NOT NULL,
                set_id TEXT,
                skeleton_id TEXT,
                detection_model_run_id TEXT,
                onnx_run_id TEXT,
                precision TEXT NOT NULL,
                nms_conf REAL,
                nms_iou REAL,
                nms_topk INTEGER,
                path TEXT,
                sha256 TEXT,
                manifest_path TEXT,
                manifest_sha256 TEXT,
                input_shape TEXT,
                img_h INTEGER,
                img_w INTEGER,
                max_batch INTEGER,
                dynamic_shapes INTEGER,
                file_size_bytes INTEGER,
                trt_version TEXT,
                cuda_version TEXT,
                compute_capability TEXT,
                gpu_name TEXT,
                gpu_uuid TEXT,
                system_hostname TEXT,
                requires_plugins INTEGER,
                plugin_ops_json TEXT,
                plugin_versions_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (run_id, precision),
                FOREIGN KEY(skeleton_id) REFERENCES pose_skeleton_specs(skeleton_id),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_set_id ON training_models(set_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_task_status ON training_models(task_type, status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_label_schema ON training_models(label_schema_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_component_coverage ON training_models(component_coverage_key);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_sets_skeleton_id ON training_sets(skeleton_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_sets_task_type ON training_sets(task_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_runs_skeleton_id ON training_runs(skeleton_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_runs_task_type ON training_runs(task_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_recording_id ON datasets(recording_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_artifact_kind ON datasets(artifact_kind);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_origin_use ON datasets(zarr_origin, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recordings_session_uuid ON recordings(session_uuid);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recordings_type_subtype ON recordings(recording_type, recording_subtype);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recordings_behavior_mode ON recordings(behavior_mode);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subtype_vocab_type_active ON recording_subtype_vocab(recording_type, active);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_artifacts_recording_id ON recording_artifacts(recording_id);"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_recording_artifacts_recording_path ON recording_artifacts(recording_id, path);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_lineage_child_rel ON dataset_lineage(child_dataset_id, relationship_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_lineage_parent_rel ON dataset_lineage(parent_dataset_id, relationship_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_onnx_models_set_id ON onnx_models(set_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_onnx_models_skeleton_id ON onnx_models(skeleton_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tensorrt_models_set_id ON tensorrt_models(set_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tensorrt_models_skeleton_id ON tensorrt_models(skeleton_id);"
        )
        # Migrate legacy detection_models rows into training_models.
        legacy_detection_models = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='detection_models';"
        ).fetchone()
        if legacy_detection_models is not None:
            cur.execute(
                """
                INSERT OR REPLACE INTO training_models (
                    run_id, set_id, model_path, model_sha256, metrics_path, metrics_sha256,
                    status, final_metrics_json, metadata_json, created_utc
                )
                SELECT
                    run_id, set_id, model_path, model_sha256, metrics_path, metrics_sha256,
                    status, final_metrics_json, metadata_json, created_utc
                FROM detection_models
                """
            )
            cur.execute("DROP TABLE detection_models;")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_quality_dataset_id ON keypoint_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_keypoint_quality_gate
            ON keypoint_quality(review_state, review_intended_use, keypoint_method, usable_keypoints_rate);
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_quality_dataset_id ON detect_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_detect_quality_gate
            ON detect_quality(review_state, review_intended_use, detect_method, interpolated_detections_rate);
            """
        )
        # Ensure additive review columns exist before refreshing views on legacy registries.
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_current;")
        cur.execute(
            """
            CREATE VIEW keypoint_quality_current AS
            WITH ranked AS (
                SELECT
                    kq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY kq.dataset_id, COALESCE(kq.keypoint_method, '')
                        ORDER BY
                            COALESCE(kq.review_timestamp_utc, kq.refined_created_utc, kq.quality_updated_utc) DESC,
                            COALESCE(kq.refined_created_utc, '') DESC,
                            kq.refined_run DESC
                    ) AS _rn
                FROM keypoint_quality kq
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_keypoint_run,
                keypoint_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_policy_id,
                review_policy_version,
                review_timestamp_utc,
                usable_keypoints,
                total_keypoints,
                usable_keypoints_rate,
                raw_keypoints_success_rate,
                raw_keypoints_successful,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_overview;")
        cur.execute(
            """
            CREATE VIEW keypoint_quality_overview AS
            SELECT
                kqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                kqc.keypoint_method AS keypoint_method,
                kqc.source_keypoint_run AS source_keypoint_run,
                kqc.refined_run AS refined_run,
                kqc.review_state AS review_state,
                kqc.review_method AS review_method,
                kqc.review_intended_use AS review_intended_use,
                kqc.review_policy_id AS review_policy_id,
                kqc.review_policy_version AS review_policy_version,
                kqc.usable_keypoints AS usable_keypoints,
                kqc.total_keypoints AS total_keypoints,
                kqc.usable_keypoints_rate AS usable_keypoints_rate,
                kqc.quality_updated_utc AS quality_updated_utc,
                kqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN kqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM keypoint_quality_current kqc
            LEFT JOIN datasets d ON d.dataset_id = kqc.dataset_id;
            """
        )
        cur.execute("DROP VIEW IF EXISTS detect_quality_current;")
        cur.execute(
            """
            CREATE VIEW detect_quality_current AS
            WITH ranked AS (
                SELECT
                    dq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dq.dataset_id, COALESCE(dq.detect_method, '')
                        ORDER BY
                            COALESCE(dq.review_timestamp_utc, dq.refined_created_utc, dq.quality_updated_utc) DESC,
                            COALESCE(dq.refined_created_utc, '') DESC,
                            dq.refined_run DESC
                    ) AS _rn
                FROM detect_quality dq
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_detect_run,
                detect_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_timestamp_utc,
                review_resolved_group,
                total_detections,
                real_detections,
                interpolated_detections,
                interpolated_detections_rate,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS refined_detect_review_current;")
        cur.execute(
            """
            CREATE VIEW refined_detect_review_current AS
            SELECT * FROM detect_quality_current;
            """
        )
        cur.execute("DROP VIEW IF EXISTS merged_training_datasets;")
        cur.execute(
            """
            CREATE VIEW merged_training_datasets AS
            SELECT
                d.dataset_id,
                d.recording_id,
                d.session_uuid,
                d.zarr_path,
                d.status,
                d.artifact_kind,
                d.zarr_origin,
                d.zarr_use,
                d.zarr_use AS zarr_purpose,
                d.last_seen_utc
            FROM datasets d
            WHERE
                d.artifact_kind = 'derived_training_merge'
                OR d.dataset_id LIKE '%_merged';
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_overview;")
        cur.execute(
            """
            CREATE VIEW recording_overview AS
            SELECT
                r.recording_id AS recording_id,
                r.session_uuid AS session_uuid,
                r.recording_name AS recording_name,
                r.recording_path AS recording_path,
                r.started_utc AS started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.artifact_schema_id AS artifact_schema_id,
                r.experiment_context_status AS experiment_context_status,
                r.experiment_context_source AS experiment_context_source,
                r.experiment_context_status_detail AS experiment_context_status_detail,
                r.stimulus_runs_available AS stimulus_runs_available,
                COALESCE(
                    NULLIF(TRIM(r.dish_design), ''),
                    GROUP_CONCAT(DISTINCT NULLIF(TRIM(dcc.dish_design), ''))
                ) AS dish_design,
                r.rig_id AS rig_id,
                r.arena_id AS arena_id,
                r.camera_id AS camera_id,
                r.protocol_name AS protocol_name,
                COUNT(DISTINCT d.dataset_id) AS dataset_count,
                SUM(CASE WHEN d.zarr_use = 'training' THEN 1 ELSE 0 END) AS training_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'analysis' THEN 1 ELSE 0 END) AS analysis_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'inference' THEN 1 ELSE 0 END) AS inference_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'export' THEN 1 ELSE 0 END) AS export_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, 'active')) = 'active' THEN 1 ELSE 0 END) AS active_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, '')) = 'missing' THEN 1 ELSE 0 END) AS missing_dataset_count,
                COALESCE(MAX(d.last_seen_utc), r.updated_utc, r.created_utc) AS last_seen_utc
            FROM recordings r
            LEFT JOIN datasets d ON d.recording_id = r.recording_id
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = d.dataset_id
            GROUP BY
                r.recording_id,
                r.session_uuid,
                r.recording_name,
                r.recording_path,
                r.started_utc,
                r.recording_type,
                r.recording_subtype,
                r.behavior_mode,
                r.artifact_schema_id,
                r.experiment_context_status,
                r.experiment_context_source,
                r.experiment_context_status_detail,
                r.stimulus_runs_available,
                r.dish_design,
                r.rig_id,
                r.arena_id,
                r.camera_id,
                r.protocol_name;
            """
        )
        cur.execute("DROP VIEW IF EXISTS dataset_lineage_current;")
        cur.execute(
            """
            CREATE VIEW dataset_lineage_current AS
            WITH ranked AS (
                SELECT
                    dl.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dl.child_dataset_id, dl.parent_dataset_id, dl.relationship_type
                        ORDER BY COALESCE(dl.updated_utc, dl.created_utc) DESC
                    ) AS _rn
                FROM dataset_lineage dl
            )
            SELECT
                child_dataset_id,
                parent_dataset_id,
                relationship_type,
                source_set_id,
                metadata_json,
                created_utc,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )
        self._ensure_columns(
            "datasets",
            {
                "recording_id": "TEXT",
                "artifact_kind": "TEXT",
                "zarr_origin": "TEXT",
                "zarr_use": "TEXT",
            },
        )
        self._ensure_columns(
            "provenance",
            {
                "fish_id": "TEXT",
                "subject_count": "INTEGER",
                "rig_id": "TEXT",
                "arena_id": "TEXT",
                "camera_id": "TEXT",
                "canvas_name": "TEXT",
                "dish_design": "TEXT",
                "fps": "REAL",
                "video_codec": "TEXT",
                "video_pix_fmt": "TEXT",
                "format_title": "TEXT",
                "format_comment": "TEXT",
                "format_encoder": "TEXT",
                "encoder_name": "TEXT",
                "encoder_codec": "TEXT",
                "encoder_preset": "TEXT",
                "encoder_tuning": "TEXT",
                "encoder_rc": "TEXT",
                "encoder_bpp": "REAL",
                "encoder_target_bps": "INTEGER",
                "encoder_res": "TEXT",
                "encoder_res_width": "INTEGER",
                "encoder_res_height": "INTEGER",
                "encoder_fps": "REAL",
                "encoder_color": "INTEGER",
                "encoder_params_json": "TEXT",
                "source_video": "TEXT",
                "compression_name": "TEXT",
                "compression_level": "INTEGER",
                "exposure": "REAL",
                "exposure_unit": "TEXT",
                "gain": "REAL",
                "frame_rate": "REAL",
                "pixel_format": "TEXT",
                "binning": "TEXT",
                "adc": "TEXT",
                "camera_model": "TEXT",
                "camera_serial": "TEXT",
                "camera_metadata_json": "TEXT",
                "has_images_ds": "INTEGER",
                "has_images_ds_rgb": "INTEGER",
                "downsample_formats_json": "TEXT",
            },
        )
        # Backfill normalized zarr origin/use for legacy registries.
        self.conn.execute(
            """
            UPDATE datasets
            SET zarr_origin = CASE
                WHEN artifact_kind = 'source_recording' THEN 'source'
                WHEN artifact_kind IN ('derived_analysis', 'derived_training_merge', 'model_input_export') THEN 'derived'
                ELSE zarr_origin
            END
            WHERE zarr_origin IS NULL;
            """
        )
        self.conn.execute(
            """
            UPDATE datasets
            SET zarr_use = CASE
                WHEN artifact_kind = 'derived_training_merge' THEN 'training'
                WHEN artifact_kind = 'derived_analysis' THEN 'analysis'
                WHEN artifact_kind = 'model_input_export' THEN 'export'
                ELSE zarr_use
            END
            WHERE zarr_use IS NULL;
            """
        )
        self._ensure_columns(
            "detect_quality",
            {
                "refined_created_utc": "TEXT",
                "source_detect_run": "TEXT",
                "detect_method": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_notes": "TEXT",
                "review_timestamp_utc": "TEXT",
                "review_resolved_group": "TEXT",
                "total_detections": "INTEGER",
                "real_detections": "INTEGER",
                "interpolated_detections": "INTEGER",
                "interpolated_detections_rate": "REAL",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )
        self._ensure_columns(
            "keypoint_quality",
            {
                "refined_created_utc": "TEXT",
                "source_keypoint_run": "TEXT",
                "keypoint_method": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_notes": "TEXT",
                "review_timestamp_utc": "TEXT",
                "usable_keypoints": "INTEGER",
                "total_keypoints": "INTEGER",
                "usable_keypoints_rate": "REAL",
                "raw_keypoints_success_rate": "REAL",
                "raw_keypoints_successful": "INTEGER",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )
        self._ensure_columns(
            "training_sets",
            {"invocation_json": "TEXT", "skeleton_id": "TEXT", "task_type": "TEXT"},
        )
        self._ensure_columns(
            "training_runs",
            {
                "invocation_json": "TEXT",
                "skeleton_id": "TEXT",
                "task_type": "TEXT",
                "config_sha256": "TEXT",
                "manifest_sha256": "TEXT",
                "model_sha256": "TEXT",
                "metrics_sha256": "TEXT",
                "status": "TEXT",
                "final_metrics_json": "TEXT",
            },
        )
        self._ensure_columns(
            "training_models",
            {
                "set_id": "TEXT",
                "model_path": "TEXT",
                "model_sha256": "TEXT",
                "metrics_path": "TEXT",
                "metrics_sha256": "TEXT",
                "status": "TEXT",
                "input_shape": "TEXT",
                "input_layout": "TEXT",
                "input_channels": "INTEGER",
                "img_h": "INTEGER",
                "img_w": "INTEGER",
                "max_batch": "INTEGER",
                "dynamic_shapes": "INTEGER",
                "input_dtype": "TEXT",
                "input_color_space": "TEXT",
                "input_shape_source": "TEXT",
                "input_shape_status": "TEXT",
                "final_metrics_json": "TEXT",
                "metadata_json": "TEXT",
                "created_utc": "TEXT",
            },
        )
        self._ensure_columns(
            "onnx_models",
            {
                "set_id": "TEXT",
                "skeleton_id": "TEXT",
                "detection_model_run_id": "TEXT",
                "path": "TEXT",
                "sha256": "TEXT",
                "manifest_path": "TEXT",
                "manifest_sha256": "TEXT",
                "opset": "INTEGER",
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
                "input_shape": "TEXT",
                "img_h": "INTEGER",
                "img_w": "INTEGER",
                "max_batch": "INTEGER",
                "dynamic_shapes": "INTEGER",
                "file_size_bytes": "INTEGER",
                "exporter_torch_version": "TEXT",
                "exporter_cuda_version": "TEXT",
                "exporter_hostname": "TEXT",
                "requires_plugins": "INTEGER",
                "plugin_ops_json": "TEXT",
                "plugin_versions_json": "TEXT",
                "metadata_json": "TEXT",
                "created_utc": "TEXT",
            },
        )
        self._ensure_columns(
            "tensorrt_models",
            {
                "set_id": "TEXT",
                "skeleton_id": "TEXT",
                "detection_model_run_id": "TEXT",
                "onnx_run_id": "TEXT",
                "precision": "TEXT",
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
                "path": "TEXT",
                "sha256": "TEXT",
                "manifest_path": "TEXT",
                "manifest_sha256": "TEXT",
                "input_shape": "TEXT",
                "img_h": "INTEGER",
                "img_w": "INTEGER",
                "max_batch": "INTEGER",
                "dynamic_shapes": "INTEGER",
                "file_size_bytes": "INTEGER",
                "trt_version": "TEXT",
                "cuda_version": "TEXT",
                "compute_capability": "TEXT",
                "gpu_name": "TEXT",
                "gpu_uuid": "TEXT",
                "system_hostname": "TEXT",
                "requires_plugins": "INTEGER",
                "plugin_ops_json": "TEXT",
                "plugin_versions_json": "TEXT",
                "metadata_json": "TEXT",
                "created_utc": "TEXT",
            },
        )

    def _migration_002_reserved_noop(self) -> None:
        # Intentionally no-op. Serves as a stable template slot for future append-only migrations.
        return

    def _migration_003_recording_columns_reconcile(self) -> None:
        if not self._table_exists("recordings"):
            return
        # Legacy bootstrapped registries can skip migration_001 execution.
        # Reconcile additive recording columns needed by current maintenance flows.
        self._ensure_columns(
            "recordings",
            {
                "recording_subtype": "TEXT",
                "behavior_mode": "TEXT",
                "dish_design": "TEXT",
                "experiment_context_status": "TEXT",
                "experiment_context_source": "TEXT",
                "experiment_context_status_detail": "TEXT",
                "stimulus_runs_available": "INTEGER",
            },
        )
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_overview;")
        cur.execute(
            """
            CREATE VIEW recording_overview AS
            SELECT
                r.recording_id AS recording_id,
                r.session_uuid AS session_uuid,
                r.recording_name AS recording_name,
                r.recording_path AS recording_path,
                r.started_utc AS started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.artifact_schema_id AS artifact_schema_id,
                r.experiment_context_status AS experiment_context_status,
                r.experiment_context_source AS experiment_context_source,
                r.experiment_context_status_detail AS experiment_context_status_detail,
                r.stimulus_runs_available AS stimulus_runs_available,
                COALESCE(
                    NULLIF(TRIM(r.dish_design), ''),
                    GROUP_CONCAT(DISTINCT NULLIF(TRIM(dcc.dish_design), ''))
                ) AS dish_design,
                r.rig_id AS rig_id,
                r.arena_id AS arena_id,
                r.camera_id AS camera_id,
                r.protocol_name AS protocol_name,
                COUNT(DISTINCT d.dataset_id) AS dataset_count,
                SUM(CASE WHEN d.zarr_use = 'training' THEN 1 ELSE 0 END) AS training_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'analysis' THEN 1 ELSE 0 END) AS analysis_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'inference' THEN 1 ELSE 0 END) AS inference_dataset_count,
                SUM(CASE WHEN d.zarr_use = 'export' THEN 1 ELSE 0 END) AS export_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, 'active')) = 'active' THEN 1 ELSE 0 END) AS active_dataset_count,
                SUM(CASE WHEN lower(COALESCE(d.status, '')) = 'missing' THEN 1 ELSE 0 END) AS missing_dataset_count,
                COALESCE(MAX(d.last_seen_utc), r.updated_utc, r.created_utc) AS last_seen_utc
            FROM recordings r
            LEFT JOIN datasets d ON d.recording_id = r.recording_id
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = d.dataset_id
            GROUP BY
                r.recording_id,
                r.session_uuid,
                r.recording_name,
                r.recording_path,
                r.started_utc,
                r.recording_type,
                r.recording_subtype,
                r.behavior_mode,
                r.artifact_schema_id,
                r.experiment_context_status,
                r.experiment_context_source,
                r.experiment_context_status_detail,
                r.stimulus_runs_available,
                r.dish_design,
                r.rig_id,
                r.arena_id,
                r.camera_id,
                r.protocol_name;
            """
        )

    def _migration_004_recording_overview_refresh(self) -> None:
        # Refresh view definition for registries that already applied v3.
        self._migration_003_recording_columns_reconcile()

    def _migration_005_drop_provenance_zarr_purpose(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_overview;")
        cur.execute("DROP VIEW IF EXISTS merged_training_datasets;")

        provenance_cols = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(provenance);").fetchall()
            if row["name"] is not None
        }
        if "zarr_purpose" in provenance_cols:
            cur.execute("ALTER TABLE provenance DROP COLUMN zarr_purpose;")

        cur.execute(
            """
            CREATE VIEW keypoint_quality_overview AS
            SELECT
                kqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                kqc.keypoint_method AS keypoint_method,
                kqc.source_keypoint_run AS source_keypoint_run,
                kqc.refined_run AS refined_run,
                kqc.review_state AS review_state,
                kqc.review_method AS review_method,
                kqc.review_intended_use AS review_intended_use,
                kqc.review_policy_id AS review_policy_id,
                kqc.review_policy_version AS review_policy_version,
                kqc.usable_keypoints AS usable_keypoints,
                kqc.total_keypoints AS total_keypoints,
                kqc.usable_keypoints_rate AS usable_keypoints_rate,
                kqc.quality_updated_utc AS quality_updated_utc,
                kqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN kqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM keypoint_quality_current kqc
            LEFT JOIN datasets d ON d.dataset_id = kqc.dataset_id;
            """
        )
        cur.execute(
            """
            CREATE VIEW merged_training_datasets AS
            SELECT
                d.dataset_id,
                d.recording_id,
                d.session_uuid,
                d.zarr_path,
                d.status,
                d.artifact_kind,
                d.zarr_origin,
                d.zarr_use,
                d.zarr_use AS zarr_purpose,
                d.last_seen_utc
            FROM datasets d
            WHERE
                d.artifact_kind = 'derived_training_merge'
                OR d.dataset_id LIKE '%_merged';
            """
        )

    def _migration_006_subject_dish_cross_entities(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crosses (
                cross_id TEXT PRIMARY KEY,
                line_strain TEXT,
                genotype TEXT,
                parents_json TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dishes (
                dish_id TEXT PRIMARY KEY,
                cross_id TEXT,
                species TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                FOREIGN KEY(cross_id) REFERENCES crosses(cross_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_subjects (
                recording_id TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                dataset_id TEXT,
                dish_id TEXT,
                cross_id TEXT,
                dpf_at_acquisition INTEGER,
                species TEXT,
                sex TEXT,
                genotype TEXT,
                line_strain TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                PRIMARY KEY (recording_id, subject_id),
                FOREIGN KEY(recording_id) REFERENCES recordings(recording_id) ON DELETE CASCADE,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE SET NULL,
                FOREIGN KEY(dish_id) REFERENCES dishes(dish_id),
                FOREIGN KEY(cross_id) REFERENCES crosses(cross_id)
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dishes_cross_id ON dishes(cross_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_dataset_id ON recording_subjects(dataset_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_dish_id ON recording_subjects(dish_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_cross_id ON recording_subjects(cross_id);"
        )

    def _migration_007_subjects_entities_and_query_indexes(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subjects (
                subject_id TEXT PRIMARY KEY,
                dish_id TEXT,
                species TEXT,
                sex TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                updated_utc TEXT,
                FOREIGN KEY(dish_id) REFERENCES dishes(dish_id)
            );
            """
        )
        # Common query path: recording_subjects -> subjects -> dishes -> crosses(genotype).
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crosses_genotype ON crosses(genotype);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subjects_dish_id ON subjects(dish_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_subject_dpf ON recording_subjects(subject_id, dpf_at_acquisition);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recording_subjects_recording_id ON recording_subjects(recording_id);"
        )

    def _migration_008_recording_subject_overview_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_subject_overview;")
        cur.execute(
            """
            CREATE VIEW recording_subject_overview AS
            SELECT
                rs.recording_id AS recording_id,
                rs.subject_id AS subject_id,
                rs.dataset_id AS dataset_id,
                COALESCE(rs.dish_id, s.dish_id) AS dish_id,
                COALESCE(rs.cross_id, d.cross_id) AS cross_id,
                c.genotype AS genotype,
                c.line_strain AS line_strain,
                rs.dpf_at_acquisition AS dpf_at_acquisition,
                COALESCE(rs.species, s.species, d.species) AS species,
                COALESCE(rs.sex, s.sex) AS sex,
                r.started_utc AS recording_started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.protocol_name AS protocol_name,
                r.rig_id AS rig_id,
                r.arena_id AS arena_id,
                r.camera_id AS camera_id
            FROM recording_subjects rs
            LEFT JOIN subjects s
              ON s.subject_id = rs.subject_id
            LEFT JOIN dishes d
              ON d.dish_id = COALESCE(rs.dish_id, s.dish_id)
            LEFT JOIN crosses c
              ON c.cross_id = COALESCE(rs.cross_id, d.cross_id)
            LEFT JOIN recordings r
              ON r.recording_id = rs.recording_id;
            """
        )

    def _migration_009_training_task_type_columns(self) -> None:
        self._ensure_columns("training_sets", {"task_type": "TEXT"})
        self._ensure_columns("training_runs", {"task_type": "TEXT"})
        cur = self.conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_sets_task_type ON training_sets(task_type);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_runs_task_type ON training_runs(task_type);"
        )

        set_rows = self.conn.execute(
            "SELECT set_id, task_type, query_filter, invocation_json FROM training_sets;"
        ).fetchall()
        for row in set_rows:
            if _normalize_task_type(row["task_type"]):
                continue
            query_filter = _json_loads(row["query_filter"])
            invocation = _json_loads(row["invocation_json"])
            inferred = _infer_task_type(
                set_id=row["set_id"],
                query_filter=query_filter,
                invocation=invocation,
            )
            if inferred:
                self.conn.execute(
                    "UPDATE training_sets SET task_type = ? WHERE set_id = ?;",
                    (inferred, str(row["set_id"])),
                )

        run_rows = self.conn.execute(
            """
            SELECT
                tr.run_id,
                tr.set_id,
                tr.task_type,
                tr.config_path,
                tr.manifest_path,
                tr.model_path,
                tr.invocation_json,
                ts.task_type AS set_task_type
            FROM training_runs tr
            LEFT JOIN training_sets ts ON ts.set_id = tr.set_id;
            """
        ).fetchall()
        for row in run_rows:
            if _normalize_task_type(row["task_type"]):
                continue
            invocation = _json_loads(row["invocation_json"])
            inferred = _infer_task_type(
                set_id=row["set_id"],
                run_id=row["run_id"],
                config_path=row["config_path"],
                manifest_path=row["manifest_path"],
                model_path=row["model_path"],
                invocation=invocation,
                explicit=row["set_task_type"],
            )
            if inferred:
                self.conn.execute(
                    "UPDATE training_runs SET task_type = ? WHERE run_id = ?;",
                    (inferred, str(row["run_id"])),
                )

    def _migration_010_detect_performance_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detect_performance (
                dataset_id TEXT NOT NULL,
                detect_run TEXT NOT NULL,
                detect_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                detection_method TEXT,
                model_run_id TEXT,
                model_set_id TEXT,
                model_path TEXT,
                model_name TEXT,
                coverage_percent REAL,
                frames_with_detections INTEGER,
                frames_zero_detections INTEGER,
                total_frames INTEGER,
                mean_confidence REAL,
                min_confidence REAL,
                max_confidence REAL,
                inference_duration_seconds REAL,
                inference_average_fps REAL,
                inference_avg_batch_ms REAL,
                inference_avg_read_ms REAL,
                conf_threshold REAL,
                iou_threshold REAL,
                batch_size INTEGER,
                inference_width INTEGER,
                inference_height INTEGER,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, detect_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "detect_performance",
            {
                "detect_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "detection_method": "TEXT",
                "model_run_id": "TEXT",
                "model_set_id": "TEXT",
                "model_path": "TEXT",
                "model_name": "TEXT",
                "coverage_percent": "REAL",
                "frames_with_detections": "INTEGER",
                "frames_zero_detections": "INTEGER",
                "total_frames": "INTEGER",
                "mean_confidence": "REAL",
                "min_confidence": "REAL",
                "max_confidence": "REAL",
                "inference_duration_seconds": "REAL",
                "inference_average_fps": "REAL",
                "inference_avg_batch_ms": "REAL",
                "inference_avg_read_ms": "REAL",
                "conf_threshold": "REAL",
                "iou_threshold": "REAL",
                "batch_size": "INTEGER",
                "inference_width": "INTEGER",
                "inference_height": "INTEGER",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_recording ON detect_performance(recording_id, detect_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_coverage ON detect_performance(coverage_percent);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_runtime ON detect_performance(inference_average_fps, inference_avg_read_ms);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_method ON detect_performance(detection_method, model_name);"
        )
        cur.execute("DROP VIEW IF EXISTS detect_performance_latest;")
        cur.execute(
            """
            CREATE VIEW detect_performance_latest AS
            WITH ranked AS (
                SELECT
                    dp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dp.dataset_id
                        ORDER BY
                            COALESCE(dp.detect_created_utc, dp.updated_utc) DESC,
                            dp.detect_run DESC
                    ) AS _rn
                FROM detect_performance dp
            )
            SELECT
                dataset_id,
                detect_run,
                detect_created_utc,
                recording_id,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_detect_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_detect_performance_latest AS
            WITH ranked AS (
                SELECT
                    dpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY dpl.recording_id
                        ORDER BY
                            COALESCE(dpl.detect_created_utc, dpl.updated_utc) DESC,
                            dpl.detect_run DESC
                    ) AS _rn
                FROM detect_performance_latest dpl
                LEFT JOIN datasets d ON d.dataset_id = dpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = dpl.dataset_id
                WHERE dpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                detect_run,
                detect_created_utc,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_011_detect_model_performance_views(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detect_perf_model_path ON detect_performance(model_path, model_name, detect_created_utc);"
        )

        cur.execute("DROP VIEW IF EXISTS detect_model_performance_latest;")
        cur.execute(
            """
            CREATE VIEW detect_model_performance_latest AS
            WITH model_rows AS (
                SELECT dp.*
                FROM detect_performance dp
                WHERE
                    trim(COALESCE(dp.model_path, '')) <> ''
                    OR trim(COALESCE(dp.model_name, '')) <> ''
            ),
            ranked AS (
                SELECT
                    mr.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY mr.dataset_id
                        ORDER BY
                            COALESCE(mr.detect_created_utc, mr.updated_utc) DESC,
                            mr.detect_run DESC
                    ) AS _rn
                FROM model_rows mr
            )
            SELECT
                dataset_id,
                detect_run,
                detect_created_utc,
                recording_id,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_detect_model_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_detect_model_performance_latest AS
            WITH ranked AS (
                SELECT
                    dmpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY dmpl.recording_id
                        ORDER BY
                            COALESCE(dmpl.detect_created_utc, dmpl.updated_utc) DESC,
                            dmpl.detect_run DESC
                    ) AS _rn
                FROM detect_model_performance_latest dmpl
                LEFT JOIN datasets d ON d.dataset_id = dmpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = dmpl.dataset_id
                WHERE dmpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                detect_run,
                detect_created_utc,
                zarr_use,
                detection_method,
                model_run_id,
                model_set_id,
                model_path,
                model_name,
                coverage_percent,
                frames_with_detections,
                frames_zero_detections,
                total_frames,
                mean_confidence,
                min_confidence,
                max_confidence,
                inference_duration_seconds,
                inference_average_fps,
                inference_avg_batch_ms,
                inference_avg_read_ms,
                conf_threshold,
                iou_threshold,
                batch_size,
                inference_width,
                inference_height,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_012_detect_performance_model_identity(self) -> None:
        # Additive migration: add run/set identity columns and rebuild detect
        # performance views to project them.
        self._ensure_columns(
            "detect_performance",
            {
                "model_run_id": "TEXT",
                "model_set_id": "TEXT",
            },
        )
        self._migration_010_detect_performance_registry()
        self._migration_011_detect_model_performance_views()

    def _create_detect_model_performance_summary_view(self, *, source_view: str, target_view: str) -> None:
        cur = self.conn.cursor()
        cur.execute(f"DROP VIEW IF EXISTS {target_view};")
        cur.execute(
            f"""
            CREATE VIEW {target_view} AS
            WITH base AS (
                SELECT
                    COALESCE(trim(model_run_id), '') AS model_run_id_key,
                    COALESCE(trim(model_set_id), '') AS model_set_id_key,
                    COALESCE(trim(model_name), '') AS model_name_key,
                    COALESCE(trim(model_path), '') AS model_path_key,
                    NULLIF(trim(model_run_id), '') AS model_run_id,
                    NULLIF(trim(model_set_id), '') AS model_set_id,
                    NULLIF(trim(model_name), '') AS model_name,
                    NULLIF(trim(model_path), '') AS model_path,
                    NULLIF(trim(detection_method), '') AS detection_method,
                    dataset_id,
                    recording_id,
                    COALESCE(detect_created_utc, updated_utc) AS detect_created_utc,
                    coverage_percent,
                    inference_average_fps,
                    inference_avg_read_ms
                FROM {source_view}
            ),
            grouped AS (
                SELECT
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key,
                    MIN(model_run_id) AS model_run_id,
                    MIN(model_set_id) AS model_set_id,
                    MIN(model_name) AS model_name,
                    MIN(model_path) AS model_path,
                    GROUP_CONCAT(DISTINCT detection_method) AS detection_methods_csv,
                    COUNT(*) AS detect_rows,
                    COUNT(DISTINCT dataset_id) AS dataset_count,
                    COUNT(DISTINCT recording_id) AS recording_count,
                    MIN(detect_created_utc) AS first_detect_created_utc,
                    MAX(detect_created_utc) AS latest_detect_created_utc,
                    AVG(coverage_percent) AS coverage_avg,
                    MIN(coverage_percent) AS coverage_min,
                    MAX(coverage_percent) AS coverage_max,
                    AVG(inference_average_fps) AS fps_avg,
                    MIN(inference_average_fps) AS fps_min,
                    MAX(inference_average_fps) AS fps_max,
                    AVG(inference_avg_read_ms) AS read_ms_avg,
                    MIN(inference_avg_read_ms) AS read_ms_min,
                    MAX(inference_avg_read_ms) AS read_ms_max
                FROM base
                GROUP BY
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key
            ),
            coverage_pct AS (
                SELECT
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key,
                    MIN(CASE WHEN coverage_cume >= 0.10 THEN coverage_percent END) AS coverage_p10,
                    MIN(CASE WHEN coverage_cume >= 0.50 THEN coverage_percent END) AS coverage_p50,
                    MIN(CASE WHEN coverage_cume >= 0.90 THEN coverage_percent END) AS coverage_p90
                FROM (
                    SELECT
                        model_run_id_key,
                        model_set_id_key,
                        model_name_key,
                        model_path_key,
                        coverage_percent,
                        CUME_DIST() OVER (
                            PARTITION BY
                                model_run_id_key,
                                model_set_id_key,
                                model_name_key,
                                model_path_key
                            ORDER BY coverage_percent
                        ) AS coverage_cume
                    FROM base
                    WHERE coverage_percent IS NOT NULL
                ) ranked
                GROUP BY
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key
            ),
            fps_pct AS (
                SELECT
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key,
                    MIN(CASE WHEN fps_cume >= 0.10 THEN inference_average_fps END) AS fps_p10,
                    MIN(CASE WHEN fps_cume >= 0.50 THEN inference_average_fps END) AS fps_p50,
                    MIN(CASE WHEN fps_cume >= 0.90 THEN inference_average_fps END) AS fps_p90
                FROM (
                    SELECT
                        model_run_id_key,
                        model_set_id_key,
                        model_name_key,
                        model_path_key,
                        inference_average_fps,
                        CUME_DIST() OVER (
                            PARTITION BY
                                model_run_id_key,
                                model_set_id_key,
                                model_name_key,
                                model_path_key
                            ORDER BY inference_average_fps
                        ) AS fps_cume
                    FROM base
                    WHERE inference_average_fps IS NOT NULL
                ) ranked
                GROUP BY
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key
            ),
            read_pct AS (
                SELECT
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key,
                    MIN(CASE WHEN read_cume >= 0.10 THEN inference_avg_read_ms END) AS read_ms_p10,
                    MIN(CASE WHEN read_cume >= 0.50 THEN inference_avg_read_ms END) AS read_ms_p50,
                    MIN(CASE WHEN read_cume >= 0.90 THEN inference_avg_read_ms END) AS read_ms_p90
                FROM (
                    SELECT
                        model_run_id_key,
                        model_set_id_key,
                        model_name_key,
                        model_path_key,
                        inference_avg_read_ms,
                        CUME_DIST() OVER (
                            PARTITION BY
                                model_run_id_key,
                                model_set_id_key,
                                model_name_key,
                                model_path_key
                            ORDER BY inference_avg_read_ms
                        ) AS read_cume
                    FROM base
                    WHERE inference_avg_read_ms IS NOT NULL
                ) ranked
                GROUP BY
                    model_run_id_key,
                    model_set_id_key,
                    model_name_key,
                    model_path_key
            )
            SELECT
                g.model_run_id,
                g.model_set_id,
                g.model_name,
                g.model_path,
                g.detection_methods_csv,
                g.detect_rows,
                g.dataset_count,
                g.recording_count,
                g.first_detect_created_utc,
                g.latest_detect_created_utc,
                g.coverage_avg,
                g.coverage_min,
                g.coverage_max,
                cp.coverage_p10,
                cp.coverage_p50,
                cp.coverage_p90,
                g.fps_avg,
                g.fps_min,
                g.fps_max,
                fp.fps_p10,
                fp.fps_p50,
                fp.fps_p90,
                g.read_ms_avg,
                g.read_ms_min,
                g.read_ms_max,
                rp.read_ms_p10,
                rp.read_ms_p50,
                rp.read_ms_p90
            FROM grouped g
            LEFT JOIN coverage_pct cp
                USING (model_run_id_key, model_set_id_key, model_name_key, model_path_key)
            LEFT JOIN fps_pct fp
                USING (model_run_id_key, model_set_id_key, model_name_key, model_path_key)
            LEFT JOIN read_pct rp
                USING (model_run_id_key, model_set_id_key, model_name_key, model_path_key)
            ORDER BY
                g.recording_count DESC,
                g.dataset_count DESC,
                g.model_set_id,
                g.model_run_id,
                g.model_name,
                g.model_path;
            """
        )

    def _migration_013_detect_model_performance_summary_views(self) -> None:
        # Build additive summary views over model-backed latest detect performance.
        self._migration_011_detect_model_performance_views()
        self._create_detect_model_performance_summary_view(
            source_view="detect_model_performance_latest",
            target_view="detect_model_performance_summary",
        )
        self._create_detect_model_performance_summary_view(
            source_view="recording_detect_model_performance_latest",
            target_view="recording_detect_model_performance_summary",
        )

    def _migration_014_crop_quality_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crop_quality (
                dataset_id TEXT NOT NULL,
                crop_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                crop_created_utc TEXT,
                source_detect_run TEXT,
                source_refined_run TEXT,
                detection_source_type TEXT,
                detection_source_path TEXT,
                total_rois INTEGER,
                frames_with_crops INTEGER,
                total_frames INTEGER,
                percent_frames_with_crops REAL,
                includes_interpolated INTEGER,
                n_real_detections INTEGER,
                n_interpolated_detections INTEGER,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                review_notes TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, crop_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "crop_quality",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "crop_created_utc": "TEXT",
                "source_detect_run": "TEXT",
                "source_refined_run": "TEXT",
                "detection_source_type": "TEXT",
                "detection_source_path": "TEXT",
                "total_rois": "INTEGER",
                "frames_with_crops": "INTEGER",
                "total_frames": "INTEGER",
                "percent_frames_with_crops": "REAL",
                "includes_interpolated": "INTEGER",
                "n_real_detections": "INTEGER",
                "n_interpolated_detections": "INTEGER",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "review_notes": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_crop_quality_dataset_id ON crop_quality(dataset_id);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_quality_review_gate ON crop_quality(review_state, review_intended_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_quality_source ON crop_quality(detection_source_type, source_refined_run);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_crop_quality_recording ON crop_quality(recording_id, crop_created_utc DESC);"
        )

        cur.execute("DROP VIEW IF EXISTS crop_quality_current;")
        cur.execute(
            """
            CREATE VIEW crop_quality_current AS
            WITH ranked AS (
                SELECT
                    cq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY cq.dataset_id
                        ORDER BY
                            COALESCE(cq.review_timestamp_utc, cq.crop_created_utc, cq.updated_utc) DESC,
                            COALESCE(cq.crop_created_utc, '') DESC,
                            cq.crop_run DESC
                    ) AS _rn
                FROM crop_quality cq
            )
            SELECT
                dataset_id,
                crop_run,
                recording_id,
                zarr_use,
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
                review_notes,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_crop_quality_current;")
        cur.execute(
            """
            CREATE VIEW recording_crop_quality_current AS
            WITH ranked AS (
                SELECT
                    cqc.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY cqc.recording_id
                        ORDER BY
                            COALESCE(cqc.review_timestamp_utc, cqc.crop_created_utc, cqc.updated_utc) DESC,
                            COALESCE(cqc.crop_created_utc, '') DESC,
                            cqc.crop_run DESC
                    ) AS _rn
                FROM crop_quality_current cqc
                LEFT JOIN datasets d ON d.dataset_id = cqc.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = cqc.dataset_id
                WHERE cqc.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                crop_run,
                zarr_use,
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
                review_notes,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_015_eye_mask_performance_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eye_mask_performance (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                method TEXT,
                source_crop_run TEXT,
                source_keypoint_group TEXT,
                source_keypoints_run TEXT,
                source_eye_masks_run TEXT,
                source_eye_masks_method TEXT,
                total_rois INTEGER,
                successful_eyes INTEGER,
                successful_roi_pairs INTEGER,
                successful_roi_pair_rate REAL,
                duration_seconds REAL,
                rois_per_second REAL,
                inference_duration_seconds REAL,
                inference_average_fps REAL,
                reason_counts_json TEXT,
                summary_statistics_json TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                source_keypoint_stale_state TEXT,
                source_keypoint_stale_reason TEXT,
                source_keypoint_stale_timestamp_utc TEXT,
                source_keypoint_stale_json TEXT,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, stage_group, run_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "eye_mask_performance",
            {
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "method": "TEXT",
                "source_crop_run": "TEXT",
                "source_keypoint_group": "TEXT",
                "source_keypoints_run": "TEXT",
                "source_eye_masks_run": "TEXT",
                "source_eye_masks_method": "TEXT",
                "total_rois": "INTEGER",
                "successful_eyes": "INTEGER",
                "successful_roi_pairs": "INTEGER",
                "successful_roi_pair_rate": "REAL",
                "duration_seconds": "REAL",
                "rois_per_second": "REAL",
                "inference_duration_seconds": "REAL",
                "inference_average_fps": "REAL",
                "reason_counts_json": "TEXT",
                "summary_statistics_json": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "source_keypoint_stale_state": "TEXT",
                "source_keypoint_stale_reason": "TEXT",
                "source_keypoint_stale_timestamp_utc": "TEXT",
                "source_keypoint_stale_json": "TEXT",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_recording ON eye_mask_performance(recording_id, stage_group, run_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_stage_method ON eye_mask_performance(stage_group, method);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_runtime ON eye_mask_performance(rois_per_second, duration_seconds);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_source ON eye_mask_performance(source_keypoints_run, source_eye_masks_run);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_perf_review ON eye_mask_performance(review_state, review_intended_use, lifecycle_state);"
        )

        cur.execute("DROP VIEW IF EXISTS eye_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW eye_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    emp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY emp.dataset_id, emp.stage_group
                        ORDER BY
                            COALESCE(emp.run_created_utc, emp.updated_utc) DESC,
                            emp.run_name DESC
                    ) AS _rn
                FROM eye_mask_performance emp
            )
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
                reason_counts_json,
                summary_statistics_json,
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
                lifecycle_reason,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_eye_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_eye_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    empl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY empl.recording_id, empl.stage_group
                        ORDER BY
                            COALESCE(empl.run_created_utc, empl.updated_utc) DESC,
                            empl.run_name DESC
                    ) AS _rn
                FROM eye_mask_performance_latest empl
                LEFT JOIN datasets d ON d.dataset_id = empl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = empl.dataset_id
                WHERE empl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
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
                reason_counts_json,
                summary_statistics_json,
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
                lifecycle_reason,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_016_model_export_nms_threshold_columns(self) -> None:
        self._ensure_columns(
            "onnx_models",
            {
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
            },
        )
        self._ensure_columns(
            "tensorrt_models",
            {
                "nms_conf": "REAL",
                "nms_iou": "REAL",
                "nms_topk": "INTEGER",
            },
        )

        # Fast-path backfill from persisted metadata JSON.
        self.conn.execute(
            """
            UPDATE onnx_models
            SET
                nms_conf = COALESCE(
                    nms_conf,
                    json_extract(metadata_json, '$.nms.conf'),
                    json_extract(metadata_json, '$.nms_conf'),
                    json_extract(metadata_json, '$.conf_threshold')
                ),
                nms_iou = COALESCE(
                    nms_iou,
                    json_extract(metadata_json, '$.nms.iou'),
                    json_extract(metadata_json, '$.nms_iou'),
                    json_extract(metadata_json, '$.iou_threshold')
                ),
                nms_topk = COALESCE(
                    nms_topk,
                    json_extract(metadata_json, '$.nms.topk'),
                    json_extract(metadata_json, '$.nms_topk'),
                    json_extract(metadata_json, '$.topk')
                )
            WHERE nms_conf IS NULL OR nms_iou IS NULL OR nms_topk IS NULL;
            """
        )
        self.conn.execute(
            """
            UPDATE tensorrt_models
            SET
                nms_conf = COALESCE(
                    nms_conf,
                    json_extract(metadata_json, '$.nms.conf'),
                    json_extract(metadata_json, '$.nms_conf'),
                    json_extract(metadata_json, '$.conf_threshold')
                ),
                nms_iou = COALESCE(
                    nms_iou,
                    json_extract(metadata_json, '$.nms.iou'),
                    json_extract(metadata_json, '$.nms_iou'),
                    json_extract(metadata_json, '$.iou_threshold')
                ),
                nms_topk = COALESCE(
                    nms_topk,
                    json_extract(metadata_json, '$.nms.topk'),
                    json_extract(metadata_json, '$.nms_topk'),
                    json_extract(metadata_json, '$.topk')
                )
            WHERE nms_conf IS NULL OR nms_iou IS NULL OR nms_topk IS NULL;
            """
        )

        # Slow-path backfill from manifest files when metadata JSON did not include NMS.
        onnx_rows = self.conn.execute(
            """
            SELECT run_id, manifest_path, metadata_json, nms_conf, nms_iou, nms_topk
            FROM onnx_models;
            """
        ).fetchall()
        for row in onnx_rows:
            if (
                row["nms_conf"] is not None
                and row["nms_iou"] is not None
                and row["nms_topk"] is not None
            ):
                continue
            metadata = _json_loads(row["metadata_json"])
            metadata_map = metadata if isinstance(metadata, dict) else None
            manifest_path_text = row["manifest_path"]
            manifest_payload = self._read_json_path(Path(str(manifest_path_text))) if manifest_path_text else {}
            nms_conf, nms_iou, nms_topk = self._extract_nms_thresholds(
                manifest_payload=manifest_payload,
                metadata=metadata_map,
            )
            if nms_conf is None and nms_iou is None and nms_topk is None:
                continue
            self.conn.execute(
                """
                UPDATE onnx_models
                SET
                    nms_conf = COALESCE(nms_conf, ?),
                    nms_iou = COALESCE(nms_iou, ?),
                    nms_topk = COALESCE(nms_topk, ?)
                WHERE run_id = ?;
                """,
                (nms_conf, nms_iou, nms_topk, str(row["run_id"])),
            )

        trt_rows = self.conn.execute(
            """
            SELECT run_id, precision, manifest_path, metadata_json, nms_conf, nms_iou, nms_topk
            FROM tensorrt_models;
            """
        ).fetchall()
        for row in trt_rows:
            if (
                row["nms_conf"] is not None
                and row["nms_iou"] is not None
                and row["nms_topk"] is not None
            ):
                continue
            metadata = _json_loads(row["metadata_json"])
            metadata_map = metadata if isinstance(metadata, dict) else None
            manifest_path_text = row["manifest_path"]
            manifest_payload = self._read_json_path(Path(str(manifest_path_text))) if manifest_path_text else {}
            nms_conf, nms_iou, nms_topk = self._extract_nms_thresholds(
                manifest_payload=manifest_payload,
                metadata=metadata_map,
            )
            if nms_conf is None and nms_iou is None and nms_topk is None:
                continue
            self.conn.execute(
                """
                UPDATE tensorrt_models
                SET
                    nms_conf = COALESCE(nms_conf, ?),
                    nms_iou = COALESCE(nms_iou, ?),
                    nms_topk = COALESCE(nms_topk, ?)
                WHERE run_id = ? AND precision = ?;
                """,
                (
                    nms_conf,
                    nms_iou,
                    nms_topk,
                    str(row["run_id"]),
                    str(row["precision"] or "fp16"),
                ),
            )

    def _migration_017_eye_mask_performance_review_stale_columns(self) -> None:
        # Additive refresh of eye-mask performance schema/views for review + stale reconciliation.
        self._migration_015_eye_mask_performance_registry()

    def _migration_018_keypoint_performance_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keypoint_performance (
                dataset_id TEXT NOT NULL,
                keypoint_run TEXT NOT NULL,
                keypoint_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                keypoint_method TEXT,
                model_run_id TEXT,
                model_set_id TEXT,
                model_path TEXT,
                model_name TEXT,
                source_crop_run TEXT,
                source_detect_run TEXT,
                source_refined_run TEXT,
                total_rois INTEGER,
                successful_detections INTEGER,
                failed_detections INTEGER,
                success_rate_percent REAL,
                frames_with_keypoints INTEGER,
                mean_confidence REAL,
                duration_seconds REAL,
                inference_duration_seconds REAL,
                keypoints_per_second REAL,
                inference_average_fps REAL,
                batch_size INTEGER,
                imgsz TEXT,
                conf_threshold REAL,
                iou_threshold REAL,
                summary_statistics_json TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, keypoint_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "keypoint_performance",
            {
                "keypoint_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "keypoint_method": "TEXT",
                "model_run_id": "TEXT",
                "model_set_id": "TEXT",
                "model_path": "TEXT",
                "model_name": "TEXT",
                "source_crop_run": "TEXT",
                "source_detect_run": "TEXT",
                "source_refined_run": "TEXT",
                "total_rois": "INTEGER",
                "successful_detections": "INTEGER",
                "failed_detections": "INTEGER",
                "success_rate_percent": "REAL",
                "frames_with_keypoints": "INTEGER",
                "mean_confidence": "REAL",
                "duration_seconds": "REAL",
                "inference_duration_seconds": "REAL",
                "keypoints_per_second": "REAL",
                "inference_average_fps": "REAL",
                "batch_size": "INTEGER",
                "imgsz": "TEXT",
                "conf_threshold": "REAL",
                "iou_threshold": "REAL",
                "summary_statistics_json": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_recording ON keypoint_performance(recording_id, keypoint_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_method ON keypoint_performance(keypoint_method, model_name);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_runtime ON keypoint_performance(keypoints_per_second, duration_seconds);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_perf_source ON keypoint_performance(source_crop_run, source_detect_run, source_refined_run);"
        )

        cur.execute("DROP VIEW IF EXISTS keypoint_performance_latest;")
        cur.execute(
            """
            CREATE VIEW keypoint_performance_latest AS
            WITH ranked AS (
                SELECT
                    kp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY kp.dataset_id
                        ORDER BY
                            COALESCE(kp.keypoint_created_utc, kp.updated_utc) DESC,
                            kp.keypoint_run DESC
                    ) AS _rn
                FROM keypoint_performance kp
            )
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
                summary_statistics_json,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_keypoint_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_keypoint_performance_latest AS
            WITH ranked AS (
                SELECT
                    kpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    dcc.cross_id AS cross_id,
                    dcc.genotype AS genotype,
                    dcc.dpf_at_acquisition AS dpf_at_acquisition,
                    ROW_NUMBER() OVER (
                        PARTITION BY kpl.recording_id
                        ORDER BY
                            COALESCE(kpl.keypoint_created_utc, kpl.updated_utc) DESC,
                            kpl.keypoint_run DESC
                    ) AS _rn
                FROM keypoint_performance_latest kpl
                LEFT JOIN datasets d ON d.dataset_id = kpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = kpl.dataset_id
                WHERE kpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                keypoint_run,
                keypoint_created_utc,
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
                summary_statistics_json,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                cross_id,
                genotype,
                dpf_at_acquisition,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_019_recording_step_status_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_step_status (
                dataset_id TEXT NOT NULL,
                recording_id TEXT,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ok', 'missing', 'absent', 'na', 'error')),
                run_name TEXT,
                method TEXT,
                coverage_pct REAL,
                review_status_json TEXT,
                details_json TEXT,
                source TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT NOT NULL,
                PRIMARY KEY (dataset_id, step_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "recording_step_status",
            {
                "recording_id": "TEXT",
                "run_name": "TEXT",
                "method": "TEXT",
                "coverage_pct": "REAL",
                "review_status_json": "TEXT",
                "details_json": "TEXT",
                "source": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_step_status_history (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                recording_id TEXT,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ok', 'missing', 'absent', 'na', 'error')),
                run_name TEXT,
                method TEXT,
                coverage_pct REAL,
                review_status_json TEXT,
                details_json TEXT,
                source TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT NOT NULL,
                recorded_utc TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "recording_step_status_history",
            {
                "recording_id": "TEXT",
                "run_name": "TEXT",
                "method": "TEXT",
                "coverage_pct": "REAL",
                "review_status_json": "TEXT",
                "details_json": "TEXT",
                "source": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "recorded_utc": "TEXT",
            },
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_recording_step
            ON recording_step_status(recording_id, step_name);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_dataset_step
            ON recording_step_status(dataset_id, step_name);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_status
            ON recording_step_status(status);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_history_recording_step
            ON recording_step_status_history(recording_id, step_name, recorded_utc DESC);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recording_step_status_history_dataset_step
            ON recording_step_status_history(dataset_id, step_name, recorded_utc DESC);
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_step_status_latest;")
        cur.execute(
            """
            CREATE VIEW recording_step_status_latest AS
            SELECT
                COALESCE(NULLIF(trim(rss.recording_id), ''), dcc.recording_id) AS recording_id,
                rss.dataset_id,
                dcc.session_uuid AS session_uuid,
                dcc.zarr_path AS zarr_path,
                dcc.zarr_use AS zarr_use,
                dcc.artifact_kind AS artifact_kind,
                dcc.dataset_status AS dataset_status,
                dcc.rig_id AS rig_id,
                dcc.arena_id AS arena_id,
                dcc.camera_id AS camera_id,
                dcc.canvas_name AS canvas_name,
                dcc.dish_design AS dish_design,
                dcc.protocol_name AS protocol_name,
                dcc.cross_id AS cross_id,
                dcc.genotype AS genotype,
                dcc.dpf_at_acquisition AS dpf_at_acquisition,
                rss.step_name,
                rss.status,
                rss.run_name,
                rss.method,
                rss.coverage_pct,
                rss.review_status_json,
                rss.details_json,
                rss.source,
                rss.zarr_mtime_ns,
                rss.updated_utc
            FROM recording_step_status rss
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = rss.dataset_id;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_step_overview;")
        cur.execute(
            """
            CREATE VIEW recording_step_overview AS
            WITH base AS (
                SELECT
                    recording_id,
                    dataset_id,
                    lower(step_name) AS step_name,
                    status,
                    updated_utc
                FROM recording_step_status_latest
                WHERE recording_id IS NOT NULL AND trim(recording_id) <> ''
            ),
            dataset_counts AS (
                SELECT
                    recording_id,
                    COUNT(DISTINCT dataset_id) AS dataset_count,
                    COUNT(*) AS step_rows_total,
                    MAX(updated_utc) AS latest_step_update_utc
                FROM base
                GROUP BY recording_id
            ),
            status_counts AS (
                SELECT
                    recording_id,
                    SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_rows,
                    SUM(CASE WHEN status = 'missing' THEN 1 ELSE 0 END) AS missing_rows,
                    SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) AS absent_rows,
                    SUM(CASE WHEN status = 'na' THEN 1 ELSE 0 END) AS na_rows,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_rows,
                    GROUP_CONCAT(DISTINCT CASE WHEN status IN ('missing', 'absent', 'error') THEN step_name END)
                        AS blocking_steps_csv,
                    GROUP_CONCAT(DISTINCT CASE WHEN status = 'ok' THEN step_name END)
                        AS ok_steps_csv
                FROM base
                GROUP BY recording_id
            ),
            per_step AS (
                SELECT
                    recording_id,
                    SUM(CASE WHEN step_name = 'raw' AND status = 'ok' THEN 1 ELSE 0 END) AS raw_ok_count,
                    SUM(CASE WHEN step_name = 'raw' AND status != 'ok' THEN 1 ELSE 0 END) AS raw_non_ok_count,
                    SUM(CASE WHEN step_name = 'background' AND status = 'ok' THEN 1 ELSE 0 END) AS background_ok_count,
                    SUM(CASE WHEN step_name = 'background' AND status != 'ok' THEN 1 ELSE 0 END) AS background_non_ok_count,
                    SUM(CASE WHEN step_name = 'detect' AND status = 'ok' THEN 1 ELSE 0 END) AS detect_ok_count,
                    SUM(CASE WHEN step_name = 'detect' AND status != 'ok' THEN 1 ELSE 0 END) AS detect_non_ok_count,
                    SUM(CASE WHEN step_name = 'refined_detect' AND status = 'ok' THEN 1 ELSE 0 END) AS refined_detect_ok_count,
                    SUM(CASE WHEN step_name = 'refined_detect' AND status != 'ok' THEN 1 ELSE 0 END) AS refined_detect_non_ok_count,
                    SUM(CASE WHEN step_name = 'crop' AND status = 'ok' THEN 1 ELSE 0 END) AS crop_ok_count,
                    SUM(CASE WHEN step_name = 'crop' AND status != 'ok' THEN 1 ELSE 0 END) AS crop_non_ok_count,
                    SUM(CASE WHEN step_name = 'keypoints' AND status = 'ok' THEN 1 ELSE 0 END) AS keypoints_ok_count,
                    SUM(CASE WHEN step_name = 'keypoints' AND status != 'ok' THEN 1 ELSE 0 END) AS keypoints_non_ok_count,
                    SUM(CASE WHEN step_name = 'refined_keypoints' AND status = 'ok' THEN 1 ELSE 0 END) AS refined_keypoints_ok_count,
                    SUM(CASE WHEN step_name = 'refined_keypoints' AND status != 'ok' THEN 1 ELSE 0 END) AS refined_keypoints_non_ok_count,
                    SUM(CASE WHEN step_name = 'eye_masks' AND status = 'ok' THEN 1 ELSE 0 END) AS eye_masks_ok_count,
                    SUM(CASE WHEN step_name = 'eye_masks' AND status != 'ok' THEN 1 ELSE 0 END) AS eye_masks_non_ok_count,
                    SUM(CASE WHEN step_name = 'refined_eye_masks' AND status = 'ok' THEN 1 ELSE 0 END) AS refined_eye_masks_ok_count,
                    SUM(CASE WHEN step_name = 'refined_eye_masks' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS refined_eye_masks_non_ok_count,
                    SUM(CASE WHEN step_name = 'arena_assignment' AND status = 'ok' THEN 1 ELSE 0 END) AS arena_assignment_ok_count,
                    SUM(CASE WHEN step_name = 'arena_assignment' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS arena_assignment_non_ok_count,
                    SUM(CASE WHEN step_name = 'tracks' AND status = 'ok' THEN 1 ELSE 0 END) AS tracks_ok_count,
                    SUM(CASE WHEN step_name = 'tracks' AND status != 'ok' THEN 1 ELSE 0 END) AS tracks_non_ok_count,
                    SUM(CASE WHEN step_name = 'stimulus' AND status = 'ok' THEN 1 ELSE 0 END) AS stimulus_ok_count,
                    SUM(CASE WHEN step_name = 'stimulus' AND status != 'ok' THEN 1 ELSE 0 END) AS stimulus_non_ok_count,
                    SUM(CASE WHEN step_name = 'calibration' AND status = 'ok' THEN 1 ELSE 0 END) AS calibration_ok_count,
                    SUM(CASE WHEN step_name = 'calibration' AND status != 'ok' THEN 1 ELSE 0 END) AS calibration_non_ok_count,
                    SUM(CASE WHEN step_name = 'dish_mask' AND status = 'ok' THEN 1 ELSE 0 END) AS dish_mask_ok_count,
                    SUM(CASE WHEN step_name = 'dish_mask' AND status != 'ok' THEN 1 ELSE 0 END) AS dish_mask_non_ok_count,
                    SUM(CASE WHEN step_name = 'detection_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS detection_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'detection_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS detection_tuning_non_ok_count,
                    SUM(CASE WHEN step_name = 'keypoint_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS keypoint_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'keypoint_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS keypoint_tuning_non_ok_count,
                    SUM(CASE WHEN step_name = 'eye_mask_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS eye_mask_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'eye_mask_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS eye_mask_tuning_non_ok_count,
                    SUM(CASE WHEN step_name = 'subdish_mask_tuning' AND status = 'ok' THEN 1 ELSE 0 END)
                        AS subdish_mask_tuning_ok_count,
                    SUM(CASE WHEN step_name = 'subdish_mask_tuning' AND status != 'ok' THEN 1 ELSE 0 END)
                        AS subdish_mask_tuning_non_ok_count
                FROM base
                GROUP BY recording_id
            )
            SELECT
                dc.recording_id,
                dc.dataset_count,
                dc.step_rows_total,
                dc.latest_step_update_utc,
                sc.ok_rows,
                sc.missing_rows,
                sc.absent_rows,
                sc.na_rows,
                sc.error_rows,
                sc.blocking_steps_csv,
                sc.ok_steps_csv,
                ps.raw_ok_count,
                ps.raw_non_ok_count,
                ps.background_ok_count,
                ps.background_non_ok_count,
                ps.detect_ok_count,
                ps.detect_non_ok_count,
                ps.refined_detect_ok_count,
                ps.refined_detect_non_ok_count,
                ps.crop_ok_count,
                ps.crop_non_ok_count,
                ps.keypoints_ok_count,
                ps.keypoints_non_ok_count,
                ps.refined_keypoints_ok_count,
                ps.refined_keypoints_non_ok_count,
                ps.eye_masks_ok_count,
                ps.eye_masks_non_ok_count,
                ps.refined_eye_masks_ok_count,
                ps.refined_eye_masks_non_ok_count,
                ps.arena_assignment_ok_count,
                ps.arena_assignment_non_ok_count,
                ps.tracks_ok_count,
                ps.tracks_non_ok_count,
                ps.stimulus_ok_count,
                ps.stimulus_non_ok_count,
                ps.calibration_ok_count,
                ps.calibration_non_ok_count,
                ps.dish_mask_ok_count,
                ps.dish_mask_non_ok_count,
                ps.detection_tuning_ok_count,
                ps.detection_tuning_non_ok_count,
                ps.keypoint_tuning_ok_count,
                ps.keypoint_tuning_non_ok_count,
                ps.eye_mask_tuning_ok_count,
                ps.eye_mask_tuning_non_ok_count,
                ps.subdish_mask_tuning_ok_count,
                ps.subdish_mask_tuning_non_ok_count
            FROM dataset_counts dc
            LEFT JOIN status_counts sc ON sc.recording_id = dc.recording_id
            LEFT JOIN per_step ps ON ps.recording_id = dc.recording_id;
            """
        )

    def _migration_020_recording_step_status_wide_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_step_status_wide;")
        cur.execute(
            f"""
            CREATE VIEW recording_step_status_wide AS
            WITH step_rows AS (
                SELECT
                    dataset_id,
                    COALESCE(NULLIF(trim(recording_id), ''), '') AS recording_id,
                    COALESCE(camera_id, 'unknown') AS camera_id,
                    zarr_path,
                    zarr_use,
                    dataset_status,
                    lower(step_name) AS step_name,
                    lower(status) AS status,
                    run_name,
                    method,
                    coverage_pct,
                    review_status_json,
                    details_json
                FROM recording_step_status_latest
            ),
            pivot AS (
                SELECT
                    dataset_id,
                    MAX(NULLIF(recording_id, '')) AS recording_id,
                    MAX(camera_id) AS camera_id,
                    MAX(zarr_path) AS zarr_path,
                    MAX(zarr_use) AS zarr_use,
                    MAX(dataset_status) AS dataset_status,
                    {_recording_step_status_pivot_columns()},
                    MAX(CASE WHEN step_name = 'raw' THEN details_json END) AS raw_details_json,
                    MAX(CASE WHEN step_name = 'background' THEN details_json END) AS background_details_json,
                    MAX(CASE WHEN step_name = 'detect' THEN method END) AS detect_method,
                    MAX(CASE WHEN step_name = 'detect' THEN coverage_pct END) AS detect_coverage_pct,
                    MAX(CASE WHEN step_name = 'detect' THEN details_json END) AS detect_details_json,
                    MAX(CASE WHEN step_name = 'detect_quality' THEN run_name END) AS detect_quality_run_name,
                    MAX(CASE WHEN step_name = 'detect_quality' THEN details_json END) AS detect_quality_details_json,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN method END) AS refined_detect_method,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN coverage_pct END) AS refined_detect_coverage_pct,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN review_status_json END) AS refined_detect_review_json,
                    MAX(CASE WHEN step_name = 'refined_detect' THEN details_json END) AS refined_detect_details_json,
                    MAX(CASE WHEN step_name = 'crop' THEN review_status_json END) AS crop_review_json,
                    MAX(CASE WHEN step_name = 'crop' THEN details_json END) AS crop_details_json,
                    MAX(CASE WHEN step_name = 'keypoints' THEN details_json END) AS keypoints_details_json,
                    MAX(CASE WHEN step_name = 'refined_keypoints' THEN coverage_pct END) AS refined_keypoints_coverage_pct,
                    MAX(CASE WHEN step_name = 'refined_keypoints' THEN review_status_json END) AS refined_keypoints_review_json,
                    MAX(CASE WHEN step_name = 'refined_keypoints' THEN details_json END) AS refined_keypoints_details_json,
                    MAX(CASE WHEN step_name = 'eye_masks' THEN review_status_json END) AS eye_masks_review_json,
                    MAX(CASE WHEN step_name = 'eye_masks' THEN details_json END) AS eye_masks_details_json,
                    MAX(CASE WHEN step_name = 'refined_eye_masks' THEN review_status_json END) AS refined_eye_masks_review_json,
                    MAX(CASE WHEN step_name = 'refined_eye_masks' THEN details_json END) AS refined_eye_masks_details_json,
                    MAX(CASE WHEN step_name = 'subject_masks' THEN review_status_json END) AS subject_masks_review_json,
                    MAX(CASE WHEN step_name = 'subject_masks' THEN details_json END) AS subject_masks_details_json,
                    MAX(CASE WHEN step_name = 'refined_subject_masks' THEN review_status_json END) AS refined_subject_masks_review_json,
                    MAX(CASE WHEN step_name = 'refined_subject_masks' THEN details_json END) AS refined_subject_masks_details_json,
                    MAX(CASE WHEN step_name = 'tracks' THEN details_json END) AS tracks_details_json,
                    MAX(CASE WHEN step_name = 'track_kinematics' THEN details_json END) AS track_kinematics_details_json,
                    MAX(CASE WHEN step_name = 'swim_bouts' THEN details_json END) AS swim_bouts_details_json,
                    MAX(CASE WHEN step_name = 'bout_kinematics' THEN details_json END) AS bout_kinematics_details_json,
                    MAX(CASE WHEN step_name = 'eye_angles' THEN details_json END) AS eye_angles_details_json,
                    MAX(CASE WHEN step_name = 'subject_shape' THEN details_json END) AS subject_shape_details_json,
                    MAX(CASE WHEN step_name = 'tail_kinematics' THEN details_json END) AS tail_kinematics_details_json,
                    MAX(CASE WHEN step_name = 'tail_posture_view' THEN details_json END) AS tail_posture_view_details_json,
                    MAX(CASE WHEN step_name = 'bout_classification' THEN details_json END) AS bout_classification_details_json,
                    MAX(CASE WHEN step_name = 'stimulus_response' THEN details_json END) AS stimulus_response_details_json,
                    MAX(CASE WHEN step_name = 'stimulus' THEN details_json END) AS stimulus_details_json
                FROM step_rows
                GROUP BY dataset_id
            ),
            derived AS (
                SELECT
                    p.*,
                    COALESCE(
                        json_extract(p.raw_details_json, '$.pipeline_type'),
                        json_extract(p.background_details_json, '$.pipeline_type'),
                        json_extract(p.detect_details_json, '$.pipeline_type'),
                        json_extract(p.refined_detect_details_json, '$.pipeline_type'),
                        json_extract(p.crop_details_json, '$.pipeline_type'),
                        json_extract(p.keypoints_details_json, '$.pipeline_type'),
                        json_extract(p.refined_keypoints_details_json, '$.pipeline_type'),
                        json_extract(p.eye_masks_details_json, '$.pipeline_type'),
                        json_extract(p.refined_eye_masks_details_json, '$.pipeline_type'),
                        json_extract(p.subject_masks_details_json, '$.pipeline_type'),
                        json_extract(p.refined_subject_masks_details_json, '$.pipeline_type'),
                        json_extract(p.track_kinematics_details_json, '$.pipeline_type'),
                        json_extract(p.swim_bouts_details_json, '$.pipeline_type'),
                        json_extract(p.bout_kinematics_details_json, '$.pipeline_type'),
                        json_extract(p.eye_angles_details_json, '$.pipeline_type'),
                        json_extract(p.subject_shape_details_json, '$.pipeline_type'),
                        json_extract(p.tail_kinematics_details_json, '$.pipeline_type'),
                        json_extract(p.tail_posture_view_details_json, '$.pipeline_type'),
                        json_extract(p.bout_classification_details_json, '$.pipeline_type'),
                        json_extract(p.stimulus_response_details_json, '$.pipeline_type')
                    ) AS pipeline_type,
                    COALESCE(
                        json_extract(p.raw_details_json, '$.zarr_purpose'),
                        json_extract(p.background_details_json, '$.zarr_purpose'),
                        json_extract(p.detect_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_detect_details_json, '$.zarr_purpose'),
                        json_extract(p.crop_details_json, '$.zarr_purpose'),
                        json_extract(p.keypoints_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_keypoints_details_json, '$.zarr_purpose'),
                        json_extract(p.eye_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_eye_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.subject_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.refined_subject_masks_details_json, '$.zarr_purpose'),
                        json_extract(p.track_kinematics_details_json, '$.zarr_purpose'),
                        json_extract(p.swim_bouts_details_json, '$.zarr_purpose'),
                        json_extract(p.bout_kinematics_details_json, '$.zarr_purpose'),
                        json_extract(p.eye_angles_details_json, '$.zarr_purpose'),
                        json_extract(p.subject_shape_details_json, '$.zarr_purpose'),
                        json_extract(p.tail_kinematics_details_json, '$.zarr_purpose'),
                        json_extract(p.tail_posture_view_details_json, '$.zarr_purpose'),
                        json_extract(p.bout_classification_details_json, '$.zarr_purpose'),
                        json_extract(p.stimulus_response_details_json, '$.zarr_purpose')
                    ) AS zarr_purpose,
                    COALESCE(
                        json_extract(p.raw_details_json, '$.has_raw_video_attr'),
                        json_extract(p.background_details_json, '$.has_raw_video_attr'),
                        json_extract(p.detect_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_detect_details_json, '$.has_raw_video_attr'),
                        json_extract(p.crop_details_json, '$.has_raw_video_attr'),
                        json_extract(p.keypoints_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_keypoints_details_json, '$.has_raw_video_attr'),
                        json_extract(p.eye_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_eye_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.subject_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.refined_subject_masks_details_json, '$.has_raw_video_attr'),
                        json_extract(p.track_kinematics_details_json, '$.has_raw_video_attr'),
                        json_extract(p.swim_bouts_details_json, '$.has_raw_video_attr'),
                        json_extract(p.bout_kinematics_details_json, '$.has_raw_video_attr'),
                        json_extract(p.eye_angles_details_json, '$.has_raw_video_attr'),
                        json_extract(p.subject_shape_details_json, '$.has_raw_video_attr'),
                        json_extract(p.tail_kinematics_details_json, '$.has_raw_video_attr'),
                        json_extract(p.tail_posture_view_details_json, '$.has_raw_video_attr'),
                        json_extract(p.bout_classification_details_json, '$.has_raw_video_attr'),
                        json_extract(p.stimulus_response_details_json, '$.has_raw_video_attr')
                    ) AS has_raw_video_attr,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.raw_details_json, '$.raw_present'),
                            CASE WHEN p.raw_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS raw_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.raw_details_json, '$.full_present'),
                            CASE WHEN p.raw_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS full_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.raw_details_json, '$.ds_present'),
                            CASE WHEN p.raw_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS ds_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.background_details_json, '$.full_present'),
                            CASE WHEN p.background_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS background_full_present,
                    CASE
                        WHEN COALESCE(
                            json_extract(p.background_details_json, '$.ds_present'),
                            CASE WHEN p.background_status = 'ok' THEN 1 ELSE 0 END
                        ) = 1 THEN 1 ELSE 0
                    END AS background_ds_present,
                    CASE WHEN p.detect_status = 'ok' THEN 1 ELSE 0 END AS detect_present,
                    CASE
                        WHEN p.refined_detect_status = 'ok' AND p.refined_detect_coverage_pct IS NULL THEN 100.0
                        ELSE p.refined_detect_coverage_pct
                    END AS refined_detect_coverage_effective,
                    CASE WHEN p.keypoints_status = 'ok' THEN 1 ELSE 0 END AS keypoints_present,
                    CASE WHEN p.refined_keypoints_status = 'ok' THEN 1 ELSE 0 END AS refined_keypoints_present,
                    CASE
                        WHEN p.refined_keypoints_status = 'ok' AND p.refined_keypoints_coverage_pct IS NULL THEN 100.0
                        ELSE p.refined_keypoints_coverage_pct
                    END AS refined_keypoints_success_effective,
                    CASE WHEN p.eye_masks_status = 'ok' THEN 1 ELSE 0 END AS eye_masks_present,
                    CASE WHEN p.refined_eye_masks_status = 'ok' THEN 1 ELSE 0 END AS refined_eye_masks_present,
                    CASE WHEN p.subject_masks_status = 'ok' THEN 1 ELSE 0 END AS subject_masks_present,
                    CASE WHEN p.refined_subject_masks_status = 'ok' THEN 1 ELSE 0 END AS refined_subject_masks_present,
                    CASE WHEN p.arena_assignment_status = 'ok' THEN 1 ELSE 0 END AS arena_assignment_present,
                    CASE WHEN p.tracks_status = 'ok' THEN 1 ELSE 0 END AS track_present,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.n_unassigned_rows'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.n_unassigned_rows')
                        ) AS INTEGER
                    ) AS track_unassigned_rows,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.unassigned_row_rate_percent'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.unassigned_row_rate_percent')
                        ) AS REAL
                    ) AS track_unassigned_rate_percent,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_warn_threshold_rows'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_warn_threshold_rows'),
                            1
                        ) AS INTEGER
                    ) AS track_warn_threshold_rows,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_warn_threshold_percent'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_warn_threshold_percent'),
                            0.0
                        ) AS REAL
                    ) AS track_warn_threshold_percent,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_block_threshold_rows'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_block_threshold_rows'),
                            10
                        ) AS INTEGER
                    ) AS track_block_threshold_rows,
                    CAST(
                        COALESCE(
                            json_extract(p.tracks_details_json, '$.tracking_block_threshold_percent'),
                            json_extract(p.tracks_details_json, '$.summary_statistics.tracking_block_threshold_percent'),
                            1.0
                        ) AS REAL
                    ) AS track_block_threshold_percent,
                    CASE WHEN p.calibration_status = 'ok' THEN 1 ELSE 0 END AS calibration_present,
                    CAST(
                        COALESCE(
                            json_extract(p.stimulus_details_json, '$.stimulus_runs'),
                            CASE WHEN p.stimulus_status = 'ok' THEN 1 ELSE 0 END,
                            0
                        ) AS INTEGER
                    ) AS stimulus_runs,
                    COALESCE(
                        NULLIF(TRIM(CAST(json_extract(p.crop_details_json, '$.run_state') AS TEXT)), ''),
                        NULL
                    ) AS crop_run_state,
                    CAST(
                        COALESCE(
                            json_extract(p.refined_keypoints_details_json, '$.usable_keypoints_pct'),
                            json_extract(p.refined_keypoints_details_json, '$.usable_percent'),
                            json_extract(p.refined_keypoints_details_json, '$.train_usable_pct')
                        ) AS REAL
                    ) AS refined_keypoints_train_usable_pct,
                    COALESCE(
                        CAST(json_extract(p.detect_quality_details_json, '$.quality_grade') AS TEXT),
                        CAST(json_extract(p.detect_details_json, '$.detect_quality_grade') AS TEXT),
                        CAST(json_extract(p.detect_details_json, '$.grade') AS TEXT)
                    ) AS detect_quality_grade,
                    CAST(
                        COALESCE(
                            json_extract(p.detect_quality_details_json, '$.quality_score'),
                            json_extract(p.detect_details_json, '$.detect_quality_score'),
                            json_extract(p.detect_details_json, '$.score')
                        ) AS REAL
                    ) AS detect_quality_score,
                    CAST(
                        COALESCE(
                            json_extract(p.detect_quality_details_json, '$.clean_percentage'),
                            json_extract(p.detect_details_json, '$.detect_quality_clean_percent'),
                            json_extract(p.detect_details_json, '$.clean_percent'),
                            json_extract(p.detect_details_json, '$.clean_percentage')
                        ) AS REAL
                    ) AS detect_quality_clean_percent,
                    CAST(
                        COALESCE(
                            json_extract(p.detect_details_json, '$.detect_quality_artifacts'),
                            json_extract(p.detect_details_json, '$.artifact_count')
                        ) AS INTEGER
                    ) AS detect_quality_artifacts,
                    COALESCE(p.refined_eye_masks_review_json, p.eye_masks_review_json) AS eye_mask_review_json
                FROM pivot p
            ),
            render AS (
                SELECT
                    d.*,
                    CASE
                        WHEN lower(COALESCE(CAST(d.zarr_purpose AS TEXT), '')) = 'production' THEN 1
                        WHEN lower(COALESCE(CAST(d.pipeline_type AS TEXT), '')) = 'yolo_inference' THEN 1
                        WHEN d.has_raw_video_attr = 0 AND NOT (d.full_present = 1 OR d.ds_present = 1) THEN 1
                        ELSE 0
                    END AS is_production,
                    ({_recording_tuning_ok_count_sql("d")}) AS tuning_ok_count,
                    {len(recording_tuning_stage_ids())} AS tuning_total,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.state') AS TEXT)), '') AS detect_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.method') AS TEXT)), '') AS detect_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.intended_use') AS TEXT)), '') AS detect_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.resolved_group') AS TEXT)), '') AS detect_review_group,
                    COALESCE(
                        NULLIF(TRIM(CAST(json_extract(d.refined_detect_review_json, '$.resolved_group') AS TEXT)), ''),
                        NULLIF(TRIM(CAST(json_extract(d.refined_detect_details_json, '$.resolved_group') AS TEXT)), '')
                    ) AS detect_group,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.state') AS TEXT)), '') AS crop_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.method') AS TEXT)), '') AS crop_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.intended_use') AS TEXT)), '') AS crop_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.crop_review_json, '$.resolved_group') AS TEXT)), '') AS crop_review_group,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.state') AS TEXT)), '') AS keypoint_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.method') AS TEXT)), '') AS keypoint_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.intended_use') AS TEXT)), '') AS keypoint_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.refined_keypoints_review_json, '$.resolved_group') AS TEXT)), '') AS keypoint_review_group,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.state') AS TEXT)), '') AS eye_review_state,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.method') AS TEXT)), '') AS eye_review_method,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.intended_use') AS TEXT)), '') AS eye_review_use,
                    NULLIF(TRIM(CAST(json_extract(d.eye_mask_review_json, '$.resolved_group') AS TEXT)), '') AS eye_review_group,
                    CASE
                        WHEN d.detect_quality_grade IS NOT NULL AND d.detect_quality_score IS NOT NULL
                            THEN d.detect_quality_grade || ' ' || printf('%.1f', d.detect_quality_score)
                        WHEN d.detect_quality_grade IS NOT NULL
                            THEN d.detect_quality_grade
                        WHEN d.detect_quality_score IS NOT NULL
                            THEN printf('%.1f', d.detect_quality_score)
                        ELSE ''
                    END AS detect_quality_head,
                    COALESCE(
                        NULLIF(
                            lower(trim(CAST(json_extract(d.tracks_details_json, '$.tracking_qc_state') AS TEXT))),
                            ''
                        ),
                        CASE
                            WHEN COALESCE(d.track_unassigned_rows, 0) <= 0 THEN 'ok'
                            WHEN d.track_unassigned_rows >= COALESCE(d.track_warn_threshold_rows, 1)
                                OR COALESCE(d.track_unassigned_rate_percent, 0.0) > COALESCE(d.track_warn_threshold_percent, 0.0)
                                THEN 'warn'
                            ELSE 'ok'
                        END
                    ) AS track_qc_state
                FROM derived d
            )
            SELECT
                COALESCE(r.recording_id, r.dataset_id) AS "Recording",
                COALESCE(r.camera_id, 'unknown') AS "Camera",
                CASE
                    WHEN lower(COALESCE(CAST(r.dataset_status AS TEXT), '')) = 'missing' THEN 'MISS'
                    ELSE 'OK'
                END AS "Zarr",
                COALESCE(NULLIF(CAST(r.zarr_use AS TEXT), ''), '—') AS "Use",
                COALESCE(NULLIF(CAST(r.zarr_purpose AS TEXT), ''), '—') AS "Purpose",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.raw_present = 1 AND (r.full_present = 1 OR r.ds_present = 1) THEN 'OK'
                    ELSE 'MISS'
                END AS "Import",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.background_full_present = 1 THEN 'OK'
                    ELSE 'MISS'
                END AS "BG Full",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.background_ds_present = 1 THEN 'OK'
                    ELSE 'MISS'
                END AS "BG DS",
                CASE
                    WHEN r.detect_present != 1 THEN 'MISS'
                    WHEN r.detect_coverage_pct IS NULL AND r.detect_method IS NOT NULL THEN 'OK (' || r.detect_method || ')'
                    WHEN r.detect_coverage_pct IS NULL THEN 'OK'
                    WHEN r.detect_method IS NOT NULL THEN
                        'OK ('
                        || CASE
                            WHEN r.detect_coverage_pct >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.detect_coverage_pct)
                        END
                        || ', registry, '
                        || r.detect_method
                        || ')'
                    ELSE
                        'OK ('
                        || CASE
                            WHEN r.detect_coverage_pct >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.detect_coverage_pct)
                        END
                        || ', registry)'
                END AS "Detect",
                CASE
                    WHEN r.detect_quality_head = '' AND r.detect_quality_clean_percent IS NULL AND r.detect_quality_artifacts IS NULL
                        THEN 'MISS'
                    ELSE
                        CASE
                            WHEN (
                                r.detect_quality_head
                                || CASE
                                    WHEN r.detect_quality_clean_percent IS NOT NULL THEN
                                        CASE WHEN r.detect_quality_head <> '' THEN ', ' ELSE '' END
                                        || 'clean '
                                        || printf('%.1f%%', r.detect_quality_clean_percent)
                                    ELSE ''
                                END
                                || CASE
                                    WHEN r.detect_quality_artifacts IS NOT NULL THEN
                                        CASE
                                            WHEN r.detect_quality_head <> '' OR r.detect_quality_clean_percent IS NOT NULL THEN ', '
                                            ELSE ''
                                        END
                                        || 'art '
                                        || CAST(r.detect_quality_artifacts AS TEXT)
                                    ELSE ''
                                END
                            ) = '' THEN 'OK'
                            ELSE
                                'OK ('
                                || (
                                    r.detect_quality_head
                                    || CASE
                                        WHEN r.detect_quality_clean_percent IS NOT NULL THEN
                                            CASE WHEN r.detect_quality_head <> '' THEN ', ' ELSE '' END
                                            || 'clean '
                                            || printf('%.1f%%', r.detect_quality_clean_percent)
                                        ELSE ''
                                    END
                                    || CASE
                                        WHEN r.detect_quality_artifacts IS NOT NULL THEN
                                            CASE
                                                WHEN r.detect_quality_head <> '' OR r.detect_quality_clean_percent IS NOT NULL THEN ', '
                                                ELSE ''
                                            END
                                            || 'art '
                                            || CAST(r.detect_quality_artifacts AS TEXT)
                                        ELSE ''
                                    END
                                )
                                || ')'
                        END
                END AS "Detect Quality",
                CASE
                    WHEN r.refined_detect_coverage_effective IS NULL THEN 'MISS'
                    WHEN r.refined_detect_method IS NOT NULL THEN
                        CASE
                            WHEN r.refined_detect_coverage_effective >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.refined_detect_coverage_effective)
                        END
                        || ' ('
                        || r.refined_detect_method
                        || ')'
                    ELSE
                        CASE
                            WHEN r.refined_detect_coverage_effective >= 99.999 THEN '100%'
                            ELSE printf('%.1f%%', r.refined_detect_coverage_effective)
                        END
                END AS "Refine Detect",
                COALESCE(r.detect_group, '—') AS "Detect Group",
                CASE
                    WHEN r.detect_review_state IS NULL
                        AND r.detect_review_method IS NULL
                        AND r.detect_review_use IS NULL
                        AND r.detect_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.detect_review_method, '') <> ''
                        OR COALESCE(r.detect_review_use, '') <> ''
                        OR COALESCE(r.detect_review_group, '') <> ''
                        THEN
                            COALESCE(r.detect_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.detect_review_method, '') <> '' THEN r.detect_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.detect_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.detect_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.detect_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.detect_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.detect_review_method, '') <> '' OR COALESCE(r.detect_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.detect_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.detect_review_state IS NOT NULL THEN r.detect_review_state
                    ELSE '—'
                END AS "Detect Review",
                CASE
                    WHEN r.crop_status = 'ok'
                        THEN COALESCE(NULLIF(lower(COALESCE(r.crop_run_state, '')), ''), 'OK')
                    WHEN r.crop_status = 'error' THEN 'failed'
                    WHEN r.crop_status = 'na' THEN 'na'
                    ELSE 'MISS'
                END AS "Crop",
                CASE
                    WHEN r.crop_review_state IS NULL
                        AND r.crop_review_method IS NULL
                        AND r.crop_review_use IS NULL
                        AND r.crop_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.crop_review_method, '') <> ''
                        OR COALESCE(r.crop_review_use, '') <> ''
                        OR COALESCE(r.crop_review_group, '') <> ''
                        THEN
                            COALESCE(r.crop_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.crop_review_method, '') <> '' THEN r.crop_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.crop_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.crop_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.crop_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.crop_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.crop_review_method, '') <> '' OR COALESCE(r.crop_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.crop_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.crop_review_state IS NOT NULL THEN r.crop_review_state
                    ELSE '—'
                END AS "Crop Review",
                CASE WHEN r.keypoints_present = 1 THEN 'OK' ELSE 'MISS' END AS "Keypoints",
                CASE
                    WHEN r.refined_keypoints_success_effective IS NULL
                        AND r.refined_keypoints_train_usable_pct IS NULL
                        THEN 'MISS'
                    ELSE
                        COALESCE(
                            CASE
                                WHEN r.refined_keypoints_success_effective >= 99.999 THEN '100%'
                                WHEN r.refined_keypoints_success_effective IS NOT NULL
                                    THEN printf('%.1f%%', r.refined_keypoints_success_effective)
                                ELSE NULL
                            END,
                            '—'
                        )
                        || CASE
                            WHEN r.refined_keypoints_train_usable_pct IS NOT NULL THEN
                                ' (train '
                                || CASE
                                    WHEN r.refined_keypoints_train_usable_pct >= 99.999 THEN '100%'
                                    ELSE printf('%.1f%%', r.refined_keypoints_train_usable_pct)
                                END
                                || ')'
                            ELSE ''
                        END
                END AS "Refined Keypoints (analysis/train)",
                CASE
                    WHEN r.keypoint_review_state IS NULL
                        AND r.keypoint_review_method IS NULL
                        AND r.keypoint_review_use IS NULL
                        AND r.keypoint_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.keypoint_review_method, '') <> ''
                        OR COALESCE(r.keypoint_review_use, '') <> ''
                        OR COALESCE(r.keypoint_review_group, '') <> ''
                        THEN
                            COALESCE(r.keypoint_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.keypoint_review_method, '') <> '' THEN r.keypoint_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.keypoint_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.keypoint_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.keypoint_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.keypoint_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.keypoint_review_method, '') <> '' OR COALESCE(r.keypoint_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.keypoint_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.keypoint_review_state IS NOT NULL THEN r.keypoint_review_state
                    ELSE '—'
                END AS "Keypoint Review",
                CASE WHEN r.eye_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Eye Masks",
                CASE WHEN r.refined_eye_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Refined Eye Masks",
                CASE WHEN r.subject_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Subject Masks",
                CASE WHEN r.refined_subject_masks_present = 1 THEN 'OK' ELSE 'MISS' END AS "Refined Subject Masks",
                CASE
                    WHEN r.eye_review_state IS NULL
                        AND r.eye_review_method IS NULL
                        AND r.eye_review_use IS NULL
                        AND r.eye_review_group IS NULL
                        THEN '—'
                    WHEN COALESCE(r.eye_review_method, '') <> ''
                        OR COALESCE(r.eye_review_use, '') <> ''
                        OR COALESCE(r.eye_review_group, '') <> ''
                        THEN
                            COALESCE(r.eye_review_state, 'review')
                            || ' ('
                            || CASE WHEN COALESCE(r.eye_review_method, '') <> '' THEN r.eye_review_method ELSE '' END
                            || CASE
                                WHEN COALESCE(r.eye_review_use, '') <> '' THEN
                                    CASE WHEN COALESCE(r.eye_review_method, '') <> '' THEN ', ' ELSE '' END
                                    || r.eye_review_use
                                ELSE ''
                            END
                            || CASE
                                WHEN COALESCE(r.eye_review_group, '') <> '' THEN
                                    CASE
                                        WHEN COALESCE(r.eye_review_method, '') <> '' OR COALESCE(r.eye_review_use, '') <> ''
                                            THEN ', '
                                        ELSE ''
                                    END
                                    || 'group='
                                    || r.eye_review_group
                                ELSE ''
                            END
                            || ')'
                    WHEN r.eye_review_state IS NOT NULL THEN r.eye_review_state
                    ELSE '—'
                END AS "Eye Mask Review",
                CASE WHEN r.arena_assignment_present = 1 THEN 'OK' ELSE 'MISS' END AS "Arena Assignment",
                CASE
                    WHEN r.track_present != 1 THEN 'MISS'
                    WHEN lower(COALESCE(r.track_qc_state, '')) IN ('warn', 'block') AND r.track_unassigned_rate_percent IS NOT NULL THEN
                        'WARN ('
                        || CAST(r.track_unassigned_rows AS TEXT)
                        || ' unassigned, '
                        || printf('%.1f%%', r.track_unassigned_rate_percent)
                        || ')'
                    WHEN lower(COALESCE(r.track_qc_state, '')) IN ('warn', 'block') AND r.track_unassigned_rows IS NOT NULL THEN
                        'WARN ('
                        || CAST(r.track_unassigned_rows AS TEXT)
                        || ' unassigned)'
                    WHEN lower(COALESCE(r.track_qc_state, '')) IN ('warn', 'block') THEN 'WARN'
                    ELSE 'OK'
                END AS "Track",
                CASE
                    WHEN r.track_kinematics_status = 'ok' THEN 'OK'
                    WHEN r.track_kinematics_status = 'na' THEN 'N/A'
                    WHEN r.track_kinematics_status = 'error' THEN 'ERR'
                    ELSE 'MISS'
                END AS "Track Kinematics",
                CASE
                    WHEN r.swim_bouts_status = 'ok' THEN 'OK'
                    WHEN r.swim_bouts_status = 'na' THEN 'N/A'
                    WHEN r.swim_bouts_status = 'error' THEN 'ERR'
                    ELSE 'MISS'
                END AS "Swim Bouts",
                {_recording_step_status_display_sql("r.bout_kinematics_status", "r.bout_kinematics_details_json")} AS "Bout Kinematics",
                {_recording_step_status_display_sql("r.eye_angles_status", "r.eye_angles_details_json")} AS "Eye Angles",
                {_recording_step_status_display_sql("r.subject_shape_status", "r.subject_shape_details_json")} AS "Subject Shape",
                {_recording_step_status_display_sql("r.tail_kinematics_status", "r.tail_kinematics_details_json")} AS "Tail Kinematics",
                {_recording_step_status_display_sql("r.tail_posture_view_status", "r.tail_posture_view_details_json")} AS "Tail Posture View",
                {_recording_step_status_display_sql("r.bout_classification_status", "r.bout_classification_details_json")} AS "Bout Classification",
                {_recording_step_status_display_sql("r.stimulus_response_status", "r.stimulus_response_details_json")} AS "Stimulus Response",
                CAST(r.stimulus_runs AS TEXT) || ' (' || CASE WHEN r.stimulus_runs > 0 THEN 'OK' ELSE 'MISS' END || ')' AS "Stimulus",
                CASE WHEN r.calibration_present = 1 THEN 'OK' ELSE 'MISS' END AS "Calib",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    ELSE CAST(r.tuning_ok_count AS TEXT) || '/' || CAST(r.tuning_total AS TEXT)
                END AS "Tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.dish_mask_status = 'ok' THEN 'OK'
                    WHEN r.dish_mask_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "dish_mask",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.detection_tuning_status = 'ok' THEN 'OK'
                    WHEN r.detection_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "detection_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.keypoint_tuning_status = 'ok' THEN 'OK'
                    WHEN r.keypoint_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "keypoint_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.subject_mask_tuning_status = 'ok' THEN 'OK'
                    WHEN r.subject_mask_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "subject_mask_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.eye_mask_tuning_status = 'ok' THEN 'OK'
                    WHEN r.eye_mask_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "eye_mask_tuning",
                CASE
                    WHEN r.is_production = 1 THEN 'N/A'
                    WHEN r.subdish_mask_tuning_status = 'ok' THEN 'OK'
                    WHEN r.subdish_mask_tuning_status = 'na' THEN 'N/A'
                    ELSE 'MISS'
                END AS "subdish_mask_tuning"
            FROM render r;
            """
        )

    def _refresh_keypoint_quality_current_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_current;")
        cur.execute(
            """
            CREATE VIEW keypoint_quality_current AS
            WITH latest_keypoint AS (
                SELECT
                    dataset_id,
                    keypoint_run AS latest_keypoint_run
                FROM keypoint_performance_latest
            ),
            ranked AS (
                SELECT
                    kq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY kq.dataset_id, COALESCE(kq.keypoint_method, '')
                        ORDER BY
                            CASE
                                WHEN lk.latest_keypoint_run IS NOT NULL
                                    AND COALESCE(kq.source_keypoint_run, '') = COALESCE(lk.latest_keypoint_run, '')
                                THEN 0
                                ELSE 1
                            END,
                            COALESCE(kq.review_timestamp_utc, kq.refined_created_utc, kq.quality_updated_utc) DESC,
                            COALESCE(kq.refined_created_utc, '') DESC,
                            kq.refined_run DESC
                    ) AS _rn
                FROM keypoint_quality kq
                LEFT JOIN latest_keypoint lk ON lk.dataset_id = kq.dataset_id
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_keypoint_run,
                keypoint_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_policy_id,
                review_policy_version,
                review_timestamp_utc,
                usable_keypoints,
                total_keypoints,
                usable_keypoints_rate,
                raw_keypoints_success_rate,
                raw_keypoints_successful,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_021_detect_keypoint_quality_review_columns(self) -> None:
        # Additive migration for shared detect/keypoint review fields in quality tables.
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
                "review_policy_id": "TEXT",
                "review_policy_version": "INTEGER",
            },
        )
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        cur = self.conn.cursor()
        self._refresh_keypoint_quality_current_view()
        cur.execute("DROP VIEW IF EXISTS detect_quality_current;")
        cur.execute(
            """
            CREATE VIEW detect_quality_current AS
            WITH ranked AS (
                SELECT
                    dq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dq.dataset_id, COALESCE(dq.detect_method, '')
                        ORDER BY
                            COALESCE(dq.review_timestamp_utc, dq.refined_created_utc, dq.quality_updated_utc) DESC,
                            COALESCE(dq.refined_created_utc, '') DESC,
                            dq.refined_run DESC
                    ) AS _rn
                FROM detect_quality dq
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_detect_run,
                detect_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_timestamp_utc,
                review_resolved_group,
                total_detections,
                real_detections,
                interpolated_detections,
                interpolated_detections_rate,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS refined_detect_review_current;")
        cur.execute(
            """
            CREATE VIEW refined_detect_review_current AS
            SELECT * FROM detect_quality_current;
            """
        )

    def _refresh_detect_quality_current_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS detect_quality_current;")
        cur.execute(
            """
            CREATE VIEW detect_quality_current AS
            WITH ranked AS (
                SELECT
                    dq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY dq.dataset_id, COALESCE(dq.detect_method, '')
                        ORDER BY
                            CASE
                                WHEN dq.review_state IS NOT NULL
                                  OR dq.review_intended_use IS NOT NULL
                                  OR dq.review_timestamp_utc IS NOT NULL
                                THEN 0
                                ELSE 1
                            END ASC,
                            COALESCE(dq.review_timestamp_utc, dq.refined_created_utc, dq.quality_updated_utc) DESC,
                            COALESCE(dq.refined_created_utc, '') DESC,
                            dq.refined_run DESC
                    ) AS _rn
                FROM detect_quality dq
            )
            SELECT
                dataset_id,
                refined_run,
                refined_created_utc,
                source_detect_run,
                detect_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_notes,
                review_timestamp_utc,
                review_resolved_group,
                total_detections,
                real_detections,
                interpolated_detections,
                interpolated_detections_rate,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS refined_detect_review_current;")
        cur.execute(
            """
            CREATE VIEW refined_detect_review_current AS
            SELECT * FROM detect_quality_current;
            """
        )

    def _migration_022_detection_data_profile_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                detection_type TEXT,
                detection_path TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                frames_total INTEGER,
                frames_with_detections INTEGER,
                coverage_percent REAL,
                detections_total INTEGER,
                detections_per_frame_p50 REAL,
                detections_per_frame_p90 REAL,
                w_p10 REAL,
                w_p50 REAL,
                w_p90 REAL,
                h_p10 REAL,
                h_p50 REAL,
                h_p90 REAL,
                area_p10 REAL,
                area_p50 REAL,
                area_p90 REAL,
                aspect_ratio_p10 REAL,
                aspect_ratio_p50 REAL,
                aspect_ratio_p90 REAL,
                edge_proximity_rate REAL,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "detection_data_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "detection_type": "TEXT",
                "detection_path": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "frames_total": "INTEGER",
                "frames_with_detections": "INTEGER",
                "coverage_percent": "REAL",
                "detections_total": "INTEGER",
                "detections_per_frame_p50": "REAL",
                "detections_per_frame_p90": "REAL",
                "w_p10": "REAL",
                "w_p50": "REAL",
                "w_p90": "REAL",
                "h_p10": "REAL",
                "h_p50": "REAL",
                "h_p90": "REAL",
                "area_p10": "REAL",
                "area_p50": "REAL",
                "area_p90": "REAL",
                "aspect_ratio_p10": "REAL",
                "aspect_ratio_p50": "REAL",
                "aspect_ratio_p90": "REAL",
                "edge_proximity_rate": "REAL",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_data_profile_recording_created "
            "ON detection_data_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_data_profile_detection_scope "
            "ON detection_data_profile(detection_type, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_data_profile_coverage "
            "ON detection_data_profile(coverage_percent);"
        )
        cur.execute("DROP VIEW IF EXISTS detection_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW detection_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    ddp.dataset_id AS dataset_id,
                    ddp.profile_run AS profile_run,
                    COALESCE(dcc.recording_id, tip.recording_id) AS recording_id,
                    COALESCE(dcc.zarr_use, tip.zarr_use) AS zarr_use,
                    ddp.detection_type AS detection_type,
                    ddp.detection_path AS detection_path,
                    ddp.profile_created_utc AS profile_created_utc,
                    ddp.zarr_mtime_ns AS zarr_mtime_ns,
                    ddp.updated_utc AS updated_utc,
                    ddp.frames_total AS frames_total,
                    ddp.frames_with_detections AS frames_with_detections,
                    ddp.coverage_percent AS coverage_percent,
                    ddp.detections_total AS detections_total,
                    ddp.detections_per_frame_p50 AS detections_per_frame_p50,
                    ddp.detections_per_frame_p90 AS detections_per_frame_p90,
                    ddp.w_p10 AS w_p10,
                    ddp.w_p50 AS w_p50,
                    ddp.w_p90 AS w_p90,
                    ddp.h_p10 AS h_p10,
                    ddp.h_p50 AS h_p50,
                    ddp.h_p90 AS h_p90,
                    ddp.area_p10 AS area_p10,
                    ddp.area_p50 AS area_p50,
                    ddp.area_p90 AS area_p90,
                    ddp.aspect_ratio_p10 AS aspect_ratio_p10,
                    ddp.aspect_ratio_p50 AS aspect_ratio_p50,
                    ddp.aspect_ratio_p90 AS aspect_ratio_p90,
                    ddp.edge_proximity_rate AS edge_proximity_rate,
                    COALESCE(dcc.rig_id, ddp.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, ddp.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, ddp.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, ddp.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, ddp.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, ddp.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, ddp.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, ddp.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    ddp.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY ddp.dataset_id
                        ORDER BY
                            COALESCE(ddp.profile_created_utc, ddp.updated_utc) DESC,
                            ddp.profile_run DESC
                    ) AS _rn
                FROM detection_data_profile ddp
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = ddp.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                detection_type,
                detection_path,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                frames_total,
                frames_with_detections,
                coverage_percent,
                detections_total,
                detections_per_frame_p50,
                detections_per_frame_p90,
                w_p10,
                w_p50,
                w_p90,
                h_p10,
                h_p50,
                h_p90,
                area_p10,
                area_p50,
                area_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                edge_proximity_rate,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_detection_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW recording_detection_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    ddpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    ROW_NUMBER() OVER (
                        PARTITION BY ddpl.recording_id
                        ORDER BY
                            COALESCE(ddpl.profile_created_utc, ddpl.updated_utc) DESC,
                            ddpl.profile_run DESC,
                            ddpl.dataset_id DESC
                    ) AS _rn
                FROM detection_data_profile_latest ddpl
                LEFT JOIN datasets d ON d.dataset_id = ddpl.dataset_id
                WHERE ddpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                profile_run,
                zarr_use,
                detection_type,
                detection_path,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                frames_total,
                frames_with_detections,
                coverage_percent,
                detections_total,
                detections_per_frame_p50,
                detections_per_frame_p90,
                w_p10,
                w_p50,
                w_p90,
                h_p10,
                h_p50,
                h_p90,
                area_p10,
                area_p50,
                area_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                edge_proximity_rate,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json,
                zarr_path,
                artifact_kind,
                dataset_status
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_023_detection_data_profile_lineage_projection(self) -> None:
        # Append-only follow-up migration: existing registries may already be at
        # v22 from before lineage projection columns were added. Re-run the v22
        # reconciler to ensure columns/views are present with the latest shape.
        self._migration_022_detection_data_profile_registry()

    def _migration_024_keypoint_data_profile_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS keypoint_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                keypoint_method TEXT,
                source_keypoint_path TEXT,
                source_keypoint_run TEXT,
                skeleton_id TEXT,
                kpt_shape TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                rows_total INTEGER,
                rows_usable INTEGER,
                usable_keypoints_total INTEGER,
                usable_rate REAL,
                confidence_valid_rate REAL,
                geometry_valid_rate REAL,
                triangle_area_p10 REAL,
                triangle_area_p50 REAL,
                triangle_area_p90 REAL,
                min_angle_p10 REAL,
                min_angle_p50 REAL,
                min_angle_p90 REAL,
                heading_p10 REAL,
                heading_p50 REAL,
                heading_p90 REAL,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "keypoint_data_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "keypoint_method": "TEXT",
                "source_keypoint_path": "TEXT",
                "source_keypoint_run": "TEXT",
                "skeleton_id": "TEXT",
                "kpt_shape": "TEXT",
                "pose_schema_name": "TEXT",
                "pose_schema_json": "TEXT",
                "heading_computation_source": "TEXT",
                "heading_computation_json": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "rows_total": "INTEGER",
                "rows_usable": "INTEGER",
                "usable_keypoints_total": "INTEGER",
                "usable_rate": "REAL",
                "confidence_valid_rate": "REAL",
                "geometry_valid_rate": "REAL",
                "triangle_area_p10": "REAL",
                "triangle_area_p50": "REAL",
                "triangle_area_p90": "REAL",
                "min_angle_p10": "REAL",
                "min_angle_p50": "REAL",
                "min_angle_p90": "REAL",
                "heading_p10": "REAL",
                "heading_p50": "REAL",
                "heading_p90": "REAL",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_dataset "
            "ON keypoint_data_profile(dataset_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_recording_created "
            "ON keypoint_data_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_method_scope "
            "ON keypoint_data_profile(keypoint_method, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_method_usable_rate "
            "ON keypoint_data_profile(keypoint_method, usable_rate);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_keypoint_data_profile_lineage "
            "ON keypoint_data_profile(genotype, dpf_at_acquisition);"
        )
        cur.execute("DROP VIEW IF EXISTS keypoint_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW keypoint_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    kdp.dataset_id AS dataset_id,
                    kdp.profile_run AS profile_run,
                    dcc.recording_id AS recording_id,
                    dcc.zarr_use AS zarr_use,
                    kdp.keypoint_method AS keypoint_method,
                    kdp.source_keypoint_path AS source_keypoint_path,
                    kdp.source_keypoint_run AS source_keypoint_run,
                    kdp.skeleton_id AS skeleton_id,
                    kdp.kpt_shape AS kpt_shape,
                    kdp.pose_schema_name AS pose_schema_name,
                    kdp.pose_schema_json AS pose_schema_json,
                    kdp.heading_computation_source AS heading_computation_source,
                    kdp.heading_computation_json AS heading_computation_json,
                    kdp.profile_created_utc AS profile_created_utc,
                    kdp.zarr_mtime_ns AS zarr_mtime_ns,
                    kdp.updated_utc AS updated_utc,
                    kdp.rows_total AS rows_total,
                    kdp.rows_usable AS rows_usable,
                    kdp.usable_keypoints_total AS usable_keypoints_total,
                    kdp.usable_rate AS usable_rate,
                    kdp.confidence_valid_rate AS confidence_valid_rate,
                    kdp.geometry_valid_rate AS geometry_valid_rate,
                    kdp.triangle_area_p10 AS triangle_area_p10,
                    kdp.triangle_area_p50 AS triangle_area_p50,
                    kdp.triangle_area_p90 AS triangle_area_p90,
                    kdp.min_angle_p10 AS min_angle_p10,
                    kdp.min_angle_p50 AS min_angle_p50,
                    kdp.min_angle_p90 AS min_angle_p90,
                    kdp.heading_p10 AS heading_p10,
                    kdp.heading_p50 AS heading_p50,
                    kdp.heading_p90 AS heading_p90,
                    COALESCE(dcc.rig_id, kdp.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, kdp.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, kdp.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, kdp.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, kdp.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, kdp.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, kdp.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, kdp.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    kdp.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY kdp.dataset_id, COALESCE(kdp.keypoint_method, '')
                        ORDER BY
                            COALESCE(kdp.profile_created_utc, kdp.updated_utc) DESC,
                            kdp.profile_run DESC
                    ) AS _rn
                FROM keypoint_data_profile kdp
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = kdp.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                keypoint_method,
                source_keypoint_path,
                source_keypoint_run,
                skeleton_id,
                kpt_shape,
                pose_schema_name,
                pose_schema_json,
                heading_computation_source,
                heading_computation_json,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_keypoints_total,
                usable_rate,
                confidence_valid_rate,
                geometry_valid_rate,
                triangle_area_p10,
                triangle_area_p50,
                triangle_area_p90,
                min_angle_p10,
                min_angle_p50,
                min_angle_p90,
                heading_p10,
                heading_p50,
                heading_p90,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_keypoint_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW recording_keypoint_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    kdpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    ROW_NUMBER() OVER (
                        PARTITION BY kdpl.recording_id, COALESCE(kdpl.keypoint_method, '')
                        ORDER BY
                            COALESCE(kdpl.profile_created_utc, kdpl.updated_utc) DESC,
                            kdpl.profile_run DESC,
                            kdpl.dataset_id DESC
                    ) AS _rn
                FROM keypoint_data_profile_latest kdpl
                LEFT JOIN datasets d ON d.dataset_id = kdpl.dataset_id
                WHERE kdpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                profile_run,
                zarr_use,
                keypoint_method,
                source_keypoint_path,
                source_keypoint_run,
                skeleton_id,
                kpt_shape,
                pose_schema_name,
                pose_schema_json,
                heading_computation_source,
                heading_computation_json,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_keypoints_total,
                usable_rate,
                confidence_valid_rate,
                geometry_valid_rate,
                triangle_area_p10,
                triangle_area_p50,
                triangle_area_p90,
                min_angle_p10,
                min_angle_p50,
                min_angle_p90,
                heading_p10,
                heading_p50,
                heading_p90,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json,
                zarr_path,
                artifact_kind,
                dataset_status
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_025_eye_mask_data_profile_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eye_mask_data_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                stage_group TEXT,
                eye_mask_method TEXT,
                source_eye_mask_path TEXT,
                source_eye_mask_run TEXT,
                source_keypoint_path TEXT,
                source_keypoint_run TEXT,
                source_crop_run TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                rows_total INTEGER,
                rows_usable INTEGER,
                usable_rate REAL,
                reviewed_rate REAL,
                excluded_rate REAL,
                exclusion_reasons_json TEXT,
                ellipse_success_rate REAL,
                pair_success_rate REAL,
                area_p10 REAL,
                area_p50 REAL,
                area_p90 REAL,
                left_area_p10 REAL,
                left_area_p50 REAL,
                left_area_p90 REAL,
                right_area_p10 REAL,
                right_area_p50 REAL,
                right_area_p90 REAL,
                union_area_p10 REAL,
                union_area_p50 REAL,
                union_area_p90 REAL,
                area_lr_ratio_p10 REAL,
                area_lr_ratio_p50 REAL,
                area_lr_ratio_p90 REAL,
                major_axis_p10 REAL,
                major_axis_p50 REAL,
                major_axis_p90 REAL,
                minor_axis_p10 REAL,
                minor_axis_p50 REAL,
                minor_axis_p90 REAL,
                aspect_ratio_p10 REAL,
                aspect_ratio_p50 REAL,
                aspect_ratio_p90 REAL,
                eye_separation_p10 REAL,
                eye_separation_p50 REAL,
                eye_separation_p90 REAL,
                edge_proximity_rate REAL,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_timestamp_utc TEXT,
                source_keypoint_stale_state TEXT,
                source_keypoint_stale_reason TEXT,
                source_keypoint_stale_timestamp_utc TEXT,
                source_keypoint_stale_json TEXT,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "eye_mask_data_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "stage_group": "TEXT",
                "eye_mask_method": "TEXT",
                "source_eye_mask_path": "TEXT",
                "source_eye_mask_run": "TEXT",
                "source_keypoint_path": "TEXT",
                "source_keypoint_run": "TEXT",
                "source_crop_run": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "rows_total": "INTEGER",
                "rows_usable": "INTEGER",
                "usable_rate": "REAL",
                "reviewed_rate": "REAL",
                "excluded_rate": "REAL",
                "exclusion_reasons_json": "TEXT",
                "ellipse_success_rate": "REAL",
                "pair_success_rate": "REAL",
                "area_p10": "REAL",
                "area_p50": "REAL",
                "area_p90": "REAL",
                "left_area_p10": "REAL",
                "left_area_p50": "REAL",
                "left_area_p90": "REAL",
                "right_area_p10": "REAL",
                "right_area_p50": "REAL",
                "right_area_p90": "REAL",
                "union_area_p10": "REAL",
                "union_area_p50": "REAL",
                "union_area_p90": "REAL",
                "area_lr_ratio_p10": "REAL",
                "area_lr_ratio_p50": "REAL",
                "area_lr_ratio_p90": "REAL",
                "major_axis_p10": "REAL",
                "major_axis_p50": "REAL",
                "major_axis_p90": "REAL",
                "minor_axis_p10": "REAL",
                "minor_axis_p50": "REAL",
                "minor_axis_p90": "REAL",
                "aspect_ratio_p10": "REAL",
                "aspect_ratio_p50": "REAL",
                "aspect_ratio_p90": "REAL",
                "eye_separation_p10": "REAL",
                "eye_separation_p50": "REAL",
                "eye_separation_p90": "REAL",
                "edge_proximity_rate": "REAL",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_timestamp_utc": "TEXT",
                "source_keypoint_stale_state": "TEXT",
                "source_keypoint_stale_reason": "TEXT",
                "source_keypoint_stale_timestamp_utc": "TEXT",
                "source_keypoint_stale_json": "TEXT",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_recording_created "
            "ON eye_mask_data_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_method_scope "
            "ON eye_mask_data_profile(eye_mask_method, zarr_use);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_stage_usable_rate "
            "ON eye_mask_data_profile(stage_group, usable_rate);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_stale_state "
            "ON eye_mask_data_profile(source_keypoint_stale_state);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_data_profile_lineage "
            "ON eye_mask_data_profile(genotype, dpf_at_acquisition);"
        )

        cur.execute("DROP VIEW IF EXISTS eye_mask_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW eye_mask_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    emdp.dataset_id AS dataset_id,
                    emdp.profile_run AS profile_run,
                    dcc.recording_id AS recording_id,
                    dcc.zarr_use AS zarr_use,
                    emdp.stage_group AS stage_group,
                    emdp.eye_mask_method AS eye_mask_method,
                    emdp.source_eye_mask_path AS source_eye_mask_path,
                    emdp.source_eye_mask_run AS source_eye_mask_run,
                    emdp.source_keypoint_path AS source_keypoint_path,
                    emdp.source_keypoint_run AS source_keypoint_run,
                    emdp.source_crop_run AS source_crop_run,
                    emdp.profile_created_utc AS profile_created_utc,
                    emdp.zarr_mtime_ns AS zarr_mtime_ns,
                    emdp.updated_utc AS updated_utc,
                    emdp.rows_total AS rows_total,
                    emdp.rows_usable AS rows_usable,
                    emdp.usable_rate AS usable_rate,
                    emdp.reviewed_rate AS reviewed_rate,
                    emdp.excluded_rate AS excluded_rate,
                    emdp.exclusion_reasons_json AS exclusion_reasons_json,
                    emdp.ellipse_success_rate AS ellipse_success_rate,
                    emdp.pair_success_rate AS pair_success_rate,
                    emdp.area_p10 AS area_p10,
                    emdp.area_p50 AS area_p50,
                    emdp.area_p90 AS area_p90,
                    emdp.left_area_p10 AS left_area_p10,
                    emdp.left_area_p50 AS left_area_p50,
                    emdp.left_area_p90 AS left_area_p90,
                    emdp.right_area_p10 AS right_area_p10,
                    emdp.right_area_p50 AS right_area_p50,
                    emdp.right_area_p90 AS right_area_p90,
                    emdp.union_area_p10 AS union_area_p10,
                    emdp.union_area_p50 AS union_area_p50,
                    emdp.union_area_p90 AS union_area_p90,
                    emdp.area_lr_ratio_p10 AS area_lr_ratio_p10,
                    emdp.area_lr_ratio_p50 AS area_lr_ratio_p50,
                    emdp.area_lr_ratio_p90 AS area_lr_ratio_p90,
                    emdp.major_axis_p10 AS major_axis_p10,
                    emdp.major_axis_p50 AS major_axis_p50,
                    emdp.major_axis_p90 AS major_axis_p90,
                    emdp.minor_axis_p10 AS minor_axis_p10,
                    emdp.minor_axis_p50 AS minor_axis_p50,
                    emdp.minor_axis_p90 AS minor_axis_p90,
                    emdp.aspect_ratio_p10 AS aspect_ratio_p10,
                    emdp.aspect_ratio_p50 AS aspect_ratio_p50,
                    emdp.aspect_ratio_p90 AS aspect_ratio_p90,
                    emdp.eye_separation_p10 AS eye_separation_p10,
                    emdp.eye_separation_p50 AS eye_separation_p50,
                    emdp.eye_separation_p90 AS eye_separation_p90,
                    emdp.edge_proximity_rate AS edge_proximity_rate,
                    emdp.review_state AS review_state,
                    emdp.review_method AS review_method,
                    emdp.review_intended_use AS review_intended_use,
                    emdp.review_timestamp_utc AS review_timestamp_utc,
                    emdp.source_keypoint_stale_state AS source_keypoint_stale_state,
                    emdp.source_keypoint_stale_reason AS source_keypoint_stale_reason,
                    emdp.source_keypoint_stale_timestamp_utc AS source_keypoint_stale_timestamp_utc,
                    emdp.source_keypoint_stale_json AS source_keypoint_stale_json,
                    COALESCE(dcc.rig_id, emdp.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, emdp.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, emdp.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, emdp.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, emdp.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, emdp.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, emdp.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, emdp.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    emdp.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY emdp.dataset_id, COALESCE(emdp.stage_group, ''), COALESCE(emdp.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emdp.profile_created_utc, emdp.updated_utc) DESC,
                            emdp.profile_run DESC
                    ) AS _rn
                FROM eye_mask_data_profile emdp
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = emdp.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                stage_group,
                eye_mask_method,
                source_eye_mask_path,
                source_eye_mask_run,
                source_keypoint_path,
                source_keypoint_run,
                source_crop_run,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_rate,
                reviewed_rate,
                excluded_rate,
                exclusion_reasons_json,
                ellipse_success_rate,
                pair_success_rate,
                area_p10,
                area_p50,
                area_p90,
                left_area_p10,
                left_area_p50,
                left_area_p90,
                right_area_p10,
                right_area_p50,
                right_area_p90,
                union_area_p10,
                union_area_p50,
                union_area_p90,
                area_lr_ratio_p10,
                area_lr_ratio_p50,
                area_lr_ratio_p90,
                major_axis_p10,
                major_axis_p50,
                major_axis_p90,
                minor_axis_p10,
                minor_axis_p50,
                minor_axis_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                eye_separation_p10,
                eye_separation_p50,
                eye_separation_p90,
                edge_proximity_rate,
                review_state,
                review_method,
                review_intended_use,
                review_timestamp_utc,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_eye_mask_data_profile_latest;")
        cur.execute(
            """
            CREATE VIEW recording_eye_mask_data_profile_latest AS
            WITH ranked AS (
                SELECT
                    emdpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    ROW_NUMBER() OVER (
                        PARTITION BY emdpl.recording_id, COALESCE(emdpl.stage_group, ''), COALESCE(emdpl.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emdpl.profile_created_utc, emdpl.updated_utc) DESC,
                            emdpl.profile_run DESC,
                            emdpl.dataset_id DESC
                    ) AS _rn
                FROM eye_mask_data_profile_latest emdpl
                LEFT JOIN datasets d ON d.dataset_id = emdpl.dataset_id
                WHERE emdpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                profile_run,
                zarr_use,
                stage_group,
                eye_mask_method,
                source_eye_mask_path,
                source_eye_mask_run,
                source_keypoint_path,
                source_keypoint_run,
                source_crop_run,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                rows_total,
                rows_usable,
                usable_rate,
                reviewed_rate,
                excluded_rate,
                exclusion_reasons_json,
                ellipse_success_rate,
                pair_success_rate,
                area_p10,
                area_p50,
                area_p90,
                left_area_p10,
                left_area_p50,
                left_area_p90,
                right_area_p10,
                right_area_p50,
                right_area_p90,
                union_area_p10,
                union_area_p50,
                union_area_p90,
                area_lr_ratio_p10,
                area_lr_ratio_p50,
                area_lr_ratio_p90,
                major_axis_p10,
                major_axis_p50,
                major_axis_p90,
                minor_axis_p10,
                minor_axis_p50,
                minor_axis_p90,
                aspect_ratio_p10,
                aspect_ratio_p50,
                aspect_ratio_p90,
                eye_separation_p10,
                eye_separation_p50,
                eye_separation_p90,
                edge_proximity_rate,
                review_state,
                review_method,
                review_intended_use,
                review_timestamp_utc,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json,
                zarr_path,
                artifact_kind,
                dataset_status
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_026_eye_mask_quality_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS eye_mask_quality (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                eye_mask_method TEXT,
                source_crop_run TEXT,
                source_keypoint_group TEXT,
                source_keypoints_run TEXT,
                source_eye_masks_run TEXT,
                source_eye_masks_method TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                total_rois INTEGER,
                successful_eyes INTEGER,
                successful_roi_pairs INTEGER,
                successful_roi_pair_rate REAL,
                source_keypoint_stale_state TEXT,
                source_keypoint_stale_reason TEXT,
                source_keypoint_stale_timestamp_utc TEXT,
                source_keypoint_stale_json TEXT,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, stage_group, run_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "eye_mask_quality",
            {
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "eye_mask_method": "TEXT",
                "source_crop_run": "TEXT",
                "source_keypoint_group": "TEXT",
                "source_keypoints_run": "TEXT",
                "source_eye_masks_run": "TEXT",
                "source_eye_masks_method": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "total_rois": "INTEGER",
                "successful_eyes": "INTEGER",
                "successful_roi_pairs": "INTEGER",
                "successful_roi_pair_rate": "REAL",
                "source_keypoint_stale_state": "TEXT",
                "source_keypoint_stale_reason": "TEXT",
                "source_keypoint_stale_timestamp_utc": "TEXT",
                "source_keypoint_stale_json": "TEXT",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_dataset_id ON eye_mask_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_gate
            ON eye_mask_quality(review_state, review_intended_use, eye_mask_method, successful_roi_pair_rate);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_stage_method
            ON eye_mask_quality(stage_group, eye_mask_method);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_eye_mask_quality_recording
            ON eye_mask_quality(recording_id, stage_group, run_created_utc DESC);
            """
        )

        cur.execute(
            """
            INSERT INTO eye_mask_quality (
                dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                eye_mask_method, source_crop_run, source_keypoint_group, source_keypoints_run,
                source_eye_masks_run, source_eye_masks_method,
                review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                total_rois, successful_eyes, successful_roi_pairs, successful_roi_pair_rate,
                source_keypoint_stale_state, source_keypoint_stale_reason, source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json, lifecycle_state, lifecycle_reason, quality_updated_utc, zarr_mtime_ns
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                recording_id,
                zarr_use,
                method AS eye_mask_method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                COALESCE(updated_utc, CURRENT_TIMESTAMP) AS quality_updated_utc,
                zarr_mtime_ns
            FROM eye_mask_performance
            WHERE stage_group = 'refined_eye_masks_runs'
            ON CONFLICT(dataset_id, stage_group, run_name) DO UPDATE SET
                run_created_utc=excluded.run_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                eye_mask_method=excluded.eye_mask_method,
                source_crop_run=excluded.source_crop_run,
                source_keypoint_group=excluded.source_keypoint_group,
                source_keypoints_run=excluded.source_keypoints_run,
                source_eye_masks_run=excluded.source_eye_masks_run,
                source_eye_masks_method=excluded.source_eye_masks_method,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_timestamp_utc=excluded.review_timestamp_utc,
                total_rois=excluded.total_rois,
                successful_eyes=excluded.successful_eyes,
                successful_roi_pairs=excluded.successful_roi_pairs,
                successful_roi_pair_rate=excluded.successful_roi_pair_rate,
                source_keypoint_stale_state=excluded.source_keypoint_stale_state,
                source_keypoint_stale_reason=excluded.source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc=excluded.source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json=excluded.source_keypoint_stale_json,
                lifecycle_state=excluded.lifecycle_state,
                lifecycle_reason=excluded.lifecycle_reason,
                quality_updated_utc=excluded.quality_updated_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns;
            """
        )

        cur.execute("DROP VIEW IF EXISTS eye_mask_quality_current;")
        cur.execute(
            """
            CREATE VIEW eye_mask_quality_current AS
            WITH ranked AS (
                SELECT
                    emq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY emq.dataset_id, COALESCE(emq.stage_group, ''), COALESCE(emq.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emq.review_timestamp_utc, emq.run_created_utc, emq.quality_updated_utc) DESC,
                            COALESCE(emq.run_created_utc, '') DESC,
                            emq.run_name DESC
                    ) AS _rn
                FROM eye_mask_quality emq
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                recording_id,
                zarr_use,
                eye_mask_method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )
        cur.execute("DROP VIEW IF EXISTS eye_mask_quality_overview;")
        cur.execute(
            """
            CREATE VIEW eye_mask_quality_overview AS
            SELECT
                emqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                d.artifact_kind AS artifact_kind,
                d.status AS dataset_status,
                emqc.stage_group AS stage_group,
                emqc.run_name AS run_name,
                emqc.run_created_utc AS run_created_utc,
                emqc.recording_id AS recording_id,
                emqc.eye_mask_method AS eye_mask_method,
                emqc.source_crop_run AS source_crop_run,
                emqc.source_keypoint_group AS source_keypoint_group,
                emqc.source_keypoints_run AS source_keypoints_run,
                emqc.source_eye_masks_run AS source_eye_masks_run,
                emqc.source_eye_masks_method AS source_eye_masks_method,
                emqc.review_state AS review_state,
                emqc.review_method AS review_method,
                emqc.review_intended_use AS review_intended_use,
                emqc.review_reviewer AS review_reviewer,
                emqc.review_timestamp_utc AS review_timestamp_utc,
                emqc.total_rois AS total_rois,
                emqc.successful_eyes AS successful_eyes,
                emqc.successful_roi_pairs AS successful_roi_pairs,
                emqc.successful_roi_pair_rate AS successful_roi_pair_rate,
                emqc.source_keypoint_stale_state AS source_keypoint_stale_state,
                emqc.source_keypoint_stale_reason AS source_keypoint_stale_reason,
                emqc.source_keypoint_stale_timestamp_utc AS source_keypoint_stale_timestamp_utc,
                emqc.source_keypoint_stale_json AS source_keypoint_stale_json,
                emqc.lifecycle_state AS lifecycle_state,
                emqc.lifecycle_reason AS lifecycle_reason,
                emqc.quality_updated_utc AS quality_updated_utc,
                emqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN emqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM eye_mask_quality_current emqc
            LEFT JOIN datasets d ON d.dataset_id = emqc.dataset_id;
            """
        )
        cur.execute("DROP VIEW IF EXISTS recording_eye_mask_quality_overview;")
        cur.execute(
            """
            CREATE VIEW recording_eye_mask_quality_overview AS
            WITH ranked AS (
                SELECT
                    emqo.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY emqo.recording_id, COALESCE(emqo.stage_group, ''), COALESCE(emqo.eye_mask_method, '')
                        ORDER BY
                            COALESCE(emqo.review_timestamp_utc, emqo.run_created_utc, emqo.quality_updated_utc) DESC,
                            COALESCE(emqo.run_created_utc, '') DESC,
                            emqo.run_name DESC,
                            emqo.dataset_id DESC
                    ) AS _rn
                FROM eye_mask_quality_overview emqo
                WHERE emqo.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                run_created_utc,
                eye_mask_method,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_eye_masks_run,
                source_eye_masks_method,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                successful_eyes,
                successful_roi_pairs,
                successful_roi_pair_rate,
                source_keypoint_stale_state,
                source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_027_detect_quality_wide_view_columns(self) -> None:
        """Re-create wide view to include detect_quality step columns."""
        self._migration_020_recording_step_status_wide_view()

    def _migration_028_keypoint_auto_review_policy_columns(self) -> None:
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_policy_id": "TEXT",
                "review_policy_version": "INTEGER",
            },
        )
        cur = self.conn.cursor()
        self._refresh_keypoint_quality_current_view()
        cur.execute("DROP VIEW IF EXISTS keypoint_quality_overview;")
        cur.execute(
            """
            CREATE VIEW keypoint_quality_overview AS
            SELECT
                kqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                kqc.keypoint_method AS keypoint_method,
                kqc.source_keypoint_run AS source_keypoint_run,
                kqc.refined_run AS refined_run,
                kqc.review_state AS review_state,
                kqc.review_method AS review_method,
                kqc.review_intended_use AS review_intended_use,
                kqc.review_policy_id AS review_policy_id,
                kqc.review_policy_version AS review_policy_version,
                kqc.usable_keypoints AS usable_keypoints,
                kqc.total_keypoints AS total_keypoints,
                kqc.usable_keypoints_rate AS usable_keypoints_rate,
                kqc.quality_updated_utc AS quality_updated_utc,
                kqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN kqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM keypoint_quality_current kqc
            LEFT JOIN datasets d ON d.dataset_id = kqc.dataset_id;
            """
        )

    def _migration_029_keypoint_quality_current_latest_source_preference(self) -> None:
        self._refresh_keypoint_quality_current_view()

    def _migration_030_tracking_unassigned_warning_wide_view(self) -> None:
        """Re-create wide view to expose tracking unassigned-row warnings."""
        self._migration_020_recording_step_status_wide_view()

    def _migration_031_tracking_qc_state_wide_view(self) -> None:
        """Re-create wide view to expose structured tracking QA state."""
        self._migration_020_recording_step_status_wide_view()

    def _migration_032_subject_mask_registry(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subject_mask_performance (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                subject_mask_method TEXT,
                label_schema_id TEXT,
                source_crop_run TEXT,
                source_keypoint_group TEXT,
                source_keypoints_run TEXT,
                source_subject_mask_run TEXT,
                source_subject_mask_method TEXT,
                run_semantics TEXT,
                probability_semantics TEXT,
                source_background_run TEXT,
                source_background_array TEXT,
                source_dish_mask_array TEXT,
                tuning_source TEXT,
                tuning_timestamp TEXT,
                total_rois INTEGER,
                rows_with_any_mask INTEGER,
                coverage_percent REAL,
                duration_seconds REAL,
                rois_per_second REAL,
                available_component_count INTEGER,
                available_components_json TEXT,
                unavailable_components_json TEXT,
                component_review_states_json TEXT,
                eye_component_mode TEXT,
                reason_counts_json TEXT,
                summary_statistics_json TEXT,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                source_subject_mask_stale_state TEXT,
                source_subject_mask_stale_reason TEXT,
                source_subject_mask_stale_timestamp_utc TEXT,
                source_subject_mask_stale_json TEXT,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                PRIMARY KEY (dataset_id, stage_group, run_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS subject_mask_component_quality (
                dataset_id TEXT NOT NULL,
                stage_group TEXT NOT NULL,
                run_name TEXT NOT NULL,
                component_name TEXT NOT NULL,
                component_family TEXT,
                run_created_utc TEXT,
                recording_id TEXT,
                zarr_use TEXT,
                subject_mask_method TEXT,
                label_schema_id TEXT,
                eye_component_mode TEXT,
                source_subject_mask_run TEXT,
                available INTEGER,
                review_state TEXT,
                review_method TEXT,
                review_intended_use TEXT,
                review_reviewer TEXT,
                review_timestamp_utc TEXT,
                total_rois INTEGER,
                rows_with_component_mask INTEGER,
                rows_with_component_mask_rate REAL,
                lifecycle_state TEXT,
                lifecycle_reason TEXT,
                quality_updated_utc TEXT,
                zarr_mtime_ns INTEGER,
                PRIMARY KEY (dataset_id, stage_group, run_name, component_name),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "subject_mask_performance",
            {
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "subject_mask_method": "TEXT",
                "label_schema_id": "TEXT",
                "source_crop_run": "TEXT",
                "source_keypoint_group": "TEXT",
                "source_keypoints_run": "TEXT",
                "source_subject_mask_run": "TEXT",
                "source_subject_mask_method": "TEXT",
                "run_semantics": "TEXT",
                "probability_semantics": "TEXT",
                "source_background_run": "TEXT",
                "source_background_array": "TEXT",
                "source_dish_mask_array": "TEXT",
                "tuning_source": "TEXT",
                "tuning_timestamp": "TEXT",
                "total_rois": "INTEGER",
                "rows_with_any_mask": "INTEGER",
                "coverage_percent": "REAL",
                "duration_seconds": "REAL",
                "rois_per_second": "REAL",
                "available_component_count": "INTEGER",
                "available_components_json": "TEXT",
                "unavailable_components_json": "TEXT",
                "component_review_states_json": "TEXT",
                "eye_component_mode": "TEXT",
                "reason_counts_json": "TEXT",
                "summary_statistics_json": "TEXT",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "source_subject_mask_stale_state": "TEXT",
                "source_subject_mask_stale_reason": "TEXT",
                "source_subject_mask_stale_timestamp_utc": "TEXT",
                "source_subject_mask_stale_json": "TEXT",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
            },
        )
        self._ensure_columns(
            "subject_mask_component_quality",
            {
                "component_family": "TEXT",
                "run_created_utc": "TEXT",
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "subject_mask_method": "TEXT",
                "label_schema_id": "TEXT",
                "eye_component_mode": "TEXT",
                "source_subject_mask_run": "TEXT",
                "available": "INTEGER",
                "review_state": "TEXT",
                "review_method": "TEXT",
                "review_intended_use": "TEXT",
                "review_reviewer": "TEXT",
                "review_timestamp_utc": "TEXT",
                "total_rois": "INTEGER",
                "rows_with_component_mask": "INTEGER",
                "rows_with_component_mask_rate": "REAL",
                "lifecycle_state": "TEXT",
                "lifecycle_reason": "TEXT",
                "quality_updated_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
            },
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_recording ON subject_mask_performance(recording_id, stage_group, run_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_stage_method ON subject_mask_performance(stage_group, subject_mask_method);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_source ON subject_mask_performance(source_keypoints_run, source_subject_mask_run);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_perf_review ON subject_mask_performance(review_state, review_intended_use, lifecycle_state);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject_mask_component_dataset_id ON subject_mask_component_quality(dataset_id);"
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subject_mask_component_gate
            ON subject_mask_component_quality(review_state, review_intended_use, component_name, available);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subject_mask_component_stage
            ON subject_mask_component_quality(stage_group, component_name, subject_mask_method);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subject_mask_component_recording
            ON subject_mask_component_quality(recording_id, stage_group, component_name, run_created_utc DESC);
            """
        )

    def _migration_033_subject_mask_registry_semantics_columns(self) -> None:
        """Reconcile subject-mask registry schema after legacy bootstrap registries."""
        self._migration_032_subject_mask_registry()
        cur = self.conn.cursor()

        cur.execute("DROP VIEW IF EXISTS subject_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW subject_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    smp.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY smp.dataset_id, smp.stage_group
                        ORDER BY
                            COALESCE(smp.run_created_utc, smp.updated_utc) DESC,
                            smp.run_name DESC
                    ) AS _rn
                FROM subject_mask_performance smp
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                recording_id,
                zarr_use,
                subject_mask_method,
                label_schema_id,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_subject_mask_run,
                source_subject_mask_method,
                run_semantics,
                probability_semantics,
                source_background_run,
                source_background_array,
                source_dish_mask_array,
                tuning_source,
                tuning_timestamp,
                total_rois,
                rows_with_any_mask,
                coverage_percent,
                duration_seconds,
                rois_per_second,
                available_component_count,
                available_components_json,
                unavailable_components_json,
                component_review_states_json,
                eye_component_mode,
                reason_counts_json,
                summary_statistics_json,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_subject_mask_performance_latest;")
        cur.execute(
            """
            CREATE VIEW recording_subject_mask_performance_latest AS
            WITH ranked AS (
                SELECT
                    smpl.*,
                    d.zarr_path AS zarr_path,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    dcc.rig_id AS rig_id,
                    dcc.arena_id AS arena_id,
                    dcc.camera_id AS camera_id,
                    dcc.canvas_name AS canvas_name,
                    dcc.dish_design AS dish_design,
                    dcc.protocol_name AS protocol_name,
                    ROW_NUMBER() OVER (
                        PARTITION BY smpl.recording_id, smpl.stage_group
                        ORDER BY
                            COALESCE(smpl.run_created_utc, smpl.updated_utc) DESC,
                            smpl.run_name DESC
                    ) AS _rn
                FROM subject_mask_performance_latest smpl
                LEFT JOIN datasets d ON d.dataset_id = smpl.dataset_id
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = smpl.dataset_id
                WHERE smpl.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                stage_group,
                run_name,
                run_created_utc,
                zarr_use,
                subject_mask_method,
                label_schema_id,
                source_crop_run,
                source_keypoint_group,
                source_keypoints_run,
                source_subject_mask_run,
                source_subject_mask_method,
                run_semantics,
                probability_semantics,
                source_background_run,
                source_background_array,
                source_dish_mask_array,
                tuning_source,
                tuning_timestamp,
                total_rois,
                rows_with_any_mask,
                coverage_percent,
                duration_seconds,
                rois_per_second,
                available_component_count,
                available_components_json,
                unavailable_components_json,
                component_review_states_json,
                eye_component_mode,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                zarr_path,
                artifact_kind,
                dataset_status,
                rig_id,
                arena_id,
                camera_id,
                canvas_name,
                dish_design,
                protocol_name,
                zarr_mtime_ns,
                updated_utc
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_current;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_current AS
            WITH ranked AS (
                SELECT
                    smcq.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY smcq.dataset_id, smcq.stage_group, smcq.component_name
                        ORDER BY
                            CASE WHEN COALESCE(smcq.available, 0) = 1 THEN 1 ELSE 0 END DESC,
                            COALESCE(smcq.review_timestamp_utc, smcq.run_created_utc, smcq.quality_updated_utc) DESC,
                            COALESCE(smcq.run_created_utc, '') DESC,
                            smcq.run_name DESC
                    ) AS _rn
                FROM subject_mask_component_quality smcq
            )
            SELECT
                dataset_id,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                recording_id,
                zarr_use,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_overview;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_overview AS
            SELECT
                smcqc.dataset_id AS dataset_id,
                d.zarr_path AS zarr_path,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.zarr_use AS zarr_purpose,
                d.artifact_kind AS artifact_kind,
                d.status AS dataset_status,
                smcqc.stage_group AS stage_group,
                smcqc.run_name AS run_name,
                smcqc.component_name AS component_name,
                smcqc.component_family AS component_family,
                smcqc.run_created_utc AS run_created_utc,
                smcqc.recording_id AS recording_id,
                smcqc.subject_mask_method AS subject_mask_method,
                smcqc.label_schema_id AS label_schema_id,
                smcqc.eye_component_mode AS eye_component_mode,
                smcqc.source_subject_mask_run AS source_subject_mask_run,
                smcqc.available AS available,
                smcqc.review_state AS review_state,
                smcqc.review_method AS review_method,
                smcqc.review_intended_use AS review_intended_use,
                smcqc.review_reviewer AS review_reviewer,
                smcqc.review_timestamp_utc AS review_timestamp_utc,
                smcqc.total_rois AS total_rois,
                smcqc.rows_with_component_mask AS rows_with_component_mask,
                smcqc.rows_with_component_mask_rate AS rows_with_component_mask_rate,
                smp.source_subject_mask_stale_state AS source_subject_mask_stale_state,
                smp.source_subject_mask_stale_reason AS source_subject_mask_stale_reason,
                smp.source_subject_mask_stale_timestamp_utc AS source_subject_mask_stale_timestamp_utc,
                smp.source_subject_mask_stale_json AS source_subject_mask_stale_json,
                CASE
                    WHEN COALESCE(smcqc.available, 0) = 1
                     AND lower(trim(COALESCE(smp.source_subject_mask_stale_state, ''))) = 'stale'
                    THEN 'stale'
                    ELSE smcqc.lifecycle_state
                END AS lifecycle_state,
                CASE
                    WHEN COALESCE(smcqc.available, 0) = 1
                     AND lower(trim(COALESCE(smp.source_subject_mask_stale_state, ''))) = 'stale'
                    THEN COALESCE(NULLIF(trim(smp.source_subject_mask_stale_reason), ''), 'source_subject_mask_stale')
                    ELSE smcqc.lifecycle_reason
                END AS lifecycle_reason,
                smcqc.quality_updated_utc AS quality_updated_utc,
                smcqc.zarr_mtime_ns AS zarr_mtime_ns,
                CASE
                    WHEN smcqc.zarr_mtime_ns IS NULL THEN 1
                    ELSE 0
                END AS quality_stale
            FROM subject_mask_component_quality_current smcqc
            LEFT JOIN datasets d ON d.dataset_id = smcqc.dataset_id
            LEFT JOIN subject_mask_performance smp
              ON smp.dataset_id = smcqc.dataset_id
             AND smp.stage_group = smcqc.stage_group
             AND smp.run_name = smcqc.run_name;
            """
        )

        cur.execute("DROP VIEW IF EXISTS recording_subject_mask_component_quality_overview;")
        cur.execute(
            """
            CREATE VIEW recording_subject_mask_component_quality_overview AS
            WITH ranked AS (
                SELECT
                    smcqo.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY smcqo.recording_id, smcqo.stage_group, smcqo.component_name
                        ORDER BY
                            COALESCE(smcqo.review_timestamp_utc, smcqo.run_created_utc, smcqo.quality_updated_utc) DESC,
                            COALESCE(smcqo.run_created_utc, '') DESC,
                            smcqo.run_name DESC,
                            smcqo.dataset_id DESC
                    ) AS _rn
                FROM subject_mask_component_quality_overview smcqo
                WHERE smcqo.recording_id IS NOT NULL
            )
            SELECT
                recording_id,
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_034_dataset_context_current_view(self) -> None:
        cur = self.conn.cursor()
        if self._table_exists("recordings"):
            self._ensure_columns(
                "recordings",
                {
                    "experiment_context_status": "TEXT",
                    "experiment_context_source": "TEXT",
                    "experiment_context_status_detail": "TEXT",
                    "stimulus_runs_available": "INTEGER",
                },
            )
        cur.execute("DROP VIEW IF EXISTS dataset_context_current;")
        cur.execute(
            """
            CREATE VIEW dataset_context_current AS
            WITH recording_subject_summary AS (
                SELECT
                    rso.recording_id AS recording_id,
                    COUNT(DISTINCT NULLIF(TRIM(rso.subject_id), '')) AS subject_count_recorded,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.subject_id), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.subject_id), ''))
                        ELSE NULL
                    END AS subject_id,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.dish_id), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.dish_id), ''))
                        ELSE NULL
                    END AS dish_id,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.cross_id), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.cross_id), ''))
                        ELSE NULL
                    END AS cross_id,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.genotype), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.genotype), ''))
                        ELSE NULL
                    END AS genotype,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.line_strain), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.line_strain), ''))
                        ELSE NULL
                    END AS line_strain,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.species), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.species), ''))
                        ELSE NULL
                    END AS species,
                    CASE
                        WHEN COUNT(DISTINCT NULLIF(TRIM(rso.sex), '')) = 1
                        THEN MIN(NULLIF(TRIM(rso.sex), ''))
                        ELSE NULL
                    END AS sex,
                    CASE
                        WHEN COUNT(DISTINCT rso.dpf_at_acquisition) = 1
                        THEN MIN(rso.dpf_at_acquisition)
                        ELSE NULL
                    END AS dpf_at_acquisition,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.subject_id), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.subject_id), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS subject_ids_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.dish_id), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.dish_id), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS dish_ids_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.cross_id), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.cross_id), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS cross_ids_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.genotype), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.genotype), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS genotypes_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.line_strain), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.line_strain), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS line_strains_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.species), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.species), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS species_values_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT NULLIF(TRIM(rso2.sex), '') AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND NULLIF(TRIM(rso2.sex), '') IS NOT NULL
                            ORDER BY value
                        )
                    ) AS sex_values_json,
                    (
                        SELECT json_group_array(value)
                        FROM (
                            SELECT DISTINCT rso2.dpf_at_acquisition AS value
                            FROM recording_subject_overview rso2
                            WHERE rso2.recording_id = rso.recording_id
                              AND rso2.dpf_at_acquisition IS NOT NULL
                            ORDER BY value
                        )
                    ) AS dpf_values_json
                FROM recording_subject_overview rso
                GROUP BY rso.recording_id
            )
            SELECT
                d.dataset_id AS dataset_id,
                d.recording_id AS recording_id,
                d.session_uuid AS session_uuid,
                d.zarr_path AS zarr_path,
                d.artifact_kind AS artifact_kind,
                d.zarr_origin AS zarr_origin,
                d.zarr_use AS zarr_use,
                d.status AS dataset_status,
                d.last_seen_utc AS last_seen_utc,
                r.recording_name AS recording_name,
                r.recording_path AS recording_path,
                r.started_utc AS recording_started_utc,
                r.recording_type AS recording_type,
                r.recording_subtype AS recording_subtype,
                r.behavior_mode AS behavior_mode,
                r.artifact_schema_id AS artifact_schema_id,
                r.experiment_context_status AS experiment_context_status,
                r.experiment_context_source AS experiment_context_source,
                r.experiment_context_status_detail AS experiment_context_status_detail,
                r.stimulus_runs_available AS stimulus_runs_available,
                COALESCE(NULLIF(TRIM(r.rig_id), ''), NULLIF(TRIM(p.rig_id), '')) AS rig_id,
                COALESCE(NULLIF(TRIM(r.arena_id), ''), NULLIF(TRIM(p.arena_id), '')) AS arena_id,
                COALESCE(NULLIF(TRIM(r.camera_id), ''), NULLIF(TRIM(p.camera_id), '')) AS camera_id,
                COALESCE(NULLIF(TRIM(r.canvas_name), ''), NULLIF(TRIM(p.canvas_name), '')) AS canvas_name,
                COALESCE(NULLIF(TRIM(r.protocol_name), ''), NULLIF(TRIM(p.protocol_name), '')) AS protocol_name,
                COALESCE(NULLIF(TRIM(r.dish_design), ''), NULLIF(TRIM(p.dish_design), '')) AS dish_design,
                p.protocol_hash AS protocol_hash,
                p.snapshot_status AS snapshot_status,
                p.snapshot_missing_json AS snapshot_missing_json,
                p.fps AS fps,
                p.video_codec AS video_codec,
                p.video_pix_fmt AS video_pix_fmt,
                p.compression_name AS compression_name,
                p.compression_level AS compression_level,
                p.exposure AS exposure,
                p.exposure_unit AS exposure_unit,
                p.gain AS gain,
                p.frame_rate AS frame_rate,
                p.camera_model AS camera_model,
                p.camera_serial AS camera_serial,
                p.has_images_ds AS has_images_ds,
                p.has_images_ds_rgb AS has_images_ds_rgb,
                p.downsample_formats_json AS downsample_formats_json,
                p.subject_count AS subject_count_snapshot,
                rss.subject_count_recorded AS subject_count_recorded,
                COALESCE(rss.subject_count_recorded, p.subject_count) AS subject_count_effective,
                CASE
                    WHEN rss.recording_id IS NOT NULL THEN 'normalized'
                    WHEN (
                        NULLIF(TRIM(p.fish_id), '') IS NOT NULL
                        OR NULLIF(TRIM(p.dish_id), '') IS NOT NULL
                        OR NULLIF(TRIM(p.cross_id), '') IS NOT NULL
                        OR NULLIF(TRIM(p.genotype), '') IS NOT NULL
                        OR p.dpf_at_acquisition IS NOT NULL
                        OR p.subject_count IS NOT NULL
                    ) THEN 'legacy_provenance'
                    ELSE 'missing'
                END AS subject_context_source,
                NULLIF(TRIM(p.fish_id), '') AS legacy_fish_id,
                NULLIF(TRIM(p.dish_id), '') AS legacy_dish_id,
                NULLIF(TRIM(p.cross_id), '') AS legacy_cross_id,
                NULLIF(TRIM(p.genotype), '') AS legacy_genotype,
                NULLIF(TRIM(p.line_strain), '') AS legacy_line_strain,
                NULLIF(TRIM(p.species), '') AS legacy_species,
                NULLIF(TRIM(p.sex), '') AS legacy_sex,
                p.dpf_at_acquisition AS legacy_dpf_at_acquisition,
                rss.subject_id AS subject_id,
                rss.dish_id AS dish_id,
                rss.cross_id AS cross_id,
                rss.genotype AS genotype,
                rss.line_strain AS line_strain,
                rss.species AS species,
                rss.sex AS sex,
                rss.dpf_at_acquisition AS dpf_at_acquisition,
                rss.subject_ids_json AS subject_ids_json,
                rss.dish_ids_json AS dish_ids_json,
                rss.cross_ids_json AS cross_ids_json,
                rss.genotypes_json AS genotypes_json,
                rss.line_strains_json AS line_strains_json,
                rss.species_values_json AS species_values_json,
                rss.sex_values_json AS sex_values_json,
                rss.dpf_values_json AS dpf_values_json
            FROM datasets d
            LEFT JOIN recordings r ON r.recording_id = d.recording_id
            LEFT JOIN provenance p ON p.dataset_id = d.dataset_id
            LEFT JOIN recording_subject_summary rss ON rss.recording_id = d.recording_id;
            """
        )

    def _migration_035_recording_step_status_latest_dataset_context_current(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS recording_step_status_latest;")
        cur.execute(
            """
            CREATE VIEW recording_step_status_latest AS
            SELECT
                COALESCE(NULLIF(trim(rss.recording_id), ''), dcc.recording_id) AS recording_id,
                rss.dataset_id,
                dcc.session_uuid AS session_uuid,
                dcc.zarr_path AS zarr_path,
                dcc.zarr_use AS zarr_use,
                dcc.artifact_kind AS artifact_kind,
                dcc.dataset_status AS dataset_status,
                dcc.rig_id AS rig_id,
                dcc.arena_id AS arena_id,
                dcc.camera_id AS camera_id,
                dcc.canvas_name AS canvas_name,
                dcc.dish_design AS dish_design,
                dcc.protocol_name AS protocol_name,
                dcc.cross_id AS cross_id,
                dcc.genotype AS genotype,
                dcc.dpf_at_acquisition AS dpf_at_acquisition,
                rss.step_name,
                rss.status,
                rss.run_name,
                rss.method,
                rss.coverage_pct,
                rss.review_status_json,
                rss.details_json,
                rss.source,
                rss.zarr_mtime_ns,
                rss.updated_utc
            FROM recording_step_status rss
            LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = rss.dataset_id;
            """
        )

    def _migration_036_subject_mask_component_latest_views(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_latest;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_latest AS
            WITH latest_raw AS (
                SELECT
                    dataset_id,
                    run_name AS latest_subject_mask_run
                FROM subject_mask_performance_latest
                WHERE stage_group = 'subject_mask_runs'
            ),
            ranked AS (
                SELECT
                    smcqo.*,
                    CASE
                        WHEN smcqo.stage_group = 'refined_subject_masks_runs'
                         AND COALESCE(smcqo.source_subject_mask_run, '') <> ''
                         AND COALESCE(smcqo.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 3
                        WHEN smcqo.stage_group = 'subject_mask_runs'
                         AND smcqo.run_name = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 2
                        ELSE 1
                    END AS freshness_rank,
                    CASE
                        WHEN smcqo.stage_group = 'refined_subject_masks_runs' THEN 1
                        ELSE 0
                    END AS stage_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY smcqo.dataset_id, smcqo.component_name
                        ORDER BY
                            CASE
                                WHEN smcqo.stage_group = 'refined_subject_masks_runs'
                                 AND COALESCE(smcqo.source_subject_mask_run, '') <> ''
                                 AND COALESCE(smcqo.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                                THEN 3
                                WHEN smcqo.stage_group = 'subject_mask_runs'
                                 AND smcqo.run_name = COALESCE(lr.latest_subject_mask_run, '')
                                THEN 2
                                ELSE 1
                            END DESC,
                            COALESCE(smcqo.review_timestamp_utc, smcqo.run_created_utc, smcqo.quality_updated_utc) DESC,
                            CASE
                                WHEN smcqo.stage_group = 'refined_subject_masks_runs' THEN 1
                                ELSE 0
                            END DESC,
                            COALESCE(smcqo.run_created_utc, '') DESC,
                            smcqo.run_name DESC
                    ) AS _rn
                FROM subject_mask_component_quality_overview smcqo
                LEFT JOIN latest_raw lr ON lr.dataset_id = smcqo.dataset_id
            )
            SELECT
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                recording_id,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_037_subject_mask_component_eye_compat_latest_views(self) -> None:
        self._migration_036_subject_mask_component_latest_views()
        cur = self.conn.cursor()

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_latest;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_latest AS
            WITH latest_raw AS (
                SELECT
                    dataset_id,
                    run_name AS latest_subject_mask_run
                FROM subject_mask_performance_latest
                WHERE stage_group = 'subject_mask_runs'
            ),
            eye_components AS (
                SELECT 'eye_left' AS component_name
                UNION ALL
                SELECT 'eye_right' AS component_name
            ),
            candidate_rows AS (
                SELECT
                    dataset_id,
                    zarr_path,
                    zarr_origin,
                    zarr_use,
                    zarr_purpose,
                    artifact_kind,
                    dataset_status,
                    stage_group,
                    run_name,
                    component_name,
                    component_family,
                    run_created_utc,
                    recording_id,
                    subject_mask_method,
                    label_schema_id,
                    eye_component_mode,
                    source_subject_mask_run,
                    source_subject_mask_stale_state,
                    source_subject_mask_stale_reason,
                    source_subject_mask_stale_timestamp_utc,
                    source_subject_mask_stale_json,
                    available,
                    review_state,
                    review_method,
                    review_intended_use,
                    review_reviewer,
                    review_timestamp_utc,
                    total_rois,
                    rows_with_component_mask,
                    rows_with_component_mask_rate,
                    lifecycle_state,
                    lifecycle_reason,
                    quality_updated_utc,
                    zarr_mtime_ns,
                    quality_stale
                FROM subject_mask_component_quality_overview
                UNION ALL
                SELECT
                    empl.dataset_id AS dataset_id,
                    d.zarr_path AS zarr_path,
                    d.zarr_origin AS zarr_origin,
                    d.zarr_use AS zarr_use,
                    d.zarr_use AS zarr_purpose,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    empl.stage_group AS stage_group,
                    empl.run_name AS run_name,
                    ec.component_name AS component_name,
                    'eyes' AS component_family,
                    empl.run_created_utc AS run_created_utc,
                    empl.recording_id AS recording_id,
                    empl.method AS subject_mask_method,
                    'subject_v1_lr' AS label_schema_id,
                    'lr' AS eye_component_mode,
                    NULL AS source_subject_mask_run,
                    NULL AS source_subject_mask_stale_state,
                    NULL AS source_subject_mask_stale_reason,
                    NULL AS source_subject_mask_stale_timestamp_utc,
                    NULL AS source_subject_mask_stale_json,
                    1 AS available,
                    empl.review_state AS review_state,
                    empl.review_method AS review_method,
                    empl.review_intended_use AS review_intended_use,
                    empl.review_reviewer AS review_reviewer,
                    empl.review_timestamp_utc AS review_timestamp_utc,
                    empl.total_rois AS total_rois,
                    empl.successful_roi_pairs AS rows_with_component_mask,
                    empl.successful_roi_pair_rate AS rows_with_component_mask_rate,
                    empl.lifecycle_state AS lifecycle_state,
                    empl.lifecycle_reason AS lifecycle_reason,
                    empl.updated_utc AS quality_updated_utc,
                    empl.zarr_mtime_ns AS zarr_mtime_ns,
                    CASE
                        WHEN empl.zarr_mtime_ns IS NULL THEN 1
                        ELSE 0
                    END AS quality_stale
                FROM eye_mask_performance_latest empl
                CROSS JOIN eye_components ec
                LEFT JOIN datasets d ON d.dataset_id = empl.dataset_id
            ),
            scored AS (
                SELECT
                    cr.*,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs'
                         AND COALESCE(cr.source_subject_mask_run, '') <> ''
                         AND COALESCE(cr.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs'
                         AND cr.run_name = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 2
                        ELSE 1
                    END AS subject_mask_freshness_rank,
                    CASE
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_subject_masks_runs'
                        THEN 5
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_eye_masks_runs'
                        THEN 4
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'subject_mask_runs'
                        THEN 3
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'eye_masks_runs'
                        THEN 2
                        ELSE 1
                    END AS eye_component_rank,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs' THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs' THEN 2
                        ELSE 1
                    END AS subject_component_rank,
                    CASE
                        WHEN cr.stage_group IN ('refined_subject_masks_runs', 'refined_eye_masks_runs') THEN 1
                        ELSE 0
                    END AS refined_stage_rank
                FROM candidate_rows cr
                LEFT JOIN latest_raw lr ON lr.dataset_id = cr.dataset_id
            ),
            ranked AS (
                SELECT
                    s.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dataset_id, s.component_name
                        ORDER BY
                            CASE WHEN COALESCE(s.available, 0) = 1 THEN 1 ELSE 0 END DESC,
                            CASE
                                WHEN s.component_name IN ('eye_left', 'eye_right')
                                THEN s.eye_component_rank
                                ELSE s.subject_component_rank
                            END DESC,
                            s.subject_mask_freshness_rank DESC,
                            COALESCE(s.review_timestamp_utc, s.run_created_utc, s.quality_updated_utc) DESC,
                            s.refined_stage_rank DESC,
                            COALESCE(s.run_created_utc, '') DESC,
                            s.run_name DESC
                    ) AS _rn
                FROM scored s
            )
            SELECT
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                recording_id,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

        cur.execute("DROP VIEW IF EXISTS subject_mask_component_quality_latest_by_recording;")
        cur.execute(
            """
            CREATE VIEW subject_mask_component_quality_latest_by_recording AS
            WITH latest_raw AS (
                SELECT
                    recording_id,
                    run_name AS latest_subject_mask_run
                FROM recording_subject_mask_performance_latest
                WHERE stage_group = 'subject_mask_runs'
            ),
            eye_components AS (
                SELECT 'eye_left' AS component_name
                UNION ALL
                SELECT 'eye_right' AS component_name
            ),
            candidate_rows AS (
                SELECT
                    dataset_id,
                    zarr_path,
                    zarr_origin,
                    zarr_use,
                    zarr_purpose,
                    artifact_kind,
                    dataset_status,
                    stage_group,
                    run_name,
                    component_name,
                    component_family,
                    run_created_utc,
                    recording_id,
                    subject_mask_method,
                    label_schema_id,
                    eye_component_mode,
                    source_subject_mask_run,
                    source_subject_mask_stale_state,
                    source_subject_mask_stale_reason,
                    source_subject_mask_stale_timestamp_utc,
                    source_subject_mask_stale_json,
                    available,
                    review_state,
                    review_method,
                    review_intended_use,
                    review_reviewer,
                    review_timestamp_utc,
                    total_rois,
                    rows_with_component_mask,
                    rows_with_component_mask_rate,
                    lifecycle_state,
                    lifecycle_reason,
                    quality_updated_utc,
                    zarr_mtime_ns,
                    quality_stale
                FROM subject_mask_component_quality_overview
                UNION ALL
                SELECT
                    empl.dataset_id AS dataset_id,
                    d.zarr_path AS zarr_path,
                    d.zarr_origin AS zarr_origin,
                    d.zarr_use AS zarr_use,
                    d.zarr_use AS zarr_purpose,
                    d.artifact_kind AS artifact_kind,
                    d.status AS dataset_status,
                    empl.stage_group AS stage_group,
                    empl.run_name AS run_name,
                    ec.component_name AS component_name,
                    'eyes' AS component_family,
                    empl.run_created_utc AS run_created_utc,
                    empl.recording_id AS recording_id,
                    empl.method AS subject_mask_method,
                    'subject_v1_lr' AS label_schema_id,
                    'lr' AS eye_component_mode,
                    NULL AS source_subject_mask_run,
                    NULL AS source_subject_mask_stale_state,
                    NULL AS source_subject_mask_stale_reason,
                    NULL AS source_subject_mask_stale_timestamp_utc,
                    NULL AS source_subject_mask_stale_json,
                    1 AS available,
                    empl.review_state AS review_state,
                    empl.review_method AS review_method,
                    empl.review_intended_use AS review_intended_use,
                    empl.review_reviewer AS review_reviewer,
                    empl.review_timestamp_utc AS review_timestamp_utc,
                    empl.total_rois AS total_rois,
                    empl.successful_roi_pairs AS rows_with_component_mask,
                    empl.successful_roi_pair_rate AS rows_with_component_mask_rate,
                    empl.lifecycle_state AS lifecycle_state,
                    empl.lifecycle_reason AS lifecycle_reason,
                    empl.updated_utc AS quality_updated_utc,
                    empl.zarr_mtime_ns AS zarr_mtime_ns,
                    CASE
                        WHEN empl.zarr_mtime_ns IS NULL THEN 1
                        ELSE 0
                    END AS quality_stale
                FROM eye_mask_performance_latest empl
                CROSS JOIN eye_components ec
                LEFT JOIN datasets d ON d.dataset_id = empl.dataset_id
            ),
            scored AS (
                SELECT
                    cr.*,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs'
                         AND COALESCE(cr.source_subject_mask_run, '') <> ''
                         AND COALESCE(cr.source_subject_mask_run, '') = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs'
                         AND cr.run_name = COALESCE(lr.latest_subject_mask_run, '')
                        THEN 2
                        ELSE 1
                    END AS subject_mask_freshness_rank,
                    CASE
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_subject_masks_runs'
                        THEN 5
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'refined_eye_masks_runs'
                        THEN 4
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'subject_mask_runs'
                        THEN 3
                        WHEN cr.component_name IN ('eye_left', 'eye_right')
                         AND cr.stage_group = 'eye_masks_runs'
                        THEN 2
                        ELSE 1
                    END AS eye_component_rank,
                    CASE
                        WHEN cr.stage_group = 'refined_subject_masks_runs' THEN 3
                        WHEN cr.stage_group = 'subject_mask_runs' THEN 2
                        ELSE 1
                    END AS subject_component_rank,
                    CASE
                        WHEN cr.stage_group IN ('refined_subject_masks_runs', 'refined_eye_masks_runs') THEN 1
                        ELSE 0
                    END AS refined_stage_rank
                FROM candidate_rows cr
                LEFT JOIN latest_raw lr ON lr.recording_id = cr.recording_id
                WHERE cr.recording_id IS NOT NULL
            ),
            ranked AS (
                SELECT
                    s.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.recording_id, s.component_name
                        ORDER BY
                            CASE WHEN COALESCE(s.available, 0) = 1 THEN 1 ELSE 0 END DESC,
                            CASE
                                WHEN s.component_name IN ('eye_left', 'eye_right')
                                THEN s.eye_component_rank
                                ELSE s.subject_component_rank
                            END DESC,
                            s.subject_mask_freshness_rank DESC,
                            COALESCE(s.review_timestamp_utc, s.run_created_utc, s.quality_updated_utc) DESC,
                            s.refined_stage_rank DESC,
                            COALESCE(s.run_created_utc, '') DESC,
                            s.run_name DESC,
                            s.dataset_id DESC
                    ) AS _rn
                FROM scored s
            )
            SELECT
                recording_id,
                dataset_id,
                zarr_path,
                zarr_origin,
                zarr_use,
                zarr_purpose,
                artifact_kind,
                dataset_status,
                stage_group,
                run_name,
                component_name,
                component_family,
                run_created_utc,
                subject_mask_method,
                label_schema_id,
                eye_component_mode,
                source_subject_mask_run,
                source_subject_mask_stale_state,
                source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json,
                available,
                review_state,
                review_method,
                review_intended_use,
                review_reviewer,
                review_timestamp_utc,
                total_rois,
                rows_with_component_mask,
                rows_with_component_mask_rate,
                lifecycle_state,
                lifecycle_reason,
                quality_updated_utc,
                zarr_mtime_ns,
                quality_stale
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _migration_038_subject_mask_component_partial_run_preference(self) -> None:
        """Refresh component views so partial refined runs remain visible."""
        self._migration_033_subject_mask_registry_semantics_columns()
        self._migration_037_subject_mask_component_eye_compat_latest_views()

    def _migration_039_subject_mask_component_source_stale_views(self) -> None:
        """Expose refined-source stale metadata through component registry views."""
        self._migration_038_subject_mask_component_partial_run_preference()

    def _migration_040_subject_mask_training_model_discovery(self) -> None:
        """Index subject-mask model discovery metadata and expose a focused view."""

        if not self._table_exists("training_models"):
            return
        self._ensure_training_model_discovery_columns()
        self._backfill_training_model_discovery_metadata()
        self._refresh_subject_mask_training_models_view()

    def _migration_041_analytics_manifest_registry(self) -> None:
        """Index immutable analytics collection/export manifests."""

        self._ensure_analytics_manifest_tables()

    def _migration_042_recording_experiment_context_columns(self) -> None:
        """Expose whether a recording has experiment/stimulus context."""

        if self._table_exists("recordings"):
            self._ensure_columns(
                "recordings",
                {
                    "experiment_context_status": "TEXT",
                    "experiment_context_source": "TEXT",
                    "experiment_context_status_detail": "TEXT",
                    "stimulus_runs_available": "INTEGER",
                },
            )
        self._migration_034_dataset_context_current_view()
        if self._table_exists("recordings"):
            self._migration_003_recording_columns_reconcile()

    def _migration_043_stage_catalog_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide from the canonical stage catalog."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_044_derived_analysis_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide to expose derived-analysis stages."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_045_tail_behavior_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide to expose tail/classifier stages."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_046_source_freshness_recording_step_status_wide_view(self) -> None:
        """Refresh recording_step_status_wide to display source freshness states."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_047_bout_stimulus_source_freshness_recording_step_status_wide_view(self) -> None:
        """Refresh wide status display for bout/stimulus source freshness."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_048_eye_shape_source_freshness_recording_step_status_wide_view(self) -> None:
        """Refresh wide status display for eye/shape source freshness."""

        self._migration_020_recording_step_status_wide_view()

    def _migration_049_model_input_shape_registry(self) -> None:
        """Normalize trained-model input shape metadata and expose a shared view."""

        if not self._table_exists("training_models"):
            return
        self._ensure_training_model_input_shape_columns()
        self._backfill_training_model_input_shapes()
        self._refresh_model_input_shapes_view()

    def _migration_050_detect_quality_current_reviewed_preference(self) -> None:
        """Prefer reviewed refined-detect rows over newer unreviewed attempts."""

        if not self._table_exists("detect_quality"):
            return
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        self._refresh_detect_quality_current_view()

    def _migration_051_training_image_profile_registry(self) -> None:
        """Register image-domain training profiles for dataset-lake queries."""

        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_image_profile (
                dataset_id TEXT NOT NULL,
                profile_run TEXT NOT NULL,
                recording_id TEXT,
                zarr_use TEXT,
                source_frame_array TEXT,
                profile_created_utc TEXT,
                zarr_mtime_ns INTEGER,
                updated_utc TEXT,
                frames_total INTEGER,
                frames_profiled INTEGER,
                mean_intensity_p50 REAL,
                contrast_p50 REAL,
                sharpness_p50 REAL,
                clip_dark_rate_mean REAL,
                clip_bright_rate_mean REAL,
                illumination_center_edge_p50 REAL,
                illumination_slope_x_p50 REAL,
                illumination_slope_y_p50 REAL,
                fish_bg_contrast_p50 REAL,
                rig_id TEXT,
                camera_id TEXT,
                arena_id TEXT,
                dish_design TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                genotype TEXT,
                dpf_at_acquisition INTEGER,
                profile_json TEXT,
                PRIMARY KEY (dataset_id, profile_run),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        self._ensure_columns(
            "training_image_profile",
            {
                "recording_id": "TEXT",
                "zarr_use": "TEXT",
                "source_frame_array": "TEXT",
                "profile_created_utc": "TEXT",
                "zarr_mtime_ns": "INTEGER",
                "updated_utc": "TEXT",
                "frames_total": "INTEGER",
                "frames_profiled": "INTEGER",
                "mean_intensity_p50": "REAL",
                "contrast_p50": "REAL",
                "sharpness_p50": "REAL",
                "clip_dark_rate_mean": "REAL",
                "clip_bright_rate_mean": "REAL",
                "illumination_center_edge_p50": "REAL",
                "illumination_slope_x_p50": "REAL",
                "illumination_slope_y_p50": "REAL",
                "fish_bg_contrast_p50": "REAL",
                "rig_id": "TEXT",
                "camera_id": "TEXT",
                "arena_id": "TEXT",
                "dish_design": "TEXT",
                "canvas_name": "TEXT",
                "protocol_name": "TEXT",
                "genotype": "TEXT",
                "dpf_at_acquisition": "INTEGER",
                "profile_json": "TEXT",
            },
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_image_profile_recording_created "
            "ON training_image_profile(recording_id, profile_created_utc DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_image_profile_scope "
            "ON training_image_profile(zarr_use, source_frame_array);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_image_profile_domain_metrics "
            "ON training_image_profile(mean_intensity_p50, contrast_p50, sharpness_p50);"
        )
        cur.execute("DROP VIEW IF EXISTS training_image_profile_latest;")
        cur.execute(
            """
            CREATE VIEW training_image_profile_latest AS
            WITH ranked AS (
                SELECT
                    tip.dataset_id AS dataset_id,
                    tip.profile_run AS profile_run,
                    dcc.recording_id AS recording_id,
                    dcc.zarr_use AS zarr_use,
                    tip.source_frame_array AS source_frame_array,
                    tip.profile_created_utc AS profile_created_utc,
                    tip.zarr_mtime_ns AS zarr_mtime_ns,
                    tip.updated_utc AS updated_utc,
                    tip.frames_total AS frames_total,
                    tip.frames_profiled AS frames_profiled,
                    tip.mean_intensity_p50 AS mean_intensity_p50,
                    tip.contrast_p50 AS contrast_p50,
                    tip.sharpness_p50 AS sharpness_p50,
                    tip.clip_dark_rate_mean AS clip_dark_rate_mean,
                    tip.clip_bright_rate_mean AS clip_bright_rate_mean,
                    tip.illumination_center_edge_p50 AS illumination_center_edge_p50,
                    tip.illumination_slope_x_p50 AS illumination_slope_x_p50,
                    tip.illumination_slope_y_p50 AS illumination_slope_y_p50,
                    tip.fish_bg_contrast_p50 AS fish_bg_contrast_p50,
                    COALESCE(dcc.rig_id, tip.rig_id) AS rig_id,
                    COALESCE(dcc.camera_id, tip.camera_id) AS camera_id,
                    COALESCE(dcc.arena_id, tip.arena_id) AS arena_id,
                    COALESCE(dcc.dish_design, tip.dish_design) AS dish_design,
                    COALESCE(dcc.canvas_name, tip.canvas_name) AS canvas_name,
                    COALESCE(dcc.protocol_name, tip.protocol_name) AS protocol_name,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.genotype
                        ELSE COALESCE(dcc.genotype, tip.genotype)
                    END AS genotype,
                    CASE
                        WHEN dcc.subject_context_source = 'normalized' THEN dcc.dpf_at_acquisition
                        ELSE COALESCE(dcc.dpf_at_acquisition, tip.dpf_at_acquisition)
                    END AS dpf_at_acquisition,
                    tip.profile_json AS profile_json,
                    ROW_NUMBER() OVER (
                        PARTITION BY tip.dataset_id
                        ORDER BY
                            COALESCE(tip.profile_created_utc, tip.updated_utc) DESC,
                            tip.profile_run DESC
                    ) AS _rn
                FROM training_image_profile tip
                LEFT JOIN dataset_context_current dcc ON dcc.dataset_id = tip.dataset_id
            )
            SELECT
                dataset_id,
                profile_run,
                recording_id,
                zarr_use,
                source_frame_array,
                profile_created_utc,
                zarr_mtime_ns,
                updated_utc,
                frames_total,
                frames_profiled,
                mean_intensity_p50,
                contrast_p50,
                sharpness_p50,
                clip_dark_rate_mean,
                clip_bright_rate_mean,
                illumination_center_edge_p50,
                illumination_slope_x_p50,
                illumination_slope_y_p50,
                fish_bg_contrast_p50,
                rig_id,
                camera_id,
                arena_id,
                dish_design,
                canvas_name,
                protocol_name,
                genotype,
                dpf_at_acquisition,
                profile_json
            FROM ranked
            WHERE _rn = 1;
            """
        )

    def _ensure_training_model_input_shape_columns(self) -> None:
        self._ensure_columns(
            "training_models",
            {
                "input_shape": "TEXT",
                "input_layout": "TEXT",
                "input_channels": "INTEGER",
                "img_h": "INTEGER",
                "img_w": "INTEGER",
                "max_batch": "INTEGER",
                "dynamic_shapes": "INTEGER",
                "input_dtype": "TEXT",
                "input_color_space": "TEXT",
                "input_shape_source": "TEXT",
                "input_shape_status": "TEXT",
            },
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_input_shape ON training_models(task_type, img_h, img_w);"
        )

    def _shape_fields_from_training_payloads(
        self,
        *,
        task_type: Optional[str],
        final_metrics: Optional[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]],
        input_shape: Any = None,
        input_layout: Optional[str] = None,
        input_channels: Optional[int] = None,
        img_h: Optional[int] = None,
        img_w: Optional[int] = None,
        max_batch: Optional[int] = None,
        dynamic_shapes: Optional[bool | int] = None,
        input_dtype: Optional[str] = None,
        input_color_space: Optional[str] = None,
        input_shape_source: Optional[str] = None,
        input_shape_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        final = dict(final_metrics or {})
        meta = dict(metadata or {})
        task = (
            _normalize_task_type(task_type)
            or _normalize_task_type(meta.get("task_type"))
            or _infer_task_type(
                set_id=None,
                run_id=None,
                model_path=None,
                explicit=task_type,
            )
        )

        resolved_source = input_shape_source
        resolved_status = input_shape_status
        resolved_shape = input_shape
        resolved_imgsz: Any = None

        if resolved_shape is not None or img_h is not None or img_w is not None:
            resolved_source = resolved_source or "explicit"
            resolved_status = resolved_status or "explicit"
        else:
            for source_name, mapping in (
                ("final_metrics", final),
                ("metadata", meta),
            ):
                shape_candidate = mapping.get("input_shape") or mapping.get("model_input_shape")
                if shape_candidate is not None:
                    resolved_shape = shape_candidate
                    resolved_source = resolved_source or f"{source_name}.input_shape"
                    resolved_status = resolved_status or "explicit"
                    break
                if mapping.get("imgsz_h") is not None and mapping.get("imgsz_w") is not None:
                    resolved_imgsz = [mapping.get("imgsz_h"), mapping.get("imgsz_w")]
                    resolved_source = resolved_source or f"{source_name}.imgsz_h_imgsz_w"
                    resolved_status = resolved_status or "inferred_from_imgsz"
                    break
                for key in ("effective_imgsz", "imgsz", "model_imgsz"):
                    if mapping.get(key) is not None:
                        resolved_imgsz = mapping.get(key)
                        resolved_source = resolved_source or f"{source_name}.{key}"
                        resolved_status = resolved_status or "inferred_from_imgsz"
                        break
                if resolved_imgsz is not None:
                    break
                training_history = _coerce_mapping(mapping.get("training_history"))
                if training_history is not None:
                    for key in ("effective_imgsz", "imgsz"):
                        if training_history.get(key) is not None:
                            resolved_imgsz = training_history.get(key)
                            resolved_source = resolved_source or f"{source_name}.training_history.{key}"
                            resolved_status = resolved_status or "inferred_from_imgsz"
                            break
                if resolved_imgsz is not None:
                    break
                training_params = _coerce_mapping(mapping.get("training_params"))
                if training_params is not None and training_params.get("imgsz") is not None:
                    resolved_imgsz = training_params.get("imgsz")
                    resolved_source = resolved_source or f"{source_name}.training_params.imgsz"
                    resolved_status = resolved_status or "inferred_from_imgsz"
                    break

        (
            shape_text,
            resolved_img_h,
            resolved_img_w,
            resolved_max_batch,
            resolved_dynamic,
        ) = self._resolve_shape_fields(input_shape=resolved_shape, imgsz=resolved_imgsz)

        img_h_norm = self._int_or_none(img_h) if img_h is not None else resolved_img_h
        img_w_norm = self._int_or_none(img_w) if img_w is not None else resolved_img_w
        channels_norm = self._int_or_none(input_channels)
        if channels_norm is None and shape_text:
            shape_list = self._shape_to_list(shape_text)
            if shape_list and len(shape_list) >= 2:
                channels_norm = self._int_or_none(shape_list[1])
        if channels_norm is None and task in {"detect", "pose"} and (img_h_norm is not None or img_w_norm is not None):
            channels_norm = 3

        if shape_text is None and img_h_norm is not None and img_w_norm is not None and channels_norm is not None:
            shape_text = _json_dumps([1, int(channels_norm), int(img_h_norm), int(img_w_norm)])
            resolved_max_batch = resolved_max_batch if resolved_max_batch is not None else 1
            resolved_dynamic = resolved_dynamic if resolved_dynamic is not None else 0

        if max_batch is not None:
            resolved_max_batch = self._int_or_none(max_batch)
        if dynamic_shapes is not None:
            resolved_dynamic = int(bool(dynamic_shapes))

        layout_norm = str(input_layout).strip() if input_layout else None
        if not layout_norm and shape_text:
            shape_list = self._shape_to_list(shape_text)
            if shape_list and len(shape_list) == 4:
                layout_norm = "NCHW"

        dtype_norm = str(input_dtype).strip() if input_dtype else None
        if not dtype_norm:
            dtype_value = meta.get("input_dtype") or final.get("input_dtype")
            dtype_norm = str(dtype_value).strip() if dtype_value else None

        color_norm = str(input_color_space).strip().lower() if input_color_space else None
        if not color_norm:
            color_value = meta.get("input_color_space") or final.get("input_color_space")
            color_norm = str(color_value).strip().lower() if color_value else None
        if not color_norm and task in {"detect", "pose"} and channels_norm == 3:
            color_norm = "rgb"

        if not dtype_norm and task in {"detect", "pose"} and shape_text:
            dtype_norm = "float32"

        if shape_text is None and img_h_norm is None and img_w_norm is None:
            resolved_status = resolved_status or "unknown"
            resolved_source = resolved_source or None
        else:
            resolved_status = resolved_status or "explicit"
            resolved_source = resolved_source or "explicit"

        return {
            "input_shape": shape_text,
            "input_layout": layout_norm,
            "input_channels": channels_norm,
            "img_h": img_h_norm,
            "img_w": img_w_norm,
            "max_batch": resolved_max_batch,
            "dynamic_shapes": resolved_dynamic,
            "input_dtype": dtype_norm,
            "input_color_space": color_norm,
            "input_shape_source": resolved_source,
            "input_shape_status": resolved_status,
        }

    def _export_input_shape_fallback(
        self,
        run_id: str,
        *,
        task_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT input_shape, img_h, img_w, max_batch, dynamic_shapes
            FROM onnx_models
            WHERE run_id = ?
              AND (input_shape IS NOT NULL OR img_h IS NOT NULL OR img_w IS NOT NULL)
            ORDER BY created_utc DESC
            LIMIT 1;
            """,
            (str(run_id),),
        ).fetchone()
        source = "onnx_models"
        if row is None:
            row = self.conn.execute(
                """
                SELECT input_shape, img_h, img_w, max_batch, dynamic_shapes
                FROM tensorrt_models
                WHERE run_id = ?
                  AND (input_shape IS NOT NULL OR img_h IS NOT NULL OR img_w IS NOT NULL)
                ORDER BY created_utc DESC
                LIMIT 1;
                """,
                (str(run_id),),
            ).fetchone()
            source = "tensorrt_models"
        if row is None:
            return None
        fields = self._shape_fields_from_training_payloads(
            task_type=task_type,
            final_metrics=None,
            metadata=None,
            input_shape=row["input_shape"],
            img_h=row["img_h"],
            img_w=row["img_w"],
            max_batch=row["max_batch"],
            dynamic_shapes=row["dynamic_shapes"],
            input_shape_source=f"{source}.input_shape",
            input_shape_status="export_backfill",
        )
        return fields

    @staticmethod
    def _shape_fields_conflict(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        for key in ("img_h", "img_w", "input_channels"):
            lval = left.get(key)
            rval = right.get(key)
            if lval is not None and rval is not None and int(lval) != int(rval):
                return True
        return False

    def _backfill_training_model_input_shapes(self) -> None:
        rows = self.conn.execute(
            """
            SELECT
                tm.run_id,
                tm.final_metrics_json,
                tm.metadata_json,
                tm.task_type AS model_task_type,
                tr.task_type AS run_task_type,
                tm.input_shape,
                tm.input_layout,
                tm.input_channels,
                tm.img_h,
                tm.img_w,
                tm.max_batch,
                tm.dynamic_shapes,
                tm.input_dtype,
                tm.input_color_space,
                tm.input_shape_source,
                tm.input_shape_status
            FROM training_models tm
            LEFT JOIN training_runs tr ON tr.run_id = tm.run_id;
            """
        ).fetchall()
        for row in rows:
            final_payload = _json_loads(row["final_metrics_json"])
            metadata_payload = _json_loads(row["metadata_json"])
            final_metrics = final_payload if isinstance(final_payload, Mapping) else {}
            metadata = metadata_payload if isinstance(metadata_payload, Mapping) else {}
            task_type = (
                _normalize_task_type(row["model_task_type"])
                or _normalize_task_type(metadata.get("task_type"))
                or _normalize_task_type(row["run_task_type"])
            )
            fields = self._shape_fields_from_training_payloads(
                task_type=task_type,
                final_metrics=final_metrics,
                metadata=metadata,
                input_shape=row["input_shape"],
                input_layout=row["input_layout"],
                input_channels=row["input_channels"],
                img_h=row["img_h"],
                img_w=row["img_w"],
                max_batch=row["max_batch"],
                dynamic_shapes=row["dynamic_shapes"],
                input_dtype=row["input_dtype"],
                input_color_space=row["input_color_space"],
                input_shape_source=row["input_shape_source"],
                input_shape_status=row["input_shape_status"],
            )
            export_fields = self._export_input_shape_fallback(str(row["run_id"]), task_type=task_type)
            if fields["input_shape"] is None and export_fields is not None:
                fields = export_fields
            elif export_fields is not None and self._shape_fields_conflict(fields, export_fields):
                fields["input_shape_status"] = "conflict"
                conflict = {
                    "training": {
                        "input_shape": fields.get("input_shape"),
                        "img_h": fields.get("img_h"),
                        "img_w": fields.get("img_w"),
                        "input_channels": fields.get("input_channels"),
                        "source": fields.get("input_shape_source"),
                    },
                    "export": {
                        "input_shape": export_fields.get("input_shape"),
                        "img_h": export_fields.get("img_h"),
                        "img_w": export_fields.get("img_w"),
                        "input_channels": export_fields.get("input_channels"),
                        "source": export_fields.get("input_shape_source"),
                    },
                }
                metadata["input_shape_conflict"] = conflict

            self.conn.execute(
                """
                UPDATE training_models
                SET input_shape = ?,
                    input_layout = ?,
                    input_channels = ?,
                    img_h = ?,
                    img_w = ?,
                    max_batch = ?,
                    dynamic_shapes = ?,
                    input_dtype = ?,
                    input_color_space = ?,
                    input_shape_source = ?,
                    input_shape_status = ?,
                    metadata_json = ?
                WHERE run_id = ?;
                """,
                (
                    fields["input_shape"],
                    fields["input_layout"],
                    fields["input_channels"],
                    fields["img_h"],
                    fields["img_w"],
                    fields["max_batch"],
                    fields["dynamic_shapes"],
                    fields["input_dtype"],
                    fields["input_color_space"],
                    fields["input_shape_source"],
                    fields["input_shape_status"],
                    _json_dumps(metadata),
                    str(row["run_id"]),
                ),
            )

    def _refresh_model_input_shapes_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS model_input_shapes;")
        cur.execute(
            """
            CREATE VIEW model_input_shapes AS
            SELECT
                'training' AS artifact_kind,
                tm.run_id AS run_id,
                tm.set_id AS set_id,
                COALESCE(tm.task_type, tr.task_type) AS task_type,
                tm.model_path AS artifact_path,
                tm.model_sha256 AS artifact_sha256,
                NULL AS artifact_precision,
                tm.input_shape AS input_shape,
                tm.input_layout AS input_layout,
                tm.input_channels AS input_channels,
                tm.img_h AS img_h,
                tm.img_w AS img_w,
                tm.max_batch AS max_batch,
                tm.dynamic_shapes AS dynamic_shapes,
                tm.input_dtype AS input_dtype,
                tm.input_color_space AS input_color_space,
                tm.input_shape_source AS input_shape_source,
                tm.input_shape_status AS input_shape_status,
                tm.created_utc AS created_utc
            FROM training_models tm
            LEFT JOIN training_runs tr ON tr.run_id = tm.run_id

            UNION ALL

            SELECT
                'onnx' AS artifact_kind,
                om.run_id AS run_id,
                om.set_id AS set_id,
                COALESCE(tm.task_type, tr.task_type) AS task_type,
                om.path AS artifact_path,
                om.sha256 AS artifact_sha256,
                NULL AS artifact_precision,
                om.input_shape AS input_shape,
                CASE WHEN om.input_shape IS NOT NULL THEN 'NCHW' ELSE NULL END AS input_layout,
                CASE WHEN json_valid(om.input_shape)
                    THEN CAST(json_extract(om.input_shape, '$[1]') AS INTEGER)
                    ELSE NULL
                END AS input_channels,
                om.img_h AS img_h,
                om.img_w AS img_w,
                om.max_batch AS max_batch,
                om.dynamic_shapes AS dynamic_shapes,
                NULL AS input_dtype,
                CASE
                    WHEN COALESCE(tm.task_type, tr.task_type) IN ('detect', 'pose')
                         AND json_valid(om.input_shape)
                         AND CAST(json_extract(om.input_shape, '$[1]') AS INTEGER) = 3
                    THEN 'rgb'
                    ELSE NULL
                END AS input_color_space,
                'onnx_models.input_shape' AS input_shape_source,
                CASE
                    WHEN om.input_shape IS NOT NULL OR om.img_h IS NOT NULL OR om.img_w IS NOT NULL THEN 'explicit'
                    ELSE 'unknown'
                END AS input_shape_status,
                om.created_utc AS created_utc
            FROM onnx_models om
            LEFT JOIN training_models tm ON tm.run_id = om.run_id
            LEFT JOIN training_runs tr ON tr.run_id = om.run_id

            UNION ALL

            SELECT
                'tensorrt' AS artifact_kind,
                trt.run_id AS run_id,
                trt.set_id AS set_id,
                COALESCE(tm.task_type, tr.task_type) AS task_type,
                trt.path AS artifact_path,
                trt.sha256 AS artifact_sha256,
                trt.precision AS artifact_precision,
                trt.input_shape AS input_shape,
                CASE WHEN trt.input_shape IS NOT NULL THEN 'NCHW' ELSE NULL END AS input_layout,
                CASE WHEN json_valid(trt.input_shape)
                    THEN CAST(json_extract(trt.input_shape, '$[1]') AS INTEGER)
                    ELSE NULL
                END AS input_channels,
                trt.img_h AS img_h,
                trt.img_w AS img_w,
                trt.max_batch AS max_batch,
                trt.dynamic_shapes AS dynamic_shapes,
                NULL AS input_dtype,
                CASE
                    WHEN COALESCE(tm.task_type, tr.task_type) IN ('detect', 'pose')
                         AND json_valid(trt.input_shape)
                         AND CAST(json_extract(trt.input_shape, '$[1]') AS INTEGER) = 3
                    THEN 'rgb'
                    ELSE NULL
                END AS input_color_space,
                'tensorrt_models.input_shape' AS input_shape_source,
                CASE
                    WHEN trt.input_shape IS NOT NULL OR trt.img_h IS NOT NULL OR trt.img_w IS NOT NULL THEN 'explicit'
                    ELSE 'unknown'
                END AS input_shape_status,
                trt.created_utc AS created_utc
            FROM tensorrt_models trt
            LEFT JOIN training_models tm ON tm.run_id = trt.run_id
            LEFT JOIN training_runs tr ON tr.run_id = trt.run_id;
            """
        )

    def _ensure_analytics_manifest_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_collections (
                collection_id TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL,
                collection_name TEXT,
                manifest_path TEXT NOT NULL,
                schema_id TEXT,
                schema_version INTEGER,
                record_count INTEGER,
                included_record_count INTEGER,
                created_utc TEXT,
                indexed_utc TEXT,
                status TEXT,
                metadata_json TEXT,
                PRIMARY KEY (collection_id, manifest_sha256)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_exports (
                export_run_id TEXT PRIMARY KEY,
                collection_id TEXT,
                collection_manifest_sha256 TEXT,
                export_manifest_path TEXT NOT NULL,
                output_root TEXT,
                schema_version INTEGER,
                tool TEXT,
                palette_git_commit TEXT,
                palette_git_dirty INTEGER,
                source_recording_count INTEGER,
                table_count INTEGER,
                diagnostics_count INTEGER,
                row_counts_json TEXT,
                tables_json TEXT,
                created_at_utc TEXT,
                indexed_utc TEXT,
                status TEXT,
                metadata_json TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_export_tables (
                export_run_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                table_path TEXT,
                row_count INTEGER,
                part_count INTEGER,
                part_files_json TEXT,
                indexed_utc TEXT,
                PRIMARY KEY (export_run_id, table_name),
                FOREIGN KEY(export_run_id) REFERENCES analytics_exports(export_run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analytics_collections_status
            ON analytics_collections(status, collection_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analytics_exports_collection
            ON analytics_exports(collection_id, collection_manifest_sha256, status);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analytics_export_tables_name
            ON analytics_export_tables(table_name, export_run_id);
            """
        )
        cur.execute("DROP VIEW IF EXISTS analytics_export_overview;")
        cur.execute(
            """
            CREATE VIEW analytics_export_overview AS
            SELECT
                ae.export_run_id,
                ae.status,
                ae.collection_id,
                ae.collection_manifest_sha256,
                ac.collection_name,
                ae.export_manifest_path,
                ae.output_root,
                ae.created_at_utc,
                ae.indexed_utc,
                ae.source_recording_count,
                ae.table_count,
                ae.diagnostics_count,
                ae.row_counts_json,
                ae.tables_json,
                ae.palette_git_commit,
                ae.palette_git_dirty
            FROM analytics_exports ae
            LEFT JOIN analytics_collections ac
              ON ac.collection_id = ae.collection_id
             AND ac.manifest_sha256 = ae.collection_manifest_sha256;
            """
        )

    def _ensure_training_model_discovery_columns(self) -> None:
        self._ensure_columns(
            "training_models",
            {
                "task_type": "TEXT",
                "label_schema_id": "TEXT",
                "coverage_class": "TEXT",
                "component_coverage_key": "TEXT",
                "mask_labels_json": "TEXT",
                "component_groups_json": "TEXT",
                "best_metric_name": "TEXT",
                "best_metric_value": "REAL",
                "best_epoch": "INTEGER",
            },
        )
        cur = self.conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_task_status ON training_models(task_type, status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_label_schema ON training_models(label_schema_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_training_models_component_coverage ON training_models(component_coverage_key);"
        )

    def _backfill_training_model_discovery_metadata(self) -> None:
        rows = self.conn.execute(
            """
            SELECT
                tm.run_id,
                tm.metadata_json,
                tm.final_metrics_json,
                tm.task_type AS model_task_type,
                tr.task_type AS run_task_type
            FROM training_models tm
            LEFT JOIN training_runs tr ON tr.run_id = tm.run_id;
            """
        ).fetchall()
        for row in rows:
            final_metrics = _json_loads(row["final_metrics_json"]) or {}
            existing_metadata = _json_loads(row["metadata_json"]) or {}
            task_type = (
                _normalize_task_type(row["model_task_type"])
                or _normalize_task_type(existing_metadata.get("task_type"))
                or _normalize_task_type(row["run_task_type"])
                or ("subject_masks" if _coerce_mapping(final_metrics.get("subject_mask_model_summary")) else None)
            )
            metadata = _training_model_discovery_metadata(
                task_type=task_type,
                final_metrics=final_metrics,
                metadata=existing_metadata,
            )
            fields = _training_model_discovery_index_fields(
                task_type=task_type,
                final_metrics=final_metrics,
                metadata=metadata,
            )
            self.conn.execute(
                """
                UPDATE training_models
                SET task_type = ?,
                    label_schema_id = ?,
                    coverage_class = ?,
                    component_coverage_key = ?,
                    mask_labels_json = ?,
                    component_groups_json = ?,
                    best_metric_name = ?,
                    best_metric_value = ?,
                    best_epoch = ?,
                    metadata_json = ?
                WHERE run_id = ?;
                """,
                (
                    fields["task_type"],
                    fields["label_schema_id"],
                    fields["coverage_class"],
                    fields["component_coverage_key"],
                    fields["mask_labels_json"],
                    fields["component_groups_json"],
                    fields["best_metric_name"],
                    fields["best_metric_value"],
                    fields["best_epoch"],
                    _json_dumps(metadata),
                    str(row["run_id"]),
                ),
            )

    def _refresh_subject_mask_training_models_view(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DROP VIEW IF EXISTS subject_mask_training_models;")
        cur.execute(
            """
            CREATE VIEW subject_mask_training_models AS
            SELECT
                tm.run_id,
                tm.set_id,
                tm.status,
                COALESCE(tm.task_type, tr.task_type) AS task_type,
                tm.model_path,
                tm.model_sha256,
                tm.metrics_path,
                tm.metrics_sha256,
                tm.label_schema_id,
                tm.coverage_class,
                tm.component_coverage_key,
                tm.mask_labels_json,
                tm.component_groups_json,
                tm.best_metric_name,
                tm.best_metric_value,
                tm.best_epoch,
                tm.created_utc,
                tm.final_metrics_json,
                tm.metadata_json
            FROM training_models tm
            LEFT JOIN training_runs tr ON tr.run_id = tm.run_id
            WHERE COALESCE(tm.task_type, tr.task_type) = 'subject_masks'
               OR json_type(tm.final_metrics_json, '$.subject_mask_model_summary') IS NOT NULL;
            """
        )

    def _ensure_columns(self, table: str, columns: Dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
        for name, ddl in columns.items():
            if name in existing:
                continue
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl};")

    def _normalize_pose_schema(
        self,
        *,
        kpt_shape: Optional[Sequence[Any]] = None,
        keypoint_labels: Optional[Sequence[Any]] = None,
        edges: Optional[Sequence[Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        shape_norm: Optional[List[int]] = None
        if isinstance(kpt_shape, (list, tuple)):
            values: List[int] = []
            for item in kpt_shape:
                val = _as_int(item)
                if val is None:
                    continue
                values.append(int(val))
            if values:
                shape_norm = values

        labels_norm: Optional[List[str]] = None
        if isinstance(keypoint_labels, (list, tuple)):
            labels = [str(item).strip() for item in keypoint_labels if str(item).strip()]
            if labels:
                labels_norm = labels

        edges_norm: Optional[List[List[int]]] = None
        if isinstance(edges, (list, tuple)):
            pairs: List[List[int]] = []
            for edge in edges:
                if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                    continue
                src = _as_int(edge[0])
                dst = _as_int(edge[1])
                if src is None or dst is None:
                    continue
                pairs.append([int(src), int(dst)])
            if pairs:
                edges_norm = pairs

        if shape_norm is None and labels_norm is None and edges_norm is None:
            return None
        return {
            "kpt_shape": shape_norm,
            "keypoint_labels": labels_norm,
            "skeleton_edges": edges_norm,
        }

    def upsert_pose_skeleton_spec(
        self,
        *,
        kpt_shape: Optional[Sequence[Any]] = None,
        keypoint_labels: Optional[Sequence[Any]] = None,
        edges: Optional[Sequence[Any]] = None,
        name: Optional[str] = None,
    ) -> Optional[str]:
        spec = self._normalize_pose_schema(
            kpt_shape=kpt_shape,
            keypoint_labels=keypoint_labels,
            edges=edges,
        )
        if spec is None:
            return None

        spec_text = _canonical_json_text(spec)
        spec_sha256 = sha256(spec_text.encode("utf-8")).hexdigest()
        existing = self.conn.execute(
            "SELECT skeleton_id FROM pose_skeleton_specs WHERE spec_sha256 = ?;",
            (spec_sha256,),
        ).fetchone()
        if existing and existing["skeleton_id"]:
            return str(existing["skeleton_id"])

        skeleton_id = f"pose_skel_{spec_sha256[:12]}"
        payload = {
            "skeleton_id": skeleton_id,
            "spec_sha256": spec_sha256,
            "name": str(name).strip() if isinstance(name, str) and str(name).strip() else None,
            "kpt_shape_json": _json_dumps(spec.get("kpt_shape")),
            "keypoint_labels_json": _json_dumps(spec.get("keypoint_labels")),
            "edges_json": _json_dumps(spec.get("skeleton_edges")),
            "spec_json": spec_text,
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO pose_skeleton_specs (
                skeleton_id, spec_sha256, name, kpt_shape_json, keypoint_labels_json,
                edges_json, spec_json, created_utc
            )
            VALUES (
                :skeleton_id, :spec_sha256, :name, :kpt_shape_json, :keypoint_labels_json,
                :edges_json, :spec_json, :created_utc
            )
            ON CONFLICT(spec_sha256) DO UPDATE SET
                name=COALESCE(excluded.name, pose_skeleton_specs.name),
                kpt_shape_json=excluded.kpt_shape_json,
                keypoint_labels_json=excluded.keypoint_labels_json,
                edges_json=excluded.edges_json,
                spec_json=excluded.spec_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()
        return skeleton_id

    def _resolve_run_skeleton_id(self, run_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT skeleton_id FROM training_runs WHERE run_id = ?;",
            (run_id,),
        ).fetchone()
        if row and row["skeleton_id"]:
            return str(row["skeleton_id"])
        return None

    def upsert_dataset(
        self,
        dataset_id: str,
        *,
        session_uuid: Optional[str],
        zarr_path: Path,
        recording_id: Optional[str] = None,
        artifact_kind: Optional[str] = None,
        zarr_purpose: Optional[str] = None,
        zarr_origin: Optional[str] = None,
        zarr_use: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        resolved_recording_id = recording_id
        if resolved_recording_id is None and session_uuid:
            path_text = str(zarr_path).replace("\\", "/").lower()
            if "/recordings/" in path_text:
                resolved_recording_id = session_uuid
        resolved_artifact_kind = artifact_kind or _infer_dataset_artifact_kind(
            zarr_path=zarr_path,
            dataset_id=dataset_id,
            session_uuid=session_uuid,
        )
        inferred_origin, inferred_use = _infer_zarr_origin_use(
            artifact_kind=resolved_artifact_kind,
            zarr_purpose=zarr_purpose,
        )
        payload = {
            "dataset_id": dataset_id,
            "session_uuid": session_uuid,
            "zarr_path": str(zarr_path),
            "recording_id": resolved_recording_id,
            "artifact_kind": resolved_artifact_kind,
            "zarr_origin": _normalize_zarr_origin(zarr_origin) or inferred_origin,
            "zarr_use": _normalize_zarr_use(zarr_use) or inferred_use,
            "path_hash": _compute_path_hash(zarr_path),
            "created_utc": now,
            "last_seen_utc": now,
            "status": "active",
        }
        self.conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, session_uuid, zarr_path, recording_id, artifact_kind, zarr_origin, zarr_use,
                path_hash, created_utc, last_seen_utc, status
            )
            VALUES (
                :dataset_id, :session_uuid, :zarr_path, :recording_id, :artifact_kind, :zarr_origin, :zarr_use,
                :path_hash, :created_utc, :last_seen_utc, :status
            )
            ON CONFLICT(dataset_id) DO UPDATE SET
                session_uuid=excluded.session_uuid,
                zarr_path=excluded.zarr_path,
                recording_id=COALESCE(excluded.recording_id, datasets.recording_id),
                artifact_kind=COALESCE(excluded.artifact_kind, datasets.artifact_kind),
                zarr_origin=COALESCE(excluded.zarr_origin, datasets.zarr_origin),
                zarr_use=COALESCE(excluded.zarr_use, datasets.zarr_use),
                path_hash=excluded.path_hash,
                last_seen_utc=excluded.last_seen_utc,
                status=excluded.status;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_recording(
        self,
        *,
        recording_id: str,
        session_uuid: Optional[str] = None,
        recording_name: Optional[str] = None,
        recording_path: Optional[str] = None,
        started_utc: Optional[str] = None,
        recording_type: Optional[str] = None,
        recording_subtype: Optional[str] = None,
        behavior_mode: Optional[str] = None,
        artifact_schema_id: Optional[str] = None,
        experiment_context_status: Optional[str] = None,
        experiment_context_source: Optional[str] = None,
        experiment_context_status_detail: Optional[str] = None,
        stimulus_runs_available: Optional[int] = None,
        rig_id: Optional[str] = None,
        arena_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        canvas_name: Optional[str] = None,
        protocol_name: Optional[str] = None,
        dish_design: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        payload = {
            "recording_id": str(recording_id),
            "session_uuid": session_uuid,
            "recording_name": recording_name,
            "recording_path": recording_path,
            "started_utc": started_utc,
            "recording_type": recording_type,
            "recording_subtype": recording_subtype,
            "behavior_mode": behavior_mode,
            "artifact_schema_id": artifact_schema_id,
            "experiment_context_status": experiment_context_status,
            "experiment_context_source": experiment_context_source,
            "experiment_context_status_detail": experiment_context_status_detail,
            "stimulus_runs_available": stimulus_runs_available,
            "rig_id": rig_id,
            "arena_id": arena_id,
            "camera_id": camera_id,
            "canvas_name": canvas_name,
            "protocol_name": protocol_name,
            "dish_design": dish_design,
            "created_utc": now,
            "updated_utc": now,
        }
        self.conn.execute(
            """
            INSERT INTO recordings (
                recording_id, session_uuid, recording_name, recording_path, started_utc,
                recording_type, recording_subtype, behavior_mode, artifact_schema_id,
                experiment_context_status, experiment_context_source,
                experiment_context_status_detail, stimulus_runs_available,
                rig_id, arena_id, camera_id, canvas_name,
                protocol_name, dish_design, created_utc, updated_utc
            )
            VALUES (
                :recording_id, :session_uuid, :recording_name, :recording_path, :started_utc,
                :recording_type, :recording_subtype, :behavior_mode, :artifact_schema_id,
                :experiment_context_status, :experiment_context_source,
                :experiment_context_status_detail, :stimulus_runs_available,
                :rig_id, :arena_id, :camera_id, :canvas_name,
                :protocol_name, :dish_design, :created_utc, :updated_utc
            )
            ON CONFLICT(recording_id) DO UPDATE SET
                session_uuid=COALESCE(excluded.session_uuid, recordings.session_uuid),
                recording_name=COALESCE(excluded.recording_name, recordings.recording_name),
                recording_path=COALESCE(excluded.recording_path, recordings.recording_path),
                started_utc=COALESCE(excluded.started_utc, recordings.started_utc),
                recording_type=COALESCE(excluded.recording_type, recordings.recording_type),
                recording_subtype=COALESCE(excluded.recording_subtype, recordings.recording_subtype),
                behavior_mode=COALESCE(excluded.behavior_mode, recordings.behavior_mode),
                artifact_schema_id=COALESCE(excluded.artifact_schema_id, recordings.artifact_schema_id),
                experiment_context_status=COALESCE(
                    excluded.experiment_context_status,
                    recordings.experiment_context_status
                ),
                experiment_context_source=COALESCE(
                    excluded.experiment_context_source,
                    recordings.experiment_context_source
                ),
                experiment_context_status_detail=COALESCE(
                    excluded.experiment_context_status_detail,
                    recordings.experiment_context_status_detail
                ),
                stimulus_runs_available=COALESCE(
                    excluded.stimulus_runs_available,
                    recordings.stimulus_runs_available
                ),
                rig_id=COALESCE(excluded.rig_id, recordings.rig_id),
                arena_id=COALESCE(excluded.arena_id, recordings.arena_id),
                camera_id=COALESCE(excluded.camera_id, recordings.camera_id),
                canvas_name=COALESCE(excluded.canvas_name, recordings.canvas_name),
                protocol_name=COALESCE(excluded.protocol_name, recordings.protocol_name),
                dish_design=COALESCE(excluded.dish_design, recordings.dish_design),
                updated_utc=excluded.updated_utc;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_analytics_collection(
        self,
        *,
        collection_id: str,
        manifest_sha256: str,
        manifest_path: Path,
        collection_name: Optional[str] = None,
        schema_id: Optional[str] = None,
        schema_version: Optional[int] = None,
        record_count: Optional[int] = None,
        included_record_count: Optional[int] = None,
        created_utc: Optional[str] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index an immutable analytics collection manifest."""

        self._ensure_analytics_manifest_tables()
        payload = {
            "collection_id": str(collection_id),
            "manifest_sha256": str(manifest_sha256),
            "collection_name": collection_name,
            "manifest_path": str(manifest_path),
            "schema_id": schema_id,
            "schema_version": schema_version,
            "record_count": record_count,
            "included_record_count": included_record_count,
            "created_utc": created_utc,
            "indexed_utc": _utc_now(),
            "status": str(status),
            "metadata_json": _json_dumps(metadata),
        }
        self.conn.execute(
            """
            INSERT INTO analytics_collections (
                collection_id, manifest_sha256, collection_name, manifest_path,
                schema_id, schema_version, record_count, included_record_count,
                created_utc, indexed_utc, status, metadata_json
            )
            VALUES (
                :collection_id, :manifest_sha256, :collection_name, :manifest_path,
                :schema_id, :schema_version, :record_count, :included_record_count,
                :created_utc, :indexed_utc, :status, :metadata_json
            )
            ON CONFLICT(collection_id, manifest_sha256) DO UPDATE SET
                collection_name=excluded.collection_name,
                manifest_path=excluded.manifest_path,
                schema_id=excluded.schema_id,
                schema_version=excluded.schema_version,
                record_count=excluded.record_count,
                included_record_count=excluded.included_record_count,
                created_utc=excluded.created_utc,
                indexed_utc=excluded.indexed_utc,
                status=excluded.status,
                metadata_json=excluded.metadata_json;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_analytics_export(
        self,
        *,
        export_run_id: str,
        export_manifest_path: Path,
        collection_id: Optional[str] = None,
        collection_manifest_sha256: Optional[str] = None,
        output_root: Optional[Path] = None,
        schema_version: Optional[int] = None,
        tool: Optional[str] = None,
        palette_git_commit: Optional[str] = None,
        palette_git_dirty: Optional[bool] = None,
        source_recording_count: Optional[int] = None,
        row_counts_by_table: Optional[Mapping[str, Any]] = None,
        part_files_by_table: Optional[Mapping[str, Any]] = None,
        created_at_utc: Optional[str] = None,
        diagnostics_count: Optional[int] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index an immutable analytics export manifest and its table paths."""

        self._ensure_analytics_manifest_tables()
        row_counts = dict(row_counts_by_table or {})
        part_files = dict(part_files_by_table or {})
        table_names = sorted(set(row_counts) | set(part_files))
        payload = {
            "export_run_id": str(export_run_id),
            "collection_id": collection_id,
            "collection_manifest_sha256": collection_manifest_sha256,
            "export_manifest_path": str(export_manifest_path),
            "output_root": str(output_root) if output_root is not None else None,
            "schema_version": schema_version,
            "tool": tool,
            "palette_git_commit": palette_git_commit,
            "palette_git_dirty": None if palette_git_dirty is None else int(bool(palette_git_dirty)),
            "source_recording_count": source_recording_count,
            "table_count": len(table_names),
            "diagnostics_count": diagnostics_count,
            "row_counts_json": _json_dumps(row_counts),
            "tables_json": _json_dumps(table_names),
            "created_at_utc": created_at_utc,
            "indexed_utc": _utc_now(),
            "status": str(status),
            "metadata_json": _json_dumps(metadata),
        }
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO analytics_exports (
                    export_run_id, collection_id, collection_manifest_sha256,
                    export_manifest_path, output_root, schema_version, tool,
                    palette_git_commit, palette_git_dirty, source_recording_count,
                    table_count, diagnostics_count, row_counts_json, tables_json,
                    created_at_utc, indexed_utc, status, metadata_json
                )
                VALUES (
                    :export_run_id, :collection_id, :collection_manifest_sha256,
                    :export_manifest_path, :output_root, :schema_version, :tool,
                    :palette_git_commit, :palette_git_dirty, :source_recording_count,
                    :table_count, :diagnostics_count, :row_counts_json, :tables_json,
                    :created_at_utc, :indexed_utc, :status, :metadata_json
                )
                ON CONFLICT(export_run_id) DO UPDATE SET
                    collection_id=excluded.collection_id,
                    collection_manifest_sha256=excluded.collection_manifest_sha256,
                    export_manifest_path=excluded.export_manifest_path,
                    output_root=excluded.output_root,
                    schema_version=excluded.schema_version,
                    tool=excluded.tool,
                    palette_git_commit=excluded.palette_git_commit,
                    palette_git_dirty=excluded.palette_git_dirty,
                    source_recording_count=excluded.source_recording_count,
                    table_count=excluded.table_count,
                    diagnostics_count=excluded.diagnostics_count,
                    row_counts_json=excluded.row_counts_json,
                    tables_json=excluded.tables_json,
                    created_at_utc=excluded.created_at_utc,
                    indexed_utc=excluded.indexed_utc,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json;
                """,
                payload,
            )
            self.conn.execute(
                "DELETE FROM analytics_export_tables WHERE export_run_id = ?;",
                (str(export_run_id),),
            )
            indexed_utc = payload["indexed_utc"]
            for table_name in table_names:
                files_raw = part_files.get(table_name) or []
                files = [str(item) for item in files_raw] if isinstance(files_raw, list) else []
                table_path = None
                if files:
                    try:
                        table_path = str(Path(files[0]).parent)
                    except Exception:
                        table_path = None
                self.conn.execute(
                    """
                    INSERT INTO analytics_export_tables (
                        export_run_id, table_name, table_path, row_count,
                        part_count, part_files_json, indexed_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(export_run_id),
                        str(table_name),
                        table_path,
                        _as_int(row_counts.get(table_name)),
                        len(files),
                        _json_dumps(files),
                        indexed_utc,
                    ),
                )

    def upsert_provenance(
        self,
        dataset_id: str,
        *,
        provenance: Dict[str, Any],
        context: Dict[str, Any],
        protocol_name: Optional[str],
        protocol_hash: Optional[str],
        acquisition: Optional[Dict[str, Any]] = None,
        zarr_purpose: Optional[str] = None,
    ) -> None:
        acquisition = acquisition or {}
        recording_context_row = self.conn.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM datasets d
                INNER JOIN recordings r ON r.recording_id = d.recording_id
                WHERE d.dataset_id = ?
            ) AS has_recording_context;
            """,
            (dataset_id,),
        ).fetchone()
        write_legacy_recording_context_snapshot = not bool(
            recording_context_row is not None and int(recording_context_row["has_recording_context"] or 0) == 1
        )
        payload = {
            "dataset_id": dataset_id,
            "fish_id": provenance.get("fish_id"),
            "subject_count": provenance.get("subject_count"),
            "dish_id": provenance.get("dish_id"),
            "dish_design": acquisition.get("dish_design") if write_legacy_recording_context_snapshot else None,
            "cross_id": provenance.get("cross_id"),
            "line_strain": provenance.get("line_strain"),
            "genotype": provenance.get("genotype"),
            "parents_json": _json_dumps(provenance.get("parents")),
            "species": provenance.get("species"),
            "sex": provenance.get("sex"),
            "dpf_at_acquisition": provenance.get("dpf_at_acquisition"),
            "rig_id": context.get("rig_id") if write_legacy_recording_context_snapshot else None,
            "arena_id": context.get("arena_id") if write_legacy_recording_context_snapshot else None,
            "camera_id": context.get("camera_id") if write_legacy_recording_context_snapshot else None,
            "canvas_name": context.get("canvas_name") if write_legacy_recording_context_snapshot else None,
            "fps": acquisition.get("fps"),
            "video_codec": acquisition.get("video_codec"),
            "video_pix_fmt": acquisition.get("video_pix_fmt"),
            "format_title": acquisition.get("format_title"),
            "format_comment": acquisition.get("format_comment"),
            "format_encoder": acquisition.get("format_encoder"),
            "encoder_name": acquisition.get("encoder_name"),
            "encoder_codec": acquisition.get("encoder_codec"),
            "encoder_preset": acquisition.get("encoder_preset"),
            "encoder_tuning": acquisition.get("encoder_tuning"),
            "encoder_rc": acquisition.get("encoder_rc"),
            "encoder_bpp": acquisition.get("encoder_bpp"),
            "encoder_target_bps": acquisition.get("encoder_target_bps"),
            "encoder_res": acquisition.get("encoder_res"),
            "encoder_res_width": acquisition.get("encoder_res_width"),
            "encoder_res_height": acquisition.get("encoder_res_height"),
            "encoder_fps": acquisition.get("encoder_fps"),
            "encoder_color": acquisition.get("encoder_color"),
            "encoder_params_json": acquisition.get("encoder_params_json"),
            "source_video": acquisition.get("source_video"),
            "compression_name": acquisition.get("compression_name"),
            "compression_level": acquisition.get("compression_level"),
            "exposure": acquisition.get("exposure"),
            "exposure_unit": acquisition.get("exposure_unit"),
            "gain": acquisition.get("gain"),
            "frame_rate": acquisition.get("frame_rate"),
            "pixel_format": acquisition.get("pixel_format"),
            "binning": acquisition.get("binning"),
            "adc": acquisition.get("adc"),
            "camera_model": acquisition.get("camera_model"),
            "camera_serial": acquisition.get("camera_serial"),
            "camera_metadata_json": acquisition.get("camera_metadata_json"),
            "has_images_ds": acquisition.get("has_images_ds"),
            "has_images_ds_rgb": acquisition.get("has_images_ds_rgb"),
            "downsample_formats_json": acquisition.get("downsample_formats_json"),
            "protocol_name": protocol_name if write_legacy_recording_context_snapshot else None,
            "protocol_hash": protocol_hash,
            "snapshot_status": provenance.get("snapshot_status"),
            "snapshot_missing_json": _json_dumps(provenance.get("snapshot_missing")),
        }
        if write_legacy_recording_context_snapshot:
            recording_context_update_sql = """
                dish_design=excluded.dish_design,
                rig_id=excluded.rig_id,
                arena_id=excluded.arena_id,
                camera_id=excluded.camera_id,
                canvas_name=excluded.canvas_name,
                protocol_name=excluded.protocol_name,
            """
        else:
            recording_context_update_sql = """
                dish_design=COALESCE(provenance.dish_design, excluded.dish_design),
                rig_id=COALESCE(provenance.rig_id, excluded.rig_id),
                arena_id=COALESCE(provenance.arena_id, excluded.arena_id),
                camera_id=COALESCE(provenance.camera_id, excluded.camera_id),
                canvas_name=COALESCE(provenance.canvas_name, excluded.canvas_name),
                protocol_name=COALESCE(provenance.protocol_name, excluded.protocol_name),
            """
        self.conn.execute(
            f"""
            INSERT INTO provenance (
                dataset_id, fish_id, subject_count, dish_id, dish_design, cross_id, line_strain, genotype, parents_json,
                species, sex, dpf_at_acquisition, rig_id, arena_id, camera_id, canvas_name,
                fps, video_codec, video_pix_fmt, format_title, format_comment, format_encoder,
                encoder_name, encoder_codec, encoder_preset, encoder_tuning, encoder_rc, encoder_bpp,
                encoder_target_bps, encoder_res, encoder_res_width, encoder_res_height, encoder_fps, encoder_color,
                encoder_params_json,
                source_video, compression_name, compression_level,
                exposure, exposure_unit, gain, frame_rate, pixel_format, binning, adc, camera_model, camera_serial,
                camera_metadata_json, has_images_ds, has_images_ds_rgb, downsample_formats_json,
                protocol_name, protocol_hash, snapshot_status, snapshot_missing_json
            )
            VALUES (
                :dataset_id, :fish_id, :subject_count, :dish_id, :dish_design, :cross_id, :line_strain, :genotype, :parents_json,
                :species, :sex, :dpf_at_acquisition, :rig_id, :arena_id, :camera_id, :canvas_name,
                :fps, :video_codec, :video_pix_fmt, :format_title, :format_comment, :format_encoder,
                :encoder_name, :encoder_codec, :encoder_preset, :encoder_tuning, :encoder_rc, :encoder_bpp,
                :encoder_target_bps, :encoder_res, :encoder_res_width, :encoder_res_height, :encoder_fps, :encoder_color,
                :encoder_params_json,
                :source_video, :compression_name, :compression_level,
                :exposure, :exposure_unit, :gain, :frame_rate, :pixel_format, :binning, :adc, :camera_model, :camera_serial,
                :camera_metadata_json, :has_images_ds, :has_images_ds_rgb, :downsample_formats_json,
                :protocol_name, :protocol_hash, :snapshot_status, :snapshot_missing_json
            )
            ON CONFLICT(dataset_id) DO UPDATE SET
                fish_id=excluded.fish_id,
                subject_count=excluded.subject_count,
                dish_id=excluded.dish_id,
                cross_id=excluded.cross_id,
                line_strain=excluded.line_strain,
                genotype=excluded.genotype,
                parents_json=excluded.parents_json,
                species=excluded.species,
                sex=excluded.sex,
                dpf_at_acquisition=excluded.dpf_at_acquisition,
                {recording_context_update_sql}
                fps=excluded.fps,
                video_codec=excluded.video_codec,
                video_pix_fmt=excluded.video_pix_fmt,
                format_title=excluded.format_title,
                format_comment=excluded.format_comment,
                format_encoder=excluded.format_encoder,
                encoder_name=excluded.encoder_name,
                encoder_codec=excluded.encoder_codec,
                encoder_preset=excluded.encoder_preset,
                encoder_tuning=excluded.encoder_tuning,
                encoder_rc=excluded.encoder_rc,
                encoder_bpp=excluded.encoder_bpp,
                encoder_target_bps=excluded.encoder_target_bps,
                encoder_res=excluded.encoder_res,
                encoder_res_width=excluded.encoder_res_width,
                encoder_res_height=excluded.encoder_res_height,
                encoder_fps=excluded.encoder_fps,
                encoder_color=excluded.encoder_color,
                encoder_params_json=excluded.encoder_params_json,
                source_video=excluded.source_video,
                compression_name=excluded.compression_name,
                compression_level=excluded.compression_level,
                exposure=excluded.exposure,
                exposure_unit=excluded.exposure_unit,
                gain=excluded.gain,
                frame_rate=excluded.frame_rate,
                pixel_format=excluded.pixel_format,
                binning=excluded.binning,
                adc=excluded.adc,
                camera_model=excluded.camera_model,
                camera_serial=excluded.camera_serial,
                camera_metadata_json=excluded.camera_metadata_json,
                has_images_ds=excluded.has_images_ds,
                has_images_ds_rgb=excluded.has_images_ds_rgb,
                downsample_formats_json=excluded.downsample_formats_json,
                protocol_hash=excluded.protocol_hash,
                snapshot_status=excluded.snapshot_status,
                snapshot_missing_json=excluded.snapshot_missing_json;
            """,
            payload,
        )
        self.conn.commit()

    def replace_detection_sources(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        """Replace detection source lineage rows for a dataset."""
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                "DELETE FROM detection_sources WHERE dataset_id = ?;",
                (dataset_id,),
            )
            for record in records:
                refined_run = _normalize_path_text(record.get("refined_run"))
                source_type = _as_text(record.get("source_type"))
                if refined_run is None or source_type is None:
                    continue
                payload = {
                    "dataset_id": dataset_id,
                    "refined_run": refined_run,
                    "source_type": source_type.lower(),
                    "counts_json": _json_dumps(record.get("counts")),
                    "created_utc": _as_text(record.get("created_utc")) or now,
                }
                self.conn.execute(
                    """
                    INSERT INTO detection_sources (dataset_id, refined_run, source_type, counts_json, created_utc)
                    VALUES (:dataset_id, :refined_run, :source_type, :counts_json, :created_utc)
                    ON CONFLICT(dataset_id, refined_run, source_type) DO UPDATE SET
                        counts_json=excluded.counts_json,
                        created_utc=excluded.created_utc;
                    """,
                    payload,
                )

    def replace_dataset_lineage(
        self,
        *,
        child_dataset_id: str,
        parent_dataset_ids: Iterable[str],
        relationship_type: str,
        source_set_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Replace lineage edges for one child + relationship_type."""
        child = str(child_dataset_id)
        rel = str(relationship_type).strip()
        if not rel:
            raise ValueError("relationship_type must be non-empty")
        parents = sorted({str(parent_id) for parent_id in parent_dataset_ids if parent_id})
        if child in parents:
            raise ValueError("dataset_lineage self-edge is not allowed")
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                """
                DELETE FROM dataset_lineage
                WHERE child_dataset_id = ? AND relationship_type = ?;
                """,
                (child, rel),
            )
            for parent in parents:
                payload = {
                    "child_dataset_id": child,
                    "parent_dataset_id": parent,
                    "relationship_type": rel,
                    "source_set_id": str(source_set_id) if source_set_id else None,
                    "metadata_json": _json_dumps(metadata),
                    "created_utc": now,
                    "updated_utc": now,
                }
                self.conn.execute(
                    """
                    INSERT INTO dataset_lineage (
                        child_dataset_id, parent_dataset_id, relationship_type,
                        source_set_id, metadata_json, created_utc, updated_utc
                    )
                    VALUES (
                        :child_dataset_id, :parent_dataset_id, :relationship_type,
                        :source_set_id, :metadata_json, :created_utc, :updated_utc
                    )
                    ON CONFLICT(child_dataset_id, parent_dataset_id, relationship_type) DO UPDATE SET
                        source_set_id=excluded.source_set_id,
                        metadata_json=excluded.metadata_json,
                        updated_utc=excluded.updated_utc;
                    """,
                    payload,
                )

    def upsert_keypoint_quality(
        self,
        *,
        dataset_id: str,
        refined_run: str,
        refined_created_utc: Optional[str],
        source_keypoint_run: str,
        keypoint_method: Optional[str],
        review_state: Optional[str],
        review_intended_use: Optional[str],
        review_reviewer: Optional[str],
        review_timestamp_utc: Optional[str],
        usable_keypoints: Optional[int],
        total_keypoints: Optional[int],
        usable_keypoints_rate: Optional[float],
        raw_keypoints_success_rate: Optional[float],
        raw_keypoints_successful: Optional[int],
        review_method: Optional[str] = None,
        review_notes: Optional[str] = None,
        review_policy_id: Optional[str] = None,
        review_policy_version: Optional[int] = None,
        quality_updated_utc: Optional[str] = None,
        zarr_mtime_ns: Optional[int] = None,
    ) -> None:
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
                "review_policy_id": "TEXT",
                "review_policy_version": "INTEGER",
            },
        )
        payload = {
            "dataset_id": str(dataset_id),
            "refined_run": str(refined_run),
            "refined_created_utc": refined_created_utc,
            "source_keypoint_run": str(source_keypoint_run),
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
            "quality_updated_utc": quality_updated_utc or _utc_now(),
            "zarr_mtime_ns": zarr_mtime_ns,
        }
        self.conn.execute(
            """
            INSERT INTO keypoint_quality (
                dataset_id, refined_run, refined_created_utc, source_keypoint_run, keypoint_method,
                review_state, review_method, review_intended_use, review_reviewer, review_notes,
                review_policy_id, review_policy_version, review_timestamp_utc,
                usable_keypoints, total_keypoints, usable_keypoints_rate,
                raw_keypoints_success_rate, raw_keypoints_successful,
                quality_updated_utc, zarr_mtime_ns
            )
            VALUES (
                :dataset_id, :refined_run, :refined_created_utc, :source_keypoint_run, :keypoint_method,
                :review_state, :review_method, :review_intended_use, :review_reviewer, :review_notes,
                :review_policy_id, :review_policy_version, :review_timestamp_utc,
                :usable_keypoints, :total_keypoints, :usable_keypoints_rate,
                :raw_keypoints_success_rate, :raw_keypoints_successful,
                :quality_updated_utc, :zarr_mtime_ns
            )
            ON CONFLICT(dataset_id, refined_run) DO UPDATE SET
                refined_created_utc=excluded.refined_created_utc,
                source_keypoint_run=excluded.source_keypoint_run,
                keypoint_method=excluded.keypoint_method,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_notes=excluded.review_notes,
                review_policy_id=excluded.review_policy_id,
                review_policy_version=excluded.review_policy_version,
                review_timestamp_utc=excluded.review_timestamp_utc,
                usable_keypoints=excluded.usable_keypoints,
                total_keypoints=excluded.total_keypoints,
                usable_keypoints_rate=excluded.usable_keypoints_rate,
                raw_keypoints_success_rate=excluded.raw_keypoints_success_rate,
                raw_keypoints_successful=excluded.raw_keypoints_successful,
                quality_updated_utc=excluded.quality_updated_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_detect_performance(
        self,
        *,
        dataset_id: str,
        detect_run: str,
        detect_created_utc: Optional[str],
        recording_id: Optional[str],
        zarr_use: Optional[str],
        detection_method: Optional[str],
        model_run_id: Optional[str],
        model_set_id: Optional[str],
        model_path: Optional[str],
        model_name: Optional[str],
        coverage_percent: Optional[float],
        frames_with_detections: Optional[int],
        frames_zero_detections: Optional[int],
        total_frames: Optional[int],
        mean_confidence: Optional[float],
        min_confidence: Optional[float],
        max_confidence: Optional[float],
        inference_duration_seconds: Optional[float],
        inference_average_fps: Optional[float],
        inference_avg_batch_ms: Optional[float],
        inference_avg_read_ms: Optional[float],
        conf_threshold: Optional[float],
        iou_threshold: Optional[float],
        batch_size: Optional[int],
        inference_width: Optional[int],
        inference_height: Optional[int],
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
    ) -> None:
        payload = {
            "dataset_id": str(dataset_id),
            "detect_run": str(detect_run),
            "detect_created_utc": detect_created_utc,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "detection_method": detection_method,
            "model_run_id": model_run_id,
            "model_set_id": model_set_id,
            "model_path": model_path,
            "model_name": model_name,
            "coverage_percent": coverage_percent,
            "frames_with_detections": frames_with_detections,
            "frames_zero_detections": frames_zero_detections,
            "total_frames": total_frames,
            "mean_confidence": mean_confidence,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "inference_duration_seconds": inference_duration_seconds,
            "inference_average_fps": inference_average_fps,
            "inference_avg_batch_ms": inference_avg_batch_ms,
            "inference_avg_read_ms": inference_avg_read_ms,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "batch_size": batch_size,
            "inference_width": inference_width,
            "inference_height": inference_height,
            "zarr_mtime_ns": zarr_mtime_ns,
            "updated_utc": updated_utc or _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO detect_performance (
                dataset_id, detect_run, detect_created_utc, recording_id, zarr_use,
                detection_method, model_run_id, model_set_id, model_path, model_name,
                coverage_percent, frames_with_detections, frames_zero_detections, total_frames,
                mean_confidence, min_confidence, max_confidence,
                inference_duration_seconds, inference_average_fps, inference_avg_batch_ms, inference_avg_read_ms,
                conf_threshold, iou_threshold, batch_size, inference_width, inference_height,
                zarr_mtime_ns, updated_utc
            )
            VALUES (
                :dataset_id, :detect_run, :detect_created_utc, :recording_id, :zarr_use,
                :detection_method, :model_run_id, :model_set_id, :model_path, :model_name,
                :coverage_percent, :frames_with_detections, :frames_zero_detections, :total_frames,
                :mean_confidence, :min_confidence, :max_confidence,
                :inference_duration_seconds, :inference_average_fps, :inference_avg_batch_ms, :inference_avg_read_ms,
                :conf_threshold, :iou_threshold, :batch_size, :inference_width, :inference_height,
                :zarr_mtime_ns, :updated_utc
            )
            ON CONFLICT(dataset_id, detect_run) DO UPDATE SET
                detect_created_utc=excluded.detect_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                detection_method=excluded.detection_method,
                model_run_id=excluded.model_run_id,
                model_set_id=excluded.model_set_id,
                model_path=excluded.model_path,
                model_name=excluded.model_name,
                coverage_percent=excluded.coverage_percent,
                frames_with_detections=excluded.frames_with_detections,
                frames_zero_detections=excluded.frames_zero_detections,
                total_frames=excluded.total_frames,
                mean_confidence=excluded.mean_confidence,
                min_confidence=excluded.min_confidence,
                max_confidence=excluded.max_confidence,
                inference_duration_seconds=excluded.inference_duration_seconds,
                inference_average_fps=excluded.inference_average_fps,
                inference_avg_batch_ms=excluded.inference_avg_batch_ms,
                inference_avg_read_ms=excluded.inference_avg_read_ms,
                conf_threshold=excluded.conf_threshold,
                iou_threshold=excluded.iou_threshold,
                batch_size=excluded.batch_size,
                inference_width=excluded.inference_width,
                inference_height=excluded.inference_height,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_keypoint_performance(
        self,
        *,
        dataset_id: str,
        keypoint_run: str,
        keypoint_created_utc: Optional[str],
        recording_id: Optional[str],
        zarr_use: Optional[str],
        keypoint_method: Optional[str],
        model_run_id: Optional[str],
        model_set_id: Optional[str],
        model_path: Optional[str],
        model_name: Optional[str],
        source_crop_run: Optional[str],
        source_detect_run: Optional[str],
        source_refined_run: Optional[str],
        total_rois: Optional[int],
        successful_detections: Optional[int],
        failed_detections: Optional[int],
        success_rate_percent: Optional[float],
        frames_with_keypoints: Optional[int],
        mean_confidence: Optional[float],
        duration_seconds: Optional[float],
        inference_duration_seconds: Optional[float],
        keypoints_per_second: Optional[float],
        inference_average_fps: Optional[float],
        batch_size: Optional[int],
        imgsz: Optional[str],
        conf_threshold: Optional[float],
        iou_threshold: Optional[float],
        summary_statistics_json: Optional[str],
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
    ) -> None:
        payload = {
            "dataset_id": str(dataset_id),
            "keypoint_run": str(keypoint_run),
            "keypoint_created_utc": keypoint_created_utc,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "keypoint_method": keypoint_method,
            "model_run_id": model_run_id,
            "model_set_id": model_set_id,
            "model_path": model_path,
            "model_name": model_name,
            "source_crop_run": source_crop_run,
            "source_detect_run": source_detect_run,
            "source_refined_run": source_refined_run,
            "total_rois": total_rois,
            "successful_detections": successful_detections,
            "failed_detections": failed_detections,
            "success_rate_percent": success_rate_percent,
            "frames_with_keypoints": frames_with_keypoints,
            "mean_confidence": mean_confidence,
            "duration_seconds": duration_seconds,
            "inference_duration_seconds": inference_duration_seconds,
            "keypoints_per_second": keypoints_per_second,
            "inference_average_fps": inference_average_fps,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "summary_statistics_json": summary_statistics_json,
            "zarr_mtime_ns": zarr_mtime_ns,
            "updated_utc": updated_utc or _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO keypoint_performance (
                dataset_id, keypoint_run, keypoint_created_utc, recording_id, zarr_use,
                keypoint_method, model_run_id, model_set_id, model_path, model_name,
                source_crop_run, source_detect_run, source_refined_run,
                total_rois, successful_detections, failed_detections, success_rate_percent,
                frames_with_keypoints, mean_confidence,
                duration_seconds, inference_duration_seconds, keypoints_per_second, inference_average_fps,
                batch_size, imgsz, conf_threshold, iou_threshold, summary_statistics_json,
                zarr_mtime_ns, updated_utc
            )
            VALUES (
                :dataset_id, :keypoint_run, :keypoint_created_utc, :recording_id, :zarr_use,
                :keypoint_method, :model_run_id, :model_set_id, :model_path, :model_name,
                :source_crop_run, :source_detect_run, :source_refined_run,
                :total_rois, :successful_detections, :failed_detections, :success_rate_percent,
                :frames_with_keypoints, :mean_confidence,
                :duration_seconds, :inference_duration_seconds, :keypoints_per_second, :inference_average_fps,
                :batch_size, :imgsz, :conf_threshold, :iou_threshold, :summary_statistics_json,
                :zarr_mtime_ns, :updated_utc
            )
            ON CONFLICT(dataset_id, keypoint_run) DO UPDATE SET
                keypoint_created_utc=excluded.keypoint_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                keypoint_method=excluded.keypoint_method,
                model_run_id=excluded.model_run_id,
                model_set_id=excluded.model_set_id,
                model_path=excluded.model_path,
                model_name=excluded.model_name,
                source_crop_run=excluded.source_crop_run,
                source_detect_run=excluded.source_detect_run,
                source_refined_run=excluded.source_refined_run,
                total_rois=excluded.total_rois,
                successful_detections=excluded.successful_detections,
                failed_detections=excluded.failed_detections,
                success_rate_percent=excluded.success_rate_percent,
                frames_with_keypoints=excluded.frames_with_keypoints,
                mean_confidence=excluded.mean_confidence,
                duration_seconds=excluded.duration_seconds,
                inference_duration_seconds=excluded.inference_duration_seconds,
                keypoints_per_second=excluded.keypoints_per_second,
                inference_average_fps=excluded.inference_average_fps,
                batch_size=excluded.batch_size,
                imgsz=excluded.imgsz,
                conf_threshold=excluded.conf_threshold,
                iou_threshold=excluded.iou_threshold,
                summary_statistics_json=excluded.summary_statistics_json,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_eye_mask_performance(
        self,
        *,
        dataset_id: str,
        stage_group: str,
        run_name: str,
        run_created_utc: Optional[str],
        recording_id: Optional[str],
        zarr_use: Optional[str],
        method: Optional[str],
        source_crop_run: Optional[str],
        source_keypoint_group: Optional[str],
        source_keypoints_run: Optional[str],
        source_eye_masks_run: Optional[str],
        source_eye_masks_method: Optional[str],
        total_rois: Optional[int],
        successful_eyes: Optional[int],
        successful_roi_pairs: Optional[int],
        successful_roi_pair_rate: Optional[float],
        duration_seconds: Optional[float],
        rois_per_second: Optional[float],
        inference_duration_seconds: Optional[float],
        inference_average_fps: Optional[float],
        reason_counts_json: Optional[str],
        summary_statistics_json: Optional[str],
        review_state: Optional[str] = None,
        review_method: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        review_reviewer: Optional[str] = None,
        review_timestamp_utc: Optional[str] = None,
        source_keypoint_stale_state: Optional[str] = None,
        source_keypoint_stale_reason: Optional[str] = None,
        source_keypoint_stale_timestamp_utc: Optional[str] = None,
        source_keypoint_stale_json: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        lifecycle_reason: Optional[str] = None,
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
    ) -> None:
        payload = {
            "dataset_id": str(dataset_id),
            "stage_group": str(stage_group),
            "run_name": str(run_name),
            "run_created_utc": run_created_utc,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "method": method,
            "source_crop_run": source_crop_run,
            "source_keypoint_group": source_keypoint_group,
            "source_keypoints_run": source_keypoints_run,
            "source_eye_masks_run": source_eye_masks_run,
            "source_eye_masks_method": source_eye_masks_method,
            "total_rois": total_rois,
            "successful_eyes": successful_eyes,
            "successful_roi_pairs": successful_roi_pairs,
            "successful_roi_pair_rate": successful_roi_pair_rate,
            "duration_seconds": duration_seconds,
            "rois_per_second": rois_per_second,
            "inference_duration_seconds": inference_duration_seconds,
            "inference_average_fps": inference_average_fps,
            "reason_counts_json": reason_counts_json,
            "summary_statistics_json": summary_statistics_json,
            "review_state": review_state,
            "review_method": review_method,
            "review_intended_use": review_intended_use,
            "review_reviewer": review_reviewer,
            "review_timestamp_utc": review_timestamp_utc,
            "source_keypoint_stale_state": source_keypoint_stale_state,
            "source_keypoint_stale_reason": source_keypoint_stale_reason,
            "source_keypoint_stale_timestamp_utc": source_keypoint_stale_timestamp_utc,
            "source_keypoint_stale_json": source_keypoint_stale_json,
            "lifecycle_state": lifecycle_state,
            "lifecycle_reason": lifecycle_reason,
            "zarr_mtime_ns": zarr_mtime_ns,
            "updated_utc": updated_utc or _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO eye_mask_performance (
                dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                method, source_crop_run, source_keypoint_group, source_keypoints_run,
                source_eye_masks_run, source_eye_masks_method,
                total_rois, successful_eyes, successful_roi_pairs, successful_roi_pair_rate,
                duration_seconds, rois_per_second, inference_duration_seconds, inference_average_fps,
                reason_counts_json, summary_statistics_json,
                review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                source_keypoint_stale_state, source_keypoint_stale_reason, source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json, lifecycle_state, lifecycle_reason,
                zarr_mtime_ns, updated_utc
            )
            VALUES (
                :dataset_id, :stage_group, :run_name, :run_created_utc, :recording_id, :zarr_use,
                :method, :source_crop_run, :source_keypoint_group, :source_keypoints_run,
                :source_eye_masks_run, :source_eye_masks_method,
                :total_rois, :successful_eyes, :successful_roi_pairs, :successful_roi_pair_rate,
                :duration_seconds, :rois_per_second, :inference_duration_seconds, :inference_average_fps,
                :reason_counts_json, :summary_statistics_json,
                :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                :source_keypoint_stale_state, :source_keypoint_stale_reason, :source_keypoint_stale_timestamp_utc,
                :source_keypoint_stale_json, :lifecycle_state, :lifecycle_reason,
                :zarr_mtime_ns, :updated_utc
            )
            ON CONFLICT(dataset_id, stage_group, run_name) DO UPDATE SET
                run_created_utc=excluded.run_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                method=excluded.method,
                source_crop_run=excluded.source_crop_run,
                source_keypoint_group=excluded.source_keypoint_group,
                source_keypoints_run=excluded.source_keypoints_run,
                source_eye_masks_run=excluded.source_eye_masks_run,
                source_eye_masks_method=excluded.source_eye_masks_method,
                total_rois=excluded.total_rois,
                successful_eyes=excluded.successful_eyes,
                successful_roi_pairs=excluded.successful_roi_pairs,
                successful_roi_pair_rate=excluded.successful_roi_pair_rate,
                duration_seconds=excluded.duration_seconds,
                rois_per_second=excluded.rois_per_second,
                inference_duration_seconds=excluded.inference_duration_seconds,
                inference_average_fps=excluded.inference_average_fps,
                reason_counts_json=excluded.reason_counts_json,
                summary_statistics_json=excluded.summary_statistics_json,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_timestamp_utc=excluded.review_timestamp_utc,
                source_keypoint_stale_state=excluded.source_keypoint_stale_state,
                source_keypoint_stale_reason=excluded.source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc=excluded.source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json=excluded.source_keypoint_stale_json,
                lifecycle_state=excluded.lifecycle_state,
                lifecycle_reason=excluded.lifecycle_reason,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_subject_mask_performance(
        self,
        *,
        dataset_id: str,
        stage_group: str,
        run_name: str,
        run_created_utc: Optional[str],
        recording_id: Optional[str],
        zarr_use: Optional[str],
        subject_mask_method: Optional[str],
        label_schema_id: Optional[str],
        source_crop_run: Optional[str],
        source_keypoint_group: Optional[str],
        source_keypoints_run: Optional[str],
        source_subject_mask_run: Optional[str],
        source_subject_mask_method: Optional[str],
        run_semantics: Optional[str],
        probability_semantics: Optional[str],
        source_background_run: Optional[str],
        source_background_array: Optional[str],
        source_dish_mask_array: Optional[str],
        tuning_source: Optional[str],
        tuning_timestamp: Optional[str],
        total_rois: Optional[int],
        rows_with_any_mask: Optional[int],
        coverage_percent: Optional[float],
        duration_seconds: Optional[float],
        rois_per_second: Optional[float],
        available_component_count: Optional[int],
        available_components_json: Optional[str],
        unavailable_components_json: Optional[str],
        component_review_states_json: Optional[str],
        eye_component_mode: Optional[str],
        reason_counts_json: Optional[str],
        summary_statistics_json: Optional[str],
        review_state: Optional[str] = None,
        review_method: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        review_reviewer: Optional[str] = None,
        review_timestamp_utc: Optional[str] = None,
        source_subject_mask_stale_state: Optional[str] = None,
        source_subject_mask_stale_reason: Optional[str] = None,
        source_subject_mask_stale_timestamp_utc: Optional[str] = None,
        source_subject_mask_stale_json: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        lifecycle_reason: Optional[str] = None,
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
    ) -> None:
        payload = {
            "dataset_id": str(dataset_id),
            "stage_group": str(stage_group),
            "run_name": str(run_name),
            "run_created_utc": run_created_utc,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "subject_mask_method": subject_mask_method,
            "label_schema_id": label_schema_id,
            "source_crop_run": source_crop_run,
            "source_keypoint_group": source_keypoint_group,
            "source_keypoints_run": source_keypoints_run,
            "source_subject_mask_run": source_subject_mask_run,
            "source_subject_mask_method": source_subject_mask_method,
            "run_semantics": run_semantics,
            "probability_semantics": probability_semantics,
            "source_background_run": source_background_run,
            "source_background_array": source_background_array,
            "source_dish_mask_array": source_dish_mask_array,
            "tuning_source": tuning_source,
            "tuning_timestamp": tuning_timestamp,
            "total_rois": total_rois,
            "rows_with_any_mask": rows_with_any_mask,
            "coverage_percent": coverage_percent,
            "duration_seconds": duration_seconds,
            "rois_per_second": rois_per_second,
            "available_component_count": available_component_count,
            "available_components_json": available_components_json,
            "unavailable_components_json": unavailable_components_json,
            "component_review_states_json": component_review_states_json,
            "eye_component_mode": eye_component_mode,
            "reason_counts_json": reason_counts_json,
            "summary_statistics_json": summary_statistics_json,
            "review_state": review_state,
            "review_method": review_method,
            "review_intended_use": review_intended_use,
            "review_reviewer": review_reviewer,
            "review_timestamp_utc": review_timestamp_utc,
            "source_subject_mask_stale_state": source_subject_mask_stale_state,
            "source_subject_mask_stale_reason": source_subject_mask_stale_reason,
            "source_subject_mask_stale_timestamp_utc": source_subject_mask_stale_timestamp_utc,
            "source_subject_mask_stale_json": source_subject_mask_stale_json,
            "lifecycle_state": lifecycle_state,
            "lifecycle_reason": lifecycle_reason,
            "zarr_mtime_ns": zarr_mtime_ns,
            "updated_utc": updated_utc or _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO subject_mask_performance (
                dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                subject_mask_method, label_schema_id,
                source_crop_run, source_keypoint_group, source_keypoints_run,
                source_subject_mask_run, source_subject_mask_method,
                run_semantics, probability_semantics,
                source_background_run, source_background_array, source_dish_mask_array,
                tuning_source, tuning_timestamp,
                total_rois, rows_with_any_mask, coverage_percent,
                duration_seconds, rois_per_second,
                available_component_count, available_components_json, unavailable_components_json,
                component_review_states_json, eye_component_mode,
                reason_counts_json, summary_statistics_json,
                review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                source_subject_mask_stale_state, source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc, source_subject_mask_stale_json,
                lifecycle_state, lifecycle_reason,
                zarr_mtime_ns, updated_utc
            )
            VALUES (
                :dataset_id, :stage_group, :run_name, :run_created_utc, :recording_id, :zarr_use,
                :subject_mask_method, :label_schema_id,
                :source_crop_run, :source_keypoint_group, :source_keypoints_run,
                :source_subject_mask_run, :source_subject_mask_method,
                :run_semantics, :probability_semantics,
                :source_background_run, :source_background_array, :source_dish_mask_array,
                :tuning_source, :tuning_timestamp,
                :total_rois, :rows_with_any_mask, :coverage_percent,
                :duration_seconds, :rois_per_second,
                :available_component_count, :available_components_json, :unavailable_components_json,
                :component_review_states_json, :eye_component_mode,
                :reason_counts_json, :summary_statistics_json,
                :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                :source_subject_mask_stale_state, :source_subject_mask_stale_reason,
                :source_subject_mask_stale_timestamp_utc, :source_subject_mask_stale_json,
                :lifecycle_state, :lifecycle_reason,
                :zarr_mtime_ns, :updated_utc
            )
            ON CONFLICT(dataset_id, stage_group, run_name) DO UPDATE SET
                run_created_utc=excluded.run_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                subject_mask_method=excluded.subject_mask_method,
                label_schema_id=excluded.label_schema_id,
                source_crop_run=excluded.source_crop_run,
                source_keypoint_group=excluded.source_keypoint_group,
                source_keypoints_run=excluded.source_keypoints_run,
                source_subject_mask_run=excluded.source_subject_mask_run,
                source_subject_mask_method=excluded.source_subject_mask_method,
                run_semantics=excluded.run_semantics,
                probability_semantics=excluded.probability_semantics,
                source_background_run=excluded.source_background_run,
                source_background_array=excluded.source_background_array,
                source_dish_mask_array=excluded.source_dish_mask_array,
                tuning_source=excluded.tuning_source,
                tuning_timestamp=excluded.tuning_timestamp,
                total_rois=excluded.total_rois,
                rows_with_any_mask=excluded.rows_with_any_mask,
                coverage_percent=excluded.coverage_percent,
                duration_seconds=excluded.duration_seconds,
                rois_per_second=excluded.rois_per_second,
                available_component_count=excluded.available_component_count,
                available_components_json=excluded.available_components_json,
                unavailable_components_json=excluded.unavailable_components_json,
                component_review_states_json=excluded.component_review_states_json,
                eye_component_mode=excluded.eye_component_mode,
                reason_counts_json=excluded.reason_counts_json,
                summary_statistics_json=excluded.summary_statistics_json,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_timestamp_utc=excluded.review_timestamp_utc,
                source_subject_mask_stale_state=excluded.source_subject_mask_stale_state,
                source_subject_mask_stale_reason=excluded.source_subject_mask_stale_reason,
                source_subject_mask_stale_timestamp_utc=excluded.source_subject_mask_stale_timestamp_utc,
                source_subject_mask_stale_json=excluded.source_subject_mask_stale_json,
                lifecycle_state=excluded.lifecycle_state,
                lifecycle_reason=excluded.lifecycle_reason,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_subject_mask_component_quality(
        self,
        *,
        dataset_id: str,
        stage_group: str,
        run_name: str,
        component_name: str,
        component_family: Optional[str],
        run_created_utc: Optional[str],
        recording_id: Optional[str],
        zarr_use: Optional[str],
        subject_mask_method: Optional[str],
        label_schema_id: Optional[str],
        eye_component_mode: Optional[str],
        source_subject_mask_run: Optional[str],
        available: Optional[int],
        review_state: Optional[str],
        review_method: Optional[str],
        review_intended_use: Optional[str],
        review_reviewer: Optional[str],
        review_timestamp_utc: Optional[str],
        total_rois: Optional[int],
        rows_with_component_mask: Optional[int],
        rows_with_component_mask_rate: Optional[float],
        lifecycle_state: Optional[str] = None,
        lifecycle_reason: Optional[str] = None,
        quality_updated_utc: Optional[str] = None,
        zarr_mtime_ns: Optional[int] = None,
    ) -> None:
        payload = {
            "dataset_id": str(dataset_id),
            "stage_group": str(stage_group),
            "run_name": str(run_name),
            "component_name": str(component_name),
            "component_family": component_family,
            "run_created_utc": run_created_utc,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "subject_mask_method": subject_mask_method,
            "label_schema_id": label_schema_id,
            "eye_component_mode": eye_component_mode,
            "source_subject_mask_run": source_subject_mask_run,
            "available": available,
            "review_state": review_state,
            "review_method": review_method,
            "review_intended_use": review_intended_use,
            "review_reviewer": review_reviewer,
            "review_timestamp_utc": review_timestamp_utc,
            "total_rois": total_rois,
            "rows_with_component_mask": rows_with_component_mask,
            "rows_with_component_mask_rate": rows_with_component_mask_rate,
            "lifecycle_state": lifecycle_state,
            "lifecycle_reason": lifecycle_reason,
            "quality_updated_utc": quality_updated_utc or _utc_now(),
            "zarr_mtime_ns": zarr_mtime_ns,
        }
        self.conn.execute(
            """
            INSERT INTO subject_mask_component_quality (
                dataset_id, stage_group, run_name, component_name, component_family,
                run_created_utc, recording_id, zarr_use, subject_mask_method, label_schema_id,
                eye_component_mode, source_subject_mask_run, available,
                review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                total_rois, rows_with_component_mask, rows_with_component_mask_rate,
                lifecycle_state, lifecycle_reason, quality_updated_utc, zarr_mtime_ns
            )
            VALUES (
                :dataset_id, :stage_group, :run_name, :component_name, :component_family,
                :run_created_utc, :recording_id, :zarr_use, :subject_mask_method, :label_schema_id,
                :eye_component_mode, :source_subject_mask_run, :available,
                :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                :total_rois, :rows_with_component_mask, :rows_with_component_mask_rate,
                :lifecycle_state, :lifecycle_reason, :quality_updated_utc, :zarr_mtime_ns
            )
            ON CONFLICT(dataset_id, stage_group, run_name, component_name) DO UPDATE SET
                component_family=excluded.component_family,
                run_created_utc=excluded.run_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                subject_mask_method=excluded.subject_mask_method,
                label_schema_id=excluded.label_schema_id,
                eye_component_mode=excluded.eye_component_mode,
                source_subject_mask_run=excluded.source_subject_mask_run,
                available=excluded.available,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_timestamp_utc=excluded.review_timestamp_utc,
                total_rois=excluded.total_rois,
                rows_with_component_mask=excluded.rows_with_component_mask,
                rows_with_component_mask_rate=excluded.rows_with_component_mask_rate,
                lifecycle_state=excluded.lifecycle_state,
                lifecycle_reason=excluded.lifecycle_reason,
                quality_updated_utc=excluded.quality_updated_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns;
            """,
            payload,
        )
        self.conn.commit()

    def _profile_duplicate_context_write_policy(self, dataset_id: str) -> Tuple[bool, bool]:
        row = self.conn.execute(
            """
            SELECT
                EXISTS(
                    SELECT 1
                    FROM datasets d
                    INNER JOIN recordings r ON r.recording_id = d.recording_id
                    WHERE d.dataset_id = ?
                ) AS has_recording_context,
                EXISTS(
                    SELECT 1
                    FROM dataset_context_current dcc
                    WHERE dcc.dataset_id = ?
                      AND dcc.subject_context_source = 'normalized'
                ) AS has_normalized_subject_context;
            """,
            (str(dataset_id), str(dataset_id)),
        ).fetchone()
        has_recording_context = bool(
            row is not None and int(row["has_recording_context"] or 0) == 1
        )
        has_normalized_subject_context = bool(
            row is not None and int(row["has_normalized_subject_context"] or 0) == 1
        )
        return (not has_recording_context, not has_normalized_subject_context)

    @staticmethod
    def _apply_profile_duplicate_context_write_policy(
        payload: Dict[str, Any],
        *,
        write_legacy_recording_context_snapshot: bool,
        write_legacy_biology_snapshot: bool,
    ) -> Dict[str, Any]:
        normalized = dict(payload)
        if not write_legacy_recording_context_snapshot:
            for field in (
                "rig_id",
                "camera_id",
                "arena_id",
                "dish_design",
                "canvas_name",
                "protocol_name",
            ):
                normalized[field] = None
        if not write_legacy_biology_snapshot:
            for field in ("genotype", "dpf_at_acquisition"):
                normalized[field] = None
        return normalized

    @staticmethod
    def _profile_duplicate_context_update_sql(
        table_name: str,
        *,
        write_legacy_recording_context_snapshot: bool,
        write_legacy_biology_snapshot: bool,
    ) -> str:
        fragments: List[str] = []
        for field in (
            "rig_id",
            "camera_id",
            "arena_id",
            "dish_design",
            "canvas_name",
            "protocol_name",
        ):
            if write_legacy_recording_context_snapshot:
                fragments.append(f"{field}=excluded.{field},")
            else:
                fragments.append(f"{field}=COALESCE({table_name}.{field}, excluded.{field}),")
        for field in ("genotype", "dpf_at_acquisition"):
            if write_legacy_biology_snapshot:
                fragments.append(f"{field}=excluded.{field},")
            else:
                fragments.append(f"{field}=COALESCE({table_name}.{field}, excluded.{field}),")
        return "\n                ".join(fragments)

    def upsert_detection_data_profile(
        self,
        *,
        dataset_id: str,
        profile_run: str,
        recording_id: Optional[str],
        zarr_use: Optional[str],
        detection_type: Optional[str],
        detection_path: Optional[str],
        profile_created_utc: Optional[str],
        frames_total: Optional[int],
        frames_with_detections: Optional[int],
        coverage_percent: Optional[float],
        detections_total: Optional[int],
        detections_per_frame_p50: Optional[float],
        detections_per_frame_p90: Optional[float],
        w_p10: Optional[float],
        w_p50: Optional[float],
        w_p90: Optional[float],
        h_p10: Optional[float],
        h_p50: Optional[float],
        h_p90: Optional[float],
        area_p10: Optional[float],
        area_p50: Optional[float],
        area_p90: Optional[float],
        aspect_ratio_p10: Optional[float],
        aspect_ratio_p50: Optional[float],
        aspect_ratio_p90: Optional[float],
        edge_proximity_rate: Optional[float],
        rig_id: Optional[str],
        camera_id: Optional[str],
        arena_id: Optional[str],
        dish_design: Optional[str],
        canvas_name: Optional[str],
        protocol_name: Optional[str],
        profile_json: Optional[str],
        genotype: Optional[str] = None,
        dpf_at_acquisition: Optional[int] = None,
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
    ) -> None:
        write_legacy_recording_context_snapshot, write_legacy_biology_snapshot = (
            self._profile_duplicate_context_write_policy(str(dataset_id))
        )
        duplicate_context_update_sql = self._profile_duplicate_context_update_sql(
            "detection_data_profile",
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        payload = self._apply_profile_duplicate_context_write_policy(
            {
                "dataset_id": str(dataset_id),
                "profile_run": str(profile_run),
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "detection_type": detection_type,
                "detection_path": detection_path,
                "profile_created_utc": profile_created_utc,
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc or _utc_now(),
                "frames_total": frames_total,
                "frames_with_detections": frames_with_detections,
                "coverage_percent": coverage_percent,
                "detections_total": detections_total,
                "detections_per_frame_p50": detections_per_frame_p50,
                "detections_per_frame_p90": detections_per_frame_p90,
                "w_p10": w_p10,
                "w_p50": w_p50,
                "w_p90": w_p90,
                "h_p10": h_p10,
                "h_p50": h_p50,
                "h_p90": h_p90,
                "area_p10": area_p10,
                "area_p50": area_p50,
                "area_p90": area_p90,
                "aspect_ratio_p10": aspect_ratio_p10,
                "aspect_ratio_p50": aspect_ratio_p50,
                "aspect_ratio_p90": aspect_ratio_p90,
                "edge_proximity_rate": edge_proximity_rate,
                "rig_id": rig_id,
                "camera_id": camera_id,
                "arena_id": arena_id,
                "dish_design": dish_design,
                "canvas_name": canvas_name,
                "protocol_name": protocol_name,
                "genotype": genotype,
                "dpf_at_acquisition": dpf_at_acquisition,
                "profile_json": profile_json,
            },
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        self.conn.execute(
            f"""
            INSERT INTO detection_data_profile (
                dataset_id, profile_run, recording_id, zarr_use,
                detection_type, detection_path, profile_created_utc,
                zarr_mtime_ns, updated_utc,
                frames_total, frames_with_detections, coverage_percent, detections_total,
                detections_per_frame_p50, detections_per_frame_p90,
                w_p10, w_p50, w_p90,
                h_p10, h_p50, h_p90,
                area_p10, area_p50, area_p90,
                aspect_ratio_p10, aspect_ratio_p50, aspect_ratio_p90,
                edge_proximity_rate,
                rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name,
                genotype, dpf_at_acquisition,
                profile_json
            )
            VALUES (
                :dataset_id, :profile_run, :recording_id, :zarr_use,
                :detection_type, :detection_path, :profile_created_utc,
                :zarr_mtime_ns, :updated_utc,
                :frames_total, :frames_with_detections, :coverage_percent, :detections_total,
                :detections_per_frame_p50, :detections_per_frame_p90,
                :w_p10, :w_p50, :w_p90,
                :h_p10, :h_p50, :h_p90,
                :area_p10, :area_p50, :area_p90,
                :aspect_ratio_p10, :aspect_ratio_p50, :aspect_ratio_p90,
                :edge_proximity_rate,
                :rig_id, :camera_id, :arena_id, :dish_design, :canvas_name, :protocol_name,
                :genotype, :dpf_at_acquisition,
                :profile_json
            )
            ON CONFLICT(dataset_id, profile_run) DO UPDATE SET
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                detection_type=excluded.detection_type,
                detection_path=excluded.detection_path,
                profile_created_utc=excluded.profile_created_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc,
                frames_total=excluded.frames_total,
                frames_with_detections=excluded.frames_with_detections,
                coverage_percent=excluded.coverage_percent,
                detections_total=excluded.detections_total,
                detections_per_frame_p50=excluded.detections_per_frame_p50,
                detections_per_frame_p90=excluded.detections_per_frame_p90,
                w_p10=excluded.w_p10,
                w_p50=excluded.w_p50,
                w_p90=excluded.w_p90,
                h_p10=excluded.h_p10,
                h_p50=excluded.h_p50,
                h_p90=excluded.h_p90,
                area_p10=excluded.area_p10,
                area_p50=excluded.area_p50,
                area_p90=excluded.area_p90,
                aspect_ratio_p10=excluded.aspect_ratio_p10,
                aspect_ratio_p50=excluded.aspect_ratio_p50,
                aspect_ratio_p90=excluded.aspect_ratio_p90,
                edge_proximity_rate=excluded.edge_proximity_rate,
                {duplicate_context_update_sql}
                profile_json=excluded.profile_json;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_training_image_profile(
        self,
        *,
        dataset_id: str,
        profile_run: str,
        recording_id: Optional[str],
        zarr_use: Optional[str],
        source_frame_array: Optional[str],
        profile_created_utc: Optional[str],
        frames_total: Optional[int],
        frames_profiled: Optional[int],
        mean_intensity_p50: Optional[float],
        contrast_p50: Optional[float],
        sharpness_p50: Optional[float],
        clip_dark_rate_mean: Optional[float],
        clip_bright_rate_mean: Optional[float],
        illumination_center_edge_p50: Optional[float],
        illumination_slope_x_p50: Optional[float],
        illumination_slope_y_p50: Optional[float],
        fish_bg_contrast_p50: Optional[float],
        rig_id: Optional[str],
        camera_id: Optional[str],
        arena_id: Optional[str],
        dish_design: Optional[str],
        canvas_name: Optional[str],
        protocol_name: Optional[str],
        profile_json: Optional[str],
        genotype: Optional[str] = None,
        dpf_at_acquisition: Optional[int] = None,
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
    ) -> None:
        write_legacy_recording_context_snapshot, write_legacy_biology_snapshot = (
            self._profile_duplicate_context_write_policy(str(dataset_id))
        )
        duplicate_context_update_sql = self._profile_duplicate_context_update_sql(
            "training_image_profile",
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        payload = self._apply_profile_duplicate_context_write_policy(
            {
                "dataset_id": str(dataset_id),
                "profile_run": str(profile_run),
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "source_frame_array": source_frame_array,
                "profile_created_utc": profile_created_utc,
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc or _utc_now(),
                "frames_total": frames_total,
                "frames_profiled": frames_profiled,
                "mean_intensity_p50": mean_intensity_p50,
                "contrast_p50": contrast_p50,
                "sharpness_p50": sharpness_p50,
                "clip_dark_rate_mean": clip_dark_rate_mean,
                "clip_bright_rate_mean": clip_bright_rate_mean,
                "illumination_center_edge_p50": illumination_center_edge_p50,
                "illumination_slope_x_p50": illumination_slope_x_p50,
                "illumination_slope_y_p50": illumination_slope_y_p50,
                "fish_bg_contrast_p50": fish_bg_contrast_p50,
                "rig_id": rig_id,
                "camera_id": camera_id,
                "arena_id": arena_id,
                "dish_design": dish_design,
                "canvas_name": canvas_name,
                "protocol_name": protocol_name,
                "genotype": genotype,
                "dpf_at_acquisition": dpf_at_acquisition,
                "profile_json": profile_json,
            },
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        self.conn.execute(
            f"""
            INSERT INTO training_image_profile (
                dataset_id, profile_run, recording_id, zarr_use,
                source_frame_array, profile_created_utc,
                zarr_mtime_ns, updated_utc,
                frames_total, frames_profiled,
                mean_intensity_p50, contrast_p50, sharpness_p50,
                clip_dark_rate_mean, clip_bright_rate_mean,
                illumination_center_edge_p50, illumination_slope_x_p50, illumination_slope_y_p50,
                fish_bg_contrast_p50,
                rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name,
                genotype, dpf_at_acquisition,
                profile_json
            )
            VALUES (
                :dataset_id, :profile_run, :recording_id, :zarr_use,
                :source_frame_array, :profile_created_utc,
                :zarr_mtime_ns, :updated_utc,
                :frames_total, :frames_profiled,
                :mean_intensity_p50, :contrast_p50, :sharpness_p50,
                :clip_dark_rate_mean, :clip_bright_rate_mean,
                :illumination_center_edge_p50, :illumination_slope_x_p50, :illumination_slope_y_p50,
                :fish_bg_contrast_p50,
                :rig_id, :camera_id, :arena_id, :dish_design, :canvas_name, :protocol_name,
                :genotype, :dpf_at_acquisition,
                :profile_json
            )
            ON CONFLICT(dataset_id, profile_run) DO UPDATE SET
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                source_frame_array=excluded.source_frame_array,
                profile_created_utc=excluded.profile_created_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc,
                frames_total=excluded.frames_total,
                frames_profiled=excluded.frames_profiled,
                mean_intensity_p50=excluded.mean_intensity_p50,
                contrast_p50=excluded.contrast_p50,
                sharpness_p50=excluded.sharpness_p50,
                clip_dark_rate_mean=excluded.clip_dark_rate_mean,
                clip_bright_rate_mean=excluded.clip_bright_rate_mean,
                illumination_center_edge_p50=excluded.illumination_center_edge_p50,
                illumination_slope_x_p50=excluded.illumination_slope_x_p50,
                illumination_slope_y_p50=excluded.illumination_slope_y_p50,
                fish_bg_contrast_p50=excluded.fish_bg_contrast_p50,
                {duplicate_context_update_sql}
                profile_json=excluded.profile_json;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_keypoint_data_profile(
        self,
        *,
        dataset_id: str,
        profile_run: str,
        recording_id: Optional[str],
        zarr_use: Optional[str],
        keypoint_method: Optional[str],
        source_keypoint_path: Optional[str],
        source_keypoint_run: Optional[str],
        skeleton_id: Optional[str],
        kpt_shape: Optional[str],
        profile_created_utc: Optional[str],
        rows_total: Optional[int],
        rows_usable: Optional[int],
        usable_keypoints_total: Optional[int],
        usable_rate: Optional[float],
        confidence_valid_rate: Optional[float],
        geometry_valid_rate: Optional[float],
        triangle_area_p10: Optional[float],
        triangle_area_p50: Optional[float],
        triangle_area_p90: Optional[float],
        min_angle_p10: Optional[float],
        min_angle_p50: Optional[float],
        min_angle_p90: Optional[float],
        heading_p10: Optional[float],
        heading_p50: Optional[float],
        heading_p90: Optional[float],
        rig_id: Optional[str],
        camera_id: Optional[str],
        arena_id: Optional[str],
        dish_design: Optional[str],
        canvas_name: Optional[str],
        protocol_name: Optional[str],
        profile_json: Optional[str],
        genotype: Optional[str] = None,
        dpf_at_acquisition: Optional[int] = None,
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
        pose_schema_name: Optional[str] = None,
        pose_schema_json: Optional[str] = None,
        heading_computation_source: Optional[str] = None,
        heading_computation_json: Optional[str] = None,
    ) -> None:
        write_legacy_recording_context_snapshot, write_legacy_biology_snapshot = (
            self._profile_duplicate_context_write_policy(str(dataset_id))
        )
        duplicate_context_update_sql = self._profile_duplicate_context_update_sql(
            "keypoint_data_profile",
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        payload = self._apply_profile_duplicate_context_write_policy(
            {
                "dataset_id": str(dataset_id),
                "profile_run": str(profile_run),
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "keypoint_method": keypoint_method,
                "source_keypoint_path": source_keypoint_path,
                "source_keypoint_run": source_keypoint_run,
                "skeleton_id": skeleton_id,
                "kpt_shape": kpt_shape,
                "pose_schema_name": pose_schema_name,
                "pose_schema_json": pose_schema_json,
                "heading_computation_source": heading_computation_source,
                "heading_computation_json": heading_computation_json,
                "profile_created_utc": profile_created_utc,
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc or _utc_now(),
                "rows_total": rows_total,
                "rows_usable": rows_usable,
                "usable_keypoints_total": usable_keypoints_total,
                "usable_rate": usable_rate,
                "confidence_valid_rate": confidence_valid_rate,
                "geometry_valid_rate": geometry_valid_rate,
                "triangle_area_p10": triangle_area_p10,
                "triangle_area_p50": triangle_area_p50,
                "triangle_area_p90": triangle_area_p90,
                "min_angle_p10": min_angle_p10,
                "min_angle_p50": min_angle_p50,
                "min_angle_p90": min_angle_p90,
                "heading_p10": heading_p10,
                "heading_p50": heading_p50,
                "heading_p90": heading_p90,
                "rig_id": rig_id,
                "camera_id": camera_id,
                "arena_id": arena_id,
                "dish_design": dish_design,
                "canvas_name": canvas_name,
                "protocol_name": protocol_name,
                "genotype": genotype,
                "dpf_at_acquisition": dpf_at_acquisition,
                "profile_json": profile_json,
            },
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        self.conn.execute(
            f"""
            INSERT INTO keypoint_data_profile (
                dataset_id, profile_run, recording_id, zarr_use,
                keypoint_method, source_keypoint_path, source_keypoint_run,
                skeleton_id, kpt_shape, pose_schema_name, pose_schema_json,
                heading_computation_source, heading_computation_json, profile_created_utc,
                zarr_mtime_ns, updated_utc,
                rows_total, rows_usable, usable_keypoints_total, usable_rate,
                confidence_valid_rate, geometry_valid_rate,
                triangle_area_p10, triangle_area_p50, triangle_area_p90,
                min_angle_p10, min_angle_p50, min_angle_p90,
                heading_p10, heading_p50, heading_p90,
                rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name,
                genotype, dpf_at_acquisition,
                profile_json
            )
            VALUES (
                :dataset_id, :profile_run, :recording_id, :zarr_use,
                :keypoint_method, :source_keypoint_path, :source_keypoint_run,
                :skeleton_id, :kpt_shape, :pose_schema_name, :pose_schema_json,
                :heading_computation_source, :heading_computation_json, :profile_created_utc,
                :zarr_mtime_ns, :updated_utc,
                :rows_total, :rows_usable, :usable_keypoints_total, :usable_rate,
                :confidence_valid_rate, :geometry_valid_rate,
                :triangle_area_p10, :triangle_area_p50, :triangle_area_p90,
                :min_angle_p10, :min_angle_p50, :min_angle_p90,
                :heading_p10, :heading_p50, :heading_p90,
                :rig_id, :camera_id, :arena_id, :dish_design, :canvas_name, :protocol_name,
                :genotype, :dpf_at_acquisition,
                :profile_json
            )
            ON CONFLICT(dataset_id, profile_run) DO UPDATE SET
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                keypoint_method=excluded.keypoint_method,
                source_keypoint_path=excluded.source_keypoint_path,
                source_keypoint_run=excluded.source_keypoint_run,
                skeleton_id=excluded.skeleton_id,
                kpt_shape=excluded.kpt_shape,
                pose_schema_name=excluded.pose_schema_name,
                pose_schema_json=excluded.pose_schema_json,
                heading_computation_source=excluded.heading_computation_source,
                heading_computation_json=excluded.heading_computation_json,
                profile_created_utc=excluded.profile_created_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc,
                rows_total=excluded.rows_total,
                rows_usable=excluded.rows_usable,
                usable_keypoints_total=excluded.usable_keypoints_total,
                usable_rate=excluded.usable_rate,
                confidence_valid_rate=excluded.confidence_valid_rate,
                geometry_valid_rate=excluded.geometry_valid_rate,
                triangle_area_p10=excluded.triangle_area_p10,
                triangle_area_p50=excluded.triangle_area_p50,
                triangle_area_p90=excluded.triangle_area_p90,
                min_angle_p10=excluded.min_angle_p10,
                min_angle_p50=excluded.min_angle_p50,
                min_angle_p90=excluded.min_angle_p90,
                heading_p10=excluded.heading_p10,
                heading_p50=excluded.heading_p50,
                heading_p90=excluded.heading_p90,
                {duplicate_context_update_sql}
                profile_json=excluded.profile_json;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_eye_mask_data_profile(
        self,
        *,
        dataset_id: str,
        profile_run: str,
        recording_id: Optional[str],
        zarr_use: Optional[str],
        stage_group: Optional[str],
        eye_mask_method: Optional[str],
        source_eye_mask_path: Optional[str],
        source_eye_mask_run: Optional[str],
        source_keypoint_path: Optional[str],
        source_keypoint_run: Optional[str],
        source_crop_run: Optional[str],
        profile_created_utc: Optional[str],
        rows_total: Optional[int],
        rows_usable: Optional[int],
        usable_rate: Optional[float],
        reviewed_rate: Optional[float],
        excluded_rate: Optional[float],
        exclusion_reasons_json: Optional[str],
        ellipse_success_rate: Optional[float],
        pair_success_rate: Optional[float],
        area_p10: Optional[float],
        area_p50: Optional[float],
        area_p90: Optional[float],
        left_area_p10: Optional[float] = None,
        left_area_p50: Optional[float] = None,
        left_area_p90: Optional[float] = None,
        right_area_p10: Optional[float] = None,
        right_area_p50: Optional[float] = None,
        right_area_p90: Optional[float] = None,
        union_area_p10: Optional[float] = None,
        union_area_p50: Optional[float] = None,
        union_area_p90: Optional[float] = None,
        area_lr_ratio_p10: Optional[float] = None,
        area_lr_ratio_p50: Optional[float] = None,
        area_lr_ratio_p90: Optional[float] = None,
        major_axis_p10: Optional[float] = None,
        major_axis_p50: Optional[float] = None,
        major_axis_p90: Optional[float] = None,
        minor_axis_p10: Optional[float] = None,
        minor_axis_p50: Optional[float] = None,
        minor_axis_p90: Optional[float] = None,
        aspect_ratio_p10: Optional[float] = None,
        aspect_ratio_p50: Optional[float] = None,
        aspect_ratio_p90: Optional[float] = None,
        eye_separation_p10: Optional[float] = None,
        eye_separation_p50: Optional[float] = None,
        eye_separation_p90: Optional[float] = None,
        edge_proximity_rate: Optional[float] = None,
        review_state: Optional[str] = None,
        review_method: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        review_timestamp_utc: Optional[str] = None,
        source_keypoint_stale_state: Optional[str] = None,
        source_keypoint_stale_reason: Optional[str] = None,
        source_keypoint_stale_timestamp_utc: Optional[str] = None,
        source_keypoint_stale_json: Optional[str] = None,
        rig_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        arena_id: Optional[str] = None,
        dish_design: Optional[str] = None,
        canvas_name: Optional[str] = None,
        protocol_name: Optional[str] = None,
        genotype: Optional[str] = None,
        dpf_at_acquisition: Optional[int] = None,
        profile_json: Optional[str] = None,
        zarr_mtime_ns: Optional[int] = None,
        updated_utc: Optional[str] = None,
    ) -> None:
        write_legacy_recording_context_snapshot, write_legacy_biology_snapshot = (
            self._profile_duplicate_context_write_policy(str(dataset_id))
        )
        duplicate_context_update_sql = self._profile_duplicate_context_update_sql(
            "eye_mask_data_profile",
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        payload = self._apply_profile_duplicate_context_write_policy(
            {
                "dataset_id": str(dataset_id),
                "profile_run": str(profile_run),
                "recording_id": recording_id,
                "zarr_use": zarr_use,
                "stage_group": stage_group,
                "eye_mask_method": eye_mask_method,
                "source_eye_mask_path": source_eye_mask_path,
                "source_eye_mask_run": source_eye_mask_run,
                "source_keypoint_path": source_keypoint_path,
                "source_keypoint_run": source_keypoint_run,
                "source_crop_run": source_crop_run,
                "profile_created_utc": profile_created_utc,
                "zarr_mtime_ns": zarr_mtime_ns,
                "updated_utc": updated_utc or _utc_now(),
                "rows_total": rows_total,
                "rows_usable": rows_usable,
                "usable_rate": usable_rate,
                "reviewed_rate": reviewed_rate,
                "excluded_rate": excluded_rate,
                "exclusion_reasons_json": exclusion_reasons_json,
                "ellipse_success_rate": ellipse_success_rate,
                "pair_success_rate": pair_success_rate,
                "area_p10": area_p10,
                "area_p50": area_p50,
                "area_p90": area_p90,
                "left_area_p10": left_area_p10,
                "left_area_p50": left_area_p50,
                "left_area_p90": left_area_p90,
                "right_area_p10": right_area_p10,
                "right_area_p50": right_area_p50,
                "right_area_p90": right_area_p90,
                "union_area_p10": union_area_p10,
                "union_area_p50": union_area_p50,
                "union_area_p90": union_area_p90,
                "area_lr_ratio_p10": area_lr_ratio_p10,
                "area_lr_ratio_p50": area_lr_ratio_p50,
                "area_lr_ratio_p90": area_lr_ratio_p90,
                "major_axis_p10": major_axis_p10,
                "major_axis_p50": major_axis_p50,
                "major_axis_p90": major_axis_p90,
                "minor_axis_p10": minor_axis_p10,
                "minor_axis_p50": minor_axis_p50,
                "minor_axis_p90": minor_axis_p90,
                "aspect_ratio_p10": aspect_ratio_p10,
                "aspect_ratio_p50": aspect_ratio_p50,
                "aspect_ratio_p90": aspect_ratio_p90,
                "eye_separation_p10": eye_separation_p10,
                "eye_separation_p50": eye_separation_p50,
                "eye_separation_p90": eye_separation_p90,
                "edge_proximity_rate": edge_proximity_rate,
                "review_state": review_state,
                "review_method": review_method,
                "review_intended_use": review_intended_use,
                "review_timestamp_utc": review_timestamp_utc,
                "source_keypoint_stale_state": source_keypoint_stale_state,
                "source_keypoint_stale_reason": source_keypoint_stale_reason,
                "source_keypoint_stale_timestamp_utc": source_keypoint_stale_timestamp_utc,
                "source_keypoint_stale_json": source_keypoint_stale_json,
                "rig_id": rig_id,
                "camera_id": camera_id,
                "arena_id": arena_id,
                "dish_design": dish_design,
                "canvas_name": canvas_name,
                "protocol_name": protocol_name,
                "genotype": genotype,
                "dpf_at_acquisition": dpf_at_acquisition,
                "profile_json": profile_json,
            },
            write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
            write_legacy_biology_snapshot=write_legacy_biology_snapshot,
        )
        self.conn.execute(
            f"""
            INSERT INTO eye_mask_data_profile (
                dataset_id, profile_run, recording_id, zarr_use,
                stage_group, eye_mask_method,
                source_eye_mask_path, source_eye_mask_run,
                source_keypoint_path, source_keypoint_run, source_crop_run,
                profile_created_utc, zarr_mtime_ns, updated_utc,
                rows_total, rows_usable, usable_rate, reviewed_rate, excluded_rate,
                exclusion_reasons_json, ellipse_success_rate, pair_success_rate,
                area_p10, area_p50, area_p90,
                left_area_p10, left_area_p50, left_area_p90,
                right_area_p10, right_area_p50, right_area_p90,
                union_area_p10, union_area_p50, union_area_p90,
                area_lr_ratio_p10, area_lr_ratio_p50, area_lr_ratio_p90,
                major_axis_p10, major_axis_p50, major_axis_p90,
                minor_axis_p10, minor_axis_p50, minor_axis_p90,
                aspect_ratio_p10, aspect_ratio_p50, aspect_ratio_p90,
                eye_separation_p10, eye_separation_p50, eye_separation_p90,
                edge_proximity_rate,
                review_state, review_method, review_intended_use, review_timestamp_utc,
                source_keypoint_stale_state, source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc, source_keypoint_stale_json,
                rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name,
                genotype, dpf_at_acquisition,
                profile_json
            )
            VALUES (
                :dataset_id, :profile_run, :recording_id, :zarr_use,
                :stage_group, :eye_mask_method,
                :source_eye_mask_path, :source_eye_mask_run,
                :source_keypoint_path, :source_keypoint_run, :source_crop_run,
                :profile_created_utc, :zarr_mtime_ns, :updated_utc,
                :rows_total, :rows_usable, :usable_rate, :reviewed_rate, :excluded_rate,
                :exclusion_reasons_json, :ellipse_success_rate, :pair_success_rate,
                :area_p10, :area_p50, :area_p90,
                :left_area_p10, :left_area_p50, :left_area_p90,
                :right_area_p10, :right_area_p50, :right_area_p90,
                :union_area_p10, :union_area_p50, :union_area_p90,
                :area_lr_ratio_p10, :area_lr_ratio_p50, :area_lr_ratio_p90,
                :major_axis_p10, :major_axis_p50, :major_axis_p90,
                :minor_axis_p10, :minor_axis_p50, :minor_axis_p90,
                :aspect_ratio_p10, :aspect_ratio_p50, :aspect_ratio_p90,
                :eye_separation_p10, :eye_separation_p50, :eye_separation_p90,
                :edge_proximity_rate,
                :review_state, :review_method, :review_intended_use, :review_timestamp_utc,
                :source_keypoint_stale_state, :source_keypoint_stale_reason,
                :source_keypoint_stale_timestamp_utc, :source_keypoint_stale_json,
                :rig_id, :camera_id, :arena_id, :dish_design, :canvas_name, :protocol_name,
                :genotype, :dpf_at_acquisition,
                :profile_json
            )
            ON CONFLICT(dataset_id, profile_run) DO UPDATE SET
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                stage_group=excluded.stage_group,
                eye_mask_method=excluded.eye_mask_method,
                source_eye_mask_path=excluded.source_eye_mask_path,
                source_eye_mask_run=excluded.source_eye_mask_run,
                source_keypoint_path=excluded.source_keypoint_path,
                source_keypoint_run=excluded.source_keypoint_run,
                source_crop_run=excluded.source_crop_run,
                profile_created_utc=excluded.profile_created_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc,
                rows_total=excluded.rows_total,
                rows_usable=excluded.rows_usable,
                usable_rate=excluded.usable_rate,
                reviewed_rate=excluded.reviewed_rate,
                excluded_rate=excluded.excluded_rate,
                exclusion_reasons_json=excluded.exclusion_reasons_json,
                ellipse_success_rate=excluded.ellipse_success_rate,
                pair_success_rate=excluded.pair_success_rate,
                area_p10=excluded.area_p10,
                area_p50=excluded.area_p50,
                area_p90=excluded.area_p90,
                left_area_p10=excluded.left_area_p10,
                left_area_p50=excluded.left_area_p50,
                left_area_p90=excluded.left_area_p90,
                right_area_p10=excluded.right_area_p10,
                right_area_p50=excluded.right_area_p50,
                right_area_p90=excluded.right_area_p90,
                union_area_p10=excluded.union_area_p10,
                union_area_p50=excluded.union_area_p50,
                union_area_p90=excluded.union_area_p90,
                area_lr_ratio_p10=excluded.area_lr_ratio_p10,
                area_lr_ratio_p50=excluded.area_lr_ratio_p50,
                area_lr_ratio_p90=excluded.area_lr_ratio_p90,
                major_axis_p10=excluded.major_axis_p10,
                major_axis_p50=excluded.major_axis_p50,
                major_axis_p90=excluded.major_axis_p90,
                minor_axis_p10=excluded.minor_axis_p10,
                minor_axis_p50=excluded.minor_axis_p50,
                minor_axis_p90=excluded.minor_axis_p90,
                aspect_ratio_p10=excluded.aspect_ratio_p10,
                aspect_ratio_p50=excluded.aspect_ratio_p50,
                aspect_ratio_p90=excluded.aspect_ratio_p90,
                eye_separation_p10=excluded.eye_separation_p10,
                eye_separation_p50=excluded.eye_separation_p50,
                eye_separation_p90=excluded.eye_separation_p90,
                edge_proximity_rate=excluded.edge_proximity_rate,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_timestamp_utc=excluded.review_timestamp_utc,
                source_keypoint_stale_state=excluded.source_keypoint_stale_state,
                source_keypoint_stale_reason=excluded.source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc=excluded.source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json=excluded.source_keypoint_stale_json,
                {duplicate_context_update_sql}
                profile_json=excluded.profile_json;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_eye_mask_quality(
        self,
        *,
        dataset_id: str,
        stage_group: str,
        run_name: str,
        run_created_utc: Optional[str],
        recording_id: Optional[str],
        zarr_use: Optional[str],
        eye_mask_method: Optional[str],
        source_crop_run: Optional[str],
        source_keypoint_group: Optional[str],
        source_keypoints_run: Optional[str],
        source_eye_masks_run: Optional[str],
        source_eye_masks_method: Optional[str],
        review_state: Optional[str],
        review_method: Optional[str],
        review_intended_use: Optional[str],
        review_reviewer: Optional[str],
        review_timestamp_utc: Optional[str],
        total_rois: Optional[int],
        successful_eyes: Optional[int],
        successful_roi_pairs: Optional[int],
        successful_roi_pair_rate: Optional[float],
        source_keypoint_stale_state: Optional[str] = None,
        source_keypoint_stale_reason: Optional[str] = None,
        source_keypoint_stale_timestamp_utc: Optional[str] = None,
        source_keypoint_stale_json: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        lifecycle_reason: Optional[str] = None,
        quality_updated_utc: Optional[str] = None,
        zarr_mtime_ns: Optional[int] = None,
    ) -> None:
        payload = {
            "dataset_id": str(dataset_id),
            "stage_group": str(stage_group),
            "run_name": str(run_name),
            "run_created_utc": run_created_utc,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "eye_mask_method": eye_mask_method,
            "source_crop_run": source_crop_run,
            "source_keypoint_group": source_keypoint_group,
            "source_keypoints_run": source_keypoints_run,
            "source_eye_masks_run": source_eye_masks_run,
            "source_eye_masks_method": source_eye_masks_method,
            "review_state": review_state,
            "review_method": review_method,
            "review_intended_use": review_intended_use,
            "review_reviewer": review_reviewer,
            "review_timestamp_utc": review_timestamp_utc,
            "total_rois": total_rois,
            "successful_eyes": successful_eyes,
            "successful_roi_pairs": successful_roi_pairs,
            "successful_roi_pair_rate": successful_roi_pair_rate,
            "source_keypoint_stale_state": source_keypoint_stale_state,
            "source_keypoint_stale_reason": source_keypoint_stale_reason,
            "source_keypoint_stale_timestamp_utc": source_keypoint_stale_timestamp_utc,
            "source_keypoint_stale_json": source_keypoint_stale_json,
            "lifecycle_state": lifecycle_state,
            "lifecycle_reason": lifecycle_reason,
            "quality_updated_utc": quality_updated_utc or _utc_now(),
            "zarr_mtime_ns": zarr_mtime_ns,
        }
        self.conn.execute(
            """
            INSERT INTO eye_mask_quality (
                dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                eye_mask_method, source_crop_run, source_keypoint_group, source_keypoints_run,
                source_eye_masks_run, source_eye_masks_method,
                review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                total_rois, successful_eyes, successful_roi_pairs, successful_roi_pair_rate,
                source_keypoint_stale_state, source_keypoint_stale_reason, source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json, lifecycle_state, lifecycle_reason,
                quality_updated_utc, zarr_mtime_ns
            )
            VALUES (
                :dataset_id, :stage_group, :run_name, :run_created_utc, :recording_id, :zarr_use,
                :eye_mask_method, :source_crop_run, :source_keypoint_group, :source_keypoints_run,
                :source_eye_masks_run, :source_eye_masks_method,
                :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                :total_rois, :successful_eyes, :successful_roi_pairs, :successful_roi_pair_rate,
                :source_keypoint_stale_state, :source_keypoint_stale_reason, :source_keypoint_stale_timestamp_utc,
                :source_keypoint_stale_json, :lifecycle_state, :lifecycle_reason,
                :quality_updated_utc, :zarr_mtime_ns
            )
            ON CONFLICT(dataset_id, stage_group, run_name) DO UPDATE SET
                run_created_utc=excluded.run_created_utc,
                recording_id=excluded.recording_id,
                zarr_use=excluded.zarr_use,
                eye_mask_method=excluded.eye_mask_method,
                source_crop_run=excluded.source_crop_run,
                source_keypoint_group=excluded.source_keypoint_group,
                source_keypoints_run=excluded.source_keypoints_run,
                source_eye_masks_run=excluded.source_eye_masks_run,
                source_eye_masks_method=excluded.source_eye_masks_method,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_timestamp_utc=excluded.review_timestamp_utc,
                total_rois=excluded.total_rois,
                successful_eyes=excluded.successful_eyes,
                successful_roi_pairs=excluded.successful_roi_pairs,
                successful_roi_pair_rate=excluded.successful_roi_pair_rate,
                source_keypoint_stale_state=excluded.source_keypoint_stale_state,
                source_keypoint_stale_reason=excluded.source_keypoint_stale_reason,
                source_keypoint_stale_timestamp_utc=excluded.source_keypoint_stale_timestamp_utc,
                source_keypoint_stale_json=excluded.source_keypoint_stale_json,
                lifecycle_state=excluded.lifecycle_state,
                lifecycle_reason=excluded.lifecycle_reason,
                quality_updated_utc=excluded.quality_updated_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_detect_quality(
        self,
        *,
        dataset_id: str,
        refined_run: str,
        refined_created_utc: Optional[str],
        source_detect_run: str,
        detect_method: Optional[str],
        review_state: Optional[str],
        review_intended_use: Optional[str],
        review_reviewer: Optional[str],
        review_timestamp_utc: Optional[str],
        review_resolved_group: Optional[str],
        total_detections: Optional[int],
        real_detections: Optional[int],
        interpolated_detections: Optional[int],
        interpolated_detections_rate: Optional[float],
        review_method: Optional[str] = None,
        review_notes: Optional[str] = None,
        quality_updated_utc: Optional[str] = None,
        zarr_mtime_ns: Optional[int] = None,
    ) -> None:
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        payload = {
            "dataset_id": str(dataset_id),
            "refined_run": str(refined_run),
            "refined_created_utc": refined_created_utc,
            "source_detect_run": str(source_detect_run),
            "detect_method": detect_method,
            "review_state": review_state,
            "review_method": review_method,
            "review_intended_use": review_intended_use,
            "review_reviewer": review_reviewer,
            "review_notes": review_notes,
            "review_timestamp_utc": review_timestamp_utc,
            "review_resolved_group": review_resolved_group,
            "total_detections": total_detections,
            "real_detections": real_detections,
            "interpolated_detections": interpolated_detections,
            "interpolated_detections_rate": interpolated_detections_rate,
            "quality_updated_utc": quality_updated_utc or _utc_now(),
            "zarr_mtime_ns": zarr_mtime_ns,
        }
        self.conn.execute(
            """
            INSERT INTO detect_quality (
                dataset_id, refined_run, refined_created_utc, source_detect_run, detect_method,
                review_state, review_method, review_intended_use, review_reviewer, review_notes,
                review_timestamp_utc, review_resolved_group,
                total_detections, real_detections, interpolated_detections, interpolated_detections_rate,
                quality_updated_utc, zarr_mtime_ns
            )
            VALUES (
                :dataset_id, :refined_run, :refined_created_utc, :source_detect_run, :detect_method,
                :review_state, :review_method, :review_intended_use, :review_reviewer, :review_notes,
                :review_timestamp_utc, :review_resolved_group,
                :total_detections, :real_detections, :interpolated_detections, :interpolated_detections_rate,
                :quality_updated_utc, :zarr_mtime_ns
            )
            ON CONFLICT(dataset_id, refined_run) DO UPDATE SET
                refined_created_utc=excluded.refined_created_utc,
                source_detect_run=excluded.source_detect_run,
                detect_method=excluded.detect_method,
                review_state=excluded.review_state,
                review_method=excluded.review_method,
                review_intended_use=excluded.review_intended_use,
                review_reviewer=excluded.review_reviewer,
                review_notes=excluded.review_notes,
                review_timestamp_utc=excluded.review_timestamp_utc,
                review_resolved_group=excluded.review_resolved_group,
                total_detections=excluded.total_detections,
                real_detections=excluded.real_detections,
                interpolated_detections=excluded.interpolated_detections,
                interpolated_detections_rate=excluded.interpolated_detections_rate,
                quality_updated_utc=excluded.quality_updated_utc,
                zarr_mtime_ns=excluded.zarr_mtime_ns;
            """,
            payload,
        )
        self.conn.commit()

    def replace_detect_performance(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM detect_performance WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                for key in (
                    "review_state",
                    "review_method",
                    "review_intended_use",
                    "review_reviewer",
                    "review_timestamp_utc",
                    "source_keypoint_stale_state",
                    "source_keypoint_stale_reason",
                    "source_keypoint_stale_timestamp_utc",
                    "source_keypoint_stale_json",
                    "lifecycle_state",
                    "lifecycle_reason",
                ):
                    payload.setdefault(key, None)
                self.conn.execute(
                    """
                    INSERT INTO detect_performance (
                        dataset_id, detect_run, detect_created_utc, recording_id, zarr_use,
                        detection_method, model_run_id, model_set_id, model_path, model_name,
                        coverage_percent, frames_with_detections, frames_zero_detections, total_frames,
                        mean_confidence, min_confidence, max_confidence,
                        inference_duration_seconds, inference_average_fps, inference_avg_batch_ms, inference_avg_read_ms,
                        conf_threshold, iou_threshold, batch_size, inference_width, inference_height,
                        zarr_mtime_ns, updated_utc
                    )
                    VALUES (
                        :dataset_id, :detect_run, :detect_created_utc, :recording_id, :zarr_use,
                        :detection_method, :model_run_id, :model_set_id, :model_path, :model_name,
                        :coverage_percent, :frames_with_detections, :frames_zero_detections, :total_frames,
                        :mean_confidence, :min_confidence, :max_confidence,
                        :inference_duration_seconds, :inference_average_fps, :inference_avg_batch_ms, :inference_avg_read_ms,
                        :conf_threshold, :iou_threshold, :batch_size, :inference_width, :inference_height,
                        :zarr_mtime_ns, :updated_utc
                    );
                    """,
                    payload,
                )

    def replace_detection_data_profile(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        write_legacy_recording_context_snapshot, write_legacy_biology_snapshot = (
            self._profile_duplicate_context_write_policy(str(dataset_id))
        )
        with self.conn:
            self.conn.execute("DELETE FROM detection_data_profile WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                for key in (
                    "rig_id",
                    "camera_id",
                    "arena_id",
                    "dish_design",
                    "canvas_name",
                    "protocol_name",
                    "genotype",
                    "dpf_at_acquisition",
                ):
                    payload.setdefault(key, None)
                payload = self._apply_profile_duplicate_context_write_policy(
                    payload,
                    write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
                    write_legacy_biology_snapshot=write_legacy_biology_snapshot,
                )
                self.conn.execute(
                    """
                    INSERT INTO detection_data_profile (
                        dataset_id, profile_run, recording_id, zarr_use,
                        detection_type, detection_path, profile_created_utc,
                        zarr_mtime_ns, updated_utc,
                        frames_total, frames_with_detections, coverage_percent, detections_total,
                        detections_per_frame_p50, detections_per_frame_p90,
                        w_p10, w_p50, w_p90,
                        h_p10, h_p50, h_p90,
                        area_p10, area_p50, area_p90,
                        aspect_ratio_p10, aspect_ratio_p50, aspect_ratio_p90,
                        edge_proximity_rate,
                        rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name,
                        genotype, dpf_at_acquisition,
                        profile_json
                    )
                    VALUES (
                        :dataset_id, :profile_run, :recording_id, :zarr_use,
                        :detection_type, :detection_path, :profile_created_utc,
                        :zarr_mtime_ns, :updated_utc,
                        :frames_total, :frames_with_detections, :coverage_percent, :detections_total,
                        :detections_per_frame_p50, :detections_per_frame_p90,
                        :w_p10, :w_p50, :w_p90,
                        :h_p10, :h_p50, :h_p90,
                        :area_p10, :area_p50, :area_p90,
                        :aspect_ratio_p10, :aspect_ratio_p50, :aspect_ratio_p90,
                        :edge_proximity_rate,
                        :rig_id, :camera_id, :arena_id, :dish_design, :canvas_name, :protocol_name,
                        :genotype, :dpf_at_acquisition,
                        :profile_json
                    );
                    """,
                    payload,
                )

    def replace_keypoint_data_profile(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        write_legacy_recording_context_snapshot, write_legacy_biology_snapshot = (
            self._profile_duplicate_context_write_policy(str(dataset_id))
        )
        with self.conn:
            self.conn.execute("DELETE FROM keypoint_data_profile WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                for key in (
                    "pose_schema_name",
                    "pose_schema_json",
                    "heading_computation_source",
                    "heading_computation_json",
                    "rig_id",
                    "camera_id",
                    "arena_id",
                    "dish_design",
                    "canvas_name",
                    "protocol_name",
                    "genotype",
                    "dpf_at_acquisition",
                ):
                    payload.setdefault(key, None)
                payload = self._apply_profile_duplicate_context_write_policy(
                    payload,
                    write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
                    write_legacy_biology_snapshot=write_legacy_biology_snapshot,
                )
                self.conn.execute(
                    """
                    INSERT INTO keypoint_data_profile (
                        dataset_id, profile_run, recording_id, zarr_use,
                        keypoint_method, source_keypoint_path, source_keypoint_run,
                        skeleton_id, kpt_shape, pose_schema_name, pose_schema_json,
                        heading_computation_source, heading_computation_json, profile_created_utc,
                        zarr_mtime_ns, updated_utc,
                        rows_total, rows_usable, usable_keypoints_total, usable_rate,
                        confidence_valid_rate, geometry_valid_rate,
                        triangle_area_p10, triangle_area_p50, triangle_area_p90,
                        min_angle_p10, min_angle_p50, min_angle_p90,
                        heading_p10, heading_p50, heading_p90,
                        rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name,
                        genotype, dpf_at_acquisition,
                        profile_json
                    )
                    VALUES (
                        :dataset_id, :profile_run, :recording_id, :zarr_use,
                        :keypoint_method, :source_keypoint_path, :source_keypoint_run,
                        :skeleton_id, :kpt_shape, :pose_schema_name, :pose_schema_json,
                        :heading_computation_source, :heading_computation_json, :profile_created_utc,
                        :zarr_mtime_ns, :updated_utc,
                        :rows_total, :rows_usable, :usable_keypoints_total, :usable_rate,
                        :confidence_valid_rate, :geometry_valid_rate,
                        :triangle_area_p10, :triangle_area_p50, :triangle_area_p90,
                        :min_angle_p10, :min_angle_p50, :min_angle_p90,
                        :heading_p10, :heading_p50, :heading_p90,
                        :rig_id, :camera_id, :arena_id, :dish_design, :canvas_name, :protocol_name,
                        :genotype, :dpf_at_acquisition,
                        :profile_json
                    );
                    """,
                    payload,
                )

    def replace_eye_mask_data_profile(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        write_legacy_recording_context_snapshot, write_legacy_biology_snapshot = (
            self._profile_duplicate_context_write_policy(str(dataset_id))
        )
        with self.conn:
            self.conn.execute("DELETE FROM eye_mask_data_profile WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                for key in (
                    "source_keypoint_stale_state",
                    "source_keypoint_stale_reason",
                    "source_keypoint_stale_timestamp_utc",
                    "source_keypoint_stale_json",
                    "left_area_p10",
                    "left_area_p50",
                    "left_area_p90",
                    "right_area_p10",
                    "right_area_p50",
                    "right_area_p90",
                    "union_area_p10",
                    "union_area_p50",
                    "union_area_p90",
                    "area_lr_ratio_p10",
                    "area_lr_ratio_p50",
                    "area_lr_ratio_p90",
                    "rig_id",
                    "camera_id",
                    "arena_id",
                    "dish_design",
                    "canvas_name",
                    "protocol_name",
                    "genotype",
                    "dpf_at_acquisition",
                ):
                    payload.setdefault(key, None)
                payload = self._apply_profile_duplicate_context_write_policy(
                    payload,
                    write_legacy_recording_context_snapshot=write_legacy_recording_context_snapshot,
                    write_legacy_biology_snapshot=write_legacy_biology_snapshot,
                )
                self.conn.execute(
                    """
                    INSERT INTO eye_mask_data_profile (
                        dataset_id, profile_run, recording_id, zarr_use,
                        stage_group, eye_mask_method,
                        source_eye_mask_path, source_eye_mask_run,
                        source_keypoint_path, source_keypoint_run, source_crop_run,
                        profile_created_utc, zarr_mtime_ns, updated_utc,
                        rows_total, rows_usable, usable_rate, reviewed_rate, excluded_rate,
                        exclusion_reasons_json, ellipse_success_rate, pair_success_rate,
                        area_p10, area_p50, area_p90,
                        left_area_p10, left_area_p50, left_area_p90,
                        right_area_p10, right_area_p50, right_area_p90,
                        union_area_p10, union_area_p50, union_area_p90,
                        area_lr_ratio_p10, area_lr_ratio_p50, area_lr_ratio_p90,
                        major_axis_p10, major_axis_p50, major_axis_p90,
                        minor_axis_p10, minor_axis_p50, minor_axis_p90,
                        aspect_ratio_p10, aspect_ratio_p50, aspect_ratio_p90,
                        eye_separation_p10, eye_separation_p50, eye_separation_p90,
                        edge_proximity_rate,
                        review_state, review_method, review_intended_use, review_timestamp_utc,
                        source_keypoint_stale_state, source_keypoint_stale_reason,
                        source_keypoint_stale_timestamp_utc, source_keypoint_stale_json,
                        rig_id, camera_id, arena_id, dish_design, canvas_name, protocol_name,
                        genotype, dpf_at_acquisition,
                        profile_json
                    )
                    VALUES (
                        :dataset_id, :profile_run, :recording_id, :zarr_use,
                        :stage_group, :eye_mask_method,
                        :source_eye_mask_path, :source_eye_mask_run,
                        :source_keypoint_path, :source_keypoint_run, :source_crop_run,
                        :profile_created_utc, :zarr_mtime_ns, :updated_utc,
                        :rows_total, :rows_usable, :usable_rate, :reviewed_rate, :excluded_rate,
                        :exclusion_reasons_json, :ellipse_success_rate, :pair_success_rate,
                        :area_p10, :area_p50, :area_p90,
                        :left_area_p10, :left_area_p50, :left_area_p90,
                        :right_area_p10, :right_area_p50, :right_area_p90,
                        :union_area_p10, :union_area_p50, :union_area_p90,
                        :area_lr_ratio_p10, :area_lr_ratio_p50, :area_lr_ratio_p90,
                        :major_axis_p10, :major_axis_p50, :major_axis_p90,
                        :minor_axis_p10, :minor_axis_p50, :minor_axis_p90,
                        :aspect_ratio_p10, :aspect_ratio_p50, :aspect_ratio_p90,
                        :eye_separation_p10, :eye_separation_p50, :eye_separation_p90,
                        :edge_proximity_rate,
                        :review_state, :review_method, :review_intended_use, :review_timestamp_utc,
                        :source_keypoint_stale_state, :source_keypoint_stale_reason,
                        :source_keypoint_stale_timestamp_utc, :source_keypoint_stale_json,
                        :rig_id, :camera_id, :arena_id, :dish_design, :canvas_name, :protocol_name,
                        :genotype, :dpf_at_acquisition,
                        :profile_json
                    );
                    """,
                    payload,
                )

    def replace_keypoint_performance(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM keypoint_performance WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                self.conn.execute(
                    """
                    INSERT INTO keypoint_performance (
                        dataset_id, keypoint_run, keypoint_created_utc, recording_id, zarr_use,
                        keypoint_method, model_run_id, model_set_id, model_path, model_name,
                        source_crop_run, source_detect_run, source_refined_run,
                        total_rois, successful_detections, failed_detections, success_rate_percent,
                        frames_with_keypoints, mean_confidence,
                        duration_seconds, inference_duration_seconds, keypoints_per_second, inference_average_fps,
                        batch_size, imgsz, conf_threshold, iou_threshold, summary_statistics_json,
                        zarr_mtime_ns, updated_utc
                    )
                    VALUES (
                        :dataset_id, :keypoint_run, :keypoint_created_utc, :recording_id, :zarr_use,
                        :keypoint_method, :model_run_id, :model_set_id, :model_path, :model_name,
                        :source_crop_run, :source_detect_run, :source_refined_run,
                        :total_rois, :successful_detections, :failed_detections, :success_rate_percent,
                        :frames_with_keypoints, :mean_confidence,
                        :duration_seconds, :inference_duration_seconds, :keypoints_per_second, :inference_average_fps,
                        :batch_size, :imgsz, :conf_threshold, :iou_threshold, :summary_statistics_json,
                        :zarr_mtime_ns, :updated_utc
                    );
                    """,
                    payload,
                )

    def replace_eye_mask_performance(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM eye_mask_performance WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                self.conn.execute(
                    """
                    INSERT INTO eye_mask_performance (
                        dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                        method, source_crop_run, source_keypoint_group, source_keypoints_run,
                        source_eye_masks_run, source_eye_masks_method,
                        total_rois, successful_eyes, successful_roi_pairs, successful_roi_pair_rate,
                        duration_seconds, rois_per_second, inference_duration_seconds, inference_average_fps,
                        reason_counts_json, summary_statistics_json,
                        review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                        source_keypoint_stale_state, source_keypoint_stale_reason, source_keypoint_stale_timestamp_utc,
                        source_keypoint_stale_json, lifecycle_state, lifecycle_reason,
                        zarr_mtime_ns, updated_utc
                    )
                    VALUES (
                        :dataset_id, :stage_group, :run_name, :run_created_utc, :recording_id, :zarr_use,
                        :method, :source_crop_run, :source_keypoint_group, :source_keypoints_run,
                        :source_eye_masks_run, :source_eye_masks_method,
                        :total_rois, :successful_eyes, :successful_roi_pairs, :successful_roi_pair_rate,
                        :duration_seconds, :rois_per_second, :inference_duration_seconds, :inference_average_fps,
                        :reason_counts_json, :summary_statistics_json,
                        :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                        :source_keypoint_stale_state, :source_keypoint_stale_reason, :source_keypoint_stale_timestamp_utc,
                        :source_keypoint_stale_json, :lifecycle_state, :lifecycle_reason,
                        :zarr_mtime_ns, :updated_utc
                    );
                    """,
                    payload,
                )

    def replace_eye_mask_quality(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM eye_mask_quality WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("quality_updated_utc", _utc_now())
                payload.setdefault("zarr_mtime_ns", None)
                self.conn.execute(
                    """
                    INSERT INTO eye_mask_quality (
                        dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                        eye_mask_method, source_crop_run, source_keypoint_group, source_keypoints_run,
                        source_eye_masks_run, source_eye_masks_method,
                        review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                        total_rois, successful_eyes, successful_roi_pairs, successful_roi_pair_rate,
                        source_keypoint_stale_state, source_keypoint_stale_reason, source_keypoint_stale_timestamp_utc,
                        source_keypoint_stale_json, lifecycle_state, lifecycle_reason,
                        quality_updated_utc, zarr_mtime_ns
                    )
                    VALUES (
                        :dataset_id, :stage_group, :run_name, :run_created_utc, :recording_id, :zarr_use,
                        :eye_mask_method, :source_crop_run, :source_keypoint_group, :source_keypoints_run,
                        :source_eye_masks_run, :source_eye_masks_method,
                        :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                        :total_rois, :successful_eyes, :successful_roi_pairs, :successful_roi_pair_rate,
                        :source_keypoint_stale_state, :source_keypoint_stale_reason, :source_keypoint_stale_timestamp_utc,
                        :source_keypoint_stale_json, :lifecycle_state, :lifecycle_reason,
                        :quality_updated_utc, :zarr_mtime_ns
                    );
                    """,
                    payload,
                )

    def replace_subject_mask_performance(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM subject_mask_performance WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                for key in (
                    "recording_id",
                    "zarr_use",
                    "subject_mask_method",
                    "label_schema_id",
                    "source_crop_run",
                    "source_keypoint_group",
                    "source_keypoints_run",
                    "source_subject_mask_run",
                    "source_subject_mask_method",
                    "run_semantics",
                    "probability_semantics",
                    "source_background_run",
                    "source_background_array",
                    "source_dish_mask_array",
                    "tuning_source",
                    "tuning_timestamp",
                    "total_rois",
                    "rows_with_any_mask",
                    "coverage_percent",
                    "duration_seconds",
                    "rois_per_second",
                    "available_component_count",
                    "available_components_json",
                    "unavailable_components_json",
                    "component_review_states_json",
                    "eye_component_mode",
                    "reason_counts_json",
                    "summary_statistics_json",
                    "review_state",
                    "review_method",
                    "review_intended_use",
                    "review_reviewer",
                    "review_timestamp_utc",
                    "source_subject_mask_stale_state",
                    "source_subject_mask_stale_reason",
                    "source_subject_mask_stale_timestamp_utc",
                    "source_subject_mask_stale_json",
                    "lifecycle_state",
                    "lifecycle_reason",
                    "zarr_mtime_ns",
                ):
                    payload.setdefault(key, None)
                self.conn.execute(
                    """
                    INSERT INTO subject_mask_performance (
                        dataset_id, stage_group, run_name, run_created_utc, recording_id, zarr_use,
                        subject_mask_method, label_schema_id,
                        source_crop_run, source_keypoint_group, source_keypoints_run,
                        source_subject_mask_run, source_subject_mask_method,
                        run_semantics, probability_semantics,
                        source_background_run, source_background_array, source_dish_mask_array,
                        tuning_source, tuning_timestamp,
                        total_rois, rows_with_any_mask, coverage_percent,
                        duration_seconds, rois_per_second,
                        available_component_count, available_components_json, unavailable_components_json,
                        component_review_states_json, eye_component_mode,
                        reason_counts_json, summary_statistics_json,
                        review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                        source_subject_mask_stale_state, source_subject_mask_stale_reason,
                        source_subject_mask_stale_timestamp_utc, source_subject_mask_stale_json,
                        lifecycle_state, lifecycle_reason,
                        zarr_mtime_ns, updated_utc
                    )
                    VALUES (
                        :dataset_id, :stage_group, :run_name, :run_created_utc, :recording_id, :zarr_use,
                        :subject_mask_method, :label_schema_id,
                        :source_crop_run, :source_keypoint_group, :source_keypoints_run,
                        :source_subject_mask_run, :source_subject_mask_method,
                        :run_semantics, :probability_semantics,
                        :source_background_run, :source_background_array, :source_dish_mask_array,
                        :tuning_source, :tuning_timestamp,
                        :total_rois, :rows_with_any_mask, :coverage_percent,
                        :duration_seconds, :rois_per_second,
                        :available_component_count, :available_components_json, :unavailable_components_json,
                        :component_review_states_json, :eye_component_mode,
                        :reason_counts_json, :summary_statistics_json,
                        :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                        :source_subject_mask_stale_state, :source_subject_mask_stale_reason,
                        :source_subject_mask_stale_timestamp_utc, :source_subject_mask_stale_json,
                        :lifecycle_state, :lifecycle_reason,
                        :zarr_mtime_ns, :updated_utc
                    );
                    """,
                    payload,
                )

    def replace_subject_mask_component_quality(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM subject_mask_component_quality WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("quality_updated_utc", _utc_now())
                payload.setdefault("zarr_mtime_ns", None)
                for key in (
                    "component_family",
                    "run_created_utc",
                    "recording_id",
                    "zarr_use",
                    "subject_mask_method",
                    "label_schema_id",
                    "eye_component_mode",
                    "source_subject_mask_run",
                    "available",
                    "review_state",
                    "review_method",
                    "review_intended_use",
                    "review_reviewer",
                    "review_timestamp_utc",
                    "total_rois",
                    "rows_with_component_mask",
                    "rows_with_component_mask_rate",
                    "lifecycle_state",
                    "lifecycle_reason",
                ):
                    payload.setdefault(key, None)
                self.conn.execute(
                    """
                    INSERT INTO subject_mask_component_quality (
                        dataset_id, stage_group, run_name, component_name, component_family,
                        run_created_utc, recording_id, zarr_use, subject_mask_method, label_schema_id,
                        eye_component_mode, source_subject_mask_run, available,
                        review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc,
                        total_rois, rows_with_component_mask, rows_with_component_mask_rate,
                        lifecycle_state, lifecycle_reason, quality_updated_utc, zarr_mtime_ns
                    )
                    VALUES (
                        :dataset_id, :stage_group, :run_name, :component_name, :component_family,
                        :run_created_utc, :recording_id, :zarr_use, :subject_mask_method, :label_schema_id,
                        :eye_component_mode, :source_subject_mask_run, :available,
                        :review_state, :review_method, :review_intended_use, :review_reviewer, :review_timestamp_utc,
                        :total_rois, :rows_with_component_mask, :rows_with_component_mask_rate,
                        :lifecycle_state, :lifecycle_reason, :quality_updated_utc, :zarr_mtime_ns
                    );
                    """,
                    payload,
                )

    def refresh_detect_performance_for_dataset(
        self,
        dataset_id: str,
        *,
        zarr_path: Path,
        recording_id: Optional[str],
        zarr_use: Optional[str],
    ) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_detect_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_detect_performance(dataset_id, rows)
        return len(rows)

    def refresh_keypoint_performance_for_dataset(
        self,
        dataset_id: str,
        *,
        zarr_path: Path,
        recording_id: Optional[str],
        zarr_use: Optional[str],
    ) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_keypoint_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_keypoint_performance(dataset_id, rows)
        return len(rows)

    def refresh_eye_mask_performance_for_dataset(
        self,
        dataset_id: str,
        *,
        zarr_path: Path,
        recording_id: Optional[str],
        zarr_use: Optional[str],
    ) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_eye_mask_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_eye_mask_performance(dataset_id, rows)
        return len(rows)

    def refresh_eye_mask_quality_for_dataset(
        self,
        dataset_id: str,
        *,
        zarr_path: Path,
        recording_id: Optional[str],
        zarr_use: Optional[str],
    ) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_eye_mask_quality_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_eye_mask_quality(dataset_id, rows)
        return len(rows)

    def refresh_subject_mask_performance_for_dataset(
        self,
        dataset_id: str,
        *,
        zarr_path: Path,
        recording_id: Optional[str],
        zarr_use: Optional[str],
    ) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_subject_mask_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_subject_mask_performance(dataset_id, rows)
        return len(rows)

    def refresh_subject_mask_component_quality_for_dataset(
        self,
        dataset_id: str,
        *,
        zarr_path: Path,
        recording_id: Optional[str],
        zarr_use: Optional[str],
    ) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_subject_mask_component_quality_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_subject_mask_component_quality(dataset_id, rows)
        return len(rows)

    def replace_detect_quality(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        self._ensure_columns(
            "detect_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
            },
        )
        with self.conn:
            self.conn.execute("DELETE FROM detect_quality WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("quality_updated_utc", _utc_now())
                payload.setdefault("review_method", None)
                payload.setdefault("review_notes", None)
                payload.setdefault("zarr_mtime_ns", None)
                self.conn.execute(
                    """
                    INSERT INTO detect_quality (
                        dataset_id, refined_run, refined_created_utc, source_detect_run, detect_method,
                        review_state, review_method, review_intended_use, review_reviewer, review_notes,
                        review_timestamp_utc, review_resolved_group,
                        total_detections, real_detections, interpolated_detections, interpolated_detections_rate,
                        quality_updated_utc, zarr_mtime_ns
                    )
                    VALUES (
                        :dataset_id, :refined_run, :refined_created_utc, :source_detect_run, :detect_method,
                        :review_state, :review_method, :review_intended_use, :review_reviewer, :review_notes,
                        :review_timestamp_utc, :review_resolved_group,
                        :total_detections, :real_detections, :interpolated_detections, :interpolated_detections_rate,
                        :quality_updated_utc, :zarr_mtime_ns
                    );
                    """,
                    payload,
                )

    def replace_crop_quality(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM crop_quality WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("updated_utc", _utc_now())
                self.conn.execute(
                    """
                    INSERT INTO crop_quality (
                        dataset_id, crop_run, recording_id, zarr_use, crop_created_utc,
                        source_detect_run, source_refined_run, detection_source_type, detection_source_path,
                        total_rois, frames_with_crops, total_frames, percent_frames_with_crops,
                        includes_interpolated, n_real_detections, n_interpolated_detections,
                        review_state, review_method, review_intended_use, review_reviewer,
                        review_timestamp_utc, review_notes, zarr_mtime_ns, updated_utc
                    )
                    VALUES (
                        :dataset_id, :crop_run, :recording_id, :zarr_use, :crop_created_utc,
                        :source_detect_run, :source_refined_run, :detection_source_type, :detection_source_path,
                        :total_rois, :frames_with_crops, :total_frames, :percent_frames_with_crops,
                        :includes_interpolated, :n_real_detections, :n_interpolated_detections,
                        :review_state, :review_method, :review_intended_use, :review_reviewer,
                        :review_timestamp_utc, :review_notes, :zarr_mtime_ns, :updated_utc
                    );
                    """,
                    payload,
                )

    def refresh_detect_quality_for_dataset(self, dataset_id: str, *, zarr_path: Path) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_detect_quality_rows(root, zarr_path=zarr_path)
        self.replace_detect_quality(dataset_id, rows)
        return len(rows)

    def replace_keypoint_quality(self, dataset_id: str, records: Iterable[Dict[str, Any]]) -> None:
        self._ensure_columns(
            "keypoint_quality",
            {
                "review_method": "TEXT",
                "review_notes": "TEXT",
                "review_policy_id": "TEXT",
                "review_policy_version": "INTEGER",
            },
        )
        with self.conn:
            self.conn.execute("DELETE FROM keypoint_quality WHERE dataset_id = ?;", (str(dataset_id),))
            for record in records:
                payload = dict(record)
                payload["dataset_id"] = str(dataset_id)
                payload.setdefault("quality_updated_utc", _utc_now())
                payload.setdefault("review_method", None)
                payload.setdefault("review_notes", None)
                payload.setdefault("review_policy_id", None)
                payload.setdefault("review_policy_version", None)
                payload.setdefault("zarr_mtime_ns", None)
                self.conn.execute(
                    """
                    INSERT INTO keypoint_quality (
                        dataset_id, refined_run, refined_created_utc, source_keypoint_run, keypoint_method,
                        review_state, review_method, review_intended_use, review_reviewer, review_notes,
                        review_policy_id, review_policy_version, review_timestamp_utc,
                        usable_keypoints, total_keypoints, usable_keypoints_rate,
                        raw_keypoints_success_rate, raw_keypoints_successful,
                        quality_updated_utc, zarr_mtime_ns
                    )
                    VALUES (
                        :dataset_id, :refined_run, :refined_created_utc, :source_keypoint_run, :keypoint_method,
                        :review_state, :review_method, :review_intended_use, :review_reviewer, :review_notes,
                        :review_policy_id, :review_policy_version, :review_timestamp_utc,
                        :usable_keypoints, :total_keypoints, :usable_keypoints_rate,
                        :raw_keypoints_success_rate, :raw_keypoints_successful,
                        :quality_updated_utc, :zarr_mtime_ns
                    );
                    """,
                    payload,
                )

    def refresh_keypoint_quality_for_dataset(self, dataset_id: str, *, zarr_path: Path) -> int:
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        rows = _extract_keypoint_quality_rows(root, zarr_path=zarr_path)
        self.replace_keypoint_quality(dataset_id, rows)
        return len(rows)

    def query_keypoint_quality_current(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        keypoint_method: Optional[str] = None,
        review_state: Optional[str] = None,
        review_method: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        review_policy_id: Optional[str] = None,
        review_policy_version: Optional[int] = None,
        min_usable_keypoints_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM keypoint_quality_current WHERE 1=1"]
        params: List[Any] = []

        if dataset_ids:
            normalized_ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(normalized_ids)
        if keypoint_method is not None:
            sql.append("AND keypoint_method = ?")
            params.append(str(keypoint_method))
        if review_state is not None:
            sql.append("AND review_state = ?")
            params.append(str(review_state))
        if review_method is not None:
            sql.append("AND review_method = ?")
            params.append(str(review_method))
        if review_intended_use is not None:
            sql.append("AND review_intended_use = ?")
            params.append(str(review_intended_use))
        if review_policy_id is not None:
            sql.append("AND review_policy_id = ?")
            params.append(str(review_policy_id))
        if review_policy_version is not None:
            sql.append("AND review_policy_version = ?")
            params.append(int(review_policy_version))
        if min_usable_keypoints_rate is not None:
            sql.append("AND usable_keypoints_rate IS NOT NULL AND usable_keypoints_rate >= ?")
            params.append(float(min_usable_keypoints_rate))
        sql.append("ORDER BY dataset_id, keypoint_method")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def _query_detect_review_current_view(
        self,
        *,
        view_name: str,
        dataset_ids: Optional[Sequence[str]] = None,
        detect_method: Optional[str] = None,
        review_state: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        max_interpolated_detections_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        if not self._sqlite_object_exists(view_name):
            # `refined_detect_review_current` is a newer semantic alias for
            # `detect_quality_current`. Some long-lived registries have the
            # migration marked applied but are missing the alias view, so keep
            # registry-gated training selection working without mutating the DB.
            if view_name == "refined_detect_review_current" and self._sqlite_object_exists("detect_quality_current"):
                view_name = "detect_quality_current"
        sql = [f"SELECT * FROM {view_name} WHERE 1=1"]
        params: List[Any] = []

        if dataset_ids:
            normalized_ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(normalized_ids)
        if detect_method is not None:
            sql.append("AND detect_method = ?")
            params.append(str(detect_method))
        if review_state is not None:
            sql.append("AND review_state = ?")
            params.append(str(review_state))
        if review_intended_use is not None:
            sql.append("AND review_intended_use = ?")
            params.append(str(review_intended_use))
        if max_interpolated_detections_rate is not None:
            sql.append(
                "AND interpolated_detections_rate IS NOT NULL "
                "AND interpolated_detections_rate <= ?"
            )
            params.append(float(max_interpolated_detections_rate))
        sql.append("ORDER BY dataset_id, detect_method")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_detect_quality_current(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        detect_method: Optional[str] = None,
        review_state: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        max_interpolated_detections_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        return self._query_detect_review_current_view(
            view_name="detect_quality_current",
            dataset_ids=dataset_ids,
            detect_method=detect_method,
            review_state=review_state,
            review_intended_use=review_intended_use,
            max_interpolated_detections_rate=max_interpolated_detections_rate,
        )

    def query_refined_detect_review_current(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        detect_method: Optional[str] = None,
        review_state: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        max_interpolated_detections_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        return self._query_detect_review_current_view(
            view_name="refined_detect_review_current",
            dataset_ids=dataset_ids,
            detect_method=detect_method,
            review_state=review_state,
            review_intended_use=review_intended_use,
            max_interpolated_detections_rate=max_interpolated_detections_rate,
        )

    def query_eye_mask_quality_current(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        stage_group: Optional[str] = None,
        eye_mask_method: Optional[str] = None,
        review_state: Optional[str] = None,
        review_intended_use: Optional[str] = None,
        min_successful_roi_pair_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM eye_mask_quality_current WHERE 1=1"]
        params: List[Any] = []

        if dataset_ids:
            normalized_ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(normalized_ids)
        if stage_group is not None:
            sql.append("AND stage_group = ?")
            params.append(str(stage_group))
        if eye_mask_method is not None:
            sql.append("AND eye_mask_method = ?")
            params.append(str(eye_mask_method))
        if review_state is not None:
            sql.append("AND review_state = ?")
            params.append(str(review_state))
        if review_intended_use is not None:
            sql.append("AND review_intended_use = ?")
            params.append(str(review_intended_use))
        if min_successful_roi_pair_rate is not None:
            sql.append("AND successful_roi_pair_rate IS NOT NULL AND successful_roi_pair_rate >= ?")
            params.append(float(min_successful_roi_pair_rate))
        sql.append("ORDER BY dataset_id, stage_group, eye_mask_method")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_detection_data_profile_latest(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        recording_ids: Optional[Sequence[str]] = None,
        zarr_use: Optional[str] = None,
        detection_type: Optional[str] = None,
        min_coverage_percent: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM detection_data_profile_latest WHERE 1=1"]
        params: List[Any] = []

        if dataset_ids:
            normalized_ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(normalized_ids)
        if recording_ids:
            normalized_ids = [str(recording_id) for recording_id in recording_ids if recording_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            params.extend(normalized_ids)
        if zarr_use is not None:
            sql.append("AND zarr_use = ?")
            params.append(str(zarr_use))
        if detection_type is not None:
            sql.append("AND detection_type = ?")
            params.append(str(detection_type))
        if min_coverage_percent is not None:
            sql.append("AND coverage_percent IS NOT NULL AND coverage_percent >= ?")
            params.append(float(min_coverage_percent))
        sql.append("ORDER BY dataset_id")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_training_image_profile_latest(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        recording_ids: Optional[Sequence[str]] = None,
        zarr_use: Optional[str] = None,
        min_frames_profiled: Optional[int] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM training_image_profile_latest WHERE 1=1"]
        params: List[Any] = []

        if dataset_ids:
            normalized_ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(normalized_ids)
        if recording_ids:
            normalized_ids = [str(recording_id) for recording_id in recording_ids if recording_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            params.extend(normalized_ids)
        if zarr_use is not None:
            sql.append("AND zarr_use = ?")
            params.append(str(zarr_use))
        if min_frames_profiled is not None:
            sql.append("AND frames_profiled IS NOT NULL AND frames_profiled >= ?")
            params.append(int(min_frames_profiled))
        sql.append("ORDER BY dataset_id")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_recording_detection_data_profile_latest(
        self,
        *,
        recording_ids: Optional[Sequence[str]] = None,
        zarr_use: Optional[str] = None,
        detection_type: Optional[str] = None,
        min_coverage_percent: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM recording_detection_data_profile_latest WHERE 1=1"]
        params: List[Any] = []

        if recording_ids:
            normalized_ids = [str(recording_id) for recording_id in recording_ids if recording_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            params.extend(normalized_ids)
        if zarr_use is not None:
            sql.append("AND zarr_use = ?")
            params.append(str(zarr_use))
        if detection_type is not None:
            sql.append("AND detection_type = ?")
            params.append(str(detection_type))
        if min_coverage_percent is not None:
            sql.append("AND coverage_percent IS NOT NULL AND coverage_percent >= ?")
            params.append(float(min_coverage_percent))
        sql.append("ORDER BY recording_id")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_keypoint_data_profile_latest(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        recording_ids: Optional[Sequence[str]] = None,
        zarr_use: Optional[str] = None,
        keypoint_method: Optional[str] = None,
        min_usable_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM keypoint_data_profile_latest WHERE 1=1"]
        params: List[Any] = []

        if dataset_ids:
            normalized_ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(normalized_ids)
        if recording_ids:
            normalized_ids = [str(recording_id) for recording_id in recording_ids if recording_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            params.extend(normalized_ids)
        if zarr_use is not None:
            sql.append("AND zarr_use = ?")
            params.append(str(zarr_use))
        if keypoint_method is not None:
            sql.append("AND keypoint_method = ?")
            params.append(str(keypoint_method))
        if min_usable_rate is not None:
            sql.append("AND usable_rate IS NOT NULL AND usable_rate >= ?")
            params.append(float(min_usable_rate))
        sql.append("ORDER BY dataset_id, keypoint_method")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_recording_keypoint_data_profile_latest(
        self,
        *,
        recording_ids: Optional[Sequence[str]] = None,
        zarr_use: Optional[str] = None,
        keypoint_method: Optional[str] = None,
        min_usable_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM recording_keypoint_data_profile_latest WHERE 1=1"]
        params: List[Any] = []

        if recording_ids:
            normalized_ids = [str(recording_id) for recording_id in recording_ids if recording_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            params.extend(normalized_ids)
        if zarr_use is not None:
            sql.append("AND zarr_use = ?")
            params.append(str(zarr_use))
        if keypoint_method is not None:
            sql.append("AND keypoint_method = ?")
            params.append(str(keypoint_method))
        if min_usable_rate is not None:
            sql.append("AND usable_rate IS NOT NULL AND usable_rate >= ?")
            params.append(float(min_usable_rate))
        sql.append("ORDER BY recording_id, keypoint_method")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_eye_mask_data_profile_latest(
        self,
        *,
        dataset_ids: Optional[Sequence[str]] = None,
        recording_ids: Optional[Sequence[str]] = None,
        zarr_use: Optional[str] = None,
        stage_group: Optional[str] = None,
        eye_mask_method: Optional[str] = None,
        min_usable_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM eye_mask_data_profile_latest WHERE 1=1"]
        params: List[Any] = []

        if dataset_ids:
            normalized_ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND dataset_id IN ({placeholders})")
            params.extend(normalized_ids)
        if recording_ids:
            normalized_ids = [str(recording_id) for recording_id in recording_ids if recording_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            params.extend(normalized_ids)
        if zarr_use is not None:
            sql.append("AND zarr_use = ?")
            params.append(str(zarr_use))
        if stage_group is not None:
            sql.append("AND stage_group = ?")
            params.append(str(stage_group))
        if eye_mask_method is not None:
            sql.append("AND eye_mask_method = ?")
            params.append(str(eye_mask_method))
        if min_usable_rate is not None:
            sql.append("AND usable_rate IS NOT NULL AND usable_rate >= ?")
            params.append(float(min_usable_rate))
        sql.append("ORDER BY dataset_id, stage_group, eye_mask_method")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def query_recording_eye_mask_data_profile_latest(
        self,
        *,
        recording_ids: Optional[Sequence[str]] = None,
        zarr_use: Optional[str] = None,
        stage_group: Optional[str] = None,
        eye_mask_method: Optional[str] = None,
        min_usable_rate: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        sql = ["SELECT * FROM recording_eye_mask_data_profile_latest WHERE 1=1"]
        params: List[Any] = []

        if recording_ids:
            normalized_ids = [str(recording_id) for recording_id in recording_ids if recording_id]
            if not normalized_ids:
                return []
            placeholders = ", ".join("?" for _ in normalized_ids)
            sql.append(f"AND recording_id IN ({placeholders})")
            params.extend(normalized_ids)
        if zarr_use is not None:
            sql.append("AND zarr_use = ?")
            params.append(str(zarr_use))
        if stage_group is not None:
            sql.append("AND stage_group = ?")
            params.append(str(stage_group))
        if eye_mask_method is not None:
            sql.append("AND eye_mask_method = ?")
            params.append(str(eye_mask_method))
        if min_usable_rate is not None:
            sql.append("AND usable_rate IS NOT NULL AND usable_rate >= ?")
            params.append(float(min_usable_rate))
        sql.append("ORDER BY recording_id, stage_group, eye_mask_method")
        return list(self.conn.execute(" ".join(sql), params).fetchall())

    def _resolve_effective_dataset_id(
        self,
        *,
        base_dataset_id: str,
        session_uuid: Optional[str],
        zarr_path: Path,
    ) -> str:
        """
        Resolve dataset identity for registration without breaking legacy IDs.

        Behavior:
        - If base dataset_id is unused, keep it.
        - If base dataset_id already points to the same path_hash, keep it.
        - If base dataset_id collides with a different path and a session UUID is present,
          deterministically suffix with the current path hash so multiple datasets can
          coexist per recording/session.
        """
        current_hash = _compute_path_hash(zarr_path)
        row = self.conn.execute(
            "SELECT path_hash FROM datasets WHERE dataset_id = ?;",
            (base_dataset_id,),
        ).fetchone()
        existing_hash = str(row["path_hash"]) if row and row["path_hash"] is not None else ""
        if row is not None and existing_hash == current_hash:
            return base_dataset_id

        is_source_recording = bool(session_uuid) and "/recordings/" in str(zarr_path).replace("\\", "/").lower()
        if not is_source_recording:
            # For non-recording artifacts, preserve caller-provided IDs.
            return base_dataset_id

        # For source recordings, prefer canonical dataset IDs derived from path hash.
        # This prevents reintroducing legacy dataset_id=session_uuid rows during rescans.
        assert session_uuid is not None  # for type checkers
        candidate = f"{session_uuid}:z{current_hash[:12]}"
        for extra in ("", current_hash[12:16], current_hash[16:20], current_hash[20:24]):
            resolved = candidate if not extra else f"{candidate}{extra}"
            existing = self.conn.execute(
                "SELECT path_hash FROM datasets WHERE dataset_id = ?;",
                (resolved,),
            ).fetchone()
            if existing is None:
                return resolved
            if str(existing["path_hash"] or "") == current_hash:
                return resolved

        # Extremely unlikely fallback: use full hash to guarantee uniqueness.
        return f"{session_uuid}:z{current_hash}"

    def register_from_root(self, root: zarr.Group, zarr_path: Path) -> str:
        metadata = extract_dataset_metadata(root, zarr_path)
        base_dataset_id = metadata.dataset_id
        session_uuid = metadata.session_uuid
        zarr_purpose = metadata.zarr_purpose
        dataset_id = self._resolve_effective_dataset_id(
            base_dataset_id=base_dataset_id,
            session_uuid=session_uuid,
            zarr_path=zarr_path,
        )
        self.upsert_dataset(
            dataset_id,
            session_uuid=session_uuid,
            zarr_path=zarr_path,
            recording_id=metadata.recording_id,
            zarr_purpose=zarr_purpose,
            zarr_use=metadata.zarr_use,
        )
        dataset_row = self.conn.execute(
            "SELECT recording_id, zarr_use FROM datasets WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchone()
        recording_id = _decode_attr(dataset_row["recording_id"]) if dataset_row is not None else None
        zarr_use = _decode_attr(dataset_row["zarr_use"]) if dataset_row is not None else None

        protocol_name, protocol_hash = _extract_protocol(root)
        snapshot, _ = _extract_snapshot(root)
        provenance = _extract_provenance(snapshot)
        context = _extract_session_context(root)
        acquisition = _extract_acquisition(root)
        recording_context = _extract_recording_context(
            root,
            zarr_path,
            metadata,
            context=context,
            acquisition=acquisition,
            protocol_name=protocol_name,
        )
        if recording_context:
            self.upsert_recording(**recording_context)
        self.upsert_provenance(
            dataset_id,
            provenance=provenance,
            context=context,
            protocol_name=protocol_name,
            protocol_hash=protocol_hash,
            acquisition=acquisition,
            zarr_purpose=zarr_purpose,
        )
        detection_records = _build_detection_source_records(root)
        self.replace_detection_sources(dataset_id, detection_records)
        detect_quality_rows = _extract_detect_quality_rows(root, zarr_path=zarr_path)
        self.replace_detect_quality(dataset_id, detect_quality_rows)
        detect_performance_rows = _extract_detect_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_detect_performance(dataset_id, detect_performance_rows)
        crop_quality_rows = _extract_crop_quality_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_crop_quality(dataset_id, crop_quality_rows)
        keypoint_quality_rows = _extract_keypoint_quality_rows(root, zarr_path=zarr_path)
        self.replace_keypoint_quality(dataset_id, keypoint_quality_rows)
        keypoint_performance_rows = _extract_keypoint_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_keypoint_performance(dataset_id, keypoint_performance_rows)
        eye_mask_performance_rows = _extract_eye_mask_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_eye_mask_performance(dataset_id, eye_mask_performance_rows)
        eye_mask_quality_rows = _extract_eye_mask_quality_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_eye_mask_quality(dataset_id, eye_mask_quality_rows)
        subject_mask_performance_rows = _extract_subject_mask_performance_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_subject_mask_performance(dataset_id, subject_mask_performance_rows)
        subject_mask_component_quality_rows = _extract_subject_mask_component_quality_rows(
            root,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
        )
        self.replace_subject_mask_component_quality(dataset_id, subject_mask_component_quality_rows)
        return dataset_id

    def record_training_run(
        self,
        *,
        run_id: str,
        set_id: Optional[str],
        task_type: Optional[str] = None,
        config_path: Optional[Path],
        manifest_path: Optional[Path],
        skeleton_id: Optional[str] = None,
        model_path: Optional[Path],
        metrics_path: Optional[Path],
        config_sha256: Optional[str] = None,
        manifest_sha256: Optional[str] = None,
        model_sha256: Optional[str] = None,
        metrics_sha256: Optional[str] = None,
        status: Optional[str] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        invocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        inferred_task_type = _infer_task_type(
            explicit=task_type,
            set_id=set_id,
            run_id=run_id,
            config_path=config_path,
            manifest_path=manifest_path,
            model_path=model_path,
            invocation=invocation,
        )
        payload = {
            "run_id": run_id,
            "set_id": set_id,
            "task_type": inferred_task_type,
            "config_path": str(config_path) if config_path else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "skeleton_id": str(skeleton_id) if skeleton_id else None,
            "model_path": str(model_path) if model_path else None,
            "metrics_path": str(metrics_path) if metrics_path else None,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "model_sha256": model_sha256,
            "metrics_sha256": metrics_sha256,
            "status": status,
            "final_metrics_json": _json_dumps(final_metrics),
            "invocation_json": _json_dumps(invocation),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO training_runs (
                run_id, set_id, task_type, config_path, manifest_path, skeleton_id, model_path, metrics_path,
                config_sha256, manifest_sha256, model_sha256, metrics_sha256,
                status, final_metrics_json,
                invocation_json, created_utc
            )
            VALUES (
                :run_id, :set_id, :task_type, :config_path, :manifest_path, :skeleton_id, :model_path, :metrics_path,
                :config_sha256, :manifest_sha256, :model_sha256, :metrics_sha256,
                :status, :final_metrics_json,
                :invocation_json, :created_utc
            )
            ON CONFLICT(run_id) DO UPDATE SET
                set_id=excluded.set_id,
                task_type=excluded.task_type,
                config_path=excluded.config_path,
                manifest_path=excluded.manifest_path,
                skeleton_id=excluded.skeleton_id,
                model_path=excluded.model_path,
                metrics_path=excluded.metrics_path,
                config_sha256=excluded.config_sha256,
                manifest_sha256=excluded.manifest_sha256,
                model_sha256=excluded.model_sha256,
                metrics_sha256=excluded.metrics_sha256,
                status=excluded.status,
                final_metrics_json=excluded.final_metrics_json,
                invocation_json=excluded.invocation_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        if set_id and (skeleton_id or inferred_task_type):
            self.conn.execute(
                """
                UPDATE training_sets
                SET skeleton_id = COALESCE(?, skeleton_id),
                    task_type = COALESCE(?, task_type)
                WHERE set_id = ?;
                """,
                (
                    (str(skeleton_id) if skeleton_id else None),
                    inferred_task_type,
                    str(set_id),
                ),
            )
        self.conn.commit()
        self.record_training_model(
            run_id=run_id,
            set_id=set_id,
            task_type=inferred_task_type,
            model_path=model_path,
            model_sha256=model_sha256,
            metrics_path=metrics_path,
            metrics_sha256=metrics_sha256,
            status=status,
            final_metrics=final_metrics,
            metadata=_training_model_discovery_metadata(
                task_type=inferred_task_type,
                final_metrics=final_metrics,
                metadata={"source": "training_runs"},
            ),
        )

    def upsert_training_set(
        self,
        *,
        set_id: str,
        name: Optional[str],
        task_type: Optional[str] = None,
        query_filter: Optional[Dict[str, Any]],
        dataset_ids: Iterable[str],
        skeleton_id: Optional[str] = None,
        invocation: Optional[Dict[str, Any]] = None,
    ) -> None:
        dataset_ids_norm = sorted({str(dataset_id) for dataset_id in dataset_ids if dataset_id})
        inferred_task_type = _infer_task_type(
            explicit=task_type,
            set_id=set_id,
            query_filter=query_filter,
            invocation=invocation,
        )
        payload = {
            "set_id": str(set_id),
            "name": name,
            "task_type": inferred_task_type,
            "query_filter": _json_dumps(query_filter),
            "dataset_ids_json": _json_dumps(dataset_ids_norm),
            "skeleton_id": str(skeleton_id) if skeleton_id else None,
            "invocation_json": _json_dumps(invocation),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO training_sets (
                set_id, name, task_type, query_filter, dataset_ids_json, skeleton_id, invocation_json, created_utc
            )
            VALUES (
                :set_id, :name, :task_type, :query_filter, :dataset_ids_json, :skeleton_id, :invocation_json, :created_utc
            )
            ON CONFLICT(set_id) DO UPDATE SET
                name=excluded.name,
                task_type=excluded.task_type,
                query_filter=excluded.query_filter,
                dataset_ids_json=excluded.dataset_ids_json,
                skeleton_id=excluded.skeleton_id,
                invocation_json=excluded.invocation_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def record_model_export(
        self,
        *,
        run_id: str,
        export_type: str,
        path: Optional[Path],
        manifest_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Migration state: new format-specific tables are the source of truth.
        self._record_export_model_row(
            run_id=run_id,
            export_type=export_type,
            path=path,
            manifest_path=manifest_path,
            metadata=metadata,
        )

    def _resolve_run_set_id(self, run_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT set_id FROM training_runs WHERE run_id = ?;",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        set_id = row["set_id"]
        return str(set_id) if set_id else None

    def _infer_tensorrt_precision(
        self,
        *,
        path: Optional[Path],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        if metadata:
            for key in ("precision",):
                value = metadata.get(key)
                if value:
                    return str(value).strip().lower()
            trt_meta = metadata.get("trt")
            if isinstance(trt_meta, dict):
                value = trt_meta.get("precision")
                if value:
                    return str(value).strip().lower()
        if path:
            stem = path.stem.lower()
            if stem.endswith("_fp16"):
                return "fp16"
            if stem.endswith("_int8"):
                return "int8"
        return "fp16"

    def _read_json_path(self, path: Optional[Path]) -> Dict[str, Any]:
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _extract_nms_thresholds(
        self,
        *,
        manifest_payload: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        conf_val: Optional[float] = None
        iou_val: Optional[float] = None
        topk_val: Optional[int] = None

        def _apply(mapping: Optional[Mapping[str, Any]]) -> None:
            nonlocal conf_val, iou_val, topk_val
            if not isinstance(mapping, Mapping):
                return
            nms = _coerce_mapping(mapping.get("nms"))
            if nms:
                if conf_val is None:
                    conf_val = _as_float(nms.get("conf"))
                if iou_val is None:
                    iou_val = _as_float(nms.get("iou"))
                if topk_val is None:
                    topk_val = self._int_or_none(nms.get("topk"))

            if conf_val is None:
                conf_val = _as_float(mapping.get("nms_conf"))
            if conf_val is None:
                conf_val = _as_float(mapping.get("conf_threshold"))
            if conf_val is None:
                conf_val = _as_float(mapping.get("conf"))

            if iou_val is None:
                iou_val = _as_float(mapping.get("nms_iou"))
            if iou_val is None:
                iou_val = _as_float(mapping.get("iou_threshold"))
            if iou_val is None:
                iou_val = _as_float(mapping.get("iou"))

            if topk_val is None:
                topk_val = self._int_or_none(mapping.get("nms_topk"))
            if topk_val is None:
                topk_val = self._int_or_none(mapping.get("topk"))

        export_payload = _coerce_mapping((manifest_payload or {}).get("export"))
        _apply(export_payload)
        _apply(metadata)
        return conf_val, iou_val, topk_val

    def _int_or_none(self, value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text)
        except Exception:
            return None

    def _shape_to_list(self, input_shape: Any) -> Optional[List[Any]]:
        if isinstance(input_shape, (list, tuple)):
            return list(input_shape)
        if isinstance(input_shape, str):
            text = input_shape.strip()
            if not text:
                return None
            try:
                payload = json.loads(text)
            except Exception:
                return None
            if isinstance(payload, list):
                return payload
        return None

    def _resolve_shape_fields(
        self,
        *,
        input_shape: Any,
        imgsz: Any,
    ) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int]]:
        shape_list = self._shape_to_list(input_shape)
        input_shape_text = _json_dumps(shape_list) if shape_list else None

        max_batch = None
        img_h = None
        img_w = None
        dynamic_shapes = None

        if shape_list:
            max_batch = self._int_or_none(shape_list[0]) if len(shape_list) >= 1 else None
            img_h = self._int_or_none(shape_list[2]) if len(shape_list) >= 3 else None
            img_w = self._int_or_none(shape_list[3]) if len(shape_list) >= 4 else None
            dynamic_shapes = int(
                any(self._int_or_none(dimension) is None for dimension in shape_list)
            )

        if (img_h is None or img_w is None) and isinstance(imgsz, (list, tuple)):
            if len(imgsz) >= 2:
                img_h = img_h if img_h is not None else self._int_or_none(imgsz[0])
                img_w = img_w if img_w is not None else self._int_or_none(imgsz[1])
            elif len(imgsz) == 1:
                val = self._int_or_none(imgsz[0])
                if img_h is None:
                    img_h = val
                if img_w is None:
                    img_w = val
        elif (img_h is None or img_w is None) and imgsz is not None:
            val = self._int_or_none(imgsz)
            if img_h is None:
                img_h = val
            if img_w is None:
                img_w = val

        return input_shape_text, img_h, img_w, max_batch, dynamic_shapes

    def _file_size_bytes(self, path: Optional[Path]) -> Optional[int]:
        if path is None:
            return None
        try:
            return int(path.stat().st_size)
        except Exception:
            return None

    def _record_export_model_row(
        self,
        *,
        run_id: str,
        export_type: str,
        path: Optional[Path],
        manifest_path: Optional[Path],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        export_type_norm = str(export_type).strip().lower()
        set_id = self._resolve_run_set_id(run_id)
        run_skeleton_id = self._resolve_run_skeleton_id(run_id)
        if export_type_norm == "onnx":
            plugin_ops = None
            plugin_versions = None
            requires_plugins = None
            onnx_opset = None
            input_shape_text = None
            img_h = None
            img_w = None
            max_batch = None
            dynamic_shapes = None
            file_size_bytes = self._file_size_bytes(path)
            exporter_torch_version = None
            exporter_cuda_version = None
            exporter_hostname = None
            skeleton_id = run_skeleton_id
            if metadata:
                raw_skeleton_id = metadata.get("skeleton_id")
                if raw_skeleton_id:
                    skeleton_id = str(raw_skeleton_id)
                pose_schema = metadata.get("pose_schema")
                if isinstance(pose_schema, dict):
                    schema_skeleton_id = self.upsert_pose_skeleton_spec(
                        kpt_shape=pose_schema.get("kpt_shape"),
                        keypoint_labels=pose_schema.get("keypoint_labels"),
                        edges=pose_schema.get("skeleton"),
                        name="pose_schema",
                    )
                    if schema_skeleton_id:
                        skeleton_id = schema_skeleton_id
                raw_ops = metadata.get("plugin_ops")
                if isinstance(raw_ops, (list, tuple)):
                    plugin_ops = [str(item) for item in raw_ops if item]

                raw_versions = metadata.get("plugin_versions")
                if isinstance(raw_versions, dict):
                    plugin_versions = {
                        str(key): str(value)
                        for key, value in raw_versions.items()
                        if key and value is not None
                    }

                raw_requires = metadata.get("requires_plugins")
                if raw_requires is not None:
                    if isinstance(raw_requires, str):
                        requires_plugins = raw_requires.strip().lower() in {
                            "1",
                            "true",
                            "yes",
                            "y",
                            "on",
                        }
                    else:
                        requires_plugins = bool(raw_requires)

                build_env = metadata.get("build_env")
                if isinstance(build_env, dict):
                    exporter_torch_version = (
                        str(build_env.get("torch_version")) if build_env.get("torch_version") else None
                    )
                    exporter_cuda_version = (
                        str(build_env.get("cuda_version")) if build_env.get("cuda_version") else None
                    )
                    exporter_hostname = (
                        str(build_env.get("system_hostname"))
                        if build_env.get("system_hostname")
                        else None
                    )

            manifest_payload = self._read_json_path(manifest_path)
            export_payload = manifest_payload.get("export") if isinstance(manifest_payload, dict) else {}
            if not isinstance(export_payload, dict):
                export_payload = {}
            onnx_opset = self._int_or_none(export_payload.get("opset"))
            (
                input_shape_text,
                img_h,
                img_w,
                max_batch,
                dynamic_shapes,
            ) = self._resolve_shape_fields(
                input_shape=export_payload.get("input_shape"),
                imgsz=export_payload.get("imgsz"),
            )
            nms_conf, nms_iou, nms_topk = self._extract_nms_thresholds(
                manifest_payload=manifest_payload if isinstance(manifest_payload, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
            if metadata:
                if onnx_opset is None:
                    onnx_opset = self._int_or_none(metadata.get("opset"))
                if onnx_opset is None:
                    meta_props = metadata.get("metadata_props")
                    if isinstance(meta_props, dict):
                        onnx_opset = self._int_or_none(meta_props.get("opset"))
                if input_shape_text is None:
                    (
                        input_shape_text,
                        img_h,
                        img_w,
                        max_batch,
                        dynamic_shapes,
                    ) = self._resolve_shape_fields(
                        input_shape=metadata.get("input_shape"),
                        imgsz=metadata.get("imgsz"),
                    )

            if requires_plugins is None:
                if plugin_ops:
                    requires_plugins = True
                elif plugin_versions:
                    requires_plugins = True

            self.record_onnx_model(
                run_id=run_id,
                set_id=set_id,
                skeleton_id=skeleton_id,
                detection_model_run_id=run_id,
                path=path,
                sha256=str(metadata.get("sha256")) if metadata and metadata.get("sha256") else None,
                manifest_path=manifest_path,
                manifest_sha256=(
                    str(metadata.get("manifest_sha256"))
                    if metadata and metadata.get("manifest_sha256")
                    else None
                ),
                opset=onnx_opset,
                nms_conf=nms_conf,
                nms_iou=nms_iou,
                nms_topk=nms_topk,
                input_shape=input_shape_text,
                img_h=img_h,
                img_w=img_w,
                max_batch=max_batch,
                dynamic_shapes=(bool(dynamic_shapes) if dynamic_shapes is not None else None),
                file_size_bytes=file_size_bytes,
                exporter_torch_version=exporter_torch_version,
                exporter_cuda_version=exporter_cuda_version,
                exporter_hostname=exporter_hostname,
                requires_plugins=requires_plugins,
                plugin_ops=plugin_ops,
                plugin_versions=plugin_versions,
                metadata=metadata,
            )
            return

        if export_type_norm in {"tensorrt", "trt"}:
            input_shape_text = None
            img_h = None
            img_w = None
            max_batch = None
            dynamic_shapes = None
            file_size_bytes = self._file_size_bytes(path)
            trt_version = None
            cuda_version = None
            compute_capability = None
            gpu_name = None
            gpu_uuid = None
            system_hostname = None
            plugin_ops = None
            plugin_versions = None
            requires_plugins = None
            skeleton_id = run_skeleton_id
            if metadata:
                raw_skeleton_id = metadata.get("skeleton_id")
                if raw_skeleton_id:
                    skeleton_id = str(raw_skeleton_id)
                pose_schema = metadata.get("pose_schema")
                if isinstance(pose_schema, dict):
                    schema_skeleton_id = self.upsert_pose_skeleton_spec(
                        kpt_shape=pose_schema.get("kpt_shape"),
                        keypoint_labels=pose_schema.get("keypoint_labels"),
                        edges=pose_schema.get("skeleton"),
                        name="pose_schema",
                    )
                    if schema_skeleton_id:
                        skeleton_id = schema_skeleton_id
                build_env = metadata.get("build_env")
                if isinstance(build_env, dict):
                    trt_version = (
                        str(build_env.get("tensorrt_version"))
                        if build_env.get("tensorrt_version")
                        else None
                    )
                    cuda_version = (
                        str(build_env.get("cuda_version")) if build_env.get("cuda_version") else None
                    )
                    system_hostname = (
                        str(build_env.get("system_hostname"))
                        if build_env.get("system_hostname")
                        else None
                    )
                    gpu_name = (
                        str(build_env.get("gpu_name")) if build_env.get("gpu_name") else None
                    )
                    torch_device = build_env.get("torch_device")
                    if isinstance(torch_device, dict):
                        compute_capability = (
                            str(torch_device.get("compute_capability"))
                            if torch_device.get("compute_capability")
                            else compute_capability
                        )
                        if not gpu_name and torch_device.get("selected_device_name"):
                            gpu_name = str(torch_device.get("selected_device_name"))
                trt_device = metadata.get("trt_device_info")
                if isinstance(trt_device, dict):
                    if trt_device.get("selected_device_name"):
                        gpu_name = str(trt_device.get("selected_device_name"))
                    if trt_device.get("selected_device_uuid"):
                        gpu_uuid = str(trt_device.get("selected_device_uuid"))
                    if trt_device.get("compute_capability"):
                        compute_capability = str(trt_device.get("compute_capability"))

                raw_ops = metadata.get("plugin_ops")
                if isinstance(raw_ops, (list, tuple)):
                    plugin_ops = [str(item) for item in raw_ops if item]

                raw_versions = metadata.get("plugin_versions")
                if isinstance(raw_versions, dict):
                    plugin_versions = {
                        str(key): str(value)
                        for key, value in raw_versions.items()
                        if key and value is not None
                    }

                raw_requires = metadata.get("requires_plugins")
                if raw_requires is not None:
                    if isinstance(raw_requires, str):
                        requires_plugins = raw_requires.strip().lower() in {
                            "1",
                            "true",
                            "yes",
                            "y",
                            "on",
                        }
                    else:
                        requires_plugins = bool(raw_requires)

            manifest_payload = self._read_json_path(manifest_path)
            export_payload = manifest_payload.get("export") if isinstance(manifest_payload, dict) else {}
            if not isinstance(export_payload, dict):
                export_payload = {}
            (
                input_shape_text,
                img_h,
                img_w,
                max_batch,
                dynamic_shapes,
            ) = self._resolve_shape_fields(
                input_shape=export_payload.get("input_shape"),
                imgsz=export_payload.get("imgsz"),
            )
            nms_conf, nms_iou, nms_topk = self._extract_nms_thresholds(
                manifest_payload=manifest_payload if isinstance(manifest_payload, dict) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
            if metadata and input_shape_text is None:
                (
                    input_shape_text,
                    img_h,
                    img_w,
                    max_batch,
                    dynamic_shapes,
                ) = self._resolve_shape_fields(
                    input_shape=metadata.get("input_shape"),
                    imgsz=metadata.get("imgsz"),
                )

            if (plugin_ops is None and plugin_versions is None and requires_plugins is None):
                onnx_ref = metadata.get("onnx_run_id") if isinstance(metadata, dict) else None
                onnx_run_id = str(onnx_ref or run_id)
                onnx_row = self.conn.execute(
                    """
                    SELECT requires_plugins, plugin_ops_json, plugin_versions_json
                    FROM onnx_models
                    WHERE run_id = ?;
                    """,
                    (onnx_run_id,),
                ).fetchone()
                if onnx_row:
                    raw_requires = onnx_row["requires_plugins"]
                    if raw_requires is not None:
                        requires_plugins = bool(raw_requires)
                    raw_ops_json = onnx_row["plugin_ops_json"]
                    if raw_ops_json:
                        try:
                            parsed_ops = json.loads(str(raw_ops_json))
                            if isinstance(parsed_ops, list):
                                plugin_ops = [str(item) for item in parsed_ops if item]
                        except Exception:
                            plugin_ops = None
                    raw_versions_json = onnx_row["plugin_versions_json"]
                    if raw_versions_json:
                        try:
                            parsed_versions = json.loads(str(raw_versions_json))
                            if isinstance(parsed_versions, dict):
                                plugin_versions = {
                                    str(key): str(value)
                                    for key, value in parsed_versions.items()
                                    if key and value is not None
                                }
                        except Exception:
                            plugin_versions = None

            if requires_plugins is None:
                if plugin_ops:
                    requires_plugins = True
                elif plugin_versions:
                    requires_plugins = True

            onnx_run_ref = (
                str(metadata.get("onnx_run_id"))
                if isinstance(metadata, dict) and metadata.get("onnx_run_id")
                else str(run_id)
            )

            self.record_tensorrt_model(
                run_id=run_id,
                set_id=set_id,
                skeleton_id=skeleton_id,
                detection_model_run_id=run_id,
                onnx_run_id=onnx_run_ref,
                precision=self._infer_tensorrt_precision(path=path, metadata=metadata),
                path=path,
                sha256=str(metadata.get("sha256")) if metadata and metadata.get("sha256") else None,
                manifest_path=manifest_path,
                manifest_sha256=(
                    str(metadata.get("manifest_sha256"))
                    if metadata and metadata.get("manifest_sha256")
                    else None
                ),
                nms_conf=nms_conf,
                nms_iou=nms_iou,
                nms_topk=nms_topk,
                input_shape=input_shape_text,
                img_h=img_h,
                img_w=img_w,
                max_batch=max_batch,
                dynamic_shapes=(bool(dynamic_shapes) if dynamic_shapes is not None else None),
                file_size_bytes=file_size_bytes,
                trt_version=trt_version,
                cuda_version=cuda_version,
                compute_capability=compute_capability,
                gpu_name=gpu_name,
                gpu_uuid=gpu_uuid,
                system_hostname=system_hostname,
                requires_plugins=requires_plugins,
                plugin_ops=plugin_ops,
                plugin_versions=plugin_versions,
                metadata=metadata,
            )

    def record_training_model(
        self,
        *,
        run_id: str,
        set_id: Optional[str],
        model_path: Optional[Path],
        model_sha256: Optional[str],
        metrics_path: Optional[Path],
        metrics_sha256: Optional[str],
        status: Optional[str],
        final_metrics: Optional[Dict[str, Any]],
        task_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_shape: Any = None,
        input_layout: Optional[str] = None,
        input_channels: Optional[int] = None,
        img_h: Optional[int] = None,
        img_w: Optional[int] = None,
        max_batch: Optional[int] = None,
        dynamic_shapes: Optional[bool | int] = None,
        input_dtype: Optional[str] = None,
        input_color_space: Optional[str] = None,
        input_shape_source: Optional[str] = None,
        input_shape_status: Optional[str] = None,
    ) -> None:
        resolved_task_type = (
            _normalize_task_type(task_type)
            or _normalize_task_type(metadata.get("task_type") if isinstance(metadata, Mapping) else None)
            or _infer_task_type(
                set_id=set_id,
                run_id=run_id,
                model_path=model_path,
            )
        )
        metadata_payload = _training_model_discovery_metadata(
            task_type=resolved_task_type,
            final_metrics=final_metrics,
            metadata=metadata,
        )
        fields = _training_model_discovery_index_fields(
            task_type=resolved_task_type,
            final_metrics=final_metrics,
            metadata=metadata_payload,
        )
        input_fields = self._shape_fields_from_training_payloads(
            task_type=resolved_task_type,
            final_metrics=final_metrics,
            metadata=metadata_payload,
            input_shape=input_shape,
            input_layout=input_layout,
            input_channels=input_channels,
            img_h=img_h,
            img_w=img_w,
            max_batch=max_batch,
            dynamic_shapes=dynamic_shapes,
            input_dtype=input_dtype,
            input_color_space=input_color_space,
            input_shape_source=input_shape_source,
            input_shape_status=input_shape_status,
        )
        payload = {
            "run_id": str(run_id),
            "set_id": str(set_id) if set_id else None,
            "model_path": str(model_path) if model_path else None,
            "model_sha256": model_sha256,
            "metrics_path": str(metrics_path) if metrics_path else None,
            "metrics_sha256": metrics_sha256,
            "status": status,
            "task_type": fields["task_type"],
            "label_schema_id": fields["label_schema_id"],
            "coverage_class": fields["coverage_class"],
            "component_coverage_key": fields["component_coverage_key"],
            "mask_labels_json": fields["mask_labels_json"],
            "component_groups_json": fields["component_groups_json"],
            "best_metric_name": fields["best_metric_name"],
            "best_metric_value": fields["best_metric_value"],
            "best_epoch": fields["best_epoch"],
            "input_shape": input_fields["input_shape"],
            "input_layout": input_fields["input_layout"],
            "input_channels": input_fields["input_channels"],
            "img_h": input_fields["img_h"],
            "img_w": input_fields["img_w"],
            "max_batch": input_fields["max_batch"],
            "dynamic_shapes": input_fields["dynamic_shapes"],
            "input_dtype": input_fields["input_dtype"],
            "input_color_space": input_fields["input_color_space"],
            "input_shape_source": input_fields["input_shape_source"],
            "input_shape_status": input_fields["input_shape_status"],
            "final_metrics_json": _json_dumps(final_metrics),
            "metadata_json": _json_dumps(metadata_payload),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO training_models (
                run_id, set_id, model_path, model_sha256, metrics_path, metrics_sha256,
                status, task_type, label_schema_id, coverage_class, component_coverage_key,
                mask_labels_json, component_groups_json, best_metric_name, best_metric_value,
                best_epoch, input_shape, input_layout, input_channels, img_h, img_w,
                max_batch, dynamic_shapes, input_dtype, input_color_space,
                input_shape_source, input_shape_status,
                final_metrics_json, metadata_json, created_utc
            )
            VALUES (
                :run_id, :set_id, :model_path, :model_sha256, :metrics_path, :metrics_sha256,
                :status, :task_type, :label_schema_id, :coverage_class, :component_coverage_key,
                :mask_labels_json, :component_groups_json, :best_metric_name, :best_metric_value,
                :best_epoch, :input_shape, :input_layout, :input_channels, :img_h, :img_w,
                :max_batch, :dynamic_shapes, :input_dtype, :input_color_space,
                :input_shape_source, :input_shape_status,
                :final_metrics_json, :metadata_json, :created_utc
            )
            ON CONFLICT(run_id) DO UPDATE SET
                set_id=excluded.set_id,
                model_path=excluded.model_path,
                model_sha256=excluded.model_sha256,
                metrics_path=excluded.metrics_path,
                metrics_sha256=excluded.metrics_sha256,
                status=excluded.status,
                task_type=excluded.task_type,
                label_schema_id=excluded.label_schema_id,
                coverage_class=excluded.coverage_class,
                component_coverage_key=excluded.component_coverage_key,
                mask_labels_json=excluded.mask_labels_json,
                component_groups_json=excluded.component_groups_json,
                best_metric_name=excluded.best_metric_name,
                best_metric_value=excluded.best_metric_value,
                best_epoch=excluded.best_epoch,
                input_shape=excluded.input_shape,
                input_layout=excluded.input_layout,
                input_channels=excluded.input_channels,
                img_h=excluded.img_h,
                img_w=excluded.img_w,
                max_batch=excluded.max_batch,
                dynamic_shapes=excluded.dynamic_shapes,
                input_dtype=excluded.input_dtype,
                input_color_space=excluded.input_color_space,
                input_shape_source=excluded.input_shape_source,
                input_shape_status=excluded.input_shape_status,
                final_metrics_json=excluded.final_metrics_json,
                metadata_json=excluded.metadata_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    # Backward-compatible alias while callers migrate to the new name.
    def record_detection_model(
        self,
        *,
        run_id: str,
        set_id: Optional[str],
        model_path: Optional[Path],
        model_sha256: Optional[str],
        metrics_path: Optional[Path],
        metrics_sha256: Optional[str],
        status: Optional[str],
        final_metrics: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.record_training_model(
            run_id=run_id,
            set_id=set_id,
            task_type="detect",
            model_path=model_path,
            model_sha256=model_sha256,
            metrics_path=metrics_path,
            metrics_sha256=metrics_sha256,
            status=status,
            final_metrics=final_metrics,
            metadata=metadata,
        )

    def record_onnx_model(
        self,
        *,
        run_id: str,
        set_id: Optional[str],
        skeleton_id: Optional[str] = None,
        detection_model_run_id: Optional[str],
        path: Optional[Path],
        sha256: Optional[str],
        manifest_path: Optional[Path],
        manifest_sha256: Optional[str],
        opset: Optional[int] = None,
        nms_conf: Optional[float] = None,
        nms_iou: Optional[float] = None,
        nms_topk: Optional[int] = None,
        input_shape: Optional[str] = None,
        img_h: Optional[int] = None,
        img_w: Optional[int] = None,
        max_batch: Optional[int] = None,
        dynamic_shapes: Optional[bool] = None,
        file_size_bytes: Optional[int] = None,
        exporter_torch_version: Optional[str] = None,
        exporter_cuda_version: Optional[str] = None,
        exporter_hostname: Optional[str] = None,
        requires_plugins: Optional[bool] = None,
        plugin_ops: Optional[List[str]] = None,
        plugin_versions: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        plugin_ops_norm = [str(item) for item in (plugin_ops or []) if item]
        plugin_versions_norm = {
            str(key): str(value)
            for key, value in (plugin_versions or {}).items()
            if key and value is not None
        }
        requires_plugins_norm = (
            int(bool(requires_plugins)) if requires_plugins is not None else None
        )
        payload = {
            "run_id": str(run_id),
            "set_id": str(set_id) if set_id else None,
            "skeleton_id": str(skeleton_id) if skeleton_id else None,
            "detection_model_run_id": str(detection_model_run_id) if detection_model_run_id else None,
            "path": str(path) if path else None,
            "sha256": sha256,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_sha256": manifest_sha256,
            "opset": self._int_or_none(opset),
            "nms_conf": _as_float(nms_conf),
            "nms_iou": _as_float(nms_iou),
            "nms_topk": self._int_or_none(nms_topk),
            "input_shape": str(input_shape) if input_shape else None,
            "img_h": self._int_or_none(img_h),
            "img_w": self._int_or_none(img_w),
            "max_batch": self._int_or_none(max_batch),
            "dynamic_shapes": (
                int(bool(dynamic_shapes)) if dynamic_shapes is not None else None
            ),
            "file_size_bytes": self._int_or_none(file_size_bytes),
            "exporter_torch_version": (
                str(exporter_torch_version) if exporter_torch_version else None
            ),
            "exporter_cuda_version": (
                str(exporter_cuda_version) if exporter_cuda_version else None
            ),
            "exporter_hostname": str(exporter_hostname) if exporter_hostname else None,
            "requires_plugins": requires_plugins_norm,
            "plugin_ops_json": _json_dumps(plugin_ops_norm) if plugin_ops_norm else None,
            "plugin_versions_json": (
                _json_dumps(plugin_versions_norm) if plugin_versions_norm else None
            ),
            "metadata_json": _json_dumps(metadata),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO onnx_models (
                run_id, set_id, skeleton_id, detection_model_run_id, path, sha256, manifest_path,
                manifest_sha256, opset, nms_conf, nms_iou, nms_topk,
                input_shape, img_h, img_w, max_batch, dynamic_shapes,
                file_size_bytes, exporter_torch_version, exporter_cuda_version, exporter_hostname,
                requires_plugins, plugin_ops_json, plugin_versions_json, metadata_json, created_utc
            )
            VALUES (
                :run_id, :set_id, :skeleton_id, :detection_model_run_id, :path, :sha256, :manifest_path,
                :manifest_sha256, :opset, :nms_conf, :nms_iou, :nms_topk,
                :input_shape, :img_h, :img_w, :max_batch, :dynamic_shapes,
                :file_size_bytes, :exporter_torch_version, :exporter_cuda_version, :exporter_hostname,
                :requires_plugins, :plugin_ops_json, :plugin_versions_json, :metadata_json, :created_utc
            )
            ON CONFLICT(run_id) DO UPDATE SET
                set_id=excluded.set_id,
                skeleton_id=excluded.skeleton_id,
                detection_model_run_id=excluded.detection_model_run_id,
                path=excluded.path,
                sha256=excluded.sha256,
                manifest_path=excluded.manifest_path,
                manifest_sha256=excluded.manifest_sha256,
                opset=excluded.opset,
                nms_conf=excluded.nms_conf,
                nms_iou=excluded.nms_iou,
                nms_topk=excluded.nms_topk,
                input_shape=excluded.input_shape,
                img_h=excluded.img_h,
                img_w=excluded.img_w,
                max_batch=excluded.max_batch,
                dynamic_shapes=excluded.dynamic_shapes,
                file_size_bytes=excluded.file_size_bytes,
                exporter_torch_version=excluded.exporter_torch_version,
                exporter_cuda_version=excluded.exporter_cuda_version,
                exporter_hostname=excluded.exporter_hostname,
                requires_plugins=excluded.requires_plugins,
                plugin_ops_json=excluded.plugin_ops_json,
                plugin_versions_json=excluded.plugin_versions_json,
                metadata_json=excluded.metadata_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def record_tensorrt_model(
        self,
        *,
        run_id: str,
        set_id: Optional[str],
        skeleton_id: Optional[str] = None,
        detection_model_run_id: Optional[str],
        onnx_run_id: Optional[str],
        precision: str,
        path: Optional[Path],
        sha256: Optional[str],
        manifest_path: Optional[Path],
        manifest_sha256: Optional[str],
        nms_conf: Optional[float] = None,
        nms_iou: Optional[float] = None,
        nms_topk: Optional[int] = None,
        input_shape: Optional[str] = None,
        img_h: Optional[int] = None,
        img_w: Optional[int] = None,
        max_batch: Optional[int] = None,
        dynamic_shapes: Optional[bool] = None,
        file_size_bytes: Optional[int] = None,
        trt_version: Optional[str] = None,
        cuda_version: Optional[str] = None,
        compute_capability: Optional[str] = None,
        gpu_name: Optional[str] = None,
        gpu_uuid: Optional[str] = None,
        system_hostname: Optional[str] = None,
        requires_plugins: Optional[bool] = None,
        plugin_ops: Optional[List[str]] = None,
        plugin_versions: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        plugin_ops_norm = [str(item) for item in (plugin_ops or []) if item]
        plugin_versions_norm = {
            str(key): str(value)
            for key, value in (plugin_versions or {}).items()
            if key and value is not None
        }
        requires_plugins_norm = (
            int(bool(requires_plugins)) if requires_plugins is not None else None
        )
        payload = {
            "run_id": str(run_id),
            "set_id": str(set_id) if set_id else None,
            "skeleton_id": str(skeleton_id) if skeleton_id else None,
            "detection_model_run_id": str(detection_model_run_id) if detection_model_run_id else None,
            "onnx_run_id": str(onnx_run_id) if onnx_run_id else None,
            "precision": str(precision).strip().lower() if precision else "fp16",
            "path": str(path) if path else None,
            "sha256": sha256,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_sha256": manifest_sha256,
            "nms_conf": _as_float(nms_conf),
            "nms_iou": _as_float(nms_iou),
            "nms_topk": self._int_or_none(nms_topk),
            "input_shape": str(input_shape) if input_shape else None,
            "img_h": self._int_or_none(img_h),
            "img_w": self._int_or_none(img_w),
            "max_batch": self._int_or_none(max_batch),
            "dynamic_shapes": (
                int(bool(dynamic_shapes)) if dynamic_shapes is not None else None
            ),
            "file_size_bytes": self._int_or_none(file_size_bytes),
            "trt_version": str(trt_version) if trt_version else None,
            "cuda_version": str(cuda_version) if cuda_version else None,
            "compute_capability": str(compute_capability) if compute_capability else None,
            "gpu_name": str(gpu_name) if gpu_name else None,
            "gpu_uuid": str(gpu_uuid) if gpu_uuid else None,
            "system_hostname": str(system_hostname) if system_hostname else None,
            "requires_plugins": requires_plugins_norm,
            "plugin_ops_json": _json_dumps(plugin_ops_norm) if plugin_ops_norm else None,
            "plugin_versions_json": (
                _json_dumps(plugin_versions_norm) if plugin_versions_norm else None
            ),
            "metadata_json": _json_dumps(metadata),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO tensorrt_models (
                run_id, set_id, skeleton_id, detection_model_run_id, onnx_run_id, precision, path, sha256,
                manifest_path, manifest_sha256, nms_conf, nms_iou, nms_topk,
                input_shape, img_h, img_w, max_batch,
                dynamic_shapes, file_size_bytes, trt_version, cuda_version, compute_capability,
                gpu_name, gpu_uuid, system_hostname, requires_plugins, plugin_ops_json,
                plugin_versions_json, metadata_json, created_utc
            )
            VALUES (
                :run_id, :set_id, :skeleton_id, :detection_model_run_id, :onnx_run_id, :precision, :path, :sha256,
                :manifest_path, :manifest_sha256, :nms_conf, :nms_iou, :nms_topk,
                :input_shape, :img_h, :img_w, :max_batch,
                :dynamic_shapes, :file_size_bytes, :trt_version, :cuda_version, :compute_capability,
                :gpu_name, :gpu_uuid, :system_hostname, :requires_plugins, :plugin_ops_json,
                :plugin_versions_json, :metadata_json, :created_utc
            )
            ON CONFLICT(run_id, precision) DO UPDATE SET
                set_id=excluded.set_id,
                skeleton_id=excluded.skeleton_id,
                detection_model_run_id=excluded.detection_model_run_id,
                onnx_run_id=excluded.onnx_run_id,
                path=excluded.path,
                sha256=excluded.sha256,
                manifest_path=excluded.manifest_path,
                manifest_sha256=excluded.manifest_sha256,
                nms_conf=excluded.nms_conf,
                nms_iou=excluded.nms_iou,
                nms_topk=excluded.nms_topk,
                input_shape=excluded.input_shape,
                img_h=excluded.img_h,
                img_w=excluded.img_w,
                max_batch=excluded.max_batch,
                dynamic_shapes=excluded.dynamic_shapes,
                file_size_bytes=excluded.file_size_bytes,
                trt_version=excluded.trt_version,
                cuda_version=excluded.cuda_version,
                compute_capability=excluded.compute_capability,
                gpu_name=excluded.gpu_name,
                gpu_uuid=excluded.gpu_uuid,
                system_hostname=excluded.system_hostname,
                requires_plugins=excluded.requires_plugins,
                plugin_ops_json=excluded.plugin_ops_json,
                plugin_versions_json=excluded.plugin_versions_json,
                metadata_json=excluded.metadata_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def scan_zarr(self, zarr_path: Path) -> Optional[str]:
        if not zarr_path.exists():
            return None
        root = _open_zarr_group_non_consolidated(zarr_path, mode="r")
        return self.register_from_root(root, zarr_path)

    def reconcile_missing_datasets(self, *, scope_paths: Optional[Iterable[Path]] = None) -> Dict[str, int]:
        """
        Mark datasets as missing when their registered zarr_path no longer exists.

        When scope_paths are provided, reconciliation is limited to dataset paths
        inside those roots (or exact path matches).
        """
        scope_roots = [_normalize_fs_path(path) for path in (scope_paths or [])]
        rows = self.conn.execute(
            "SELECT dataset_id, zarr_path FROM datasets WHERE status IS NULL OR status != 'missing';"
        ).fetchall()

        checked = 0
        marked_missing = 0
        with self.conn:
            for row in rows:
                dataset_path = _normalize_fs_path(row["zarr_path"])
                if scope_roots and not _path_matches_scope(dataset_path, scope_roots):
                    continue
                checked += 1
                if _is_zarr_root(dataset_path):
                    continue
                self.conn.execute(
                    "UPDATE datasets SET status = 'missing' WHERE dataset_id = ?;",
                    (row["dataset_id"],),
                )
                marked_missing += 1

        return {"checked": checked, "marked_missing": marked_missing}

    def query_datasets(
        self,
        *,
        dish_design: Optional[str] = None,
        dish_design_like: Optional[str] = None,
        fish_id: Optional[str] = None,
        subject_count_min: Optional[int] = None,
        subject_count_max: Optional[int] = None,
        zarr_origin: Optional[str] = None,
        zarr_use: Optional[str] = None,
        experiment_context_status: Optional[str] = None,
        experiment_context_source: Optional[str] = None,
        stimulus_runs_available: Optional[bool] = None,
        fps_min: Optional[float] = None,
        fps_max: Optional[float] = None,
        exposure_min: Optional[float] = None,
        exposure_max: Optional[float] = None,
        frame_rate_min: Optional[float] = None,
        frame_rate_max: Optional[float] = None,
        gain_min: Optional[float] = None,
        gain_max: Optional[float] = None,
        video_codec: Optional[str] = None,
        video_pix_fmt: Optional[str] = None,
        format_encoder: Optional[str] = None,
        format_title: Optional[str] = None,
        format_comment: Optional[str] = None,
        encoder_name: Optional[str] = None,
        encoder_codec: Optional[str] = None,
        encoder_preset: Optional[str] = None,
        encoder_tuning: Optional[str] = None,
        encoder_rc: Optional[str] = None,
        compression_name: Optional[str] = None,
        camera_model: Optional[str] = None,
        camera_serial: Optional[str] = None,
        camera_id: Optional[str] = None,
        rig_id: Optional[str] = None,
        arena_id: Optional[str] = None,
        model_input: Optional[str] = None,
        path_contains: Optional[str] = None,
        status: Optional[str] = None,
        exclude_status: Optional[str] = None,
        require_recording: bool = False,
        exclude_step_ok: Optional[str] = None,
        require_steps_ok: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[sqlite3.Row]:
        sql = [
            "SELECT dcc.dataset_id, dcc.session_uuid, dcc.zarr_path,",
            "dcc.recording_id, dcc.zarr_origin, dcc.zarr_use, dcc.dataset_status AS status,",
            "dcc.experiment_context_status, dcc.experiment_context_source,",
            "dcc.experiment_context_status_detail, dcc.stimulus_runs_available,",
            "dcc.dish_design, COALESCE(dcc.subject_id, dcc.legacy_fish_id) AS fish_id, dcc.subject_id AS subject_id,",
            "dcc.subject_count_effective AS subject_count,",
            "dcc.subject_count_snapshot, dcc.subject_count_recorded, dcc.subject_context_source,",
            "dcc.fps, dcc.exposure, dcc.exposure_unit, dcc.frame_rate, dcc.gain,",
            "dcc.cross_id, dcc.genotype, dcc.line_strain, dcc.species, dcc.sex, dcc.dpf_at_acquisition,",
            "dcc.protocol_name, dcc.protocol_hash,",
            "dcc.video_codec, dcc.video_pix_fmt, p.format_title, p.format_comment, p.format_encoder,",
            "p.encoder_name, p.encoder_codec, p.encoder_preset, p.encoder_tuning, p.encoder_rc,",
            "p.encoder_bpp, p.encoder_target_bps, p.encoder_res, p.encoder_res_width, p.encoder_res_height,",
            "p.encoder_fps, p.encoder_color, p.encoder_params_json,",
            "dcc.compression_name, dcc.compression_level,",
            "dcc.camera_model, dcc.camera_serial, dcc.camera_id, dcc.rig_id, dcc.arena_id, dcc.canvas_name,",
            "dcc.has_images_ds, dcc.has_images_ds_rgb, dcc.downsample_formats_json,",
            "dcc.subject_ids_json, dcc.dish_ids_json, dcc.cross_ids_json, dcc.genotypes_json,",
            "dcc.line_strains_json, dcc.species_values_json, dcc.sex_values_json, dcc.dpf_values_json",
            "FROM dataset_context_current dcc",
            "LEFT JOIN provenance p ON dcc.dataset_id = p.dataset_id",
        ]
        params: List[Any] = []

        if exclude_step_ok is not None:
            sql.append(
                "LEFT JOIN recording_step_status rss_excl "
                "ON dcc.dataset_id = rss_excl.dataset_id "
                "AND rss_excl.step_name = ?"
            )
            params.append(str(exclude_step_ok).strip().lower())

        if require_steps_ok is not None:
            for i, step in enumerate(require_steps_ok):
                alias = f"rss_req_{i}"
                sql.append(
                    f"INNER JOIN recording_step_status {alias} "
                    f"ON dcc.dataset_id = {alias}.dataset_id "
                    f"AND {alias}.step_name = ? AND {alias}.status = 'ok'"
                )
                params.append(str(step).strip().lower())

        sql.append("WHERE 1=1")

        def add_clause(clause: str, value: Any) -> None:
            if value is None:
                return
            sql.append(clause)
            params.append(value)

        add_clause("AND dcc.dish_design = ?", dish_design)
        if dish_design_like:
            sql.append("AND dcc.dish_design LIKE ?")
            params.append(f"%{dish_design_like}%")
        if fish_id is not None:
            fish_id_text = str(fish_id).strip()
            sql.append(
                "AND (COALESCE(dcc.subject_id, dcc.legacy_fish_id) = ? "
                "OR EXISTS ("
                "SELECT 1 FROM json_each(COALESCE(dcc.subject_ids_json, '[]')) "
                "WHERE CAST(json_each.value AS TEXT) = ?"
                "))"
            )
            params.extend([fish_id_text, fish_id_text])
        add_clause("AND dcc.subject_count_effective >= ?", subject_count_min)
        add_clause("AND dcc.subject_count_effective <= ?", subject_count_max)
        add_clause("AND dcc.zarr_origin = ?", _normalize_zarr_origin(zarr_origin))
        add_clause("AND dcc.zarr_use = ?", _normalize_zarr_use(zarr_use))
        add_clause("AND dcc.experiment_context_status = ?", experiment_context_status)
        add_clause("AND dcc.experiment_context_source = ?", experiment_context_source)
        if stimulus_runs_available is not None:
            add_clause("AND dcc.stimulus_runs_available = ?", int(bool(stimulus_runs_available)))
        add_clause("AND dcc.fps >= ?", fps_min)
        add_clause("AND dcc.fps <= ?", fps_max)
        add_clause("AND dcc.exposure >= ?", exposure_min)
        add_clause("AND dcc.exposure <= ?", exposure_max)
        add_clause("AND dcc.frame_rate >= ?", frame_rate_min)
        add_clause("AND dcc.frame_rate <= ?", frame_rate_max)
        add_clause("AND dcc.gain >= ?", gain_min)
        add_clause("AND dcc.gain <= ?", gain_max)
        add_clause("AND dcc.video_codec = ?", video_codec)
        add_clause("AND dcc.video_pix_fmt = ?", video_pix_fmt)
        add_clause("AND p.format_encoder = ?", format_encoder)
        add_clause("AND p.format_title = ?", format_title)
        add_clause("AND p.format_comment = ?", format_comment)
        add_clause("AND p.encoder_name = ?", encoder_name)
        add_clause("AND p.encoder_codec = ?", encoder_codec)
        add_clause("AND p.encoder_preset = ?", encoder_preset)
        add_clause("AND p.encoder_tuning = ?", encoder_tuning)
        add_clause("AND p.encoder_rc = ?", encoder_rc)
        add_clause("AND dcc.compression_name = ?", compression_name)
        add_clause("AND dcc.camera_model = ?", camera_model)
        add_clause("AND dcc.camera_serial = ?", camera_serial)
        add_clause("AND dcc.camera_id = ?", camera_id)
        add_clause("AND dcc.rig_id = ?", rig_id)
        add_clause("AND dcc.arena_id = ?", arena_id)
        if model_input is not None:
            mode = str(model_input).strip().lower()
            if mode == "gray":
                sql.append(
                    "AND (COALESCE(dcc.has_images_ds, 0) = 1 "
                    "OR dcc.downsample_formats_json LIKE '%\"gray\"%')"
                )
            elif mode == "rgb":
                sql.append(
                    "AND (COALESCE(dcc.has_images_ds_rgb, 0) = 1 "
                    "OR dcc.downsample_formats_json LIKE '%\"rgb\"%')"
                )
            else:
                raise ValueError(f"Unsupported model_input '{model_input}'. Expected 'gray' or 'rgb'.")
        if path_contains:
            sql.append("AND dcc.zarr_path LIKE ?")
            params.append(f"%{path_contains}%")
        add_clause("AND dcc.dataset_status = ?", status)
        if exclude_status is not None:
            sql.append("AND (dcc.dataset_status IS NULL OR dcc.dataset_status != ?)")
            params.append(exclude_status)
        if require_recording:
            sql.append("AND dcc.recording_id IS NOT NULL AND TRIM(dcc.recording_id) != ''")
        if exclude_step_ok is not None:
            sql.append("AND (rss_excl.status IS NULL OR rss_excl.status != 'ok')")

        sql.append("ORDER BY dcc.dish_design, dcc.fps, dcc.dataset_id")
        if limit is not None:
            sql.append("LIMIT ?")
            params.append(int(limit))

        query = " ".join(sql)
        return list(self.conn.execute(query, params).fetchall())


def scan_paths(
    registry: Registry,
    paths: Iterable[Path],
    *,
    recursive: bool = False,
) -> List[str]:
    normalized_paths = [Path(path).expanduser() for path in paths]
    dataset_ids: List[str] = []
    for path in normalized_paths:
        if path.is_dir() and _is_zarr_root(path):
            dataset_id = registry.scan_zarr(path)
            if dataset_id:
                dataset_ids.append(dataset_id)
            continue
        if path.is_dir() and recursive:
            for candidate in _find_zarr_roots(path):
                dataset_id = registry.scan_zarr(candidate)
                if dataset_id:
                    dataset_ids.append(dataset_id)
    registry.reconcile_missing_datasets(scope_paths=normalized_paths)
    return dataset_ids


def _is_zarr_root(path: Path) -> bool:
    return (path / "zarr.json").exists() or (path / ".zgroup").exists()


def _normalize_fs_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except Exception:
        return candidate.absolute()


def _path_matches_scope(candidate: Path, scope_roots: List[Path]) -> bool:
    for root in scope_roots:
        if candidate == root:
            return True
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _find_zarr_roots(root: Path) -> List[Path]:
    roots: List[Path] = []
    seen: set[Path] = set()
    for candidate in root.rglob("*.zarr"):
        if not candidate.is_dir():
            continue
        if not _is_zarr_root(candidate):
            continue
        resolved = _normalize_fs_path(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(candidate)
    roots.sort(key=lambda path: str(path))
    return roots
