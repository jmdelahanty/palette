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
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name
from .extractors.crop import _extract_crop_quality_rows
from .extractors.detect_performance import _extract_detect_performance_rows
from .extractors.keypoint_performance import (
    _extract_keypoint_performance_rows,
    _extract_keypoint_profile_rows,
)
from .extractors.masks import (
    _extract_eye_mask_performance_rows,
    _extract_eye_mask_quality_rows,
    _extract_subject_mask_component_quality_rows,
    _extract_subject_mask_performance_rows,
)
from .extractors.quality import _extract_detect_quality_rows, _extract_keypoint_quality_rows
from .migration_bodies import RegistryMigrationMixin
from .migrations import bind_migrations
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
    source_layout: Optional[str]
    source_frame_index_path: Optional[str]
    source_recording_frame_index_path: Optional[str]
    source_frame_index_schema: Optional[str]


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
        source_layout=_decode_attr(root.attrs.get("source_layout")),
        source_frame_index_path=_decode_attr(root.attrs.get("source_frame_index_path")),
        source_recording_frame_index_path=_decode_attr(root.attrs.get("source_recording_frame_index_path")),
        source_frame_index_schema=_decode_attr(root.attrs.get("source_frame_index_schema")),
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
    return resolve_latest_complete_run_name(parent, legacy_default=True)


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


class Registry(RegistryMigrationMixin):
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
        return bind_migrations(self)

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
        source_layout: Optional[str] = None,
        source_frame_index_path: Optional[str] = None,
        source_recording_frame_index_path: Optional[str] = None,
        source_frame_index_schema: Optional[str] = None,
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
            "source_layout": _decode_attr(source_layout),
            "source_frame_index_path": _decode_attr(source_frame_index_path),
            "source_recording_frame_index_path": _decode_attr(source_recording_frame_index_path),
            "source_frame_index_schema": _decode_attr(source_frame_index_schema),
            "path_hash": _compute_path_hash(zarr_path),
            "created_utc": now,
            "last_seen_utc": now,
            "status": "active",
        }
        self.conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, session_uuid, zarr_path, recording_id, artifact_kind, zarr_origin, zarr_use,
                source_layout, source_frame_index_path, source_recording_frame_index_path, source_frame_index_schema,
                path_hash, created_utc, last_seen_utc, status
            )
            VALUES (
                :dataset_id, :session_uuid, :zarr_path, :recording_id, :artifact_kind, :zarr_origin, :zarr_use,
                :source_layout, :source_frame_index_path, :source_recording_frame_index_path, :source_frame_index_schema,
                :path_hash, :created_utc, :last_seen_utc, :status
            )
            ON CONFLICT(dataset_id) DO UPDATE SET
                session_uuid=excluded.session_uuid,
                zarr_path=excluded.zarr_path,
                recording_id=COALESCE(excluded.recording_id, datasets.recording_id),
                artifact_kind=COALESCE(excluded.artifact_kind, datasets.artifact_kind),
                zarr_origin=COALESCE(excluded.zarr_origin, datasets.zarr_origin),
                zarr_use=COALESCE(excluded.zarr_use, datasets.zarr_use),
                source_layout=COALESCE(excluded.source_layout, datasets.source_layout),
                source_frame_index_path=COALESCE(excluded.source_frame_index_path, datasets.source_frame_index_path),
                source_recording_frame_index_path=COALESCE(excluded.source_recording_frame_index_path, datasets.source_recording_frame_index_path),
                source_frame_index_schema=COALESCE(excluded.source_frame_index_schema, datasets.source_frame_index_schema),
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
            source_layout=metadata.source_layout,
            source_frame_index_path=metadata.source_frame_index_path,
            source_recording_frame_index_path=metadata.source_recording_frame_index_path,
            source_frame_index_schema=metadata.source_frame_index_schema,
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
        if _is_empty_zarr_stub(zarr_path):
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
                if _is_zarr_root(dataset_path) and not _is_empty_zarr_stub(dataset_path):
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
        source_layout: Optional[str] = None,
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
            "dcc.source_layout, dcc.source_frame_index_path, dcc.source_recording_frame_index_path,",
            "dcc.source_frame_index_schema,",
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
        add_clause("AND dcc.source_layout = ?", source_layout)
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


def _is_empty_zarr_stub(path: Path) -> bool:
    """Return true for aborted Zarr roots with only root metadata and no attrs."""

    if not path.is_dir() or not _is_zarr_root(path):
        return False

    metadata_names = {"zarr.json", ".zgroup", ".zattrs", ".zmetadata"}
    try:
        non_metadata_children = [child for child in path.iterdir() if child.name not in metadata_names]
    except OSError:
        return False
    if non_metadata_children:
        return False

    if (path / ".zattrs").exists():
        try:
            attrs = json.loads((path / ".zattrs").read_text(encoding="utf-8"))
        except Exception:
            return False
        if attrs:
            return False

    zarr_json = path / "zarr.json"
    if zarr_json.exists():
        try:
            data = json.loads(zarr_json.read_text(encoding="utf-8"))
        except Exception:
            return False
        attrs = data.get("attributes")
        if isinstance(attrs, Mapping) and attrs:
            return False
        if attrs is not None and not isinstance(attrs, Mapping):
            return False

    return True


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
        if _is_empty_zarr_stub(candidate):
            continue
        resolved = _normalize_fs_path(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(candidate)
    roots.sort(key=lambda path: str(path))
    return roots
