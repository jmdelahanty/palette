from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from fisheye.utils.detection_profile import infer_zarr_use


SCHEMA_NAME = "keypoint_dataset_profile"
SCHEMA_VERSION = "v1"

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

REVIEW_TIMESTAMP_KEYS = ("timestamp_utc", "timestamp", "reviewed_at_utc", "reviewed_at", "updated_utc")
REFINED_PARENT_NAMES = ("refined_keypoints_runs", "keypoints_refined_runs")
GEOMETRY_ARRAY_NAMES = ("triangle_area", "min_angle", "heading")


class KeypointProfileError(RuntimeError):
    """Base error for keypoint profile read/write failures."""


class KeypointSourceError(KeypointProfileError):
    """Raised when no usable keypoint source can be resolved."""


@dataclass(frozen=True)
class ResolvedKeypointSource:
    keypoint_path: str
    keypoint_type: str
    keypoint_method: Optional[str]
    keypoint_run: Optional[str]
    refined_run: Optional[str]
    review_state: Optional[str]
    review_method: Optional[str]
    review_intended_use: Optional[str]
    review_timestamp_utc: Optional[str]
    skeleton_id: Optional[str]
    kpt_shape: Optional[Sequence[int]]
    pose_schema_name: Optional[str]
    pose_schema: Optional[dict[str, Any]]
    heading_computation_source: Optional[str]
    heading_computation: Optional[dict[str, Any]]


@dataclass(frozen=True)
class KeypointProfileWriteResult:
    run_name: str
    source_keypoint_path: str
    source_keypoint_method: Optional[str]
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
    return f"keypoint_profile_{stamp.strftime('%Y-%m-%d_%H-%M-%S')}"


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


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
        return float(value)
    except Exception:
        return None


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


def _coerce_int_sequence(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            parsed = _as_int(item)
            if parsed is None:
                return None
            out.append(int(parsed))
        return out
    return None


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            text_key = _normalize_text(key)
            if text_key is None:
                continue
            normalized[text_key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    text = _normalize_text(value)
    return text


def _coerce_json_mapping(value: Any) -> Optional[dict[str, Any]]:
    payload = _coerce_mapping(value)
    if not payload:
        return None
    normalized = _normalize_json_value(payload)
    return dict(normalized) if isinstance(normalized, dict) else None


def _resolve_source_metadata(
    group: zarr.Group,
) -> tuple[
    Optional[str],
    Optional[list[int]],
    Optional[str],
    Optional[dict[str, Any]],
    Optional[str],
    Optional[dict[str, Any]],
]:
    pose_schema = _coerce_json_mapping(group.attrs.get("pose_schema"))
    pose_schema_name = _normalize_text(pose_schema.get("name")) if pose_schema else None

    skeleton_id = _normalize_text(group.attrs.get("skeleton_id"))
    if skeleton_id is None and pose_schema is not None:
        skeleton_id = _normalize_text(pose_schema.get("skeleton_id"))
    if skeleton_id is None and pose_schema_name is not None:
        skeleton_id = f"pose_schema:{pose_schema_name}"

    kpt_shape = _coerce_int_sequence(group.attrs.get("kpt_shape"))
    if kpt_shape is None and pose_schema is not None:
        kpt_shape = _coerce_int_sequence(pose_schema.get("kpt_shape"))

    heading_computation_source = None
    heading_computation = _coerce_json_mapping(group.attrs.get("heading_computation_override"))
    if heading_computation is not None:
        heading_computation_source = "heading_computation_override"
    else:
        metadata = _coerce_json_mapping(pose_schema.get("metadata")) if pose_schema else None
        if metadata is not None:
            heading_computation = _coerce_json_mapping(metadata.get("heading_computation"))
            if heading_computation is not None:
                heading_computation_source = "pose_schema.metadata.heading_computation"
        if heading_computation is None:
            heading_computation = _coerce_json_mapping(group.attrs.get("heading_computation"))
            if heading_computation is not None:
                heading_computation_source = "heading_computation"

    return (
        skeleton_id,
        kpt_shape,
        pose_schema_name,
        pose_schema,
        heading_computation_source,
        heading_computation,
    )


def _select_latest_run(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = sorted(str(name) for name in parent.group_keys())
    except Exception:
        names = sorted(str(name) for name in parent.keys())
    return names[-1] if names else None


def _resolve_refined_parent(root: zarr.Group) -> tuple[Optional[str], Optional[zarr.Group]]:
    for name in REFINED_PARENT_NAMES:
        parent = root.get(name)
        if parent is not None:
            return name, parent
    return None, None


def _resolve_review_fields(refined_group: Optional[zarr.Group]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    if refined_group is None:
        return None, None, None, None
    review = _coerce_mapping(refined_group.attrs.get("keypoint_review_status"))
    if not review:
        return None, None, None, None
    timestamp = None
    for key in REVIEW_TIMESTAMP_KEYS:
        timestamp = _normalize_text(review.get(key))
        if timestamp is not None:
            break
    return (
        _normalize_text(review.get("state")),
        _normalize_text(review.get("method")),
        _normalize_text(review.get("intended_use")),
        timestamp,
    )


def _extract_subject_snapshot(root: zarr.Group) -> dict[str, Any]:
    analysis_meta = root.get("analysis_metadata")
    if analysis_meta is None:
        return {}
    for key in ("subject_metadata", "zebrobot_snapshot"):
        payload = _coerce_mapping(analysis_meta.attrs.get(key))
        if payload:
            return payload
    return {}


def _extract_composition(root: zarr.Group) -> dict[str, Any]:
    session_context: dict[str, Any] = {}
    analysis_meta = root.get("analysis_metadata")
    if analysis_meta is not None:
        payload = _coerce_mapping(analysis_meta.attrs.get("session_context"))
        if payload:
            session_context = payload
    subject_snapshot = _extract_subject_snapshot(root)
    dish_map = _coerce_mapping(subject_snapshot.get("dish")) if subject_snapshot else None
    dish_map = dish_map or {}

    protocol_from_context = _normalize_text(
        session_context.get("protocol_name") or session_context.get("protocol_name_from_definition")
    )
    genotype_from_snapshot = _normalize_text(
        dish_map.get("genotype") or subject_snapshot.get("genotype")
    )
    dpf_from_snapshot = _as_int(
        subject_snapshot.get("dpf_at_acquisition")
        or subject_snapshot.get("days_post_fertilization")
    )

    composition: dict[str, Any] = {}
    for key in COMPOSITION_FIELDS:
        if key == "protocol_name":
            value = _normalize_text(root.attrs.get("protocol_name")) or protocol_from_context
        elif key == "genotype":
            value = (
                _normalize_text(root.attrs.get("genotype"))
                or _normalize_text(session_context.get("genotype"))
                or genotype_from_snapshot
            )
        elif key == "dpf_at_acquisition":
            value = _as_int(root.attrs.get("dpf_at_acquisition"))
            if value is None:
                value = _as_int(session_context.get("dpf_at_acquisition"))
            if value is None:
                value = _as_int(session_context.get("days_post_fertilization"))
            if value is None:
                value = dpf_from_snapshot
        else:
            value = _normalize_text(root.attrs.get(key)) or _normalize_text(session_context.get(key))
        if value is not None:
            composition[key] = value
    return composition


def resolve_keypoint_source(
    root: zarr.Group,
    *,
    source_keypoint_path: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    refined_run: Optional[str] = None,
) -> ResolvedKeypointSource:
    keypoints_parent = root.get("keypoints_runs")
    refined_parent_name, refined_parent = _resolve_refined_parent(root)

    keypoint_run = _normalize_text(keypoint_run)
    refined_run = _normalize_text(refined_run)

    def _keypoint_only(run_name: str) -> ResolvedKeypointSource:
        if keypoints_parent is None or run_name not in keypoints_parent:
            raise KeypointSourceError(f"keypoint run not found: {run_name}")
        kp_group = keypoints_parent[run_name]
        (
            skeleton_id,
            kpt_shape,
            pose_schema_name,
            pose_schema,
            heading_computation_source,
            heading_computation,
        ) = _resolve_source_metadata(kp_group)
        return ResolvedKeypointSource(
            keypoint_path=f"keypoints_runs/{run_name}",
            keypoint_type="keypoints",
            keypoint_method=_normalize_text(kp_group.attrs.get("method")),
            keypoint_run=run_name,
            refined_run=None,
            review_state=None,
            review_method=None,
            review_intended_use=None,
            review_timestamp_utc=None,
            skeleton_id=skeleton_id,
            kpt_shape=kpt_shape,
            pose_schema_name=pose_schema_name,
            pose_schema=pose_schema,
            heading_computation_source=heading_computation_source,
            heading_computation=heading_computation,
        )

    if source_keypoint_path:
        source_path = source_keypoint_path.strip("/")
        parts = source_path.split("/")
        if len(parts) >= 2 and parts[0] == "keypoints_runs":
            return _keypoint_only(parts[1])
        if len(parts) >= 2 and parts[0] in REFINED_PARENT_NAMES:
            parent_name = parts[0]
            refined_name = parts[1]
            parent = root.get(parent_name)
            if parent is None or refined_name not in parent:
                raise KeypointSourceError(f"refined run not found: {refined_name}")
            refined_group = parent[refined_name]
            source_run = _normalize_text(
                refined_group.attrs.get("source_keypoints_run")
                or refined_group.attrs.get("source_keypoint_run")
            )
            method = None
            if source_run and keypoints_parent is not None and source_run in keypoints_parent:
                method = _normalize_text(keypoints_parent[source_run].attrs.get("method"))
            review_state, review_method, review_use, review_timestamp = _resolve_review_fields(refined_group)
            (
                skeleton_id,
                kpt_shape,
                pose_schema_name,
                pose_schema,
                heading_computation_source,
                heading_computation,
            ) = _resolve_source_metadata(refined_group)
            return ResolvedKeypointSource(
                keypoint_path=f"{parent_name}/{refined_name}",
                keypoint_type="refined",
                keypoint_method=method,
                keypoint_run=source_run,
                refined_run=refined_name,
                review_state=review_state,
                review_method=review_method,
                review_intended_use=review_use,
                review_timestamp_utc=review_timestamp,
                skeleton_id=skeleton_id,
                kpt_shape=kpt_shape,
                pose_schema_name=pose_schema_name,
                pose_schema=pose_schema,
                heading_computation_source=heading_computation_source,
                heading_computation=heading_computation,
            )
        raise KeypointSourceError(f"unsupported source_keypoint_path format: {source_path}")

    if refined_parent is not None:
        chosen_refined = refined_run or _select_latest_run(refined_parent)
        if chosen_refined is not None:
            if chosen_refined not in refined_parent:
                raise KeypointSourceError(f"refined run not found: {chosen_refined}")
            refined_group = refined_parent[chosen_refined]
            source_run = _normalize_text(
                refined_group.attrs.get("source_keypoints_run")
                or refined_group.attrs.get("source_keypoint_run")
            )
            if keypoint_run and source_run and source_run != keypoint_run:
                raise KeypointSourceError(
                    f"refined run source keypoint mismatch: expected={keypoint_run} found={source_run}"
                )
            if source_run is None:
                source_run = keypoint_run
            method = None
            if source_run and keypoints_parent is not None and source_run in keypoints_parent:
                method = _normalize_text(keypoints_parent[source_run].attrs.get("method"))
            review_state, review_method, review_use, review_timestamp = _resolve_review_fields(refined_group)
            (
                skeleton_id,
                kpt_shape,
                pose_schema_name,
                pose_schema,
                heading_computation_source,
                heading_computation,
            ) = _resolve_source_metadata(refined_group)
            return ResolvedKeypointSource(
                keypoint_path=f"{refined_parent_name}/{chosen_refined}",
                keypoint_type="refined",
                keypoint_method=method,
                keypoint_run=source_run,
                refined_run=chosen_refined,
                review_state=review_state,
                review_method=review_method,
                review_intended_use=review_use,
                review_timestamp_utc=review_timestamp,
                skeleton_id=skeleton_id,
                kpt_shape=kpt_shape,
                pose_schema_name=pose_schema_name,
                pose_schema=pose_schema,
                heading_computation_source=heading_computation_source,
                heading_computation=heading_computation,
            )

    if keypoints_parent is None:
        raise KeypointSourceError("keypoints_runs group missing")
    chosen_run = keypoint_run or _select_latest_run(keypoints_parent)
    if chosen_run is None:
        raise KeypointSourceError("no keypoint run available")
    return _keypoint_only(chosen_run)


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _metric_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    stats: dict[str, Any] = {"count": int(arr.size)}
    if arr.size == 0:
        stats.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "p10": None,
                "p50": None,
                "p90": None,
            }
        )
        return stats
    stats.update(
        {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
        }
    )
    return stats


def _edge_labels_from_source(source_group: zarr.Group, edge_pairs: np.ndarray) -> list[str]:
    labels_raw = source_group.attrs.get("edge_distance_labels")
    labels: list[str] = []
    if isinstance(labels_raw, (list, tuple)):
        labels = [str(item).strip() for item in labels_raw if str(item).strip()]
    if len(labels) == int(edge_pairs.shape[0]):
        return labels

    keypoint_labels_raw = source_group.attrs.get("keypoint_labels")
    keypoint_labels: list[str] = []
    if isinstance(keypoint_labels_raw, (list, tuple)):
        keypoint_labels = [str(item).strip() for item in keypoint_labels_raw if str(item).strip()]

    derived: list[str] = []
    for src, dst in edge_pairs.tolist():
        src_i = int(src)
        dst_i = int(dst)
        if 0 <= src_i < len(keypoint_labels) and 0 <= dst_i < len(keypoint_labels):
            derived.append(f"{keypoint_labels[src_i]}-{keypoint_labels[dst_i]}")
        else:
            derived.append(f"{src_i}-{dst_i}")
    return derived


def _build_edge_distance_payload(
    source_group: zarr.Group,
    *,
    rows_total: Optional[int],
) -> Optional[dict[str, Any]]:
    if "edge_pairs" not in source_group or "edge_distances" not in source_group:
        return None
    edge_pairs = np.asarray(source_group["edge_pairs"][:], dtype=np.int64)
    if edge_pairs.ndim != 2 or edge_pairs.shape[1] < 2:
        return None
    edge_pairs = edge_pairs[:, :2]
    edge_count = int(edge_pairs.shape[0])
    if edge_count <= 0:
        return None

    edge_distances = np.asarray(source_group["edge_distances"][:], dtype=np.float64)
    if edge_distances.ndim != 2 or edge_distances.shape[1] != edge_count:
        return None

    edge_distances_norm = None
    if "edge_distances_norm" in source_group:
        candidate = np.asarray(source_group["edge_distances_norm"][:], dtype=np.float64)
        if candidate.shape == edge_distances.shape:
            edge_distances_norm = candidate

    edge_distance_valid = None
    if "edge_distance_valid" in source_group:
        candidate = np.asarray(source_group["edge_distance_valid"][:], dtype=bool)
        if candidate.shape == edge_distances.shape:
            edge_distance_valid = candidate
    if edge_distance_valid is None:
        edge_distance_valid = np.isfinite(edge_distances)

    total_rows = int(edge_distances.shape[0]) if rows_total is None else int(rows_total)
    normalization = _coerce_mapping(source_group.attrs.get("edge_distance_normalization"))
    labels = _edge_labels_from_source(source_group, edge_pairs)
    edges: list[dict[str, Any]] = []
    for edge_idx in range(edge_count):
        valid = np.asarray(edge_distance_valid[:, edge_idx], dtype=bool)
        valid_count = int(np.count_nonzero(valid))
        rate = _safe_ratio(float(valid_count), float(total_rows))
        edge_info: dict[str, Any] = {
            "edge": [int(edge_pairs[edge_idx, 0]), int(edge_pairs[edge_idx, 1])],
            "label": labels[edge_idx],
            "valid_count": valid_count,
            "valid_rate": rate,
            "distance": _metric_stats(edge_distances[valid, edge_idx]),
        }
        if edge_distances_norm is not None:
            edge_info["distance_norm"] = _metric_stats(edge_distances_norm[valid, edge_idx])
        edges.append(edge_info)

    return {
        "total_rows": total_rows,
        "edge_order": edge_pairs.tolist(),
        "edge_labels": labels,
        "normalization": normalization or {},
        "edges": edges,
    }


def _build_derived_metric_payload(
    source_group: zarr.Group,
    *,
    rows_total: Optional[int],
) -> Optional[dict[str, Any]]:
    required_arrays = ("derived_metric_values", "derived_metric_valid")
    if not all(name in source_group for name in required_arrays):
        return None

    values = np.asarray(source_group["derived_metric_values"][:], dtype=np.float64)
    if values.ndim != 2:
        return None
    metric_count = int(values.shape[1])
    if metric_count <= 0:
        return None

    valid = np.asarray(source_group["derived_metric_valid"][:], dtype=bool)
    if valid.shape != values.shape:
        return None

    values_norm = None
    if "derived_metric_values_norm" in source_group:
        candidate = np.asarray(source_group["derived_metric_values_norm"][:], dtype=np.float64)
        if candidate.shape == values.shape:
            values_norm = candidate

    labels_raw = source_group.attrs.get("derived_metric_labels")
    labels: list[str] = []
    if isinstance(labels_raw, (list, tuple)):
        labels = [str(item).strip() for item in labels_raw if str(item).strip()]
    if len(labels) != metric_count:
        labels = [f"metric_{idx}" for idx in range(metric_count)]

    total_rows = int(values.shape[0]) if rows_total is None else int(rows_total)
    payload: dict[str, Any] = {
        "schema_id": _normalize_text(source_group.attrs.get("derived_metric_schema_id")),
        "schema_version": _normalize_text(source_group.attrs.get("derived_metric_schema_version")),
        "labels": labels,
        "normalization": _coerce_mapping(source_group.attrs.get("derived_metric_normalization")) or {},
        "metrics": [],
    }
    for metric_idx, label in enumerate(labels):
        metric_valid = np.asarray(valid[:, metric_idx], dtype=bool)
        valid_count = int(np.count_nonzero(metric_valid))
        metric_info: dict[str, Any] = {
            "name": label,
            "valid_count": valid_count,
            "valid_rate": _safe_ratio(float(valid_count), float(total_rows)),
            "stats": _metric_stats(values[metric_valid, metric_idx]),
        }
        if values_norm is not None:
            metric_info["stats_norm"] = _metric_stats(values_norm[metric_valid, metric_idx])
        payload["metrics"].append(metric_info)
    return payload


def _load_source_group(root: zarr.Group, source: ResolvedKeypointSource) -> zarr.Group:
    try:
        group = root[source.keypoint_path]
    except Exception as exc:
        raise KeypointProfileError(f"source keypoint path missing: {source.keypoint_path}") from exc
    return group


def build_keypoint_profile_summary(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    source_keypoint_path: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    created_at_utc = created_at_utc or _utc_now()
    source = resolve_keypoint_source(
        root,
        source_keypoint_path=source_keypoint_path,
        keypoint_run=keypoint_run,
        refined_run=refined_run,
    )
    source_group = _load_source_group(root, source)

    keypoints = None
    if "keypoints_roi" in source_group:
        keypoints = np.asarray(source_group["keypoints_roi"][:], dtype=np.float64)
    rows_total = int(keypoints.shape[0]) if isinstance(keypoints, np.ndarray) and keypoints.ndim >= 1 else None

    summary_stats = _coerce_mapping(source_group.attrs.get("summary_statistics")) or {}
    rows_usable = None
    usable_rate = None
    if "usable_keypoints" in source_group:
        usable = np.asarray(source_group["usable_keypoints"][:], dtype=np.bool_)
        rows_usable = int(np.sum(usable))
        if rows_total is None:
            rows_total = int(usable.size)
        usable_rate = _safe_ratio(float(rows_usable), float(rows_total)) if rows_total is not None else None
    if rows_usable is None:
        rows_usable = _as_int(summary_stats.get("usable_keypoints"))
    if rows_total is None:
        rows_total = _as_int(summary_stats.get("total_rois"))
    if usable_rate is None:
        usable_rate = _safe_ratio(
            float(rows_usable) if rows_usable is not None else None,
            float(rows_total) if rows_total is not None else None,
        )

    confidence_valid_rate = None
    if "confidence_valid" in source_group:
        conf_valid = np.asarray(source_group["confidence_valid"][:], dtype=np.bool_)
        confidence_valid_rate = _safe_ratio(float(np.sum(conf_valid)), float(conf_valid.size))
    elif rows_total is not None:
        low_confidence = _as_int(summary_stats.get("low_confidence"))
        confidence_missing = _as_int(summary_stats.get("confidence_missing"))
        if low_confidence is not None and confidence_missing is not None:
            valid = int(rows_total) - int(low_confidence) - int(confidence_missing)
            confidence_valid_rate = _safe_ratio(float(max(0, valid)), float(rows_total))

    geometry_valid_rate = None
    if "geometry_valid" in source_group:
        geom_valid = np.asarray(source_group["geometry_valid"][:], dtype=np.bool_)
        geometry_valid_rate = _safe_ratio(float(np.sum(geom_valid)), float(geom_valid.size))
    elif rows_total is not None:
        geometry_issues = _as_int(summary_stats.get("geometry_issues"))
        if geometry_issues is not None:
            valid = int(rows_total) - int(geometry_issues)
            geometry_valid_rate = _safe_ratio(float(max(0, valid)), float(rows_total))

    geometry_payload: dict[str, Any] = {}
    for metric_name in GEOMETRY_ARRAY_NAMES:
        if metric_name in source_group:
            values = np.asarray(source_group[metric_name][:], dtype=np.float64)
            geometry_payload[metric_name] = {"stats": _metric_stats(values)}
        else:
            geometry_payload[metric_name] = {"stats": _metric_stats(np.asarray([], dtype=np.float64))}
    edge_distance_payload = _build_edge_distance_payload(
        source_group,
        rows_total=rows_total,
    )
    if edge_distance_payload is not None:
        geometry_payload["edge_distance"] = edge_distance_payload
    derived_metrics_schema = _coerce_json_mapping(source_group.attrs.get("derived_metrics_schema"))
    if derived_metrics_schema is not None:
        geometry_payload["derived_metrics_schema"] = derived_metrics_schema
    derived_metric_payload = _build_derived_metric_payload(
        source_group,
        rows_total=rows_total,
    )
    if derived_metric_payload is not None:
        geometry_payload["derived_metrics"] = derived_metric_payload

    dataset_id = _normalize_text(dataset_id) or _normalize_text(root.attrs.get("dataset_id"))
    recording_id = (
        _normalize_text(recording_id)
        or _normalize_text(root.attrs.get("recording_id"))
        or _normalize_text(root.attrs.get("session_uuid"))
    )
    zarr_use = _normalize_text(zarr_use) or infer_zarr_use(root, zarr_path)
    zarr_path_text = str(zarr_path) if zarr_path is not None else None

    summary: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "dataset": {
            "dataset_id": dataset_id,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "zarr_path": zarr_path_text,
        },
        "source": {
            "keypoint_path": source.keypoint_path,
            "keypoint_type": source.keypoint_type,
            "keypoint_method": source.keypoint_method,
            "keypoint_run": source.keypoint_run,
            "refined_run": source.refined_run,
            "review_state": source.review_state,
            "review_method": source.review_method,
            "review_intended_use": source.review_intended_use,
            "review_timestamp_utc": source.review_timestamp_utc,
            "skeleton_id": source.skeleton_id,
            "kpt_shape": list(source.kpt_shape) if source.kpt_shape is not None else None,
            "pose_schema_name": source.pose_schema_name,
            "pose_schema": source.pose_schema,
            "heading_computation_source": source.heading_computation_source,
            "heading_computation": source.heading_computation,
        },
        "quality": {
            "rows_total": rows_total,
            "rows_usable": rows_usable,
            "usable_keypoints_total": rows_usable,
            "usable_rate": usable_rate,
            "confidence_valid_rate": confidence_valid_rate,
            "geometry_valid_rate": geometry_valid_rate,
        },
        "geometry": geometry_payload,
    }

    composition = _extract_composition(root)
    if composition:
        summary["composition"] = composition

    return summary


def write_keypoint_profile(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    source_keypoint_path: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> KeypointProfileWriteResult:
    summary = build_keypoint_profile_summary(
        root,
        zarr_path=zarr_path,
        source_keypoint_path=source_keypoint_path,
        keypoint_run=keypoint_run,
        refined_run=refined_run,
        dataset_id=dataset_id,
        recording_id=recording_id,
        zarr_use=zarr_use,
        created_at_utc=created_at_utc,
    )
    created = _normalize_text(summary.get("created_at_utc")) or _utc_now()
    run_name = _normalize_text(run_name) or _default_profile_run_name(created)

    def _open_child_group(parent: zarr.Group, name: str) -> Optional[zarr.Group]:
        store = getattr(parent, "store", None)
        if store is None:
            return None
        parent_path = _normalize_text(getattr(parent, "path", None))
        child_path = f"{parent_path}/{name}" if parent_path else name
        try:
            return zarr.open_group(store=store, path=child_path, mode="a")
        except TypeError:
            try:
                return zarr.open_group(store, mode="a", path=child_path)
            except Exception:
                return None
        except Exception:
            return None

    def _get_or_create_group(parent: zarr.Group, name: str) -> zarr.Group:
        existing = parent.get(name)
        if existing is None:
            try:
                existing = parent[name]
            except Exception:
                existing = None
        if existing is None:
            existing = _open_child_group(parent, name)
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
                existing = _open_child_group(parent, name)
            if existing is not None:
                return existing
            raise

    analysis_group = _get_or_create_group(root, "analysis")
    runs_parent = require_runs_parent(analysis_group, "keypoint_profile_runs")
    if run_name in runs_parent:
        if not overwrite:
            raise FileExistsError(f"analysis/keypoint_profile_runs/{run_name} already exists")
        del runs_parent[run_name]
    run_group = runs_parent.create_group(run_name)
    mark_run_started(run_group, run_name=run_name, stage="keypoint_profile")

    dataset = summary.get("dataset", {})
    source = summary.get("source", {})
    quality = summary.get("quality", {})

    run_group.attrs.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": created,
            "source_dataset_id": dataset.get("dataset_id"),
            "source_recording_id": dataset.get("recording_id"),
            "source_zarr_use": dataset.get("zarr_use"),
            "source_keypoint_path": source.get("keypoint_path"),
            "source_keypoint_method": source.get("keypoint_method"),
            "source_keypoint_run": source.get("keypoint_run"),
            "source_refined_run": source.get("refined_run"),
            "source_skeleton_id": source.get("skeleton_id"),
            "source_kpt_shape": source.get("kpt_shape"),
            "source_pose_schema_name": source.get("pose_schema_name"),
            "source_pose_schema": source.get("pose_schema"),
            "source_heading_computation_source": source.get("heading_computation_source"),
            "source_heading_computation": source.get("heading_computation"),
            "source_row_count": _as_int(quality.get("rows_total")),
            "profile_summary": summary,
        }
    )
    runs_parent.attrs["latest"] = run_name
    mark_run_complete(run_group, parent_group=runs_parent, run_name=run_name)

    return KeypointProfileWriteResult(
        run_name=run_name,
        source_keypoint_path=str(source.get("keypoint_path") or ""),
        source_keypoint_method=_normalize_text(source.get("keypoint_method")),
        profile_summary=summary,
    )


__all__ = [
    "KeypointProfileError",
    "KeypointProfileWriteResult",
    "KeypointSourceError",
    "ResolvedKeypointSource",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "build_keypoint_profile_summary",
    "infer_zarr_use",
    "resolve_keypoint_source",
    "write_keypoint_profile",
]
