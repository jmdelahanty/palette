from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run
from fisheye.utils.detection_profile import infer_zarr_use
from fisheye.utils.refined_eye_masks_compat import refined_eye_masks_compat_context


SCHEMA_NAME = "eye_mask_dataset_profile"
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

REVIEW_TIMESTAMP_KEYS = (
    "review_timestamp_utc",
    "timestamp_utc",
    "timestamp",
    "reviewed_at_utc",
    "reviewed_at",
    "updated_utc",
)


class EyeMaskProfileError(RuntimeError):
    """Base error for eye-mask profile read/write failures."""


class EyeMaskSourceError(EyeMaskProfileError):
    """Raised when no usable eye-mask source run can be resolved."""


@dataclass(frozen=True)
class ResolvedEyeMaskSource:
    eye_mask_path: str
    stage_group: str
    stage_role: str
    stage_label: str
    authority_stage_group: str
    compatibility_role: Optional[str]
    eye_mask_run: str
    eye_mask_method: Optional[str]
    source_eye_masks_run: Optional[str]
    source_refined_subject_masks_run: Optional[str]
    source_subject_mask_run: Optional[str]
    source_keypoint_group: Optional[str]
    source_keypoint_run: Optional[str]
    source_keypoint_path: Optional[str]
    source_crop_run: Optional[str]
    review_state: Optional[str]
    review_method: Optional[str]
    review_intended_use: Optional[str]
    review_reviewer: Optional[str]
    review_notes: Optional[str]
    review_timestamp_utc: Optional[str]
    source_keypoint_stale: Optional[dict[str, Any]]


@dataclass(frozen=True)
class EyeMaskProfileWriteResult:
    run_name: str
    source_eye_mask_path: str
    source_eye_mask_method: Optional[str]
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
    return f"eye_mask_profile_{stamp.strftime('%Y-%m-%d_%H-%M-%S')}"


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
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _to_json_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return None


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="linear"))


def _metric_stats(values: Optional[np.ndarray]) -> dict[str, Any]:
    if values is None:
        arr = np.asarray([], dtype=np.float64)
    else:
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
            "p10": _quantile(arr, 0.10),
            "p50": _quantile(arr, 0.50),
            "p90": _quantile(arr, 0.90),
        }
    )
    return stats


def _metric_payload(values: Optional[np.ndarray]) -> dict[str, Any]:
    return {"stats": _metric_stats(values)}


def _mask_metric_values(values: Optional[np.ndarray], mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if values is None or mask is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if arr.ndim == 0:
        return None
    if int(arr.shape[0]) != int(mask_arr.shape[0]):
        return None
    return arr[mask_arr]


def _open_child_group(parent: zarr.Group, key: str, *, mode: str = "r") -> Optional[zarr.Group]:
    store = getattr(parent, "store", None)
    if store is None:
        return None
    parent_path = _normalize_text(getattr(parent, "path", None))
    child_path = f"{parent_path}/{key}" if parent_path else key
    try:
        return zarr.open_group(store=store, path=child_path, mode=mode)
    except TypeError:
        try:
            return zarr.open_group(store, mode=mode, path=child_path)
        except Exception:
            return None
    except Exception:
        return None


def _get_group(parent: zarr.Group, key: str) -> Optional[zarr.Group]:
    child = parent.get(key)
    if child is not None:
        return child
    try:
        child = parent[key]
    except Exception:
        child = None
    if child is None:
        child = _open_child_group(parent, key)
    return child


def _get_group_by_path(root: zarr.Group, path: str) -> Optional[zarr.Group]:
    current: Optional[zarr.Group] = root
    for token in str(path).split("/"):
        name = token.strip()
        if not name:
            continue
        if current is None:
            return None
        current = _get_group(current, name)
    return current


def _select_latest_run(parent: zarr.Group) -> Optional[str]:
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and _get_group(parent, latest) is not None:
        return latest
    try:
        names = sorted(str(name) for name in parent.group_keys())
    except Exception:
        names = sorted(str(name) for name in parent.keys())
    return names[-1] if names else None


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
    genotype_from_snapshot = _normalize_text(dish_map.get("genotype") or subject_snapshot.get("genotype"))
    dpf_from_snapshot = _as_int(
        subject_snapshot.get("dpf_at_acquisition") or subject_snapshot.get("days_post_fertilization")
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


def _resolve_source_keypoint_path(
    root: zarr.Group,
    *,
    source_group: Optional[str],
    source_run: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    run_name = _normalize_text(source_run)
    group_name = _normalize_text(source_group)
    if run_name is None:
        return None, group_name

    candidates: list[str] = []
    if group_name:
        candidates.append(group_name)
    for fallback in ("refined_keypoints_runs", "keypoints_runs"):
        if fallback not in candidates:
            candidates.append(fallback)

    for candidate in candidates:
        parent = _get_group(root, candidate)
        if parent is not None and _get_group(parent, run_name) is not None:
            return f"{candidate}/{run_name}", candidate
    return None, group_name


def _review_payload(attrs: Mapping[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    review = _coerce_mapping(attrs.get("eye_mask_review_status")) or {}
    timestamp = None
    for key in REVIEW_TIMESTAMP_KEYS:
        timestamp = _normalize_text(review.get(key))
        if timestamp is not None:
            break
    return review, timestamp


def resolve_eye_mask_source(
    root: zarr.Group,
    *,
    source_eye_mask_path: Optional[str] = None,
    eye_mask_run: Optional[str] = None,
    refined_run: Optional[str] = None,
) -> ResolvedEyeMaskSource:
    eye_parent = _get_group(root, "eye_masks_runs")
    refined_parent = _get_group(root, "refined_eye_masks_runs")

    eye_mask_run = _normalize_text(eye_mask_run)
    refined_run = _normalize_text(refined_run)

    def _from_stage(stage_group: str, run_name: str) -> ResolvedEyeMaskSource:
        parent = _get_group(root, stage_group)
        if parent is None:
            raise EyeMaskSourceError(f"eye-mask run not found: {stage_group}/{run_name}")
        run_group = _get_group(parent, run_name)
        if run_group is None:
            raise EyeMaskSourceError(f"eye-mask run not found: {stage_group}/{run_name}")
        attrs = dict(run_group.attrs)
        compat_context = refined_eye_masks_compat_context(attrs, base_stage=stage_group)

        source_keypoint_run = _normalize_text(resolve_source_keypoints_run(attrs))
        source_keypoint_group = _normalize_text(attrs.get("source_keypoint_group"))
        source_keypoint_path, resolved_source_group = _resolve_source_keypoint_path(
            root,
            source_group=source_keypoint_group,
            source_run=source_keypoint_run,
        )
        if source_keypoint_group is None:
            source_keypoint_group = resolved_source_group

        review_map, review_timestamp = _review_payload(attrs)

        return ResolvedEyeMaskSource(
            eye_mask_path=f"{stage_group}/{run_name}",
            stage_group=stage_group,
            stage_role=str(compat_context.get("stage_role") or "canonical"),
            stage_label=str(compat_context.get("stage_label") or stage_group),
            authority_stage_group=str(compat_context.get("authority_stage_group") or stage_group),
            compatibility_role=_normalize_text(compat_context.get("compatibility_role")),
            eye_mask_run=run_name,
            eye_mask_method=_normalize_text(attrs.get("method")),
            source_eye_masks_run=_normalize_text(attrs.get("source_eye_masks_run")),
            source_refined_subject_masks_run=_normalize_text(compat_context.get("source_refined_subject_masks_run")),
            source_subject_mask_run=_normalize_text(compat_context.get("source_subject_mask_run")),
            source_keypoint_group=source_keypoint_group,
            source_keypoint_run=source_keypoint_run,
            source_keypoint_path=source_keypoint_path,
            source_crop_run=_normalize_text(attrs.get("source_crop_run")),
            review_state=_normalize_text(review_map.get("state")),
            review_method=_normalize_text(review_map.get("method")),
            review_intended_use=_normalize_text(review_map.get("intended_use")),
            review_reviewer=_normalize_text(review_map.get("reviewer")),
            review_notes=_normalize_text(review_map.get("notes")),
            review_timestamp_utc=review_timestamp,
            source_keypoint_stale=_coerce_mapping(attrs.get("source_keypoint_stale")),
        )

    if source_eye_mask_path:
        source_path = source_eye_mask_path.strip("/")
        parts = source_path.split("/")
        if len(parts) >= 2 and parts[0] in {"eye_masks_runs", "refined_eye_masks_runs"}:
            return _from_stage(parts[0], parts[1])
        raise EyeMaskSourceError(f"unsupported source_eye_mask_path format: {source_path}")

    if refined_run and refined_parent is None:
        raise EyeMaskSourceError("refined_eye_masks_runs group missing")

    if refined_parent is not None:
        refined_choice = refined_run or _select_latest_run(refined_parent)
        if refined_choice is not None:
            source = _from_stage("refined_eye_masks_runs", refined_choice)
            if eye_mask_run and source.source_eye_masks_run and source.source_eye_masks_run != eye_mask_run:
                raise EyeMaskSourceError(
                    f"refined run source eye-mask mismatch: expected={eye_mask_run} found={source.source_eye_masks_run}"
                )
            return source

    if eye_parent is None:
        raise EyeMaskSourceError("eye_masks_runs group missing")

    raw_choice = eye_mask_run or _select_latest_run(eye_parent)
    if raw_choice is None:
        raise EyeMaskSourceError("no eye-mask run available")
    return _from_stage("eye_masks_runs", raw_choice)


def _load_source_group(root: zarr.Group, source: ResolvedEyeMaskSource) -> zarr.Group:
    group = _get_group_by_path(root, source.eye_mask_path)
    if group is None:
        raise EyeMaskProfileError(f"source eye-mask path missing: {source.eye_mask_path}")
    return group


def _metric_from_axis(array: Optional[np.ndarray], axis: int) -> Optional[np.ndarray]:
    if array is None:
        return None
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[1] <= axis:
        return None
    values = arr[:, axis]
    values[~np.isfinite(values)] = np.nan
    return values


def _pairwise_mean(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim < 2:
        return None
    out = np.full((arr.shape[0],), np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=1)
    if np.any(counts > 0):
        sums = np.nansum(arr, axis=1)
        valid_rows = counts > 0
        out[valid_rows] = sums[valid_rows] / counts[valid_rows]
    return out


def _load_metrics_geometry(source_group: zarr.Group) -> dict[str, Optional[np.ndarray]]:
    metrics_group = source_group.get("metrics")
    if metrics_group is None:
        return {
            "left_area": None,
            "right_area": None,
            "union_area": None,
            "area_lr_ratio": None,
            "aspect_ratio": None,
            "eye_separation": None,
            "interocular_px": None,
            "circularity": None,
        }

    area_refined = None
    if "area_refined" in metrics_group:
        area_refined = np.asarray(metrics_group["area_refined"][:], dtype=np.float64)

    axis_ratio = None
    if "axis_ratio" in metrics_group:
        axis_ratio = np.asarray(metrics_group["axis_ratio"][:], dtype=np.float64)

    circularity = None
    if "circularity" in metrics_group:
        circularity = np.asarray(metrics_group["circularity"][:], dtype=np.float64)

    return {
        "left_area": _metric_from_axis(area_refined, 0),
        "right_area": _metric_from_axis(area_refined, 1),
        "union_area": (
            np.asarray(metrics_group["area_union_refined"][:], dtype=np.float64)
            if "area_union_refined" in metrics_group
            else None
        ),
        "area_lr_ratio": (
            np.asarray(metrics_group["area_ratio_left_right"][:], dtype=np.float64)
            if "area_ratio_left_right" in metrics_group
            else None
        ),
        "aspect_ratio": _pairwise_mean(axis_ratio),
        "eye_separation": (
            np.asarray(metrics_group["separation_refined"][:], dtype=np.float64)
            if "separation_refined" in metrics_group
            else None
        ),
        "interocular_px": (
            np.asarray(metrics_group["separation_keypoint"][:], dtype=np.float64)
            if "separation_keypoint" in metrics_group
            else None
        ),
        "circularity": _pairwise_mean(circularity),
    }


def _load_ellipse_axes(source_group: zarr.Group) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if "ellipse_params" not in source_group:
        return None, None
    params = np.asarray(source_group["ellipse_params"][:], dtype=np.float64)
    if params.ndim != 3 or params.shape[1] < 2 or params.shape[2] < 4:
        return None, None

    major = params[:, :2, 2]
    minor = params[:, :2, 3]
    if "ellipse_success" in source_group:
        success = np.asarray(source_group["ellipse_success"][:], dtype=np.bool_)
        if success.ndim == 2 and success.shape[:2] == major.shape:
            major = np.where(success, major, np.nan)
            minor = np.where(success, minor, np.nan)

    major_pair = _pairwise_mean(major)
    minor_pair = _pairwise_mean(minor)
    return major_pair, minor_pair


def _rows_total(source_group: zarr.Group, attrs: Mapping[str, Any]) -> Optional[int]:
    for key in ("total_rois", "rows_total"):
        value = _as_int(attrs.get(key))
        if value is not None:
            return value

    for array_name in ("ellipse_success", "masks_roi", "eye_separation"):
        if array_name in source_group:
            shape = getattr(source_group[array_name], "shape", None)
            if isinstance(shape, tuple) and shape:
                value = _as_int(shape[0])
                if value is not None:
                    return value
    return None


def _load_ellipse_success(source_group: zarr.Group) -> Optional[np.ndarray]:
    if "ellipse_success" not in source_group:
        return None
    values = np.asarray(source_group["ellipse_success"][:], dtype=np.bool_)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        return None
    return values


def _chunk_rows(array_obj: Any, fallback: int = 128) -> int:
    chunks = getattr(array_obj, "chunks", None)
    if isinstance(chunks, tuple) and chunks:
        return max(1, int(chunks[0]))
    if isinstance(chunks, int):
        return max(1, int(chunks))
    return max(1, int(fallback))


def _load_area_and_edge_from_masks(
    source_group: zarr.Group,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    if "masks_roi" not in source_group:
        return None, None, None, None
    masks_arr = source_group["masks_roi"]
    shape = getattr(masks_arr, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 4:
        return None, None, None, None

    n_rows = int(shape[0])
    if n_rows <= 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, None

    chunk_rows = _chunk_rows(masks_arr, fallback=256)
    left_area = np.zeros((n_rows,), dtype=np.float64)
    right_area = np.zeros((n_rows,), dtype=np.float64)
    union_area = np.zeros((n_rows,), dtype=np.float64)
    edge_hits = np.zeros((n_rows,), dtype=np.bool_)

    for start in range(0, n_rows, chunk_rows):
        stop = min(n_rows, start + chunk_rows)
        block = np.asarray(masks_arr[start:stop], dtype=np.uint8)
        if block.ndim != 4:
            continue
        if block.shape[1] < 1:
            continue

        left = block[:, 0] > 0
        right = block[:, 1] > 0 if block.shape[1] > 1 else left.copy()
        union = np.logical_or(left, right)

        left_area[start:stop] = np.sum(left, axis=(1, 2), dtype=np.int64)
        right_area[start:stop] = np.sum(right, axis=(1, 2), dtype=np.int64)
        union_area[start:stop] = np.sum(union, axis=(1, 2), dtype=np.int64)

        edge = (
            np.any(union[:, 0, :], axis=1)
            | np.any(union[:, -1, :], axis=1)
            | np.any(union[:, :, 0], axis=1)
            | np.any(union[:, :, -1], axis=1)
        )
        edge_hits[start:stop] = edge

    edge_rate = float(np.mean(edge_hits)) if edge_hits.size else None
    return left_area, right_area, union_area, edge_rate


def _first_review_timestamp(source: ResolvedEyeMaskSource) -> Optional[str]:
    return source.review_timestamp_utc


def build_eye_mask_profile_summary(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    source_eye_mask_path: Optional[str] = None,
    eye_mask_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    created_at_utc = created_at_utc or _utc_now()
    source = resolve_eye_mask_source(
        root,
        source_eye_mask_path=source_eye_mask_path,
        eye_mask_run=eye_mask_run,
        refined_run=refined_run,
    )
    source_group = _load_source_group(root, source)
    source_attrs = dict(source_group.attrs)
    summary_stats = _coerce_mapping(source_attrs.get("summary_statistics")) or {}

    rows_total = _rows_total(source_group, source_attrs)

    ellipse_success = _load_ellipse_success(source_group)
    pair_success_mask: Optional[np.ndarray] = None
    if ellipse_success is not None and ellipse_success.size > 0:
        pair_success_mask = np.all(ellipse_success, axis=1)
    rows_usable = _as_int(source_attrs.get("successful_roi_pairs"))
    if rows_usable is None:
        rows_usable = _as_int(summary_stats.get("successful_roi_pairs"))
    if rows_usable is None and pair_success_mask is not None:
        rows_usable = int(np.sum(pair_success_mask))

    usable_rate = _as_float(source_attrs.get("successful_roi_pair_rate"))
    if usable_rate is None:
        usable_rate = _as_float(summary_stats.get("successful_roi_pair_rate"))
    if usable_rate is None:
        usable_rate = _safe_ratio(
            float(rows_usable) if rows_usable is not None else None,
            float(rows_total) if rows_total is not None else None,
        )

    ellipse_success_rate = None
    if ellipse_success is not None and ellipse_success.size > 0:
        ellipse_success_rate = float(np.mean(ellipse_success.astype(np.float64)))
    if ellipse_success_rate is None:
        successful_eyes = _as_int(source_attrs.get("successful_eyes"))
        if successful_eyes is not None and rows_total is not None and rows_total > 0:
            ellipse_success_rate = _safe_ratio(float(successful_eyes), float(rows_total) * 2.0)

    pair_success_rate = usable_rate
    reviewed_rate = 1.0 if source.review_state is not None else None
    excluded_rate = None
    if usable_rate is not None:
        excluded_rate = float(max(0.0, min(1.0, 1.0 - usable_rate)))

    reason_counts = _coerce_mapping(source_attrs.get("reason_counts"))
    if reason_counts is None:
        reason_counts = _coerce_mapping(summary_stats.get("reason_counts"))
    if reason_counts is None:
        postprocess_stats = _coerce_mapping(summary_stats.get("postprocess")) or {}
        reason_counts = _coerce_mapping(postprocess_stats.get("reason_counts"))

    metrics_geometry = _load_metrics_geometry(source_group)
    mask_left, mask_right, mask_union, mask_edge_rate = _load_area_and_edge_from_masks(source_group)

    # Prefer mask-derived areas when available. Manual review edits update masks_roi
    # directly, while metrics arrays can be stale until explicitly refreshed.
    left_area = mask_left if mask_left is not None else metrics_geometry["left_area"]
    right_area = mask_right if mask_right is not None else metrics_geometry["right_area"]
    union_area = mask_union if mask_union is not None else metrics_geometry["union_area"]
    left_area_usable = _mask_metric_values(left_area, pair_success_mask)
    right_area_usable = _mask_metric_values(right_area, pair_success_mask)
    union_area_usable = _mask_metric_values(union_area, pair_success_mask)
    edge_rate = mask_edge_rate

    area_lr_ratio = None
    if left_area is not None and right_area is not None:
        denom = np.maximum(right_area, 1e-6)
        area_lr_ratio = np.asarray(left_area / denom, dtype=np.float64)
    else:
        area_lr_ratio = metrics_geometry["area_lr_ratio"]

    major_axis, minor_axis = _load_ellipse_axes(source_group)
    aspect_ratio = metrics_geometry["aspect_ratio"]
    if aspect_ratio is None and major_axis is not None and minor_axis is not None:
        aspect_ratio = np.asarray(major_axis / np.maximum(minor_axis, 1e-6), dtype=np.float64)

    eye_separation = metrics_geometry["eye_separation"]
    if eye_separation is None and "eye_separation" in source_group:
        eye_separation = np.asarray(source_group["eye_separation"][:], dtype=np.float64)

    interocular_px = metrics_geometry["interocular_px"]
    if interocular_px is None:
        interocular_px = eye_separation

    circularity = metrics_geometry["circularity"]

    ellipse_area = None
    if major_axis is not None and minor_axis is not None:
        ellipse_area = np.asarray(np.pi * np.maximum(major_axis, 0.0) * np.maximum(minor_axis, 0.0) * 0.25)
    if ellipse_area is None:
        ellipse_area = union_area

    area_values = union_area if union_area is not None else ellipse_area
    area_values_usable = union_area_usable if union_area_usable is not None else _mask_metric_values(ellipse_area, pair_success_mask)

    if edge_rate is None:
        edge_rate = _as_float(source_attrs.get("edge_proximity_rate"))
    if edge_rate is None:
        edge_rate = _as_float(summary_stats.get("edge_proximity_rate"))

    review_approved_rate = None
    review_rejected_rate = None
    review_state_norm = (source.review_state or "").strip().lower() if source.review_state else None
    if review_state_norm == "approved":
        review_approved_rate = 1.0
        review_rejected_rate = 0.0
    elif review_state_norm == "rejected":
        review_approved_rate = 0.0
        review_rejected_rate = 1.0

    dataset_id = _normalize_text(dataset_id) or _normalize_text(root.attrs.get("dataset_id"))
    recording_id = (
        _normalize_text(recording_id)
        or _normalize_text(root.attrs.get("recording_id"))
        or _normalize_text(root.attrs.get("session_uuid"))
    )
    zarr_use = _normalize_text(zarr_use) or infer_zarr_use(root, zarr_path)
    zarr_path_text = str(zarr_path) if zarr_path is not None else None

    source_map: dict[str, Any] = {
        "stage_group": source.stage_group,
        "eye_stage_group": source.stage_group,
        "source_eye_stage": source.stage_group,
        "eye_stage_role": source.stage_role,
        "eye_stage_label": source.stage_label,
        "stage_label": source.stage_label,
        "authority_stage_group": source.authority_stage_group,
        "compatibility_role": source.compatibility_role,
        "eye_mask_path": source.eye_mask_path,
        "source_eye_mask_path": source.eye_mask_path,
        "eye_mask_run": source.eye_mask_run,
        "source_eye_mask_run": source.eye_mask_run,
        "eye_mask_method": source.eye_mask_method,
        "method": source.eye_mask_method,
        "eye_masks_run": source.source_eye_masks_run or (source.eye_mask_run if source.stage_group == "eye_masks_runs" else None),
        "refined_eye_masks_run": source.eye_mask_run if source.stage_group == "refined_eye_masks_runs" else None,
        "source_refined_subject_masks_run": source.source_refined_subject_masks_run,
        "canonical_refined_subject_masks_run": source.source_refined_subject_masks_run,
        "source_subject_mask_run": source.source_subject_mask_run,
        "source_keypoint_group": source.source_keypoint_group,
        "source_keypoint_path": source.source_keypoint_path,
        "keypoint_path": source.source_keypoint_path,
        "source_keypoint_run": source.source_keypoint_run,
        "keypoint_run": source.source_keypoint_run,
        "keypoints_run": source.source_keypoint_run,
        "source_crop_run": source.source_crop_run,
        "crop_run": source.source_crop_run,
        "review_state": source.review_state,
        "review_method": source.review_method,
        "review_intended_use": source.review_intended_use,
        "review_reviewer": source.review_reviewer,
        "review_notes": source.review_notes,
        "review_timestamp_utc": _first_review_timestamp(source),
    }

    review_payload = {
        "state": source.review_state,
        "method": source.review_method,
        "intended_use": source.review_intended_use,
        "reviewer": source.review_reviewer,
        "notes": source.review_notes,
        "timestamp_utc": _first_review_timestamp(source),
    }
    review_payload = {k: v for k, v in review_payload.items() if v is not None}
    if review_payload:
        source_map["review"] = review_payload

    stale_map = source.source_keypoint_stale or None
    if stale_map:
        source_map["source_keypoint_stale"] = stale_map

    quality_map: dict[str, Any] = {
        "rows_total": rows_total,
        "total_rows": rows_total,
        "total_rois": rows_total,
        "rows_usable": rows_usable,
        "usable_rows": rows_usable,
        "usable_rois": rows_usable,
        "rows_training_usable": rows_usable,
        "successful_roi_pairs": rows_usable,
        "usable_rate": usable_rate,
        "usable_rows_rate": usable_rate,
        "usable_roi_pair_rate": usable_rate,
        "successful_roi_pair_rate": pair_success_rate,
        "reviewed_rate": reviewed_rate,
        "review_rate": reviewed_rate,
        "excluded_rate": excluded_rate,
        "exclude_rate": excluded_rate,
        "exclusion_reasons": reason_counts,
        "excluded_reasons": reason_counts,
        "exclusion_reasons_json": _to_json_text(reason_counts),
        "ellipse_success_rate": ellipse_success_rate,
        "successful_ellipse_rate": ellipse_success_rate,
        "pair_success_rate": pair_success_rate,
        "review_approved_rate": review_approved_rate,
        "review_rejected_rate": review_rejected_rate,
    }

    geometry_map: dict[str, Any] = {
        "area": _metric_payload(area_values),
        "area_usable": _metric_payload(area_values_usable),
        "left_area": _metric_payload(left_area),
        "left_area_usable": _metric_payload(left_area_usable),
        "area_left": _metric_payload(left_area),
        "area_left_usable": _metric_payload(left_area_usable),
        "left_eye_area": _metric_payload(left_area),
        "left_eye_area_usable": _metric_payload(left_area_usable),
        "right_area": _metric_payload(right_area),
        "right_area_usable": _metric_payload(right_area_usable),
        "area_right": _metric_payload(right_area),
        "area_right_usable": _metric_payload(right_area_usable),
        "right_eye_area": _metric_payload(right_area),
        "right_eye_area_usable": _metric_payload(right_area_usable),
        "union_area": _metric_payload(union_area),
        "union_area_usable": _metric_payload(union_area_usable),
        "area_union": _metric_payload(union_area),
        "area_union_usable": _metric_payload(union_area_usable),
        "combined_area": _metric_payload(union_area),
        "combined_area_usable": _metric_payload(union_area_usable),
        "area_lr_ratio": _metric_payload(area_lr_ratio),
        "left_right_area_ratio": _metric_payload(area_lr_ratio),
        "area_ratio_left_right": _metric_payload(area_lr_ratio),
        "lr_area_ratio": _metric_payload(area_lr_ratio),
        "major_axis": _metric_payload(major_axis),
        "ellipse_major": _metric_payload(major_axis),
        "minor_axis": _metric_payload(minor_axis),
        "ellipse_minor": _metric_payload(minor_axis),
        "aspect_ratio": _metric_payload(aspect_ratio),
        "axis_ratio": _metric_payload(aspect_ratio),
        "eye_separation": _metric_payload(eye_separation),
        "interocular_px": _metric_payload(interocular_px),
        "ellipse_area": _metric_payload(ellipse_area),
        "circularity": _metric_payload(circularity),
    }

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
        "source": source_map,
        "quality": quality_map,
        "geometry": geometry_map,
        "spatial": {
            "edge_proximity_rate": edge_rate,
            "edge_touch_rate": edge_rate,
        },
    }

    composition = _extract_composition(root)
    if composition:
        summary["composition"] = composition

    if stale_map:
        summary["freshness"] = {"source_keypoint_stale": stale_map}

    return summary


def write_eye_mask_profile(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    source_eye_mask_path: Optional[str] = None,
    eye_mask_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> EyeMaskProfileWriteResult:
    summary = build_eye_mask_profile_summary(
        root,
        zarr_path=zarr_path,
        source_eye_mask_path=source_eye_mask_path,
        eye_mask_run=eye_mask_run,
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
    runs_parent = _get_or_create_group(analysis_group, "eye_mask_profile_runs")
    if run_name in runs_parent:
        if not overwrite:
            raise FileExistsError(f"analysis/eye_mask_profile_runs/{run_name} already exists")
        del runs_parent[run_name]
    run_group = runs_parent.create_group(run_name)

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
            "source_stage_group": source.get("stage_group"),
            "source_stage_role": source.get("eye_stage_role"),
            "source_stage_label": source.get("eye_stage_label"),
            "source_authority_stage_group": source.get("authority_stage_group"),
            "source_compatibility_role": source.get("compatibility_role"),
            "source_eye_mask_path": source.get("eye_mask_path"),
            "source_eye_mask_run": source.get("eye_mask_run"),
            "source_eye_mask_method": source.get("eye_mask_method"),
            "source_keypoint_path": source.get("source_keypoint_path"),
            "source_keypoint_run": source.get("source_keypoint_run"),
            "source_keypoints_run": source.get("source_keypoint_run"),
            "source_crop_run": source.get("source_crop_run"),
            "source_eye_masks_run": source.get("eye_masks_run"),
            "source_refined_eye_masks_run": source.get("refined_eye_masks_run"),
            "source_refined_subject_masks_run": source.get("source_refined_subject_masks_run"),
            "source_subject_mask_run": source.get("source_subject_mask_run"),
            "source_row_count": _as_int(quality.get("rows_total")),
            "profile_summary": summary,
        }
    )
    runs_parent.attrs["latest"] = run_name

    return EyeMaskProfileWriteResult(
        run_name=run_name,
        source_eye_mask_path=str(source.get("eye_mask_path") or ""),
        source_eye_mask_method=_normalize_text(source.get("eye_mask_method")),
        profile_summary=summary,
    )


__all__ = [
    "EyeMaskProfileError",
    "EyeMaskProfileWriteResult",
    "EyeMaskSourceError",
    "ResolvedEyeMaskSource",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "build_eye_mask_profile_summary",
    "infer_zarr_use",
    "resolve_eye_mask_source",
    "write_eye_mask_profile",
]
