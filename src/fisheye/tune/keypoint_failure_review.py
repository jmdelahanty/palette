"""
Manual review tool for keypoints in a refined keypoint run.

Edits are applied to the refined run only (raw keypoints remain untouched).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple
from datetime import datetime, timezone
import hashlib
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zarr

from ..pose.heuristics import (
    heuristic_profile_from_package,
    maybe_geometry_qc_from_attrs,
    require_geometry_qc,
)
from ..pose.metric_schema import (
    DerivedMetricStorage,
    compute_derived_metric_results,
    ensure_derived_metric_storage,
    resolve_metric_schema_for_group,
)
from ..pose.heading import compute_heading_from_attrs
from ..shared.detect_reason_codec import (
    MutableReasonColumn,
    open_mutable_reason_column,
    read_reason_labels,
    write_reason_columns,
)
from ..shared.frame_flags import (
    append_flagged_frame as _append_shared_flagged_frame,
    load_frame_flags as _load_shared_frame_flags,
    load_row_identity_arrays,
    row_identity_payload,
)
from ..shared.keypoint_temporal_heading import refresh_refined_keypoint_heading_fields
from ..shared.subject_mask_stale import mark_downstream_subject_mask_runs_stale
from ..registry.stage_complete import DatasetMetadata, emit_stage_completion
from ..shared.type_conversions import normalize_attr as _normalize_attr
from ..shared.zarr_run_completion import resolve_latest_complete_run_name
from ..shared.zarr_io import open_zarr_root
from ..refinement.keypoint_quality import (
    compute_geometry_metrics,
    resolve_head_triangle_for_labels,
    select_head_triangle_points,
)
from ..registry.db import Registry, RegistryPaths


_DEFAULT_LABELS = ("swim_bladder", "eye_left", "eye_right")
_DEFAULT_COLORS = ("#22c55e", "#1a66f3", "#f85151")
_NO_REVIEWABLE_FAILURES_POLICY_ID = "keypoint_no_reviewable_failures_v1"
_NO_REVIEWABLE_FAILURES_POLICY_VERSION = 1
_NO_REVIEWABLE_FAILURES_ALLOWED_TAGS = ("fish_present_no_keypoints",)
_NO_REVIEWABLE_FAILURES_BLOCKED_TAGS = ("detection_issue",)
_TRADITIONAL_HEURISTIC_METHOD = "traditional_pose"
_DEFAULT_CONFIDENCE_THRESHOLD = 0.3

TRADITIONAL_REVIEW_HEURISTIC_PROFILE = heuristic_profile_from_package(
    _TRADITIONAL_HEURISTIC_METHOD,
    "traditional_v1",
)
TRADITIONAL_REVIEW_GEOMETRY_QC = require_geometry_qc(TRADITIONAL_REVIEW_HEURISTIC_PROFILE)


def _required_geometry_default(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(
            "Traditional pose heuristic profile is missing "
            f"'geometry_qc.{field_name}'."
        ) from None
    return result


DEFAULT_MIN_TRIANGLE_ANGLE = _required_geometry_default(
    TRADITIONAL_REVIEW_GEOMETRY_QC.min_triangle_angle_deg,
    "min_triangle_angle_deg",
)
DEFAULT_MIN_TRIANGLE_AREA = _required_geometry_default(
    TRADITIONAL_REVIEW_GEOMETRY_QC.min_triangle_area_px,
    "min_triangle_area_px",
)
DEFAULT_MAX_TRIANGLE_AREA = (
    float(TRADITIONAL_REVIEW_GEOMETRY_QC.max_triangle_area_px)
    if TRADITIONAL_REVIEW_GEOMETRY_QC.max_triangle_area_px is not None
    else None
)


def _resolve_review_geometry_defaults(
    attrs: Dict[str, object],
) -> tuple[float, float, Optional[float]]:
    geometry_qc = maybe_geometry_qc_from_attrs(_TRADITIONAL_HEURISTIC_METHOD, attrs)
    if geometry_qc is None:
        return (
            DEFAULT_MIN_TRIANGLE_ANGLE,
            DEFAULT_MIN_TRIANGLE_AREA,
            DEFAULT_MAX_TRIANGLE_AREA,
        )
    min_angle = _required_geometry_default(
        geometry_qc.min_triangle_angle_deg,
        "min_triangle_angle_deg",
    )
    min_area = _required_geometry_default(
        geometry_qc.min_triangle_area_px,
        "min_triangle_area_px",
    )
    max_area = (
        float(geometry_qc.max_triangle_area_px)
        if geometry_qc.max_triangle_area_px is not None
        else None
    )
    return (min_angle, min_area, max_area)


def _display_colors(label_count: int) -> list[str]:
    if label_count <= len(_DEFAULT_COLORS):
        return list(_DEFAULT_COLORS[:label_count])
    colors = list(_DEFAULT_COLORS)
    cmap = mpl.colormaps["tab10"]
    extra_needed = int(label_count - len(colors))
    for idx in range(extra_needed):
        colors.append(mpl.colors.to_hex(cmap((idx + len(_DEFAULT_COLORS)) % cmap.N)))
    return colors


def _advance_active_index(current_idx: int, *, label_count: int, step: int) -> int:
    if label_count <= 0:
        return 0
    return int((current_idx + step) % label_count)


def _active_index_from_key(key: str | None, *, label_count: int, current_idx: int) -> int | None:
    if not key or label_count <= 0:
        return None
    if key.isdigit():
        idx = int(key) - 1
        if 0 <= idx < label_count:
            return idx
        return None
    if key == "]":
        return _advance_active_index(current_idx, label_count=label_count, step=1)
    if key == "[":
        return _advance_active_index(current_idx, label_count=label_count, step=-1)
    return None


def _get_latest_run(root: zarr.Group, group_name: str) -> str:
    parent_name = f"{group_name}_runs"
    if parent_name not in root:
        raise RuntimeError(f"No '{parent_name}' group found in Zarr store.")
    latest = resolve_latest_complete_run_name(root[parent_name])
    if not latest:
        raise RuntimeError(f"No runs recorded under '{parent_name}'.")
    return latest


def _total_keypoints(refined: zarr.Group) -> int:
    for key in ("keypoints_roi", "keypoints_img", "keypoints_norm"):
        if key in refined:
            return int(refined[key].shape[0])
    for key in ("refined_success", "source_success", "failure_indices"):
        if key in refined:
            return int(refined[key].shape[0])
    return 0


def _as_positive_int(value: object) -> Optional[int]:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _resolve_full_frame_dimensions(root: zarr.Group) -> tuple[int, int]:
    raw = root.get("raw_video")
    if raw is not None and "images_full" in raw:
        full = raw["images_full"]
        return int(full.shape[1]), int(full.shape[2])

    width = (
        _as_positive_int(root.attrs.get("width"))
        or _as_positive_int(root.attrs.get("video_width"))
        or _as_positive_int(root.attrs.get("palette_video_width"))
        or _as_positive_int(root.attrs.get("source_full_width"))
        or _as_positive_int(root.attrs.get("source_video_width"))
    )
    height = (
        _as_positive_int(root.attrs.get("height"))
        or _as_positive_int(root.attrs.get("video_height"))
        or _as_positive_int(root.attrs.get("palette_video_height"))
        or _as_positive_int(root.attrs.get("source_full_height"))
        or _as_positive_int(root.attrs.get("source_video_height"))
    )

    if raw is not None:
        original_resolution = raw.attrs.get("original_resolution")
        if isinstance(original_resolution, (list, tuple, np.ndarray)) and len(original_resolution) == 2:
            res_h = _as_positive_int(original_resolution[0])
            res_w = _as_positive_int(original_resolution[1])
            if height is None and res_h is not None:
                height = res_h
            if width is None and res_w is not None:
                width = res_w

        if "images_ds" in raw and (height is None or width is None):
            ds = raw["images_ds"]
            if height is None:
                height = _as_positive_int(ds.shape[1])
            if width is None:
                width = _as_positive_int(ds.shape[2])

    return int(height or 640), int(width or 640)


def _load_failure_indices(refined: zarr.Group, include_all: bool = False) -> np.ndarray:
    if include_all:
        total = _total_keypoints(refined)
        if total <= 0:
            return np.zeros(0, dtype="i4")
        return np.arange(total, dtype="i4")
    failures = _load_raw_failure_indices(refined)
    if failures.size == 0:
        return failures
    reason_vals = read_reason_labels(refined)
    if reason_vals is None:
        return failures
    reason_vals = np.asarray(reason_vals, dtype=object)
    if reason_vals.size == 0:
        return failures
    keep_mask = []
    for idx in failures:
        try:
            text = str(reason_vals[int(idx)]) if reason_vals[int(idx)] is not None else ""
        except Exception:
            text = ""
        keep_mask.append(
            "fish_present_no_keypoints" not in text
            and "detection_issue" not in text
        )
    return failures[np.array(keep_mask, dtype=bool)]


def _load_raw_failure_indices(refined: zarr.Group) -> np.ndarray:
    if "refined_success" in refined:
        success = np.asarray(refined["refined_success"][:], dtype=bool)
        return np.where(~success)[0].astype("i4", copy=False)
    elif "source_success" in refined:
        success = np.asarray(refined["source_success"][:], dtype=bool)
        return np.where(~success)[0].astype("i4", copy=False)
    elif "failure_indices" in refined:
        return np.asarray(refined["failure_indices"][:], dtype="i4")
    else:
        return np.zeros(0, dtype="i4")


def _empty_review_auto_state(refined: zarr.Group, raw_failures: np.ndarray) -> Tuple[bool, Optional[str]]:
    if raw_failures.size == 0:
        return True, None

    reason_vals = read_reason_labels(refined)
    if reason_vals is None:
        return False, "missing_reason_labels"
    reason_vals = np.asarray(reason_vals, dtype=object)
    if reason_vals.size == 0:
        return False, "missing_reason_labels"

    has_detection_issue = False
    has_other_failure = False
    for idx in raw_failures:
        try:
            raw_text = reason_vals[int(idx)]
        except Exception:
            has_other_failure = True
            continue
        text = str(raw_text) if raw_text is not None else ""
        tags = {token.strip() for token in text.split("|") if token.strip()}
        if "detection_issue" in tags:
            has_detection_issue = True
            continue
        if "fish_present_no_keypoints" in tags and tags.issubset({"fish_present_no_keypoints"}):
            continue
        has_other_failure = True

    if has_detection_issue:
        return False, "detection_issue_present"
    if has_other_failure:
        return False, "non_reviewable_failures_present"
    return True, None


def _failure_reason_tag_counts(refined: zarr.Group, raw_failures: np.ndarray) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if raw_failures.size == 0:
        return counts
    reason_vals = read_reason_labels(refined)
    if reason_vals is None:
        return counts
    reason_vals = np.asarray(reason_vals, dtype=object)
    if reason_vals.size == 0:
        return counts
    for idx in raw_failures:
        try:
            raw_text = reason_vals[int(idx)]
        except Exception:
            continue
        text = str(raw_text) if raw_text is not None else ""
        for token in text.split("|"):
            tag = token.strip()
            if not tag:
                continue
            counts[tag] = int(counts.get(tag, 0) + 1)
    return counts


def _build_no_reviewable_failures_auto_review(
    *,
    refined: zarr.Group,
    raw_failures: np.ndarray,
    state: str,
) -> Dict[str, object]:
    now_utc = datetime.now(timezone.utc).isoformat()
    reason_counts = _failure_reason_tag_counts(refined, raw_failures)
    blocked_tags = {
        tag: int(reason_counts.get(tag, 0))
        for tag in _NO_REVIEWABLE_FAILURES_BLOCKED_TAGS
    }
    blocked_total = int(sum(blocked_tags.values()))
    allowed = set(_NO_REVIEWABLE_FAILURES_ALLOWED_TAGS)
    blocked = set(_NO_REVIEWABLE_FAILURES_BLOCKED_TAGS)
    other_tags = {
        tag: count
        for tag, count in reason_counts.items()
        if tag not in allowed and tag not in blocked
    }
    other_total = int(sum(other_tags.values()))
    return {
        "policy_id": _NO_REVIEWABLE_FAILURES_POLICY_ID,
        "policy_version": int(_NO_REVIEWABLE_FAILURES_POLICY_VERSION),
        "result": state,
        "applied_at_utc": now_utc,
        "thresholds": {
            "allowed_failure_tags": list(_NO_REVIEWABLE_FAILURES_ALLOWED_TAGS),
            "blocked_failure_tags": list(_NO_REVIEWABLE_FAILURES_BLOCKED_TAGS),
            "other_failure_tag_count_max": 0,
        },
        "checks": {
            "blocked_failure_tags": {
                "observed_total": blocked_total,
                "tags": blocked_tags,
                "pass": blocked_total == 0,
            },
            "other_failure_tags": {
                "observed_total": other_total,
                "tags": other_tags,
                "pass": other_total == 0,
            },
        },
        "evidence": {
            "raw_failure_count": int(raw_failures.size),
            "reason_counts": reason_counts,
            "fish_present_no_keypoints_count": int(reason_counts.get("fish_present_no_keypoints", 0)),
            "detection_issue_count": int(reason_counts.get("detection_issue", 0)),
        },
    }


def _clean_reason(existing: str, new_tags: Sequence[str]) -> str:
    tags = [tag for tag in existing.split("|") if tag and tag != "detection_failed"]
    tags.extend(new_tags)
    unique = sorted(set(tags))
    return "|".join(unique) if unique else "manual_correction"


def _values_equal(current: object, new_value: object) -> bool:
    current_arr = np.asarray(current)
    new_arr = np.asarray(new_value)
    if current_arr.shape != new_arr.shape:
        return False
    try:
        return bool(np.array_equal(current_arr, new_arr, equal_nan=True))
    except TypeError:
        # Object arrays can raise with equal_nan; compare normalized string forms.
        current_obj = current_arr.astype(object, copy=False)
        new_obj = new_arr.astype(object, copy=False)
        if current_obj.ndim == 0:
            return str(current_obj.item()) == str(new_obj.item())
        return all(str(lhs) == str(rhs) for lhs, rhs in zip(current_obj.flat, new_obj.flat))


def _set_roi_value_if_changed(arr: Optional[zarr.Array], roi_idx: int, value: object) -> bool:
    if arr is None:
        return False
    current = arr[roi_idx]
    if _values_equal(current, value):
        return False
    arr[roi_idx] = value
    return True


def _mark_edit_applied(edit_applied_arr: Optional[zarr.Array], roi_idx: int) -> bool:
    if edit_applied_arr is None:
        return False
    if bool(np.asarray(edit_applied_arr[roi_idx], dtype=bool)):
        return False
    edit_applied_arr[roi_idx] = True
    return True


def _build_manual_reason(existing: str, *, geom_ok: bool) -> str:
    existing_tags = [tag for tag in existing.split("|") if tag]
    drop_tags = {
        "detection_failed",
        "low_confidence",
        "confidence_missing",
        "fish_present_no_keypoints",
        "detection_issue",
        "manual_correction",
        "geometry_issue",
    }
    kept_tags = [tag for tag in existing_tags if tag not in drop_tags]
    tags = kept_tags + ["manual_correction"]
    if not geom_ok:
        tags.append("geometry_issue")
    unique: list[str] = []
    seen = set()
    for tag in tags:
        if not tag or tag in seen:
            continue
        unique.append(tag)
        seen.add(tag)
    return "|".join(unique) if unique else "manual_correction"


def _build_cleared_failure_reason(existing: str) -> str:
    existing_tags = [tag for tag in existing.split("|") if tag]
    drop_tags = {
        "detection_failed",
        "fish_present_no_keypoints",
        "detection_issue",
    }
    kept_tags = [tag for tag in existing_tags if tag not in drop_tags]
    if "manual_correction" not in kept_tags:
        kept_tags.append("manual_correction")
    unique: list[str] = []
    seen = set()
    for tag in kept_tags:
        if not tag or tag in seen:
            continue
        unique.append(tag)
        seen.add(tag)
    return "|".join(unique) if unique else "manual_correction"


def _sanitize_reason_array(reason_arr: MutableReasonColumn | zarr.Array) -> None:
    try:
        raw = reason_arr[:]
    except Exception:
        return
    if raw.size == 0:
        return

    def coerce(val: object) -> str:
        if val is None:
            return ""
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            if val.size == 1:
                return str(val.item())
            return "|".join(str(item) for item in val.tolist())
        return str(val)

    cleaned = np.array([coerce(v) for v in raw], dtype=object)
    reason_arr[:] = cleaned


def _write_reason_labels(refined: zarr.Group, labels: np.ndarray) -> None:
    heading = refined.get("heading")
    if heading is not None and heading.chunks:
        chunk_size = int(heading.chunks[0])
    else:
        chunk_size = max(1, min(1024, int(np.asarray(labels).shape[0])))
    write_reason_columns(
        refined,
        np.asarray(labels, dtype=object),
        chunk_size,
        overwrite=True,
    )


def _roi_diagonal_from_roi_images(roi_images: np.ndarray) -> Optional[float]:
    shape = np.shape(roi_images)
    if len(shape) < 3:
        return None
    try:
        roi_h = float(shape[1])
        roi_w = float(shape[2])
    except (TypeError, ValueError):
        return None
    diagonal = float(np.hypot(roi_w, roi_h))
    return diagonal if np.isfinite(diagonal) and diagonal > 0 else None


def _ensure_review_derived_metric_storage(
    refined: zarr.Group,
    *,
    row_count: int,
    chunk_len: int,
    roi_diagonal: Optional[float],
) -> DerivedMetricStorage | None:
    schema = resolve_metric_schema_for_group(refined, required=False)
    if schema is None:
        return None
    return ensure_derived_metric_storage(
        refined,
        schema=schema,
        row_count=row_count,
        chunk_len=chunk_len,
        roi_diagonal=roi_diagonal,
        overwrite=False,
    )


def _set_review_derived_metric_row(
    storage: Optional[DerivedMetricStorage],
    *,
    roi_idx: int,
    keypoints_roi: Optional[np.ndarray],
    keypoint_labels: Sequence[str],
    roi_diagonal: Optional[float],
) -> bool:
    if storage is None:
        return False

    metric_count = int(len(storage.schema.metrics))
    if metric_count <= 0:
        return False

    if keypoints_roi is None:
        values = np.full(metric_count, np.nan, dtype=np.float32)
        valid = np.zeros(metric_count, dtype=bool)
        changed = _set_roi_value_if_changed(storage.values, roi_idx, values)
        changed |= _set_roi_value_if_changed(storage.values_norm, roi_idx, values)
        changed |= _set_roi_value_if_changed(storage.valid, roi_idx, valid)
        return changed

    result = compute_derived_metric_results(
        np.asarray(keypoints_roi, dtype=np.float64),
        keypoint_labels=keypoint_labels,
        schema=storage.schema,
        roi_diagonal=roi_diagonal,
    )
    changed = _set_roi_value_if_changed(storage.values, roi_idx, result.values)
    changed |= _set_roi_value_if_changed(storage.values_norm, roi_idx, result.values_norm)
    changed |= _set_roi_value_if_changed(storage.valid, roi_idx, result.valid)
    return changed


def _load_frame_flags(path: Path) -> Dict[str, list[Dict[str, Optional[int]]]]:
    return _load_shared_frame_flags(path)  # type: ignore[return-value]


def _append_flagged_frame(
    flag_path: Path,
    zarr_path: str,
    frame_idx: int,
    roi_idx: Optional[int],
    *,
    extra_fields: Optional[dict[str, object]] = None,
) -> None:
    _append_shared_flagged_frame(
        flag_path,
        zarr_path,
        frame_idx,
        roi_idx,
        extra_fields=extra_fields,
    )


def _load_detection_frame_flags(path: Path) -> Dict[str, list[int]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    parsed: Dict[str, list[int]] = {}
    for key, value in data.items():
        frames: list[int] = []
        if isinstance(value, list):
            for item in value:
                try:
                    frames.append(int(item))
                except (TypeError, ValueError):
                    continue
        parsed[str(key)] = sorted(set(frames))
    return parsed


def _append_detection_frame(flag_path: Path, zarr_path: str, frame_idx: int) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_detection_frame_flags(flag_path)
    frames = set(data.get(zarr_path, []))
    frames.add(int(frame_idx))
    data[zarr_path] = sorted(frames)
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _append_flagged_path(flag_path: Path, zarr_path: str) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if flag_path.exists():
        with flag_path.open("r", encoding="utf-8") as handle:
            existing = {line.strip() for line in handle if line.strip()}
    if zarr_path in existing:
        return
    with flag_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{zarr_path}\n")


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_keypoint_signature(attrs: Dict[str, Any]) -> Dict[str, object]:
    params = attrs.get("parameters")
    if not isinstance(params, dict):
        provenance = attrs.get("provenance")
        if isinstance(provenance, dict):
            params = provenance.get("parameters")
    if not isinstance(params, dict):
        params = None

    parameter_source = attrs.get("parameter_source")
    if parameter_source is None and isinstance(params, dict):
        parameter_source = params.get("parameter_source")

    return {
        "signature_version": 2,
        "source_keypoints_run": attrs.get("source_keypoints_run"),
        "source_crop_run": attrs.get("source_crop_run"),
        "source_crop_storage_mode": attrs.get("source_crop_storage_mode"),
        "source_crop_signature": attrs.get("source_crop_signature"),
        "source_crop_revision": attrs.get("source_crop_revision"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "source_detect_review_status_ref": attrs.get("source_detect_review_status_ref"),
        "parameter_source": parameter_source,
        "parameters_hash": _hash_parameters(params),
    }


def _jsonable_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_json_mapping(value: object) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return dict(parsed)
    return None


def _resolve_registry_dataset_context(
    zarr_path: Path,
    *,
    registry_path: Optional[Path] = None,
) -> Optional[Dict[str, Optional[str]]]:
    resolved_zarr = zarr_path.expanduser().resolve()
    resolved_registry = (
        registry_path.expanduser().resolve()
        if registry_path is not None
        else RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    )
    if not resolved_registry.exists():
        return None

    registry = Registry(resolved_registry)
    try:
        path_hash = hashlib.sha256(str(resolved_zarr).encode("utf-8")).hexdigest()
        row = registry.conn.execute(
            """
            SELECT dataset_id, recording_id, zarr_use
            FROM datasets
            WHERE path_hash = ?
              AND (status IS NULL OR lower(status) = 'active')
            ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
            LIMIT 1;
            """,
            (path_hash,),
        ).fetchone()
        if row is None:
            row = registry.conn.execute(
                """
                SELECT dataset_id, recording_id, zarr_use
                FROM datasets
                WHERE zarr_path = ?
                  AND (status IS NULL OR lower(status) = 'active')
                ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                LIMIT 1;
                """,
                (str(resolved_zarr),),
            ).fetchone()
        if row is None:
            row = registry.conn.execute(
                """
                SELECT dataset_id, recording_id, zarr_use
                FROM datasets
                WHERE lower(COALESCE(zarr_path, '')) = lower(?)
                  AND (status IS NULL OR lower(status) = 'active')
                ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                LIMIT 1;
                """,
                (str(resolved_zarr),),
            ).fetchone()
        if row is None:
            return None
        dataset_id = _normalize_attr(row["dataset_id"])
        if dataset_id is None:
            return None
        return {
            "registry_path": str(resolved_registry),
            "dataset_id": dataset_id,
            "recording_id": _normalize_attr(row["recording_id"]),
            "zarr_use": _normalize_attr(row["zarr_use"]),
        }
    finally:
        registry.close()


def _refined_coverage_pct(refined: zarr.Group) -> Optional[float]:
    summary_raw = refined.attrs.get("summary_statistics")
    candidates: list[object] = []
    if isinstance(summary_raw, dict):
        postprocess = summary_raw.get("postprocess")
        refine = summary_raw.get("refine")
        if isinstance(postprocess, dict):
            candidates.extend([postprocess.get("pass_rate_percent"), postprocess.get("success_rate_percent")])
        if isinstance(refine, dict):
            candidates.extend([refine.get("pass_rate_percent"), refine.get("success_rate_percent")])
        candidates.extend([summary_raw.get("pass_rate_percent"), summary_raw.get("success_rate_percent")])
    candidates.append(refined.attrs.get("success_rate"))

    for raw in candidates:
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value <= 1.0:
            value *= 100.0
        return value
    return None


def _build_refined_keypoint_details(refined: zarr.Group) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "reason": "present",
        "latest_selector": "keypoint_failure_review_manual",
    }
    for key in ("source_keypoints_run", "source_crop_run", "source_detect_run"):
        value = refined.attrs.get(key)
        if value is not None:
            details[key] = _jsonable_scalar(value)

    summary_raw = refined.attrs.get("summary_statistics")
    if isinstance(summary_raw, dict):
        for key in ("total_rois", "refined_success", "remaining_failures", "usable_keypoints", "pass_rate_percent"):
            if key in summary_raw and summary_raw.get(key) is not None:
                details[key] = _jsonable_scalar(summary_raw.get(key))
        postprocess = summary_raw.get("postprocess")
        if isinstance(postprocess, dict):
            for key in ("total_rois", "refined_success", "remaining_failures", "usable_keypoints", "pass_rate_percent"):
                if postprocess.get(key) is not None:
                    details[key] = _jsonable_scalar(postprocess.get(key))
    return details


def _sync_registry_after_review_status(
    *,
    root: Optional[zarr.Group] = None,
    zarr_path: str,
    refined_run: str,
    refined: zarr.Group,
    review_status: Dict[str, object],
    registry_path: Optional[Path] = None,
) -> Dict[str, object]:
    context = _resolve_registry_dataset_context(Path(zarr_path), registry_path=registry_path)
    if context is None:
        return {"synced": False, "reason": "status_context_unavailable"}

    resolved_registry = Path(str(context["registry_path"]))
    resolved_zarr = Path(zarr_path).expanduser().resolve()
    registry = Registry(resolved_registry)
    try:
        completion_root = root if root is not None else open_zarr_root(resolved_zarr, mode="r")
        dataset_id = str(context["dataset_id"])
        row = registry.conn.execute(
            """
            SELECT details_json
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = 'refined_keypoints'
            LIMIT 1;
            """,
            (dataset_id,),
        ).fetchone()
        existing_details = _parse_json_mapping(row["details_json"]) if row is not None else None
        merged_details: Dict[str, Any] = {}
        if isinstance(existing_details, dict):
            merged_details.update(existing_details)
        merged_details.update(_build_refined_keypoint_details(refined))

        method = (
            _normalize_attr(refined.attrs.get("method"))
            or _normalize_attr(refined.attrs.get("refinement_role"))
            or "refine_keypoints"
        )
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            session_uuid=None,
            recording_id=_normalize_attr(context.get("recording_id")),
            zarr_use=_normalize_attr(context.get("zarr_use")),
            zarr_purpose=None,
            source_layout=None,
            source_frame_index_path=None,
            source_recording_frame_index_path=None,
            source_frame_index_schema=None,
        )
        wrote = emit_stage_completion(
            completion_root,
            resolved_zarr,
            step_name="refined_keypoints",
            status="ok",
            source="keypoint_failure_review_manual",
            run_name=refined_run,
            method=method,
            coverage_pct=_refined_coverage_pct(refined),
            review_status_json=review_status,
            details_json=merged_details,
            console=None,
            registry=registry,
            auto_registry_from_env=False,
            invalidate_on_ok=False,
            metadata=metadata,
            upsert_dataset_row=False,
        )
        if not wrote:
            return {"synced": False, "reason": "sync_failed", "error": "emit_stage_completion returned false"}
        perf_rows = int(
            registry.refresh_keypoint_performance_for_dataset(
                dataset_id,
                zarr_path=resolved_zarr,
                recording_id=context.get("recording_id"),
                zarr_use=context.get("zarr_use"),
            )
        )
        quality_rows = int(registry.refresh_keypoint_quality_for_dataset(dataset_id, zarr_path=resolved_zarr))
        return {
            "synced": True,
            "dataset_id": dataset_id,
            "recording_id": context.get("recording_id"),
            "keypoint_performance_rows": perf_rows,
            "keypoint_quality_rows": quality_rows,
        }
    except Exception as exc:
        return {"synced": False, "reason": "sync_failed", "error": str(exc)}
    finally:
        registry.close()


def _apply_review_status(
    refined_parent: zarr.Group,
    refined_run: str,
    refined: zarr.Group,
    *,
    root: Optional[zarr.Group] = None,
    zarr_path: str,
    state: str,
    method: str,
    intended_use: str,
    reviewer: Optional[str],
    notes: Optional[str],
    registry_path: Optional[Path] = None,
    auto_review: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    now_utc = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, object] = {
        "state": state,
        "method": method,
        "intended_use": intended_use,
        "timestamp_utc": now_utc,
        "timestamp": now_utc,
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes
    if isinstance(auto_review, dict):
        payload["auto_review"] = dict(auto_review)

    refined.attrs["keypoint_review_status"] = payload

    signature = refined.attrs.get("keypoint_signature")
    if not isinstance(signature, dict):
        signature = _build_keypoint_signature(dict(refined.attrs))
        refined.attrs["keypoint_signature"] = signature
    refined.attrs["keypoint_review_signature"] = signature

    refined_parent.attrs["keypoint_review_status_latest"] = refined_run
    sync_result = _sync_registry_after_review_status(
        root=root,
        zarr_path=zarr_path,
        refined_run=refined_run,
        refined=refined,
        review_status=payload,
        registry_path=registry_path,
    )
    return payload, sync_result


def _normalize_optional_text(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def _infer_zarr_use(root: zarr.Group, zarr_path: str) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        value = _normalize_optional_text(root.attrs.get(key))
        if value in {"analysis", "training"}:
            return value
    name = Path(zarr_path).name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _resolve_review_intended_use(
    *,
    requested: Optional[str],
    refined: zarr.Group,
    root: zarr.Group,
    zarr_path: str,
) -> str:
    normalized_requested = _normalize_optional_text(requested)
    if normalized_requested in {"training", "full_recording"}:
        return str(normalized_requested)

    existing_status = refined.attrs.get("keypoint_review_status")
    existing_use = None
    if isinstance(existing_status, dict):
        existing_use = _normalize_optional_text(existing_status.get("intended_use"))
    if existing_use in {"training", "full_recording"}:
        return str(existing_use)

    observed_use = _infer_zarr_use(root, zarr_path)
    if observed_use == "training":
        return "training"
    return "full_recording"


def launch_review(
    zarr_path: str,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    include_all: bool = False,
    target_frames: Optional[Sequence[int]] = None,
    target_roi_indices: Optional[Sequence[int]] = None,
    review_state: str = "approved",
    review_method: str = "manual",
    review_intended_use: Optional[str] = None,
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
    frame_flag_file: Optional[str] = None,
    detect_flag_file: Optional[str] = None,
    detect_frame_flag_file: Optional[str] = None,
) -> None:
    root = open_zarr_root(zarr_path, mode="a")

    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")

    using_latest = refined_run is None
    if refined_run is None:
        try:
            refined_run = resolve_latest_complete_run_name(refined_parent)
        except Exception:
            refined_run = None
        if refined_run is None:
            refined_run = _normalize_attr(refined_parent.attrs.get("latest"))
    if not refined_run:
        raise RuntimeError("Refined keypoint run not found.")
    try:
        refined = root[f"refined_keypoints_runs/{refined_run}"]
    except KeyError as exc:
        raise RuntimeError("Refined keypoint run not found.") from exc
    resolved_review_intended_use = _resolve_review_intended_use(
        requested=review_intended_use,
        refined=refined,
        root=root,
        zarr_path=zarr_path,
    )

    flag_path = Path(frame_flag_file).expanduser() if frame_flag_file else None
    detect_flag_path = Path(detect_flag_file).expanduser() if detect_flag_file else None
    detect_frame_flag_path = (
        Path(detect_frame_flag_file).expanduser() if detect_frame_flag_file else None
    )
    crop_run = crop_run or refined.attrs.get("source_crop_run") or _get_latest_run(root, "crop")
    crop_group = root[f"crop_runs/{crop_run}"]
    roi_images = crop_group["roi_images"]
    roi_coords = crop_group["roi_coordinates_full"]
    frame_indices = crop_group["frame_indices"][:]
    source_refined_row_ids, source_detect_row_index = load_row_identity_arrays(
        crop_group,
        total_rois=int(frame_indices.shape[0]),
    )

    full_h, full_w = _resolve_full_frame_dimensions(root)
    norm_factor = np.array([full_w, full_h], dtype=np.float64)

    failures = _load_failure_indices(refined, include_all=include_all)
    targeted = False
    if target_frames or target_roi_indices:
        selected: set[int] = set()
        if target_frames:
            target_frames_arr = np.array(sorted(set(int(f) for f in target_frames)), dtype=np.int64)
            frame_hits = np.where(np.isin(frame_indices, target_frames_arr))[0].astype("i4", copy=False)
            selected.update(int(v) for v in frame_hits.tolist())
        if target_roi_indices:
            total_rows = int(frame_indices.shape[0])
            for roi_idx in sorted(set(int(v) for v in target_roi_indices)):
                if 0 <= roi_idx < total_rows:
                    selected.add(int(roi_idx))
        targeted = True
        failures = np.array(sorted(selected), dtype="i4")
    if failures.size == 0:
        raw_failures = _load_raw_failure_indices(refined)
        if targeted:
            print("No matching keypoints found for requested targets.")
            return
        if include_all:
            print("No keypoints found to review.")
            return
        else:
            can_auto_apply, blocked_reason = _empty_review_auto_state(refined, raw_failures)
            if can_auto_apply:
                auto_review_payload = _build_no_reviewable_failures_auto_review(
                    refined=refined,
                    raw_failures=raw_failures,
                    state=review_state,
                )
                auto_reviewer = f"auto:{_NO_REVIEWABLE_FAILURES_POLICY_ID}"
                auto_notes = review_notes or (
                    "Auto-approved: no reviewable failures "
                    "(fish_present_no_keypoints-only or none)."
                )
                payload, sync = _apply_review_status(
                    refined_parent,
                    refined_run,
                    refined,
                    root=root,
                    zarr_path=zarr_path,
                    state=review_state,
                    method="algorithmic",
                    intended_use=resolved_review_intended_use,
                    reviewer=auto_reviewer,
                    notes=auto_notes,
                    auto_review=auto_review_payload,
                )
                if raw_failures.size > 0:
                    print(
                        "No reviewable failures found; auto-applied keypoint review status "
                        "for fish_present_no_keypoints-only failures."
                    )
                else:
                    print("No failed keypoints to review; auto-applied keypoint review status.")
                print(
                    f"✓ Keypoint review set: {payload.get('state')} "
                    f"({payload.get('method')}/{payload.get('intended_use')})"
                )
                if not bool(sync.get("synced")):
                    reason = str(sync.get("reason") or "unknown")
                    if reason != "status_context_unavailable":
                        detail = str(sync.get("error") or "")
                        suffix = f" ({detail})" if detail else ""
                        print(f"Warning: registry sync skipped: {reason}{suffix}")
            else:
                print("No failed keypoints to review.")
                print(
                    "Review status not auto-applied because remaining failures are not "
                    f"fish_present_no_keypoints-only ({blocked_reason})."
                )
                return
            return
    if flag_path is not None:
        print(f"Frame flag file: {flag_path.expanduser().resolve(strict=False)}")
    if detect_flag_path is not None:
        print(f"Detection flag file: {detect_flag_path.expanduser().resolve(strict=False)}")
    if detect_frame_flag_path is not None:
        print(f"Detection frame flag file: {detect_frame_flag_path.expanduser().resolve(strict=False)}")

    summary_raw = refined.attrs.get("summary_statistics", {})
    summary = summary_raw.get("refine", summary_raw) if isinstance(summary_raw, dict) else {}
    default_min_triangle_angle, default_min_triangle_area, default_max_triangle_area = (
        _resolve_review_geometry_defaults(refined.attrs)
    )
    confidence_threshold = float(
        summary.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD)
    )
    min_triangle_angle = float(
        summary.get("min_triangle_angle", default_min_triangle_angle)
    )
    min_triangle_area = float(
        summary.get("min_triangle_area", default_min_triangle_area)
    )
    max_tri_val = summary.get("max_triangle_area")
    try:
        max_triangle_area = (
            float(max_tri_val)
            if max_tri_val is not None
            else default_max_triangle_area
        )
    except (TypeError, ValueError):
        max_triangle_area = default_max_triangle_area

    if "keypoints_roi" not in refined:
        raise RuntimeError("Refined run is missing keypoints_roi.")
    kp_roi_arr = refined["keypoints_roi"]
    kp_img_arr = refined.get("keypoints_img")
    kp_norm_arr = refined.get("keypoints_norm")
    heading_arr = refined.get("heading")
    confidence_arr = refined.get("confidence")
    conf_arr = refined.get("keypoint_confidences")
    triangle_area_arr = refined.get("triangle_area")
    min_angle_arr = refined.get("min_angle")
    triangle_angles_arr = refined.get("triangle_angles")
    refined_success_arr = refined.get("refined_success")
    flip_corrected_arr = refined.get("flip_corrected")
    quality_labels_arr = refined.get("quality_labels")
    confidence_valid_arr = refined.get("confidence_valid")
    geometry_valid_arr = refined.get("geometry_valid")
    usable_arr = refined.get("usable_keypoints")
    edit_applied_arr = refined.get("edit_applied")
    reason_chunk = (
        int(heading_arr.chunks[0])
        if heading_arr is not None and heading_arr.chunks
        else max(1, min(1024, int(kp_roi_arr.shape[0]) or 1))
    )
    reason_arr = open_mutable_reason_column(refined, chunk_size=reason_chunk)
    heading_finite_arr = refined.get("heading_finite")
    heading_usable_arr = refined.get("heading_usable")
    detection_source_arr = refined.get("detection_source")
    if reason_arr is not None:
        _sanitize_reason_array(reason_arr)

    keypoint_count = (
        int(kp_roi_arr.shape[1])
        if len(kp_roi_arr.shape) >= 2
        else len(_DEFAULT_LABELS)
    )
    labels = list(refined.attrs.get("keypoint_labels", _DEFAULT_LABELS))
    if len(labels) != keypoint_count:
        raise ValueError(
            "Refined keypoint run keypoint_labels count does not match keypoints_roi "
            f"K ({len(labels)} vs {keypoint_count})."
        )
    head_triangle_indices = resolve_head_triangle_for_labels(
        labels,
        keypoint_count=keypoint_count,
        allow_legacy_3point_fallback=True,
    )
    colors = _display_colors(len(labels))
    roi_diagonal = _roi_diagonal_from_roi_images(roi_images)
    derived_metric_storage: Optional[DerivedMetricStorage] = None

    def _get_derived_metric_storage() -> Optional[DerivedMetricStorage]:
        nonlocal derived_metric_storage
        if derived_metric_storage is not None:
            return derived_metric_storage
        chunk_len = (
            int(heading_arr.chunks[0])
            if heading_arr is not None and heading_arr.chunks
            else max(1, min(1024, int(kp_roi_arr.shape[0])))
        )
        derived_metric_storage = _ensure_review_derived_metric_storage(
            refined,
            row_count=int(kp_roi_arr.shape[0]),
            chunk_len=chunk_len,
            roi_diagonal=roi_diagonal,
        )
        return derived_metric_storage

    idx_pos = 0
    active_idx = 0
    points = np.full((len(labels), 2), np.nan, dtype=np.float64)
    show_text = True

    mpl.rcParams["keymap.save"] = []
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)

    def load_current_points() -> None:
        nonlocal points
        roi_idx = int(failures[idx_pos])
        existing = np.asarray(kp_roi_arr[roi_idx], dtype=np.float64)
        if existing.shape == points.shape and np.isfinite(existing).any():
            points = existing.copy()
        else:
            points = np.full_like(points, np.nan)

    def update_display() -> None:
        roi_idx = int(failures[idx_pos])
        roi_img = roi_images[roi_idx]
        frame_idx = int(frame_indices[roi_idx])

        ax.clear()
        ax.imshow(roi_img, cmap="gray")

        for i, (label, color) in enumerate(zip(labels, colors)):
            if np.isfinite(points[i]).all():
                ax.scatter(points[i, 0], points[i, 1], s=60, c=color, edgecolors="black", linewidths=1.0)
                if show_text:
                    ax.text(points[i, 0] + 3, points[i, 1] - 3, label, color=color, fontsize=8, weight="bold")

        if show_text:
            active_label = labels[active_idx] if active_idx < len(labels) else "unknown"
            flag_label = ""
            if reason_arr is not None:
                try:
                    raw_reason = reason_arr[roi_idx]
                except Exception:
                    raw_reason = ""
                reason_text = str(raw_reason) if raw_reason is not None else ""
                if reason_text:
                    tags = [tag.strip() for tag in reason_text.split("|") if tag.strip()]
                    flagged = [tag for tag in tags if tag in {"fish_present_no_keypoints", "detection_issue"}]
                    if flagged:
                        flag_label = "Flagged: " + ", ".join(flagged)
            ax.set_title(
                f"ROI {roi_idx} | Frame {frame_idx} | {idx_pos + 1}/{len(failures)} "
                f"| Active: {active_label}",
                fontsize=10,
            )
            if flag_label:
                ax.text(
                    0.02,
                    0.02,
                    flag_label,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#f97316",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2),
                )
        else:
            ax.set_title("")
        ax.set_axis_off()
        fig.canvas.draw_idle()

    def save_current() -> None:
        nonlocal active_idx, idx_pos
        roi_idx = int(failures[idx_pos])
        frame_idx = int(frame_indices[roi_idx])
        if not np.isfinite(points).all():
            missing_labels = [str(label) for label, xy in zip(labels, points) if not np.isfinite(xy).all()]
            detail = f" Missing: {', '.join(missing_labels)}." if missing_labels else ""
            print(f"Set all keypoints before saving.{detail}")
            return

        changed = False
        changed |= _set_roi_value_if_changed(kp_roi_arr, roi_idx, points)
        full_points = points + roi_coords[roi_idx]
        changed |= _set_roi_value_if_changed(kp_img_arr, roi_idx, full_points)
        changed |= _set_roi_value_if_changed(kp_norm_arr, roi_idx, full_points / norm_factor)

        heading_val = compute_heading_from_attrs(
            refined.attrs,
            labels=labels,
            points=points,
        )
        changed |= _set_roi_value_if_changed(heading_arr, roi_idx, heading_val)

        metrics = compute_geometry_metrics(
            select_head_triangle_points(points, head_triangle_indices)
        )
        max_ok = max_triangle_area is None or metrics.area <= max_triangle_area
        geom_ok = bool(
            np.isfinite(metrics.min_angle)
            and np.isfinite(metrics.area)
            and metrics.min_angle >= min_triangle_angle
            and metrics.area >= min_triangle_area
            and max_ok
        )

        changed |= _set_roi_value_if_changed(triangle_area_arr, roi_idx, metrics.area)
        changed |= _set_roi_value_if_changed(min_angle_arr, roi_idx, metrics.min_angle)
        changed |= _set_roi_value_if_changed(triangle_angles_arr, roi_idx, metrics.angles)
        changed |= _set_review_derived_metric_row(
            _get_derived_metric_storage(),
            roi_idx=roi_idx,
            keypoints_roi=points,
            keypoint_labels=labels,
            roi_diagonal=roi_diagonal,
        )

        conf_ok = True
        if conf_arr is not None:
            conf_vals = np.ones(len(labels), dtype=np.float64)
            changed |= _set_roi_value_if_changed(conf_arr, roi_idx, conf_vals)
            conf_ok = bool(np.all(conf_vals >= confidence_threshold))

        changed |= _set_roi_value_if_changed(confidence_arr, roi_idx, 1.0)

        refined_success_val = True
        changed |= _set_roi_value_if_changed(refined_success_arr, roi_idx, refined_success_val)
        changed |= _set_roi_value_if_changed(flip_corrected_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(quality_labels_arr, roi_idx, 0)
        changed |= _set_roi_value_if_changed(confidence_valid_arr, roi_idx, conf_ok)
        changed |= _set_roi_value_if_changed(geometry_valid_arr, roi_idx, geom_ok)
        changed |= _set_roi_value_if_changed(usable_arr, roi_idx, conf_ok and geom_ok)
        heading_is_finite = bool(np.isfinite(heading_val))
        changed |= _set_roi_value_if_changed(heading_finite_arr, roi_idx, heading_is_finite)
        if heading_usable_arr is not None:
            det_src = int(detection_source_arr[roi_idx]) if detection_source_arr is not None else 0
            changed |= _set_roi_value_if_changed(
                heading_usable_arr,
                roi_idx,
                refined_success_val and det_src == 0 and heading_is_finite,
            )
        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            reason_value = _build_manual_reason(existing, geom_ok=geom_ok)
            if existing != reason_value:
                reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)
                changed = True

        if changed:
            _mark_edit_applied(edit_applied_arr, roi_idx)
            stale_touched = mark_downstream_subject_mask_runs_stale(
                root,
                source_keypoint_group="refined_keypoints_runs",
                source_keypoints_run=str(refined_run),
                roi_indices=[roi_idx],
                frame_indices=[frame_idx],
                reason="keypoint_manual_correction",
            )
            print(f"Saved manual correction for ROI {roi_idx}.")
            if stale_touched:
                print(f"Marked {stale_touched} downstream refined-subject mask run(s) stale.")
        else:
            print(f"No changes for ROI {roi_idx}; skipped stale marker update.")

        active_idx = 0
        if idx_pos < len(failures) - 1:
            idx_pos += 1
            load_current_points()
            update_display()

    def mark_no_keypoints() -> None:
        nonlocal active_idx, idx_pos, failures
        roi_idx = int(failures[idx_pos])
        frame_idx = int(frame_indices[roi_idx])

        changed = False
        if kp_roi_arr is not None:
            changed |= _set_roi_value_if_changed(kp_roi_arr, roi_idx, np.full_like(np.asarray(kp_roi_arr[roi_idx]), np.nan))
        if kp_img_arr is not None:
            changed |= _set_roi_value_if_changed(kp_img_arr, roi_idx, np.full_like(np.asarray(kp_img_arr[roi_idx]), np.nan))
        if kp_norm_arr is not None:
            changed |= _set_roi_value_if_changed(kp_norm_arr, roi_idx, np.full_like(np.asarray(kp_norm_arr[roi_idx]), np.nan))
        changed |= _set_roi_value_if_changed(heading_arr, roi_idx, np.nan)
        changed |= _set_roi_value_if_changed(confidence_arr, roi_idx, np.nan)
        if conf_arr is not None:
            changed |= _set_roi_value_if_changed(conf_arr, roi_idx, np.full_like(np.asarray(conf_arr[roi_idx]), np.nan))
        changed |= _set_roi_value_if_changed(triangle_area_arr, roi_idx, np.nan)
        changed |= _set_roi_value_if_changed(min_angle_arr, roi_idx, np.nan)
        if triangle_angles_arr is not None:
            changed |= _set_roi_value_if_changed(
                triangle_angles_arr,
                roi_idx,
                np.full_like(np.asarray(triangle_angles_arr[roi_idx]), np.nan),
            )
        changed |= _set_review_derived_metric_row(
            _get_derived_metric_storage(),
            roi_idx=roi_idx,
            keypoints_roi=None,
            keypoint_labels=labels,
            roi_diagonal=roi_diagonal,
        )
        changed |= _set_roi_value_if_changed(refined_success_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(flip_corrected_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(quality_labels_arr, roi_idx, 0)
        changed |= _set_roi_value_if_changed(confidence_valid_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(geometry_valid_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(usable_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(heading_finite_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(heading_usable_arr, roi_idx, False)
        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            reason_value = str(_clean_reason(existing, ["fish_present_no_keypoints"]))
            if existing != reason_value:
                reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)
                changed = True

        if changed:
            _mark_edit_applied(edit_applied_arr, roi_idx)
            stale_touched = mark_downstream_subject_mask_runs_stale(
                root,
                source_keypoint_group="refined_keypoints_runs",
                source_keypoints_run=str(refined_run),
                roi_indices=[roi_idx],
                frame_indices=[frame_idx],
                reason="keypoint_mark_no_keypoints",
            )
            print(f"Marked fish-present/no-keypoints for ROI {roi_idx} (frame {frame_idx}).")
            if stale_touched:
                print(f"Marked {stale_touched} downstream refined-subject mask run(s) stale.")
        else:
            print(f"No changes for ROI {roi_idx} (frame {frame_idx}); skipped stale marker update.")

        failures = np.delete(failures, idx_pos)
        if failures.size == 0:
            plt.close(fig)
            return
        idx_pos = min(idx_pos, len(failures) - 1)
        active_idx = 0
        load_current_points()
        update_display()

    def mark_detection_issue() -> None:
        nonlocal active_idx, idx_pos, failures
        roi_idx = int(failures[idx_pos])
        frame_idx = int(frame_indices[roi_idx])

        if detect_frame_flag_path is not None:
            try:
                _append_detection_frame(detect_frame_flag_path, zarr_path, frame_idx)
            except Exception as exc:
                print(f"Failed to flag detection frame: {exc}")
        if detect_flag_path is not None:
            try:
                _append_flagged_path(detect_flag_path, zarr_path)
            except Exception as exc:
                print(f"Failed to flag detection path: {exc}")

        changed = False
        if kp_roi_arr is not None:
            changed |= _set_roi_value_if_changed(kp_roi_arr, roi_idx, np.full_like(np.asarray(kp_roi_arr[roi_idx]), np.nan))
        if kp_img_arr is not None:
            changed |= _set_roi_value_if_changed(kp_img_arr, roi_idx, np.full_like(np.asarray(kp_img_arr[roi_idx]), np.nan))
        if kp_norm_arr is not None:
            changed |= _set_roi_value_if_changed(kp_norm_arr, roi_idx, np.full_like(np.asarray(kp_norm_arr[roi_idx]), np.nan))
        changed |= _set_roi_value_if_changed(heading_arr, roi_idx, np.nan)
        changed |= _set_roi_value_if_changed(confidence_arr, roi_idx, np.nan)
        if conf_arr is not None:
            changed |= _set_roi_value_if_changed(conf_arr, roi_idx, np.full_like(np.asarray(conf_arr[roi_idx]), np.nan))
        changed |= _set_roi_value_if_changed(triangle_area_arr, roi_idx, np.nan)
        changed |= _set_roi_value_if_changed(min_angle_arr, roi_idx, np.nan)
        if triangle_angles_arr is not None:
            changed |= _set_roi_value_if_changed(
                triangle_angles_arr,
                roi_idx,
                np.full_like(np.asarray(triangle_angles_arr[roi_idx]), np.nan),
            )
        changed |= _set_review_derived_metric_row(
            _get_derived_metric_storage(),
            roi_idx=roi_idx,
            keypoints_roi=None,
            keypoint_labels=labels,
            roi_diagonal=roi_diagonal,
        )
        changed |= _set_roi_value_if_changed(refined_success_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(flip_corrected_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(quality_labels_arr, roi_idx, 0)
        changed |= _set_roi_value_if_changed(confidence_valid_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(geometry_valid_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(usable_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(heading_finite_arr, roi_idx, False)
        changed |= _set_roi_value_if_changed(heading_usable_arr, roi_idx, False)
        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            reason_value = str(_clean_reason(existing, ["detection_issue"]))
            if existing != reason_value:
                reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)
                changed = True

        if changed:
            _mark_edit_applied(edit_applied_arr, roi_idx)
            stale_touched = mark_downstream_subject_mask_runs_stale(
                root,
                source_keypoint_group="refined_keypoints_runs",
                source_keypoints_run=str(refined_run),
                roi_indices=[roi_idx],
                frame_indices=[frame_idx],
                reason="keypoint_mark_detection_issue",
            )
            print(f"Marked detection issue for ROI {roi_idx} (frame {frame_idx}).")
            if stale_touched:
                print(f"Marked {stale_touched} downstream refined-subject mask run(s) stale.")
        else:
            print(f"No changes for ROI {roi_idx} (frame {frame_idx}); skipped stale marker update.")

        failures = np.delete(failures, idx_pos)
        if failures.size == 0:
            plt.close(fig)
            return
        idx_pos = min(idx_pos, len(failures) - 1)
        active_idx = 0
        load_current_points()
        update_display()

    def clear_failure_label() -> None:
        roi_idx = int(failures[idx_pos])
        frame_idx = int(frame_indices[roi_idx])
        if reason_arr is None:
            print("Cannot clear failure label: reason labels are unavailable.")
            return

        existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
        tags = {token.strip() for token in existing.split("|") if token.strip()}
        target_tags = {"fish_present_no_keypoints", "detection_issue"}
        present = sorted(tags & target_tags)
        if not present:
            print(f"ROI {roi_idx} has no clearable failure label.")
            return

        reason_value = _build_cleared_failure_reason(existing)
        changed = False
        if existing != reason_value:
            reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)
            changed = True

        if changed:
            _mark_edit_applied(edit_applied_arr, roi_idx)
            stale_touched = mark_downstream_subject_mask_runs_stale(
                root,
                source_keypoint_group="refined_keypoints_runs",
                source_keypoints_run=str(refined_run),
                roi_indices=[roi_idx],
                frame_indices=[frame_idx],
                reason="keypoint_clear_failure_label",
            )
            print(
                f"Cleared failure label for ROI {roi_idx} (frame {frame_idx}): "
                f"{', '.join(present)}."
            )
            if stale_touched:
                print(f"Marked {stale_touched} downstream refined-subject mask run(s) stale.")
        else:
            print(f"No changes for ROI {roi_idx} (frame {frame_idx}); skipped stale marker update.")
        update_display()

    def next_failure() -> None:
        nonlocal idx_pos
        if idx_pos < len(failures) - 1:
            idx_pos += 1
            load_current_points()
            update_display()

    def prev_failure() -> None:
        nonlocal idx_pos
        if idx_pos > 0:
            idx_pos -= 1
            load_current_points()
            update_display()

    def reset_points() -> None:
        load_current_points()
        update_display()

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        points[active_idx] = [event.xdata, event.ydata]
        update_display()

    def on_key(event) -> None:
        nonlocal active_idx, show_text
        def apply_state(state: str) -> None:
            payload, sync = _apply_review_status(
                refined_parent,
                refined_run,
                refined,
                root=root,
                zarr_path=zarr_path,
                state=state,
                method=review_method,
                intended_use=resolved_review_intended_use,
                reviewer=reviewer or os.environ.get("USER"),
                notes=review_notes,
            )
            print(f"✓ Keypoint review set: {payload.get('state')} ({payload.get('method')}/{payload.get('intended_use')})")
            if not bool(sync.get("synced")):
                reason = str(sync.get("reason") or "unknown")
                if reason != "status_context_unavailable":
                    detail = str(sync.get("error") or "")
                    suffix = f" ({detail})" if detail else ""
                    print(f"Warning: registry sync skipped: {reason}{suffix}")

        selected_idx = _active_index_from_key(event.key, label_count=len(labels), current_idx=active_idx)
        if selected_idx is not None:
            active_idx = int(selected_idx)
            update_display()
        elif event.key == "n":
            next_failure()
        elif event.key == "p":
            prev_failure()
        elif event.key == "r":
            reset_points()
        elif event.key == "t":
            show_text = not show_text
            update_display()
        elif event.key == "s":
            save_current()
        elif event.key == "b":
            if flag_path is None:
                print("No frame flag file configured. Pass --frame-flag-file to enable frame flagging.")
            else:
                roi_idx = int(failures[idx_pos])
                frame_idx = int(frame_indices[roi_idx])
                try:
                    _append_flagged_frame(
                        flag_path,
                        zarr_path,
                        frame_idx,
                        roi_idx,
                        extra_fields=row_identity_payload(
                            roi_idx,
                            source_refined_row_ids=source_refined_row_ids,
                            source_detect_row_index=source_detect_row_index,
                        ),
                    )
                    print(f"Flagged frame {frame_idx} (ROI {roi_idx}) for keypoint follow-up.")
                    print(f"Frame flag file: {flag_path}")
                except Exception as exc:
                    print(f"Failed to flag frame: {exc}")
        elif event.key == "d":
            mark_detection_issue()
        elif event.key == "x":
            mark_no_keypoints()
        elif event.key == "c":
            clear_failure_label()
        elif event.key == "a":
            apply_state(review_state)
        elif event.key == "R":
            apply_state("rejected")
        elif event.key == "N":
            apply_state("needs_review")
        elif event.key == "P":
            apply_state("pending")
        elif event.key == "q":
            plt.close(fig)

    print("\nKeypoint Review")
    print(f"  Zarr: {zarr_path}")
    refined_label = f"{refined_run} (latest)" if using_latest else refined_run
    print(f"  Refined run: {refined_label}")
    print(f"  Crop run: {crop_run}")
    if targeted:
        print(f"  ROIs to review (target selection): {len(failures)}")
    elif include_all:
        print(f"  ROIs to review: {len(failures)}")
    else:
        print(f"  Failures to review: {len(failures)}")
    print("\nControls:")
    print("  Click: set active keypoint")
    if len(labels) <= 9:
        label_help = ", ".join(f"{idx + 1}={label}" for idx, label in enumerate(labels))
        print(f"  1-{len(labels)}: select keypoint ({label_help})")
    else:
        print("  1-9: select corresponding keypoint when available")
    print("  [/]: cycle active keypoint backward/forward")
    print("  s: save correction")
    print("  t: toggle text overlays")
    print("  b: flag frame for follow-up (writes --frame-flag-file)")
    print("  d: flag detection issue (writes retune flags)")
    print("  x: mark fish present but no keypoints")
    print("  c: clear fish_present_no_keypoints/detection_issue label")
    print("  r: reset points from current data")
    print("  n/p: next/previous failure")
    print("  a: approve keypoints")
    print("  N: mark needs_review")
    print("  R: mark rejected")
    print("  P: mark pending")
    print("  q: quit")

    load_current_points()
    update_display()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if reason_arr is not None:
        _write_reason_labels(refined, np.asarray(reason_arr[:], dtype=object))
    refresh_refined_keypoint_heading_fields(refined, root=root)


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(
        "The keypoint_failure_review entrypoint has been removed. "
        "Use `python -m fisheye.tune.keypoint_review --manual`."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
