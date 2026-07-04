"""Backend primitives for keypoint manual review web and shared review flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import base64
import numpy as np
import zarr

from ..pose.heading import compute_heading_from_attrs
from ..refinement.keypoint_quality import (
    compute_geometry_metrics,
    resolve_head_triangle_for_labels,
    select_head_triangle_points,
)
from ..shared.detect_reason_codec import read_reason_labels, write_reason_columns
from ..shared.frame_flags import append_flagged_frame, load_row_identity_arrays, row_identity_payload
from ..shared.subject_mask_stale import mark_downstream_subject_mask_runs_stale
from ..shared.zarr_run_completion import resolve_latest_complete_run_name, set_authoritative_run
from ..shared.zarr_io import open_zarr_root
from .keypoint_failure_review import (
    _DEFAULT_CONFIDENCE_THRESHOLD,
    _DEFAULT_LABELS,
    _apply_review_status,
    _build_cleared_failure_reason,
    _build_manual_reason,
    _clean_reason,
    _ensure_review_derived_metric_storage,
    _load_failure_indices,
    _mark_edit_applied,
    _resolve_full_frame_dimensions,
    _resolve_review_intended_use,
    _resolve_review_geometry_defaults,
    _roi_diagonal_from_roi_images,
    _set_review_derived_metric_row,
    _set_roi_value_if_changed,
)
from .keypoint_review import _update_postprocess_summary


def _coerce_status_scalar(value: object) -> object:
    if value is None:
        return None
    try:
        scalar = np.asarray(value).item()
    except Exception:
        return str(value)
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, (bytes, bytearray, memoryview)):
        try:
            return bytes(scalar).decode("utf-8")
        except Exception:
            return str(scalar)
    if isinstance(scalar, np.ndarray):  # pragma: no cover - defensive
        return str(scalar.tolist())
    if isinstance(scalar, float):
        return scalar
    if isinstance(scalar, (int, bool, str)):
        return scalar
    return str(scalar)


def _coerce_float_or_none(value: object) -> float | None:
    scalar = _coerce_status_scalar(value)
    try:
        if scalar is None:
            return None
        value_f = float(scalar)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _coerce_bool(value: object) -> bool:
    scalar = _coerce_status_scalar(value)
    return bool(scalar)


def _resolve_latest_refined_run(root: zarr.Group) -> str:
    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")
    latest = resolve_latest_complete_run_name(refined_parent)
    if not latest:
        raise RuntimeError("No refined keypoint runs recorded.")
    return latest


@dataclass(frozen=False)
class ReviewSession:
    zarr_path: str
    root: zarr.Group
    refined: zarr.Group
    crop: zarr.Group
    refined_run: str
    crop_run: str

    failures: np.ndarray
    frame_indices: np.ndarray
    roi_images: zarr.Array
    roi_coordinates_full: zarr.Array

    source_refined_row_ids: Optional[np.ndarray]
    source_detect_row_index: Optional[np.ndarray]

    keypoint_labels: list[str]
    keypoint_count: int
    roi_diagonal: Optional[float]
    norm_factor: np.ndarray

    kp_roi_arr: zarr.Array
    kp_img_arr: Optional[zarr.Array]
    kp_norm_arr: Optional[zarr.Array]
    heading_arr: Optional[zarr.Array]
    confidence_arr: Optional[zarr.Array]
    conf_arr: Optional[zarr.Array]
    triangle_area_arr: Optional[zarr.Array]
    min_angle_arr: Optional[zarr.Array]
    triangle_angles_arr: Optional[zarr.Array]
    refined_success_arr: Optional[zarr.Array]
    flip_corrected_arr: Optional[zarr.Array]
    quality_labels_arr: Optional[zarr.Array]
    confidence_valid_arr: Optional[zarr.Array]
    geometry_valid_arr: Optional[zarr.Array]
    usable_arr: Optional[zarr.Array]
    edit_applied_arr: Optional[zarr.Array]
    reason_arr: Optional[zarr.Array]
    heading_finite_arr: Optional[zarr.Array]
    heading_usable_arr: Optional[zarr.Array]
    detection_source_arr: Optional[zarr.Array]

    min_triangle_angle: float
    min_triangle_area: float
    max_triangle_area: Optional[float]
    confidence_threshold: float
    head_triangle_indices: object
    derived_metric_storage: Optional[object]


def _coerce_ints(values: Optional[Sequence[object]]) -> list[int]:
    if not values:
        return []
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _build_reason_array(refined: zarr.Group) -> Optional[zarr.Array]:
    existing = refined.get("reason")
    if existing is not None:
        return existing

    reason_values = read_reason_labels(refined)
    if reason_values is None:
        return None

    chunk_size = 1
    if "keypoints_roi" in refined and np.asarray(refined["keypoints_roi"].shape)[0] > 0:
        chunk_size = int(np.asarray(refined["keypoints_roi"].shape)[0])
    write_reason_columns(
        refined,
        np.asarray(reason_values, dtype=object),
        max(1, chunk_size),
        include_reason_text=True,
        overwrite=True,
    )
    return refined.get("reason")


def _resolve_session_geometry(
    refined: zarr.Group,
    roi_images: zarr.Array,
) -> tuple[float, float, Optional[float], float, Optional[float], object]:
    summary_raw = refined.attrs.get("summary_statistics", {})
    summary = summary_raw.get("refine", summary_raw) if isinstance(summary_raw, dict) else {}

    min_triangle_angle, min_triangle_area, max_triangle_area = _resolve_review_geometry_defaults(refined.attrs)
    min_triangle_angle = float(summary.get("min_triangle_angle", min_triangle_angle))
    min_triangle_area = float(summary.get("min_triangle_area", min_triangle_area))
    max_area_value = summary.get("max_triangle_area")
    if max_area_value is None:
        # Default from review heuristics schema; this value may be None for 3pt schemas.
        max_triangle_area = max_triangle_area
    else:
        max_triangle_area = float(max_area_value)

    confidence_threshold = float(summary.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD))

    roi_diagonal = _roi_diagonal_from_roi_images(roi_images)

    rows = int(np.asarray(refined["keypoints_roi"].shape[0]))
    heading = refined.get("heading")
    if heading is not None and heading.chunks:
        chunk_len = int(heading.chunks[0])
    else:
        chunk_len = max(1, min(1024, rows))
    derived_metric_storage = _ensure_review_derived_metric_storage(
        refined,
        row_count=rows,
        chunk_len=chunk_len,
        roi_diagonal=roi_diagonal,
    )

    return (
        min_triangle_angle,
        min_triangle_area,
        max_triangle_area,
        confidence_threshold,
        roi_diagonal,
        derived_metric_storage,
    )


def resolve_latest_refined_and_crop(
    zarr_path: str,
    *,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    mode: str = "a",
) -> tuple[zarr.Group, zarr.Group, zarr.Group, str, str]:
    root = open_zarr_root(zarr_path, mode=mode)
    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")

    chosen_refined = refined_run or _resolve_latest_refined_run(root)
    refined = refined_parent[chosen_refined]

    chosen_crop = crop_run or refined.attrs.get("source_crop_run")
    if not chosen_crop:
        crop_parent = root.get("crop_runs")
        if crop_parent is None:
            raise RuntimeError("No crop_runs found in archive.")
        chosen_crop = resolve_latest_complete_run_name(crop_parent)
        if not chosen_crop:
            raise RuntimeError("Cannot resolve default crop run.")
    crop_parent = root.get("crop_runs")
    if crop_parent is None or chosen_crop not in crop_parent:
        raise RuntimeError(f"Cannot resolve crop run '{chosen_crop}'.")
    crop = crop_parent[chosen_crop]
    return root, refined, crop, chosen_refined, str(chosen_crop)


def list_review_rois(
    refined: zarr.Group,
    frame_indices: np.ndarray,
    *,
    include_all: bool,
    target_frames: Optional[Sequence[int]] = None,
    target_roi_indices: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, bool]:
    failures = _load_failure_indices(refined, include_all=include_all)
    if not (target_frames or target_roi_indices):
        return np.asarray(failures, dtype="i4"), False

    selected: set[int] = set()
    frame_values = np.asarray(_coerce_ints(target_frames), dtype="i8")
    if frame_values.size > 0:
        selected.update(np.where(np.isin(frame_indices, frame_values))[0].tolist())

    for roi_idx in _coerce_ints(target_roi_indices):
        if 0 <= roi_idx < int(frame_indices.shape[0]):
            selected.add(roi_idx)

    if not selected:
        return np.zeros(0, dtype="i4"), True
    return np.asarray(sorted(selected), dtype="i4"), True


def _reason_at(session: ReviewSession, roi_idx: int) -> str:
    if session.reason_arr is None:
        return ""
    value = session.reason_arr[int(roi_idx)]
    return "" if value is None else str(value)


def _roi_status(session: ReviewSession, roi_idx: int) -> dict[str, object]:
    idx = int(roi_idx)
    status: dict[str, object] = {}
    if session.heading_arr is not None:
        status["heading"] = _coerce_float_or_none(session.heading_arr[idx])
    if session.refined_success_arr is not None:
        status["refined_success"] = _coerce_bool(np.asarray(session.refined_success_arr[idx], dtype=bool).item())
    if session.usable_arr is not None:
        status["usable_keypoints"] = _coerce_bool(np.asarray(session.usable_arr[idx], dtype=bool).item())
    if session.edit_applied_arr is not None:
        status["edit_applied"] = _coerce_bool(np.asarray(session.edit_applied_arr[idx], dtype=bool).item())
    if session.confidence_valid_arr is not None:
        status["confidence_valid"] = _coerce_bool(np.asarray(session.confidence_valid_arr[idx], dtype=bool).item())
    if session.geometry_valid_arr is not None:
        status["geometry_valid"] = _coerce_bool(np.asarray(session.geometry_valid_arr[idx], dtype=bool).item())
    if session.heading_finite_arr is not None:
        status["heading_finite"] = _coerce_bool(np.asarray(session.heading_finite_arr[idx], dtype=bool).item())
    if session.heading_usable_arr is not None:
        status["heading_usable"] = _coerce_bool(np.asarray(session.heading_usable_arr[idx], dtype=bool).item())
    if session.source_refined_row_ids is not None:
        status["source_refined_row_id"] = int(session.source_refined_row_ids[idx])
    if session.source_detect_row_index is not None:
        status["source_detect_row_index"] = int(session.source_detect_row_index[idx])
    return status


def summarize_roi(session: ReviewSession, roi_idx: int) -> dict[str, object]:
    idx = int(roi_idx)
    return {
        "roi_idx": idx,
        "frame_idx": int(session.frame_indices[idx]),
        "reason": _reason_at(session, idx),
        "status": _roi_status(session, idx),
    }


def _count_truthy(arr: Optional[zarr.Array]) -> int:
    if arr is None:
        return 0
    try:
        return int(np.asarray(arr[:], dtype=bool).sum())
    except Exception:
        return 0


def _count_reason_tags(session: ReviewSession) -> dict[str, int]:
    if session.reason_arr is None:
        return {}
    try:
        values = np.asarray(session.reason_arr[:], dtype=object)
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for value in values.tolist():
        text = "" if value is None else str(value)
        for tag in (part.strip() for part in text.split("|")):
            if not tag:
                continue
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def review_session_summary(session: ReviewSession) -> dict[str, object]:
    total = int(session.frame_indices.shape[0])
    refined_success = _count_truthy(session.refined_success_arr)
    usable = _count_truthy(session.usable_arr)
    edit_applied = _count_truthy(session.edit_applied_arr)
    reason_counts = _count_reason_tags(session)
    review_status = session.refined.attrs.get("keypoint_review_status")
    return {
        "zarr_path": str(session.zarr_path),
        "refined_run": str(session.refined_run),
        "crop_run": str(session.crop_run),
        "total_rois": total,
        "refined_success": refined_success,
        "remaining_failures": max(0, total - refined_success),
        "reviewable_failures": int(filter_review_rois(session, filter_mode="failed").size),
        "usable_keypoints": usable,
        "manual_corrections": int(reason_counts.get("manual_correction", 0)),
        "edited_rows": edit_applied,
        "reason_counts": reason_counts,
        "review_status": dict(review_status) if isinstance(review_status, Mapping) else None,
    }


def filter_review_rois(
    session: ReviewSession,
    *,
    filter_mode: str = "failed",
    search: Optional[str] = None,
) -> np.ndarray:
    """Return row indices for the requested UI queue/filter."""

    mode = str(filter_mode or "failed").strip().lower()
    total = int(session.frame_indices.shape[0])
    if total <= 0:
        return np.zeros(0, dtype="i4")

    if mode in {"all", "any"}:
        indices = np.arange(total, dtype="i4")
    elif mode in {"edited", "manual-edited"}:
        if session.edit_applied_arr is None:
            indices = np.zeros(0, dtype="i4")
        else:
            indices = np.where(np.asarray(session.edit_applied_arr[:], dtype=bool))[0].astype("i4", copy=False)
    elif mode in {"manual", "manual_correction"}:
        if session.reason_arr is None:
            indices = np.zeros(0, dtype="i4")
        else:
            values = np.asarray(session.reason_arr[:], dtype=object)
            keep = ["manual_correction" in ("" if value is None else str(value)) for value in values]
            indices = np.where(np.asarray(keep, dtype=bool))[0].astype("i4", copy=False)
    elif mode in {"usable", "valid"}:
        if session.usable_arr is None:
            indices = np.zeros(0, dtype="i4")
        else:
            indices = np.where(np.asarray(session.usable_arr[:], dtype=bool))[0].astype("i4", copy=False)
    elif mode in {"raw_failed", "failed_raw"}:
        indices = _load_failure_indices(session.refined, include_all=True)
        if session.refined_success_arr is not None:
            success = np.asarray(session.refined_success_arr[:], dtype=bool)
            indices = np.where(~success)[0].astype("i4", copy=False)
    else:
        indices = _load_failure_indices(session.refined, include_all=False)

    query = str(search or "").strip().lower()
    if not query:
        return np.asarray(indices, dtype="i4")

    tokens = [token for token in query.replace(",", " ").split() if token]
    if not tokens:
        return np.asarray(indices, dtype="i4")

    kept: list[int] = []
    for roi_idx in np.asarray(indices, dtype="i4").tolist():
        summary = summarize_roi(session, int(roi_idx))
        status_text = " ".join(
            f"{key}={value}" for key, value in sorted(summary["status"].items())  # type: ignore[union-attr]
        )
        haystack = (
            f"roi={roi_idx} frame={summary['frame_idx']} "
            f"reason={summary['reason']} {status_text}"
        ).lower()
        if all(token in haystack for token in tokens):
            kept.append(int(roi_idx))
    return np.asarray(kept, dtype="i4")


def find_queue_position(
    session: ReviewSession,
    *,
    roi_idx: Optional[int] = None,
    frame_idx: Optional[int] = None,
) -> Optional[int]:
    if roi_idx is not None:
        matches = np.where(np.asarray(session.failures, dtype="i4") == int(roi_idx))[0]
        if matches.size > 0:
            return int(matches[0])
    if frame_idx is not None:
        current_frames = np.asarray(session.frame_indices[np.asarray(session.failures, dtype="i4")], dtype=np.int64)
        matches = np.where(current_frames == int(frame_idx))[0]
        if matches.size > 0:
            return int(matches[0])
    return None


def resolve_review_session(
    zarr_path: str,
    *,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    include_all: bool = False,
    target_frames: Optional[Sequence[int]] = None,
    target_roi_indices: Optional[Sequence[int]] = None,
) -> ReviewSession:
    root, refined, crop, resolved_refined_run, resolved_crop_run = resolve_latest_refined_and_crop(
        zarr_path,
        refined_run=refined_run,
        crop_run=crop_run,
        mode="a",
    )

    if "frame_indices" not in crop:
        raise RuntimeError("Crop run is missing frame_indices.")
    if "roi_images" not in crop:
        raise RuntimeError("Crop run is missing roi_images.")
    if "roi_coordinates_full" not in crop:
        raise RuntimeError("Crop run is missing roi_coordinates_full.")

    if "keypoints_roi" not in refined:
        raise RuntimeError("Refined run is missing keypoints_roi.")

    frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64)
    roi_images = crop["roi_images"]
    roi_coordinates_full = crop["roi_coordinates_full"]

    total_rois = int(frame_indices.shape[0])
    source_refined_row_ids, source_detect_row_index = load_row_identity_arrays(
        crop,
        total_rois=total_rois,
    )

    failures, _targeted = list_review_rois(
        refined,
        frame_indices,
        include_all=include_all,
        target_frames=target_frames,
        target_roi_indices=target_roi_indices,
    )

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
    reason_arr = _build_reason_array(refined)
    heading_finite_arr = refined.get("heading_finite")
    heading_usable_arr = refined.get("heading_usable")
    detection_source_arr = refined.get("detection_source")

    keypoint_count = int(kp_roi_arr.shape[1])
    labels = list(refined.attrs.get("keypoint_labels", _DEFAULT_LABELS))
    if len(labels) != keypoint_count:
        raise ValueError(
            "Refined keypoint run keypoint_labels count does not match keypoints_roi K "
            f"({len(labels)} vs {keypoint_count})."
        )

    min_triangle_angle, min_triangle_area, max_triangle_area, confidence_threshold, roi_diagonal, derived_metric_storage = _resolve_session_geometry(
        refined,
        roi_images,
    )
    head_triangle_indices = resolve_head_triangle_for_labels(
        labels,
        keypoint_count=keypoint_count,
        allow_legacy_3point_fallback=True,
    )

    full_h, full_w = _resolve_full_frame_dimensions(root)
    norm_factor = np.array([full_w, full_h], dtype=np.float64)

    return ReviewSession(
        zarr_path=zarr_path,
        root=root,
        refined=refined,
        crop=crop,
        refined_run=resolved_refined_run,
        crop_run=resolved_crop_run,
        failures=np.asarray(failures, dtype="i4"),
        frame_indices=frame_indices,
        roi_images=roi_images,
        roi_coordinates_full=roi_coordinates_full,
        source_refined_row_ids=source_refined_row_ids,
        source_detect_row_index=source_detect_row_index,
        keypoint_labels=labels,
        keypoint_count=keypoint_count,
        roi_diagonal=roi_diagonal,
        norm_factor=norm_factor,
        kp_roi_arr=kp_roi_arr,
        kp_img_arr=kp_img_arr,
        kp_norm_arr=kp_norm_arr,
        heading_arr=heading_arr,
        confidence_arr=confidence_arr,
        conf_arr=conf_arr,
        triangle_area_arr=triangle_area_arr,
        min_angle_arr=min_angle_arr,
        triangle_angles_arr=triangle_angles_arr,
        refined_success_arr=refined_success_arr,
        flip_corrected_arr=flip_corrected_arr,
        quality_labels_arr=quality_labels_arr,
        confidence_valid_arr=confidence_valid_arr,
        geometry_valid_arr=geometry_valid_arr,
        usable_arr=usable_arr,
        edit_applied_arr=edit_applied_arr,
        reason_arr=reason_arr,
        heading_finite_arr=heading_finite_arr,
        heading_usable_arr=heading_usable_arr,
        detection_source_arr=detection_source_arr,
        min_triangle_angle=min_triangle_angle,
        min_triangle_area=min_triangle_area,
        max_triangle_area=max_triangle_area,
        confidence_threshold=confidence_threshold,
        head_triangle_indices=head_triangle_indices,
        derived_metric_storage=derived_metric_storage,
    )


def load_roi_payload(session: ReviewSession, position: int) -> Mapping[str, object]:
    if session.failures.size == 0:
        raise IndexError("No ROIs are currently loaded for review.")
    if position < 0 or position >= int(session.failures.size):
        raise IndexError("ROI position is out of range.")

    roi_idx = int(session.failures[position])
    frame_idx = int(session.frame_indices[roi_idx])

    points = np.asarray(session.kp_roi_arr[roi_idx], dtype=float)
    image = np.asarray(session.roi_images[roi_idx])
    reason = _reason_at(session, roi_idx)
    status = _roi_status(session, roi_idx)

    image_payload = {
        "shape": [int(v) for v in image.shape],
        "channels": int(image.shape[-1]) if image.ndim >= 3 else 1,
        "dtype": str(image.dtype),
        "encoding": "base64_raw",
        "pixels": base64.b64encode(np.asarray(image, dtype=np.uint8).tobytes()).decode("ascii"),
    }

    def _json_point(point: np.ndarray) -> list[Optional[float]]:
        x = float(point[0])
        y = float(point[1])
        return [
            x if np.isfinite(x) else None,
            y if np.isfinite(y) else None,
        ]

    return {
        "roi_idx": roi_idx,
        "position": int(position),
        "total": int(session.failures.size),
        "frame_idx": frame_idx,
        "labels": list(session.keypoint_labels),
        "points": [_json_point(point) for point in points],
        "reason": reason,
        "status": status,
        "roi_image": image_payload,
    }


def save_roi_correction(
    session: ReviewSession,
    *,
    position: int,
    points: Sequence[Sequence[float]],
) -> dict[str, object]:
    if session.failures.size == 0:
        raise IndexError("No ROIs are currently loaded for review.")
    if position < 0 or position >= int(session.failures.size):
        raise IndexError("ROI position is out of range.")

    points_arr = np.asarray(points, dtype=np.float64)
    if points_arr.shape != (session.keypoint_count, 2):
        raise ValueError(
            f"Expected points shape ({session.keypoint_count}, 2), got {points_arr.shape}."
        )
    if not np.isfinite(points_arr).all():
        missing = [label for label, point in zip(session.keypoint_labels, points_arr) if not np.isfinite(point).all()]
        detail = ", ".join(missing)
        raise ValueError(
            f"Cannot save incomplete keypoints. Missing: {detail}" if detail else "Cannot save incomplete keypoints."
        )

    roi_idx = int(session.failures[position])
    frame_idx = int(session.frame_indices[roi_idx])
    changed = False

    changed |= _set_roi_value_if_changed(session.kp_roi_arr, roi_idx, points_arr)

    full_points = points_arr + np.asarray(session.roi_coordinates_full[roi_idx], dtype=np.float64)
    changed |= _set_roi_value_if_changed(session.kp_img_arr, roi_idx, full_points)
    changed |= _set_roi_value_if_changed(session.kp_norm_arr, roi_idx, full_points / session.norm_factor)

    heading_val = compute_heading_from_attrs(
        session.refined.attrs,
        labels=session.keypoint_labels,
        points=points_arr,
    )
    changed |= _set_roi_value_if_changed(session.heading_arr, roi_idx, heading_val)

    triangle = compute_geometry_metrics(
        select_head_triangle_points(points_arr, session.head_triangle_indices)
    )
    max_ok = session.max_triangle_area is None or triangle.area <= float(session.max_triangle_area)
    geom_ok = bool(
        np.isfinite(triangle.min_angle)
        and np.isfinite(triangle.area)
        and triangle.min_angle >= float(session.min_triangle_angle)
        and triangle.area >= float(session.min_triangle_area)
        and max_ok
    )

    changed |= _set_roi_value_if_changed(session.triangle_area_arr, roi_idx, triangle.area)
    changed |= _set_roi_value_if_changed(session.min_angle_arr, roi_idx, triangle.min_angle)
    changed |= _set_roi_value_if_changed(session.triangle_angles_arr, roi_idx, triangle.angles)
    changed |= _set_review_derived_metric_row(
        session.derived_metric_storage,
        roi_idx=roi_idx,
        keypoints_roi=points_arr,
        keypoint_labels=session.keypoint_labels,
        roi_diagonal=session.roi_diagonal,
    )

    conf_ok = True
    if session.conf_arr is not None:
        conf_vals = np.ones(session.keypoint_count, dtype=np.float64)
        changed |= _set_roi_value_if_changed(session.conf_arr, roi_idx, conf_vals)
        conf_ok = bool(np.all(conf_vals >= session.confidence_threshold))

    changed |= _set_roi_value_if_changed(session.confidence_arr, roi_idx, 1.0)
    changed |= _set_roi_value_if_changed(session.refined_success_arr, roi_idx, True)
    changed |= _set_roi_value_if_changed(session.flip_corrected_arr, roi_idx, False)
    changed |= _set_roi_value_if_changed(session.quality_labels_arr, roi_idx, 0)
    changed |= _set_roi_value_if_changed(session.confidence_valid_arr, roi_idx, conf_ok)
    changed |= _set_roi_value_if_changed(session.geometry_valid_arr, roi_idx, geom_ok)
    changed |= _set_roi_value_if_changed(session.usable_arr, roi_idx, conf_ok and geom_ok)

    heading_is_finite = bool(np.isfinite(heading_val))
    changed |= _set_roi_value_if_changed(session.heading_finite_arr, roi_idx, heading_is_finite)
    if session.heading_usable_arr is not None:
        detection_source = 0
        if session.detection_source_arr is not None:
            detection_source = int(session.detection_source_arr[roi_idx])
        changed |= _set_roi_value_if_changed(
            session.heading_usable_arr,
            roi_idx,
            True and (detection_source == 0) and heading_is_finite,
        )

    reason_updated = False
    if session.reason_arr is not None:
        existing = session.reason_arr[roi_idx]
        existing_text = "" if existing is None else str(existing)
        reason_value = _build_manual_reason(existing_text, geom_ok=geom_ok)
        if reason_value != existing_text:
            session.reason_arr[roi_idx : roi_idx + 1] = np.array([reason_value], dtype=object)
            reason_updated = True
            changed = True

    stale_touched = 0
    if changed:
        _mark_edit_applied(session.edit_applied_arr, roi_idx)
        stale_touched = int(
            mark_downstream_subject_mask_runs_stale(
                session.root,
                source_keypoint_group="refined_keypoints_runs",
                source_keypoints_run=str(session.refined_run),
                roi_indices=[roi_idx],
                frame_indices=[frame_idx],
                reason="keypoint_manual_correction",
            )
        )

    return {
        "roi_idx": roi_idx,
        "position": int(position),
        "frame_idx": frame_idx,
        "changed": bool(changed),
        "reason_updated": bool(reason_updated),
        "heading": float(heading_val) if np.isfinite(heading_val) else None,
        "stale_touched": stale_touched,
        "geometry_ok": bool(geom_ok),
        "confidence_ok": bool(conf_ok),
        "readback": summarize_roi(session, roi_idx),
    }


def _nan_like_row(arr: Optional[zarr.Array], roi_idx: int) -> object:
    if arr is None:
        return np.nan
    return np.full_like(np.asarray(arr[int(roi_idx)]), np.nan)


def _mark_manual_state_changed(
    session: ReviewSession,
    *,
    roi_idx: int,
    frame_idx: int,
    changed: bool,
    stale_reason: str,
) -> int:
    if not changed:
        return 0
    _mark_edit_applied(session.edit_applied_arr, roi_idx)
    return int(
        mark_downstream_subject_mask_runs_stale(
            session.root,
            source_keypoint_group="refined_keypoints_runs",
            source_keypoints_run=str(session.refined_run),
            roi_indices=[roi_idx],
            frame_indices=[frame_idx],
            reason=stale_reason,
        )
    )


def _clear_keypoint_solution(
    session: ReviewSession,
    *,
    roi_idx: int,
    reason_tag: str,
    stale_reason: str,
) -> dict[str, object]:
    idx = int(roi_idx)
    frame_idx = int(session.frame_indices[idx])
    changed = False

    changed |= _set_roi_value_if_changed(session.kp_roi_arr, idx, _nan_like_row(session.kp_roi_arr, idx))
    changed |= _set_roi_value_if_changed(session.kp_img_arr, idx, _nan_like_row(session.kp_img_arr, idx))
    changed |= _set_roi_value_if_changed(session.kp_norm_arr, idx, _nan_like_row(session.kp_norm_arr, idx))
    changed |= _set_roi_value_if_changed(session.heading_arr, idx, np.nan)
    changed |= _set_roi_value_if_changed(session.confidence_arr, idx, np.nan)
    changed |= _set_roi_value_if_changed(session.conf_arr, idx, _nan_like_row(session.conf_arr, idx))
    changed |= _set_roi_value_if_changed(session.triangle_area_arr, idx, np.nan)
    changed |= _set_roi_value_if_changed(session.min_angle_arr, idx, np.nan)
    changed |= _set_roi_value_if_changed(session.triangle_angles_arr, idx, _nan_like_row(session.triangle_angles_arr, idx))
    changed |= _set_review_derived_metric_row(
        session.derived_metric_storage,
        roi_idx=idx,
        keypoints_roi=None,
        keypoint_labels=session.keypoint_labels,
        roi_diagonal=session.roi_diagonal,
    )
    changed |= _set_roi_value_if_changed(session.refined_success_arr, idx, False)
    changed |= _set_roi_value_if_changed(session.flip_corrected_arr, idx, False)
    changed |= _set_roi_value_if_changed(session.quality_labels_arr, idx, 0)
    changed |= _set_roi_value_if_changed(session.confidence_valid_arr, idx, False)
    changed |= _set_roi_value_if_changed(session.geometry_valid_arr, idx, False)
    changed |= _set_roi_value_if_changed(session.usable_arr, idx, False)
    changed |= _set_roi_value_if_changed(session.heading_finite_arr, idx, False)
    changed |= _set_roi_value_if_changed(session.heading_usable_arr, idx, False)

    reason_updated = False
    if session.reason_arr is not None:
        existing = _reason_at(session, idx)
        reason_value = str(_clean_reason(existing, [reason_tag]))
        if existing != reason_value:
            session.reason_arr[idx : idx + 1] = np.asarray([reason_value], dtype=object)
            changed = True
            reason_updated = True

    stale_touched = _mark_manual_state_changed(
        session,
        roi_idx=idx,
        frame_idx=frame_idx,
        changed=changed,
        stale_reason=stale_reason,
    )
    return {
        "roi_idx": idx,
        "frame_idx": frame_idx,
        "changed": bool(changed),
        "reason_updated": bool(reason_updated),
        "stale_touched": int(stale_touched),
        "readback": summarize_roi(session, idx),
    }


def mark_no_keypoints(session: ReviewSession, *, position: int) -> dict[str, object]:
    if session.failures.size == 0:
        raise IndexError("No ROIs are currently loaded for review.")
    if position < 0 or position >= int(session.failures.size):
        raise IndexError("ROI position is out of range.")
    return {
        "action": "mark_no_keypoints",
        **_clear_keypoint_solution(
            session,
            roi_idx=int(session.failures[position]),
            reason_tag="fish_present_no_keypoints",
            stale_reason="keypoint_mark_no_keypoints",
        ),
    }


def mark_detection_issue(session: ReviewSession, *, position: int) -> dict[str, object]:
    if session.failures.size == 0:
        raise IndexError("No ROIs are currently loaded for review.")
    if position < 0 or position >= int(session.failures.size):
        raise IndexError("ROI position is out of range.")
    return {
        "action": "mark_detection_issue",
        **_clear_keypoint_solution(
            session,
            roi_idx=int(session.failures[position]),
            reason_tag="detection_issue",
            stale_reason="keypoint_mark_detection_issue",
        ),
    }


def clear_failure_label(session: ReviewSession, *, position: int) -> dict[str, object]:
    if session.failures.size == 0:
        raise IndexError("No ROIs are currently loaded for review.")
    if position < 0 or position >= int(session.failures.size):
        raise IndexError("ROI position is out of range.")
    if session.reason_arr is None:
        raise RuntimeError("Cannot clear failure label: reason labels are unavailable.")

    roi_idx = int(session.failures[position])
    frame_idx = int(session.frame_indices[roi_idx])
    existing = _reason_at(session, roi_idx)
    tags = {token.strip() for token in existing.split("|") if token.strip()}
    clearable = sorted(tags & {"fish_present_no_keypoints", "detection_issue"})
    changed = False
    if clearable:
        reason_value = _build_cleared_failure_reason(existing)
        if existing != reason_value:
            session.reason_arr[roi_idx : roi_idx + 1] = np.asarray([reason_value], dtype=object)
            changed = True
    stale_touched = _mark_manual_state_changed(
        session,
        roi_idx=roi_idx,
        frame_idx=frame_idx,
        changed=changed,
        stale_reason="keypoint_clear_failure_label",
    )
    return {
        "action": "clear_failure_label",
        "roi_idx": roi_idx,
        "frame_idx": frame_idx,
        "changed": bool(changed),
        "cleared_tags": clearable,
        "stale_touched": int(stale_touched),
        "readback": summarize_roi(session, roi_idx),
    }


def flag_followup_frame(
    session: ReviewSession,
    *,
    position: int,
    flag_path: Optional[str | Path],
) -> dict[str, object]:
    if flag_path is None:
        raise RuntimeError("No frame flag file configured.")
    if session.failures.size == 0:
        raise IndexError("No ROIs are currently loaded for review.")
    if position < 0 or position >= int(session.failures.size):
        raise IndexError("ROI position is out of range.")

    roi_idx = int(session.failures[position])
    frame_idx = int(session.frame_indices[roi_idx])
    extra_fields = row_identity_payload(
        roi_idx,
        source_refined_row_ids=session.source_refined_row_ids,
        source_detect_row_index=session.source_detect_row_index,
    )
    append_flagged_frame(
        Path(flag_path).expanduser(),
        session.zarr_path,
        frame_idx,
        roi_idx,
        extra_fields=extra_fields,
    )
    return {
        "action": "flag_followup",
        "roi_idx": roi_idx,
        "frame_idx": frame_idx,
        "changed": True,
        "flag_path": str(Path(flag_path).expanduser()),
        "readback": summarize_roi(session, roi_idx),
    }


def _approve_authoritative_refined_keypoints(
    session: ReviewSession,
    *,
    state: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> dict[str, object]:
    if str(state).strip().lower() != "approved":
        return {"attempted": False, "reason": "review_state_not_approved"}
    zarr_path = Path(session.zarr_path).expanduser()
    if not zarr_path.exists():
        return {"attempted": False, "reason": "zarr_path_unavailable", "zarr_path": str(session.zarr_path)}

    from ..cli.palette import ApproveRequest, approve

    envelope = approve(
        ApproveRequest(
            recording=zarr_path,
            stage="refined_keypoints",
            run=session.refined_run,
            approved_by=reviewer,
            note=notes or "keypoint review sign-off",
            apply=True,
        )
    )
    return {
        "attempted": True,
        "status": envelope.get("status"),
        "reason_code": envelope.get("reason_code"),
        "run": envelope.get("run"),
        "envelope": envelope,
    }


def _authoritative_approval_ok(payload: Mapping[str, object]) -> bool:
    return bool(payload.get("attempted")) and str(payload.get("status") or "").strip().lower() == "ok"


def _mirror_authoritative_approval(parent: zarr.Group, run_name: str, payload: Mapping[str, object]) -> None:
    envelope = payload.get("envelope")
    approval = envelope.get("approval") if isinstance(envelope, Mapping) else None
    if not isinstance(approval, Mapping):
        approval = {}
    set_authoritative_run(
        parent,
        run_name,
        approved_by=str(approval.get("approved_by") or "unknown"),
        approved_at=str(approval.get("approved_at") or ""),
        git_sha=str(approval.get("git_sha") or ""),
        note=str(approval.get("note") or ""),
    )


def apply_review_status(
    session: ReviewSession,
    *,
    state: str,
    method: str = "manual",
    intended_use: Optional[str] = None,
    reviewer: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, object]:
    refined_parent = session.root.get("refined_keypoints_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")
    resolved_intended_use = _resolve_review_intended_use(
        requested=intended_use,
        refined=session.refined,
        root=session.root,
        zarr_path=session.zarr_path,
    )
    authoritative_approval = _approve_authoritative_refined_keypoints(
        session,
        state=str(state),
        reviewer=reviewer,
        notes=notes,
    )
    if str(state).strip().lower() == "approved" and not _authoritative_approval_ok(authoritative_approval):
        return {
            "action": "apply_review_status",
            "changed": False,
            "review_status": dict(session.refined.attrs.get("keypoint_review_status") or {}),
            "registry_sync": None,
            "postprocess_summary": None,
            "authoritative_approval": authoritative_approval,
        }
    if _authoritative_approval_ok(authoritative_approval):
        _mirror_authoritative_approval(refined_parent, session.refined_run, authoritative_approval)
    payload, sync = _apply_review_status(
        refined_parent,
        session.refined_run,
        session.refined,
        zarr_path=session.zarr_path,
        state=str(state),
        method=str(method),
        intended_use=resolved_intended_use,
        reviewer=reviewer,
        notes=notes,
    )
    postprocess_summary = _update_postprocess_summary(
        session.refined,
        root=session.root,
        print_summary=False,
    )
    return {
        "action": "apply_review_status",
        "changed": True,
        "review_status": payload,
        "registry_sync": sync,
        "postprocess_summary": postprocess_summary,
        "authoritative_approval": authoritative_approval,
    }
