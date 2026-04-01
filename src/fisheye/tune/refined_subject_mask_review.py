"""Manual review/editor for refined subject-mask runs."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

# Limit threading before importing numpy/cv2 to avoid oversubscribing cores in review UIs.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter1d

from ..shared.detect_reason_codec import encode_reason_bytes, read_reason_labels, write_reason_columns
from ..shared.provenance_attrs import build_source_keypoints_attrs, resolve_source_keypoints_run
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..utils.system import get_environment_info, get_git_info
from ..utils.zarr_io import open_zarr_root

try:
    cv2_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
except (TypeError, ValueError):
    cv2_threads = 2
cv2.setNumThreads(cv2_threads)

WINDOW_NAME = "Refined Subject Mask Review"
DEFAULT_COMPONENTS = ("subject_body", "swim_bladder")
COMPONENT_ALIASES = {
    "body": "subject_body",
    "subject": "subject_body",
    "subject_body": "subject_body",
    "subject-body": "subject_body",
    "whole_subject": "subject_body",
    "whole-subject": "subject_body",
    "swimbladder": "swim_bladder",
    "swim_bladder": "swim_bladder",
    "swim-bladder": "swim_bladder",
    "eye": "eyes_union",
    "eyes": "eyes_union",
    "eyes_union": "eyes_union",
    "eye-union": "eyes_union",
    "eye_left": "eye_left",
    "left_eye": "eye_left",
    "left-eye": "eye_left",
    "eye_right": "eye_right",
    "right_eye": "eye_right",
    "right-eye": "eye_right",
}
COMPONENT_COLORS: dict[str, tuple[int, int, int]] = {
    "subject_body": (0, 220, 0),
    "swim_bladder": (255, 220, 0),
    "eyes_union": (255, 0, 255),
    "eye_left": (0, 255, 255),
    "eye_right": (255, 128, 255),
}
DISPLAY_SCALE = 2.5
DEFAULT_BRUSH_RADIUS = 6
DEFAULT_REVIEW_METHOD = "manual"
DEFAULT_REVIEW_INTENDED_USE = "training"
DEFAULT_RUN_METHOD = "refined_subject_mask_manual_review_v1"
REFINED_SUBJECT_STAGE_NAME = "refine_subject_masks"
REFINED_SUBJECT_SYNC_METHOD = "sync_refined_subject_mask_metadata"
DEFAULT_SUBJECT_BODY_COMPONENT_SCHEMA_ID = "subject_body_v1"
DEFAULT_SUBJECT_BODY_ANATOMICAL_SCOPE = "body_core"
DEFAULT_SUBJECT_BODY_PECTORAL_FIN_POLICY = "excluded_or_unresolved"
SIGMA_NOISE_SMOOTHING_SIGMA = 3.0
CURVATURE_VAR_SMOOTHING_SIGMA = 3.0
CURVATURE_VAR_STEP = 5


@dataclass(frozen=True)
class SourceSubjectMaskRun:
    run_name: str
    group: zarr.Group
    crop_run: str
    masks_roi: Any
    detection_source: Any
    mask_labels: tuple[str, ...]
    available_channels: np.ndarray
    frame_indices: Any | None
    frame_counts: Any | None
    detection_indices: Any | None
    source_method: Optional[str]
    source_keypoints_run: Optional[str]
    source_keypoint_group: Optional[str]


@dataclass(frozen=True)
class RefinedSubjectMaskRun:
    run_name: str
    parent: zarr.Group
    group: zarr.Group
    component_names: tuple[str, ...]
    component_to_index: dict[str, int]


@dataclass(frozen=True)
class RefinedSubjectComponentSeed:
    component_name: str
    masks: np.ndarray
    source_payload: Mapping[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_component_name(name: object) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    return COMPONENT_ALIASES.get(text, text)


def _normalize_component_list(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not values:
        return tuple(DEFAULT_COMPONENTS)
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        normalized = _normalize_component_name(raw)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    if not result:
        return tuple(DEFAULT_COMPONENTS)
    return tuple(result)


def _require_gui_display() -> None:
    display = str(os.environ.get("DISPLAY") or "").strip()
    wayland_display = str(os.environ.get("WAYLAND_DISPLAY") or "").strip()
    if display or wayland_display:
        return
    raise RuntimeError(
        "No GUI display detected for OpenCV review window. DISPLAY and WAYLAND_DISPLAY are unset."
    )


def _resolve_latest_run(parent: Any, label: str) -> str:
    if parent is None:
        raise RuntimeError(f"No {label} found in archive.")
    latest = parent.attrs.get("latest")
    if latest:
        return str(latest)
    keys = sorted(str(key) for key in parent.keys())
    if not keys:
        raise RuntimeError(f"No runs found under {label}.")
    return keys[-1]


def _default_refined_run_name() -> str:
    return f"refined_subject_masks_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}"


def _infer_refined_label_schema_id(component_names: Sequence[str]) -> str:
    labels = tuple(component_names)
    if labels == ("subject_body", "eye_left", "eye_right", "swim_bladder"):
        return "subject_v1_lr"
    if labels == ("subject_body",):
        return "refined_subject_v1_body"
    if labels == ("swim_bladder",):
        return "refined_subject_v1_swim_bladder"
    if labels == ("subject_body", "swim_bladder"):
        return "refined_subject_v1_body_swim"
    return "refined_subject_v1_custom"


def _review_payload(
    *,
    state: str,
    method: str,
    intended_use: str,
    reviewer: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "state": state,
        "method": method,
        "intended_use": intended_use,
        "timestamp_utc": _utc_now(),
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes
    return payload


def _copy_optional_array(dest: zarr.Group, source: zarr.Group, name: str) -> None:
    if name not in source or name in dest:
        return
    dest.create_array(name, data=np.asarray(source[name][:]), overwrite=True)


def _normalize_provenance_channels(value: object, *, default_component: str) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        channels = [str(item) for item in value if str(item).strip()]
        return channels or [str(default_component)]
    if value is None:
        return [str(default_component)]
    text = str(value).strip()
    return [text] if text else [str(default_component)]


def _get_component_group(parent: zarr.Group, component_name: str) -> zarr.Group | None:
    components_parent = parent.get("components")
    if components_parent is None:
        return None
    component_group = components_parent.get(str(component_name))
    if component_group is None or not hasattr(component_group, "attrs"):
        return None
    return component_group


def _get_component_provenance_group(parent: zarr.Group, component_name: str) -> zarr.Group | None:
    component_group = _get_component_group(parent, component_name)
    if component_group is None:
        return None
    provenance_group = component_group.get("provenance")
    if provenance_group is None or not hasattr(provenance_group, "attrs"):
        return None
    return provenance_group


def _set_attr_if_missing(attrs: Any, key: str, value: object) -> None:
    existing = attrs.get(key)
    if existing is None or (isinstance(existing, str) and not existing.strip()):
        attrs[key] = value


def _source_component_provenance_payload(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> dict[str, object]:
    legacy_component_payload = (
        dict(source.group.attrs.get("component_provenance") or {}).get("components", {}).get(str(component_name), {})
    )
    provenance_group = _get_component_provenance_group(source.group, component_name)
    provenance_attrs = dict(provenance_group.attrs) if provenance_group is not None else {}
    label_schema_id = source.group.attrs.get("label_schema_id")
    created_at_utc = source.group.attrs.get("created_at_utc")
    projection_mode = source.group.attrs.get("projection_mode")

    source_channels = provenance_attrs.get("source_channels")
    if source_channels is None:
        source_channels = legacy_component_payload.get("source_channels", legacy_component_payload.get("source_channel"))

    payload: dict[str, object] = {
        "source_stage": str(
            provenance_attrs.get("source_stage")
            or legacy_component_payload.get("source_stage")
            or "subject_mask_runs"
        ),
        "source_run": str(
            provenance_attrs.get("source_run")
            or legacy_component_payload.get("source_run")
            or source.run_name
        ),
        "source_method": str(
            provenance_attrs.get("source_method")
            or legacy_component_payload.get("source_method")
            or source.source_method
            or source.group.attrs.get("method")
            or "unknown"
        ),
        "source_channels": _normalize_provenance_channels(source_channels, default_component=component_name),
    }
    source_label_schema_id = provenance_attrs.get("source_label_schema_id")
    if source_label_schema_id is None:
        source_label_schema_id = legacy_component_payload.get("source_label_schema_id", label_schema_id)
    if source_label_schema_id is not None:
        payload["source_label_schema_id"] = str(source_label_schema_id)
    source_created_at = provenance_attrs.get("source_created_at_utc")
    if source_created_at is None:
        source_created_at = legacy_component_payload.get("source_created_at_utc", created_at_utc)
    if source_created_at is not None:
        payload["source_created_at_utc"] = str(source_created_at)
    projection_mode_value = provenance_attrs.get("projection_mode")
    if projection_mode_value is None:
        projection_mode_value = legacy_component_payload.get("projection_mode", projection_mode)
    if projection_mode_value is not None:
        payload["projection_mode"] = str(projection_mode_value)
    return payload


def _ensure_refined_component_provenance_payload(
    refined_group: zarr.Group,
    *,
    component_name: str,
    source_payload: Mapping[str, object],
    created_at_utc: str,
) -> zarr.Group:
    component_group = refined_group.require_group("components").require_group(component_name)
    provenance_group = component_group.require_group("provenance")
    for key, value in source_payload.items():
        _set_attr_if_missing(provenance_group.attrs, str(key), value)
    provenance_group.attrs.setdefault("last_update_stage", REFINED_SUBJECT_STAGE_NAME)
    provenance_group.attrs.setdefault("last_update_mode", "create")
    provenance_group.attrs.setdefault("last_update_method", str(refined_group.attrs.get("method") or DEFAULT_RUN_METHOD))
    provenance_group.attrs.setdefault(
        "updated_at_utc",
        str(refined_group.attrs.get("updated_at_utc") or refined_group.attrs.get("created_at_utc") or created_at_utc),
    )
    return provenance_group


def _ensure_refined_component_provenance(
    refined_group: zarr.Group,
    *,
    source: SourceSubjectMaskRun,
    component_name: str,
    created_at_utc: str,
) -> zarr.Group:
    source_payload = _source_component_provenance_payload(source, component_name)
    return _ensure_refined_component_provenance_payload(
        refined_group,
        component_name=component_name,
        source_payload=source_payload,
        created_at_utc=created_at_utc,
    )


def _component_masks_from_source(source: SourceSubjectMaskRun, component_name: str) -> np.ndarray:
    total_rois = int(source.masks_roi.shape[0])
    height = int(source.masks_roi.shape[2])
    width = int(source.masks_roi.shape[3])
    masks = np.zeros((total_rois, height, width), dtype=np.uint8)
    if component_name not in source.mask_labels:
        return masks
    source_idx = source.mask_labels.index(component_name)
    if source_idx >= int(source.available_channels.shape[0]) or not bool(source.available_channels[source_idx]):
        return masks
    return np.asarray(source.masks_roi[:, source_idx], dtype=np.uint8)


def _build_single_source_component_seeds(
    source: SourceSubjectMaskRun,
    component_names: Sequence[str],
) -> dict[str, RefinedSubjectComponentSeed]:
    return {
        str(component_name): RefinedSubjectComponentSeed(
            component_name=str(component_name),
            masks=_component_masks_from_source(source, str(component_name)),
            source_payload=_source_component_provenance_payload(source, str(component_name)),
        )
        for component_name in component_names
    }


def _capture_component_sync_states(
    run_group: zarr.Group,
    component_name: str,
    roi_indices: Sequence[int],
    *,
    comp_idx: int,
) -> dict[int, tuple[object, ...]]:
    component_group = run_group.require_group("components").require_group(component_name)
    return {
        int(roi_idx): _component_sync_state(run_group, component_group, comp_idx=comp_idx, roi_idx=int(roi_idx))
        for roi_idx in roi_indices
    }


def _component_sync_states_changed(
    run_group: zarr.Group,
    component_name: str,
    roi_indices: Sequence[int],
    *,
    comp_idx: int,
    before_states: Mapping[int, tuple[object, ...]],
) -> bool:
    component_group = run_group.require_group("components").require_group(component_name)
    for roi_idx in roi_indices:
        if before_states[int(roi_idx)] != _component_sync_state(
            run_group,
            component_group,
            comp_idx=comp_idx,
            roi_idx=int(roi_idx),
        ):
            return True
    return False


def _write_refined_component_last_update(
    refined_group: zarr.Group,
    *,
    component_name: str,
    updated_at_utc: str,
    update_mode: str,
    update_method: str,
) -> None:
    provenance_group = refined_group.require_group("components").require_group(component_name).require_group("provenance")
    provenance_group.attrs["last_update_stage"] = REFINED_SUBJECT_STAGE_NAME
    provenance_group.attrs["last_update_mode"] = str(update_mode)
    provenance_group.attrs["last_update_method"] = str(update_method)
    provenance_group.attrs["updated_at_utc"] = str(updated_at_utc)


def _compute_mask_metrics(masks_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(masks_roi, dtype=np.uint8) > 0
    area = np.sum(binary.reshape(binary.shape[0], binary.shape[1], -1), axis=2, dtype=np.int64)
    present = area > 0
    return present.astype(bool), np.asarray(area, dtype=np.float32)


def _compute_single_mask_geometry_metrics(mask: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray, bool]:
    binary = np.asarray(mask, dtype=np.uint8) > 0
    centroid_xy = np.zeros((2,), dtype=np.float32)
    bbox_xyxy = np.zeros((4,), dtype=np.float32)
    if int(np.count_nonzero(binary)) <= 0:
        return centroid_xy, False, bbox_xyxy, False

    ys, xs = np.nonzero(binary)
    centroid_xy[:] = np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
    bbox_xyxy[:] = np.asarray(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )
    return centroid_xy, True, bbox_xyxy, True


def _compute_geometry_metrics(masks_roi: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks_roi, dtype=np.uint8)
    if binary.ndim != 4:
        raise ValueError(f"Expected masks with shape (N,C,H,W), got {tuple(binary.shape)}")
    total_rois = int(binary.shape[0])
    channel_count = int(binary.shape[1])
    centroid_xy = np.zeros((total_rois, channel_count, 2), dtype=np.float32)
    centroid_valid = np.zeros((total_rois, channel_count), dtype=bool)
    bbox_xyxy = np.zeros((total_rois, channel_count, 4), dtype=np.float32)
    bbox_valid = np.zeros((total_rois, channel_count), dtype=bool)

    for roi_idx in range(total_rois):
        for comp_idx in range(channel_count):
            centroid, centroid_is_valid, bbox, bbox_is_valid = _compute_single_mask_geometry_metrics(
                binary[roi_idx, comp_idx]
            )
            centroid_xy[roi_idx, comp_idx] = centroid
            centroid_valid[roi_idx, comp_idx] = centroid_is_valid
            bbox_xyxy[roi_idx, comp_idx] = bbox
            bbox_valid[roi_idx, comp_idx] = bbox_is_valid

    return {
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
    }


def _compute_single_mask_topology_metrics(mask: np.ndarray) -> tuple[int, float, int, float]:
    binary = np.asarray(mask, dtype=np.uint8) > 0
    area = int(np.count_nonzero(binary))
    if area <= 0:
        return 0, 0.0, 0, 0.0

    component_input = binary.astype(np.uint8, copy=False)
    num_labels, labels = cv2.connectedComponents(component_input, connectivity=8)
    if num_labels <= 1:
        component_count = 0
        largest_component_fraction = 0.0
    else:
        counts = np.bincount(labels.reshape(-1), minlength=num_labels)
        component_sizes = counts[1:]
        component_count = int(component_sizes.size)
        largest_component_fraction = float(component_sizes.max() / float(area)) if component_sizes.size else 0.0

    background_input = (~binary).astype(np.uint8, copy=False)
    num_bg_labels, bg_labels = cv2.connectedComponents(background_input, connectivity=8)
    if num_bg_labels <= 1:
        hole_count = 0
        hole_area_fraction = 0.0
    else:
        border_labels = set()
        border_labels.update(int(value) for value in bg_labels[0, :].tolist())
        border_labels.update(int(value) for value in bg_labels[-1, :].tolist())
        border_labels.update(int(value) for value in bg_labels[:, 0].tolist())
        border_labels.update(int(value) for value in bg_labels[:, -1].tolist())
        hole_labels = [label for label in range(1, num_bg_labels) if label not in border_labels]
        if hole_labels:
            bg_counts = np.bincount(bg_labels.reshape(-1), minlength=num_bg_labels)
            hole_area = int(sum(int(bg_counts[label]) for label in hole_labels))
            hole_count = int(len(hole_labels))
            hole_area_fraction = float(hole_area / float(area))
        else:
            hole_count = 0
            hole_area_fraction = 0.0

    return component_count, largest_component_fraction, hole_count, hole_area_fraction


def _compute_component_topology_metrics(masks_roi: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks_roi, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(binary.shape)}")
    total = int(binary.shape[0])
    component_count = np.zeros((total,), dtype=np.int32)
    largest_component_fraction = np.zeros((total,), dtype=np.float32)
    hole_count = np.zeros((total,), dtype=np.int32)
    hole_area_fraction = np.zeros((total,), dtype=np.float32)
    for row_idx in range(total):
        comp_count, largest_frac, holes, hole_frac = _compute_single_mask_topology_metrics(binary[row_idx])
        component_count[row_idx] = comp_count
        largest_component_fraction[row_idx] = largest_frac
        hole_count[row_idx] = holes
        hole_area_fraction[row_idx] = hole_frac
    return {
        "component_count": component_count,
        "largest_component_fraction": largest_component_fraction,
        "hole_count": hole_count,
        "hole_area_fraction": hole_area_fraction,
    }


def _extract_largest_external_contour(mask: np.ndarray) -> np.ndarray | None:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if int(np.count_nonzero(binary)) <= 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = np.asarray(max(contours, key=cv2.contourArea), dtype=np.float32).reshape(-1, 1, 2)
    if contour.shape[0] < 3:
        return None
    return contour


def _extract_largest_external_contour_xy(mask: np.ndarray) -> np.ndarray | None:
    contour = _extract_largest_external_contour(mask)
    if contour is None:
        return None
    return np.asarray(contour, dtype=np.float32).reshape(-1, 2)


def _smooth_closed_contour(contour_xy: np.ndarray, *, smoothing_sigma: float) -> np.ndarray:
    sigma = max(float(smoothing_sigma), 0.0)
    if sigma == 0.0:
        return np.asarray(contour_xy, dtype=np.float32)
    xs = np.asarray(contour_xy[:, 0], dtype=np.float32)
    ys = np.asarray(contour_xy[:, 1], dtype=np.float32)
    xs_smooth = gaussian_filter1d(xs, sigma=sigma, mode="wrap")
    ys_smooth = gaussian_filter1d(ys, sigma=sigma, mode="wrap")
    return np.stack((xs_smooth, ys_smooth), axis=1).astype(np.float32, copy=False)


def _compute_single_mask_sigma_noise(mask: np.ndarray, *, smoothing_sigma: float = SIGMA_NOISE_SMOOTHING_SIGMA) -> float:
    contour_xy = _extract_largest_external_contour_xy(mask)
    if contour_xy is None:
        return 0.0

    contour_smooth = _smooth_closed_contour(contour_xy, smoothing_sigma=smoothing_sigma)
    diffs = np.asarray(contour_xy, dtype=np.float32) - np.asarray(contour_smooth, dtype=np.float32)
    distances_sq = np.sum(np.square(diffs, dtype=np.float32), axis=1, dtype=np.float32)
    return float(np.sqrt(np.mean(distances_sq, dtype=np.float32), dtype=np.float32))


def _compute_component_sigma_noise_metrics(masks_roi: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks_roi, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(binary.shape)}")
    total = int(binary.shape[0])
    sigma_noise = np.zeros((total,), dtype=np.float32)
    for row_idx in range(total):
        sigma_noise[row_idx] = _compute_single_mask_sigma_noise(binary[row_idx])
    return {"sigma_noise": sigma_noise}


def _compute_single_mask_curvature_var(
    mask: np.ndarray,
    *,
    smoothing_sigma: float = CURVATURE_VAR_SMOOTHING_SIGMA,
    step: int = CURVATURE_VAR_STEP,
) -> float:
    contour_xy = _extract_largest_external_contour_xy(mask)
    if contour_xy is None:
        return 0.0
    contour_smooth = _smooth_closed_contour(contour_xy, smoothing_sigma=smoothing_sigma)
    total = int(contour_smooth.shape[0])
    k = max(int(step), 1)
    if total < (2 * k + 1):
        return 0.0

    curvature = np.zeros((total,), dtype=np.float32)
    for idx in range(total):
        p_prev = contour_smooth[(idx - k) % total]
        p_curr = contour_smooth[idx]
        p_next = contour_smooth[(idx + k) % total]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        v3 = p_prev - p_next
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v3))
        if denom <= 0.0:
            curvature[idx] = 0.0
            continue
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        curvature[idx] = np.float32((2.0 * abs(cross)) / denom)
    return float(np.var(curvature, dtype=np.float32))


def _compute_component_curvature_var_metrics(masks_roi: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks_roi, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(binary.shape)}")
    total = int(binary.shape[0])
    curvature_var = np.zeros((total,), dtype=np.float32)
    for row_idx in range(total):
        curvature_var[row_idx] = _compute_single_mask_curvature_var(binary[row_idx])
    return {"curvature_var": curvature_var}


def _compute_single_mask_ipr(mask: np.ndarray) -> float:
    contour = _extract_largest_external_contour(mask)
    if contour is None:
        return 0.0
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return 0.0
    perimeter = float(cv2.arcLength(contour, True))
    return float((perimeter * perimeter) / (4.0 * float(np.pi) * area))


def _compute_single_mask_solidity(mask: np.ndarray) -> float:
    contour = _extract_largest_external_contour(mask)
    if contour is None:
        return 0.0
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return 0.0
    return float(area / hull_area)


def _compute_component_shape_qc_metrics(masks_roi: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks_roi, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(binary.shape)}")
    total = int(binary.shape[0])
    ipr = np.zeros((total,), dtype=np.float32)
    solidity = np.zeros((total,), dtype=np.float32)
    for row_idx in range(total):
        ipr[row_idx] = _compute_single_mask_ipr(binary[row_idx])
        solidity[row_idx] = _compute_single_mask_solidity(binary[row_idx])
    return {"ipr": ipr, "solidity": solidity}


def _source_mask_for_component(
    source: SourceSubjectMaskRun,
    component_name: str,
    roi_idx: int,
) -> np.ndarray:
    if component_name not in source.mask_labels:
        h = int(source.masks_roi.shape[2])
        w = int(source.masks_roi.shape[3])
        return np.zeros((h, w), dtype=np.uint8)
    channel_idx = source.mask_labels.index(component_name)
    if channel_idx >= int(source.available_channels.shape[0]) or not bool(source.available_channels[channel_idx]):
        h = int(source.masks_roi.shape[2])
        w = int(source.masks_roi.shape[3])
        return np.zeros((h, w), dtype=np.uint8)
    return np.asarray(source.masks_roi[roi_idx, channel_idx], dtype=np.uint8)


def _source_mask_for_resolved_component(
    component_sources: Mapping[str, SourceSubjectMaskRun],
    component_name: str,
    roi_idx: int,
) -> np.ndarray:
    if component_name not in component_sources:
        raise RuntimeError(f"No resolved source available for component {component_name!r}.")
    return _source_mask_for_component(component_sources[component_name], component_name, roi_idx)


def _default_reason_label(source_mask: np.ndarray, current_mask: np.ndarray, edit_applied: bool) -> str:
    if edit_applied:
        return "manual_correction"
    if int(np.count_nonzero(source_mask)) > 0 and np.array_equal(source_mask, current_mask):
        return "copied_from_source"
    return "clean"


def _load_or_init_reason_labels(component_group: zarr.Group, total_rois: int) -> np.ndarray:
    labels = read_reason_labels(component_group)
    if labels is None or int(labels.shape[0]) != int(total_rois):
        labels = np.full((int(total_rois),), "clean", dtype=object)
        write_reason_columns(
            component_group,
            labels,
            chunk_size=max(1, min(256, int(total_rois))),
            include_reason_text=True,
            overwrite=True,
        )
        return labels
    return np.asarray(labels, dtype=object)


def _write_reason_label_row(
    component_group: zarr.Group,
    reason_labels: np.ndarray,
    row_idx: int,
) -> None:
    label = str(reason_labels[int(row_idx)])
    reason_arr = component_group.get("reason")
    if reason_arr is not None:
        reason_arr[int(row_idx):int(row_idx) + 1] = np.asarray([label], dtype=object)

    reason_bytes_arr = component_group.get("reason_bytes")
    existing_width = 0
    if reason_bytes_arr is not None and hasattr(reason_bytes_arr, "shape"):
        existing_width = int(reason_bytes_arr.shape[1])
    encoded = encode_reason_bytes(np.asarray([label], dtype=object), min_width=max(64, existing_width))
    if reason_bytes_arr is None or int(encoded.shape[1]) > existing_width:
        write_reason_columns(
            component_group,
            reason_labels,
            chunk_size=max(1, min(256, int(reason_labels.shape[0]))),
            include_reason_text=True,
            overwrite=True,
        )
        return
    reason_bytes_arr[int(row_idx):int(row_idx) + 1] = encoded


def _ensure_component_group(
    refined: zarr.Group,
    component_name: str,
    total_rois: int,
    mask_present: np.ndarray,
    area_px: np.ndarray,
    edit_applied: np.ndarray,
    component_metrics: Optional[Mapping[str, np.ndarray]] = None,
) -> zarr.Group:
    components_parent = refined.require_group("components")
    component_group = components_parent.require_group(component_name)
    if "mask_present" not in component_group:
        component_group.create_array("mask_present", data=np.asarray(mask_present, dtype=bool), overwrite=True)
    if "area_px" not in component_group:
        component_group.create_array("area_px", data=np.asarray(area_px, dtype=np.float32), overwrite=True)
    if "edit_applied" not in component_group:
        component_group.create_array("edit_applied", data=np.asarray(edit_applied, dtype=bool), overwrite=True)
    if component_name == "subject_body":
        component_group.attrs.setdefault("component_schema_id", DEFAULT_SUBJECT_BODY_COMPONENT_SCHEMA_ID)
        component_group.attrs.setdefault("anatomical_scope", DEFAULT_SUBJECT_BODY_ANATOMICAL_SCOPE)
        component_group.attrs.setdefault("pectoral_fin_policy", DEFAULT_SUBJECT_BODY_PECTORAL_FIN_POLICY)
    if component_metrics is not None:
        metrics_group = component_group.require_group("metrics")
        metric_specs = (
            ("component_count", np.asarray(component_metrics["component_count"], dtype=np.int32)),
            (
                "largest_component_fraction",
                np.asarray(component_metrics["largest_component_fraction"], dtype=np.float32),
            ),
            ("hole_count", np.asarray(component_metrics["hole_count"], dtype=np.int32)),
            ("hole_area_fraction", np.asarray(component_metrics["hole_area_fraction"], dtype=np.float32)),
            ("sigma_noise", np.asarray(component_metrics["sigma_noise"], dtype=np.float32)),
            ("curvature_var", np.asarray(component_metrics["curvature_var"], dtype=np.float32)),
            ("ipr", np.asarray(component_metrics["ipr"], dtype=np.float32)),
            ("solidity", np.asarray(component_metrics["solidity"], dtype=np.float32)),
        )
        for metric_name, metric_values in metric_specs:
            if metric_name not in metrics_group:
                metrics_group.create_array(metric_name, data=metric_values, overwrite=True)
    _load_or_init_reason_labels(component_group, total_rois)
    return component_group


def _load_source_subject_mask_run(root: zarr.Group, subject_run: Optional[str]) -> SourceSubjectMaskRun:
    parent = root.get("subject_mask_runs")
    if parent is None:
        raise RuntimeError("No subject_mask_runs found in archive.")
    run_name = subject_run or _resolve_latest_run(parent, "subject_mask_runs")
    if run_name not in parent:
        raise RuntimeError(f"subject_mask_runs/{run_name} not found.")
    group = parent[run_name]
    if "masks_roi" not in group:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing masks_roi.")
    crop_run = str(group.attrs.get("source_crop_run") or "")
    if not crop_run:
        crop_parent = root.get("crop_runs")
        crop_run = _resolve_latest_run(crop_parent, "crop_runs")
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing usable mask_labels attr.")
    available = group.get("available_channels")
    if available is None:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing available_channels.")
    return SourceSubjectMaskRun(
        run_name=run_name,
        group=group,
        crop_run=crop_run,
        masks_roi=group["masks_roi"],
        detection_source=group["detection_source"],
        mask_labels=tuple(str(item) for item in labels_raw),
        available_channels=np.asarray(available[:], dtype=bool),
        frame_indices=group.get("frame_indices"),
        frame_counts=group.get("frame_counts"),
        detection_indices=group.get("detection_indices"),
        source_method=str(group.attrs.get("method")) if group.attrs.get("method") is not None else None,
        source_keypoints_run=resolve_source_keypoints_run(group.attrs),
        source_keypoint_group=(
            str(group.attrs.get("source_keypoint_group")) if group.attrs.get("source_keypoint_group") is not None else None
        ),
    )


def _resolve_primary_source_subject_mask_run_name(refined_group: zarr.Group) -> Optional[str]:
    direct = refined_group.attrs.get("source_subject_mask_run")
    if direct is not None and str(direct).strip():
        return str(direct)
    labels_raw = refined_group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return None
    for component_name in (str(item) for item in labels_raw):
        provenance_group = _get_component_provenance_group(refined_group, component_name)
        if provenance_group is None:
            continue
        source_stage = str(provenance_group.attrs.get("source_stage") or "")
        source_run = str(provenance_group.attrs.get("source_run") or "")
        if source_stage == "subject_mask_runs" and source_run:
            return source_run
    return None


def _load_refined_component_source_runs(
    root: zarr.Group,
    refined: RefinedSubjectMaskRun,
    *,
    default_source: Optional[SourceSubjectMaskRun] = None,
) -> tuple[SourceSubjectMaskRun, dict[str, SourceSubjectMaskRun]]:
    cache: dict[str, SourceSubjectMaskRun] = {}
    if default_source is not None:
        cache[default_source.run_name] = default_source

    primary_source = default_source
    primary_run = _resolve_primary_source_subject_mask_run_name(refined.group)
    if primary_source is None and primary_run:
        primary_source = cache.setdefault(primary_run, _load_source_subject_mask_run(root, primary_run))

    component_sources: dict[str, SourceSubjectMaskRun] = {}
    for component_name in refined.component_names:
        provenance_group = _get_component_provenance_group(refined.group, component_name)
        source_stage = str(provenance_group.attrs.get("source_stage") or "") if provenance_group is not None else ""
        source_run = str(provenance_group.attrs.get("source_run") or "") if provenance_group is not None else ""
        if source_run and source_stage and source_stage != "subject_mask_runs":
            raise RuntimeError(
                "Component source resolution currently supports only subject_mask_runs-backed refined components; "
                f"{component_name!r} is sourced from {source_stage}/{source_run}."
            )
        resolved_source: SourceSubjectMaskRun | None = None
        if source_run:
            resolved_source = cache.setdefault(source_run, _load_source_subject_mask_run(root, source_run))
        elif default_source is not None:
            resolved_source = default_source
        elif primary_source is not None:
            resolved_source = primary_source
        if resolved_source is None:
            raise RuntimeError(
                f"Unable to resolve source subject-mask lineage for component {component_name!r} "
                f"in refined_subject_masks_runs/{refined.run_name}."
            )
        component_sources[component_name] = resolved_source
        if primary_source is None:
            primary_source = resolved_source

    if primary_source is None:
        raise RuntimeError(f"Unable to resolve any source subject-mask run for refined_subject_masks_runs/{refined.run_name}.")
    return primary_source, component_sources


def _create_refined_subject_run_from_component_seeds(
    *,
    refined_parent: zarr.Group,
    target_run: str,
    reference_source: SourceSubjectMaskRun,
    component_names: Sequence[str],
    component_seeds: Mapping[str, RefinedSubjectComponentSeed],
    coarse_source_subject_mask_run: Optional[str],
    coarse_source_subject_mask_method: Optional[str],
    source_keypoints_run: Optional[str],
    source_keypoint_group: Optional[str],
    run_method: str = DEFAULT_RUN_METHOD,
    stage_command: Optional[str] = None,
    extra_attrs: Optional[Mapping[str, object]] = None,
    provenance_inputs: Optional[Mapping[str, object]] = None,
) -> RefinedSubjectMaskRun:
    total_rois = int(reference_source.masks_roi.shape[0])
    height = int(reference_source.masks_roi.shape[2])
    width = int(reference_source.masks_roi.shape[3])
    component_names = tuple(str(name) for name in component_names)
    masks = np.zeros((total_rois, len(component_names), height, width), dtype=np.uint8)
    for comp_idx, component_name in enumerate(component_names):
        if component_name not in component_seeds:
            raise RuntimeError(f"Missing component seed for {component_name!r}.")
        seed_masks = np.asarray(component_seeds[component_name].masks, dtype=np.uint8)
        expected_shape = (total_rois, height, width)
        if seed_masks.shape != expected_shape:
            raise ValueError(
                f"Component seed shape mismatch for {component_name!r}: expected {expected_shape}, got {tuple(seed_masks.shape)}"
            )
        masks[:, comp_idx] = seed_masks

    mask_present, area_px = _compute_mask_metrics(masks)
    edit_applied = np.zeros((total_rois, len(component_names)), dtype=bool)

    run_group = refined_parent.create_group(target_run)
    run_group.create_array(
        "detection_source",
        data=np.asarray(reference_source.detection_source[:], dtype=np.int8),
        overwrite=True,
    )
    run_group.create_array("masks_roi", data=masks, overwrite=True)
    run_group.create_array(
        "available_channels",
        data=np.ones((len(component_names),), dtype=bool),
        overwrite=True,
    )
    run_group.create_array("edit_applied", data=edit_applied, overwrite=True)
    _copy_optional_array(run_group, reference_source.group, "frame_indices")
    _copy_optional_array(run_group, reference_source.group, "frame_counts")
    _copy_optional_array(run_group, reference_source.group, "detection_indices")

    metrics = run_group.require_group("metrics")
    metrics.create_array("mask_present", data=mask_present, overwrite=True)
    metrics.create_array("area_px", data=area_px, overwrite=True)
    geometry_metrics = _compute_geometry_metrics(masks)
    metrics.create_array("centroid_xy", data=geometry_metrics["centroid_xy"], overwrite=True)
    metrics.create_array("centroid_valid", data=geometry_metrics["centroid_valid"], overwrite=True)
    metrics.create_array("bbox_xyxy", data=geometry_metrics["bbox_xyxy"], overwrite=True)
    metrics.create_array("bbox_valid", data=geometry_metrics["bbox_valid"], overwrite=True)
    created = _utc_now()

    for comp_idx, component_name in enumerate(component_names):
        seed = component_seeds[component_name]
        component_masks = np.asarray(seed.masks, dtype=np.uint8)
        component_metrics = _compute_component_topology_metrics(component_masks)
        component_metrics.update(_compute_component_sigma_noise_metrics(component_masks))
        component_metrics.update(_compute_component_curvature_var_metrics(component_masks))
        component_metrics.update(_compute_component_shape_qc_metrics(component_masks))
        component_group = _ensure_component_group(
            run_group,
            component_name,
            total_rois,
            mask_present[:, comp_idx],
            area_px[:, comp_idx],
            edit_applied[:, comp_idx],
            component_metrics=component_metrics,
        )
        reason_labels = _load_or_init_reason_labels(component_group, total_rois)
        for row_idx in range(total_rois):
            source_mask = np.asarray(component_masks[row_idx], dtype=np.uint8)
            reason_labels[row_idx] = _default_reason_label(source_mask, masks[row_idx, comp_idx], edit_applied=False)
        write_reason_columns(
            component_group,
            reason_labels,
            chunk_size=max(1, min(256, total_rois)),
            include_reason_text=True,
            overwrite=True,
        )
        _ensure_refined_component_provenance_payload(
            run_group,
            component_name=component_name,
            source_payload=seed.source_payload,
            created_at_utc=created,
        )

    if coarse_source_subject_mask_run:
        run_group.attrs["source_subject_mask_run"] = coarse_source_subject_mask_run
    if coarse_source_subject_mask_method:
        run_group.attrs["source_subject_mask_method"] = coarse_source_subject_mask_method
    if source_keypoints_run:
        run_group.attrs.update(build_source_keypoints_attrs(source_keypoints_run, include_legacy_alias=True))
    if source_keypoint_group:
        run_group.attrs["source_keypoint_group"] = source_keypoint_group
    run_group.attrs["source_crop_run"] = reference_source.crop_run
    run_group.attrs["mask_labels"] = list(component_names)
    run_group.attrs["label_schema_id"] = _infer_refined_label_schema_id(component_names)
    run_group.attrs["output_semantics"] = "multilabel"
    run_group.attrs["refinement_semantics"] = "canonical_component_masks"
    run_group.attrs["method"] = str(run_method)
    run_group.attrs["created_at_utc"] = created
    run_group.attrs["created_utc"] = created
    run_group.attrs["duration_seconds"] = 0.0
    if extra_attrs:
        for key, value in extra_attrs.items():
            run_group.attrs[str(key)] = value

    component_reviews = {
        component_name: _review_payload(
            state="pending",
            method=DEFAULT_REVIEW_METHOD,
            intended_use=DEFAULT_REVIEW_INTENDED_USE,
        )
        for component_name in component_names
    }
    run_group.attrs["component_review_statuses"] = component_reviews
    run_group.attrs["refined_subject_mask_review_status"] = _review_payload(
        state="pending",
        method=DEFAULT_REVIEW_METHOD,
        intended_use=DEFAULT_REVIEW_INTENDED_USE,
        notes="auto_initialized_from_components",
    )
    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    stage_inputs_payload = {
        "source_subject_mask_run": coarse_source_subject_mask_run,
        "source_subject_mask_method": coarse_source_subject_mask_method,
        "source_crop_run": reference_source.crop_run,
        "source_keypoints_run": source_keypoints_run,
        "source_keypoint_group": source_keypoint_group,
    }
    if provenance_inputs:
        stage_inputs_payload.update({str(key): value for key, value in provenance_inputs.items()})
    provenance = build_stage_provenance(
        stage=REFINED_SUBJECT_STAGE_NAME,
        command=stage_command or (" ".join(sys.argv) if sys.argv else "unknown"),
        created_at_utc=created,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        parameters={
            "method": str(run_method),
            "refinement_semantics": "canonical_component_masks",
            "components": list(component_names),
            "component_count": int(len(component_names)),
        },
        inputs=stage_inputs_payload,
    )
    write_stage_provenance(run_group, provenance)
    refined_parent.attrs["latest"] = target_run
    refined_parent.attrs["refined_subject_mask_review_status_latest"] = target_run
    return RefinedSubjectMaskRun(
        run_name=target_run,
        parent=refined_parent,
        group=run_group,
        component_names=component_names,
        component_to_index={name: idx for idx, name in enumerate(component_names)},
    )


def _open_or_create_refined_subject_run(
    root: zarr.Group,
    *,
    source: SourceSubjectMaskRun,
    refined_run: Optional[str],
    components: Sequence[str],
) -> RefinedSubjectMaskRun:
    refined_parent = root.require_group("refined_subject_masks_runs")
    target_run = refined_run
    if target_run is None:
        latest = refined_parent.attrs.get("latest")
        if latest and str(latest) in refined_parent:
            candidate = refined_parent[str(latest)]
            if str(candidate.attrs.get("source_subject_mask_run") or "") == source.run_name:
                target_run = str(latest)
        if target_run is None:
            target_run = _default_refined_run_name()

    if target_run in refined_parent:
        run_group = refined_parent[target_run]
        labels_raw = run_group.attrs.get("mask_labels")
        if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
            raise RuntimeError(f"refined_subject_masks_runs/{target_run} missing usable mask_labels attr.")
        component_names = tuple(str(item) for item in labels_raw)
        component_to_index = {name: idx for idx, name in enumerate(component_names)}
        masks_arr = run_group.get("masks_roi")
        if masks_arr is None:
            raise RuntimeError(f"refined_subject_masks_runs/{target_run} missing masks_roi.")
        total_rois = int(masks_arr.shape[0])
        metrics = run_group.require_group("metrics")
        masks_np = np.asarray(masks_arr[:], dtype=np.uint8)
        geometry_metrics = None
        if "mask_present" not in metrics or "area_px" not in metrics:
            mask_present, area_px = _compute_mask_metrics(masks_np)
            if "mask_present" not in metrics:
                metrics.create_array("mask_present", data=mask_present, overwrite=True)
            if "area_px" not in metrics:
                metrics.create_array("area_px", data=area_px, overwrite=True)
        if (
            "centroid_xy" not in metrics
            or "centroid_valid" not in metrics
            or "bbox_xyxy" not in metrics
            or "bbox_valid" not in metrics
        ):
            geometry_metrics = _compute_geometry_metrics(masks_np)
            if "centroid_xy" not in metrics:
                metrics.create_array("centroid_xy", data=geometry_metrics["centroid_xy"], overwrite=True)
            if "centroid_valid" not in metrics:
                metrics.create_array("centroid_valid", data=geometry_metrics["centroid_valid"], overwrite=True)
            if "bbox_xyxy" not in metrics:
                metrics.create_array("bbox_xyxy", data=geometry_metrics["bbox_xyxy"], overwrite=True)
            if "bbox_valid" not in metrics:
                metrics.create_array("bbox_valid", data=geometry_metrics["bbox_valid"], overwrite=True)
        if "edit_applied" not in run_group:
            run_group.create_array(
                "edit_applied",
                data=np.zeros((total_rois, len(component_names)), dtype=bool),
                overwrite=True,
            )
        mask_present_arr = np.asarray(metrics["mask_present"][:], dtype=bool)
        area_px_arr = np.asarray(metrics["area_px"][:], dtype=np.float32)
        edit_applied_arr = np.asarray(run_group["edit_applied"][:], dtype=bool)
        for comp_idx, component_name in enumerate(component_names):
            component_masks = np.asarray(masks_np[:, comp_idx], dtype=np.uint8)
            component_metrics = _compute_component_topology_metrics(component_masks)
            component_metrics.update(_compute_component_sigma_noise_metrics(component_masks))
            component_metrics.update(_compute_component_curvature_var_metrics(component_masks))
            component_metrics.update(_compute_component_shape_qc_metrics(component_masks))
            _ensure_component_group(
                run_group,
                component_name,
                total_rois,
                mask_present_arr[:, comp_idx],
                area_px_arr[:, comp_idx],
                edit_applied_arr[:, comp_idx],
                component_metrics=component_metrics,
            )
            _ensure_refined_component_provenance(
                run_group,
                source=source,
                component_name=component_name,
                created_at_utc=str(run_group.attrs.get("created_at_utc") or _utc_now()),
            )
        return RefinedSubjectMaskRun(
            run_name=target_run,
            parent=refined_parent,
            group=run_group,
            component_names=component_names,
            component_to_index=component_to_index,
        )

    component_names = tuple(components)
    component_seeds = _build_single_source_component_seeds(source, component_names)
    return _create_refined_subject_run_from_component_seeds(
        refined_parent=refined_parent,
        target_run=target_run,
        reference_source=source,
        component_names=component_names,
        component_seeds=component_seeds,
        coarse_source_subject_mask_run=source.run_name,
        coarse_source_subject_mask_method=source.source_method,
        source_keypoints_run=source.source_keypoints_run,
        source_keypoint_group=source.source_keypoint_group,
    )


def prepare_refined_subject_run(
    root: zarr.Group,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
) -> tuple[SourceSubjectMaskRun, RefinedSubjectMaskRun]:
    resolved_subject_run = subject_run
    if resolved_subject_run is None and refined_run is not None:
        refined_parent = root.get("refined_subject_masks_runs")
        if refined_parent is not None and str(refined_run) in refined_parent:
            resolved_subject_run = _resolve_primary_source_subject_mask_run_name(refined_parent[str(refined_run)])
    source = _load_source_subject_mask_run(root, resolved_subject_run)
    normalized_components = _normalize_component_list(components)
    refined = _open_or_create_refined_subject_run(
        root,
        source=source,
        refined_run=refined_run,
        components=normalized_components,
    )
    return source, refined


def _open_existing_refined_subject_run(
    root: zarr.Group,
    refined_run: str,
) -> RefinedSubjectMaskRun:
    refined_parent = root.get("refined_subject_masks_runs")
    if refined_parent is None or str(refined_run) not in refined_parent:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} not found.")

    run_group = refined_parent[str(refined_run)]
    labels_raw = run_group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} missing usable mask_labels attr.")
    if "masks_roi" not in run_group:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} missing masks_roi.")
    component_names = tuple(str(item) for item in labels_raw)
    return RefinedSubjectMaskRun(
        run_name=str(refined_run),
        parent=refined_parent,
        group=run_group,
        component_names=component_names,
        component_to_index={name: idx for idx, name in enumerate(component_names)},
    )


def _aggregate_run_state(component_reviews: Mapping[str, Mapping[str, object]]) -> str:
    states = [str(payload.get("state") or "pending") for payload in component_reviews.values()]
    if not states:
        return "pending"
    if any(state == "rejected" for state in states):
        return "rejected"
    if any(state == "needs_review" for state in states):
        return "needs_review"
    if all(state == "approved" for state in states):
        return "approved"
    if any(state == "pending" for state in states):
        return "pending"
    return states[0]


def apply_component_review_status(
    refined_parent: zarr.Group,
    refined_run: str,
    refined: zarr.Group,
    *,
    component_name: str,
    state: str,
    method: str,
    intended_use: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> tuple[Dict[str, object], Dict[str, object]]:
    component_reviews = dict(refined.attrs.get("component_review_statuses") or {})
    component_reviews[str(component_name)] = _review_payload(
        state=state,
        method=method,
        intended_use=intended_use,
        reviewer=reviewer,
        notes=notes,
    )
    refined.attrs["component_review_statuses"] = component_reviews

    run_state = _aggregate_run_state(component_reviews)
    run_payload = _review_payload(
        state=run_state,
        method=method,
        intended_use=intended_use,
        reviewer=reviewer,
        notes="auto_aggregated_from_component_review_statuses",
    )
    refined.attrs["refined_subject_mask_review_status"] = run_payload
    refined_parent.attrs["refined_subject_mask_review_status_latest"] = refined_run
    return component_reviews[str(component_name)], run_payload


def _normalize_refined_component_names(
    refined: RefinedSubjectMaskRun,
    component_names: Optional[Sequence[str]],
) -> tuple[str, ...]:
    if component_names is None:
        return tuple(refined.component_names)

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in component_names:
        component_name = _normalize_component_name(raw_name)
        if component_name is None:
            continue
        if component_name not in refined.component_to_index:
            raise ValueError(
                f"Component '{component_name}' not available in refined_subject_masks_runs/{refined.run_name}."
            )
        if component_name in seen:
            continue
        seen.add(component_name)
        normalized.append(component_name)

    if not normalized:
        raise ValueError("At least one refined subject-mask component is required.")
    return tuple(normalized)


def _normalize_refined_edited_masks_batch(
    refined: RefinedSubjectMaskRun,
    roi_indices: Sequence[int],
    edited_masks_batch: np.ndarray,
) -> np.ndarray:
    expected_row_shape = tuple(int(dim) for dim in refined.group["masks_roi"].shape[1:])
    edited = np.asarray(edited_masks_batch, dtype=np.uint8)
    if edited.shape == expected_row_shape:
        if len(roi_indices) != 1:
            raise ValueError(
                "edited_masks_batch must include one mask stack per ROI row when multiple roi_indices are provided."
            )
        edited = edited[np.newaxis, ...]

    expected_batch_shape = (len(roi_indices),) + expected_row_shape
    if edited.shape != expected_batch_shape:
        raise ValueError(
            f"edited_masks_batch shape mismatch: expected {expected_batch_shape}, got {tuple(edited.shape)}"
        )
    return edited


def _compute_refined_subject_component_apply_rows(
    *,
    component_sources: Optional[Mapping[str, SourceSubjectMaskRun]] = None,
    source: Optional[SourceSubjectMaskRun] = None,
    refined: RefinedSubjectMaskRun,
    component_name: str,
    roi_indices: Sequence[int],
    edited_masks_batch: np.ndarray,
) -> dict[str, np.ndarray]:
    if component_sources is None:
        if source is None:
            raise RuntimeError("component_sources or source is required.")
        component_sources = {name: source for name in refined.component_names}
    comp_idx = int(refined.component_to_index[component_name])
    row_count = int(len(roi_indices))
    mask_present = np.zeros((row_count,), dtype=bool)
    area_px = np.zeros((row_count,), dtype=np.float32)
    edit_applied = np.zeros((row_count,), dtype=bool)
    centroid_xy = np.zeros((row_count, 2), dtype=np.float32)
    centroid_valid = np.zeros((row_count,), dtype=bool)
    bbox_xyxy = np.zeros((row_count, 4), dtype=np.float32)
    bbox_valid = np.zeros((row_count,), dtype=bool)
    component_count = np.zeros((row_count,), dtype=np.int32)
    largest_component_fraction = np.zeros((row_count,), dtype=np.float32)
    hole_count = np.zeros((row_count,), dtype=np.int32)
    hole_area_fraction = np.zeros((row_count,), dtype=np.float32)
    sigma_noise = np.zeros((row_count,), dtype=np.float32)
    curvature_var = np.zeros((row_count,), dtype=np.float32)
    ipr = np.zeros((row_count,), dtype=np.float32)
    solidity = np.zeros((row_count,), dtype=np.float32)
    reason_labels = np.empty((row_count,), dtype=object)

    for row_offset, roi_idx in enumerate(roi_indices):
        current_mask = np.asarray(edited_masks_batch[row_offset, comp_idx], dtype=np.uint8)
        source_mask = _source_mask_for_resolved_component(component_sources, component_name, int(roi_idx))
        area = float(np.count_nonzero(current_mask))
        present = bool(area > 0.0)
        edited = not np.array_equal(current_mask, source_mask)
        centroid, centroid_is_valid, bbox, bbox_is_valid = _compute_single_mask_geometry_metrics(current_mask)
        comp_count, largest_frac, holes, hole_frac = _compute_single_mask_topology_metrics(current_mask)

        mask_present[row_offset] = present
        area_px[row_offset] = np.float32(area)
        edit_applied[row_offset] = edited
        centroid_xy[row_offset] = centroid
        centroid_valid[row_offset] = centroid_is_valid
        bbox_xyxy[row_offset] = bbox
        bbox_valid[row_offset] = bbox_is_valid
        component_count[row_offset] = np.int32(comp_count)
        largest_component_fraction[row_offset] = np.float32(largest_frac)
        hole_count[row_offset] = np.int32(holes)
        hole_area_fraction[row_offset] = np.float32(hole_frac)
        sigma_noise[row_offset] = np.float32(_compute_single_mask_sigma_noise(current_mask))
        curvature_var[row_offset] = np.float32(_compute_single_mask_curvature_var(current_mask))
        ipr[row_offset] = np.float32(_compute_single_mask_ipr(current_mask))
        solidity[row_offset] = np.float32(_compute_single_mask_solidity(current_mask))
        reason_labels[row_offset] = _default_reason_label(source_mask, current_mask, edited)

    return {
        "mask_present": mask_present,
        "area_px": area_px,
        "edit_applied": edit_applied,
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
        "component_count": component_count,
        "largest_component_fraction": largest_component_fraction,
        "hole_count": hole_count,
        "hole_area_fraction": hole_area_fraction,
        "sigma_noise": sigma_noise,
        "curvature_var": curvature_var,
        "ipr": ipr,
        "solidity": solidity,
        "reason_labels": reason_labels,
    }


def _write_refined_subject_component_apply_rows(
    *,
    refined: RefinedSubjectMaskRun,
    component_name: str,
    roi_indices: Sequence[int],
    edited_masks_batch: np.ndarray,
    component_updates: Mapping[str, np.ndarray],
) -> None:
    run_group = refined.group
    comp_idx = int(refined.component_to_index[component_name])
    masks_arr = run_group["masks_roi"]
    metrics = run_group["metrics"]
    edit_arr = run_group["edit_applied"]
    metrics_mask_present = metrics["mask_present"]
    metrics_area_px = metrics["area_px"]
    metrics_centroid_xy = metrics["centroid_xy"]
    metrics_centroid_valid = metrics["centroid_valid"]
    metrics_bbox_xyxy = metrics["bbox_xyxy"]
    metrics_bbox_valid = metrics["bbox_valid"]
    component_group = run_group.require_group("components").require_group(component_name)
    component_metrics = component_group.require_group("metrics")
    reason_labels = _load_or_init_reason_labels(component_group, int(run_group["masks_roi"].shape[0]))

    for row_offset, roi_idx in enumerate(roi_indices):
        masks_arr[int(roi_idx), comp_idx] = np.asarray(edited_masks_batch[row_offset, comp_idx], dtype=np.uint8)
        edit_arr[int(roi_idx), comp_idx] = bool(component_updates["edit_applied"][row_offset])
        metrics_mask_present[int(roi_idx), comp_idx] = bool(component_updates["mask_present"][row_offset])
        metrics_area_px[int(roi_idx), comp_idx] = np.float32(component_updates["area_px"][row_offset])
        metrics_centroid_xy[int(roi_idx), comp_idx] = np.asarray(
            component_updates["centroid_xy"][row_offset],
            dtype=np.float32,
        )
        metrics_centroid_valid[int(roi_idx), comp_idx] = bool(component_updates["centroid_valid"][row_offset])
        metrics_bbox_xyxy[int(roi_idx), comp_idx] = np.asarray(
            component_updates["bbox_xyxy"][row_offset],
            dtype=np.float32,
        )
        metrics_bbox_valid[int(roi_idx), comp_idx] = bool(component_updates["bbox_valid"][row_offset])

        component_group["mask_present"][int(roi_idx)] = bool(component_updates["mask_present"][row_offset])
        component_group["area_px"][int(roi_idx)] = np.float32(component_updates["area_px"][row_offset])
        component_group["edit_applied"][int(roi_idx)] = bool(component_updates["edit_applied"][row_offset])
        component_metrics["component_count"][int(roi_idx)] = np.int32(component_updates["component_count"][row_offset])
        component_metrics["largest_component_fraction"][int(roi_idx)] = np.float32(
            component_updates["largest_component_fraction"][row_offset]
        )
        component_metrics["hole_count"][int(roi_idx)] = np.int32(component_updates["hole_count"][row_offset])
        component_metrics["hole_area_fraction"][int(roi_idx)] = np.float32(
            component_updates["hole_area_fraction"][row_offset]
        )
        component_metrics["sigma_noise"][int(roi_idx)] = np.float32(component_updates["sigma_noise"][row_offset])
        component_metrics["curvature_var"][int(roi_idx)] = np.float32(component_updates["curvature_var"][row_offset])
        component_metrics["ipr"][int(roi_idx)] = np.float32(component_updates["ipr"][row_offset])
        component_metrics["solidity"][int(roi_idx)] = np.float32(component_updates["solidity"][row_offset])
        reason_labels[int(roi_idx)] = str(component_updates["reason_labels"][row_offset])
        _write_reason_label_row(component_group, reason_labels, int(roi_idx))


def _finalize_refined_subject_apply(refined: RefinedSubjectMaskRun) -> str:
    updated_at_utc = _utc_now()
    refined.group.attrs["updated_at_utc"] = updated_at_utc
    refined.parent.attrs["latest"] = refined.run_name
    return updated_at_utc


def _apply_refined_subject_roi_rows(
    *,
    component_sources: Optional[Mapping[str, SourceSubjectMaskRun]] = None,
    source: Optional[SourceSubjectMaskRun] = None,
    refined: RefinedSubjectMaskRun,
    roi_indices: Sequence[int],
    edited_masks_batch: np.ndarray,
    component_names: Optional[Sequence[str]] = None,
    update_mode: str = "interactive",
    update_method: Optional[str] = None,
) -> tuple[int, ...]:
    if component_sources is None:
        if source is None:
            raise RuntimeError("component_sources or source is required.")
        component_sources = {name: source for name in refined.component_names}
    run_group = refined.group
    total_rois = int(run_group["masks_roi"].shape[0])
    normalized_rows = tuple(_normalize_roi_indices(roi_indices, total_rois))
    normalized_components = _normalize_refined_component_names(refined, component_names)
    edited_batch = _normalize_refined_edited_masks_batch(refined, normalized_rows, edited_masks_batch)
    before_states = {
        component_name: _capture_component_sync_states(
            run_group,
            component_name,
            normalized_rows,
            comp_idx=int(refined.component_to_index[component_name]),
        )
        for component_name in normalized_components
    }

    for component_name in normalized_components:
        component_updates = _compute_refined_subject_component_apply_rows(
            component_sources=component_sources,
            source=source,
            refined=refined,
            component_name=component_name,
            roi_indices=normalized_rows,
            edited_masks_batch=edited_batch,
        )
        _write_refined_subject_component_apply_rows(
            refined=refined,
            component_name=component_name,
            roi_indices=normalized_rows,
            edited_masks_batch=edited_batch,
            component_updates=component_updates,
        )

    updated_at_utc = _finalize_refined_subject_apply(refined)
    resolved_method = str(update_method or DEFAULT_RUN_METHOD)
    for component_name in normalized_components:
        comp_idx = int(refined.component_to_index[component_name])
        if _component_sync_states_changed(
            run_group,
            component_name,
            normalized_rows,
            comp_idx=comp_idx,
            before_states=before_states[component_name],
        ):
            _write_refined_component_last_update(
                run_group,
                component_name=component_name,
                updated_at_utc=updated_at_utc,
                update_mode=update_mode,
                update_method=resolved_method,
            )
    return normalized_rows


def save_refined_subject_roi(
    *,
    source: SourceSubjectMaskRun,
    refined: RefinedSubjectMaskRun,
    roi_idx: int,
    edited_masks: np.ndarray,
    component_sources: Optional[Mapping[str, SourceSubjectMaskRun]] = None,
) -> None:
    _apply_refined_subject_roi_rows(
        component_sources=(
            dict(component_sources)
            if component_sources is not None
            else {component_name: source for component_name in refined.component_names}
        ),
        refined=refined,
        roi_indices=[int(roi_idx)],
        edited_masks_batch=edited_masks,
        update_mode="interactive",
        update_method=DEFAULT_RUN_METHOD,
    )


def _normalize_roi_indices(roi_indices: Sequence[int], total_rois: int) -> list[int]:
    if total_rois <= 0:
        raise RuntimeError("refined_subject_masks run has no ROI rows.")
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_idx in roi_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= total_rois:
            raise ValueError(f"roi index {idx} is out of bounds for run with {total_rois} rows.")
        if idx in seen:
            continue
        seen.add(idx)
        normalized.append(idx)
    if not normalized:
        raise ValueError("At least one ROI index is required.")
    return normalized


def _component_sync_state(
    run_group: zarr.Group,
    component_group: zarr.Group,
    *,
    comp_idx: int,
    roi_idx: int,
) -> tuple[bool, float, bool, str]:
    reason_labels = _load_or_init_reason_labels(component_group, int(run_group["masks_roi"].shape[0]))
    component_metrics = component_group.require_group("metrics")
    return (
        bool(np.asarray(run_group["metrics/mask_present"][int(roi_idx), comp_idx], dtype=bool)),
        float(np.asarray(run_group["metrics/area_px"][int(roi_idx), comp_idx], dtype=np.float32)),
        tuple(
            float(value)
            for value in np.asarray(run_group["metrics/centroid_xy"][int(roi_idx), comp_idx], dtype=np.float32).tolist()
        ),
        bool(np.asarray(run_group["metrics/centroid_valid"][int(roi_idx), comp_idx], dtype=bool)),
        tuple(
            float(value)
            for value in np.asarray(run_group["metrics/bbox_xyxy"][int(roi_idx), comp_idx], dtype=np.float32).tolist()
        ),
        bool(np.asarray(run_group["metrics/bbox_valid"][int(roi_idx), comp_idx], dtype=bool)),
        bool(np.asarray(run_group["edit_applied"][int(roi_idx), comp_idx], dtype=bool)),
        int(np.asarray(component_metrics["component_count"][int(roi_idx)], dtype=np.int32)),
        float(np.asarray(component_metrics["largest_component_fraction"][int(roi_idx)], dtype=np.float32)),
        int(np.asarray(component_metrics["hole_count"][int(roi_idx)], dtype=np.int32)),
        float(np.asarray(component_metrics["hole_area_fraction"][int(roi_idx)], dtype=np.float32)),
        float(np.asarray(component_metrics["sigma_noise"][int(roi_idx)], dtype=np.float32)),
        float(np.asarray(component_metrics["curvature_var"][int(roi_idx)], dtype=np.float32)),
        float(np.asarray(component_metrics["ipr"][int(roi_idx)], dtype=np.float32)),
        float(np.asarray(component_metrics["solidity"][int(roi_idx)], dtype=np.float32)),
        str(reason_labels[int(roi_idx)]),
    )


def _format_component_summary_lines(
    run_group: zarr.Group,
    component_group: zarr.Group,
    *,
    comp_idx: int,
    roi_idx: int,
) -> list[str]:
    metrics = run_group["metrics"]
    component_metrics = component_group.require_group("metrics")
    reason_labels = _load_or_init_reason_labels(component_group, int(run_group["masks_roi"].shape[0]))
    present = bool(np.asarray(metrics["mask_present"][int(roi_idx), comp_idx], dtype=bool))
    area = float(np.asarray(metrics["area_px"][int(roi_idx), comp_idx], dtype=np.float32))
    centroid_xy = np.asarray(metrics["centroid_xy"][int(roi_idx), comp_idx], dtype=np.float32)
    centroid_valid = bool(np.asarray(metrics["centroid_valid"][int(roi_idx), comp_idx], dtype=bool))
    bbox_xyxy = np.asarray(metrics["bbox_xyxy"][int(roi_idx), comp_idx], dtype=np.float32)
    bbox_valid = bool(np.asarray(metrics["bbox_valid"][int(roi_idx), comp_idx], dtype=bool))
    component_count = int(np.asarray(component_metrics["component_count"][int(roi_idx)], dtype=np.int32))
    largest_fraction = float(
        np.asarray(component_metrics["largest_component_fraction"][int(roi_idx)], dtype=np.float32)
    )
    hole_count = int(np.asarray(component_metrics["hole_count"][int(roi_idx)], dtype=np.int32))
    hole_area_fraction = float(
        np.asarray(component_metrics["hole_area_fraction"][int(roi_idx)], dtype=np.float32)
    )
    sigma_noise = float(np.asarray(component_metrics["sigma_noise"][int(roi_idx)], dtype=np.float32))
    curvature_var = float(np.asarray(component_metrics["curvature_var"][int(roi_idx)], dtype=np.float32))
    ipr = float(np.asarray(component_metrics["ipr"][int(roi_idx)], dtype=np.float32))
    solidity = float(np.asarray(component_metrics["solidity"][int(roi_idx)], dtype=np.float32))
    centroid_text = (
        f"centroid=({float(centroid_xy[0]):.1f}, {float(centroid_xy[1]):.1f})"
        if centroid_valid
        else "centroid=--"
    )
    bbox_text = (
        "bbox=["
        f"{float(bbox_xyxy[0]):.1f},{float(bbox_xyxy[1]):.1f},"
        f"{float(bbox_xyxy[2]):.1f},{float(bbox_xyxy[3]):.1f}]"
        if bbox_valid
        else "bbox=--"
    )
    return [
        f"present={int(present)} area_px={area:.1f}",
        f"{centroid_text}  {bbox_text}",
        f"components={component_count} largest_frac={largest_fraction:.3f}",
        f"holes={hole_count} hole_area_frac={hole_area_fraction:.3f}",
        f"sigma_noise={sigma_noise:.3f} curvature_var={curvature_var:.6f}",
        f"ipr={ipr:.3f} solidity={solidity:.3f}",
        f"reason={str(reason_labels[int(roi_idx)])}",
    ]


def _draw_summary_lines(
    panel: np.ndarray,
    lines: Sequence[str],
    *,
    start_y: int = 46,
    line_height: int = 18,
) -> np.ndarray:
    output = np.asarray(panel, dtype=np.uint8).copy()
    for idx, line in enumerate(lines):
        cv2.putText(
            output,
            str(line),
            (10, start_y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    return output


def sync_refined_subject_mask_metadata(
    zarr_path: str | Path,
    *,
    refined_run: str,
    component_name: str,
    roi_indices: Sequence[int],
    source_subject_mask_run: Optional[str] = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    refined_parent = root.get("refined_subject_masks_runs")
    if refined_parent is None or str(refined_run) not in refined_parent:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} not found.")

    existing_run = refined_parent[str(refined_run)]
    labels_raw = existing_run.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} missing usable mask_labels attr.")

    normalized_component = _normalize_component_name(component_name)
    if normalized_component is None:
        raise ValueError("component_name is required.")
    if normalized_component not in tuple(str(item) for item in labels_raw):
        raise RuntimeError(
            f"Component '{normalized_component}' not available in refined_subject_masks_runs/{refined_run}."
        )

    resolved_source_run = source_subject_mask_run or str(existing_run.attrs.get("source_subject_mask_run") or "")
    if not resolved_source_run:
        raise RuntimeError(
            f"refined_subject_masks_runs/{refined_run} is missing source_subject_mask_run and no override was provided."
        )

    source, refined = prepare_refined_subject_run(
        root,
        subject_run=resolved_source_run,
        refined_run=str(refined_run),
        components=tuple(str(item) for item in labels_raw),
    )
    if refined.run_name != str(refined_run):
        raise RuntimeError(f"Resolved refined run mismatch: expected {refined_run}, got {refined.run_name}.")
    if source.run_name != str(resolved_source_run):
        raise RuntimeError(
            f"Resolved source subject-mask run mismatch: expected {resolved_source_run}, got {source.run_name}."
        )
    _primary_source, component_sources = _load_refined_component_source_runs(root, refined, default_source=source)

    run_group = refined.group
    total_rois = int(run_group["masks_roi"].shape[0])
    normalized_rows = _normalize_roi_indices(roi_indices, total_rois)
    comp_idx = int(refined.component_to_index[normalized_component])
    component_group = run_group.require_group("components").require_group(normalized_component)

    before_states = {
        int(roi_idx): _component_sync_state(run_group, component_group, comp_idx=comp_idx, roi_idx=int(roi_idx))
        for roi_idx in normalized_rows
    }
    edited_masks_batch = np.stack(
        [np.asarray(run_group["masks_roi"][int(roi_idx)], dtype=np.uint8) for roi_idx in normalized_rows],
        axis=0,
    )
    _apply_refined_subject_roi_rows(
        component_sources=component_sources,
        refined=refined,
        roi_indices=normalized_rows,
        edited_masks_batch=edited_masks_batch,
        component_names=(normalized_component,),
        update_mode="interactive",
        update_method=REFINED_SUBJECT_SYNC_METHOD,
    )

    changed_count = 0
    noop_count = 0
    for roi_idx in normalized_rows:
        before = before_states[int(roi_idx)]
        after = _component_sync_state(run_group, component_group, comp_idx=comp_idx, roi_idx=int(roi_idx))
        if before == after:
            noop_count += 1
        else:
            changed_count += 1

    return {
        "status": "updated",
        "zarr_path": str(Path(zarr_path)),
        "refined_run": refined.run_name,
        "component_name": normalized_component,
        "source_subject_mask_run": source.run_name,
        "roi_indices": normalized_rows,
        "roi_count": int(len(normalized_rows)),
        "changed_roi_count": int(changed_count),
        "noop_roi_count": int(noop_count),
        "updated_at_utc": str(run_group.attrs.get("updated_at_utc") or ""),
    }


def _overlay_components(
    roi_image: np.ndarray,
    component_names: Sequence[str],
    masks: Sequence[np.ndarray],
) -> np.ndarray:
    display = cv2.cvtColor(np.asarray(roi_image, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = display.copy()
    for component_name, mask in zip(component_names, masks):
        color = COMPONENT_COLORS.get(component_name, (255, 255, 255))
        overlay[np.asarray(mask, dtype=np.uint8) > 0] = color
    return cv2.addWeighted(overlay, 0.45, display, 0.55, 0)


def _panel(title: str, image: np.ndarray) -> np.ndarray:
    panel = np.asarray(image, dtype=np.uint8).copy()
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return panel


def _blank_panel_like(roi_image: np.ndarray, title: str, note: str) -> np.ndarray:
    panel = np.zeros((int(roi_image.shape[0]), int(roi_image.shape[1]), 3), dtype=np.uint8)
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(panel, note, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return panel


def launch_review(
    zarr_path: str,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    component: Optional[str] = None,
    roi_index: int = 0,
    review_method: str = DEFAULT_REVIEW_METHOD,
    review_intended_use: str = DEFAULT_REVIEW_INTENDED_USE,
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
) -> str:
    root = open_zarr_root(zarr_path, mode="a")
    source, refined = prepare_refined_subject_run(
        root,
        subject_run=subject_run,
        refined_run=refined_run,
        components=components,
    )
    crop_run_name = str(crop_run or source.crop_run)
    crop_parent = root.get("crop_runs")
    if crop_parent is None or crop_run_name not in crop_parent:
        raise RuntimeError(f"crop_runs/{crop_run_name} not found.")
    crop_group = crop_parent[crop_run_name]
    if "roi_images" not in crop_group:
        raise RuntimeError(f"crop_runs/{crop_run_name} missing roi_images.")
    roi_images = crop_group["roi_images"]

    component_names = refined.component_names
    if not component_names:
        raise RuntimeError("No components available in refined subject-mask run.")
    active_component = _normalize_component_name(component) or component_names[0]
    if active_component not in refined.component_to_index:
        raise RuntimeError(
            f"Component '{active_component}' not available in refined_subject_masks_runs/{refined.run_name}."
        )
    active_idx = int(refined.component_to_index[active_component])
    total_rois = int(roi_images.shape[0])
    current_pos = max(0, min(int(roi_index), total_rois - 1))
    brush_radius = DEFAULT_BRUSH_RADIUS
    drawing = False
    erase_mode = False
    display_scale = max(1.0, float(DISPLAY_SCALE))
    cursor_pos: Optional[tuple[str, int, int]] = None

    _require_gui_display()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1900, 1100)

    refined_masks_arr = refined.group["masks_roi"]
    _primary_source, component_sources = _load_refined_component_source_runs(root, refined, default_source=source)
    display_layout = {"all": (0, 0, 0, 0), "edit": (0, 0, 0, 0)}
    roi_img: np.ndarray | None = None
    original_masks: list[np.ndarray] = []
    edit_masks: list[np.ndarray] = []

    def load_current_roi() -> None:
        nonlocal roi_img, original_masks, edit_masks
        roi_img = np.asarray(roi_images[current_pos], dtype=np.uint8)
        current_masks = np.asarray(refined_masks_arr[current_pos], dtype=np.uint8)
        original_masks = [current_masks[idx].copy() for idx in range(len(component_names))]
        edit_masks = [current_masks[idx].copy() for idx in range(len(component_names))]

    def update_display() -> None:
        if roi_img is None:
            return
        source_active_mask = _source_mask_for_resolved_component(component_sources, component_names[active_idx], current_pos)
        source_panel = (
            _panel(
                f"Source {component_names[active_idx]}",
                _overlay_components(roi_img, (component_names[active_idx],), (source_active_mask,)),
            )
            if np.count_nonzero(source_active_mask) > 0
            else _blank_panel_like(roi_img, f"Source {component_names[active_idx]}", "Unavailable / empty")
        )
        roi_panel = _panel("Crop ROI", cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR))
        refined_panel = _panel("Refined Overlay", _overlay_components(roi_img, component_names, edit_masks))
        active_mask = edit_masks[active_idx]
        active_panel = _panel(
            f"Editor {component_names[active_idx]}",
            _overlay_components(roi_img, (component_names[active_idx],), (active_mask,)),
        )
        active_component_group = refined.group.require_group("components").require_group(component_names[active_idx])
        summary_lines = _format_component_summary_lines(
            refined.group,
            active_component_group,
            comp_idx=active_idx,
            roi_idx=current_pos,
        )
        active_panel = _draw_summary_lines(active_panel, summary_lines)

        header = refined_panel.copy()
        state_payload = dict(refined.group.attrs.get("component_review_statuses") or {}).get(component_names[active_idx], {})
        state_text = str(state_payload.get("state") or "pending")
        cv2.putText(
            header,
            f"ROI {current_pos + 1}/{total_rois}  Active: {component_names[active_idx]}  Brush: {brush_radius}  Review: {state_text}",
            (10, int(header.shape[0]) - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        top = np.hstack([roi_panel, source_panel])
        bottom = np.hstack([header, active_panel])
        combined = np.vstack([top, bottom])

        h = int(roi_img.shape[0])
        w = int(roi_img.shape[1])
        display_layout["all"] = (0, h, w, h)
        display_layout["edit"] = (w, h, w, h)

        if cursor_pos is not None:
            panel_name, local_x, local_y = cursor_pos
            base_x, base_y, panel_w, panel_h = display_layout[panel_name]
            x1 = max(base_x, base_x + local_x - brush_radius)
            y1 = max(base_y, base_y + local_y - brush_radius)
            x2 = min(base_x + panel_w - 1, base_x + local_x + brush_radius)
            y2 = min(base_y + panel_h - 1, base_y + local_y + brush_radius)
            cv2.rectangle(combined, (x1, y1), (x2, y2), COMPONENT_COLORS.get(component_names[active_idx], (255, 255, 255)), 1)

        if display_scale != 1.0:
            combined = cv2.resize(combined, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WINDOW_NAME, combined)

    def save_current() -> None:
        stacked = np.stack(edit_masks, axis=0)
        save_refined_subject_roi(
            source=source,
            refined=refined,
            roi_idx=current_pos,
            edited_masks=stacked,
            component_sources=component_sources,
        )
        print(f"Saved refined subject masks for ROI {current_pos}.")

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal drawing, erase_mode, cursor_pos
        if display_scale != 1.0:
            x = int(x / display_scale)
            y = int(y / display_scale)

        edit_x, edit_y, edit_w, edit_h = display_layout["edit"]
        all_x, all_y, all_w, all_h = display_layout["all"]
        in_edit = edit_x <= x < edit_x + edit_w and edit_y <= y < edit_y + edit_h
        in_all = all_x <= x < all_x + all_w and all_y <= y < all_y + all_h
        if not (in_edit or in_all):
            if cursor_pos is not None:
                cursor_pos = None
                update_display()
            if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                drawing = False
            return
        if in_edit:
            local_x = x - edit_x
            local_y = y - edit_y
            panel_name = "edit"
        else:
            local_x = x - all_x
            local_y = y - all_y
            panel_name = "all"

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            erase_mode = False
            cursor_pos = (panel_name, int(local_x), int(local_y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            erase_mode = True
            cursor_pos = (panel_name, int(local_x), int(local_y))
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
            cursor_pos = None
            update_display()

        if event == cv2.EVENT_MOUSEMOVE and not drawing:
            cursor_pos = (panel_name, int(local_x), int(local_y))
            update_display()
        elif drawing:
            cursor_pos = (panel_name, int(local_x), int(local_y))
            color = 0 if erase_mode else 1
            cv2.circle(edit_masks[active_idx], (local_x, local_y), brush_radius, color, -1)
            update_display()

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    load_current_roi()
    update_display()

    print("\nRefined Subject Mask Review")
    print(f"  Source run: subject_mask_runs/{source.run_name}")
    print(f"  Refined run: refined_subject_masks_runs/{refined.run_name}")
    print(f"  Crop run: {crop_run_name}")
    print(f"  Components: {', '.join(component_names)}")
    print("Controls:")
    print("  Mouse: paint (LMB) / erase (RMB) on refined overlay or editor")
    print("  1..9: select active component by order shown")
    print("  [ / ]: brush size")
    print("  s: save current ROI edits")
    print("  r: reset current ROI to stored refined masks")
    print("  n/p: next/previous ROI")
    print("  a: approve active component")
    print("  N: mark active component needs_review")
    print("  R: mark active component rejected")
    print("  P: mark active component pending")
    print("  q/ESC: quit")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("1") <= key <= ord("9"):
            choice = key - ord("1")
            if choice < len(component_names):
                active_idx = choice
                update_display()
        elif key == ord("["):
            brush_radius = max(1, brush_radius - 1)
            update_display()
        elif key == ord("]"):
            brush_radius = min(64, brush_radius + 1)
            update_display()
        elif key == ord("r"):
            edit_masks = [mask.copy() for mask in original_masks]
            update_display()
        elif key == ord("s"):
            save_current()
            original_masks = [mask.copy() for mask in edit_masks]
        elif key == ord("n"):
            if current_pos < total_rois - 1:
                current_pos += 1
                load_current_roi()
                update_display()
        elif key == ord("p"):
            if current_pos > 0:
                current_pos -= 1
                load_current_roi()
                update_display()
        elif key in (ord("a"), ord("N"), ord("R"), ord("P")):
            state = {
                ord("a"): "approved",
                ord("N"): "needs_review",
                ord("R"): "rejected",
                ord("P"): "pending",
            }[key]
            reviewer_name = reviewer or os.environ.get("USER") or os.environ.get("USERNAME")
            component_payload, run_payload = apply_component_review_status(
                refined.parent,
                refined.run_name,
                refined.group,
                component_name=component_names[active_idx],
                state=state,
                method=review_method,
                intended_use=review_intended_use,
                reviewer=reviewer_name,
                notes=review_notes,
            )
            print(
                f"Set {component_names[active_idx]} review to {component_payload.get('state')} "
                f"(run={run_payload.get('state')})"
            )
            update_display()

    cv2.destroyAllWindows()
    return refined.run_name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual paint/review UI for refined subject masks.")
    parser.add_argument("zarr_path", help="Path to Palette zarr archive.")
    parser.add_argument("--subject-run", help="Source subject_mask_runs/<run> to refine (default: latest).")
    parser.add_argument("--refined-run", help="Existing refined_subject_masks run to open, or target name to create.")
    parser.add_argument("--crop-run", help="Crop run to use for ROI images (default: source subject mask crop run).")
    parser.add_argument(
        "--components",
        nargs="+",
        help="Components to include when creating a new refined run (default: subject_body swim_bladder).",
    )
    parser.add_argument("--component", help="Initial active component.")
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index.")
    parser.add_argument("--review-method", default=DEFAULT_REVIEW_METHOD)
    parser.add_argument("--review-intended-use", default=DEFAULT_REVIEW_INTENDED_USE)
    parser.add_argument("--reviewer", help="Reviewer name to record in review payloads.")
    parser.add_argument("--review-notes", help="Optional note attached to review payload updates.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> str:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return launch_review(
        args.zarr_path,
        subject_run=args.subject_run,
        refined_run=args.refined_run,
        crop_run=args.crop_run,
        components=args.components,
        component=args.component,
        roi_index=args.roi_index,
        review_method=args.review_method,
        review_intended_use=args.review_intended_use,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
