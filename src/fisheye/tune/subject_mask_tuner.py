#!/usr/bin/env python3
"""Interactive tuner for ROI-local subject-mask seed generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import argparse
import cv2
import numpy as np
import yaml
from skimage import measure, morphology
import zarr

from ..diagnostics.preview_eye_mask_background_subtraction import (
    _extract_background_roi_ds,
    _extract_background_roi_full,
    _prepare_panel,
    _resolve_background_source,
    _resolve_run_name,
)
from . import eye_mask_tuner as eye_mask_ops
from ..utils.zarr_io import open_zarr_root


TUNING_KEY = "subject_mask_tuning"
EYE_TUNING_COMPAT_KEY = "eye_mask_tuning"
WINDOW_NAME = "Subject Mask Tuner"
DEBUG_WINDOW_SUFFIX = " Eye Debug"
DEFAULT_COMPONENT = "subject_body"
CANONICAL_COMPONENTS = ("subject_body", "eyes_union", "swim_bladder")
EYE_COMPONENTS = ("eyes_union", "eye_left", "eye_right")
SUPPORTED_COMPONENTS = ("subject_body", "eyes_union", "eye_left", "eye_right", "swim_bladder")
SUBJECT_TUNING_VERSION = "2.0"
DEFAULT_EYE_ROTATE_ROI = True
COMPONENT_ALIASES = {
    "subject": "subject_body",
    "subject_body": "subject_body",
    "body": "subject_body",
    "whole_subject": "subject_body",
    "whole-subject": "subject_body",
    "subject-body": "subject_body",
    "subject-body-mask": "subject_body",
    "eye": "eyes_union",
    "eyes": "eyes_union",
    "eye_union": "eyes_union",
    "eye-union": "eyes_union",
    "eye-union-mask": "eyes_union",
    "eyes_union": "eyes_union",
    "eye_left": "eye_left",
    "left_eye": "eye_left",
    "left-eye": "eye_left",
    "left-eye-mask": "eye_left",
    "eye-left-mask": "eye_left",
    "eye_right": "eye_right",
    "right_eye": "eye_right",
    "right-eye": "eye_right",
    "right-eye-mask": "eye_right",
    "eye-right-mask": "eye_right",
    "swimbladder": "swim_bladder",
    "swim-bladder": "swim_bladder",
    "swim_bladder": "swim_bladder",
    "swimbladder-mask": "swim_bladder",
    "swim-bladder-mask": "swim_bladder",
}
DEFAULT_THRESHOLD = 50
DEFAULT_BLUR_KERNEL = 0
DEFAULT_CLOSING_RADIUS = 3
DEFAULT_OPENING_RADIUS = 1
DEFAULT_MIN_AREA = 80
DEFAULT_KEEP_LARGEST_COMPONENT = 1
DEFAULT_PANEL_SIZE = 360
TRACKBAR_BLUR_MAX = 12
TRACKBAR_RADIUS_MAX = 20
TRACKBAR_MIN_AREA_MAX = 5000

current_roi_index = 0
diff_threshold = DEFAULT_THRESHOLD
gaussian_blur_kernel = DEFAULT_BLUR_KERNEL
closing_radius = DEFAULT_CLOSING_RADIUS
opening_radius = DEFAULT_OPENING_RADIUS
min_area = DEFAULT_MIN_AREA
keep_largest_component = DEFAULT_KEEP_LARGEST_COMPONENT


@dataclass(frozen=True)
class SubjectMaskSource:
    run_name: str
    group: zarr.Group
    mask_labels: tuple[str, ...]
    available_channels: np.ndarray
    masks_roi: Any
    mask_probs_roi: Any | None
    probability_encoding: str


@dataclass(frozen=True)
class EyeKeypointSource:
    group_name: str
    run_name: str
    group: zarr.Group
    keypoints_roi: Any
    success_flags: np.ndarray
    heading_values: Any | None


def update_roi_index(val: int) -> None:
    global current_roi_index
    current_roi_index = max(0, int(val))


def update_threshold(val: int) -> None:
    global diff_threshold
    diff_threshold = int(np.clip(val, 0, 255))


def update_blur_kernel(val: int) -> None:
    global gaussian_blur_kernel
    gaussian_blur_kernel = max(0, int(val))


def update_closing_radius(val: int) -> None:
    global closing_radius
    closing_radius = max(0, int(val))


def update_opening_radius(val: int) -> None:
    global opening_radius
    opening_radius = max(0, int(val))


def update_min_area(val: int) -> None:
    global min_area
    min_area = max(1, int(val))


def update_keep_largest_component(val: int) -> None:
    global keep_largest_component
    keep_largest_component = 1 if int(val) > 0 else 0


def _normalize_kernel_size(value: object) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 0
    if numeric <= 1:
        return 0
    if numeric % 2 == 0:
        numeric += 1
    return numeric


def _normalize_component_name(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "_")
    if not text:
        return None
    return COMPONENT_ALIASES.get(text, text if text in SUPPORTED_COMPONENTS else None)


def _is_eye_component(component_name: Optional[str]) -> bool:
    normalized = _normalize_component_name(component_name)
    return normalized in EYE_COMPONENTS


def _current_params() -> Dict[str, Any]:
    return {
        "diff_threshold": int(diff_threshold),
        "gaussian_blur_kernel": int(_normalize_kernel_size(gaussian_blur_kernel)),
        "closing_radius": int(max(0, closing_radius)),
        "opening_radius": int(max(0, opening_radius)),
        "min_area": int(max(1, min_area)),
        "keep_largest_component": bool(keep_largest_component),
    }


def _normalize_subject_component_params(
    component_name: str,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    component_name = _normalize_component_name(component_name) or DEFAULT_COMPONENT
    if component_name == "eyes_union":
        return {
            "roi_padding": int(params.get("roi_padding", 12)),
            "pre_threshold": (
                int(params["pre_threshold"]) if params.get("pre_threshold") is not None else None
            ),
            "sobel_strength": float(params.get("sobel_strength", 0.0)),
            "min_area": int(params.get("min_area", 15)),
            "max_area": int(params["max_area"]) if params.get("max_area") is not None else None,
            "min_circularity": (
                float(params["min_circularity"]) if params.get("min_circularity") is not None else None
            ),
            "closing_radius": int(params.get("closing_radius", 3)),
            "opening_radius": int(params.get("opening_radius", 1)),
            "min_eye_separation": (
                float(params["min_eye_separation"])
                if params.get("min_eye_separation") is not None
                else None
            ),
            "max_eye_separation": (
                float(params["max_eye_separation"])
                if params.get("max_eye_separation") is not None
                else None
            ),
        }

    return {
        "diff_threshold": int(np.clip(params.get("diff_threshold", DEFAULT_THRESHOLD), 0, 255)),
        "gaussian_blur_kernel": int(
            _normalize_kernel_size(params.get("gaussian_blur_kernel", DEFAULT_BLUR_KERNEL))
        ),
        "closing_radius": int(max(0, int(params.get("closing_radius", DEFAULT_CLOSING_RADIUS)))),
        "opening_radius": int(max(0, int(params.get("opening_radius", DEFAULT_OPENING_RADIUS)))),
        "min_area": int(max(1, int(params.get("min_area", DEFAULT_MIN_AREA)))),
        "keep_largest_component": bool(params.get("keep_largest_component", DEFAULT_KEEP_LARGEST_COMPONENT)),
    }


def _apply_subject_mask_params(params: Optional[Dict[str, Any]]) -> None:
    global diff_threshold, gaussian_blur_kernel
    global closing_radius, opening_radius, min_area, keep_largest_component
    if not isinstance(params, dict):
        return
    if "diff_threshold" in params:
        diff_threshold = int(np.clip(params["diff_threshold"], 0, 255))
    if "gaussian_blur_kernel" in params:
        gaussian_blur_kernel = _normalize_kernel_size(params["gaussian_blur_kernel"])
    if "closing_radius" in params:
        closing_radius = max(0, int(params["closing_radius"]))
    if "opening_radius" in params:
        opening_radius = max(0, int(params["opening_radius"]))
    if "min_area" in params:
        min_area = max(1, int(params["min_area"]))
    if "keep_largest_component" in params:
        keep_largest_component = 1 if bool(params["keep_largest_component"]) else 0


def _normalize_subject_tuning_payload(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"version": SUBJECT_TUNING_VERSION, "components": {}}

    components_raw = raw.get("components")
    if isinstance(components_raw, dict):
        components: Dict[str, Dict[str, Any]] = {}
        for key, value in components_raw.items():
            normalized = _normalize_component_name(key)
            if normalized is None or not isinstance(value, dict):
                continue
            components[normalized] = dict(value)
        payload = dict(raw)
        payload["version"] = str(payload.get("version") or SUBJECT_TUNING_VERSION)
        payload["components"] = components
        return payload

    tuned = raw.get("tuned_parameters")
    if not isinstance(tuned, dict):
        return {"version": SUBJECT_TUNING_VERSION, "components": {}}
    return {
        "version": SUBJECT_TUNING_VERSION,
        "components": {
            DEFAULT_COMPONENT: {
                "method": str(raw.get("method") or "traditional_subject_mask_seed"),
                "version": str(raw.get("version") or "1.0"),
                "tuned_timestamp": raw.get("tuned_timestamp"),
                "tuned_parameters": dict(tuned),
                "context": dict(raw.get("context")) if isinstance(raw.get("context"), dict) else {},
            }
        },
        "migrated_from_flat": True,
    }


def _load_subject_tuning_entry_from_root(
    root: zarr.Group,
    component_name: str,
) -> Optional[Dict[str, Any]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    payload = _normalize_subject_tuning_payload(analysis.attrs.get(TUNING_KEY))
    components = payload.get("components", {})
    if not isinstance(components, dict):
        return None
    entry = components.get(component_name)
    return dict(entry) if isinstance(entry, dict) else None


def _load_eye_tuned_params_from_root(root: zarr.Group) -> Optional[Dict[str, Any]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    tuning = analysis.attrs.get(EYE_TUNING_COMPAT_KEY)
    if not isinstance(tuning, dict):
        return None
    tuned = tuning.get("tuned_parameters")
    return dict(tuned) if isinstance(tuned, dict) else None


def _load_tuned_params_from_root(
    root: zarr.Group,
    component_name: str = DEFAULT_COMPONENT,
) -> Optional[Dict[str, Any]]:
    normalized = _normalize_component_name(component_name) or DEFAULT_COMPONENT
    entry = _load_subject_tuning_entry_from_root(root, normalized)
    tuned = entry.get("tuned_parameters") if isinstance(entry, dict) else None
    if isinstance(tuned, dict):
        return dict(tuned)
    if normalized == "eyes_union":
        return _load_eye_tuned_params_from_root(root)
    return None


def _build_subject_eye_tuning_entry(
    *,
    tuned_parameters: Mapping[str, Any],
    context: Mapping[str, Any],
    timestamp: str,
    version: str,
    method: Optional[str],
    extra_entry_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_context = dict(context)
    if "component_name" in normalized_context and "storage_component_name" not in normalized_context:
        normalized_context["storage_component_name"] = normalized_context.pop("component_name")
    entry: Dict[str, Any] = {
        "method": str(method or "global_threshold_otsu"),
        "subject_method_family": "eye_mask_threshold_lr_v1",
        "version": str(version),
        "tuned_timestamp": timestamp,
        "tuned_parameters": dict(tuned_parameters),
        "context": normalized_context,
        "output_labels": ["eye_left", "eye_right"],
        "storage_component": "eyes_union",
    }
    if extra_entry_fields:
        entry.update(dict(extra_entry_fields))
    entry.pop("union_component", None)
    entry.pop("legacy_eye_mask_method", None)
    return entry


def _load_initial_params(config_path: Optional[Path]) -> None:
    if config_path is None or not config_path.exists():
        return
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        return
    section = config.get("subject_masks")
    if isinstance(section, dict) and isinstance(section.get("traditional"), dict):
        section = section["traditional"]
    if not isinstance(section, dict):
        section = config.get("subject_mask")
    if isinstance(section, dict):
        _apply_subject_mask_params(section)


def _require_gui_display() -> None:
    display = str(os.environ.get("DISPLAY") or "").strip()
    wayland_display = str(os.environ.get("WAYLAND_DISPLAY") or "").strip()
    if display or wayland_display:
        return

    message = (
        "No GUI display is available for the OpenCV tuner window. "
        "DISPLAY and WAYLAND_DISPLAY are unset in this shell."
    )
    if os.environ.get("TMUX"):
        message += (
            " This shell is running inside tmux, so the pane likely missed the GUI "
            "environment. Open a new tmux window from a GUI-capable session or export "
            "DISPLAY/XAUTHORITY into the current pane before rerunning."
        )
    raise RuntimeError(message)


def _set_initial_window_size(
    window_name: str,
    *,
    panel_size: int,
    footer_height: int,
    columns: int = 3,
    rows: int = 2,
) -> None:
    width = max(960, int(panel_size) * int(columns))
    height = max(720, int(panel_size) * int(rows) + int(footer_height))
    cv2.resizeWindow(window_name, width, height)


def _load_subject_mask_source(root: zarr.Group, explicit: Optional[str]) -> Optional[SubjectMaskSource]:
    parent = root.get("subject_mask_runs")
    if parent is None:
        return None
    run_name = _resolve_run_name(root, "subject_mask_runs", explicit)
    run_group = parent[run_name]
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        raise ValueError(f"subject_mask_runs/{run_name} missing masks_roi.")
    mask_probs_roi = run_group.get("mask_probs_roi")
    mask_labels_attr = run_group.attrs.get("mask_labels")
    if isinstance(mask_labels_attr, (list, tuple)) and mask_labels_attr:
        mask_labels = tuple(str(item) for item in mask_labels_attr)
    else:
        channel_count = int(masks_roi.shape[1]) if len(masks_roi.shape) >= 2 else 0
        if channel_count == len(CANONICAL_COMPONENTS):
            mask_labels = CANONICAL_COMPONENTS
        else:
            raise ValueError(f"subject_mask_runs/{run_name} missing usable mask_labels attr.")
    available_raw = run_group.get("available_channels")
    if available_raw is not None:
        available_channels = np.asarray(available_raw[:], dtype=bool)
    else:
        available_channels = np.ones(len(mask_labels), dtype=bool)
    if available_channels.shape != (len(mask_labels),):
        raise ValueError(
            f"subject_mask_runs/{run_name} available_channels shape {available_channels.shape} "
            f"does not match mask_labels length {len(mask_labels)}."
        )
    probability_encoding = str(run_group.attrs.get("probabilities_encoding") or "unit_float")
    return SubjectMaskSource(
        run_name=str(run_name),
        group=run_group,
        mask_labels=mask_labels,
        available_channels=available_channels,
        masks_roi=masks_roi,
        mask_probs_roi=mask_probs_roi,
        probability_encoding=probability_encoding,
    )


def _find_subject_label_index(subject_source: SubjectMaskSource, label: str) -> Optional[int]:
    try:
        return int(subject_source.mask_labels.index(label))
    except ValueError:
        return None


def _has_available_subject_label(subject_source: SubjectMaskSource, label: str) -> bool:
    idx = _find_subject_label_index(subject_source, label)
    return idx is not None and bool(subject_source.available_channels[idx])


def _subject_source_has_lr_pair(subject_source: SubjectMaskSource) -> bool:
    return _has_available_subject_label(subject_source, "eye_left") and _has_available_subject_label(
        subject_source,
        "eye_right",
    )


def _resolve_eye_keypoint_source(
    root: zarr.Group,
    explicit: Optional[str],
) -> EyeKeypointSource:
    group, run_name = eye_mask_ops._resolve_keypoints_group(root, explicit)
    refined_parent = root.get("refined_keypoints_runs")
    if isinstance(refined_parent, zarr.Group) and run_name in refined_parent:
        group_name = "refined_keypoints_runs"
    else:
        group_name = "keypoints_runs"
    keypoints_roi = group["keypoints_roi"]
    if "refined_success" in group:
        success_flags = np.asarray(group["refined_success"][:], dtype=bool)
    elif "detection_success" in group:
        success_flags = np.asarray(group["detection_success"][:], dtype=bool)
    elif "source_success" in group:
        success_flags = np.asarray(group["source_success"][:], dtype=bool)
    else:
        success_flags = np.ones(int(keypoints_roi.shape[0]), dtype=bool)
    heading_values = group.get("heading")
    return EyeKeypointSource(
        group_name=group_name,
        run_name=str(run_name),
        group=group,
        keypoints_roi=keypoints_roi,
        success_flags=success_flags,
        heading_values=heading_values,
    )


def _resolve_subject_component(
    subject_source: Optional[SubjectMaskSource],
    requested: Optional[str],
) -> tuple[str, Optional[int]]:
    normalized = _normalize_component_name(requested)
    if subject_source is None:
        return normalized or DEFAULT_COMPONENT, None

    if normalized is not None:
        if normalized == "eyes_union" and normalized not in subject_source.mask_labels and _subject_source_has_lr_pair(
            subject_source
        ):
            return normalized, None
        if normalized not in subject_source.mask_labels:
            available_text = ", ".join(subject_source.mask_labels)
            raise ValueError(
                f"Component '{requested}' not found in subject mask labels. Available: {available_text}"
            )
        component_index = int(subject_source.mask_labels.index(normalized))
        if not bool(subject_source.available_channels[component_index]):
            raise ValueError(
                f"Component '{normalized}' is unavailable in subject_mask_runs/{subject_source.run_name}."
            )
        return normalized, component_index

    available_names = [
        label
        for label, available in zip(subject_source.mask_labels, subject_source.available_channels.tolist())
        if bool(available)
    ]
    if _has_available_subject_label(subject_source, "eyes_union"):
        selected = "eyes_union"
    elif _subject_source_has_lr_pair(subject_source):
        selected = "eyes_union"
    elif len(available_names) == 1:
        selected = available_names[0]
    elif DEFAULT_COMPONENT in available_names:
        selected = DEFAULT_COMPONENT
    elif available_names:
        selected = available_names[0]
    else:
        selected = subject_source.mask_labels[0]
    if selected == "eyes_union" and selected not in subject_source.mask_labels:
        return selected, None
    return selected, int(subject_source.mask_labels.index(selected))


def _default_eye_union_params() -> Dict[str, Any]:
    return {
        "roi_padding": 12,
        "pre_threshold": None,
        "sobel_strength": 0.0,
        "min_area": 15,
        "max_area": None,
        "min_circularity": None,
        "closing_radius": 3,
        "opening_radius": 1,
        "min_eye_separation": 4.0,
        "max_eye_separation": None,
        "rotate_roi": DEFAULT_EYE_ROTATE_ROI,
    }


def _build_eye_union_params(
    tuned_params: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    params = _default_eye_union_params()
    if isinstance(tuned_params, Mapping):
        params.update(_normalize_subject_component_params("eyes_union", tuned_params))
        if "rotate_roi" in tuned_params:
            params["rotate_roi"] = bool(tuned_params["rotate_roi"])
    return params


def _rotate_image_nearest(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    return cv2.warpAffine(
        np.asarray(image),
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _text_panel(message: str, *, shape: tuple[int, int] = (120, 180)) -> np.ndarray:
    h, w = shape
    panel = np.zeros((int(h), int(w), 3), dtype=np.uint8)
    lines = [line for line in str(message).splitlines() if line.strip()] or ["n/a"]
    for idx, line in enumerate(lines[:4]):
        cv2.putText(
            panel,
            line,
            (8, 26 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return panel


def _decode_probability_image(image: np.ndarray, *, encoding: str) -> np.ndarray:
    arr = np.asarray(image)
    source_dtype = arr.dtype
    decoded = arr.astype(np.float32, copy=False)
    if np.issubdtype(source_dtype, np.integer) and encoding == "linear_uint8_0_255":
        max_value = float(np.iinfo(source_dtype).max)
        if max_value > 1.0:
            decoded /= max_value
    decoded = np.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(decoded, 0.0, 1.0, out=decoded)


def _postprocess_subject_mask(
    binary: np.ndarray,
    *,
    closing_radius_value: int,
    opening_radius_value: int,
    min_area_value: int,
    keep_largest: bool,
) -> tuple[np.ndarray, Dict[str, Any]]:
    mask = np.asarray(binary, dtype=bool)
    if closing_radius_value > 0:
        mask = morphology.binary_closing(mask, morphology.disk(int(closing_radius_value)))
    if opening_radius_value > 0:
        mask = morphology.binary_opening(mask, morphology.disk(int(opening_radius_value)))

    labeled = measure.label(mask)
    if int(labeled.max()) <= 0:
        return np.zeros_like(mask, dtype=np.uint8), {
            "candidate_components": 0,
            "kept_components": 0,
            "largest_area": 0,
            "kept_area": 0,
            "bbox_xyxy": None,
        }

    regions = sorted(measure.regionprops(labeled), key=lambda region: region.area, reverse=True)
    largest_area = int(regions[0].area) if regions else 0
    keep_labels: list[int] = []
    min_area_int = max(1, int(min_area_value))
    for region in regions:
        if int(region.area) < min_area_int:
            continue
        keep_labels.append(int(region.label))
        if keep_largest:
            break

    if not keep_labels:
        return np.zeros_like(mask, dtype=np.uint8), {
            "candidate_components": len(regions),
            "kept_components": 0,
            "largest_area": largest_area,
            "kept_area": 0,
            "bbox_xyxy": None,
        }

    filtered = np.isin(labeled, np.asarray(keep_labels, dtype=np.int32))
    coords = np.argwhere(filtered)
    bbox_xyxy = None
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        bbox_xyxy = [int(x0), int(y0), int(x1) + 1, int(y1) + 1]

    return filtered.astype(np.uint8, copy=False), {
        "candidate_components": len(regions),
        "kept_components": len(keep_labels),
        "largest_area": largest_area,
        "kept_area": int(filtered.sum()),
        "bbox_xyxy": bbox_xyxy,
    }


def _compute_subject_mask_seed(
    roi_image: np.ndarray,
    background_roi: np.ndarray,
    params: Dict[str, Any],
    *,
    allowed_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    diff_raw = np.clip(
        background_roi.astype(np.int16) - roi_image.astype(np.int16),
        0,
        255,
    ).astype(np.uint8)
    if allowed_mask is not None:
        allowed = np.asarray(allowed_mask > 0, dtype=np.uint8)
        if allowed.shape != diff_raw.shape:
            raise ValueError(
                f"allowed_mask shape {allowed.shape} does not match ROI shape {diff_raw.shape}."
            )
        diff_raw = np.where(allowed > 0, diff_raw, 0).astype(np.uint8, copy=False)
    blur_kernel = _normalize_kernel_size(params.get("gaussian_blur_kernel", 0))
    if blur_kernel >= 3:
        diff_filtered = cv2.GaussianBlur(diff_raw, (blur_kernel, blur_kernel), 0)
    else:
        diff_filtered = diff_raw
    if allowed_mask is not None:
        allowed = np.asarray(allowed_mask > 0, dtype=np.uint8)
        diff_filtered = np.where(allowed > 0, diff_filtered, 0).astype(np.uint8, copy=False)

    otsu_threshold, _ = cv2.threshold(
        diff_filtered,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    seed_mask = (diff_filtered >= int(np.clip(params["diff_threshold"], 0, 255))).astype(np.uint8)
    if allowed_mask is not None:
        allowed = np.asarray(allowed_mask > 0, dtype=np.uint8)
        seed_mask = np.where(allowed > 0, seed_mask, 0).astype(np.uint8, copy=False)
    processed_mask, stats = _postprocess_subject_mask(
        seed_mask,
        closing_radius_value=int(params["closing_radius"]),
        opening_radius_value=int(params["opening_radius"]),
        min_area_value=int(params["min_area"]),
        keep_largest=bool(params["keep_largest_component"]),
    )
    return {
        "diff_raw": diff_raw,
        "diff_filtered": diff_filtered,
        "seed_mask": seed_mask,
        "processed_mask": processed_mask,
        "otsu_threshold": int(otsu_threshold),
        "stats": stats,
    }


def _resolve_full_frame_shape(root: zarr.Group) -> Optional[tuple[int, int]]:
    raw_video = root.get("raw_video")
    images_full = raw_video.get("images_full") if raw_video is not None else None
    if images_full is not None and len(images_full.shape) >= 3:
        return int(images_full.shape[1]), int(images_full.shape[2])
    height = root.attrs.get("height") or root.attrs.get("video_height")
    width = root.attrs.get("width") or root.attrs.get("video_width")
    if height is not None and width is not None:
        return int(height), int(width)
    return None


def _create_dish_mask_image(mask_params: Mapping[str, Any], img_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_params:
        return None
    dish_shape = mask_params.get("shape", "circle")
    mask = None
    if dish_shape == "rectangle" and "roi" in mask_params:
        x, y, w, h = [int(v) for v in mask_params["roi"]]
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    elif dish_shape == "circle":
        center = mask_params.get("center")
        radius = mask_params.get("radius")
        if center and radius:
            mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.circle(mask, tuple(int(v) for v in center), int(radius), 255, -1)
    return mask


def _resolve_dish_mask_projection(
    root: zarr.Group,
) -> tuple[Optional[np.ndarray], Optional[str], Optional[tuple[int, int]]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None, None, None
    raw_mask = analysis.attrs.get("dish_mask")
    if not isinstance(raw_mask, Mapping):
        return None, None, None
    mask_data = dict(raw_mask)
    shape = mask_data.get("shape")
    if not shape:
        if "detected_circle" in mask_data:
            shape = "circle"
        elif "rectangle" in mask_data:
            shape = "rectangle"
    tuned_on_array = (
        mask_data.get("tuned_on_array")
        or mask_data.get("source", {}).get("array")
        or "images_full"
    )
    metrics = mask_data.get("metrics", {}) if isinstance(mask_data.get("metrics"), Mapping) else {}
    image_shape_raw = metrics.get("image_shape")
    image_shape = None
    if isinstance(image_shape_raw, (list, tuple)) and len(image_shape_raw) == 2:
        image_shape = (int(image_shape_raw[0]), int(image_shape_raw[1]))

    if shape == "rectangle" and isinstance(mask_data.get("rectangle"), Mapping):
        params = {
            "shape": "rectangle",
            "roi": [int(v) for v in mask_data["rectangle"]["roi"]],
        }
    elif shape == "circle" and isinstance(mask_data.get("detected_circle"), Mapping):
        circle = mask_data["detected_circle"]
        params = {
            "shape": "circle",
            "center": [int(v) for v in circle["center"]],
            "radius": int(circle["radius"]),
        }
    else:
        return None, None, None

    if image_shape is None:
        if tuned_on_array == "images_ds":
            raw_video = root.get("raw_video")
            images_ds = raw_video.get("images_ds") if raw_video is not None else None
            if images_ds is not None and len(images_ds.shape) >= 3:
                image_shape = (int(images_ds.shape[1]), int(images_ds.shape[2]))
        if image_shape is None and tuned_on_array == "images_full":
            image_shape = _resolve_full_frame_shape(root)
    if image_shape is None:
        return None, None, None
    return _create_dish_mask_image(params, image_shape), str(tuned_on_array), image_shape


def _extract_dish_mask_roi(
    dish_mask_image: np.ndarray,
    *,
    tuned_on_array: str,
    top_left_xy_full: np.ndarray,
    roi_shape: tuple[int, int],
    full_shape: Optional[tuple[int, int]],
) -> np.ndarray:
    if tuned_on_array == "images_ds":
        if full_shape is None:
            raise ValueError("Cannot project a dish mask tuned on images_ds without a full-frame shape.")
        return _extract_background_roi_ds(
            dish_mask_image,
            top_left_xy_full,
            roi_shape,
            full_shape,
        )
    return _extract_background_roi_full(dish_mask_image, top_left_xy_full, roi_shape)


def _draw_subject_overlay(
    roi_image: np.ndarray,
    processed_mask: np.ndarray,
    stats: Dict[str, Any],
) -> np.ndarray:
    base = cv2.cvtColor(np.asarray(roi_image, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    overlay[processed_mask > 0] = (0, 255, 0)
    out = cv2.addWeighted(overlay, 0.35, base, 0.65, 0.0)
    bbox_xyxy = stats.get("bbox_xyxy")
    if isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
        cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 255), 1)
    return out


def _subject_component_preview(
    subject_source: SubjectMaskSource,
    *,
    roi_index: int,
    component_name: str,
    component_index: Optional[int],
) -> Dict[str, Any]:
    if component_name == "eyes_union" and component_index is None:
        left_idx = _find_subject_label_index(subject_source, "eye_left")
        right_idx = _find_subject_label_index(subject_source, "eye_right")
        if left_idx is None or right_idx is None:
            raise ValueError(f"subject_mask_runs/{subject_source.run_name} cannot derive eyes_union preview.")
        left_mask = np.asarray(subject_source.masks_roi[roi_index, left_idx], dtype=np.uint8)
        right_mask = np.asarray(subject_source.masks_roi[roi_index, right_idx], dtype=np.uint8)
        mask = np.maximum(left_mask, right_mask)
        preview: Dict[str, Any] = {
            "mask": mask,
            "component_name": "eyes_union",
            "run_name": subject_source.run_name,
            "derived": True,
            "derived_from": ["eye_left", "eye_right"],
        }
        if subject_source.mask_probs_roi is not None:
            left_prob = _decode_probability_image(
                np.asarray(subject_source.mask_probs_roi[roi_index, left_idx]),
                encoding=subject_source.probability_encoding,
            )
            right_prob = _decode_probability_image(
                np.asarray(subject_source.mask_probs_roi[roi_index, right_idx]),
                encoding=subject_source.probability_encoding,
            )
            preview["prob"] = np.maximum(left_prob, right_prob)
        return preview

    assert component_index is not None
    mask = np.asarray(subject_source.masks_roi[roi_index, component_index], dtype=np.uint8)
    preview: Dict[str, Any] = {
        "mask": mask,
        "component_name": component_name,
        "run_name": subject_source.run_name,
        "derived": False,
    }
    if subject_source.mask_probs_roi is not None:
        prob = np.asarray(subject_source.mask_probs_roi[roi_index, component_index])
        preview["prob"] = _decode_probability_image(prob, encoding=subject_source.probability_encoding)
    return preview


def _compute_eye_lr_preview(
    roi_image: np.ndarray,
    keypoints_row: np.ndarray,
    *,
    success_flag: bool,
    heading_deg: Optional[float],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    working_roi = np.asarray(roi_image, dtype=np.uint8)
    working_keypoints = np.asarray(keypoints_row, dtype=np.float32)
    rotation_applied: Optional[float] = None
    if bool(params.get("rotate_roi")) and heading_deg is not None:
        rotation_applied = -float(heading_deg)
        working_roi, working_keypoints = eye_mask_ops.rotate_image_and_points(
            working_roi,
            working_keypoints,
            rotation_applied,
        )

    masks: list[Optional[np.ndarray]] = [None, None]
    contours: list[Optional[np.ndarray]] = [None, None]
    regions_info: list[Optional[dict[str, Any]]] = [None, None]
    debug_panels: list[np.ndarray] = []
    eye_centers_roi: list[Optional[tuple[float, float]]] = [None, None]
    info_lines: list[str] = []

    if success_flag:
        for eye_idx in (0, 1):
            center = np.asarray(working_keypoints[1 + eye_idx], dtype=np.float32)
            cx, cy = float(center[0]), float(center[1])
            eye_label = "Left" if eye_idx == 0 else "Right"
            if not np.isfinite(cx) or not np.isfinite(cy):
                info_lines.append(f"{eye_label}: keypoint missing")
                debug_panels.append(_text_panel(f"{eye_label}\nkeypoint missing"))
                continue

            roi_h, roi_w = working_roi.shape
            padding = int(params["roi_padding"])
            x0 = max(0, int(round(cx)) - padding)
            x1 = min(roi_w, int(round(cx)) + padding + 1)
            y0 = max(0, int(round(cy)) - padding)
            y1 = min(roi_h, int(round(cy)) + padding + 1)

            patch = working_roi[y0:y1, x0:x1]
            if patch.size == 0:
                info_lines.append(f"{eye_label}: patch empty")
                debug_panels.append(_text_panel(f"{eye_label}\npatch empty"))
                continue

            filtered_patch, sobel_panel = eye_mask_ops.apply_sobel_filter(
                patch,
                float(params["sobel_strength"]),
            )
            binary, threshold_value = eye_mask_ops.global_mask(filtered_patch)
            if params.get("pre_threshold") is not None:
                binary = np.logical_and(binary, filtered_patch < int(params["pre_threshold"]))

            region_mask = eye_mask_ops.select_region(
                binary,
                (cx - x0, cy - y0),
                int(params["min_area"]),
                int(params["max_area"]) if params.get("max_area") is not None else None,
                float(params["min_circularity"]) if params.get("min_circularity") is not None else None,
                int(params["closing_radius"]),
                int(params["opening_radius"]),
            )
            debug_panels.append(
                eye_mask_ops.create_debug_panel(
                    patch,
                    filtered_patch,
                    binary,
                    region_mask,
                    (cx - x0, cy - y0),
                    eye_label,
                    int(params["pre_threshold"]) if params.get("pre_threshold") is not None else None,
                    sobel_panel=sobel_panel,
                )
            )

            if region_mask is None:
                info_lines.append(f"{eye_label}: no region")
                continue

            full_mask = np.zeros_like(working_roi, dtype=np.uint8)
            full_mask[y0:y1, x0:x1][region_mask] = 1
            masks[eye_idx] = full_mask

            region = measure.regionprops(region_mask.astype(int))[0]
            regions_info[eye_idx] = {
                "centroid": (region.centroid[0] + y0, region.centroid[1] + x0),
                "orientation": region.orientation,
                "major_axis_length": region.major_axis_length,
                "minor_axis_length": region.minor_axis_length,
            }
            eye_centers_roi[eye_idx] = (
                float(region.centroid[1] + x0),
                float(region.centroid[0] + y0),
            )

            info_line = (
                f"{eye_label}: area={region.area:.0f} "
                f"major={region.major_axis_length:.1f} minor={region.minor_axis_length:.1f} "
                f"thr={threshold_value:.1f}"
            )
            perimeter = float(region.perimeter)
            if perimeter > 0.0:
                circularity = float((4.0 * np.pi * float(region.area)) / (perimeter * perimeter))
                info_line += f" circ={circularity:.2f}"
            info_lines.append(info_line)

            contour = measure.find_contours(region_mask.astype(float), 0.5)
            if contour:
                best = max(contour, key=lambda c: c.shape[0])
                if best.shape[0] >= 5:
                    best = best[:, ::-1]
                    best[:, 0] += x0
                    best[:, 1] += y0
                    contours[eye_idx] = best
    else:
        info_lines.append("Keypoints failed for this ROI")

    overlay = eye_mask_ops.draw_overlay(
        working_roi,
        (masks[0], masks[1]),
        (contours[0], contours[1]),
        (regions_info[0], regions_info[1]),
    )
    union_mask = np.zeros_like(working_roi, dtype=np.uint8)
    for mask in masks:
        if mask is not None:
            union_mask = np.maximum(union_mask, np.asarray(mask, dtype=np.uint8))

    if eye_centers_roi[0] is not None and eye_centers_roi[1] is not None:
        separation = float(
            np.hypot(
                eye_centers_roi[0][0] - eye_centers_roi[1][0],
                eye_centers_roi[0][1] - eye_centers_roi[1][1],
            )
        )
        info_lines.append(f"Eye separation: {separation:.1f}px")
    else:
        separation = None

    if rotation_applied is not None:
        overlay = eye_mask_ops.apply_circular_window(overlay)
        union_mask = eye_mask_ops.apply_circular_window(union_mask)

    while len(debug_panels) < 2:
        debug_panels.append(_text_panel("No debug panel"))

    return {
        "roi_image": working_roi,
        "overlay": overlay,
        "union_mask": union_mask,
        "debug_panels": debug_panels[:2],
        "info_lines": info_lines,
        "separation_px": separation,
        "rotation_applied": rotation_applied,
        "success_flag": bool(success_flag),
    }


def _render_eye_union_panel(
    *,
    preview: Mapping[str, Any],
    panel_size: int,
    crop_meta: Mapping[str, Any],
    selected_component_name: str,
    subject_preview: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    stored_title = f"Stored {selected_component_name}"
    stored_mask = np.zeros_like(np.asarray(preview["union_mask"], dtype=np.uint8))
    if subject_preview is not None:
        stored_mask = np.asarray(subject_preview["mask"], dtype=np.uint8)
        derived_note = " (derived)" if bool(subject_preview.get("derived")) else ""
        stored_title = f"Stored {subject_preview['component_name']}{derived_note} ({subject_preview['run_name']})"
        if preview.get("rotation_applied") is not None:
            stored_mask = _rotate_image_nearest(stored_mask, float(preview["rotation_applied"]))
            stored_mask = eye_mask_ops.apply_circular_window(stored_mask)

    panels = [
        _prepare_panel(np.asarray(preview["roi_image"], dtype=np.uint8), "Crop ROI", panel_size),
        _prepare_panel(stored_mask * 255, stored_title, panel_size, nearest=True),
        _prepare_panel(np.asarray(preview["overlay"]), "Eye Left/Right Overlay", panel_size),
        _prepare_panel(np.asarray(preview["union_mask"], dtype=np.uint8) * 255, "Eye Union Proposal", panel_size, nearest=True),
    ]
    top_row = cv2.hconcat(panels[:2])
    bottom_row = cv2.hconcat(panels[2:])
    grid = cv2.vconcat([top_row, bottom_row])

    footer = np.zeros((140, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        footer,
        (
            f"ROI {int(crop_meta['roi_index']) + 1}/{int(crop_meta['total_rois'])} | "
            f"frame={crop_meta.get('frame_index')} | detection={crop_meta.get('detection_index')}"
        ),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    status_line = " | ".join(str(line) for line in list(preview.get("info_lines", []))[:2]) or "No eye preview information."
    cv2.putText(
        footer,
        status_line[: max(10, min(len(status_line), 140))],
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    extra_line = " | ".join(str(line) for line in list(preview.get("info_lines", []))[2:4])
    if extra_line:
        cv2.putText(
            footer,
            extra_line[: max(10, min(len(extra_line), 140))],
            (10, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        footer,
        (
            f"Eye method controls ({selected_component_name} view): padding, pre-threshold, Sobel, "
            "area, circularity, close/open, eye gap, rotate | debug panels in separate Eye Debug window"
        ),
        (10, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "Controls: n/p next-prev, j/k +/-10 ROI, s save eye-method tuning + compatibility mirror, q/ESC quit",
        (10, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([grid, footer])


def _render_eye_debug_display(preview: Mapping[str, Any]) -> np.ndarray:
    debug_panels = [np.asarray(panel, dtype=np.uint8) for panel in list(preview.get("debug_panels", [])) if panel is not None]
    if not debug_panels:
        empty = np.zeros((180, 720, 3), dtype=np.uint8)
        cv2.putText(
            empty,
            "No eye debug panels available",
            (18, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return empty
    if len(debug_panels) == 1:
        return debug_panels[0]
    return np.vstack(debug_panels[:2])


def _render_subject_mask_panel(
    *,
    roi_image: np.ndarray,
    background_roi: np.ndarray,
    params: Dict[str, Any],
    dish_mask_roi: Optional[np.ndarray],
    panel_size: int,
    crop_meta: Dict[str, Any],
    component_name: str,
    subject_preview: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    preview = _compute_subject_mask_seed(
        roi_image,
        background_roi,
        params,
        allowed_mask=dish_mask_roi,
    )
    overlay = _draw_subject_overlay(roi_image, preview["processed_mask"], preview["stats"])
    existing_mask = None
    existing_title = f"Stored {component_name}"
    if subject_preview is not None:
        existing_mask = np.asarray(subject_preview["mask"], dtype=np.uint8)
        existing_title = f"Stored {component_name} ({subject_preview['run_name']})"
    else:
        existing_mask = np.zeros_like(preview["processed_mask"], dtype=np.uint8)
    dish_panel = (
        np.asarray(dish_mask_roi, dtype=np.uint8) * 255
        if dish_mask_roi is not None
        else np.zeros_like(preview["processed_mask"], dtype=np.uint8)
    )
    dish_title = "Dish Mask ROI" if dish_mask_roi is not None else "Dish Mask ROI (none)"

    panels = [
        _prepare_panel(roi_image, "Crop ROI", panel_size),
        _prepare_panel(background_roi, "Background ROI", panel_size),
        _prepare_panel(dish_panel, dish_title, panel_size, nearest=True),
        _prepare_panel(preview["diff_raw"], "Traditional Diff (BG - Crop)", panel_size),
        _prepare_panel(
            preview["processed_mask"] * 255,
            f"Traditional Proposal {component_name}",
            panel_size,
            nearest=True,
        ),
        _prepare_panel(existing_mask * 255, existing_title, panel_size, nearest=True),
    ]
    top_row = cv2.hconcat(panels[:3])
    bottom_row = cv2.hconcat(panels[3:])
    middle_row = cv2.hconcat(
        [
            _prepare_panel(preview["seed_mask"] * 255, "Thresholded Seed", panel_size, nearest=True),
            _prepare_panel(overlay, "Traditional Proposal Overlay", panel_size),
            np.zeros_like(_prepare_panel(overlay, "Traditional Proposal Overlay", panel_size)),
        ]
    )
    grid = cv2.vconcat([top_row, bottom_row, middle_row])

    footer = np.zeros((112, grid.shape[1], 3), dtype=np.uint8)
    stats = preview["stats"]
    frame_text = crop_meta.get("frame_index")
    detect_text = crop_meta.get("detection_index")
    cv2.putText(
        footer,
        (
            f"ROI {crop_meta['roi_index'] + 1}/{crop_meta['total_rois']} | frame={frame_text} | "
            f"detection={detect_text} | component={component_name}"
        ),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        (
            f"manual_thr={int(params['diff_threshold'])} | otsu={int(preview['otsu_threshold'])} | "
            f"blur={int(params['gaussian_blur_kernel'])} | close={int(params['closing_radius'])} | "
            f"open={int(params['opening_radius'])} | min_area={int(params['min_area'])} | "
            f"largest_only={'yes' if params['keep_largest_component'] else 'no'} | "
            f"dish_mask={'on' if dish_mask_roi is not None else 'off'}"
        ),
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        (
            f"components={int(stats['candidate_components'])} | kept={int(stats['kept_components'])} | "
            f"largest_area={int(stats['largest_area'])} | kept_area={int(stats['kept_area'])}"
        ),
        (10, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "Controls: n/p next-prev, j/k +/-10 ROI, s save traditional tuning, q/ESC quit",
        (10, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([grid, footer])


def save_subject_mask_params(
    zarr_path: str | Path,
    params: Dict[str, Any],
    *,
    context: Dict[str, Any],
    component_name: str = DEFAULT_COMPONENT,
    method: Optional[str] = None,
    version: str = "1.0",
    extra_entry_fields: Optional[Dict[str, Any]] = None,
    mirror_eye_tuning_compat: bool = False,
    normalize_params: bool = True,
) -> tuple[bool, str]:
    try:
        normalized_component = _normalize_component_name(component_name) or DEFAULT_COMPONENT
        normalized_params = (
            _normalize_subject_component_params(normalized_component, params)
            if normalize_params
            else dict(params)
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        root = open_zarr_root(zarr_path, mode="r+")
        analysis = root.require_group("analysis_metadata")
        metadata = dict(analysis.attrs) if analysis.attrs else {}
        normalized_context = dict(context)
        if "component_name" in normalized_context and "storage_component_name" not in normalized_context:
            normalized_context["storage_component_name"] = normalized_context.pop("component_name")

        subject_payload = _normalize_subject_tuning_payload(metadata.get(TUNING_KEY))
        components = dict(subject_payload.get("components", {}))
        if normalized_component == "eyes_union":
            entry = _build_subject_eye_tuning_entry(
                tuned_parameters=normalized_params,
                context=normalized_context,
                timestamp=timestamp,
                version=str(version),
                method=method,
                extra_entry_fields=extra_entry_fields,
            )
        else:
            entry = {
                "method": method or "traditional_subject_mask_seed",
                "version": str(version),
                "tuned_timestamp": timestamp,
                "tuned_parameters": normalized_params,
                "context": normalized_context,
            }
            if extra_entry_fields:
                entry.update(dict(extra_entry_fields))
        components[normalized_component] = entry
        subject_payload["version"] = SUBJECT_TUNING_VERSION
        subject_payload["components"] = components
        subject_payload["latest_component"] = normalized_component
        subject_payload["latest_timestamp"] = timestamp
        metadata[TUNING_KEY] = subject_payload

        if mirror_eye_tuning_compat:
            metadata[EYE_TUNING_COMPAT_KEY] = {
                "method": "global_threshold_otsu",
                "version": "1.0",
                "tuned_timestamp": timestamp,
                "tuned_parameters": {
                    key: value
                    for key, value in normalized_params.items()
                    if key != "rotate_roi"
                },
                "context": dict(context),
            }

        analysis.attrs.update(metadata)
        if mirror_eye_tuning_compat:
            return True, "eyes_union tuning saved with eye_mask_tuning compatibility mirror"
        return True, f"{normalized_component} tuning saved"
    except Exception as exc:
        return False, f"Error saving parameters: {exc}"


def _run_eye_union_component_tuner(
    zarr_path: str | Path,
    *,
    root: zarr.Group,
    crop_group: zarr.Group,
    resolved_crop_run: str,
    subject_source: Optional[SubjectMaskSource],
    component_name: str,
    component_index: Optional[int],
    keypoint_run: Optional[str],
    roi_index: int,
    panel_size: int,
    window_name: str,
) -> int:
    keypoint_source = _resolve_eye_keypoint_source(root, keypoint_run)
    roi_images = crop_group["roi_images"]
    frame_indices = crop_group.get("frame_indices")
    detection_indices = crop_group.get("detection_indices")
    total_rois = int(roi_images.shape[0])
    if total_rois <= 0:
        raise ValueError(f"crop_runs/{resolved_crop_run}/roi_images is empty.")

    current_eye_roi_index = int(np.clip(roi_index, 0, max(0, total_rois - 1)))
    initial_params = _build_eye_union_params(_load_tuned_params_from_root(root, "eyes_union"))
    selected_component_name = component_name
    debug_window_name = f"{window_name}{DEBUG_WINDOW_SUFFIX}"
    debug_window_sized = False

    def update_eye_roi_index(val: int) -> None:
        nonlocal current_eye_roi_index
        current_eye_roi_index = max(0, min(int(val), total_rois - 1))

    def eye_params_from_trackbars() -> Dict[str, Any]:
        pre_thresh_val = cv2.getTrackbarPos("PreThresh", window_name)
        max_area_slider = cv2.getTrackbarPos("Max Area", window_name)
        min_circularity_slider = cv2.getTrackbarPos("Min Circ %", window_name)
        max_gap_slider = cv2.getTrackbarPos("Max Gap", window_name)
        return {
            "roi_padding": cv2.getTrackbarPos("ROI Padding", window_name),
            "pre_threshold": pre_thresh_val if pre_thresh_val > 0 else None,
            "sobel_strength": cv2.getTrackbarPos("Sobel %", window_name) / float(eye_mask_ops.SLIDER_MAX_SOBEL),
            "min_area": cv2.getTrackbarPos("Min Area", window_name),
            "max_area": max_area_slider if max_area_slider > 0 else None,
            "min_circularity": (
                float(min_circularity_slider) / float(eye_mask_ops.SLIDER_MAX_CIRCULARITY)
                if min_circularity_slider > 0
                else None
            ),
            "closing_radius": cv2.getTrackbarPos("Closing r", window_name),
            "opening_radius": cv2.getTrackbarPos("Opening r", window_name),
            "min_eye_separation": float(cv2.getTrackbarPos("Min Gap", window_name)) or None,
            "max_eye_separation": float(max_gap_slider) if max_gap_slider > 0 else None,
            "rotate_roi": cv2.getTrackbarPos("Rotate ROI", window_name) > 0,
        }

    _require_gui_display()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    _set_initial_window_size(window_name, panel_size=panel_size, footer_height=140, columns=2, rows=2)
    cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar("ROI", window_name, current_eye_roi_index, max(0, total_rois - 1), update_eye_roi_index)
    cv2.createTrackbar(
        "ROI Padding",
        window_name,
        min(int(initial_params["roi_padding"]), eye_mask_ops.SLIDER_MAX_PADDING),
        eye_mask_ops.SLIDER_MAX_PADDING,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "PreThresh",
        window_name,
        int(initial_params["pre_threshold"]) if initial_params["pre_threshold"] is not None else 0,
        eye_mask_ops.SLIDER_MAX_PRETHRESH,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Sobel %",
        window_name,
        int(float(initial_params["sobel_strength"]) * eye_mask_ops.SLIDER_MAX_SOBEL),
        eye_mask_ops.SLIDER_MAX_SOBEL,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Min Area",
        window_name,
        min(int(initial_params["min_area"]), eye_mask_ops.SLIDER_MAX_AREA),
        eye_mask_ops.SLIDER_MAX_AREA,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Max Area",
        window_name,
        int(initial_params["max_area"]) if initial_params["max_area"] is not None else 0,
        eye_mask_ops.SLIDER_MAX_AREA,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Min Circ %",
        window_name,
        int(round(float(initial_params["min_circularity"]) * eye_mask_ops.SLIDER_MAX_CIRCULARITY))
        if initial_params["min_circularity"] is not None
        else 0,
        eye_mask_ops.SLIDER_MAX_CIRCULARITY,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Closing r",
        window_name,
        min(int(initial_params["closing_radius"]), eye_mask_ops.SLIDER_MAX_RADIUS),
        eye_mask_ops.SLIDER_MAX_RADIUS,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Opening r",
        window_name,
        min(int(initial_params["opening_radius"]), eye_mask_ops.SLIDER_MAX_RADIUS),
        eye_mask_ops.SLIDER_MAX_RADIUS,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Min Gap",
        window_name,
        int(initial_params["min_eye_separation"]) if initial_params["min_eye_separation"] is not None else 0,
        eye_mask_ops.SLIDER_MAX_EYE_GAP,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Max Gap",
        window_name,
        int(initial_params["max_eye_separation"]) if initial_params["max_eye_separation"] is not None else 0,
        eye_mask_ops.SLIDER_MAX_EYE_GAP,
        lambda _val: None,
    )
    cv2.createTrackbar(
        "Rotate ROI",
        window_name,
        1 if bool(initial_params["rotate_roi"]) else 0,
        1,
        lambda _val: None,
    )

    print("\nStarting Subject Mask Tuner")
    print(f"  crop_run={resolved_crop_run}")
    print(f"  keypoint_run={keypoint_source.group_name}/{keypoint_source.run_name}")
    if subject_source is not None:
        print(f"  subject_run={subject_source.run_name}")
    print(f"  component={selected_component_name}")
    if subject_source is not None and component_index is None:
        print("  preview=stored derived eyes_union mask + left/right eye proposal")
    else:
        print(f"  preview=stored {selected_component_name} mask + left/right eye proposal")
    print("Controls:")
    print("  - ROI trackbar or n/p, j/k to navigate crops")
    print("  - Adjust eye-mask parameters for automated left/right eye segmentation")
    print(f"  - Left/right debug panels open in a second window: {debug_window_name}")
    print("  - Press 's' to save eye-method tuning and mirror eye_mask_tuning for compatibility")
    print("  - Press 'q' or ESC to quit")

    try:
        while True:
            roi_idx = int(np.clip(current_eye_roi_index, 0, total_rois - 1))
            roi_image = np.asarray(roi_images[roi_idx], dtype=np.uint8)
            keypoints_row = np.asarray(keypoint_source.keypoints_roi[roi_idx], dtype=np.float32)
            success_flag = bool(keypoint_source.success_flags[roi_idx])
            heading_deg: Optional[float] = None
            if keypoint_source.heading_values is not None:
                heading_val = float(keypoint_source.heading_values[roi_idx])
                if np.isfinite(heading_val):
                    heading_deg = heading_val
            if heading_deg is None:
                heading_deg = eye_mask_ops.compute_heading_deg(keypoints_row)

            frame_index = int(frame_indices[roi_idx]) if frame_indices is not None else None
            detection_index = int(detection_indices[roi_idx]) if detection_indices is not None else None
            subject_preview = None
            if subject_source is not None and component_index is not None:
                subject_preview = _subject_component_preview(
                    subject_source,
                    roi_index=roi_idx,
                    component_name=selected_component_name,
                    component_index=component_index,
                )
            elif subject_source is not None:
                subject_preview = _subject_component_preview(
                    subject_source,
                    roi_index=roi_idx,
                    component_name="eyes_union",
                    component_index=None,
                )

            params = eye_params_from_trackbars()
            preview = _compute_eye_lr_preview(
                roi_image,
                keypoints_row,
                success_flag=success_flag,
                heading_deg=heading_deg,
                params=params,
            )
            panel = _render_eye_union_panel(
                preview=preview,
                panel_size=panel_size,
                selected_component_name=selected_component_name,
                subject_preview=subject_preview,
                crop_meta={
                    "roi_index": roi_idx,
                    "total_rois": total_rois,
                    "frame_index": frame_index if frame_index is not None else "n/a",
                    "detection_index": detection_index if detection_index is not None else "n/a",
                },
            )
            debug_display = _render_eye_debug_display(preview)
            cv2.imshow(window_name, panel)
            cv2.imshow(debug_window_name, debug_display)
            if not debug_window_sized:
                cv2.resizeWindow(
                    debug_window_name,
                    max(900, int(debug_display.shape[1])),
                    max(700, int(debug_display.shape[0])),
                )
                debug_window_sized = True
            key = cv2.waitKeyEx(30)

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("n"), ord("N")):
                current_eye_roi_index = min(total_rois - 1, current_eye_roi_index + 1)
                cv2.setTrackbarPos("ROI", window_name, current_eye_roi_index)
                continue
            if key in (ord("p"), ord("P")):
                current_eye_roi_index = max(0, current_eye_roi_index - 1)
                cv2.setTrackbarPos("ROI", window_name, current_eye_roi_index)
                continue
            if key == ord("j"):
                current_eye_roi_index = min(total_rois - 1, current_eye_roi_index + 10)
                cv2.setTrackbarPos("ROI", window_name, current_eye_roi_index)
                continue
            if key == ord("k"):
                current_eye_roi_index = max(0, current_eye_roi_index - 10)
                cv2.setTrackbarPos("ROI", window_name, current_eye_roi_index)
                continue
            if key in (ord("s"), ord("S")):
                save_ok, message = save_subject_mask_params(
                    zarr_path,
                    params,
                    component_name="eyes_union",
                    method="global_threshold_otsu",
                    mirror_eye_tuning_compat=True,
                    extra_entry_fields={
                        "migrated_from_eye_mask_tuning": False,
                    },
                    context={
                        "crop_run": resolved_crop_run,
                        "keypoint_run": keypoint_source.run_name,
                        "keypoint_source": keypoint_source.group_name,
                        "subject_run": subject_source.run_name if subject_source is not None else None,
                        "storage_component_name": "eyes_union",
                        "view_component_name": selected_component_name,
                        "roi_index": roi_idx,
                        "frame_index": frame_index,
                        "detection_index": detection_index,
                        "roi_success": success_flag,
                    },
                )
                if save_ok:
                    print(f"✓ {message}")
                    print(f"  Saved as analysis_metadata.attrs['{TUNING_KEY}'].components['eyes_union']")
                    print(f"  Mirrored to analysis_metadata.attrs['{EYE_TUNING_COMPAT_KEY}'] for compatibility")
                else:
                    print(f"✗ {message}")
                continue
    finally:
        cv2.destroyWindow(window_name)
        cv2.destroyWindow(debug_window_name)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument("--config", type=Path, help="Optional config YAML with subject_masks defaults.")
    parser.add_argument("--crop-run", help="Crop run name (default: latest).")
    parser.add_argument("--background-run", help="Background run name (default: latest).")
    parser.add_argument(
        "--keypoint-run",
        help="Keypoint run for eyes_union mode (default: latest refined_keypoints/keypoints run).",
    )
    parser.add_argument("--subject-run", help="Subject mask run name (default: latest if subject_mask_runs exists).")
    parser.add_argument(
        "--component",
        help="Canonical or alias component name (subject_body, eyes_union, eye_left, eye_right, swim_bladder).",
    )
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index (default: 0).")
    parser.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE, help="Per-panel display size.")
    parser.add_argument("--window-name", default=WINDOW_NAME, help="OpenCV window name.")
    return parser


def run(
    zarr_path: str | Path,
    *,
    config_path: Optional[str | Path] = None,
    crop_run: Optional[str] = None,
    background_run: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    subject_run: Optional[str] = None,
    component: Optional[str] = None,
    roi_index: int = 0,
    panel_size: int = DEFAULT_PANEL_SIZE,
    window_name: str = WINDOW_NAME,
) -> int:
    global current_roi_index
    current_roi_index = max(0, int(roi_index))

    _load_initial_params(Path(config_path) if config_path else None)

    root = open_zarr_root(zarr_path, mode="r")
    subject_source = _load_subject_mask_source(root, subject_run)
    component_name, component_index = _resolve_subject_component(subject_source, component)

    resolved_crop_run = _resolve_run_name(root, "crop_runs", crop_run)
    crop_group = root[f"crop_runs/{resolved_crop_run}"]

    if "roi_images" not in crop_group:
        raise ValueError(f"crop_runs/{resolved_crop_run} missing roi_images.")

    roi_images = crop_group["roi_images"]
    total_rois = int(roi_images.shape[0])
    if total_rois <= 0:
        raise ValueError(f"crop_runs/{resolved_crop_run}/roi_images is empty.")
    current_roi_index = int(np.clip(current_roi_index, 0, max(0, total_rois - 1)))
    panel_size = max(96, int(panel_size))

    if _is_eye_component(component_name):
        return _run_eye_union_component_tuner(
            zarr_path,
            root=root,
            crop_group=crop_group,
            resolved_crop_run=resolved_crop_run,
            subject_source=subject_source,
            component_name=component_name,
            component_index=component_index,
            keypoint_run=keypoint_run,
            roi_index=current_roi_index,
            panel_size=panel_size,
            window_name=window_name,
        )

    _apply_subject_mask_params(_load_tuned_params_from_root(root, component_name))

    resolved_background_run = _resolve_run_name(root, "background_runs", background_run)
    bg_source = _resolve_background_source(root, resolved_background_run)
    bg_group = root[f"background_runs/{resolved_background_run}"]
    if "roi_coordinates_full" not in crop_group:
        raise ValueError(f"crop_runs/{resolved_crop_run} missing roi_coordinates_full.")
    roi_coords = crop_group["roi_coordinates_full"]
    frame_indices = crop_group.get("frame_indices")
    detection_indices = crop_group.get("detection_indices")
    bg_array = bg_group[bg_source.array_name]
    dish_mask_image, dish_mask_array_name, _dish_mask_image_shape = _resolve_dish_mask_projection(root)
    full_frame_shape = _resolve_full_frame_shape(root)

    _require_gui_display()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    _set_initial_window_size(window_name, panel_size=panel_size, footer_height=112, columns=3, rows=3)
    cv2.createTrackbar("ROI", window_name, current_roi_index, max(0, total_rois - 1), update_roi_index)
    cv2.createTrackbar("Threshold", window_name, int(diff_threshold), 255, update_threshold)
    cv2.createTrackbar("Blur", window_name, int(gaussian_blur_kernel), TRACKBAR_BLUR_MAX, update_blur_kernel)
    cv2.createTrackbar("Close", window_name, int(closing_radius), TRACKBAR_RADIUS_MAX, update_closing_radius)
    cv2.createTrackbar("Open", window_name, int(opening_radius), TRACKBAR_RADIUS_MAX, update_opening_radius)
    cv2.createTrackbar("Min Area", window_name, int(min_area), TRACKBAR_MIN_AREA_MAX, update_min_area)
    cv2.createTrackbar("Largest Only", window_name, int(bool(keep_largest_component)), 1, update_keep_largest_component)

    print("\nStarting Subject Mask Tuner")
    print(f"  crop_run={resolved_crop_run}")
    print(f"  background_run={resolved_background_run} ({bg_source.array_name})")
    if dish_mask_image is not None:
        print(f"  dish_mask={dish_mask_array_name}")
    else:
        print("  dish_mask=none")
    if subject_source is not None:
        print(f"  subject_run={subject_source.run_name}")
        print(f"  component={component_name}")
        print("  preview=stored subject-mask channel + dish-masked traditional proposal")
    else:
        print(f"  component={component_name} (dish-masked traditional-proposal-only mode)")
    print("Controls:")
    print("  - ROI trackbar or n/p, j/k to navigate crops")
    print("  - Adjust threshold, blur, morphology, and area sliders for the traditional proposal")
    print("  - Press 's' to save traditional subject-mask tuning to analysis_metadata")
    print("  - Press 'q' or ESC to quit")

    try:
        while True:
            roi_idx = int(np.clip(current_roi_index, 0, total_rois - 1))
            roi_image = np.asarray(roi_images[roi_idx], dtype=np.uint8)
            top_left_xy = np.asarray(roi_coords[roi_idx], dtype=np.float64)
            roi_shape = (int(roi_image.shape[0]), int(roi_image.shape[1]))

            if bg_source.array_name == "background_full":
                background_roi = _extract_background_roi_full(bg_array, top_left_xy, roi_shape)
            else:
                assert bg_source.full_shape is not None
                background_roi = _extract_background_roi_ds(
                    bg_array,
                    top_left_xy,
                    roi_shape,
                    bg_source.full_shape,
                )
            background_roi = np.asarray(background_roi, dtype=np.uint8)
            dish_mask_roi = None
            if dish_mask_image is not None and dish_mask_array_name is not None:
                dish_mask_roi = np.asarray(
                    _extract_dish_mask_roi(
                        dish_mask_image,
                        tuned_on_array=dish_mask_array_name,
                        top_left_xy_full=top_left_xy,
                        roi_shape=roi_shape,
                        full_shape=full_frame_shape,
                    ),
                    dtype=np.uint8,
                )

            frame_index = int(frame_indices[roi_idx]) if frame_indices is not None else None
            detection_index = int(detection_indices[roi_idx]) if detection_indices is not None else None
            subject_preview = None
            if subject_source is not None and component_index is not None:
                subject_preview = _subject_component_preview(
                    subject_source,
                    roi_index=roi_idx,
                    component_name=component_name,
                    component_index=component_index,
                )
            panel = _render_subject_mask_panel(
                roi_image=roi_image,
                background_roi=background_roi,
                params=_current_params(),
                dish_mask_roi=dish_mask_roi,
                panel_size=panel_size,
                component_name=component_name,
                subject_preview=subject_preview,
                crop_meta={
                    "roi_index": roi_idx,
                    "total_rois": total_rois,
                    "frame_index": frame_index if frame_index is not None else "n/a",
                    "detection_index": detection_index if detection_index is not None else "n/a",
                },
            )
            cv2.imshow(window_name, panel)
            key = cv2.waitKeyEx(30)

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("n"), ord("N")):
                current_roi_index = min(total_rois - 1, current_roi_index + 1)
                cv2.setTrackbarPos("ROI", window_name, current_roi_index)
                continue
            if key in (ord("p"), ord("P")):
                current_roi_index = max(0, current_roi_index - 1)
                cv2.setTrackbarPos("ROI", window_name, current_roi_index)
                continue
            if key == ord("j"):
                current_roi_index = min(total_rois - 1, current_roi_index + 10)
                cv2.setTrackbarPos("ROI", window_name, current_roi_index)
                continue
            if key == ord("k"):
                current_roi_index = max(0, current_roi_index - 10)
                cv2.setTrackbarPos("ROI", window_name, current_roi_index)
                continue
            if key in (ord("s"), ord("S")):
                save_ok, message = save_subject_mask_params(
                    zarr_path,
                    _current_params(),
                    component_name=component_name,
                    context={
                        "crop_run": resolved_crop_run,
                        "background_run": resolved_background_run,
                        "background_array": bg_source.array_name,
                        "subject_run": subject_source.run_name if subject_source is not None else None,
                        "storage_component_name": component_name,
                        "roi_index": roi_idx,
                        "frame_index": frame_index,
                        "detection_index": detection_index,
                    },
                )
                if save_ok:
                    print(f"✓ {message}")
                    print(f"  Saved as analysis_metadata.attrs['{TUNING_KEY}']")
                else:
                    print(f"✗ {message}")
                continue
    finally:
        cv2.destroyWindow(window_name)

    return 0


def main(
    zarr_path: Optional[str] = None,
    *,
    config_path: Optional[str] = None,
    keypoint_run: Optional[str] = None,
    subject_run: Optional[str] = None,
    component: Optional[str] = None,
    roi_index: int = 0,
    crop_run: Optional[str] = None,
    background_run: Optional[str] = None,
    panel_size: int = DEFAULT_PANEL_SIZE,
    window_name: str = WINDOW_NAME,
    argv: Optional[Iterable[str]] = None,
) -> int:
    if argv is not None or zarr_path is None:
        parser = build_parser()
        args = parser.parse_args(list(argv) if argv is not None else None)
        return run(
            args.zarr_path,
            config_path=str(args.config) if args.config else None,
            crop_run=args.crop_run,
            background_run=args.background_run,
            keypoint_run=args.keypoint_run,
            subject_run=args.subject_run,
            component=args.component,
            roi_index=int(args.roi_index),
            panel_size=int(args.panel_size),
            window_name=str(args.window_name),
        )
    return run(
        zarr_path,
        config_path=config_path,
        crop_run=crop_run,
        background_run=background_run,
        keypoint_run=keypoint_run,
        subject_run=subject_run,
        component=component,
        roi_index=roi_index,
        panel_size=panel_size,
        window_name=window_name,
    )


if __name__ == "__main__":
    raise SystemExit(main(argv=None))
