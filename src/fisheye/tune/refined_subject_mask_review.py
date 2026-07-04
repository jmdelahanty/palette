"""Manual review/editor for refined subject-mask runs."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
from ..shared.mask_store import (
    MaskStore,
    MaskStoreError,
    mark_mask_rle_stale_attrs,
    materialize_dense_masks_roi_from_store,
    open_mask_store,
    refresh_bitpacked_mask_store_from_dense,
)
from ..shared.mask_geometry import batch_mask_spatial_metrics
from ..shared.mask_probability_encoding import (
    decode_probability_values,
    probabilities_encoding_from_attrs,
)
from ..shared.provenance_attrs import (
    build_source_crop_snapshot_attrs,
    build_source_keypoints_attrs,
    extract_source_crop_snapshot_attrs,
    resolve_assignment_keypoint_group,
    resolve_assignment_keypoints_run,
    resolve_source_keypoints_run,
)
from ..shared.refined_subject_component_contours import refresh_component_contour_rows_from_masks
from ..shared.row_lineage import copy_row_lineage_arrays_from_sources, resolve_source_crop_row_ids
from ..shared.subject_mask_chunks import (
    refined_subject_mask_metric_row_chunk,
    refined_subject_mask_storage_chunks,
)
from ..shared.subject_mask_stale import sync_source_subject_mask_stale_payload
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
    set_authoritative_run,
)
from ..shared.system_metadata import get_environment_info, get_git_info
from ..utils.zarr_io import open_zarr_root

try:
    cv2_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
except (TypeError, ValueError):
    cv2_threads = 2
cv2.setNumThreads(cv2_threads)

WINDOW_NAME = "Refined Subject Mask Review"
EDIT_WINDOW_NAME = "Refined Subject Mask Edit Patch"
DEFAULT_COMPONENTS = ("subject_body", "swim_bladder")
REVIEW_STATE_CHOICES = ("approved", "pending", "rejected", "needs_review")
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
DISPLAY_SCALE = 1.0
DEFAULT_EDIT_ZOOM = 4
MAX_EDIT_ZOOM = 12
DEFAULT_EDIT_PADDING = 32
MAX_EDIT_PADDING = 192
MIN_WINDOW_SCALE_PERCENT = 50
MAX_WINDOW_SCALE_PERCENT = 300
PANEL_GAP = 8
PANEL_LABEL_HEIGHT = 18
PATCH_PAN_STEP = 8
PATCH_PAN_STEP_LARGE = 24
DEFAULT_BRUSH_RADIUS = 6
MAX_BRUSH_RADIUS = 64
LASSO_OUTLINE_COLOR = (0, 0, 255)
LASSO_PREVIEW_COLOR = (64, 64, 255)
LASSO_START_COLOR = (255, 255, 255)
LASSO_LINE_THICKNESS = 2
LASSO_PREVIEW_THICKNESS = 2
LASSO_POINT_RADIUS = 2
LASSO_MIN_POINT_STEP_PX = 2
DEFAULT_REVIEW_METHOD = "manual"
DEFAULT_REVIEW_INTENDED_USE = "training"
DEFAULT_RUN_METHOD = "refined_subject_mask_manual_review_v1"
REFINED_SUBJECT_STAGE_NAME = "refine_subject_masks"
REFINED_SUBJECT_SYNC_METHOD = "sync_refined_subject_mask_metadata"
REFINED_SUBJECT_SOURCE_UPDATE_METHOD = "check_refined_subject_source_updates"
REFINED_SUBJECT_WRITEBACK_METHOD = "palette_write_refined_subject_mask_edit"
DEFAULT_REFINED_SUBJECT_WRITEBACK_REASON = "crimson_refined_subject_mask_edit"
DEFAULT_SUBJECT_BODY_COMPONENT_SCHEMA_ID = "subject_body_v1"
DEFAULT_SUBJECT_BODY_ANATOMICAL_SCOPE = "body_core"
DEFAULT_SUBJECT_BODY_PECTORAL_FIN_POLICY = "excluded_or_unresolved"
REFINED_SUBJECT_SOURCE_SYNC_SCHEMA_ID = "refined_subject_source_sync_v1"
SIGMA_NOISE_SMOOTHING_SIGMA = 3.0
CURVATURE_VAR_SMOOTHING_SIGMA = 3.0
CURVATURE_VAR_STEP = 5
REFINED_EYE_COMPONENTS = ("eye_left", "eye_right")


@dataclass(frozen=True)
class SourceSubjectMaskRun:
    run_name: str
    group: zarr.Group
    crop_run: str
    source_crop_snapshot: dict[str, object]
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
    assignment_keypoints_run: Optional[str] = None
    assignment_keypoint_group: Optional[str] = None
    source_crop_row_ids: Any | None = None
    source_refined_row_ids: Any | None = None
    source_detect_row_index: Any | None = None
    instance_key: Any | None = None
    mask_surface_kind: str = "binary"
    mask_surface_path: str = "masks_roi"
    probability_thresholds: tuple[float, ...] = ()
    probability_encoding: Optional[str] = None


class _ThresholdedProbabilityMaskArray:
    """Array-like binary view over subject-mask probability surfaces."""

    dtype = np.dtype(np.uint8)

    def __init__(
        self,
        probabilities: Any,
        *,
        thresholds: Sequence[float],
        encoding: Optional[str],
        source_path: str,
    ) -> None:
        self._probabilities = probabilities
        self._thresholds = np.asarray(thresholds, dtype=np.float32)
        self._encoding = encoding
        self._source_path = str(source_path)
        self.shape = tuple(int(dim) for dim in probabilities.shape)

    def __getitem__(self, key: object) -> np.ndarray:
        raw = np.asarray(self._probabilities[key])
        probabilities = self._decode(raw)
        thresholds = self._thresholds_for_key(key, probabilities.ndim)
        return (probabilities >= thresholds).astype(np.uint8, copy=False)

    def _decode(self, values: np.ndarray) -> np.ndarray:
        return decode_probability_values(values, encoding=self._encoding, source_path=self._source_path)

    def _thresholds_for_key(self, key: object, ndim: int) -> np.ndarray | np.float32:
        if self._thresholds.size <= 0:
            return np.float32(0.5)

        channel_key: object = slice(None)
        if isinstance(key, tuple) and len(key) > 1:
            channel_key = key[1]
        channel_thresholds = np.asarray(self._thresholds[channel_key], dtype=np.float32)
        if channel_thresholds.ndim == 0:
            return np.float32(channel_thresholds)
        if ndim == 4:
            return channel_thresholds.reshape((1, -1, 1, 1))
        if ndim == 3:
            return channel_thresholds.reshape((-1, 1, 1))
        return channel_thresholds


class _MaskStoreArray:
    """Array-like dense binary view over a compact mask store."""

    dtype = np.dtype(np.uint8)

    def __init__(self, mask_store: MaskStore) -> None:
        self._mask_store = mask_store
        self.shape = tuple(int(dim) for dim in mask_store.shape)

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple):
            if len(key) > 4:
                raise IndexError("too many indices for mask store array")
            normalized = key + (slice(None),) * (4 - len(key))
        else:
            normalized = (key, slice(None), slice(None), slice(None))
        row_key, channel_key, y_key, x_key = normalized
        row_scalar = isinstance(row_key, (int, np.integer))
        channel_scalar = isinstance(channel_key, (int, np.integer))
        dense = self._mask_store.read_dense(rows=row_key, channels=channel_key)
        dense = dense[:, :, y_key, x_key]
        if row_scalar and channel_scalar:
            return dense[0, 0]
        if row_scalar:
            return dense[0]
        if channel_scalar:
            return dense[:, 0]
        return dense


class _RefinedSubjectComponentMaskArray:
    """Array-like single-channel source view over a refined-subject component."""

    dtype = np.dtype(np.uint8)

    def __init__(self, masks_roi: Any, component_index: int) -> None:
        self._masks_roi = masks_roi
        self._component_index = int(component_index)
        self.shape = (
            int(masks_roi.shape[0]),
            1,
            int(masks_roi.shape[2]),
            int(masks_roi.shape[3]),
        )

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple):
            if len(key) > 4:
                raise IndexError("too many indices for refined-subject component source")
            normalized = key + (slice(None),) * (4 - len(key))
        else:
            normalized = (key, slice(None), slice(None), slice(None))
        row_key, channel_key, y_key, x_key = normalized
        row_scalar = isinstance(row_key, (int, np.integer))
        channel_scalar = isinstance(channel_key, (int, np.integer))
        if channel_scalar and int(channel_key) != 0:
            raise IndexError("refined-subject component source has one channel")
        dense = np.asarray(self._masks_roi[row_key, self._component_index, y_key, x_key], dtype=np.uint8)
        if row_scalar and channel_scalar:
            return dense
        if row_scalar:
            return dense.reshape((1, *dense.shape[-2:]))
        if channel_scalar:
            return dense
        return dense[:, np.newaxis, ...]


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
    reason_labels: Optional[np.ndarray] = None
    source_masks: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _RefinedSubjectApplyContext:
    source: SourceSubjectMaskRun
    refined: RefinedSubjectMaskRun
    component_sources: dict[str, SourceSubjectMaskRun]
    component_name: str
    comp_idx: int
    component_group: zarr.Group


@dataclass(frozen=True)
class ReviewPanelRegion:
    name: str
    x: int
    y: int
    width: int
    height: int
    zoom: int = 1
    label_h: int = 0
    patch_x0: int = 0
    patch_y0: int = 0
    patch_w: int | None = None
    patch_h: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_component_name(name: object) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    if not text:
        return None
    return COMPONENT_ALIASES.get(text, text)


def _default_components_for_source(source: SourceSubjectMaskRun) -> tuple[str, ...]:
    available: list[str] = []
    seen: set[str] = set()
    for idx, raw_name in enumerate(source.mask_labels):
        if idx >= int(source.available_channels.shape[0]) or not bool(source.available_channels[idx]):
            continue
        component_name = _normalize_component_name(raw_name)
        if component_name is None or component_name in seen:
            continue
        seen.add(component_name)
        available.append(component_name)
    if available:
        return tuple(available)

    fallback_labels: list[str] = []
    for raw_name in source.mask_labels:
        component_name = _normalize_component_name(raw_name)
        if component_name is None or component_name in seen:
            continue
        seen.add(component_name)
        fallback_labels.append(component_name)
    if fallback_labels:
        return tuple(fallback_labels)

    return tuple(DEFAULT_COMPONENTS)


def _normalize_component_list(
    values: Optional[Sequence[str]],
    *,
    default_source: Optional[SourceSubjectMaskRun] = None,
) -> tuple[str, ...]:
    if not values:
        if default_source is not None:
            return _default_components_for_source(default_source)
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
        if default_source is not None:
            return _default_components_for_source(default_source)
        return tuple(DEFAULT_COMPONENTS)
    return tuple(result)


def _coerce_probability_threshold(value: object, *, default: float = 0.5) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float(default)
    return float(np.clip(threshold, 0.0, 1.0))


def _probability_thresholds_for_labels(group: zarr.Group, labels: Sequence[str]) -> tuple[float, ...]:
    default_threshold = _coerce_probability_threshold(group.attrs.get("mask_probability_threshold"), default=0.5)
    thresholds = [default_threshold for _ in labels]
    raw = (
        group.attrs.get("thresholds_by_label")
        or group.attrs.get("threshold_by_component")
        or group.attrs.get("threshold_by_label")
    )
    if isinstance(raw, Mapping):
        for idx, raw_label in enumerate(labels):
            label = str(raw_label)
            normalized = _normalize_component_name(label) or label
            value = raw.get(label, raw.get(normalized))
            if value is not None:
                thresholds[idx] = _coerce_probability_threshold(value, default=default_threshold)
    elif isinstance(raw, (list, tuple)) and len(raw) == len(labels):
        thresholds = [
            _coerce_probability_threshold(value, default=default_threshold)
            for value in raw
        ]
    return tuple(float(value) for value in thresholds)


def _probability_encoding_for_group(
    group: zarr.Group,
    *,
    source_path: Optional[str] = None,
    observed_dtype: object | None = None,
) -> str:
    if observed_dtype is None:
        probabilities = group.get("mask_probs_roi")
        observed_dtype = probabilities.dtype if probabilities is not None else np.dtype(np.float32)
    resolved_source_path = source_path or f"{getattr(group, 'path', 'subject_mask_runs/<unknown>')}/mask_probs_roi"
    return probabilities_encoding_from_attrs(
        group.attrs,
        source_path=str(resolved_source_path),
        observed_dtype=observed_dtype,
    )


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


def _refined_metric_chunks(total_rows: int) -> tuple[int]:
    return (refined_subject_mask_metric_row_chunk(total_rows),)


def _refined_metric_chunks_2d(total_rows: int) -> tuple[int, int]:
    return (refined_subject_mask_metric_row_chunk(total_rows), 1)


def _refined_metric_chunks_lastdim(total_rows: int, width: int) -> tuple[int, int, int]:
    return (refined_subject_mask_metric_row_chunk(total_rows), 1, int(width))


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
        "source_crop_run": source.crop_run,
        "source_channels": _normalize_provenance_channels(source_channels, default_component=component_name),
    }
    payload.update(source.source_crop_snapshot)
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
    if source.mask_surface_kind == "thresholded_probability":
        source_idx = source.mask_labels.index(component_name) if component_name in source.mask_labels else None
        threshold = (
            float(source.probability_thresholds[source_idx])
            if source_idx is not None and source_idx < len(source.probability_thresholds)
            else 0.5
        )
        payload["source_probability_path"] = (
            f"subject_mask_runs/{source.run_name}/{source.mask_surface_path}"
        )
        payload["source_probability_encoding"] = str(source.probability_encoding or "")
        payload["source_binary_derivation"] = "threshold(mask_probs_roi)"
        payload["source_probability_threshold"] = float(threshold)
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


def _mask_row_fingerprint(mask: np.ndarray) -> np.uint64:
    payload = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.blake2b(payload.tobytes(), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def _compute_mask_row_fingerprints(masks: np.ndarray) -> np.ndarray:
    rows = np.asarray(masks, dtype=np.uint8)
    if rows.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(rows.shape)}")
    out = np.zeros((int(rows.shape[0]),), dtype=np.uint64)
    for row_idx in range(int(rows.shape[0])):
        out[row_idx] = _mask_row_fingerprint(rows[row_idx])
    return out


def _compute_manual_override_flags(
    *,
    source_masks: np.ndarray,
    current_masks: np.ndarray,
) -> np.ndarray:
    source_arr = np.asarray(source_masks, dtype=np.uint8)
    current_arr = np.asarray(current_masks, dtype=np.uint8)
    if source_arr.shape != current_arr.shape:
        raise ValueError(
            f"source/current mask shape mismatch for source-sync tracking: {tuple(source_arr.shape)} vs {tuple(current_arr.shape)}"
        )
    if source_arr.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(source_arr.shape)}")
    changed = np.any(source_arr != current_arr, axis=(1, 2))
    return np.asarray(changed, dtype=bool)


def _ensure_component_source_sync_arrays(
    component_group: zarr.Group,
    *,
    total_rois: int,
    source_masks: np.ndarray,
    current_masks: np.ndarray,
) -> None:
    component_chunks = _refined_metric_chunks(total_rois)
    source_hashes = _compute_mask_row_fingerprints(source_masks)
    stale = np.zeros((int(total_rois),), dtype=bool)

    source_hash_arr = component_group.get("source_row_fingerprint")
    if source_hash_arr is None or int(source_hash_arr.shape[0]) != int(total_rois):
        component_group.create_array(
            "source_row_fingerprint",
            data=np.asarray(source_hashes, dtype=np.uint64),
            chunks=component_chunks,
            overwrite=True,
        )

    manual_override_arr = component_group.get("manual_override")
    if manual_override_arr is None or int(manual_override_arr.shape[0]) != int(total_rois):
        edit_applied_arr = component_group.get("edit_applied")
        if edit_applied_arr is not None and int(edit_applied_arr.shape[0]) == int(total_rois):
            manual_override = np.asarray(edit_applied_arr[:], dtype=bool)
        else:
            manual_override = _compute_manual_override_flags(source_masks=source_masks, current_masks=current_masks)
        component_group.create_array(
            "manual_override",
            data=np.asarray(manual_override, dtype=bool),
            chunks=component_chunks,
            overwrite=True,
        )

    stale_arr = component_group.get("source_row_stale")
    if stale_arr is None or int(stale_arr.shape[0]) != int(total_rois):
        component_group.create_array(
            "source_row_stale",
            data=stale,
            chunks=component_chunks,
            overwrite=True,
        )

    component_group.attrs.setdefault("source_sync_schema_id", REFINED_SUBJECT_SOURCE_SYNC_SCHEMA_ID)


def _has_component_source_sync_tracking(component_group: zarr.Group, *, total_rois: int) -> bool:
    if str(component_group.attrs.get("source_sync_schema_id") or "") != REFINED_SUBJECT_SOURCE_SYNC_SCHEMA_ID:
        return False
    required_names = ("source_row_fingerprint", "manual_override", "source_row_stale")
    for name in required_names:
        arr = component_group.get(name)
        if arr is None or int(arr.shape[0]) != int(total_rois):
            return False
    return True


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
    metrics = batch_mask_spatial_metrics(masks_roi)
    return np.asarray(metrics["mask_present"], dtype=bool), np.asarray(metrics["area_px"], dtype=np.float32)


def _compute_geometry_metrics(masks_roi: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks_roi, dtype=np.uint8)
    if binary.ndim != 4:
        raise ValueError(f"Expected masks with shape (N,C,H,W), got {tuple(binary.shape)}")
    metrics = batch_mask_spatial_metrics(binary)
    return {
        "centroid_xy": np.asarray(metrics["centroid_xy"], dtype=np.float32),
        "centroid_valid": np.asarray(metrics["centroid_valid"], dtype=bool),
        "bbox_xyxy": np.asarray(metrics["bbox_xyxy"], dtype=np.float32),
        "bbox_valid": np.asarray(metrics["bbox_valid"], dtype=bool),
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


def _source_seed_mask_for_component(
    refined: RefinedSubjectMaskRun,
    component_name: str,
    roi_idx: int,
) -> np.ndarray | None:
    component_group = _get_component_group(refined.group, component_name)
    if component_group is None:
        return None
    seed_masks = component_group.get("source_seed_masks_roi")
    if seed_masks is None:
        return None
    return np.asarray(seed_masks[int(roi_idx)], dtype=np.uint8)


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
    source_masks: Optional[np.ndarray] = None,
    current_masks: Optional[np.ndarray] = None,
    source_seed_masks: Optional[np.ndarray] = None,
) -> zarr.Group:
    components_parent = refined.require_group("components")
    component_group = components_parent.require_group(component_name)
    component_chunks = _refined_metric_chunks(total_rois)
    if "mask_present" not in component_group:
        component_group.create_array(
            "mask_present",
            data=np.asarray(mask_present, dtype=bool),
            chunks=component_chunks,
            overwrite=True,
        )
    if "area_px" not in component_group:
        component_group.create_array(
            "area_px",
            data=np.asarray(area_px, dtype=np.float32),
            chunks=component_chunks,
            overwrite=True,
        )
    if "edit_applied" not in component_group:
        component_group.create_array(
            "edit_applied",
            data=np.asarray(edit_applied, dtype=bool),
            chunks=component_chunks,
            overwrite=True,
        )
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
                metrics_group.create_array(metric_name, data=metric_values, chunks=component_chunks, overwrite=True)
    _load_or_init_reason_labels(component_group, total_rois)
    if source_masks is not None and current_masks is not None:
        _ensure_component_source_sync_arrays(
            component_group,
            total_rois=int(total_rois),
            source_masks=np.asarray(source_masks, dtype=np.uint8),
            current_masks=np.asarray(current_masks, dtype=np.uint8),
        )
    if source_seed_masks is not None:
        source_seed_arr = np.asarray(source_seed_masks, dtype=np.uint8)
        if source_seed_arr.ndim != 3 or int(source_seed_arr.shape[0]) != int(total_rois):
            raise ValueError(
                f"source_seed_masks for {component_name!r} must have shape (N,H,W), "
                f"got {tuple(source_seed_arr.shape)}."
            )
        chunk_rows = refined_subject_mask_metric_row_chunk(total_rois)
        component_group.create_array(
            "source_seed_masks_roi",
            data=source_seed_arr,
            chunks=(chunk_rows, int(source_seed_arr.shape[1]), int(source_seed_arr.shape[2])),
            overwrite=True,
        )
        component_group.attrs["source_seed_masks_schema_id"] = "refined_subject_component_source_seed_masks_v1"
    return component_group


def _load_source_subject_mask_run(root: zarr.Group, subject_run: Optional[str]) -> SourceSubjectMaskRun:
    parent = root.get("subject_mask_runs")
    if parent is None:
        raise RuntimeError("No subject_mask_runs found in archive.")
    run_name = subject_run or _resolve_latest_run(parent, "subject_mask_runs")
    if run_name not in parent:
        raise RuntimeError(f"subject_mask_runs/{run_name} not found.")
    group = parent[run_name]
    crop_run = str(group.attrs.get("source_crop_run") or "")
    if not crop_run:
        crop_parent = root.get("crop_runs")
        crop_run = _resolve_latest_run(crop_parent, "crop_runs")
    crop_group = root.get(f"crop_runs/{crop_run}") if crop_run else None
    stored_crop_snapshot = extract_source_crop_snapshot_attrs(group.attrs)
    inferred_crop_storage_mode = stored_crop_snapshot.get("source_crop_storage_mode")
    if inferred_crop_storage_mode is None and crop_group is not None:
        inferred_crop_storage_mode = (
            crop_group.attrs.get("crop_storage_mode")
            or ("materialized" if crop_group.get("roi_images") is not None else "geometry_only")
        )
    source_crop_snapshot = build_source_crop_snapshot_attrs(
        crop_group.attrs if crop_group is not None else None,
        source_crop_storage_mode=inferred_crop_storage_mode,
    )
    source_crop_snapshot.update(stored_crop_snapshot)
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing usable mask_labels attr.")
    mask_labels = tuple(str(item) for item in labels_raw)
    available = group.get("available_channels")
    if available is None:
        raise RuntimeError(f"subject_mask_runs/{run_name} missing available_channels.")
    masks_roi = group.get("masks_roi")
    mask_surface_kind = "binary"
    mask_surface_path = "masks_roi"
    probability_thresholds: tuple[float, ...] = ()
    probability_encoding: Optional[str] = None
    if masks_roi is None:
        probabilities = group.get("mask_probs_roi")
        if probabilities is not None:
            if len(probabilities.shape) != 4:
                raise RuntimeError(
                    f"subject_mask_runs/{run_name}/mask_probs_roi must have shape (N, C, H, W), "
                    f"got {probabilities.shape}."
                )
            probability_thresholds = _probability_thresholds_for_labels(group, mask_labels)
            probability_source_path = f"subject_mask_runs/{run_name}/mask_probs_roi"
            probability_encoding = _probability_encoding_for_group(
                group,
                source_path=probability_source_path,
                observed_dtype=probabilities.dtype,
            )
            masks_roi = _ThresholdedProbabilityMaskArray(
                probabilities,
                thresholds=probability_thresholds,
                encoding=probability_encoding,
                source_path=probability_source_path,
            )
            mask_surface_kind = "thresholded_probability"
            mask_surface_path = "mask_probs_roi"
        else:
            try:
                mask_store = open_mask_store(
                    group,
                    source_path=f"subject_mask_runs/{run_name}",
                    prefer="rle",
                )
            except MaskStoreError as exc:
                raise RuntimeError(f"subject_mask_runs/{run_name} missing masks_roi, mask_probs_roi, or mask_rle.") from exc
            masks_roi = _MaskStoreArray(mask_store)
            mask_surface_kind = "binary"
            mask_surface_path = "mask_rle"
    def _lineage_array(name: str) -> Any | None:
        if name in group:
            return group[name]
        if crop_group is not None:
            return crop_group.get(name)
        return None

    total_rows = int(masks_roi.shape[0])
    frame_indices = _lineage_array("frame_indices")
    source_crop_row_ids = resolve_source_crop_row_ids(
        group,
        crop_group,
        total_rois=total_rows,
        frame_indices=frame_indices,
    )

    return SourceSubjectMaskRun(
        run_name=run_name,
        group=group,
        crop_run=crop_run,
        source_crop_snapshot=source_crop_snapshot,
        masks_roi=masks_roi,
        detection_source=group["detection_source"],
        mask_labels=mask_labels,
        available_channels=np.asarray(available[:], dtype=bool),
        frame_indices=frame_indices,
        frame_counts=_lineage_array("frame_counts"),
        detection_indices=_lineage_array("detection_indices"),
        source_method=str(group.attrs.get("method")) if group.attrs.get("method") is not None else None,
        source_keypoints_run=resolve_source_keypoints_run(group.attrs),
        source_keypoint_group=(
            str(group.attrs.get("source_keypoint_group")) if group.attrs.get("source_keypoint_group") is not None else None
        ),
        assignment_keypoints_run=(
            str(resolve_assignment_keypoints_run(group.attrs))
            if resolve_assignment_keypoints_run(group.attrs) is not None
            else None
        ),
        assignment_keypoint_group=(
            str(resolve_assignment_keypoint_group(group.attrs))
            if resolve_assignment_keypoint_group(group.attrs) is not None
            else None
        ),
        source_crop_row_ids=source_crop_row_ids,
        source_refined_row_ids=_lineage_array("source_refined_row_ids"),
        source_detect_row_index=_lineage_array("source_detect_row_index"),
        instance_key=_lineage_array("instance_key"),
        mask_surface_kind=mask_surface_kind,
        mask_surface_path=mask_surface_path,
        probability_thresholds=probability_thresholds,
        probability_encoding=probability_encoding,
    )


def _normalize_refined_eye_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"eye_left", "left", "left_eye"}:
        return "eye_left"
    if text in {"eye_right", "right", "right_eye"}:
        return "eye_right"
    return text


def _refined_eye_channel_indices(group: zarr.Group) -> tuple[int, int]:
    labels_raw = group.attrs.get("eye_labels")
    if labels_raw is None:
        return 0, 1
    if not isinstance(labels_raw, (list, tuple)):
        raise ValueError("refined_eye_masks_runs source has non-list eye_labels attr.")
    normalized = [_normalize_refined_eye_label(value) for value in labels_raw]
    try:
        left_idx = normalized.index("eye_left")
        right_idx = normalized.index("eye_right")
    except ValueError as exc:
        raise ValueError(
            "refined_eye_masks_runs source must expose anatomical eye_labels for direct refined-subject seeding."
        ) from exc
    return int(left_idx), int(right_idx)


def _resolve_source_lineage_array(
    root: zarr.Group,
    *,
    source_group: zarr.Group,
    source_crop_run: str,
    array_name: str,
) -> Any | None:
    if array_name in source_group:
        return source_group[array_name]
    crop_parent = root.get("crop_runs")
    if crop_parent is None or source_crop_run not in crop_parent:
        return None
    crop_group = crop_parent[source_crop_run]
    return crop_group.get(array_name)


def _load_refined_eye_mask_source(root: zarr.Group, refined_eye_run: Optional[str]) -> SourceSubjectMaskRun:
    parent = root.get("refined_eye_masks_runs")
    run_name = refined_eye_run or _resolve_latest_run(parent, "refined_eye_masks_runs")
    if parent is None or run_name not in parent:
        raise RuntimeError(f"refined_eye_masks_runs/{run_name} not found.")
    group = parent[run_name]
    masks_arr = group.get("masks_roi")
    if masks_arr is None:
        raise RuntimeError(f"refined_eye_masks_runs/{run_name} missing masks_roi.")
    if len(masks_arr.shape) != 4 or int(masks_arr.shape[1]) < 2:
        raise RuntimeError(
            f"refined_eye_masks_runs/{run_name}/masks_roi must have shape (N, >=2, H, W), got {masks_arr.shape}."
        )
    left_idx, right_idx = _refined_eye_channel_indices(group)
    if max(left_idx, right_idx) >= int(masks_arr.shape[1]):
        raise RuntimeError(
            f"refined_eye_masks_runs/{run_name} eye_labels reference channels outside masks_roi shape {masks_arr.shape}."
        )
    raw_masks = np.asarray(masks_arr[:], dtype=np.uint8)
    masks = raw_masks[:, [left_idx, right_idx], :, :]

    crop_run = str(group.attrs.get("source_crop_run") or "")
    if not crop_run:
        crop_run = _resolve_latest_run(root.get("crop_runs"), "crop_runs")
    crop_group = root.get(f"crop_runs/{crop_run}") if crop_run else None
    stored_crop_snapshot = extract_source_crop_snapshot_attrs(group.attrs)
    inferred_crop_storage_mode = stored_crop_snapshot.get("source_crop_storage_mode")
    if inferred_crop_storage_mode is None and crop_group is not None:
        inferred_crop_storage_mode = (
            crop_group.attrs.get("crop_storage_mode")
            or ("materialized" if crop_group.get("roi_images") is not None else "geometry_only")
        )
    source_crop_snapshot = build_source_crop_snapshot_attrs(
        crop_group.attrs if crop_group is not None else None,
        source_crop_storage_mode=inferred_crop_storage_mode,
    )
    source_crop_snapshot.update(stored_crop_snapshot)

    detection_source = _resolve_source_lineage_array(
        root,
        source_group=group,
        source_crop_run=crop_run,
        array_name="detection_source",
    )
    if detection_source is None:
        raise RuntimeError(f"Could not resolve detection_source for refined_eye_masks_runs/{run_name}.")
    frame_indices = _resolve_source_lineage_array(
        root,
        source_group=group,
        source_crop_run=crop_run,
        array_name="frame_indices",
    )
    source_crop_row_ids = resolve_source_crop_row_ids(
        group,
        crop_group,
        total_rois=int(masks.shape[0]),
        frame_indices=frame_indices,
    )

    return SourceSubjectMaskRun(
        run_name=run_name,
        group=group,
        crop_run=crop_run,
        source_crop_snapshot=source_crop_snapshot,
        masks_roi=masks,
        detection_source=detection_source,
        mask_labels=REFINED_EYE_COMPONENTS,
        available_channels=np.ones((2,), dtype=bool),
        frame_indices=frame_indices,
        frame_counts=_resolve_source_lineage_array(root, source_group=group, source_crop_run=crop_run, array_name="frame_counts"),
        detection_indices=_resolve_source_lineage_array(root, source_group=group, source_crop_run=crop_run, array_name="detection_indices"),
        source_method=str(group.attrs.get("method")) if group.attrs.get("method") is not None else None,
        source_keypoints_run=resolve_source_keypoints_run(group.attrs),
        source_keypoint_group=(
            str(group.attrs.get("source_keypoint_group")) if group.attrs.get("source_keypoint_group") is not None else None
        ),
        assignment_keypoints_run=(
            str(resolve_assignment_keypoints_run(group.attrs))
            if resolve_assignment_keypoints_run(group.attrs) is not None
            else None
        ),
        assignment_keypoint_group=(
            str(resolve_assignment_keypoint_group(group.attrs))
            if resolve_assignment_keypoint_group(group.attrs) is not None
            else None
        ),
        source_crop_row_ids=source_crop_row_ids,
        source_refined_row_ids=_resolve_source_lineage_array(
            root,
            source_group=group,
            source_crop_run=crop_run,
            array_name="source_refined_row_ids",
        ),
        source_detect_row_index=_resolve_source_lineage_array(
            root,
            source_group=group,
            source_crop_run=crop_run,
            array_name="source_detect_row_index",
        ),
        instance_key=_resolve_source_lineage_array(
            root,
            source_group=group,
            source_crop_run=crop_run,
            array_name="instance_key",
        ),
    )


def _load_refined_subject_component_source(
    root: zarr.Group,
    refined_run: str,
    component_name: str,
) -> SourceSubjectMaskRun:
    refined = _open_existing_refined_subject_run(root, refined_run)
    if component_name not in refined.component_to_index:
        raise RuntimeError(
            f"refined_subject_masks_runs/{refined_run} does not contain component {component_name!r}."
        )
    masks_roi = refined.group.get("masks_roi")
    if masks_roi is None:
        raise RuntimeError(f"refined_subject_masks_runs/{refined_run} missing masks_roi.")
    crop_run = str(refined.group.attrs.get("source_crop_run") or "")
    if not crop_run:
        crop_parent = root.get("crop_runs")
        crop_run = _resolve_latest_run(crop_parent, "crop_runs")
    crop_group = root.get(f"crop_runs/{crop_run}") if crop_run else None
    source_crop_snapshot = build_source_crop_snapshot_attrs(
        crop_group.attrs if crop_group is not None else None,
        source_crop_storage_mode=(
            crop_group.attrs.get("crop_storage_mode")
            if crop_group is not None
            else None
        ),
    )
    return SourceSubjectMaskRun(
        run_name=str(refined_run),
        group=refined.group,
        crop_run=crop_run,
        source_crop_snapshot=source_crop_snapshot,
        masks_roi=_RefinedSubjectComponentMaskArray(masks_roi, int(refined.component_to_index[component_name])),
        detection_source=refined.group.get("detection_source"),
        mask_labels=(str(component_name),),
        available_channels=np.ones((1,), dtype=bool),
        frame_indices=_resolve_source_lineage_array(root, source_group=refined.group, source_crop_run=crop_run, array_name="frame_indices"),
        frame_counts=_resolve_source_lineage_array(root, source_group=refined.group, source_crop_run=crop_run, array_name="frame_counts"),
        detection_indices=_resolve_source_lineage_array(root, source_group=refined.group, source_crop_run=crop_run, array_name="detection_indices"),
        source_method=str(refined.group.attrs.get("method")) if refined.group.attrs.get("method") is not None else None,
        source_keypoints_run=resolve_source_keypoints_run(refined.group.attrs),
        source_keypoint_group=(
            str(refined.group.attrs.get("source_keypoint_group"))
            if refined.group.attrs.get("source_keypoint_group") is not None
            else None
        ),
        assignment_keypoints_run=(
            str(resolve_assignment_keypoints_run(refined.group.attrs))
            if resolve_assignment_keypoints_run(refined.group.attrs) is not None
            else None
        ),
        assignment_keypoint_group=(
            str(resolve_assignment_keypoint_group(refined.group.attrs))
            if resolve_assignment_keypoint_group(refined.group.attrs) is not None
            else None
        ),
        source_crop_row_ids=_resolve_source_lineage_array(
            root,
            source_group=refined.group,
            source_crop_run=crop_run,
            array_name="source_crop_row_ids",
        ),
        source_refined_row_ids=_resolve_source_lineage_array(
            root,
            source_group=refined.group,
            source_crop_run=crop_run,
            array_name="source_refined_row_ids",
        ),
        source_detect_row_index=_resolve_source_lineage_array(
            root,
            source_group=refined.group,
            source_crop_run=crop_run,
            array_name="source_detect_row_index",
        ),
        instance_key=_resolve_source_lineage_array(
            root,
            source_group=refined.group,
            source_crop_run=crop_run,
            array_name="instance_key",
        ),
        mask_surface_kind="refined_subject_component",
        mask_surface_path=f"refined_subject_masks_runs/{refined_run}/masks_roi[{component_name}]",
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
        cache[f"subject_mask_runs/{default_source.run_name}"] = default_source

    primary_source = default_source
    primary_run = _resolve_primary_source_subject_mask_run_name(refined.group)
    if primary_source is None and primary_run:
        primary_source = cache.setdefault(
            f"subject_mask_runs/{primary_run}",
            _load_source_subject_mask_run(root, primary_run),
        )

    component_sources: dict[str, SourceSubjectMaskRun] = {}
    for component_name in refined.component_names:
        provenance_group = _get_component_provenance_group(refined.group, component_name)
        source_stage = str(provenance_group.attrs.get("source_stage") or "") if provenance_group is not None else ""
        source_run = str(provenance_group.attrs.get("source_run") or "") if provenance_group is not None else ""
        resolved_source: SourceSubjectMaskRun | None = None
        if source_run and source_stage == "subject_mask_runs":
            resolved_source = cache.setdefault(
                f"subject_mask_runs/{source_run}",
                _load_source_subject_mask_run(root, source_run),
            )
        elif source_run and source_stage == "refined_eye_masks_runs":
            cache_key = f"refined_eye_masks_runs/{source_run}"
            if cache_key in cache:
                resolved_source = cache[cache_key]
            else:
                try:
                    resolved_source = _load_refined_eye_mask_source(root, source_run)
                except RuntimeError:
                    # Older refined-subject runs may preserve legacy refined-eye
                    # provenance even when the editor source is the coarse
                    # subject-mask run.
                    if default_source is not None:
                        resolved_source = default_source
                    elif primary_source is not None:
                        resolved_source = primary_source
                    else:
                        raise
                cache[cache_key] = resolved_source
        elif source_run and source_stage == "refined_subject_masks_runs":
            source_component = str(
                provenance_group.attrs.get("source_component")
                or provenance_group.attrs.get("source_channel")
                or component_name
            ) if provenance_group is not None else component_name
            cache_key = f"refined_subject_masks_runs/{source_run}:{source_component}"
            if cache_key in cache:
                resolved_source = cache[cache_key]
            else:
                resolved_source = _load_refined_subject_component_source(root, source_run, source_component)
                cache[cache_key] = resolved_source
        elif source_run and source_stage and source_stage != "subject_mask_runs":
            # Editing still operates against a coarse subject_mask_runs source even
            # when the component's original lineage points at a legacy stage such
            # as eye_masks_runs. Preserve the true provenance on the
            # component itself, but use the coarse source subject run for editor
            # overlay/save comparisons.
            if default_source is not None:
                resolved_source = default_source
            elif primary_source is not None:
                resolved_source = primary_source
        elif default_source is not None:
            resolved_source = default_source
        elif primary_source is not None:
            resolved_source = primary_source
        if resolved_source is None:
            raise RuntimeError(
                f"Unable to resolve source mask lineage for component {component_name!r} "
                f"in refined_subject_masks_runs/{refined.run_name}."
            )
        component_sources[component_name] = resolved_source
        if primary_source is None:
            primary_source = resolved_source

    if primary_source is None:
        raise RuntimeError(f"Unable to resolve any source mask run for refined_subject_masks_runs/{refined.run_name}.")
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
    if reference_source.source_crop_row_ids is None:
        source_path = getattr(reference_source.group, "path", reference_source.run_name)
        raise ValueError(
            f"{source_path} cannot create refined_subject_masks_runs without source_crop_row_ids."
        )
    component_names = tuple(str(name) for name in component_names)
    storage_chunks = refined_subject_mask_storage_chunks(total_rois, height, width)
    metric_chunks = _refined_metric_chunks_2d(total_rois)
    metric_chunks_2 = _refined_metric_chunks_lastdim(total_rois, 2)
    metric_chunks_4 = _refined_metric_chunks_lastdim(total_rois, 4)
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
    mark_run_started(run_group, run_name=target_run, stage=REFINED_SUBJECT_STAGE_NAME)
    note_pending_latest(refined_parent, target_run)
    run_group.create_array(
        "detection_source",
        data=np.asarray(reference_source.detection_source[:], dtype=np.int8),
        chunks=_refined_metric_chunks(total_rois),
        overwrite=True,
    )
    run_group.create_array("masks_roi", data=masks, chunks=storage_chunks, overwrite=True)
    run_group.create_array(
        "available_channels",
        data=np.ones((len(component_names),), dtype=bool),
        overwrite=True,
    )
    run_group.create_array("edit_applied", data=edit_applied, chunks=metric_chunks, overwrite=True)
    copy_row_lineage_arrays_from_sources(
        run_group,
        {
            "frame_indices": reference_source.frame_indices,
            "frame_counts": reference_source.frame_counts,
            "detection_indices": reference_source.detection_indices,
            "source_crop_row_ids": reference_source.source_crop_row_ids,
            "source_refined_row_ids": reference_source.source_refined_row_ids,
            "source_detect_row_index": reference_source.source_detect_row_index,
            "instance_key": reference_source.instance_key,
        },
        total_rois=total_rois,
    )

    metrics = run_group.require_group("metrics")
    metrics.create_array("mask_present", data=mask_present, chunks=metric_chunks, overwrite=True)
    metrics.create_array("area_px", data=area_px, chunks=metric_chunks, overwrite=True)
    geometry_metrics = _compute_geometry_metrics(masks)
    metrics.create_array("centroid_xy", data=geometry_metrics["centroid_xy"], chunks=metric_chunks_2, overwrite=True)
    metrics.create_array("centroid_valid", data=geometry_metrics["centroid_valid"], chunks=metric_chunks, overwrite=True)
    metrics.create_array("bbox_xyxy", data=geometry_metrics["bbox_xyxy"], chunks=metric_chunks_4, overwrite=True)
    metrics.create_array("bbox_valid", data=geometry_metrics["bbox_valid"], chunks=metric_chunks, overwrite=True)
    created = _utc_now()

    for comp_idx, component_name in enumerate(component_names):
        seed = component_seeds[component_name]
        component_masks = np.asarray(seed.masks, dtype=np.uint8)
        component_source_masks = (
            np.asarray(seed.source_masks, dtype=np.uint8)
            if seed.source_masks is not None
            else component_masks
        )
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
            source_masks=component_source_masks,
            current_masks=np.asarray(masks[:, comp_idx], dtype=np.uint8),
            source_seed_masks=(
                np.asarray(seed.source_masks, dtype=np.uint8)
                if seed.source_masks is not None
                else None
            ),
        )
        reason_labels = _load_or_init_reason_labels(component_group, total_rois)
        if seed.reason_labels is not None:
            seed_reason_labels = np.asarray(seed.reason_labels, dtype=object).reshape(-1)
            if int(seed_reason_labels.shape[0]) != int(total_rois):
                raise ValueError(
                    f"Reason-label row count mismatch for {component_name!r}: "
                    f"expected {total_rois}, got {int(seed_reason_labels.shape[0])}."
                )
            reason_labels[:] = seed_reason_labels
        else:
            for row_idx in range(total_rois):
                source_mask = np.asarray(component_source_masks[row_idx], dtype=np.uint8)
                reason_labels[row_idx] = _default_reason_label(
                    source_mask,
                    masks[row_idx, comp_idx],
                    edit_applied=False,
                )
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
    run_group.attrs.update(reference_source.source_crop_snapshot)
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
        **reference_source.source_crop_snapshot,
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
    if {"eye_left", "eye_right"}.issubset(set(component_names)):
        from ..shared.refined_subject_eye_geometry import write_refined_subject_eye_geometry

        write_refined_subject_eye_geometry(run_group)
    mark_run_complete(run_group, parent_group=refined_parent, run_name=target_run)
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
    refined_parent = require_runs_parent(root, "refined_subject_masks_runs")
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
            if "mask_rle" not in run_group:
                raise RuntimeError(f"refined_subject_masks_runs/{target_run} missing masks_roi or mask_rle.")
            try:
                materialize_dense_masks_roi_from_store(
                    run_group,
                    chunk_size=256,
                    overwrite=False,
                    source_path=f"refined_subject_masks_runs/{target_run}",
                )
            except (MaskStoreError, ValueError) as exc:
                raise RuntimeError(
                    f"refined_subject_masks_runs/{target_run} missing masks_roi and dense cache "
                    f"could not be materialized from mask_rle."
                ) from exc
            masks_arr = run_group.get("masks_roi")
            if masks_arr is None:
                raise RuntimeError(
                    f"refined_subject_masks_runs/{target_run} dense masks_roi was not materialized from mask_rle."
                )
        total_rois = int(masks_arr.shape[0])
        metric_chunks = _refined_metric_chunks_2d(total_rois)
        metric_chunks_2 = _refined_metric_chunks_lastdim(total_rois, 2)
        metric_chunks_4 = _refined_metric_chunks_lastdim(total_rois, 4)
        metrics = run_group.require_group("metrics")
        metric_names = ("mask_present", "area_px", "centroid_xy", "centroid_valid", "bbox_xyxy", "bbox_valid")
        needs_metric_backfill = any(name not in metrics for name in metric_names)
        components_group = run_group.get("components")
        needs_component_backfill = not (
            components_group is not None and all(component_name in components_group for component_name in component_names)
        )
        masks_np: np.ndarray | None = None
        if needs_metric_backfill or needs_component_backfill:
            masks_np = np.asarray(masks_arr[:], dtype=np.uint8)
        geometry_metrics = None
        if "mask_present" not in metrics or "area_px" not in metrics:
            if masks_np is None:
                masks_np = np.asarray(masks_arr[:], dtype=np.uint8)
            mask_present, area_px = _compute_mask_metrics(masks_np)
            if "mask_present" not in metrics:
                metrics.create_array("mask_present", data=mask_present, chunks=metric_chunks, overwrite=True)
            if "area_px" not in metrics:
                metrics.create_array("area_px", data=area_px, chunks=metric_chunks, overwrite=True)
        if (
            "centroid_xy" not in metrics
            or "centroid_valid" not in metrics
            or "bbox_xyxy" not in metrics
            or "bbox_valid" not in metrics
        ):
            if masks_np is None:
                masks_np = np.asarray(masks_arr[:], dtype=np.uint8)
            geometry_metrics = _compute_geometry_metrics(masks_np)
            if "centroid_xy" not in metrics:
                metrics.create_array(
                    "centroid_xy",
                    data=geometry_metrics["centroid_xy"],
                    chunks=metric_chunks_2,
                    overwrite=True,
                )
            if "centroid_valid" not in metrics:
                metrics.create_array(
                    "centroid_valid",
                    data=geometry_metrics["centroid_valid"],
                    chunks=metric_chunks,
                    overwrite=True,
                )
            if "bbox_xyxy" not in metrics:
                metrics.create_array(
                    "bbox_xyxy",
                    data=geometry_metrics["bbox_xyxy"],
                    chunks=metric_chunks_4,
                    overwrite=True,
                )
            if "bbox_valid" not in metrics:
                metrics.create_array(
                    "bbox_valid",
                    data=geometry_metrics["bbox_valid"],
                    chunks=metric_chunks,
                    overwrite=True,
                )
        if "edit_applied" not in run_group:
            run_group.create_array(
                "edit_applied",
                data=np.zeros((total_rois, len(component_names)), dtype=bool),
                chunks=metric_chunks,
                overwrite=True,
            )
        if needs_component_backfill:
            if masks_np is None:
                masks_np = np.asarray(masks_arr[:], dtype=np.uint8)
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
                    source_masks=_component_masks_from_source(source, component_name),
                    current_masks=np.asarray(masks_np[:, comp_idx], dtype=np.uint8),
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
    normalized_components = _normalize_component_list(components, default_source=source)
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
        if "mask_rle" not in run_group:
            raise RuntimeError(f"refined_subject_masks_runs/{refined_run} missing masks_roi or mask_rle.")
        try:
            materialize_dense_masks_roi_from_store(
                run_group,
                chunk_size=256,
                overwrite=False,
                source_path=f"refined_subject_masks_runs/{refined_run}",
            )
        except (MaskStoreError, ValueError) as exc:
            raise RuntimeError(
                f"refined_subject_masks_runs/{refined_run} missing masks_roi and dense cache "
                f"could not be materialized from mask_rle."
            ) from exc
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


def _approve_authoritative_refined_subject_masks(
    zarr_path: str | Path | None,
    *,
    refined_run: str,
    run_state: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> dict[str, object]:
    if str(run_state).strip().lower() != "approved":
        return {"attempted": False, "reason": "run_review_state_not_approved"}
    if zarr_path is None:
        return {"attempted": False, "reason": "zarr_path_unavailable"}
    resolved_path = Path(zarr_path).expanduser()
    if not resolved_path.exists():
        return {"attempted": False, "reason": "zarr_path_unavailable", "zarr_path": str(zarr_path)}

    from ..cli.palette import ApproveRequest, approve

    envelope = approve(
        ApproveRequest(
            recording=resolved_path,
            stage="refined_subject_masks",
            run=refined_run,
            approved_by=reviewer,
            note=notes or "refined subject-mask review sign-off",
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
    zarr_path: str | Path | None = None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    existing_component_reviews = dict(refined.attrs.get("component_review_statuses") or {})
    component_reviews = dict(existing_component_reviews)
    component_payload = _review_payload(
        state=state,
        method=method,
        intended_use=intended_use,
        reviewer=reviewer,
        notes=notes,
    )
    component_reviews[str(component_name)] = component_payload

    run_state = _aggregate_run_state(component_reviews)
    run_payload = _review_payload(
        state=run_state,
        method=method,
        intended_use=intended_use,
        reviewer=reviewer,
        notes="auto_aggregated_from_component_review_statuses",
    )
    authoritative_approval = _approve_authoritative_refined_subject_masks(
        zarr_path,
        refined_run=refined_run,
        run_state=run_state,
        reviewer=reviewer,
        notes=notes,
    )
    if run_state == "approved" and not _authoritative_approval_ok(authoritative_approval):
        existing_run_payload = dict(refined.attrs.get("refined_subject_mask_review_status") or {})
        existing_run_payload["authoritative_approval"] = authoritative_approval
        existing_component_payload = dict(existing_component_reviews.get(str(component_name)) or {})
        return existing_component_payload, existing_run_payload

    if _authoritative_approval_ok(authoritative_approval):
        _mirror_authoritative_approval(refined_parent, refined_run, authoritative_approval)
    refined.attrs["component_review_statuses"] = component_reviews
    refined.attrs["refined_subject_mask_review_status"] = run_payload
    refined_parent.attrs["refined_subject_mask_review_status_latest"] = refined_run
    returned_run_payload = dict(run_payload)
    returned_run_payload["authoritative_approval"] = authoritative_approval
    return component_payload, returned_run_payload


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
    compute_workers: int = 1,
) -> dict[str, np.ndarray]:
    if component_sources is None:
        if source is None:
            raise RuntimeError("component_sources or source is required.")
        component_sources = {name: source for name in refined.component_names}
    comp_idx = int(refined.component_to_index[component_name])
    row_count = int(len(roi_indices))
    component_masks = np.asarray(edited_masks_batch[:, comp_idx], dtype=np.uint8)
    spatial_metrics = batch_mask_spatial_metrics(component_masks)
    mask_present = np.asarray(spatial_metrics["mask_present"], dtype=bool)
    area_px = np.asarray(spatial_metrics["area_px"], dtype=np.float32)
    edit_applied = np.zeros((row_count,), dtype=bool)
    centroid_xy = np.asarray(spatial_metrics["centroid_xy"], dtype=np.float32)
    centroid_valid = np.asarray(spatial_metrics["centroid_valid"], dtype=bool)
    bbox_xyxy = np.asarray(spatial_metrics["bbox_xyxy"], dtype=np.float32)
    bbox_valid = np.asarray(spatial_metrics["bbox_valid"], dtype=bool)
    component_count = np.zeros((row_count,), dtype=np.int32)
    largest_component_fraction = np.zeros((row_count,), dtype=np.float32)
    hole_count = np.zeros((row_count,), dtype=np.int32)
    hole_area_fraction = np.zeros((row_count,), dtype=np.float32)
    sigma_noise = np.zeros((row_count,), dtype=np.float32)
    curvature_var = np.zeros((row_count,), dtype=np.float32)
    ipr = np.zeros((row_count,), dtype=np.float32)
    solidity = np.zeros((row_count,), dtype=np.float32)
    source_row_fingerprint = np.zeros((row_count,), dtype=np.uint64)
    manual_override = np.zeros((row_count,), dtype=bool)
    source_row_stale = np.zeros((row_count,), dtype=bool)
    reason_labels = np.empty((row_count,), dtype=object)

    source_masks: list[np.ndarray] = []
    for roi_idx in roi_indices:
        source_mask = _source_seed_mask_for_component(refined, component_name, int(roi_idx))
        if source_mask is None:
            source_mask = _source_mask_for_resolved_component(component_sources, component_name, int(roi_idx))
        source_masks.append(np.asarray(source_mask, dtype=np.uint8).copy())

    def _compute_single_row(row_offset: int) -> tuple[int, bool, int, float, int, float, float, float, float, float, np.uint64, str]:
        current_mask = np.asarray(component_masks[int(row_offset)], dtype=np.uint8)
        source_mask = source_masks[int(row_offset)]
        edited = not np.array_equal(current_mask, source_mask)
        comp_count, largest_frac, holes, hole_frac = _compute_single_mask_topology_metrics(current_mask)
        return (
            int(row_offset),
            bool(edited),
            int(comp_count),
            float(largest_frac),
            int(holes),
            float(hole_frac),
            float(_compute_single_mask_sigma_noise(current_mask)),
            float(_compute_single_mask_curvature_var(current_mask)),
            float(_compute_single_mask_ipr(current_mask)),
            float(_compute_single_mask_solidity(current_mask)),
            _mask_row_fingerprint(source_mask),
            _default_reason_label(source_mask, current_mask, edited),
        )

    worker_count = max(1, int(compute_workers or 1))
    if worker_count > 1 and row_count > 1:
        with ThreadPoolExecutor(max_workers=min(worker_count, row_count)) as executor:
            row_results = list(executor.map(_compute_single_row, range(row_count)))
    else:
        row_results = [_compute_single_row(row_offset) for row_offset in range(row_count)]

    for (
        row_offset,
        edited,
        comp_count,
        largest_frac,
        holes,
        hole_frac,
        sigma_noise_value,
        curvature_var_value,
        ipr_value,
        solidity_value,
        source_fingerprint,
        reason_label,
    ) in row_results:
        edit_applied[row_offset] = edited
        component_count[row_offset] = np.int32(comp_count)
        largest_component_fraction[row_offset] = np.float32(largest_frac)
        hole_count[row_offset] = np.int32(holes)
        hole_area_fraction[row_offset] = np.float32(hole_frac)
        sigma_noise[row_offset] = np.float32(sigma_noise_value)
        curvature_var[row_offset] = np.float32(curvature_var_value)
        ipr[row_offset] = np.float32(ipr_value)
        solidity[row_offset] = np.float32(solidity_value)
        source_row_fingerprint[row_offset] = source_fingerprint
        manual_override[row_offset] = bool(edited)
        source_row_stale[row_offset] = False
        reason_labels[row_offset] = reason_label

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
        "source_row_fingerprint": source_row_fingerprint,
        "manual_override": manual_override,
        "source_row_stale": source_row_stale,
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
    source_row_fingerprint = component_group["source_row_fingerprint"]
    manual_override = component_group["manual_override"]
    source_row_stale = component_group["source_row_stale"]
    reason_labels = _load_or_init_reason_labels(component_group, int(run_group["masks_roi"].shape[0]))
    pending_rows = sorted(set(int(v) for v in (component_group.attrs.get("source_update_pending_rows") or [])))

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
        source_row_fingerprint[int(roi_idx)] = np.uint64(component_updates["source_row_fingerprint"][row_offset])
        manual_override[int(roi_idx)] = bool(component_updates["manual_override"][row_offset])
        source_row_stale[int(roi_idx)] = bool(component_updates["source_row_stale"][row_offset])
        reason_labels[int(roi_idx)] = str(component_updates["reason_labels"][row_offset])
        _write_reason_label_row(component_group, reason_labels, int(roi_idx))
        if int(roi_idx) in pending_rows:
            pending_rows.remove(int(roi_idx))

    component_group.attrs["source_update_pending_rows"] = pending_rows
    sync_source_subject_mask_stale_payload(run_group)


def _finalize_refined_subject_apply(
    refined: RefinedSubjectMaskRun,
    *,
    updated_components: Optional[Sequence[str]] = None,
    updated_rows: Sequence[int] = (),
    stale_reason: str = "dense_refined_subject_mask_edit",
) -> str:
    updated_at_utc = _utc_now()
    refined.group.attrs["updated_at_utc"] = updated_at_utc
    refined.parent.attrs["latest"] = refined.run_name
    if {"eye_left", "eye_right"}.issubset(set(refined.component_names)):
        from ..shared.refined_subject_eye_geometry import write_refined_subject_eye_geometry

        write_refined_subject_eye_geometry(refined.group, updated_components=updated_components)
    if "mask_bitpacked" in refined.group:
        refresh_bitpacked_mask_store_from_dense(
            refined.group,
            component_names=refined.component_names,
            refresh_components=updated_components,
            refresh_rows=updated_rows,
            source_path=f"refined_subject_masks_runs/{refined.run_name}",
            validation_mode="invariants",
        )
    mark_mask_rle_stale_attrs(
        refined.group,
        updated_at_utc=updated_at_utc,
        updated_components=updated_components,
        updated_rows=updated_rows,
        reason=stale_reason,
    )
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
    update_reason: Optional[str] = None,
    compute_workers: int = 1,
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
            compute_workers=compute_workers,
        )
        _write_refined_subject_component_apply_rows(
            refined=refined,
            component_name=component_name,
            roi_indices=normalized_rows,
            edited_masks_batch=edited_batch,
            component_updates=component_updates,
        )
        refresh_component_contour_rows_from_masks(
            run_group,
            component_name,
            normalized_rows,
            reason=str(update_reason or f"{update_mode}_refined_subject_mask_edit"),
        )

    updated_at_utc = _finalize_refined_subject_apply(
        refined,
        updated_components=normalized_components,
        updated_rows=normalized_rows,
        stale_reason=str(update_reason or f"{update_mode}_refined_subject_mask_edit"),
    )
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
    component_names: Optional[Sequence[str]] = None,
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
        component_names=component_names,
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
    manual_override_arr = component_group.get("manual_override")
    source_row_stale_arr = component_group.get("source_row_stale")
    source_row_fingerprint_arr = component_group.get("source_row_fingerprint")
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
        bool(np.asarray(manual_override_arr[int(roi_idx)], dtype=bool)) if manual_override_arr is not None else False,
        bool(np.asarray(source_row_stale_arr[int(roi_idx)], dtype=bool)) if source_row_stale_arr is not None else False,
        int(np.asarray(source_row_fingerprint_arr[int(roi_idx)], dtype=np.uint64))
        if source_row_fingerprint_arr is not None
        else 0,
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


def _resolve_refined_subject_apply_context(
    zarr_path: str | Path,
    *,
    refined_run: str,
    component_name: str,
    source_subject_mask_run: Optional[str] = None,
) -> _RefinedSubjectApplyContext:
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

    comp_idx = int(refined.component_to_index[normalized_component])
    available_arr = refined.group.get("available_channels")
    if available_arr is not None:
        available = np.asarray(available_arr[:], dtype=bool).reshape(-1)
        if comp_idx >= int(available.shape[0]) or not bool(available[comp_idx]):
            raise RuntimeError(
                f"Component '{normalized_component}' is marked unavailable in refined_subject_masks_runs/{refined_run}."
            )

    component_group = refined.group.require_group("components").require_group(normalized_component)
    return _RefinedSubjectApplyContext(
        source=source,
        refined=refined,
        component_sources=component_sources,
        component_name=normalized_component,
        comp_idx=comp_idx,
        component_group=component_group,
    )


def _read_component_row_revision(component_group: zarr.Group, roi_idx: int) -> int:
    revision_arr = component_group.get("row_revision")
    if revision_arr is None:
        return 0
    return int(np.asarray(revision_arr[int(roi_idx)], dtype=np.int64))


def _read_component_contour_len(component_group: zarr.Group, roi_idx: int) -> int:
    contours = component_group.get("contours")
    if not isinstance(contours, zarr.Group) or "len" not in contours:
        return 0
    return int(np.asarray(contours["len"][int(roi_idx)], dtype=np.int32))


def _validate_refined_subject_mask_edit_row(
    *,
    run_group: zarr.Group,
    component_group: zarr.Group,
    comp_idx: int,
    roi_idx: int,
    mask: np.ndarray,
) -> None:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    expected_area = float(np.count_nonzero(binary))
    expected_present = bool(expected_area > 0.0)
    metrics = run_group["metrics"]
    actual_present = bool(np.asarray(metrics["mask_present"][int(roi_idx), int(comp_idx)], dtype=bool))
    actual_area = float(np.asarray(metrics["area_px"][int(roi_idx), int(comp_idx)], dtype=np.float32))
    component_present = bool(np.asarray(component_group["mask_present"][int(roi_idx)], dtype=bool))
    component_area = float(np.asarray(component_group["area_px"][int(roi_idx)], dtype=np.float32))
    if actual_present != expected_present or component_present != expected_present:
        raise RuntimeError(
            f"Post-write validation failed for row {int(roi_idx)}: mask_present does not match mask pixels."
        )
    if not np.isclose(actual_area, expected_area) or not np.isclose(component_area, expected_area):
        raise RuntimeError(
            f"Post-write validation failed for row {int(roi_idx)}: area_px does not match mask pixels."
        )


def write_refined_subject_mask_edit(
    zarr_path: str | Path,
    *,
    refined_run: str,
    component_name: str,
    roi_index: int,
    mask: np.ndarray,
    reason: str = DEFAULT_REFINED_SUBJECT_WRITEBACK_REASON,
    source_subject_mask_run: Optional[str] = None,
    validate: bool = False,
) -> dict[str, object]:
    """Write one refined subject-mask row/component and refresh Palette-owned metadata."""

    context = _resolve_refined_subject_apply_context(
        zarr_path,
        refined_run=refined_run,
        component_name=component_name,
        source_subject_mask_run=source_subject_mask_run,
    )
    run_group = context.refined.group
    total_rois = int(run_group["masks_roi"].shape[0])
    normalized_rows = _normalize_roi_indices([int(roi_index)], total_rois)
    roi_idx = int(normalized_rows[0])
    expected_mask_shape = tuple(int(dim) for dim in run_group["masks_roi"].shape[2:])
    binary_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if tuple(binary_mask.shape) != expected_mask_shape:
        raise ValueError(f"mask shape mismatch: expected {expected_mask_shape}, got {tuple(binary_mask.shape)}")

    row_revision_before = _read_component_row_revision(context.component_group, roi_idx)
    current_stack = np.asarray(run_group["masks_roi"][roi_idx], dtype=np.uint8)
    previous_mask = np.asarray(current_stack[context.comp_idx], dtype=np.uint8)
    mask_changed = not np.array_equal(previous_mask, binary_mask)
    if mask_changed:
        edited_stack = current_stack.copy()
        edited_stack[context.comp_idx] = binary_mask
        _apply_refined_subject_roi_rows(
            component_sources=context.component_sources,
            refined=context.refined,
            roi_indices=(roi_idx,),
            edited_masks_batch=edited_stack,
            component_names=(context.component_name,),
            update_mode="external_writeback",
            update_method=REFINED_SUBJECT_WRITEBACK_METHOD,
            update_reason=str(reason or DEFAULT_REFINED_SUBJECT_WRITEBACK_REASON),
        )

    if validate:
        _validate_refined_subject_mask_edit_row(
            run_group=run_group,
            component_group=context.component_group,
            comp_idx=context.comp_idx,
            roi_idx=roi_idx,
            mask=binary_mask,
        )

    row_revision_after = _read_component_row_revision(context.component_group, roi_idx)
    edit_applied = bool(np.asarray(run_group["edit_applied"][roi_idx, context.comp_idx], dtype=bool))
    return {
        "ok": True,
        "status": "updated" if mask_changed else "noop",
        "zarr_path": str(Path(zarr_path)),
        "refined_run": context.refined.run_name,
        "source_subject_mask_run": context.source.run_name,
        "roi_index": roi_idx,
        "component_name": context.component_name,
        "row_revision_before": int(row_revision_before),
        "row_revision_after": int(row_revision_after),
        "edit_applied": edit_applied,
        "mask_changed": bool(mask_changed),
        "contour_points": int(_read_component_contour_len(context.component_group, roi_idx)),
        "updated_at_utc": str(run_group.attrs.get("updated_at_utc") or ""),
        "reason": str(reason or DEFAULT_REFINED_SUBJECT_WRITEBACK_REASON),
        "validated": bool(validate),
    }


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


def check_refined_subject_source_updates(
    zarr_path: str | Path,
    *,
    refined_run: str,
    component_name: str,
    roi_indices: Sequence[int],
    source_subject_mask_run: Optional[str] = None,
    assume_source_changed_untracked: bool = False,
    force_source_changed: bool = False,
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

    existing_component_group = existing_run.require_group("components").require_group(normalized_component)
    total_rois_existing = int(existing_run["masks_roi"].shape[0])
    had_source_fingerprint_tracking = _has_component_source_sync_tracking(
        existing_component_group,
        total_rois=total_rois_existing,
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
    _primary_source, component_sources = _load_refined_component_source_runs(root, refined, default_source=source)

    run_group = refined.group
    total_rois = int(run_group["masks_roi"].shape[0])
    normalized_rows = _normalize_roi_indices(roi_indices, total_rois)
    comp_idx = int(refined.component_to_index[normalized_component])
    component_group = run_group.require_group("components").require_group(normalized_component)

    current_component_masks = np.asarray(run_group["masks_roi"][:, comp_idx], dtype=np.uint8)
    source_seed_masks = component_group.get("source_seed_masks_roi")
    if source_seed_masks is not None:
        source_component_masks = np.asarray(source_seed_masks[:], dtype=np.uint8)
    else:
        source_component_masks = _component_masks_from_source(
            component_sources[normalized_component],
            normalized_component,
        )
    _ensure_component_source_sync_arrays(
        component_group,
        total_rois=total_rois,
        source_masks=source_component_masks,
        current_masks=current_component_masks,
    )

    source_row_fingerprint_arr = component_group["source_row_fingerprint"]
    manual_override_arr = component_group["manual_override"]
    source_row_stale_arr = component_group["source_row_stale"]

    auto_sync_rows: list[int] = []
    stale_rows: list[int] = []
    unchanged_rows: list[int] = []

    for roi_idx in normalized_rows:
        current_source_mask = np.asarray(source_component_masks[int(roi_idx)], dtype=np.uint8)
        current_source_fingerprint = _mask_row_fingerprint(current_source_mask)
        previous_source_fingerprint = np.uint64(np.asarray(source_row_fingerprint_arr[int(roi_idx)], dtype=np.uint64))
        treat_as_source_changed = bool(force_source_changed) or (
            bool(assume_source_changed_untracked) and not had_source_fingerprint_tracking
        )
        if not treat_as_source_changed and int(current_source_fingerprint) == int(previous_source_fingerprint):
            unchanged_rows.append(int(roi_idx))
            continue

        source_row_fingerprint_arr[int(roi_idx)] = np.uint64(current_source_fingerprint)
        if bool(np.asarray(manual_override_arr[int(roi_idx)], dtype=bool)):
            source_row_stale_arr[int(roi_idx)] = True
            stale_rows.append(int(roi_idx))
        else:
            source_row_stale_arr[int(roi_idx)] = False
            auto_sync_rows.append(int(roi_idx))

    auto_synced_rows: list[int] = []
    updated_at_utc = str(run_group.attrs.get("updated_at_utc") or "")
    if auto_sync_rows:
        edited_masks_batch = np.stack(
            [np.asarray(run_group["masks_roi"][int(roi_idx)], dtype=np.uint8) for roi_idx in auto_sync_rows],
            axis=0,
        )
        for row_offset, roi_idx in enumerate(auto_sync_rows):
            edited_masks_batch[row_offset, comp_idx] = np.asarray(source_component_masks[int(roi_idx)], dtype=np.uint8)
        _apply_refined_subject_roi_rows(
            component_sources=component_sources,
            refined=refined,
            roi_indices=auto_sync_rows,
            edited_masks_batch=edited_masks_batch,
            component_names=(normalized_component,),
            update_mode="sync",
            update_method=REFINED_SUBJECT_SOURCE_UPDATE_METHOD,
        )
        auto_synced_rows = list(auto_sync_rows)
        updated_at_utc = str(run_group.attrs.get("updated_at_utc") or "")

    if stale_rows:
        updated_at_utc = _utc_now()
        run_group.attrs["updated_at_utc"] = updated_at_utc
        existing_pending_raw = component_group.attrs.get("source_update_pending_rows")
        if isinstance(existing_pending_raw, (list, tuple)):
            existing_pending = [int(v) for v in existing_pending_raw]
        elif existing_pending_raw is None:
            existing_pending = []
        else:
            existing_pending = [int(existing_pending_raw)]
        component_group.attrs["source_update_pending_rows"] = sorted(set(existing_pending + stale_rows))
        _write_refined_component_last_update(
            run_group,
            component_name=normalized_component,
            updated_at_utc=updated_at_utc,
            update_mode="sync",
            update_method=REFINED_SUBJECT_SOURCE_UPDATE_METHOD,
        )
        existing_review = dict(run_group.attrs.get("component_review_statuses") or {}).get(normalized_component, {})
        existing_intended_use = str(existing_review.get("intended_use") or DEFAULT_REVIEW_INTENDED_USE)
        apply_component_review_status(
            refined.parent,
            refined.run_name,
            refined.group,
            component_name=normalized_component,
            state="needs_review",
            method=REFINED_SUBJECT_SOURCE_UPDATE_METHOD,
            intended_use=existing_intended_use,
            reviewer=None,
            notes=f"source_subject_mask_rows_changed:{','.join(str(idx) for idx in stale_rows)}",
        )

    sync_source_subject_mask_stale_payload(run_group)
    stale_total = int(np.count_nonzero(np.asarray(source_row_stale_arr[:], dtype=bool)))
    return {
        "status": "updated",
        "zarr_path": str(Path(zarr_path)),
        "refined_run": refined.run_name,
        "component_name": normalized_component,
        "source_subject_mask_run": source.run_name,
        "assume_source_changed_untracked": bool(assume_source_changed_untracked),
        "force_source_changed": bool(force_source_changed),
        "roi_indices": normalized_rows,
        "roi_count": int(len(normalized_rows)),
        "unchanged_roi_count": int(len(unchanged_rows)),
        "source_changed_roi_count": int(len(auto_sync_rows) + len(stale_rows)),
        "auto_synced_roi_count": int(len(auto_synced_rows)),
        "stale_marked_roi_count": int(len(stale_rows)),
        "auto_synced_roi_indices": auto_synced_rows,
        "stale_roi_indices": stale_rows,
        "unchanged_roi_indices": unchanged_rows,
        "stale_total": stale_total,
        "updated_at_utc": updated_at_utc,
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
    image_bgr = np.asarray(image, dtype=np.uint8)
    h, w = image_bgr.shape[:2]
    panel = np.zeros((h + PANEL_LABEL_HEIGHT, w, 3), dtype=np.uint8)
    panel[PANEL_LABEL_HEIGHT:, :] = image_bgr
    cv2.putText(panel, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 255, 220), 1, cv2.LINE_AA)
    return panel


def _blank_panel_like(roi_image: np.ndarray, title: str, note: str) -> np.ndarray:
    image = np.zeros((int(roi_image.shape[0]), int(roi_image.shape[1]), 3), dtype=np.uint8)
    panel = _panel(title, image)
    cv2.putText(panel, note, (10, PANEL_LABEL_HEIGHT + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return panel


def _noop_trackbar(_value: int) -> None:
    return


def _scale_panel_image(image: np.ndarray, *, zoom: int) -> np.ndarray:
    zoom_value = max(1, int(zoom))
    if zoom_value <= 1:
        return np.asarray(image, dtype=np.uint8).copy()
    return cv2.resize(
        np.asarray(image, dtype=np.uint8),
        None,
        fx=float(zoom_value),
        fy=float(zoom_value),
        interpolation=cv2.INTER_NEAREST,
    )


def _mask_patch_bounds(mask: np.ndarray, *, padding: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.uint8) > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    pad = max(0, int(padding))
    return (
        max(0, int(xs.min()) - pad),
        int(xs.max()) + pad + 1,
        max(0, int(ys.min()) - pad),
        int(ys.max()) + pad + 1,
    )


def _resolve_active_patch_bounds(
    roi_shape: tuple[int, int],
    anchor_mask: np.ndarray,
    source_mask: np.ndarray,
    *,
    padding: int,
) -> tuple[int, int, int, int]:
    roi_h = int(roi_shape[0])
    roi_w = int(roi_shape[1])
    combined_mask = np.logical_or(
        np.asarray(anchor_mask, dtype=np.uint8) > 0,
        np.asarray(source_mask, dtype=np.uint8) > 0,
    ).astype(np.uint8)
    bounds = _mask_patch_bounds(combined_mask, padding=padding)
    if bounds is not None:
        x0, x1, y0, y1 = bounds
        return max(0, x0), min(roi_w, x1), max(0, y0), min(roi_h, y1)

    pad = max(1, int(padding))
    cx = roi_w // 2
    cy = roi_h // 2
    x0 = max(0, cx - pad)
    x1 = min(roi_w, cx + pad + 1)
    y0 = max(0, cy - pad)
    y1 = min(roi_h, cy + pad + 1)
    return x0, x1, y0, y1


def _shift_patch_bounds(
    bounds: tuple[int, int, int, int],
    roi_shape: tuple[int, int],
    *,
    dx: int = 0,
    dy: int = 0,
) -> tuple[int, int, int, int]:
    roi_h = int(roi_shape[0])
    roi_w = int(roi_shape[1])
    x0, x1, y0, y1 = (int(value) for value in bounds)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    if width >= roi_w:
        new_x0 = 0
        new_x1 = roi_w
    else:
        new_x0 = min(max(0, x0 + int(dx)), roi_w - width)
        new_x1 = new_x0 + width

    if height >= roi_h:
        new_y0 = 0
        new_y1 = roi_h
    else:
        new_y0 = min(max(0, y0 + int(dy)), roi_h - height)
        new_y1 = new_y0 + height

    return int(new_x0), int(new_x1), int(new_y0), int(new_y1)


def _fill_polygon_mask(
    mask: np.ndarray,
    roi_points: Sequence[tuple[int, int]],
    *,
    fill_value: int,
    invert: bool = False,
) -> bool:
    if len(roi_points) < 3:
        return False
    mask_arr = np.asarray(mask, dtype=np.uint8)
    roi_h, roi_w = (int(dim) for dim in mask_arr.shape[:2])
    polygon = np.asarray(roi_points, dtype=np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(0, roi_w - 1))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(0, roi_h - 1))
    if invert:
        fill_mask = np.zeros(mask_arr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(fill_mask, [polygon.reshape(-1, 1, 2)], 1)
        mask_arr[fill_mask == 0] = np.uint8(fill_value)
    else:
        cv2.fillPoly(mask_arr, [polygon.reshape(-1, 1, 2)], int(fill_value))
    return True


def _append_lasso_point(
    roi_points: list[tuple[int, int]],
    point: tuple[int, int],
    *,
    min_step_px: int = LASSO_MIN_POINT_STEP_PX,
) -> bool:
    candidate = (int(point[0]), int(point[1]))
    if not roi_points:
        roi_points.append(candidate)
        return True
    last_x, last_y = roi_points[-1]
    dx = int(candidate[0]) - int(last_x)
    dy = int(candidate[1]) - int(last_y)
    min_step = max(0, int(min_step_px))
    if (dx * dx) + (dy * dy) < (min_step * min_step):
        return False
    roi_points.append(candidate)
    return True


def _panel_xy_for_roi_point(
    panel_region: ReviewPanelRegion,
    roi_point: tuple[int, int],
) -> tuple[int, int]:
    zoom = max(1, int(panel_region.zoom))
    local_x = int(roi_point[0]) - int(panel_region.patch_x0)
    local_y = int(roi_point[1]) - int(panel_region.patch_y0)
    panel_x = int(panel_region.x + round(float(local_x) * zoom))
    panel_y = int(panel_region.y + int(panel_region.label_h) + round(float(local_y) * zoom))
    return panel_x, panel_y


def _draw_panel_cursor(
    image: np.ndarray,
    *,
    panel_region: ReviewPanelRegion | None,
    roi_xy: tuple[int, int] | None,
    brush_radius: int,
    color: tuple[int, int, int],
    style: str = "circle",
) -> None:
    if panel_region is None or roi_xy is None:
        return
    local_roi_x = int(roi_xy[0]) - int(panel_region.patch_x0)
    local_roi_y = int(roi_xy[1]) - int(panel_region.patch_y0)
    patch_w = int(panel_region.patch_w) if panel_region.patch_w is not None else 0
    patch_h = int(panel_region.patch_h) if panel_region.patch_h is not None else 0
    if patch_w > 0 and patch_h > 0 and not (0 <= local_roi_x < patch_w and 0 <= local_roi_y < patch_h):
        return
    zoom = max(1, int(panel_region.zoom))
    center_x = int(panel_region.x + round(float(local_roi_x) * zoom))
    center_y = int(panel_region.y + int(panel_region.label_h) + round(float(local_roi_y) * zoom))
    if str(style).strip().lower() == "crosshair":
        marker_size = max(11, int(round(float(max(1, brush_radius)) * zoom * 3.0)))
        cv2.drawMarker(
            image,
            (center_x, center_y),
            color,
            cv2.MARKER_CROSS,
            markerSize=marker_size,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        return
    cv2.circle(
        image,
        (center_x, center_y),
        max(1, int(round(float(brush_radius) * zoom))),
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_lasso_overlay(
    image: np.ndarray,
    *,
    panel_region: ReviewPanelRegion | None,
    roi_points: Sequence[tuple[int, int]],
    cursor_roi_xy: tuple[int, int] | None,
) -> None:
    if panel_region is None or not roi_points:
        return
    panel_points = np.asarray(
        [_panel_xy_for_roi_point(panel_region, point) for point in roi_points],
        dtype=np.int32,
    )
    if panel_points.shape[0] >= 2:
        cv2.polylines(
            image,
            [panel_points.reshape(-1, 1, 2)],
            False,
            LASSO_OUTLINE_COLOR,
            LASSO_LINE_THICKNESS,
            cv2.LINE_AA,
        )
    marker_indices = {0, int(panel_points.shape[0] - 1)}
    for idx in sorted(marker_indices):
        point = panel_points[idx]
        color = LASSO_START_COLOR if idx == 0 else LASSO_OUTLINE_COLOR
        cv2.circle(
            image,
            (int(point[0]), int(point[1])),
            LASSO_POINT_RADIUS,
            color,
            -1,
            cv2.LINE_AA,
        )
    if cursor_roi_xy is not None:
        cursor_point = _panel_xy_for_roi_point(panel_region, cursor_roi_xy)
        last_point = tuple(int(v) for v in panel_points[-1])
        cv2.line(
            image,
            last_point,
            cursor_point,
            LASSO_PREVIEW_COLOR,
            LASSO_PREVIEW_THICKNESS,
            cv2.LINE_AA,
        )


def _stack_h(
    panels: Sequence[np.ndarray],
    *,
    gap: int = PANEL_GAP,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8), []
    normalized = [np.asarray(panel, dtype=np.uint8) for panel in panels]
    height = max(int(panel.shape[0]) for panel in normalized)
    width = sum(int(panel.shape[1]) for panel in normalized) + max(0, len(normalized) - 1) * int(gap)
    stacked = np.zeros((height, width, 3), dtype=np.uint8)
    regions: list[tuple[int, int, int, int]] = []
    offset_x = 0
    for panel in normalized:
        panel_h = int(panel.shape[0])
        panel_w = int(panel.shape[1])
        stacked[0:panel_h, offset_x : offset_x + panel_w] = panel
        regions.append((int(offset_x), 0, panel_w, panel_h))
        offset_x += panel_w + int(gap)
    return stacked, regions


def _stack_v(
    panels: Sequence[np.ndarray],
    *,
    gap: int = PANEL_GAP,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8), []
    normalized = [np.asarray(panel, dtype=np.uint8) for panel in panels]
    width = max(int(panel.shape[1]) for panel in normalized)
    height = sum(int(panel.shape[0]) for panel in normalized) + max(0, len(normalized) - 1) * int(gap)
    stacked = np.zeros((height, width, 3), dtype=np.uint8)
    regions: list[tuple[int, int, int, int]] = []
    offset_y = 0
    for panel in normalized:
        panel_h = int(panel.shape[0])
        panel_w = int(panel.shape[1])
        stacked[offset_y : offset_y + panel_h, 0:panel_w] = panel
        regions.append((0, int(offset_y), panel_w, panel_h))
        offset_y += panel_h + int(gap)
    return stacked, regions


def _map_review_pointer_to_roi(
    *,
    x: int,
    y: int,
    display_scale: float,
    panel_regions: Mapping[str, ReviewPanelRegion],
    roi_shape: tuple[int, int],
) -> tuple[str, int, int] | None:
    raw_x = int(x / display_scale) if display_scale > 0 else int(x)
    raw_y = int(y / display_scale) if display_scale > 0 else int(y)
    roi_h = int(roi_shape[0])
    roi_w = int(roi_shape[1])
    for panel_name in ("edit", "all", "roi"):
        region = panel_regions.get(panel_name)
        if region is None:
            continue
        if not (region.x <= raw_x < region.x + region.width and region.y <= raw_y < region.y + region.height):
            continue
        local_x = raw_x - region.x
        local_y = raw_y - region.y
        label_h = max(0, int(region.label_h))
        if local_y < label_h:
            return None
        zoom = max(1, int(region.zoom))
        patch_x = int(local_x / zoom)
        patch_y = int((local_y - label_h) / zoom)
        patch_w = int(region.patch_w) if region.patch_w is not None else roi_w
        patch_h = int(region.patch_h) if region.patch_h is not None else roi_h
        if not (0 <= patch_x < patch_w and 0 <= patch_y < patch_h):
            return None
        roi_x = int(region.patch_x0) + patch_x
        roi_y = int(region.patch_y0) + patch_y
        if 0 <= roi_x < roi_w and 0 <= roi_y < roi_h:
            return region.name, roi_x, roi_y
        return None
    return None


def _mouse_modifier_state(flags: int) -> tuple[bool, bool, bool]:
    ctrl_down = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
    shift_down = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
    left_down = bool(flags & cv2.EVENT_FLAG_LBUTTON)
    return ctrl_down, shift_down, left_down


def _resolve_erase_mode(base_erase_mode: bool, shift_down: bool) -> bool:
    # Shift acts as a temporary inverse while drawing.
    return (not base_erase_mode) if shift_down else base_erase_mode


def launch_review(
    zarr_path: str,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    component: Optional[str] = None,
    roi_index: int = 0,
    approve_state: str = "approved",
    review_method: str = DEFAULT_REVIEW_METHOD,
    review_intended_use: str = DEFAULT_REVIEW_INTENDED_USE,
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
    edit_zoom: int = DEFAULT_EDIT_ZOOM,
    edit_padding: int = DEFAULT_EDIT_PADDING,
    display_scale: float = DISPLAY_SCALE,
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
    approve_state = str(approve_state).strip().lower() or "approved"
    if approve_state not in REVIEW_STATE_CHOICES:
        raise RuntimeError(
            f"Unsupported approve_state={approve_state!r}; expected one of {', '.join(REVIEW_STATE_CHOICES)}."
        )
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
    lasso_mode = False
    lasso_drawing = False
    lasso_points: list[tuple[int, int]] = []
    edit_zoom = max(1, min(int(edit_zoom), MAX_EDIT_ZOOM))
    edit_padding = max(1, min(int(edit_padding), MAX_EDIT_PADDING))
    display_scale = max(MIN_WINDOW_SCALE_PERCENT / 100.0, float(display_scale))

    _require_gui_display()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1900, 1100)
    cv2.createTrackbar("Padding", WINDOW_NAME, int(edit_padding), MAX_EDIT_PADDING, _noop_trackbar)
    cv2.createTrackbar("Edit Zoom", WINDOW_NAME, int(edit_zoom), MAX_EDIT_ZOOM, _noop_trackbar)
    cv2.createTrackbar(
        "Scale %",
        WINDOW_NAME,
        max(
            MIN_WINDOW_SCALE_PERCENT,
            min(int(round(float(display_scale) * 100.0)), MAX_WINDOW_SCALE_PERCENT),
        ),
        MAX_WINDOW_SCALE_PERCENT,
        _noop_trackbar,
    )

    refined_masks_arr = refined.group["masks_roi"]
    _primary_source, component_sources = _load_refined_component_source_runs(root, refined, default_source=source)
    cv2.namedWindow(EDIT_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(EDIT_WINDOW_NAME, 1200, 900)

    main_display_layout: dict[str, ReviewPanelRegion] = {}
    edit_display_layout: dict[str, ReviewPanelRegion] = {}
    roi_img: np.ndarray | None = None
    original_masks: list[np.ndarray] = []
    edit_masks: list[np.ndarray] = []
    patch_anchor_masks: list[np.ndarray] = []
    patch_pan_offsets: list[tuple[int, int]] = []
    cached_roi_panel: np.ndarray | None = None
    cached_source_panel: np.ndarray | None = None
    cached_refined_panel: np.ndarray | None = None
    cached_active_panel: np.ndarray | None = None
    cached_main_combined: np.ndarray | None = None
    cached_edit_combined: np.ndarray | None = None
    cached_header_signature: tuple[object, ...] | None = None
    last_draw_signature: tuple[int, int, bool, int, int] | None = None
    cursor_roi_xy: tuple[int, int] | None = None
    needs_main_redraw = False
    needs_edit_redraw = False

    def request_redraw(*, main: bool = True, edit: bool = True) -> None:
        nonlocal needs_main_redraw, needs_edit_redraw
        if main:
            needs_main_redraw = True
        if edit:
            needs_edit_redraw = True

    def invalidate_main_combined_cache(*, reset_layout: bool = False) -> None:
        nonlocal cached_main_combined, cached_header_signature, main_display_layout
        cached_main_combined = None
        cached_header_signature = None
        if reset_layout:
            main_display_layout = {}

    def invalidate_edit_combined_cache(*, reset_layout: bool = False) -> None:
        nonlocal cached_edit_combined, edit_display_layout
        cached_edit_combined = None
        if reset_layout:
            edit_display_layout = {}

    def invalidate_active_panel_cache() -> None:
        nonlocal cached_active_panel
        cached_active_panel = None
        invalidate_edit_combined_cache(reset_layout=True)

    def invalidate_source_panel_cache() -> None:
        nonlocal cached_source_panel
        cached_source_panel = None
        invalidate_edit_combined_cache(reset_layout=True)

    def invalidate_refined_panel_cache() -> None:
        nonlocal cached_refined_panel
        cached_refined_panel = None
        invalidate_main_combined_cache(reset_layout=True)

    def invalidate_active_component_caches() -> None:
        invalidate_source_panel_cache()
        invalidate_active_panel_cache()
        invalidate_main_combined_cache(reset_layout=True)

    def invalidate_mask_panel_caches(*, refresh_main: bool = True) -> None:
        if refresh_main:
            invalidate_refined_panel_cache()
        invalidate_active_panel_cache()

    def invalidate_all_panel_caches() -> None:
        nonlocal cached_roi_panel, cached_source_panel, cached_refined_panel, cached_active_panel, main_display_layout, edit_display_layout
        cached_roi_panel = None
        cached_source_panel = None
        cached_refined_panel = None
        cached_active_panel = None
        main_display_layout = {}
        edit_display_layout = {}
        invalidate_main_combined_cache(reset_layout=True)
        invalidate_edit_combined_cache(reset_layout=True)

    def _clear_lasso() -> None:
        nonlocal lasso_points, lasso_drawing
        lasso_points = []
        lasso_drawing = False

    def load_current_roi() -> None:
        nonlocal roi_img, original_masks, edit_masks, patch_anchor_masks, patch_pan_offsets, cursor_roi_xy, last_draw_signature
        roi_img = np.asarray(roi_images[current_pos], dtype=np.uint8)
        current_masks = np.asarray(refined_masks_arr[current_pos], dtype=np.uint8)
        original_masks = [current_masks[idx].copy() for idx in range(len(component_names))]
        edit_masks = [current_masks[idx].copy() for idx in range(len(component_names))]
        patch_anchor_masks = []
        patch_pan_offsets = [(0, 0) for _ in component_names]
        for comp_idx, component_name in enumerate(component_names):
            source_mask = _source_mask_for_resolved_component(component_sources, component_name, current_pos)
            patch_anchor_masks.append(
                np.logical_or(
                    np.asarray(original_masks[comp_idx], dtype=np.uint8) > 0,
                    np.asarray(source_mask, dtype=np.uint8) > 0,
                ).astype(np.uint8)
            )
        cursor_roi_xy = None
        last_draw_signature = None
        _clear_lasso()
        invalidate_all_panel_caches()

    def update_display(*, refresh_main: bool = True, refresh_edit: bool = True) -> None:
        nonlocal main_display_layout, edit_display_layout, cached_roi_panel, cached_source_panel, cached_refined_panel, cached_active_panel, cached_main_combined, cached_edit_combined, cached_header_signature, needs_main_redraw, needs_edit_redraw
        if roi_img is None:
            return
        focus_color = COMPONENT_COLORS.get(component_names[active_idx], (255, 255, 255))
        if refresh_edit and not refresh_main and cached_edit_combined is not None and edit_display_layout:
            edit_combined = cached_edit_combined.copy()
            _draw_lasso_overlay(
                edit_combined,
                panel_region=edit_display_layout.get("edit"),
                roi_points=lasso_points,
                cursor_roi_xy=cursor_roi_xy if lasso_mode else None,
            )
            _draw_panel_cursor(
                edit_combined,
                panel_region=edit_display_layout.get("edit"),
                roi_xy=cursor_roi_xy,
                brush_radius=brush_radius,
                color=LASSO_OUTLINE_COLOR if lasso_mode else focus_color,
                style="crosshair" if lasso_mode else "circle",
            )
            if display_scale != 1.0:
                edit_combined = cv2.resize(
                    edit_combined,
                    None,
                    fx=display_scale,
                    fy=display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(EDIT_WINDOW_NAME, edit_combined)
            needs_edit_redraw = False
            return
        source_active_mask = _source_mask_for_resolved_component(component_sources, component_names[active_idx], current_pos)
        active_mask = np.asarray(edit_masks[active_idx], dtype=np.uint8)
        anchor_mask = (
            np.asarray(patch_anchor_masks[active_idx], dtype=np.uint8)
            if active_idx < len(patch_anchor_masks)
            else active_mask
        )
        base_patch_bounds = _resolve_active_patch_bounds(
            tuple(int(dim) for dim in roi_img.shape[:2]),
            anchor_mask,
            source_active_mask,
            padding=edit_padding,
        )
        patch_pan_x, patch_pan_y = patch_pan_offsets[active_idx] if active_idx < len(patch_pan_offsets) else (0, 0)
        patch_x0, patch_x1, patch_y0, patch_y1 = _shift_patch_bounds(
            base_patch_bounds,
            tuple(int(dim) for dim in roi_img.shape[:2]),
            dx=int(patch_pan_x),
            dy=int(patch_pan_y),
        )
        crop_patch = np.asarray(roi_img[patch_y0:patch_y1, patch_x0:patch_x1], dtype=np.uint8)
        source_patch_mask = np.asarray(source_active_mask[patch_y0:patch_y1, patch_x0:patch_x1], dtype=np.uint8)
        active_patch_mask = np.asarray(active_mask[patch_y0:patch_y1, patch_x0:patch_x1], dtype=np.uint8)
        if refresh_main and cached_roi_panel is None:
            cached_roi_panel = _panel("Crop ROI", cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR))
        if refresh_edit and cached_source_panel is None:
            cached_source_panel = (
                _panel(
                    f"Source Patch {component_names[active_idx]}",
                    _overlay_components(crop_patch, (component_names[active_idx],), (source_patch_mask,)),
                )
                if np.count_nonzero(source_patch_mask) > 0
                else _blank_panel_like(crop_patch, f"Source Patch {component_names[active_idx]}", "Unavailable / empty")
            )
        if refresh_main and cached_refined_panel is None:
            cached_refined_panel = _panel("Refined Overlay", _overlay_components(roi_img, component_names, edit_masks))
        if refresh_edit and cached_active_panel is None:
            active_panel = _panel(
                f"Edit Patch {component_names[active_idx]} x{int(edit_zoom)}",
                _scale_panel_image(
                    _overlay_components(crop_patch, (component_names[active_idx],), (active_patch_mask,)),
                    zoom=int(edit_zoom),
                ),
            )
            cached_active_panel = active_panel

        state_payload = dict(refined.group.attrs.get("component_review_statuses") or {}).get(component_names[active_idx], {})
        state_text = str(state_payload.get("state") or "pending")
        brush_mode_text = "erase" if erase_mode else "paint"
        tool_mode_text = "lasso" if lasso_mode else "brush"
        header_signature = (
            int(current_pos),
            int(active_idx),
            int(brush_radius),
            str(state_text),
            bool(erase_mode),
            bool(lasso_mode),
            int(edit_padding),
            int(patch_pan_x),
            int(patch_pan_y),
        )
        if refresh_main and (cached_main_combined is None or cached_header_signature != header_signature):
            header = cached_refined_panel.copy()
            cv2.putText(
                header,
                (
                    f"ROI {current_pos + 1}/{total_rois}  Active: {component_names[active_idx]}  "
                    f"Tool: {tool_mode_text} ({brush_mode_text})  Brush: {brush_radius}  Padding: {edit_padding}  "
                    f"Pan: {patch_pan_x:+d},{patch_pan_y:+d}  Review: {state_text}"
                ),
                (10, int(header.shape[0]) - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cached_main_combined, main_regions = _stack_h((cached_roi_panel, header), gap=PANEL_GAP)
            roi_region = main_regions[0]
            header_region = main_regions[1]
            main_display_layout = {
                "roi": ReviewPanelRegion(
                    name="roi",
                    x=int(roi_region[0]),
                    y=int(roi_region[1]),
                    width=int(roi_region[2]),
                    height=int(roi_region[3]),
                    zoom=1,
                    label_h=int(PANEL_LABEL_HEIGHT),
                    patch_x0=0,
                    patch_y0=0,
                    patch_w=int(roi_img.shape[1]),
                    patch_h=int(roi_img.shape[0]),
                ),
                "all": ReviewPanelRegion(
                    name="all",
                    x=int(header_region[0]),
                    y=int(header_region[1]),
                    width=int(header_region[2]),
                    height=int(header_region[3]),
                    zoom=1,
                    label_h=int(PANEL_LABEL_HEIGHT),
                    patch_x0=0,
                    patch_y0=0,
                    patch_w=int(roi_img.shape[1]),
                    patch_h=int(roi_img.shape[0]),
                ),
            }
            cached_header_signature = header_signature
        if refresh_edit and cached_edit_combined is None:
            cached_edit_combined, edit_regions = _stack_h((cached_source_panel, cached_active_panel), gap=PANEL_GAP)
            active_region = edit_regions[1]
            edit_display_layout = {
                "edit": ReviewPanelRegion(
                    name="edit",
                    x=int(active_region[0]),
                    y=int(active_region[1]),
                    width=int(active_region[2]),
                    height=int(active_region[3]),
                    zoom=int(edit_zoom),
                    label_h=int(PANEL_LABEL_HEIGHT),
                    patch_x0=int(patch_x0),
                    patch_y0=int(patch_y0),
                    patch_w=int(crop_patch.shape[1]),
                    patch_h=int(crop_patch.shape[0]),
                ),
            }

        if refresh_main:
            main_combined = cached_main_combined.copy()
            for panel_name in ("roi", "all"):
                panel_region = main_display_layout.get(panel_name)
                if panel_region is None:
                    continue
                cv2.rectangle(
                    main_combined,
                    (
                        int(panel_region.x + patch_x0),
                        int(panel_region.y + panel_region.label_h + patch_y0),
                    ),
                    (
                        int(panel_region.x + patch_x1 - 1),
                        int(panel_region.y + panel_region.label_h + patch_y1 - 1),
                    ),
                    focus_color,
                    1,
                    cv2.LINE_AA,
                )
        if refresh_edit:
            edit_combined = cached_edit_combined.copy()
            _draw_lasso_overlay(
                edit_combined,
                panel_region=edit_display_layout.get("edit"),
                roi_points=lasso_points,
                cursor_roi_xy=cursor_roi_xy if lasso_mode else None,
            )

        if cursor_roi_xy is not None and refresh_main:
            for panel_name in ("roi", "all"):
                _draw_panel_cursor(
                    main_combined,
                    panel_region=main_display_layout.get(panel_name),
                    roi_xy=cursor_roi_xy,
                    brush_radius=brush_radius,
                    color=LASSO_OUTLINE_COLOR if lasso_mode else focus_color,
                    style="crosshair" if lasso_mode else "circle",
                )
        if cursor_roi_xy is not None and refresh_edit:
            _draw_panel_cursor(
                edit_combined,
                panel_region=edit_display_layout.get("edit"),
                roi_xy=cursor_roi_xy,
                brush_radius=brush_radius,
                color=LASSO_OUTLINE_COLOR if lasso_mode else focus_color,
                style="crosshair" if lasso_mode else "circle",
            )

        if refresh_main and display_scale != 1.0:
            main_combined = cv2.resize(
                main_combined,
                None,
                fx=display_scale,
                fy=display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        if refresh_edit and display_scale != 1.0:
            edit_combined = cv2.resize(
                edit_combined,
                None,
                fx=display_scale,
                fy=display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        if refresh_main:
            cv2.imshow(WINDOW_NAME, main_combined)
            needs_main_redraw = False
        if refresh_edit:
            cv2.imshow(EDIT_WINDOW_NAME, edit_combined)
            needs_edit_redraw = False

    def save_current() -> None:
        stacked = np.stack(edit_masks, axis=0)
        save_refined_subject_roi(
            source=source,
            refined=refined,
            roi_idx=current_pos,
            edited_masks=stacked,
            component_sources=component_sources,
            component_names=(component_names[active_idx],),
        )
        invalidate_mask_panel_caches(refresh_main=True)
        request_redraw(main=True, edit=True)
        print(f"Saved refined subject masks for ROI {current_pos}.")

    def _pan_active_patch(dx: int, dy: int) -> None:
        if active_idx >= len(patch_pan_offsets):
            return
        current_dx, current_dy = patch_pan_offsets[active_idx]
        updated = (int(current_dx) + int(dx), int(current_dy) + int(dy))
        if updated == (int(current_dx), int(current_dy)):
            return
        patch_pan_offsets[active_idx] = updated
        invalidate_active_component_caches()
        request_redraw(main=True, edit=True)

    def _recenter_active_patch() -> None:
        if active_idx >= len(patch_pan_offsets):
            return
        if patch_pan_offsets[active_idx] == (0, 0):
            return
        patch_pan_offsets[active_idx] = (0, 0)
        invalidate_active_component_caches()
        request_redraw(main=True, edit=True)

    def on_edit_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        nonlocal drawing, erase_mode, cursor_roi_xy, last_draw_signature, lasso_points, lasso_drawing
        if roi_img is None:
            return
        ctrl_down, shift_down, left_down = _mouse_modifier_state(int(flags))
        mapped = _map_review_pointer_to_roi(
            x=int(x),
            y=int(y),
            display_scale=float(display_scale),
            panel_regions=edit_display_layout,
            roi_shape=tuple(int(dim) for dim in roi_img.shape[:2]),
        )
        if mapped is None:
            if cursor_roi_xy is not None:
                cursor_roi_xy = None
                request_redraw(main=False, edit=True)
            if lasso_mode and (event == cv2.EVENT_LBUTTONUP or not left_down):
                lasso_drawing = False
            if not lasso_mode and (event == cv2.EVENT_LBUTTONUP or not left_down):
                drawing = False
                last_draw_signature = None
            return
        _panel_name, roi_x, roi_y = mapped

        if lasso_mode:
            point = (int(roi_x), int(roi_y))
            cursor_roi_xy = point
            if event == cv2.EVENT_LBUTTONDOWN:
                lasso_drawing = True
                _append_lasso_point(lasso_points, point)
                request_redraw(main=False, edit=True)
            elif event == cv2.EVENT_MOUSEMOVE:
                if lasso_drawing and left_down:
                    if _append_lasso_point(lasso_points, point):
                        request_redraw(main=False, edit=True)
                else:
                    request_redraw(main=False, edit=True)
            elif event == cv2.EVENT_LBUTTONUP:
                if lasso_drawing:
                    _append_lasso_point(lasso_points, point)
                lasso_drawing = False
                request_redraw(main=False, edit=True)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = bool(ctrl_down)
            last_draw_signature = None
            cursor_roi_xy = (int(roi_x), int(roi_y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if not (ctrl_down and left_down):
                if drawing:
                    invalidate_refined_panel_cache()
                    request_redraw(main=True, edit=True)
                drawing = False
                last_draw_signature = None
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                invalidate_refined_panel_cache()
                request_redraw(main=True, edit=True)
            drawing = False
            last_draw_signature = None
            cursor_roi_xy = None
            if not needs_main_redraw:
                request_redraw(main=False, edit=True)

        if event == cv2.EVENT_MOUSEMOVE and not drawing:
            cursor_roi_xy = (int(roi_x), int(roi_y))
            request_redraw(main=False, edit=True)
        elif drawing:
            cursor_roi_xy = (int(roi_x), int(roi_y))
            current_erase_mode = _resolve_erase_mode(bool(erase_mode), shift_down)
            draw_signature = (int(roi_x), int(roi_y), bool(current_erase_mode), int(active_idx), int(brush_radius))
            if last_draw_signature != draw_signature:
                color = 0 if current_erase_mode else 1
                cv2.circle(edit_masks[active_idx], (roi_x, roi_y), brush_radius, color, -1)
                last_draw_signature = draw_signature
                invalidate_mask_panel_caches(refresh_main=False)
                request_redraw(main=False, edit=True)

    cv2.setMouseCallback(EDIT_WINDOW_NAME, on_edit_mouse)
    load_current_roi()
    request_redraw(main=True, edit=True)

    print("\nRefined Subject Mask Review")
    print(f"  Source run: subject_mask_runs/{source.run_name}")
    print(f"  Refined run: refined_subject_masks_runs/{refined.run_name}")
    print(f"  Crop run: {crop_run_name}")
    print(f"  Components: {', '.join(component_names)}")
    print("Controls:")
    print(f"  Main window: {WINDOW_NAME}")
    print(f"  Edit window: {EDIT_WINDOW_NAME}")
    print("  Mouse on Edit window: hold Ctrl+LMB to draw")
    print("  Mouse while drawing: hold Shift to temporarily invert brush mode")
    print("  1..9: select active component by order shown")
    print("  [ / ]: brush size")
    print("  x: toggle brush mode (paint/erase)")
    print("  v: toggle lasso fill mode")
    print(
        "  In lasso mode: click or drag LMB to add contour points, "
        "f fills inside, g fills outside, e erases outside, u removes last point, d clears contour"
    )
    print("  Padding trackbar: expand/shrink the focused crop around the active mask")
    print("  i/j/k/l: pan edit crop up/left/down/right")
    print("  I/J/K/L: coarse pan edit crop up/left/down/right")
    print("  c: recenter edit crop on the anchored mask bounds")
    print("  - / =: decrease/increase editor zoom")
    print("  , / .: decrease/increase window scale")
    print("  Padding / Edit Zoom / Scale % trackbars: focused crop, editor zoom, and shared window scale")
    print("  s: save current ROI edits")
    print("  r: reset current ROI to stored refined masks")
    print("  n/p: next/previous ROI")
    print(f"  a: set active component review to {approve_state}")
    print("  N: mark active component needs_review")
    print("  R: mark active component rejected")
    print("  P: mark active component pending")
    print("  q/ESC: quit")

    def _adjust_trackbar(name: str, delta: int, minimum: int, maximum: int) -> int:
        current = int(cv2.getTrackbarPos(name, WINDOW_NAME))
        updated = max(int(minimum), min(int(maximum), current + int(delta)))
        if updated != current:
            cv2.setTrackbarPos(name, WINDOW_NAME, updated)
        return updated

    while True:
        current_padding = max(1, int(cv2.getTrackbarPos("Padding", WINDOW_NAME)))
        current_edit_zoom = max(1, int(cv2.getTrackbarPos("Edit Zoom", WINDOW_NAME)))
        current_scale_percent = max(MIN_WINDOW_SCALE_PERCENT, int(cv2.getTrackbarPos("Scale %", WINDOW_NAME)))
        current_display_scale = float(current_scale_percent) / 100.0
        if (
            current_padding != int(edit_padding)
            or current_edit_zoom != int(edit_zoom)
            or abs(current_display_scale - float(display_scale)) > 1e-6
        ):
            if current_padding != int(edit_padding):
                edit_padding = int(current_padding)
                invalidate_active_component_caches()
            if current_edit_zoom != int(edit_zoom):
                invalidate_active_panel_cache()
            edit_zoom = int(current_edit_zoom)
            display_scale = float(current_display_scale)
            request_redraw(main=True, edit=True)
        if needs_main_redraw or needs_edit_redraw:
            update_display(refresh_main=needs_main_redraw, refresh_edit=needs_edit_redraw)
        wait_ms = 1 if drawing or needs_main_redraw or needs_edit_redraw else 30
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("1") <= key <= ord("9"):
            choice = key - ord("1")
            if choice < len(component_names):
                active_idx = choice
                _clear_lasso()
                invalidate_active_component_caches()
                request_redraw(main=True, edit=True)
        elif key == ord("["):
            brush_radius = max(1, brush_radius - 1)
            invalidate_main_combined_cache()
            request_redraw(main=True, edit=True)
        elif key == ord("]"):
            brush_radius = min(MAX_BRUSH_RADIUS, brush_radius + 1)
            invalidate_main_combined_cache()
            request_redraw(main=True, edit=True)
        elif key in (ord("v"), ord("V")):
            lasso_mode = not bool(lasso_mode)
            if not lasso_mode:
                _clear_lasso()
            invalidate_main_combined_cache()
            request_redraw(main=True, edit=True)
        elif key in (ord("u"), ord("U"), 8):
            if lasso_points:
                lasso_points.pop()
                request_redraw(main=False, edit=True)
        elif key in (ord("d"), ord("D")):
            if lasso_points:
                _clear_lasso()
                request_redraw(main=False, edit=True)
        elif key in (ord("f"), ord("F")):
            fill_value = 0 if erase_mode else 1
            if _fill_polygon_mask(edit_masks[active_idx], lasso_points, fill_value=fill_value):
                _clear_lasso()
                invalidate_mask_panel_caches(refresh_main=True)
                request_redraw(main=True, edit=True)
            else:
                print("Lasso fill requires at least 3 contour points.")
        elif key in (ord("g"), ord("G")):
            fill_value = 0 if erase_mode else 1
            if _fill_polygon_mask(edit_masks[active_idx], lasso_points, fill_value=fill_value, invert=True):
                _clear_lasso()
                invalidate_mask_panel_caches(refresh_main=True)
                request_redraw(main=True, edit=True)
            else:
                print("Lasso inverse fill requires at least 3 contour points.")
        elif key in (ord("e"), ord("E")):
            if _fill_polygon_mask(edit_masks[active_idx], lasso_points, fill_value=0, invert=True):
                _clear_lasso()
                invalidate_mask_panel_caches(refresh_main=True)
                request_redraw(main=True, edit=True)
            else:
                print("Lasso erase-outside requires at least 3 contour points.")
        elif key == ord("i"):
            _pan_active_patch(0, -PATCH_PAN_STEP)
        elif key == ord("k"):
            _pan_active_patch(0, PATCH_PAN_STEP)
        elif key == ord("j"):
            _pan_active_patch(-PATCH_PAN_STEP, 0)
        elif key == ord("l"):
            _pan_active_patch(PATCH_PAN_STEP, 0)
        elif key == ord("I"):
            _pan_active_patch(0, -PATCH_PAN_STEP_LARGE)
        elif key == ord("K"):
            _pan_active_patch(0, PATCH_PAN_STEP_LARGE)
        elif key == ord("J"):
            _pan_active_patch(-PATCH_PAN_STEP_LARGE, 0)
        elif key == ord("L"):
            _pan_active_patch(PATCH_PAN_STEP_LARGE, 0)
        elif key in (ord("c"), ord("C")):
            _recenter_active_patch()
        elif key in (ord("-"), ord("_")):
            edit_zoom = _adjust_trackbar("Edit Zoom", -1, 1, MAX_EDIT_ZOOM)
            invalidate_active_panel_cache()
            request_redraw(main=False, edit=True)
        elif key in (ord("="), ord("+")):
            edit_zoom = _adjust_trackbar("Edit Zoom", 1, 1, MAX_EDIT_ZOOM)
            invalidate_active_panel_cache()
            request_redraw(main=False, edit=True)
        elif key == ord(","):
            display_scale = float(_adjust_trackbar("Scale %", -10, MIN_WINDOW_SCALE_PERCENT, MAX_WINDOW_SCALE_PERCENT)) / 100.0
            request_redraw(main=True, edit=True)
        elif key == ord("."):
            display_scale = float(_adjust_trackbar("Scale %", 10, MIN_WINDOW_SCALE_PERCENT, MAX_WINDOW_SCALE_PERCENT)) / 100.0
            request_redraw(main=True, edit=True)
        elif key == ord("r"):
            edit_masks = [mask.copy() for mask in original_masks]
            _clear_lasso()
            invalidate_mask_panel_caches(refresh_main=True)
            request_redraw(main=True, edit=True)
        elif key == ord("x"):
            erase_mode = not bool(erase_mode)
            invalidate_main_combined_cache()
            request_redraw(main=True, edit=True)
        elif key == ord("s"):
            save_current()
            original_masks = [mask.copy() for mask in edit_masks]
            source_mask = _source_mask_for_resolved_component(component_sources, component_names[active_idx], current_pos)
            if active_idx < len(patch_anchor_masks):
                patch_anchor_masks[active_idx] = np.logical_or(
                    np.asarray(original_masks[active_idx], dtype=np.uint8) > 0,
                    np.asarray(source_mask, dtype=np.uint8) > 0,
                ).astype(np.uint8)
            _clear_lasso()
            invalidate_active_component_caches()
            request_redraw(main=True, edit=True)
        elif key == ord("n"):
            if current_pos < total_rois - 1:
                current_pos += 1
                load_current_roi()
                request_redraw(main=True, edit=True)
        elif key == ord("p"):
            if current_pos > 0:
                current_pos -= 1
                load_current_roi()
                request_redraw(main=True, edit=True)
        elif key in (ord("a"), ord("N"), ord("R"), ord("P")):
            state = {
                ord("a"): approve_state,
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
                zarr_path=zarr_path,
            )
            print(
                f"Set {component_names[active_idx]} review to {component_payload.get('state')} "
                f"(run={run_payload.get('state')})"
            )
            invalidate_main_combined_cache()
            request_redraw(main=True, edit=False)

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
    parser.add_argument(
        "--approve-state",
        default="approved",
        choices=list(REVIEW_STATE_CHOICES),
        help="Review state applied by the 'a' hotkey (default: approved).",
    )
    parser.add_argument("--review-method", default=DEFAULT_REVIEW_METHOD)
    parser.add_argument("--review-intended-use", default=DEFAULT_REVIEW_INTENDED_USE)
    parser.add_argument("--reviewer", help="Reviewer name to record in review payloads.")
    parser.add_argument("--review-notes", help="Optional note attached to review payload updates.")
    parser.add_argument(
        "--edit-zoom",
        type=int,
        default=DEFAULT_EDIT_ZOOM,
        help="Initial zoom factor for the editor panel (default: %(default)s).",
    )
    parser.add_argument(
        "--edit-padding",
        type=int,
        default=DEFAULT_EDIT_PADDING,
        help="Initial padding around the active-mask crop used for the edit panel (default: %(default)s).",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=DISPLAY_SCALE,
        help="Initial whole-window display scale (default: %(default)s).",
    )
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
        approve_state=args.approve_state,
        review_method=args.review_method,
        review_intended_use=args.review_intended_use,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
        edit_zoom=args.edit_zoom,
        edit_padding=args.edit_padding,
        display_scale=args.display_scale,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
