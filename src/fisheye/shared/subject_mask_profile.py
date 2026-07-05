"""Subject-mask dataset profile builder (parity with detection/keypoint profiles).

Implements the design in
``docs/diagnostics/subject_mask_profile_design_2026-06-18.md``:

1. ``build_subject_mask_profile_summary`` -> ``subject_mask_dataset_profile`` v1
   summary with sections ``dataset`` / ``source`` / ``coverage`` / ``components``
   / ``composition``.
2. ``write_subject_mask_profile`` -> stores the summary under
   ``analysis/subject_mask_profile_runs/<run>/`` with a ``latest`` pointer,
   identical convention to the detection/keypoint profile runs.

Registry projection is NOT done here: the profile is picked up by
``_extract_subject_mask_data_profile_rows`` under
``Registry.reconcile_dataset_from_root`` (no standalone sync script).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import as_float, as_int, normalize_attr
from fisheye.shared.zarr_helpers import infer_zarr_use, zarr_group_names
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
    resolve_authoritative_run_name,
)


SCHEMA_NAME = "subject_mask_dataset_profile"
SCHEMA_VERSION = "v1"

RAW_PARENT_NAME = "subject_mask_runs"
REFINED_PARENT_NAME = "refined_subject_masks_runs"
SUBJECT_MASK_PARENT_NAMES = (REFINED_PARENT_NAME, RAW_PARENT_NAME)

REVIEW_TIMESTAMP_KEYS = ("timestamp_utc", "timestamp", "reviewed_at_utc", "reviewed_at", "updated_utc")
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
PROFILE_PERCENTILES = (10, 50, 90)

_as_float = as_float
_as_int = as_int
_normalize_text = normalize_attr
_utc_now = utc_now


class SubjectMaskProfileError(RuntimeError):
    """Base error for subject-mask profile read/write failures."""


class SubjectMaskSourceError(SubjectMaskProfileError):
    """Raised when no usable subject-mask source can be resolved."""


@dataclass(frozen=True)
class ResolvedSubjectMaskSource:
    mask_path: str
    stage_group: str
    mask_run: str
    subject_mask_method: Optional[str]
    label_schema_id: Optional[str]
    source_keypoints_run: Optional[str]
    source_crop_run: Optional[str]
    run_semantics: Optional[str]
    source_subject_mask_run: Optional[str]
    review_state: Optional[str]
    review_method: Optional[str]
    review_intended_use: Optional[str]
    review_timestamp_utc: Optional[str]
    mask_labels: tuple[str, ...]
    available_flags: tuple[bool, ...]
    eye_component_mode: Optional[str]


@dataclass(frozen=True)
class SubjectMaskProfileWriteResult:
    run_name: str
    source_mask_path: str
    source_subject_mask_method: Optional[str]
    profile_summary: dict[str, Any]


def _default_profile_run_name(created_at_utc: str) -> str:
    try:
        stamp = datetime.fromisoformat(str(created_at_utc))
    except Exception:
        stamp = datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    return f"subject_mask_profile_{stamp.strftime('%Y-%m-%d_%H-%M-%S')}"


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


def _select_source_run(parent: zarr.Group) -> Optional[str]:
    """Resolve the mask source run using authority first, then latest-complete."""

    try:
        return resolve_authoritative_run_name(parent)
    except Exception:
        names = zarr_group_names(parent)
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
        stats.update({"min": None, "max": None, "mean": None, "std": None})
        for percentile in PROFILE_PERCENTILES:
            stats[f"p{percentile}"] = None
        return stats
    stats.update(
        {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    )
    for percentile in PROFILE_PERCENTILES:
        stats[f"p{percentile}"] = float(np.quantile(arr, percentile / 100.0))
    return stats


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray, str)):
        text = _normalize_text(value)
        return [text] if text else []
    if isinstance(value, Sequence):
        values: list[str] = []
        for item in value:
            text = _normalize_text(item)
            if text:
                values.append(text)
        return values
    return []


def _coerce_bool_list(value: Any) -> list[bool]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [bool(item) for item in value]
    return []


def _resolve_review_fields(
    attrs: Mapping[str, Any],
    *,
    stage_group: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    review = None
    if stage_group == REFINED_PARENT_NAME:
        review = _coerce_mapping(attrs.get("refined_subject_mask_review_status"))
    if review is None:
        review = _coerce_mapping(attrs.get("subject_mask_review_status"))
    if not review:
        return None, None, None, None
    timestamp = None
    for key in REVIEW_TIMESTAMP_KEYS:
        timestamp = _normalize_text(review.get(key))
        if timestamp is not None:
            break
    return (
        _normalize_text(review.get("state")) or _normalize_text(review.get("review_state")),
        _normalize_text(review.get("method")) or _normalize_text(review.get("review_method")),
        _normalize_text(review.get("intended_use")) or _normalize_text(review.get("review_intended_use")),
        timestamp,
    )


def _available_flags_for_labels(run_group: zarr.Group, mask_labels: Sequence[str]) -> list[bool]:
    flags: list[bool] = []
    available_channels = run_group.get("available_channels")
    if available_channels is not None:
        try:
            flags = _coerce_bool_list(available_channels[:])
        except Exception:
            flags = []
    if not flags:
        flags = [True] * len(mask_labels)
    if len(flags) < len(mask_labels):
        flags.extend([False] * (len(mask_labels) - len(flags)))
    return flags[: len(mask_labels)]


def _eye_component_mode(mask_labels: Sequence[str]) -> Optional[str]:
    if "eye_left" in mask_labels or "eye_right" in mask_labels:
        return "lr"
    if "eyes_union" in mask_labels:
        return "union"
    return None


def _resolve_run(root: zarr.Group, stage_group: str, run_name: str) -> ResolvedSubjectMaskSource:
    parent = root.get(stage_group)
    if parent is None or run_name not in parent:
        raise SubjectMaskSourceError(f"subject mask run not found: {stage_group}/{run_name}")
    run_group = parent[run_name]
    attrs = dict(run_group.attrs)
    provenance = _coerce_mapping(attrs.get("provenance")) or {}
    provenance_parameters = _coerce_mapping(provenance.get("parameters")) or {}
    provenance_inputs = _coerce_mapping(provenance.get("inputs")) or {}

    method = (
        _normalize_text(attrs.get("method"))
        or _normalize_text(provenance_parameters.get("method"))
        or _normalize_text(provenance.get("method"))
    )
    if not method and stage_group == REFINED_PARENT_NAME:
        method = "refine_subject_masks"

    review_state, review_method, review_intended_use, review_timestamp_utc = _resolve_review_fields(
        attrs,
        stage_group=stage_group,
    )

    mask_labels = _coerce_text_list(attrs.get("mask_labels"))
    available_flags = _available_flags_for_labels(run_group, mask_labels)

    return ResolvedSubjectMaskSource(
        mask_path=f"{stage_group}/{run_name}",
        stage_group=stage_group,
        mask_run=str(run_name),
        subject_mask_method=method,
        label_schema_id=_normalize_text(attrs.get("label_schema_id")),
        source_keypoints_run=(
            _normalize_text(attrs.get("source_keypoints_run"))
            or _normalize_text(attrs.get("source_keypoint_run"))
            or _normalize_text(provenance_inputs.get("source_keypoints_run"))
            or _normalize_text(provenance_inputs.get("source_keypoint_run"))
        ),
        source_crop_run=(
            _normalize_text(attrs.get("source_crop_run"))
            or _normalize_text(provenance_inputs.get("source_crop_run"))
        ),
        run_semantics=(
            _normalize_text(attrs.get("run_semantics"))
            or _normalize_text(provenance_parameters.get("run_semantics"))
        ),
        source_subject_mask_run=(
            _normalize_text(attrs.get("source_subject_mask_run"))
            or _normalize_text(provenance_inputs.get("source_subject_mask_run"))
        ),
        review_state=review_state,
        review_method=review_method,
        review_intended_use=review_intended_use,
        review_timestamp_utc=review_timestamp_utc,
        mask_labels=tuple(mask_labels),
        available_flags=tuple(available_flags),
        eye_component_mode=_eye_component_mode(mask_labels),
    )


def resolve_subject_mask_source(
    root: zarr.Group,
    *,
    source_mask_path: Optional[str] = None,
    refined_run: Optional[str] = None,
    subject_mask_run: Optional[str] = None,
) -> ResolvedSubjectMaskSource:
    """Resolve the subject-mask run to profile.

    Preference order: an explicit path/run wins, then the authoritative refined
    run (falling back to latest-complete when no pointer exists), then the raw
    ``subject_mask_runs`` parent with the same authoritative-first resolution.
    """

    if source_mask_path:
        source_path = str(source_mask_path).strip("/")
        parts = source_path.split("/")
        if len(parts) >= 2 and parts[0] in SUBJECT_MASK_PARENT_NAMES:
            return _resolve_run(root, parts[0], parts[1])
        raise SubjectMaskSourceError(f"unsupported source_mask_path format: {source_path}")

    refined_run = _normalize_text(refined_run)
    subject_mask_run = _normalize_text(subject_mask_run)

    refined_parent = root.get(REFINED_PARENT_NAME)
    if refined_run is not None:
        return _resolve_run(root, REFINED_PARENT_NAME, refined_run)
    if subject_mask_run is not None:
        return _resolve_run(root, RAW_PARENT_NAME, subject_mask_run)
    if refined_parent is not None:
        chosen = _select_source_run(refined_parent)
        if chosen is not None:
            return _resolve_run(root, REFINED_PARENT_NAME, chosen)

    raw_parent = root.get(RAW_PARENT_NAME)
    if raw_parent is None:
        raise SubjectMaskSourceError("no subject mask run parent found")
    chosen = _select_source_run(raw_parent)
    if chosen is None:
        raise SubjectMaskSourceError("no subject mask run available")
    return _resolve_run(root, RAW_PARENT_NAME, chosen)


def _load_dense_masks(run_group: zarr.Group) -> Optional[np.ndarray]:
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        return None
    try:
        values = np.asarray(masks_roi[:])
    except Exception:
        return None
    if values.ndim != 4:
        return None
    return values


def _mask_area_matrix(
    run_group: zarr.Group,
    *,
    n_labels: int,
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Return ``(area_px matrix (N, n_labels), roi_pixels)``.

    ``roi_pixels`` (H*W) is only known when the dense ``masks_roi`` store is
    present; it normalizes component areas into ROI fractions.
    """

    roi_pixels: Optional[int] = None
    dense = _load_dense_masks(run_group)
    if dense is not None:
        roi_pixels = int(dense.shape[2]) * int(dense.shape[3])

    metrics = run_group.get("metrics")
    area_px = metrics.get("area_px") if metrics is not None else None
    if area_px is not None:
        try:
            values = np.asarray(area_px[:], dtype=np.float64)
            if values.ndim == 2 and values.shape[1] >= n_labels:
                return values[:, :n_labels], roi_pixels
        except Exception:
            pass

    if dense is not None and dense.shape[1] >= n_labels:
        binary = dense[:, :n_labels, :, :] > 0
        return binary.sum(axis=(2, 3)).astype(np.float64), roi_pixels
    return None, roi_pixels


def _mask_present_matrix(
    run_group: zarr.Group,
    *,
    n_labels: int,
    areas: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    metrics = run_group.get("metrics")
    mask_present = metrics.get("mask_present") if metrics is not None else None
    if mask_present is not None:
        try:
            values = np.asarray(mask_present[:], dtype=bool)
            if values.ndim == 2 and values.shape[1] >= n_labels:
                return values[:, :n_labels]
        except Exception:
            pass
    if areas is not None:
        return areas > 0.0
    return None


def build_subject_mask_profile_summary(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    source_mask_path: Optional[str] = None,
    refined_run: Optional[str] = None,
    subject_mask_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    """Build a ``subject_mask_dataset_profile`` v1 summary for one mask run.

    Sections: ``dataset`` / ``source`` / ``coverage`` / ``components`` /
    ``composition``. Component area distributions are reported both in pixels
    (``area_px``) and, when the dense mask store reveals the ROI size, as ROI
    fractions (``area_norm``).
    """

    created_at_utc = created_at_utc or _utc_now()
    source = resolve_subject_mask_source(
        root,
        source_mask_path=source_mask_path,
        refined_run=refined_run,
        subject_mask_run=subject_mask_run,
    )
    run_group = root[source.mask_path]
    attrs = dict(run_group.attrs)

    mask_labels = list(source.mask_labels)
    n_labels = len(mask_labels)
    if n_labels == 0:
        raise SubjectMaskProfileError(f"subject mask run has no mask_labels: {source.mask_path}")

    areas, roi_pixels = _mask_area_matrix(run_group, n_labels=n_labels)
    present = _mask_present_matrix(run_group, n_labels=n_labels, areas=areas)

    total_rois = _as_int(attrs.get("total_rois"))
    if total_rois is None and present is not None:
        total_rois = int(present.shape[0])
    if total_rois is None and areas is not None:
        total_rois = int(areas.shape[0])

    available_flags = list(source.available_flags)
    available_labels = [label for label, flag in zip(mask_labels, available_flags) if flag]
    available_indices = [index for index, flag in enumerate(available_flags) if flag]

    rows_with_any_mask: Optional[int] = None
    rows_with_all_available: Optional[int] = None
    if present is not None and available_indices:
        available_present = present[:, available_indices]
        rows_with_any_mask = int(available_present.any(axis=1).sum())
        rows_with_all_available = int(available_present.all(axis=1).sum())
    coverage_percent = None
    any_rate = _safe_ratio(
        float(rows_with_any_mask) if rows_with_any_mask is not None else None,
        float(total_rois) if total_rois is not None else None,
    )
    if any_rate is not None:
        coverage_percent = float(any_rate) * 100.0

    components: dict[str, Any] = {}
    for index, label in enumerate(mask_labels):
        available = bool(available_flags[index]) if index < len(available_flags) else False
        presence_count: Optional[int] = None
        if present is not None:
            presence_count = int(present[:, index].sum())
        presence_rate = _safe_ratio(
            float(presence_count) if presence_count is not None else None,
            float(total_rois) if total_rois is not None else None,
        )
        component: dict[str, Any] = {
            "available": available,
            "presence_count": presence_count,
            "presence_rate": presence_rate,
        }
        if areas is not None and present is not None:
            present_areas = areas[present[:, index], index]
            component["area_px"] = _metric_stats(present_areas)
            if roi_pixels:
                component["area_norm"] = _metric_stats(present_areas / float(roi_pixels))
        components[label] = component

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
            "mask_path": source.mask_path,
            "stage_group": source.stage_group,
            "mask_run": source.mask_run,
            "subject_mask_method": source.subject_mask_method,
            "label_schema_id": source.label_schema_id,
            "source_keypoints_run": source.source_keypoints_run,
            "source_crop_run": source.source_crop_run,
            "run_semantics": source.run_semantics,
            "source_subject_mask_run": source.source_subject_mask_run,
            "review_state": source.review_state,
            "review_method": source.review_method,
            "review_intended_use": source.review_intended_use,
            "review_timestamp_utc": source.review_timestamp_utc,
            "mask_labels": mask_labels,
            "available_components": available_labels,
            "eye_component_mode": source.eye_component_mode,
        },
        "coverage": {
            "total_rois": total_rois,
            "rows_with_any_mask": rows_with_any_mask,
            "coverage_percent": coverage_percent,
            "available_component_count": len(available_labels),
            "rows_with_all_available_components": rows_with_all_available,
            "all_available_components_rate": _safe_ratio(
                float(rows_with_all_available) if rows_with_all_available is not None else None,
                float(total_rois) if total_rois is not None else None,
            ),
            "roi_pixels": roi_pixels,
        },
        "components": components,
    }

    composition = _extract_composition(root)
    if composition:
        summary["composition"] = composition

    return summary


def write_subject_mask_profile(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    source_mask_path: Optional[str] = None,
    refined_run: Optional[str] = None,
    subject_mask_run: Optional[str] = None,
    dataset_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_use: Optional[str] = None,
    created_at_utc: Optional[str] = None,
) -> SubjectMaskProfileWriteResult:
    """Write a profile run under ``analysis/subject_mask_profile_runs/<run>``.

    Identical zarr convention to the detection/keypoint profile writers:
    ``profile_summary`` stored as a run-group attr, ``latest`` pointer on the
    parent, run completion markers on the run group.
    """

    summary = build_subject_mask_profile_summary(
        root,
        zarr_path=zarr_path,
        source_mask_path=source_mask_path,
        refined_run=refined_run,
        subject_mask_run=subject_mask_run,
        dataset_id=dataset_id,
        recording_id=recording_id,
        zarr_use=zarr_use,
        created_at_utc=created_at_utc,
    )
    created = _normalize_text(summary.get("created_at_utc")) or _utc_now()
    run_name = _normalize_text(run_name) or _default_profile_run_name(created)

    analysis_group = root.require_group("analysis")
    runs_parent = require_runs_parent(analysis_group, "subject_mask_profile_runs")
    if run_name in runs_parent:
        if not overwrite:
            raise FileExistsError(f"analysis/subject_mask_profile_runs/{run_name} already exists")
        del runs_parent[run_name]
    run_group = runs_parent.create_group(run_name)
    mark_run_started(run_group, run_name=run_name, stage="subject_mask_profile")

    dataset = summary.get("dataset", {})
    source = summary.get("source", {})
    coverage = summary.get("coverage", {})

    run_group.attrs.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": created,
            "source_dataset_id": dataset.get("dataset_id"),
            "source_recording_id": dataset.get("recording_id"),
            "source_zarr_use": dataset.get("zarr_use"),
            "source_mask_path": source.get("mask_path"),
            "source_stage_group": source.get("stage_group"),
            "source_mask_run": source.get("mask_run"),
            "source_subject_mask_method": source.get("subject_mask_method"),
            "source_label_schema_id": source.get("label_schema_id"),
            "source_keypoints_run": source.get("source_keypoints_run"),
            "source_crop_run": source.get("source_crop_run"),
            "source_run_semantics": source.get("run_semantics"),
            "source_row_count": _as_int(coverage.get("total_rois")),
            "profile_summary": summary,
        }
    )
    runs_parent.attrs["latest"] = run_name
    mark_run_complete(run_group, parent_group=runs_parent, run_name=run_name)

    return SubjectMaskProfileWriteResult(
        run_name=run_name,
        source_mask_path=str(source.get("mask_path") or ""),
        source_subject_mask_method=_normalize_text(source.get("subject_mask_method")),
        profile_summary=summary,
    )


__all__ = [
    "ResolvedSubjectMaskSource",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SubjectMaskProfileError",
    "SubjectMaskProfileWriteResult",
    "SubjectMaskSourceError",
    "build_subject_mask_profile_summary",
    "resolve_subject_mask_source",
    "write_subject_mask_profile",
]
