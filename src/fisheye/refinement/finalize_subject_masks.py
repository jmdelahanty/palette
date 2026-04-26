"""Create refined subject-mask candidates from raw subject-mask outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import dask
from dask import delayed
import numpy as np
import zarr

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:  # pragma: no cover - depends on optional dependency
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    HAVE_DISTRIBUTED = False

from ..shared.detect_reason_codec import write_reason_columns
from ..shared.provenance_attrs import (
    ASSIGNMENT_KEYPOINT_CONTRACT_VALUE,
    build_assignment_keypoint_attrs,
    build_source_keypoints_attrs,
)
from ..shared.row_lineage import copy_row_lineage_arrays_from_sources
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import (
    refined_subject_mask_metric_row_chunk,
    refined_subject_mask_storage_chunks,
)
from ..shared.subject_mask_registry_status import emit_refined_subject_mask_stage_completion
from ..tune.refined_subject_mask_review import (
    DEFAULT_REVIEW_INTENDED_USE,
    DEFAULT_REVIEW_METHOD,
    DEFAULT_SUBJECT_BODY_ANATOMICAL_SCOPE,
    DEFAULT_SUBJECT_BODY_COMPONENT_SCHEMA_ID,
    DEFAULT_SUBJECT_BODY_PECTORAL_FIN_POLICY,
    REFINED_SUBJECT_SOURCE_SYNC_SCHEMA_ID,
    REFINED_SUBJECT_STAGE_NAME,
    SourceSubjectMaskRun,
    _compute_component_curvature_var_metrics,
    _compute_component_shape_qc_metrics,
    _compute_component_sigma_noise_metrics,
    _compute_component_topology_metrics,
    _compute_geometry_metrics,
    _compute_mask_metrics,
    _compute_mask_row_fingerprints,
    _default_refined_run_name,
    _ensure_refined_component_provenance_payload,
    _infer_refined_label_schema_id,
    _load_source_subject_mask_run,
    _normalize_component_name,
    _probability_encoding_for_group,
    _probability_thresholds_for_labels,
    _review_payload,
    _source_component_provenance_payload,
)
from ..utils.system import get_environment_info, get_git_info
from ..utils.zarr_io import open_zarr_root
from .assemble_refined_subject_masks import (
    CANONICAL_COMPONENT_ORDER,
    _has_available_component,
    _require_available_component,
    _resolve_eye_keypoint_indices,
    _resolve_keypoint_success_array,
    _resolve_subject_keypoint_group,
)
from ..shared.refined_subject_eye_geometry import write_refined_subject_eye_geometry
from .subject_eye_assignment import EYES_UNION_ASSIGNMENT_METHOD, assign_eyes_union_to_lr
from .subject_mask_finalization import (
    ComponentFinalizationPolicy,
    _default_policy_for_component,
    finalize_component_mask,
)

SMART_FINALIZE_SUBJECT_MASKS_METHOD = "smart_finalize_subject_masks_v1"
_REFINED_SUBJECT_MASKS_STATUS_SOURCE = "runtime_smart_finalize_subject_masks"
_RAW_EYE_UNION_COMPONENT = "eyes_union"
_EYE_COMPONENTS = ("eye_left", "eye_right")
_FINALIZABLE_RAW_COMPONENTS = ("subject_body", "swim_bladder", _RAW_EYE_UNION_COMPONENT)
_METRIC_LEVELS = ("cheap", "full")
_SUPPORTED_SCHEDULERS = ("single-threaded", "threads", "processes", "distributed")
_EXECUTION_BACKENDS = ("serial_driver", "dask_worker_chunks")
_SERIAL_EXECUTION_BACKEND = "serial_driver"
_DASK_WORKER_EXECUTION_BACKEND = "dask_worker_chunks"
_FINALIZATION_METRIC_NAMES = (
    "added_area_px",
    "area_px_after",
    "area_px_before",
    "changed_area_fraction",
    "changed_area_px",
    "component_count_after",
    "component_count_before",
    "hole_area_fraction_after",
    "hole_area_fraction_before",
    "hole_count_after",
    "hole_count_before",
    "largest_component_fraction_after",
    "largest_component_fraction_before",
    "removed_area_fraction",
    "removed_area_px",
    "removed_component_count",
    "removed_high_prob_area_px",
    "removed_prob_mass",
    "removed_prob_mass_fraction",
)


@dataclass(frozen=True)
class _FinalizedComponentBatch:
    component_name: str
    masks: np.ndarray
    source_masks: np.ndarray
    reason_labels: np.ndarray
    quality_code: np.ndarray
    quality_score: np.ndarray
    review_recommendation: np.ndarray
    metrics: dict[str, np.ndarray]
    policy: ComponentFinalizationPolicy
    source_surface_path: str
    source_surface_kind: str
    source_probability_encoding: Optional[str]
    source_probability_threshold: float


@dataclass(frozen=True)
class _EyeAssignmentContext:
    keypoints_roi: Any
    keypoint_success: np.ndarray
    eye_keypoint_indices: tuple[int, int]
    keypoint_run_name: str
    keypoint_group_name: str
    keypoint_success_dataset: str
    keypoint_source_kind: str


@dataclass(frozen=True)
class _EyeAssignmentChunk:
    masks: dict[str, np.ndarray]
    reason_labels: dict[str, np.ndarray]
    summary: dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_scheduler(scheduler: object) -> str:
    scheduler_key = str(scheduler or "single-threaded").strip().lower()
    if scheduler_key in {"single-thread", "single_thread"}:
        scheduler_key = "single-threaded"
    if scheduler_key not in _SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler {scheduler!r}; expected one of {', '.join(_SUPPORTED_SCHEDULERS)}."
        )
    return scheduler_key


def _normalize_execution_backend(execution_backend: object) -> str:
    backend = str(execution_backend or _SERIAL_EXECUTION_BACKEND).strip().lower()
    if backend not in _EXECUTION_BACKENDS:
        raise ValueError(
            f"Unsupported execution_backend {execution_backend!r}; expected one of "
            f"{', '.join(_EXECUTION_BACKENDS)}."
        )
    return backend


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class _TimingRecorder:
    phase_seconds: dict[str, float] = field(default_factory=dict)
    phase_counts: dict[str, int] = field(default_factory=dict)
    chunk_timings: list[dict[str, object]] = field(default_factory=list)

    def add(self, phase: str, seconds: float) -> float:
        key = str(phase)
        elapsed = float(seconds)
        self.phase_seconds[key] = float(self.phase_seconds.get(key, 0.0) + elapsed)
        self.phase_counts[key] = int(self.phase_counts.get(key, 0) + 1)
        return elapsed

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, time.perf_counter() - start)

    def summary(self, *, total_rows: int, duration_seconds: float) -> dict[str, object]:
        total = float(duration_seconds)
        rows = int(total_rows)
        phase_seconds = {
            str(key): float(value)
            for key, value in sorted(self.phase_seconds.items())
        }
        return {
            "duration_seconds": total,
            "rows_total": rows,
            "rows_per_second": float(rows / total) if total > 0 else 0.0,
            "phase_seconds": phase_seconds,
            "phase_counts": {
                str(key): int(value)
                for key, value in sorted(self.phase_counts.items())
            },
            "chunk_count": int(len(self.chunk_timings)),
            "chunk_seconds_total": float(sum(float(item.get("total_seconds") or 0.0) for item in self.chunk_timings)),
            "slowest_chunks": sorted(
                (
                    {
                        "chunk_index": int(item.get("chunk_index") or 0),
                        "start_row": int(item.get("start_row") or 0),
                        "stop_row": int(item.get("stop_row") or 0),
                        "total_seconds": float(item.get("total_seconds") or 0.0),
                    }
                    for item in self.chunk_timings
                ),
                key=lambda item: float(item["total_seconds"]),
                reverse=True,
            )[:5],
        }


def _decode_probabilities(values: np.ndarray, *, encoding: Optional[str]) -> np.ndarray:
    arr = np.asarray(values)
    normalized_encoding = str(encoding or "").strip().lower()
    if arr.dtype == np.uint8 and normalized_encoding in {"", "linear_uint8_0_255"}:
        return arr.astype(np.float32) / np.float32(255.0)
    return arr.astype(np.float32, copy=False)


def _join_reason_tags(tags: Sequence[object], *, probability_source: bool) -> str:
    merged: list[str] = []
    if probability_source:
        merged.append("cleanup_thresholded_probability")
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if not tag:
            continue
        if probability_source and tag == "clean":
            continue
        if tag not in merged:
            merged.append(tag)
    if not merged:
        merged.append("clean")
    return "|".join(merged)


def _combine_reason_labels(*labels: object) -> str:
    merged: list[str] = []
    for label in labels:
        for raw_tag in str(label or "").split("|"):
            tag = raw_tag.strip()
            if not tag or tag == "clean":
                continue
            if tag not in merged:
                merged.append(tag)
    return "|".join(merged) if merged else "clean"


def _policy_payload(policy: ComponentFinalizationPolicy) -> dict[str, object]:
    return {str(key): _json_safe(value) for key, value in asdict(policy).items()}


def _component_threshold(source: SourceSubjectMaskRun, component_name: str, component_idx: int) -> float:
    if component_idx < len(source.probability_thresholds):
        return float(source.probability_thresholds[component_idx])
    labels = tuple(str(label) for label in source.mask_labels)
    thresholds = _probability_thresholds_for_labels(source.group, labels)
    if component_idx < len(thresholds):
        return float(thresholds[component_idx])
    return 0.5


def _component_surface_rows(
    source: SourceSubjectMaskRun,
    component_name: str,
    *,
    start_row: int,
    stop_row: int,
) -> tuple[np.ndarray, bool, str, Optional[str], float, int]:
    component_idx = _require_available_component(source, component_name, "subject_mask_runs")
    probabilities = source.group.get("mask_probs_roi")
    threshold = _component_threshold(source, component_name, component_idx)
    if probabilities is not None:
        encoding = source.probability_encoding or _probability_encoding_for_group(source.group)
        raw = np.asarray(probabilities[int(start_row) : int(stop_row), component_idx])
        return (
            _decode_probabilities(raw, encoding=encoding),
            True,
            "mask_probs_roi",
            encoding,
            threshold,
            component_idx,
        )

    masks = source.group.get("masks_roi")
    if masks is None:
        raise RuntimeError(f"subject_mask_runs/{source.run_name} missing masks_roi or mask_probs_roi.")
    return (
        np.asarray(masks[int(start_row) : int(stop_row), component_idx], dtype=np.uint8),
        False,
        "masks_roi",
        None,
        threshold,
        component_idx,
    )


def _component_surface_batch(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> tuple[np.ndarray, bool, str, Optional[str], float, int]:
    return _component_surface_rows(
        source,
        component_name,
        start_row=0,
        stop_row=int(source.masks_roi.shape[0]),
    )


def _finalize_source_component(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> _FinalizedComponentBatch:
    return _finalize_source_component_rows(
        source,
        component_name,
        start_row=0,
        stop_row=int(source.masks_roi.shape[0]),
    )


def _finalize_source_component_rows(
    source: SourceSubjectMaskRun,
    component_name: str,
    *,
    start_row: int,
    stop_row: int,
) -> _FinalizedComponentBatch:
    surfaces, is_probability, surface_path, encoding, threshold, _component_idx = _component_surface_rows(
        source,
        component_name,
        start_row=start_row,
        stop_row=stop_row,
    )
    if surfaces.ndim != 3:
        raise ValueError(
            f"subject_mask_runs/{source.run_name}/{surface_path} component {component_name!r} "
            f"must have shape (N,H,W), got {tuple(surfaces.shape)}."
        )
    base_policy = _default_policy_for_component(component_name)
    policy = replace(base_policy, threshold=float(threshold))

    total_rows = int(surfaces.shape[0])
    masks = np.zeros(surfaces.shape, dtype=np.uint8)
    source_masks = np.zeros(surfaces.shape, dtype=np.uint8)
    reason_labels = np.full((total_rows,), "clean", dtype=object)
    quality_code = np.zeros((total_rows,), dtype=np.int16)
    quality_score = np.zeros((total_rows,), dtype=np.float32)
    review_recommendation = np.full((total_rows,), "pending", dtype=object)
    metric_values: dict[str, list[float]] = {}

    for row_idx in range(total_rows):
        result = finalize_component_mask(
            component_name,
            surfaces[row_idx],
            policy=policy,
            surface_is_probability=is_probability,
        )
        masks[row_idx] = np.asarray(result.mask, dtype=np.uint8)
        source_masks[row_idx] = np.asarray(result.source_mask, dtype=np.uint8)
        reason_labels[row_idx] = _join_reason_tags(
            result.reason_tags,
            probability_source=bool(is_probability),
        )
        quality_code[row_idx] = np.int16(result.quality_code)
        quality_score[row_idx] = np.float32(result.quality_score)
        review_recommendation[row_idx] = str(result.review_recommendation)
        for name, value in result.metrics.items():
            metric_values.setdefault(str(name), []).append(float(value))

    metrics = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in sorted(metric_values.items())
    }
    return _FinalizedComponentBatch(
        component_name=component_name,
        masks=masks,
        source_masks=source_masks,
        reason_labels=reason_labels,
        quality_code=quality_code,
        quality_score=quality_score,
        review_recommendation=review_recommendation,
        metrics=metrics,
        policy=policy,
        source_surface_path=surface_path,
        source_surface_kind="probability" if is_probability else "binary",
        source_probability_encoding=encoding,
        source_probability_threshold=float(threshold),
    )


def _source_payload_for_finalized_component(
    source: SourceSubjectMaskRun,
    batch: _FinalizedComponentBatch,
) -> dict[str, object]:
    payload = _source_component_provenance_payload(source, batch.component_name)
    payload["finalization_method"] = SMART_FINALIZE_SUBJECT_MASKS_METHOD
    payload["finalization_policy"] = _policy_payload(batch.policy)
    payload["source_surface_path"] = (
        f"subject_mask_runs/{source.run_name}/{batch.source_surface_path}"
    )
    payload["source_surface_kind"] = batch.source_surface_kind
    if batch.source_surface_kind == "probability":
        payload["source_probability_path"] = (
            f"subject_mask_runs/{source.run_name}/{batch.source_surface_path}"
        )
        payload["source_probability_encoding"] = str(batch.source_probability_encoding or "")
        payload["source_probability_threshold"] = float(batch.source_probability_threshold)
        payload["source_binary_derivation"] = "smart_finalize(mask_probs_roi)"
    else:
        payload["source_binary_derivation"] = "smart_finalize(masks_roi)"
    return payload


def _source_payload_for_assigned_eye_component(
    source: SourceSubjectMaskRun,
    *,
    component_name: str,
    union_batch: _FinalizedComponentBatch,
    assignment_summary: Mapping[str, object],
    keypoint_run_name: str,
    keypoint_group_name: str,
    keypoint_success_dataset: str,
    keypoint_source_kind: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_stage": "subject_mask_runs",
        "source_run": source.run_name,
        "source_method": source.source_method or source.group.attrs.get("method") or "unknown",
        "source_channels": [_RAW_EYE_UNION_COMPONENT],
        "source_crop_run": source.crop_run,
        "derived_component": str(component_name),
        "derived_from_component": _RAW_EYE_UNION_COMPONENT,
        "assignment_method": EYES_UNION_ASSIGNMENT_METHOD,
        "assignment_summary": dict(_json_safe(dict(assignment_summary))),
        "assignment_keypoint_run": str(keypoint_run_name),
        "assignment_keypoint_group": str(keypoint_group_name),
        "assignment_keypoint_success_dataset": str(keypoint_success_dataset),
        "assignment_keypoint_source_kind": str(keypoint_source_kind),
        "finalization_method": SMART_FINALIZE_SUBJECT_MASKS_METHOD,
        "finalization_policy": _policy_payload(union_batch.policy),
        "source_surface_path": f"subject_mask_runs/{source.run_name}/{union_batch.source_surface_path}",
        "source_surface_kind": union_batch.source_surface_kind,
        **source.source_crop_snapshot,
    }
    label_schema_id = source.group.attrs.get("label_schema_id")
    if label_schema_id is not None:
        payload["source_label_schema_id"] = str(label_schema_id)
    created_at = source.group.attrs.get("created_at_utc") or source.group.attrs.get("created_utc")
    if created_at is not None:
        payload["source_created_at_utc"] = str(created_at)
    if union_batch.source_surface_kind == "probability":
        payload["source_probability_path"] = (
            f"subject_mask_runs/{source.run_name}/{union_batch.source_surface_path}"
        )
        payload["source_probability_encoding"] = str(union_batch.source_probability_encoding or "")
        payload["source_probability_threshold"] = float(union_batch.source_probability_threshold)
        payload["source_binary_derivation"] = "smart_finalize(mask_probs_roi)"
    else:
        payload["source_binary_derivation"] = "smart_finalize(masks_roi)"
    return payload


def _resolve_eye_assignment_context(
    root: zarr.Group,
    source: SourceSubjectMaskRun,
    *,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> _EyeAssignmentContext:
    kp_group, keypoint_run_name, keypoint_group_name, keypoint_source_kind = _resolve_subject_keypoint_group(
        root,
        source,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    keypoints_roi = kp_group.get("keypoints_roi")
    if keypoints_roi is None:
        raise ValueError(f"Keypoint run {keypoint_run_name!r} missing keypoints_roi; cannot assign eyes_union.")
    keypoint_success, success_dataset = _resolve_keypoint_success_array(kp_group, keypoint_run_name)
    eye_keypoint_indices = _resolve_eye_keypoint_indices(kp_group, keypoint_run_name)
    return _EyeAssignmentContext(
        keypoints_roi=keypoints_roi,
        keypoint_success=np.asarray(keypoint_success, dtype=bool),
        eye_keypoint_indices=eye_keypoint_indices,
        keypoint_run_name=str(keypoint_run_name),
        keypoint_group_name=str(keypoint_group_name),
        keypoint_success_dataset=str(success_dataset),
        keypoint_source_kind=str(keypoint_source_kind),
    )


def _assign_finalized_eyes_union_rows(
    source: SourceSubjectMaskRun,
    union_batch: _FinalizedComponentBatch,
    context: _EyeAssignmentContext,
    *,
    start_row: int,
    stop_row: int,
) -> _EyeAssignmentChunk:
    assignment = assign_eyes_union_to_lr(
        np.asarray(union_batch.masks, dtype=np.uint8),
        keypoints_roi=np.asarray(context.keypoints_roi[int(start_row) : int(stop_row)], dtype=np.float32),
        keypoint_success=np.asarray(context.keypoint_success[int(start_row) : int(stop_row)], dtype=bool),
        eye_keypoint_indices=context.eye_keypoint_indices,
    )
    summary = dict(assignment.summary)
    summary["keypoint_run"] = context.keypoint_run_name
    summary["keypoint_group"] = context.keypoint_group_name
    summary["keypoint_success_dataset"] = context.keypoint_success_dataset
    summary["keypoint_source_kind"] = context.keypoint_source_kind
    summary["assignment_keypoint_contract"] = ASSIGNMENT_KEYPOINT_CONTRACT_VALUE

    reason_labels_by_component: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for component_name in _EYE_COMPONENTS:
        assignment_reasons = np.asarray(assignment.reason_labels[component_name], dtype=object)
        reason_labels = np.asarray(
            [
                _combine_reason_labels(union_batch.reason_labels[row_idx], assignment_reasons[row_idx])
                for row_idx in range(int(union_batch.reason_labels.shape[0]))
            ],
            dtype=object,
        )
        reason_labels_by_component[component_name] = reason_labels
        masks[component_name] = np.asarray(assignment.masks[component_name], dtype=np.uint8)
    return _EyeAssignmentChunk(masks=masks, reason_labels=reason_labels_by_component, summary=summary)


def _merge_assignment_summary(target: dict[str, object], chunk_summary: Mapping[str, object]) -> dict[str, object]:
    if not target:
        target.update(
            {
                "assignment_method": EYES_UNION_ASSIGNMENT_METHOD,
                "total_rows": 0,
                "assigned_rows": 0,
                "assigned_needs_review_rows": 0,
                "failed_rows": 0,
                "status_counts": {},
                "reason_counts": {},
                "keypoint_run": chunk_summary.get("keypoint_run"),
                "keypoint_group": chunk_summary.get("keypoint_group"),
                "keypoint_success_dataset": chunk_summary.get("keypoint_success_dataset"),
                "keypoint_source_kind": chunk_summary.get("keypoint_source_kind"),
                "assignment_keypoint_contract": ASSIGNMENT_KEYPOINT_CONTRACT_VALUE,
            }
        )
    for key in ("total_rows", "assigned_rows", "assigned_needs_review_rows", "failed_rows"):
        target[key] = int(target.get(key) or 0) + int(chunk_summary.get(key) or 0)
    for key in ("status_counts", "reason_counts"):
        merged = dict(target.get(key) or {})
        for item_key, item_value in dict(chunk_summary.get(key) or {}).items():
            merged[str(item_key)] = int(merged.get(str(item_key), 0)) + int(item_value)
        target[key] = merged
    return target


def _requested_output_components(
    source: SourceSubjectMaskRun,
    components: Optional[Sequence[str]],
) -> tuple[str, ...]:
    if not components:
        output: list[str] = []
        if _has_available_component(source, "subject_body"):
            output.append("subject_body")
        if _has_available_component(source, _RAW_EYE_UNION_COMPONENT):
            output.extend(_EYE_COMPONENTS)
        if _has_available_component(source, "swim_bladder"):
            output.append("swim_bladder")
        return tuple(component for component in CANONICAL_COMPONENT_ORDER if component in output)

    normalized = []
    seen: set[str] = set()
    for raw_component in components:
        component = _normalize_component_name(raw_component)
        if component is None:
            continue
        if component == _RAW_EYE_UNION_COMPONENT:
            for eye_component in _EYE_COMPONENTS:
                if eye_component not in seen:
                    seen.add(eye_component)
                    normalized.append(eye_component)
            continue
        if component not in seen:
            seen.add(component)
            normalized.append(component)
    if "eye_left" in seen or "eye_right" in seen:
        if not _has_available_component(source, _RAW_EYE_UNION_COMPONENT):
            raise ValueError("Requested eye_left/eye_right finalization requires an available eyes_union source channel.")
        if not set(_EYE_COMPONENTS).issubset(seen):
            raise ValueError("Request both eye_left and eye_right, or request eyes_union, for smart eye finalization.")
    invalid = [component for component in normalized if component not in CANONICAL_COMPONENT_ORDER]
    if invalid:
        raise ValueError(f"Unsupported smart-finalizer component(s): {invalid}.")
    return tuple(component for component in CANONICAL_COMPONENT_ORDER if component in normalized)


def _review_counts_from_labels(labels: np.ndarray) -> dict[str, int]:
    needs_review = 0
    pending = 0
    for label in np.asarray(labels, dtype=object).reshape(-1):
        if "needs_review" in str(label):
            needs_review += 1
        else:
            pending += 1
    return {"pending": int(pending), "needs_review": int(needs_review)}


def _add_review_counts(target: dict[str, dict[str, int]], component_name: str, labels: np.ndarray) -> None:
    counts = _review_counts_from_labels(labels)
    existing = target.setdefault(str(component_name), {"pending": 0, "needs_review": 0})
    for key, value in counts.items():
        existing[str(key)] = int(existing.get(str(key), 0)) + int(value)


def _merge_review_counts(target: dict[str, dict[str, int]], source: Mapping[str, object]) -> None:
    for component_name, counts_raw in dict(source).items():
        existing = target.setdefault(str(component_name), {"pending": 0, "needs_review": 0})
        for key, value in dict(counts_raw or {}).items():
            existing[str(key)] = int(existing.get(str(key), 0)) + int(value)


def _row_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    total = max(0, int(total_rows))
    size = max(1, int(chunk_size))
    return [
        (start, min(total, start + size))
        for start in range(0, total, size)
    ]


def _create_filled_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    fill_value: object = 0,
) -> Any:
    return group.create_array(
        name,
        shape=tuple(int(dim) for dim in shape),
        dtype=np.dtype(dtype),
        chunks=tuple(int(dim) for dim in chunks),
        fill_value=fill_value,
        overwrite=True,
    )


def _metric_chunks_2d(total_rows: int) -> tuple[int, int]:
    return (refined_subject_mask_metric_row_chunk(total_rows), 1)


def _metric_chunks_lastdim(total_rows: int, width: int) -> tuple[int, int, int]:
    return (refined_subject_mask_metric_row_chunk(total_rows), 1, int(width))


def _source_lineage_map(source: SourceSubjectMaskRun) -> dict[str, object | None]:
    return {
        "frame_indices": source.frame_indices,
        "frame_counts": source.frame_counts,
        "detection_indices": source.detection_indices,
        "source_refined_row_ids": source.source_refined_row_ids,
        "source_detect_row_index": source.source_detect_row_index,
    }


def _create_component_shell(
    run_group: zarr.Group,
    *,
    component_name: str,
    total_rows: int,
    height: int,
    width: int,
) -> zarr.Group:
    component_group = run_group.require_group("components").require_group(component_name)
    metric_chunks = (refined_subject_mask_metric_row_chunk(total_rows),)
    _create_filled_array(
        component_group,
        "mask_present",
        shape=(total_rows,),
        dtype=bool,
        chunks=metric_chunks,
    )
    _create_filled_array(
        component_group,
        "area_px",
        shape=(total_rows,),
        dtype=np.float32,
        chunks=metric_chunks,
    )
    _create_filled_array(
        component_group,
        "edit_applied",
        shape=(total_rows,),
        dtype=bool,
        chunks=metric_chunks,
    )
    component_group.attrs["source_sync_schema_id"] = REFINED_SUBJECT_SOURCE_SYNC_SCHEMA_ID
    _create_filled_array(
        component_group,
        "source_row_fingerprint",
        shape=(total_rows,),
        dtype=np.uint64,
        chunks=metric_chunks,
    )
    _create_filled_array(
        component_group,
        "manual_override",
        shape=(total_rows,),
        dtype=bool,
        chunks=metric_chunks,
    )
    _create_filled_array(
        component_group,
        "source_row_stale",
        shape=(total_rows,),
        dtype=bool,
        chunks=metric_chunks,
    )
    _create_filled_array(
        component_group,
        "source_seed_masks_roi",
        shape=(total_rows, height, width),
        dtype=np.uint8,
        chunks=(refined_subject_mask_metric_row_chunk(total_rows), int(height), int(width)),
    )
    component_group.attrs["source_seed_masks_schema_id"] = "refined_subject_component_source_seed_masks_v1"
    if component_name == "subject_body":
        component_group.attrs["component_schema_id"] = DEFAULT_SUBJECT_BODY_COMPONENT_SCHEMA_ID
        component_group.attrs["anatomical_scope"] = DEFAULT_SUBJECT_BODY_ANATOMICAL_SCOPE
        component_group.attrs["pectoral_fin_policy"] = DEFAULT_SUBJECT_BODY_PECTORAL_FIN_POLICY

    metrics_group = component_group.require_group("metrics")
    for metric_name, dtype in (
        ("component_count", np.int32),
        ("largest_component_fraction", np.float32),
        ("hole_count", np.int32),
        ("hole_area_fraction", np.float32),
    ):
        _create_filled_array(
            metrics_group,
            metric_name,
            shape=(total_rows,),
            dtype=dtype,
            chunks=metric_chunks,
        )
    for metric_name, dtype, fill_value in (
        ("sigma_noise", np.float32, np.nan),
        ("curvature_var", np.float32, np.nan),
        ("ipr", np.float32, np.nan),
        ("solidity", np.float32, np.nan),
    ):
        _create_filled_array(
            metrics_group,
            metric_name,
            shape=(total_rows,),
            dtype=dtype,
            chunks=metric_chunks,
            fill_value=fill_value,
        )
    return component_group


def _create_finalization_metric_shell(
    component_group: zarr.Group,
    *,
    batch: _FinalizedComponentBatch | None = None,
    metric_names: Sequence[str] = (),
    total_rows: int,
) -> None:
    metrics_group = component_group.require_group("finalization_metrics")
    metric_chunks = (refined_subject_mask_metric_row_chunk(total_rows),)
    names_source = metric_names or (batch.metrics if batch is not None else ())
    names = sorted(str(name) for name in names_source)
    for metric_name in names:
        if metric_name not in metrics_group:
            _create_filled_array(
                metrics_group,
                metric_name,
                shape=(total_rows,),
                dtype=np.float32,
                chunks=metric_chunks,
            )
    if "quality_code" not in metrics_group:
        _create_filled_array(
            metrics_group,
            "quality_code",
            shape=(total_rows,),
            dtype=np.int16,
            chunks=metric_chunks,
        )
    if "quality_score" not in metrics_group:
        _create_filled_array(
            metrics_group,
            "quality_score",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=metric_chunks,
        )
    metrics_group.attrs["schema_id"] = "refined_subject_component_finalization_metrics_v1"
    metrics_group.attrs["method"] = SMART_FINALIZE_SUBJECT_MASKS_METHOD
    if batch is not None:
        metrics_group.attrs["source_component"] = str(batch.component_name)
        metrics_group.attrs["source_surface_path"] = str(batch.source_surface_path)
        metrics_group.attrs["source_surface_kind"] = str(batch.source_surface_kind)


def _create_refined_run_shell(
    *,
    refined_parent: zarr.Group,
    target_run: str,
    source: SourceSubjectMaskRun,
    component_names: Sequence[str],
    extra_attrs: Mapping[str, object],
    provenance_inputs: Mapping[str, object],
    stage_command: Optional[str],
) -> zarr.Group:
    total_rows = int(source.masks_roi.shape[0])
    height = int(source.masks_roi.shape[2])
    width = int(source.masks_roi.shape[3])
    component_names = tuple(str(name) for name in component_names)
    created = _utc_now()

    run_group = refined_parent.create_group(target_run)
    run_group.create_array(
        "detection_source",
        data=np.asarray(source.detection_source[:], dtype=np.int8),
        chunks=(refined_subject_mask_metric_row_chunk(total_rows),),
        overwrite=True,
    )
    _create_filled_array(
        run_group,
        "masks_roi",
        shape=(total_rows, len(component_names), height, width),
        dtype=np.uint8,
        chunks=refined_subject_mask_storage_chunks(total_rows, height, width),
    )
    run_group.create_array(
        "available_channels",
        data=np.ones((len(component_names),), dtype=bool),
        overwrite=True,
    )
    _create_filled_array(
        run_group,
        "edit_applied",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_metric_chunks_2d(total_rows),
    )
    copy_row_lineage_arrays_from_sources(run_group, _source_lineage_map(source), total_rois=total_rows)

    metrics_group = run_group.require_group("metrics")
    _create_filled_array(
        metrics_group,
        "mask_present",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_metric_chunks_2d(total_rows),
    )
    _create_filled_array(
        metrics_group,
        "area_px",
        shape=(total_rows, len(component_names)),
        dtype=np.float32,
        chunks=_metric_chunks_2d(total_rows),
    )
    _create_filled_array(
        metrics_group,
        "centroid_xy",
        shape=(total_rows, len(component_names), 2),
        dtype=np.float32,
        chunks=_metric_chunks_lastdim(total_rows, 2),
        fill_value=np.nan,
    )
    _create_filled_array(
        metrics_group,
        "centroid_valid",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_metric_chunks_2d(total_rows),
    )
    _create_filled_array(
        metrics_group,
        "bbox_xyxy",
        shape=(total_rows, len(component_names), 4),
        dtype=np.float32,
        chunks=_metric_chunks_lastdim(total_rows, 4),
    )
    _create_filled_array(
        metrics_group,
        "bbox_valid",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_metric_chunks_2d(total_rows),
    )

    for component_name in component_names:
        _create_component_shell(
            run_group,
            component_name=component_name,
            total_rows=total_rows,
            height=height,
            width=width,
        )

    if source.run_name:
        run_group.attrs["source_subject_mask_run"] = source.run_name
    if source.source_method:
        run_group.attrs["source_subject_mask_method"] = source.source_method
    if source.source_keypoints_run:
        run_group.attrs.update(build_source_keypoints_attrs(source.source_keypoints_run, include_legacy_alias=True))
    if source.source_keypoint_group:
        run_group.attrs["source_keypoint_group"] = source.source_keypoint_group
    run_group.attrs["source_crop_run"] = source.crop_run
    run_group.attrs.update(source.source_crop_snapshot)
    run_group.attrs["mask_labels"] = list(component_names)
    run_group.attrs["label_schema_id"] = _infer_refined_label_schema_id(component_names)
    run_group.attrs["output_semantics"] = "multilabel"
    run_group.attrs["refinement_semantics"] = "canonical_component_masks"
    run_group.attrs["method"] = SMART_FINALIZE_SUBJECT_MASKS_METHOD
    run_group.attrs["created_at_utc"] = created
    run_group.attrs["created_utc"] = created
    run_group.attrs["duration_seconds"] = 0.0
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
        "source_subject_mask_run": source.run_name,
        "source_subject_mask_method": source.source_method,
        "source_crop_run": source.crop_run,
        **source.source_crop_snapshot,
        "source_keypoints_run": source.source_keypoints_run,
        "source_keypoint_group": source.source_keypoint_group,
    }
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
            "method": SMART_FINALIZE_SUBJECT_MASKS_METHOD,
            "refinement_semantics": "canonical_component_masks",
            "components": list(component_names),
            "component_count": int(len(component_names)),
        },
        inputs=stage_inputs_payload,
    )
    write_stage_provenance(run_group, provenance)
    return run_group


def _write_component_metrics_chunk(
    component_group: zarr.Group,
    *,
    row_slice: slice,
    masks: np.ndarray,
    metric_level: str,
    write_attrs: bool = True,
) -> None:
    if metric_level not in _METRIC_LEVELS:
        raise ValueError(f"metric_level must be one of {_METRIC_LEVELS}; got {metric_level!r}.")
    component_metrics = _compute_component_topology_metrics(masks)
    if metric_level == "full":
        component_metrics.update(_compute_component_sigma_noise_metrics(masks))
        component_metrics.update(_compute_component_curvature_var_metrics(masks))
        component_metrics.update(_compute_component_shape_qc_metrics(masks))
    metrics_group = component_group["metrics"]
    for metric_name, values in component_metrics.items():
        metrics_group[str(metric_name)][row_slice] = np.asarray(values)
    if not write_attrs:
        return
    _set_component_metric_attrs(component_group, metric_level=metric_level)


def _set_component_metric_attrs(component_group: zarr.Group, *, metric_level: str) -> None:
    metrics_group = component_group["metrics"]
    metrics_group.attrs["metric_level"] = metric_level
    component_group.attrs["component_metric_level"] = metric_level
    if metric_level == "full":
        component_group.attrs["shape_qc_metrics_status"] = "computed"
        component_group.attrs.pop("shape_qc_metrics_deferred_reason", None)
    else:
        component_group.attrs["shape_qc_metrics_status"] = "deferred"
        component_group.attrs["shape_qc_metrics_deferred_reason"] = "metric_level=cheap"


def _write_canonical_component_chunk(
    run_group: zarr.Group,
    *,
    component_name: str,
    component_idx: int,
    row_slice: slice,
    masks: np.ndarray,
    source_masks: np.ndarray,
    metric_level: str,
    write_metric_attrs: bool = True,
) -> np.ndarray:
    masks_u8 = np.asarray(masks, dtype=np.uint8)
    source_u8 = np.asarray(source_masks, dtype=np.uint8)
    run_group["masks_roi"][row_slice, int(component_idx)] = masks_u8
    mask_present, area_px = _compute_mask_metrics(masks_u8[:, None, :, :])
    run_group["metrics/mask_present"][row_slice, int(component_idx)] = mask_present[:, 0]
    run_group["metrics/area_px"][row_slice, int(component_idx)] = area_px[:, 0]
    geometry_metrics = _compute_geometry_metrics(masks_u8[:, None, :, :])
    run_group["metrics/centroid_xy"][row_slice, int(component_idx), :] = geometry_metrics["centroid_xy"][:, 0, :]
    run_group["metrics/centroid_valid"][row_slice, int(component_idx)] = geometry_metrics["centroid_valid"][:, 0]
    run_group["metrics/bbox_xyxy"][row_slice, int(component_idx), :] = geometry_metrics["bbox_xyxy"][:, 0, :]
    run_group["metrics/bbox_valid"][row_slice, int(component_idx)] = geometry_metrics["bbox_valid"][:, 0]

    component_group = run_group["components"][component_name]
    component_group["mask_present"][row_slice] = mask_present[:, 0]
    component_group["area_px"][row_slice] = area_px[:, 0]
    component_group["source_seed_masks_roi"][row_slice] = source_u8
    component_group["source_row_fingerprint"][row_slice] = _compute_mask_row_fingerprints(source_u8)
    _write_component_metrics_chunk(
        component_group,
        row_slice=row_slice,
        masks=masks_u8,
        metric_level=metric_level,
        write_attrs=write_metric_attrs,
    )
    return np.asarray(mask_present[:, 0], dtype=bool)


def _write_finalization_metrics_chunk(
    run_group: zarr.Group,
    *,
    component_name: str,
    row_slice: slice,
    batch: _FinalizedComponentBatch,
    total_rows: int,
    ensure_shell: bool = True,
) -> None:
    component_group = run_group["components"][component_name]
    if ensure_shell:
        _create_finalization_metric_shell(component_group, batch=batch, total_rows=total_rows)
    metrics_group = component_group["finalization_metrics"]
    for metric_name, values in batch.metrics.items():
        metrics_group[str(metric_name)][row_slice] = np.asarray(values, dtype=np.float32)
    metrics_group["quality_code"][row_slice] = np.asarray(batch.quality_code, dtype=np.int16)
    metrics_group["quality_score"][row_slice] = np.asarray(batch.quality_score, dtype=np.float32)


def _process_and_write_finalizer_chunk(
    zarr_path: str,
    *,
    subject_run: str,
    refined_run: str,
    component_names: Sequence[str],
    required_raw_components: Sequence[str],
    start_row: int,
    stop_row: int,
    chunk_index: int,
    metric_level: str,
    assignment_keypoint_group: Optional[str],
    assignment_keypoints_run: Optional[str],
    total_rows: int,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    source = _load_source_subject_mask_run(root, subject_run)
    run_group = root["refined_subject_masks_runs"][refined_run]
    component_names = tuple(str(name) for name in component_names)
    component_to_index = {name: idx for idx, name in enumerate(component_names)}
    timing = _TimingRecorder()

    eye_assignment_context: _EyeAssignmentContext | None = None
    if set(_EYE_COMPONENTS).issubset(component_names):
        eye_assignment_context = _resolve_eye_assignment_context(
            root,
            source,
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
        )

    row_slice = slice(int(start_row), int(stop_row))
    chunk_timing: dict[str, object] = {
        "chunk_index": int(chunk_index),
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "row_count": int(stop_row) - int(start_row),
        "execution_backend": _DASK_WORKER_EXECUTION_BACKEND,
    }
    chunk_start = time.perf_counter()
    chunk_any = np.zeros((int(stop_row) - int(start_row),), dtype=bool)
    chunk_batches: dict[str, _FinalizedComponentBatch] = {}
    review_counts: dict[str, dict[str, int]] = {}
    reason_labels_by_component: dict[str, list[str]] = {}
    eyes_union_assignment_summary: dict[str, object] = {}

    for raw_component in required_raw_components:
        phase_start = time.perf_counter()
        batch = _finalize_source_component_rows(
            source,
            raw_component,
            start_row=start_row,
            stop_row=stop_row,
        )
        elapsed = timing.add(f"finalize_{raw_component}", time.perf_counter() - phase_start)
        chunk_timing[f"finalize_{raw_component}_seconds"] = elapsed
        chunk_batches[raw_component] = batch
        if raw_component == _RAW_EYE_UNION_COMPONENT:
            continue

        component_idx = int(component_to_index[raw_component])
        phase_start = time.perf_counter()
        present = _write_canonical_component_chunk(
            run_group,
            component_name=raw_component,
            component_idx=component_idx,
            row_slice=row_slice,
            masks=batch.masks,
            source_masks=batch.source_masks,
            metric_level=metric_level,
            write_metric_attrs=False,
        )
        _write_finalization_metrics_chunk(
            run_group,
            component_name=raw_component,
            row_slice=row_slice,
            batch=batch,
            total_rows=total_rows,
            ensure_shell=False,
        )
        elapsed = timing.add(f"write_{raw_component}", time.perf_counter() - phase_start)
        chunk_timing[f"write_{raw_component}_seconds"] = elapsed
        chunk_any |= present
        labels = np.asarray(batch.reason_labels, dtype=object)
        reason_labels_by_component[raw_component] = [str(value) for value in labels.tolist()]
        _add_review_counts(review_counts, raw_component, labels)

    if eye_assignment_context is not None:
        union_batch = chunk_batches[_RAW_EYE_UNION_COMPONENT]
        phase_start = time.perf_counter()
        assignment_chunk = _assign_finalized_eyes_union_rows(
            source,
            union_batch,
            eye_assignment_context,
            start_row=start_row,
            stop_row=stop_row,
        )
        elapsed = timing.add("eye_assignment", time.perf_counter() - phase_start)
        chunk_timing["eye_assignment_seconds"] = elapsed
        _merge_assignment_summary(eyes_union_assignment_summary, assignment_chunk.summary)
        for component_name in _EYE_COMPONENTS:
            component_idx = int(component_to_index[component_name])
            masks = np.asarray(assignment_chunk.masks[component_name], dtype=np.uint8)
            phase_start = time.perf_counter()
            present = _write_canonical_component_chunk(
                run_group,
                component_name=component_name,
                component_idx=component_idx,
                row_slice=row_slice,
                masks=masks,
                source_masks=masks,
                metric_level=metric_level,
                write_metric_attrs=False,
            )
            elapsed = timing.add(f"write_{component_name}", time.perf_counter() - phase_start)
            chunk_timing[f"write_{component_name}_seconds"] = elapsed
            chunk_any |= present
            labels = np.asarray(assignment_chunk.reason_labels[component_name], dtype=object)
            reason_labels_by_component[component_name] = [str(value) for value in labels.tolist()]
            _add_review_counts(review_counts, component_name, labels)

    chunk_timing["total_seconds"] = float(time.perf_counter() - chunk_start)
    return {
        "chunk_timing": chunk_timing,
        "phase_seconds": dict(timing.phase_seconds),
        "phase_counts": dict(timing.phase_counts),
        "review_counts": review_counts,
        "reason_labels_by_component": reason_labels_by_component,
        "rows_with_nonempty_masks": int(np.count_nonzero(chunk_any)),
        "eyes_union_assignment_summary": dict(_json_safe(eyes_union_assignment_summary)),
    }


def _compute_finalizer_dask_tasks(
    tasks: Sequence[object],
    *,
    scheduler_key: str,
    num_workers: Optional[int],
) -> list[dict[str, object]]:
    if not tasks:
        return []

    cluster = None
    client = None
    try:
        if scheduler_key == "distributed":
            if not HAVE_DISTRIBUTED:
                raise RuntimeError(
                    "Dask distributed is not available. Install dask[distributed] "
                    "or choose a different scheduler."
                )
            cluster_kwargs: dict[str, object] = {}
            if num_workers is not None:
                cluster_kwargs["n_workers"] = int(num_workers)
            cluster = LocalCluster(**cluster_kwargs)
            client = Client(cluster)
            results = list(client.gather(client.compute(list(tasks))))
        else:
            compute_kwargs: dict[str, object] = {"scheduler": scheduler_key}
            if num_workers is not None and scheduler_key != "single-threaded":
                compute_kwargs["num_workers"] = int(num_workers)
            results = list(dask.compute(*tasks, **compute_kwargs))
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
    return [dict(result) for result in results]


def finalize_subject_mask_run(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    metric_level: str = "cheap",
    write_eye_geometry: bool = False,
    execution_backend: str = _SERIAL_EXECUTION_BACKEND,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> dict[str, object]:
    """Finalize one subject-mask run into a canonical refined-subject run."""

    if bool(assignment_keypoint_group) != bool(assignment_keypoints_run):
        raise ValueError("Pass both assignment_keypoint_group and assignment_keypoints_run, or neither.")
    metric_level = str(metric_level)
    if metric_level not in _METRIC_LEVELS:
        raise ValueError(f"metric_level must be one of {_METRIC_LEVELS}; got {metric_level!r}.")
    scheduler_key = _normalize_scheduler(scheduler)
    normalized_num_workers = int(num_workers) if num_workers is not None else None
    execution_backend = _normalize_execution_backend(execution_backend)
    if execution_backend == _DASK_WORKER_EXECUTION_BACKEND and zarr_path is None:
        raise ValueError("execution_backend='dask_worker_chunks' requires a filesystem zarr_path.")
    dask_metadata: dict[str, object] = {
        "execution_backend": execution_backend,
        "dask_execution_enabled": execution_backend == _DASK_WORKER_EXECUTION_BACKEND,
        "dask_scheduler": scheduler_key,
        "dask_num_workers": normalized_num_workers,
        "dask_chunk_size": max(1, int(chunk_size)),
        "dask_version": getattr(dask, "__version__", "unknown"),
    }
    source = _load_source_subject_mask_run(root, subject_run)
    component_names = _requested_output_components(source, components)
    if not component_names:
        raise ValueError(f"subject_mask_runs/{source.run_name} has no finalizable subject-mask components.")
    required_raw_components = []
    if "subject_body" in component_names:
        required_raw_components.append("subject_body")
    if set(_EYE_COMPONENTS).issubset(component_names):
        required_raw_components.append(_RAW_EYE_UNION_COMPONENT)
    if "swim_bladder" in component_names:
        required_raw_components.append("swim_bladder")
    for raw_component in required_raw_components:
        if raw_component not in _FINALIZABLE_RAW_COMPONENTS:
            raise ValueError(f"Unsupported raw component {raw_component!r}.")
        _require_available_component(source, raw_component, "subject_mask_runs")

    target_run = str(refined_run or _default_refined_run_name())
    refined_parent = root.get("refined_subject_masks_runs")
    target_exists = refined_parent is not None and target_run in refined_parent
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "source_subject_mask_run": source.run_name,
        "refined_run": target_run,
        "refined_run_exists": bool(target_exists),
        "would_create_refined_run": not bool(target_exists),
        "mutates_archive": not bool(dry_run),
        "component_names": list(component_names),
        "raw_component_names": list(required_raw_components),
        "source_crop_run": source.crop_run,
        "roi_count": int(source.masks_roi.shape[0]),
        "chunk_size": max(1, int(chunk_size)),
        "chunk_count": len(_row_chunks(int(source.masks_roi.shape[0]), max(1, int(chunk_size)))),
        "metric_level": metric_level,
        "write_eye_geometry": bool(write_eye_geometry),
        **dask_metadata,
        "source_surface_kind": source.mask_surface_kind,
    }
    if dry_run:
        return summary

    refined_parent = root.require_group("refined_subject_masks_runs")
    if target_run in refined_parent:
        if not overwrite:
            raise ValueError(
                f"refined_subject_masks_runs/{target_run} already exists. Pass overwrite=True to replace it."
            )
        del refined_parent[target_run]

    stage_start = time.perf_counter()
    timing = _TimingRecorder()
    total_rows = int(source.masks_roi.shape[0])
    height = int(source.masks_roi.shape[2])
    width = int(source.masks_roi.shape[3])
    component_names = tuple(component_names)
    component_to_index = {name: idx for idx, name in enumerate(component_names)}
    chunk_ranges = _row_chunks(total_rows, max(1, int(chunk_size)))
    reason_labels_by_component = {
        component_name: np.full((total_rows,), "clean", dtype=object)
        for component_name in component_names
    }
    review_counts: dict[str, dict[str, int]] = {}
    source_payloads: dict[str, dict[str, object]] = {}
    first_batches: dict[str, _FinalizedComponentBatch] = {}
    rows_with_nonempty_masks = 0
    eyes_union_assignment_summary: dict[str, object] = {}
    assignment_keypoint_attrs: dict[str, object] = {}
    eye_assignment_context: _EyeAssignmentContext | None = None
    if set(_EYE_COMPONENTS).issubset(component_names):
        eye_assignment_context = _resolve_eye_assignment_context(
            root,
            source,
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
        )
        assignment_keypoint_attrs = build_assignment_keypoint_attrs(
            eye_assignment_context.keypoint_run_name,
            assignment_keypoint_group=eye_assignment_context.keypoint_group_name,
            selection=eye_assignment_context.keypoint_source_kind,
        )
        assignment_keypoint_attrs["assignment_keypoint_success_dataset"] = str(
            eye_assignment_context.keypoint_success_dataset
        )

    extra_attrs: dict[str, object] = {
        "finalization_semantics": "smart_probability_to_refined_candidate",
        "component_metric_level": metric_level,
        "eye_geometry_requested": bool(write_eye_geometry),
        **dask_metadata,
        "source_input_subject_mask_run": source.run_name,
        "source_component_runs": {
            component_name: source.run_name
            for component_name in component_names
        },
        "source_component_sources": {
            component_name: {
                "source_stage": "subject_mask_runs",
                "source_run": source.run_name,
            }
            for component_name in component_names
        },
    }
    if eye_assignment_context is not None:
        extra_attrs.update(assignment_keypoint_attrs)

    provenance_inputs: dict[str, object] = {
        "source_input_subject_mask_run": source.run_name,
        "finalization_semantics": "smart_probability_to_refined_candidate",
        "source_component_runs": dict(extra_attrs["source_component_runs"]),
        "source_component_sources": dict(extra_attrs["source_component_sources"]),
        "chunk_size": int(max(1, int(chunk_size))),
        "chunk_count": int(len(chunk_ranges)),
        "component_metric_level": metric_level,
        "eye_geometry_requested": bool(write_eye_geometry),
        **dask_metadata,
    }
    if eye_assignment_context is not None:
        provenance_inputs.update(assignment_keypoint_attrs)

    with timing.phase("target_init"):
        run_group = _create_refined_run_shell(
            refined_parent=refined_parent,
            target_run=target_run,
            source=source,
            component_names=component_names,
            extra_attrs=extra_attrs,
            provenance_inputs=provenance_inputs,
            stage_command=" ".join(sys.argv) if sys.argv else "unknown",
        )

    if execution_backend == _DASK_WORKER_EXECUTION_BACKEND:
        assert zarr_path is not None
        with timing.phase("dask_prepare_shells"):
            for raw_component in required_raw_components:
                sample_batch = _finalize_source_component_rows(
                    source,
                    raw_component,
                    start_row=0,
                    stop_row=min(1, total_rows),
                )
                first_batches.setdefault(raw_component, sample_batch)
                if raw_component == _RAW_EYE_UNION_COMPONENT:
                    continue
                component_group = run_group["components"][raw_component]
                _create_finalization_metric_shell(
                    component_group,
                    batch=sample_batch,
                    metric_names=_FINALIZATION_METRIC_NAMES,
                    total_rows=total_rows,
                )
                source_payloads.setdefault(
                    raw_component,
                    _source_payload_for_finalized_component(source, sample_batch),
                )

        tasks = [
            delayed(_process_and_write_finalizer_chunk)(
                str(zarr_path),
                subject_run=source.run_name,
                refined_run=target_run,
                component_names=component_names,
                required_raw_components=tuple(required_raw_components),
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                metric_level=metric_level,
                assignment_keypoint_group=assignment_keypoint_group,
                assignment_keypoints_run=assignment_keypoints_run,
                total_rows=total_rows,
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunk_ranges)
        ]
        with timing.phase("dask_compute"):
            dask_results = _compute_finalizer_dask_tasks(
                tasks,
                scheduler_key=scheduler_key,
                num_workers=normalized_num_workers,
            )
        for result in sorted(dask_results, key=lambda item: int(dict(item["chunk_timing"]).get("chunk_index") or 0)):
            chunk_timing = dict(result["chunk_timing"])
            timing.chunk_timings.append(chunk_timing)
            for phase, seconds in dict(result.get("phase_seconds") or {}).items():
                timing.add(str(phase), float(seconds))
            _merge_review_counts(review_counts, dict(result.get("review_counts") or {}))
            rows_with_nonempty_masks += int(result.get("rows_with_nonempty_masks") or 0)
            row_slice = slice(int(chunk_timing["start_row"]), int(chunk_timing["stop_row"]))
            for component_name, labels in dict(result.get("reason_labels_by_component") or {}).items():
                reason_labels_by_component[str(component_name)][row_slice] = np.asarray(labels, dtype=object)
            if result.get("eyes_union_assignment_summary"):
                _merge_assignment_summary(
                    eyes_union_assignment_summary,
                    dict(result["eyes_union_assignment_summary"]),
                )
        for component_name in component_names:
            _set_component_metric_attrs(
                run_group["components"][component_name],
                metric_level=metric_level,
            )
    else:
        for chunk_index, (start_row, stop_row) in enumerate(chunk_ranges):
            chunk_start = time.perf_counter()
            chunk_timing: dict[str, object] = {
                "chunk_index": int(chunk_index),
                "start_row": int(start_row),
                "stop_row": int(stop_row),
                "row_count": int(stop_row) - int(start_row),
                "execution_backend": _SERIAL_EXECUTION_BACKEND,
            }
            row_slice = slice(int(start_row), int(stop_row))
            chunk_any = np.zeros((int(stop_row) - int(start_row),), dtype=bool)
            chunk_batches: dict[str, _FinalizedComponentBatch] = {}
            for raw_component in required_raw_components:
                phase_start = time.perf_counter()
                batch = _finalize_source_component_rows(
                    source,
                    raw_component,
                    start_row=start_row,
                    stop_row=stop_row,
                )
                elapsed = timing.add(f"finalize_{raw_component}", time.perf_counter() - phase_start)
                chunk_timing[f"finalize_{raw_component}_seconds"] = elapsed
                chunk_batches[raw_component] = batch
                first_batches.setdefault(raw_component, batch)
                if raw_component == _RAW_EYE_UNION_COMPONENT:
                    continue
                component_idx = int(component_to_index[raw_component])
                phase_start = time.perf_counter()
                present = _write_canonical_component_chunk(
                    run_group,
                    component_name=raw_component,
                    component_idx=component_idx,
                    row_slice=row_slice,
                    masks=batch.masks,
                    source_masks=batch.source_masks,
                    metric_level=metric_level,
                )
                chunk_any |= present
                _write_finalization_metrics_chunk(
                    run_group,
                    component_name=raw_component,
                    row_slice=row_slice,
                    batch=batch,
                    total_rows=total_rows,
                )
                elapsed = timing.add(f"write_{raw_component}", time.perf_counter() - phase_start)
                chunk_timing[f"write_{raw_component}_seconds"] = elapsed
                reason_labels_by_component[raw_component][row_slice] = np.asarray(batch.reason_labels, dtype=object)
                _add_review_counts(review_counts, raw_component, batch.reason_labels)
                source_payloads.setdefault(raw_component, _source_payload_for_finalized_component(source, batch))

            if eye_assignment_context is not None:
                union_batch = chunk_batches[_RAW_EYE_UNION_COMPONENT]
                phase_start = time.perf_counter()
                assignment_chunk = _assign_finalized_eyes_union_rows(
                    source,
                    union_batch,
                    eye_assignment_context,
                    start_row=start_row,
                    stop_row=stop_row,
                )
                elapsed = timing.add("eye_assignment", time.perf_counter() - phase_start)
                chunk_timing["eye_assignment_seconds"] = elapsed
                _merge_assignment_summary(eyes_union_assignment_summary, assignment_chunk.summary)
                for component_name in _EYE_COMPONENTS:
                    component_idx = int(component_to_index[component_name])
                    masks = np.asarray(assignment_chunk.masks[component_name], dtype=np.uint8)
                    phase_start = time.perf_counter()
                    present = _write_canonical_component_chunk(
                        run_group,
                        component_name=component_name,
                        component_idx=component_idx,
                        row_slice=row_slice,
                        masks=masks,
                        source_masks=masks,
                        metric_level=metric_level,
                    )
                    elapsed = timing.add(f"write_{component_name}", time.perf_counter() - phase_start)
                    chunk_timing[f"write_{component_name}_seconds"] = elapsed
                    chunk_any |= present
                    labels = np.asarray(assignment_chunk.reason_labels[component_name], dtype=object)
                    reason_labels_by_component[component_name][row_slice] = labels
                    _add_review_counts(review_counts, component_name, labels)

            rows_with_nonempty_masks += int(np.count_nonzero(chunk_any))
            chunk_timing["total_seconds"] = float(time.perf_counter() - chunk_start)
            timing.chunk_timings.append(chunk_timing)

    if eye_assignment_context is not None:
        usable_rows = int(eyes_union_assignment_summary.get("assigned_rows") or 0) + int(
            eyes_union_assignment_summary.get("assigned_needs_review_rows") or 0
        )
        if usable_rows <= 0:
            raise ValueError(
                f"eyes_union assignment for subject_mask_runs/{source.run_name} produced no usable LR eye rows; "
                f"summary={eyes_union_assignment_summary!r}."
            )
        union_batch = first_batches[_RAW_EYE_UNION_COMPONENT]
        run_group.attrs["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        provenance = dict(run_group.attrs.get("provenance") or {})
        provenance_inputs_payload = dict(provenance.get("inputs") or {})
        provenance_inputs_payload["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        provenance["inputs"] = provenance_inputs_payload
        run_group.attrs["provenance"] = provenance
        for component_name in _EYE_COMPONENTS:
            source_payloads[component_name] = _source_payload_for_assigned_eye_component(
                source,
                component_name=component_name,
                union_batch=union_batch,
                assignment_summary=eyes_union_assignment_summary,
                keypoint_run_name=eye_assignment_context.keypoint_run_name,
                keypoint_group_name=eye_assignment_context.keypoint_group_name,
                keypoint_success_dataset=eye_assignment_context.keypoint_success_dataset,
                keypoint_source_kind=eye_assignment_context.keypoint_source_kind,
            )

    created = str(run_group.attrs.get("created_at_utc") or _utc_now())
    for component_name in component_names:
        component_group = run_group["components"][component_name]
        with timing.phase(f"write_reason_columns_{component_name}"):
            write_reason_columns(
                component_group,
                reason_labels_by_component[component_name],
                chunk_size=max(1, min(256, total_rows)),
                include_reason_text=True,
                overwrite=True,
            )
        with timing.phase(f"write_component_provenance_{component_name}"):
            _ensure_refined_component_provenance_payload(
                run_group,
                component_name=component_name,
                source_payload=source_payloads[component_name],
                created_at_utc=created,
            )

    if write_eye_geometry and set(_EYE_COMPONENTS).issubset(component_names):
        with timing.phase("write_eye_geometry"):
            write_refined_subject_eye_geometry(run_group)
        run_group.attrs["eye_geometry_status"] = "computed"
    elif set(_EYE_COMPONENTS).issubset(component_names):
        run_group.attrs["eye_geometry_status"] = "deferred"
        run_group.attrs["eye_geometry_deferred_reason"] = "write_eye_geometry=false"

    duration_seconds = float(time.perf_counter() - stage_start)
    timing_summary = timing.summary(total_rows=total_rows, duration_seconds=duration_seconds)
    timing_summary.update(dask_metadata)
    run_group.attrs["duration_seconds"] = duration_seconds
    run_group.attrs["summary_statistics"] = {
        "rows_total": int(total_rows),
        "rows_with_nonempty_masks": int(rows_with_nonempty_masks),
        "duration_seconds": float(duration_seconds),
        "rows_per_second": float(timing_summary["rows_per_second"]),
    }
    run_group.attrs["smart_finalizer_timing_summary"] = dict(_json_safe(timing_summary))
    run_group.attrs["smart_finalizer_chunk_timings"] = list(_json_safe(timing.chunk_timings))
    run_group.attrs["smart_finalizer_review_counts"] = review_counts
    run_group.attrs["smart_finalizer_chunk_size"] = int(max(1, int(chunk_size)))
    run_group.attrs["smart_finalizer_chunk_count"] = int(len(chunk_ranges))
    run_group.attrs["smart_finalizer_metric_level"] = metric_level
    run_group.attrs["smart_finalizer_write_eye_geometry"] = bool(write_eye_geometry)
    run_group.attrs["smart_finalizer_execution_backend"] = execution_backend
    run_group.attrs["dask_execution_enabled"] = execution_backend == _DASK_WORKER_EXECUTION_BACKEND
    run_group.attrs["dask_scheduler"] = scheduler_key
    run_group.attrs["dask_num_workers"] = normalized_num_workers
    run_group.attrs["dask_chunk_size"] = int(max(1, int(chunk_size)))
    run_group.attrs["dask_version"] = getattr(dask, "__version__", "unknown")
    refined_parent.attrs["latest"] = target_run
    refined_parent.attrs["refined_subject_mask_review_status_latest"] = target_run

    summary.update(
        {
            "status": "updated",
            "refined_run_exists": False,
            "would_create_refined_run": True,
            "duration_seconds": duration_seconds,
            "timing_summary": dict(_json_safe(timing_summary)),
            "metric_level": metric_level,
            "write_eye_geometry": bool(write_eye_geometry),
            "review_counts": review_counts,
            "eyes_union_assignment_summary": (
                dict(_json_safe(eyes_union_assignment_summary))
                if eyes_union_assignment_summary
                else None
            ),
        }
    )
    return summary


def finalize_subject_masks(
    zarr_path: str | Path,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    metric_level: str = "cheap",
    write_eye_geometry: bool = False,
    execution_backend: str = _SERIAL_EXECUTION_BACKEND,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
    console: Any = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    summary = finalize_subject_mask_run(
        root,
        zarr_path=zarr_path,
        subject_run=subject_run,
        refined_run=refined_run,
        components=components,
        chunk_size=chunk_size,
        metric_level=metric_level,
        write_eye_geometry=write_eye_geometry,
        execution_backend=execution_backend,
        scheduler=scheduler,
        num_workers=num_workers,
        overwrite=overwrite,
        dry_run=dry_run,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    if not dry_run:
        run_name = str(summary["refined_run"])
        refined_group = root["refined_subject_masks_runs"][run_name]
        emit_refined_subject_mask_stage_completion(
            root,
            zarr_path,
            run_group=refined_group,
            run_name=run_name,
            source=_REFINED_SUBJECT_MASKS_STATUS_SOURCE,
            console=console,
            invalidate_on_ok=True,
        )
    return summary


def _parse_components(values: Optional[Sequence[Sequence[str]]], single_values: Optional[Sequence[str]]) -> Optional[list[str]]:
    merged: list[str] = []
    for group in values or ():
        merged.extend(str(value) for value in group)
    for value in single_values or ():
        merged.append(str(value))
    return merged or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the Palette zarr archive.")
    parser.add_argument(
        "--subject-run",
        "--source-run",
        dest="subject_run",
        help="Source subject_mask_runs/<run> to finalize. Defaults to latest subject-mask run.",
    )
    parser.add_argument(
        "--refined-run",
        "--run-name",
        dest="refined_run",
        help="Target refined_subject_masks_runs/<run>. Defaults to a timestamped refined run.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        action="append",
        help="Optional output components to create. Use eyes_union to request eye_left/eye_right assignment.",
    )
    parser.add_argument(
        "--component",
        action="append",
        dest="component_values",
        help="Optional single output component selector. Repeat to add more components.",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of ROI rows to finalize per write chunk.")
    parser.add_argument(
        "--metric-level",
        choices=_METRIC_LEVELS,
        default="cheap",
        help="Metric depth to compute during finalization. cheap writes topology only; full also writes slower shape QC metrics.",
    )
    parser.add_argument(
        "--write-eye-geometry",
        action="store_true",
        help="Also compute refined eye geometry/ellipse relations during finalization.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=_EXECUTION_BACKENDS,
        default=_SERIAL_EXECUTION_BACKEND,
        help="Execution backend. Use dask_worker_chunks to let Dask workers write disjoint row chunks.",
    )
    parser.add_argument(
        "--scheduler",
        type=_normalize_scheduler,
        choices=_SUPPORTED_SCHEDULERS,
        default="single-threaded",
        help=(
            "Dask scheduler used when --execution-backend=dask_worker_chunks; recorded as instrumentation "
            "for serial_driver."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Worker-count setting to record for future Dask-backed finalization.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target refined run.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the plan without mutating the archive.")
    parser.add_argument(
        "--assignment-keypoint-group",
        choices=("refined_keypoints_runs", "keypoints_runs"),
        help="Explicit keypoint group to use for eyes_union -> eye_left/eye_right assignment.",
    )
    parser.add_argument(
        "--assignment-keypoints-run",
        help="Explicit keypoint run to use for eyes_union -> eye_left/eye_right assignment.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    components = _parse_components(args.components, args.component_values)
    summary = finalize_subject_masks(
        args.zarr_path,
        subject_run=args.subject_run,
        refined_run=args.refined_run,
        components=components,
        chunk_size=int(args.chunk_size),
        metric_level=args.metric_level,
        write_eye_geometry=bool(args.write_eye_geometry),
        execution_backend=args.execution_backend,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        assignment_keypoint_group=args.assignment_keypoint_group,
        assignment_keypoints_run=args.assignment_keypoints_run,
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
