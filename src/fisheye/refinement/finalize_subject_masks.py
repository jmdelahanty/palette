"""Create refined subject-mask candidates from raw subject-mask outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..shared.provenance_attrs import (
    ASSIGNMENT_KEYPOINT_CONTRACT_VALUE,
    build_assignment_keypoint_attrs,
)
from ..shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from ..shared.subject_mask_registry_status import emit_refined_subject_mask_stage_completion
from ..tune.refined_subject_mask_review import (
    RefinedSubjectComponentSeed,
    SourceSubjectMaskRun,
    _create_refined_subject_run_from_component_seeds,
    _default_refined_run_name,
    _load_source_subject_mask_run,
    _normalize_component_name,
    _probability_encoding_for_group,
    _probability_thresholds_for_labels,
    _source_component_provenance_payload,
)
from ..utils.zarr_io import open_zarr_root
from .assemble_refined_subject_masks import (
    CANONICAL_COMPONENT_ORDER,
    _has_available_component,
    _require_available_component,
    _resolve_eye_keypoint_indices,
    _resolve_keypoint_success_array,
    _resolve_subject_keypoint_group,
)
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


def _component_surface_batch(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> tuple[np.ndarray, bool, str, Optional[str], float, int]:
    component_idx = _require_available_component(source, component_name, "subject_mask_runs")
    probabilities = source.group.get("mask_probs_roi")
    threshold = _component_threshold(source, component_name, component_idx)
    if probabilities is not None:
        encoding = source.probability_encoding or _probability_encoding_for_group(source.group)
        raw = np.asarray(probabilities[:, component_idx])
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
        np.asarray(masks[:, component_idx], dtype=np.uint8),
        False,
        "masks_roi",
        None,
        threshold,
        component_idx,
    )


def _finalize_source_component(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> _FinalizedComponentBatch:
    surfaces, is_probability, surface_path, encoding, threshold, _component_idx = _component_surface_batch(
        source,
        component_name,
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


def _assign_finalized_eyes_union(
    root: zarr.Group,
    source: SourceSubjectMaskRun,
    union_batch: _FinalizedComponentBatch,
    *,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> tuple[dict[str, RefinedSubjectComponentSeed], dict[str, object], dict[str, np.ndarray]]:
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
    assignment = assign_eyes_union_to_lr(
        np.asarray(union_batch.masks, dtype=np.uint8),
        keypoints_roi=np.asarray(keypoints_roi[:], dtype=np.float32),
        keypoint_success=np.asarray(keypoint_success, dtype=bool),
        eye_keypoint_indices=eye_keypoint_indices,
    )
    usable_rows = int(assignment.summary.get("assigned_rows") or 0) + int(
        assignment.summary.get("assigned_needs_review_rows") or 0
    )
    if usable_rows <= 0:
        raise ValueError(
            f"eyes_union assignment for subject_mask_runs/{source.run_name} produced no usable LR eye rows; "
            f"summary={assignment.summary!r}."
        )
    summary = dict(assignment.summary)
    summary["keypoint_run"] = str(keypoint_run_name)
    summary["keypoint_group"] = str(keypoint_group_name)
    summary["keypoint_success_dataset"] = str(success_dataset)
    summary["keypoint_source_kind"] = str(keypoint_source_kind)
    summary["assignment_keypoint_contract"] = ASSIGNMENT_KEYPOINT_CONTRACT_VALUE

    seeds: dict[str, RefinedSubjectComponentSeed] = {}
    reason_labels_by_component: dict[str, np.ndarray] = {}
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
        seeds[component_name] = RefinedSubjectComponentSeed(
            component_name=component_name,
            masks=np.asarray(assignment.masks[component_name], dtype=np.uint8),
            source_payload=_source_payload_for_assigned_eye_component(
                source,
                component_name=component_name,
                union_batch=union_batch,
                assignment_summary=summary,
                keypoint_run_name=keypoint_run_name,
                keypoint_group_name=keypoint_group_name,
                keypoint_success_dataset=success_dataset,
                keypoint_source_kind=keypoint_source_kind,
            ),
            reason_labels=reason_labels,
            source_masks=np.asarray(assignment.masks[component_name], dtype=np.uint8),
        )
    return seeds, summary, reason_labels_by_component


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


def _write_finalization_metrics(
    refined_group: zarr.Group,
    *,
    component_name: str,
    batch: _FinalizedComponentBatch,
) -> None:
    component_group = refined_group.require_group("components").require_group(component_name)
    metrics_group = component_group.require_group("finalization_metrics")
    total_rows = int(batch.masks.shape[0])
    chunks = (refined_subject_mask_metric_row_chunk(total_rows),)
    for metric_name, values in sorted(batch.metrics.items()):
        metrics_group.create_array(
            metric_name,
            data=np.asarray(values, dtype=np.float32),
            chunks=chunks,
            overwrite=True,
        )
    metrics_group.create_array(
        "quality_code",
        data=np.asarray(batch.quality_code, dtype=np.int16),
        chunks=chunks,
        overwrite=True,
    )
    metrics_group.create_array(
        "quality_score",
        data=np.asarray(batch.quality_score, dtype=np.float32),
        chunks=chunks,
        overwrite=True,
    )
    metrics_group.attrs["schema_id"] = "refined_subject_component_finalization_metrics_v1"
    metrics_group.attrs["method"] = SMART_FINALIZE_SUBJECT_MASKS_METHOD
    metrics_group.attrs["source_component"] = str(batch.component_name)
    metrics_group.attrs["source_surface_path"] = str(batch.source_surface_path)
    metrics_group.attrs["source_surface_kind"] = str(batch.source_surface_kind)


def _review_counts_from_labels(labels: np.ndarray) -> dict[str, int]:
    needs_review = 0
    pending = 0
    for label in np.asarray(labels, dtype=object).reshape(-1):
        if "needs_review" in str(label):
            needs_review += 1
        else:
            pending += 1
    return {"pending": int(pending), "needs_review": int(needs_review)}


def _build_summary_statistics(refined_group: zarr.Group, duration_seconds: float) -> dict[str, object]:
    masks = np.asarray(refined_group["masks_roi"][:], dtype=np.uint8)
    rows_with_any = int(np.count_nonzero(np.any(masks > 0, axis=(1, 2, 3))))
    return {
        "rows_total": int(masks.shape[0]),
        "rows_with_nonempty_masks": rows_with_any,
        "duration_seconds": float(duration_seconds),
    }


def finalize_subject_mask_run(
    root: zarr.Group,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> dict[str, object]:
    """Finalize one subject-mask run into a canonical refined-subject run."""

    if bool(assignment_keypoint_group) != bool(assignment_keypoints_run):
        raise ValueError("Pass both assignment_keypoint_group and assignment_keypoints_run, or neither.")
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
    final_batches: dict[str, _FinalizedComponentBatch] = {}
    component_seeds: dict[str, RefinedSubjectComponentSeed] = {}
    review_counts: dict[str, dict[str, int]] = {}
    for raw_component in required_raw_components:
        batch = _finalize_source_component(source, raw_component)
        final_batches[raw_component] = batch
        if raw_component == _RAW_EYE_UNION_COMPONENT:
            continue
        component_seeds[raw_component] = RefinedSubjectComponentSeed(
            component_name=raw_component,
            masks=np.asarray(batch.masks, dtype=np.uint8),
            source_payload=_source_payload_for_finalized_component(source, batch),
            reason_labels=np.asarray(batch.reason_labels, dtype=object),
            source_masks=np.asarray(batch.source_masks, dtype=np.uint8),
        )
        review_counts[raw_component] = _review_counts_from_labels(batch.reason_labels)

    eyes_union_assignment_summary: Optional[dict[str, object]] = None
    assignment_keypoint_attrs: dict[str, object] = {}
    if set(_EYE_COMPONENTS).issubset(component_names):
        union_batch = final_batches[_RAW_EYE_UNION_COMPONENT]
        eye_seeds, eyes_union_assignment_summary, eye_reason_labels = _assign_finalized_eyes_union(
            root,
            source,
            union_batch,
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
        )
        component_seeds.update(eye_seeds)
        for component_name, labels in eye_reason_labels.items():
            review_counts[component_name] = _review_counts_from_labels(labels)
        assignment_keypoint_attrs = build_assignment_keypoint_attrs(
            eyes_union_assignment_summary["keypoint_run"],
            assignment_keypoint_group=eyes_union_assignment_summary["keypoint_group"],
            selection=str(eyes_union_assignment_summary.get("keypoint_source_kind") or "unknown"),
        )
        assignment_keypoint_attrs["assignment_keypoint_success_dataset"] = str(
            eyes_union_assignment_summary["keypoint_success_dataset"]
        )

    extra_attrs: dict[str, object] = {
        "finalization_semantics": "smart_probability_to_refined_candidate",
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
    if eyes_union_assignment_summary is not None:
        extra_attrs["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        extra_attrs.update(assignment_keypoint_attrs)

    provenance_inputs: dict[str, object] = {
        "source_input_subject_mask_run": source.run_name,
        "finalization_semantics": "smart_probability_to_refined_candidate",
        "source_component_runs": dict(extra_attrs["source_component_runs"]),
        "source_component_sources": dict(extra_attrs["source_component_sources"]),
    }
    if eyes_union_assignment_summary is not None:
        provenance_inputs["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        provenance_inputs.update(assignment_keypoint_attrs)

    refined = _create_refined_subject_run_from_component_seeds(
        refined_parent=refined_parent,
        target_run=target_run,
        reference_source=source,
        component_names=component_names,
        component_seeds=component_seeds,
        coarse_source_subject_mask_run=source.run_name,
        coarse_source_subject_mask_method=source.source_method,
        source_keypoints_run=source.source_keypoints_run,
        source_keypoint_group=source.source_keypoint_group,
        run_method=SMART_FINALIZE_SUBJECT_MASKS_METHOD,
        stage_command=" ".join(sys.argv) if sys.argv else "unknown",
        extra_attrs=extra_attrs,
        provenance_inputs=provenance_inputs,
    )
    for component_name, batch in final_batches.items():
        if component_name in refined.component_to_index:
            _write_finalization_metrics(refined.group, component_name=component_name, batch=batch)
    duration_seconds = float(time.perf_counter() - stage_start)
    refined.group.attrs["duration_seconds"] = duration_seconds
    refined.group.attrs["summary_statistics"] = _build_summary_statistics(refined.group, duration_seconds)
    refined.group.attrs["smart_finalizer_review_counts"] = review_counts

    summary.update(
        {
            "status": "updated",
            "refined_run_exists": False,
            "would_create_refined_run": True,
            "duration_seconds": duration_seconds,
            "review_counts": review_counts,
            "eyes_union_assignment_summary": eyes_union_assignment_summary,
        }
    )
    return summary


def finalize_subject_masks(
    zarr_path: str | Path,
    *,
    subject_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
    console: Any = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    summary = finalize_subject_mask_run(
        root,
        subject_run=subject_run,
        refined_run=refined_run,
        components=components,
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
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        assignment_keypoint_group=args.assignment_keypoint_group,
        assignment_keypoints_run=args.assignment_keypoints_run,
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
