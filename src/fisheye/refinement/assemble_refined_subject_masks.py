"""Assemble multi-source refined subject-mask runs and finalize them immediately."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..pose.schema import resolve_required_keypoint_indices_from_attrs
from ..shared.json_safety import json_attr_safe
from ..shared.mask_store import MaskStore, MaskStoreError, open_mask_store
from ..shared.provenance_attrs import (
    ASSIGNMENT_KEYPOINT_CONTRACT_VALUE,
    CANONICAL_SOURCE_CROP_SNAPSHOT_ATTRS,
    build_assignment_keypoint_attrs,
    build_source_crop_snapshot_attrs,
    extract_source_crop_snapshot_attrs,
    resolve_assignment_keypoint_group,
    resolve_assignment_keypoints_run,
    resolve_source_keypoints_run,
)
from ..shared.row_lineage import (
    ROW_IDENTITY_MODE_SCHEMA,
    assert_row_lineage_sources_equal,
    resolve_row_lineage_identity_mode,
    resolve_source_crop_row_ids,
)
from ..shared.subject_mask_registry_status import emit_refined_subject_mask_stage_completion
from ..shared.zarr_run_completion import require_runs_parent
from ..tune.refined_subject_mask_review import (
    RefinedSubjectComponentSeed,
    SourceSubjectMaskRun,
    _aggregate_run_state,
    _build_single_source_component_seeds,
    _create_refined_subject_run_from_component_seeds,
    _default_refined_run_name,
    _get_component_provenance_group,
    _infer_refined_label_schema_id,
    _load_source_subject_mask_run,
    _normalize_component_name,
    _review_payload,
)
from .subject_eye_assignment import (
    EYES_UNION_ASSIGNMENT_METHOD,
    assign_eyes_union_to_lr,
    reconcile_keypoint_mask_row_identity,
)
from ..shared.zarr_io import open_zarr_root

ASSEMBLE_REFINED_SUBJECT_METHOD = "assemble_refined_subject_masks_v1"
CANONICAL_COMPONENT_ORDER = ("subject_body", "eye_left", "eye_right", "swim_bladder")
_REFINED_SUBJECT_MASKS_STATUS_SOURCE = "runtime_assemble_refined_subject_masks"
_EYE_COMPONENTS = ("eye_left", "eye_right")
_RAW_EYE_UNION_COMPONENT = "eyes_union"
_KEYPOINT_GROUP_CHOICES = ("refined_keypoints_runs", "keypoints_runs")
_KEYPOINT_SUCCESS_DATASET_CANDIDATES = (
    "usable_keypoints",
    "detection_success",
    "refined_success",
    "source_success",
)
_SOURCE_VIEW_CROP_SIGNATURE_DIFF_PATHS = frozenset(
    {
        "source_crop_signature.detection_source_path",
        "source_crop_signature.detection_source_type",
    }
)


_json_safe = json_attr_safe


class _MaskStoreDenseArrayView:
    """Array-like dense view for readers that still use ``masks_roi`` indexing."""

    def __init__(self, mask_store: MaskStore):
        self._mask_store = mask_store
        self.shape = tuple(int(dim) for dim in mask_store.shape)
        self.dtype = np.dtype(np.uint8)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, key: object) -> np.ndarray:
        row_scalar = False
        if not isinstance(key, tuple):
            rows = key
            channels = None
            tail = ()
            row_scalar = isinstance(rows, (int, np.integer))
        else:
            padded = tuple(key) + (slice(None),) * max(0, 4 - len(key))
            rows, channels, *tail = padded[:4]
            row_scalar = isinstance(rows, (int, np.integer))
        dense = self._mask_store.read_dense(rows=rows, channels=channels)
        if row_scalar:
            dense = dense[0]
        if channels is not None and isinstance(channels, (int, np.integer)):
            dense = dense[0] if dense.ndim == 3 else dense[:, 0]
        if tail and any(item != slice(None) for item in tail):
            dense = dense[(slice(None),) * (dense.ndim - len(tail)) + tuple(tail)]
        return dense


def _resolve_latest_parent_run(parent: Any, label: str) -> str:
    if parent is None:
        raise RuntimeError(f"No {label} found in archive.")
    latest = parent.attrs.get("latest")
    if latest and str(latest) in parent:
        return str(latest)
    keys = sorted(str(key) for key in parent.keys())
    if not keys:
        raise RuntimeError(f"No runs found under {label}.")
    return keys[-1]


def _lineage_array(root: zarr.Group, group: zarr.Group, *, crop_run: str, name: str) -> Any | None:
    if name in group:
        return group[name]
    crop_group = root.get(f"crop_runs/{crop_run}") if crop_run else None
    if crop_group is None:
        return None
    return crop_group.get(name)


def _required_array_equal(name: str, left: Any, right: Any) -> None:
    left_arr = np.asarray(left[:])
    right_arr = np.asarray(right[:])
    if left_arr.shape != right_arr.shape or not np.array_equal(left_arr, right_arr):
        raise ValueError(f"Alignment mismatch for {name}.")


def _parse_crop_signature(value: object) -> object:
    if isinstance(value, Mapping):
        return _json_safe(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return text
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        return _json_safe(parsed)
    return _json_safe(value)


def _normalized_crop_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in snapshot.items():
        if key == "source_crop_signature":
            normalized[key] = _parse_crop_signature(value)
        else:
            normalized[key] = _json_safe(value)
    return normalized


def _crop_snapshot_diff_paths(left: object, right: object, *, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(child_prefix)
            else:
                paths.update(_crop_snapshot_diff_paths(left[key], right[key], prefix=child_prefix))
        return paths
    return set() if left == right else {prefix}


def _is_allowed_source_view_crop_snapshot_mismatch(
    reference: SourceSubjectMaskRun,
    other: SourceSubjectMaskRun,
) -> bool:
    """Allow old refined views that differ only by refined/manual/interpolated labels.

    Some historical refined subject-mask components recorded the crop signature
    against a refined-detect subview such as ``manual`` or ``interpolated`` while
    still sharing the same crop run, row lineage, and ROI geometry as the
    canonical ``instances`` view. In that case the run-level assembled surface can
    use the reference crop surface while component provenance preserves the old
    source view.
    """

    reference_snapshot = _normalized_crop_snapshot(reference.source_crop_snapshot)
    other_snapshot = _normalized_crop_snapshot(other.source_crop_snapshot)
    diff_paths = _crop_snapshot_diff_paths(reference_snapshot, other_snapshot)
    if not diff_paths:
        return True
    return diff_paths <= _SOURCE_VIEW_CROP_SIGNATURE_DIFF_PATHS


def _source_lineage_arrays(source: SourceSubjectMaskRun) -> dict[str, object | None]:
    return {
        "frame_indices": source.frame_indices,
        "frame_counts": source.frame_counts,
        "detection_indices": source.detection_indices,
        "source_crop_row_ids": getattr(source, "source_crop_row_ids", None),
        "source_refined_row_ids": source.source_refined_row_ids,
        "source_detect_row_index": source.source_detect_row_index,
        "instance_key": getattr(source, "instance_key", None),
    }


def _validate_source_alignment(reference: SourceSubjectMaskRun, other: SourceSubjectMaskRun) -> str:
    if reference.crop_run != other.crop_run:
        raise ValueError(
            f"Alignment mismatch for source_crop_run: {reference.crop_run!r} != {other.crop_run!r}."
        )
    crop_snapshot_mismatches: list[str] = []
    for field_name in CANONICAL_SOURCE_CROP_SNAPSHOT_ATTRS:
        reference_value = reference.source_crop_snapshot.get(field_name)
        other_value = other.source_crop_snapshot.get(field_name)
        if reference_value != other_value:
            crop_snapshot_mismatches.append(
                f"{field_name}: {reference_value!r} != {other_value!r}"
            )
    if int(reference.masks_roi.shape[0]) != int(other.masks_roi.shape[0]):
        raise ValueError(
            f"Row-count mismatch: {reference.masks_roi.shape[0]} != {other.masks_roi.shape[0]}."
        )
    if tuple(reference.masks_roi.shape[2:]) != tuple(other.masks_roi.shape[2:]):
        raise ValueError(
            f"ROI shape mismatch: {reference.masks_roi.shape[2:]} != {other.masks_roi.shape[2:]}."
        )
    _required_array_equal("detection_source", reference.detection_source, other.detection_source)
    reference_lineage = _source_lineage_arrays(reference)
    other_lineage = _source_lineage_arrays(other)
    row_identity_mode = resolve_row_lineage_identity_mode(
        reference_lineage,
        other_lineage,
        allow_legacy_positional=True,
    )
    assert_row_lineage_sources_equal(
        reference_lineage,
        other_lineage,
        identity_mode=row_identity_mode,
    )
    if crop_snapshot_mismatches and not _is_allowed_source_view_crop_snapshot_mismatch(reference, other):
        raise ValueError(
            "Alignment mismatch for crop snapshot fields: "
            + "; ".join(crop_snapshot_mismatches)
            + "."
        )
    return row_identity_mode


def _shared_value(sources: Sequence[SourceSubjectMaskRun], attr_name: str) -> Optional[str]:
    values = [getattr(source, attr_name) for source in sources if getattr(source, attr_name) is not None]
    if not values:
        return None
    first = str(values[0])
    if all(str(value) == first for value in values[1:]):
        return first
    return None


def _shared_group_attr(sources: Sequence[SourceSubjectMaskRun], attr_name: str) -> Optional[str]:
    values = []
    for source in sources:
        value = source.group.attrs.get(attr_name)
        if value is not None and str(value).strip():
            values.append(str(value))
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values[1:]):
        return first
    return None


def _first_group_attr(sources: Sequence[SourceSubjectMaskRun], attr_name: str) -> Optional[str]:
    for source in sources:
        value = source.group.attrs.get(attr_name)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _component_seed_from_source(source: SourceSubjectMaskRun, component_name: str) -> RefinedSubjectComponentSeed:
    return _build_single_source_component_seeds(source, (component_name,))[component_name]


def _load_refined_subject_mask_source(root: zarr.Group, refined_subject_run: Optional[str]) -> SourceSubjectMaskRun:
    parent = root.get("refined_subject_masks_runs")
    run_name = refined_subject_run or _resolve_latest_parent_run(parent, "refined_subject_masks_runs")
    if parent is None or str(run_name) not in parent:
        raise RuntimeError(f"refined_subject_masks_runs/{run_name} not found.")
    group = parent[run_name]
    try:
        mask_store = open_mask_store(
            group,
            source_path=f"refined_subject_masks_runs/{run_name}",
            prefer="dense",
        )
    except (MaskStoreError, ValueError) as exc:
        raise RuntimeError(
            f"refined_subject_masks_runs/{run_name} missing usable mask store (masks_roi or mask_rle)."
        ) from exc
    dense_masks = group.get("masks_roi")
    masks_arr = dense_masks if dense_masks is not None else _MaskStoreDenseArrayView(mask_store)
    mask_surface_path = mask_store.storage_surface
    if len(tuple(masks_arr.shape)) != 4:
        raise RuntimeError(
            f"refined_subject_masks_runs/{run_name}/masks_roi must have shape (N, C, H, W), got {masks_arr.shape}."
        )
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RuntimeError(f"refined_subject_masks_runs/{run_name} missing usable mask_labels attr.")
    available = group.get("available_channels")
    if available is None:
        raise RuntimeError(f"refined_subject_masks_runs/{run_name} missing available_channels.")
    detection_source = group.get("detection_source")
    if detection_source is None:
        raise RuntimeError(f"refined_subject_masks_runs/{run_name} missing detection_source.")

    crop_run = str(group.attrs.get("source_crop_run") or "")
    if not crop_run:
        crop_run = _resolve_latest_parent_run(root.get("crop_runs"), "crop_runs")
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

    frame_indices = _lineage_array(root, group, crop_run=crop_run, name="frame_indices")
    source_crop_row_ids = resolve_source_crop_row_ids(
        group,
        crop_group,
        total_rois=int(masks_arr.shape[0]),
        frame_indices=frame_indices,
    )

    return SourceSubjectMaskRun(
        run_name=str(run_name),
        group=group,
        crop_run=crop_run,
        source_crop_snapshot=source_crop_snapshot,
        masks_roi=masks_arr,
        detection_source=detection_source,
        mask_labels=tuple(str(item) for item in labels_raw),
        available_channels=np.asarray(available[:], dtype=bool),
        frame_indices=frame_indices,
        frame_counts=_lineage_array(root, group, crop_run=crop_run, name="frame_counts"),
        detection_indices=_lineage_array(root, group, crop_run=crop_run, name="detection_indices"),
        source_method=str(group.attrs.get("method")) if group.attrs.get("method") is not None else None,
        source_keypoints_run=resolve_source_keypoints_run(group.attrs),
        source_keypoint_group=(
            str(group.attrs.get("source_keypoint_group"))
            if group.attrs.get("source_keypoint_group") is not None
            else None
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
        source_refined_row_ids=_lineage_array(root, group, crop_run=crop_run, name="source_refined_row_ids"),
        source_detect_row_index=_lineage_array(root, group, crop_run=crop_run, name="source_detect_row_index"),
        instance_key=_lineage_array(root, group, crop_run=crop_run, name="instance_key"),
        mask_surface_kind="binary",
        mask_surface_path=mask_surface_path,
    )


def _require_available_component(source: SourceSubjectMaskRun, component_name: str, stage_group: str) -> int:
    if component_name not in source.mask_labels:
        raise ValueError(f"{stage_group}/{source.run_name} does not expose {component_name}.")
    comp_idx = source.mask_labels.index(component_name)
    if comp_idx >= int(source.available_channels.shape[0]) or not bool(source.available_channels[comp_idx]):
        raise ValueError(f"{stage_group}/{source.run_name} does not have an available {component_name} channel.")
    return int(comp_idx)


def _has_available_component(source: SourceSubjectMaskRun, component_name: str) -> bool:
    if component_name not in source.mask_labels:
        return False
    comp_idx = source.mask_labels.index(component_name)
    return comp_idx < int(source.available_channels.shape[0]) and bool(source.available_channels[comp_idx])


def _resolve_subject_keypoint_group(
    root: zarr.Group,
    source: SourceSubjectMaskRun,
    *,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> tuple[zarr.Group, str, str, str]:
    if bool(assignment_keypoint_group) != bool(assignment_keypoints_run):
        raise ValueError(
            "Pass both assignment_keypoint_group and assignment_keypoints_run, or neither."
        )
    if assignment_keypoint_group and assignment_keypoints_run:
        parent = root.get(str(assignment_keypoint_group))
        if parent is None or str(assignment_keypoints_run) not in parent:
            raise ValueError(
                f"Explicit assignment keypoint source missing: "
                f"{assignment_keypoint_group}/{assignment_keypoints_run}."
            )
        return (
            parent[str(assignment_keypoints_run)],
            str(assignment_keypoints_run),
            str(assignment_keypoint_group),
            "cli_assignment_keypoint_override",
        )

    assignment_keypoint_run = source.assignment_keypoints_run
    assignment_keypoint_group = source.assignment_keypoint_group
    if assignment_keypoint_group and assignment_keypoint_run:
        parent = root.get(str(assignment_keypoint_group))
        if parent is None or str(assignment_keypoint_run) not in parent:
            raise ValueError(
                f"subject_mask_runs/{source.run_name} references missing assignment keypoint source "
                f"{assignment_keypoint_group}/{assignment_keypoint_run}; cannot assign eyes_union to eye_left/eye_right."
            )
        return (
            parent[str(assignment_keypoint_run)],
            str(assignment_keypoint_run),
            str(assignment_keypoint_group),
            "assignment_keypoint_attrs",
        )

    if assignment_keypoint_group and not assignment_keypoint_run:
        raise ValueError(
            f"subject_mask_runs/{source.run_name} has assignment_keypoint_group but no assignment_keypoints_run; "
            "cannot assign eyes_union to eye_left/eye_right."
        )
    if assignment_keypoint_run and not assignment_keypoint_group:
        raise ValueError(
            f"subject_mask_runs/{source.run_name} has assignment_keypoints_run but no assignment_keypoint_group; "
            "cannot assign eyes_union to eye_left/eye_right."
        )

    source_keypoint_run = source.source_keypoints_run
    source_keypoint_group = source.source_keypoint_group
    if source_keypoint_group and source_keypoint_run:
        parent = root.get(str(source_keypoint_group))
        if parent is None or str(source_keypoint_run) not in parent:
            raise ValueError(
                f"subject_mask_runs/{source.run_name} references missing keypoint source "
                f"{source_keypoint_group}/{source_keypoint_run}; cannot assign eyes_union to eye_left/eye_right."
            )
        return parent[str(source_keypoint_run)], str(source_keypoint_run), str(source_keypoint_group), "source_keypoint_lineage"

    if source_keypoint_group and not source_keypoint_run:
        raise ValueError(
            f"subject_mask_runs/{source.run_name} has source_keypoint_group but no source_keypoints_run; "
            "cannot assign eyes_union to eye_left/eye_right."
        )

    if source_keypoint_run:
        matches: list[tuple[zarr.Group, str, str]] = []
        for parent_name in ("refined_keypoints_runs", "keypoints_runs"):
            parent = root.get(parent_name)
            if parent is not None and str(source_keypoint_run) in parent:
                matches.append((parent[str(source_keypoint_run)], str(source_keypoint_run), parent_name))
        if len(matches) == 1:
            keypoint_group, keypoint_run, group_name = matches[0]
            return keypoint_group, keypoint_run, group_name, "source_keypoint_lineage"
        if len(matches) > 1:
            raise ValueError(
                f"subject_mask_runs/{source.run_name} references keypoint run {source_keypoint_run!r} "
                "in multiple keypoint groups; set source_keypoint_group before assigning eyes_union."
            )
        raise ValueError(
            f"subject_mask_runs/{source.run_name} references missing keypoint run {source_keypoint_run!r}; "
            "cannot assign eyes_union to eye_left/eye_right."
        )

    raise ValueError(
        f"subject_mask_runs/{source.run_name} exposes eyes_union but has no source keypoint lineage; "
        "cannot assign anatomical eye_left/eye_right without keypoints."
    )


def _resolve_keypoint_success_array(kp_group: zarr.Group, keypoint_run_name: str) -> tuple[np.ndarray, str]:
    for candidate in _KEYPOINT_SUCCESS_DATASET_CANDIDATES:
        if candidate in kp_group:
            return np.asarray(kp_group[candidate][:], dtype=bool), candidate
    raise ValueError(
        f"Keypoint run {keypoint_run_name!r} missing success flags "
        f"({', '.join(_KEYPOINT_SUCCESS_DATASET_CANDIDATES)}); cannot assign eyes_union."
    )


def _resolve_eye_keypoint_indices(kp_group: zarr.Group, keypoint_run_name: str) -> tuple[int, int]:
    keypoints_roi = kp_group.get("keypoints_roi")
    if keypoints_roi is None:
        raise ValueError(f"Keypoint run {keypoint_run_name!r} missing keypoints_roi; cannot assign eyes_union.")
    keypoint_count = int(keypoints_roi.shape[1])
    try:
        resolved = resolve_required_keypoint_indices_from_attrs(
            kp_group.attrs,
            _EYE_COMPONENTS,
            keypoint_count=keypoint_count,
        )
    except ValueError as exc:
        raise ValueError(
            f"Keypoint run {keypoint_run_name!r} is missing canonical eye labels required "
            f"for eyes_union assignment: {exc}"
        ) from exc
    return int(resolved["eye_left"]), int(resolved["eye_right"])


def _component_seed_from_eyes_union_assignment(
    source: SourceSubjectMaskRun,
    *,
    component_name: str,
    masks: np.ndarray,
    reason_labels: np.ndarray,
    assignment_summary: Mapping[str, object],
    keypoint_run_name: str,
    keypoint_group_name: str,
    keypoint_success_dataset: str,
    keypoint_source_kind: str,
) -> RefinedSubjectComponentSeed:
    source_payload: dict[str, object] = {
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
        **source.source_crop_snapshot,
    }
    label_schema_id = source.group.attrs.get("label_schema_id")
    if label_schema_id is not None:
        source_payload["source_label_schema_id"] = str(label_schema_id)
    created_at = source.group.attrs.get("created_at_utc") or source.group.attrs.get("created_utc")
    if created_at is not None:
        source_payload["source_created_at_utc"] = str(created_at)
    if source.mask_surface_kind == "thresholded_probability":
        comp_idx = source.mask_labels.index(_RAW_EYE_UNION_COMPONENT)
        threshold = (
            float(source.probability_thresholds[comp_idx])
            if comp_idx < len(source.probability_thresholds)
            else 0.5
        )
        source_payload["source_probability_path"] = (
            f"subject_mask_runs/{source.run_name}/{source.mask_surface_path}"
        )
        source_payload["source_probability_encoding"] = str(source.probability_encoding or "")
        source_payload["source_binary_derivation"] = "threshold(mask_probs_roi)"
        source_payload["source_probability_threshold"] = float(threshold)
    return RefinedSubjectComponentSeed(
        component_name=str(component_name),
        masks=np.asarray(masks, dtype=np.uint8),
        source_payload=source_payload,
        reason_labels=np.asarray(reason_labels, dtype=object),
        source_masks=np.asarray(masks, dtype=np.uint8),
    )


def _assign_eyes_union_component_seeds(
    root: zarr.Group,
    source: SourceSubjectMaskRun,
    *,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> tuple[dict[str, RefinedSubjectComponentSeed], dict[str, object]]:
    comp_idx = _require_available_component(source, _RAW_EYE_UNION_COMPONENT, "subject_mask_runs")
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
    reconcile_keypoint_mask_row_identity(
        keypoint_source_crop_row_ids=kp_group.get("source_crop_row_ids"),
        mask_source_crop_row_ids=source.group.get("source_crop_row_ids"),
        expected_rows=int(source.masks_roi.shape[0]),
        keypoint_run_name=keypoint_run_name,
        keypoint_group_name=keypoint_group_name,
        mask_run_name=source.run_name,
        mask_group_name="subject_mask_runs",
    )
    assignment = assign_eyes_union_to_lr(
        np.asarray(source.masks_roi[:, comp_idx], dtype=np.uint8),
        keypoints_roi=np.asarray(keypoints_roi[:], dtype=np.float32),
        keypoint_success=keypoint_success,
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
    seeds = {
        component_name: _component_seed_from_eyes_union_assignment(
            source,
            component_name=component_name,
            masks=np.asarray(assignment.masks[component_name], dtype=np.uint8),
            reason_labels=np.asarray(assignment.reason_labels[component_name], dtype=object),
            assignment_summary=summary,
            keypoint_run_name=keypoint_run_name,
            keypoint_group_name=keypoint_group_name,
            keypoint_success_dataset=success_dataset,
            keypoint_source_kind=keypoint_source_kind,
        )
        for component_name in _EYE_COMPONENTS
    }
    return seeds, summary


def _component_seed_from_refined_subject_source(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> RefinedSubjectComponentSeed:
    comp_idx = _require_available_component(source, component_name, "refined_subject_masks_runs")
    source_payload: dict[str, object] = {
        "source_stage": "refined_subject_masks_runs",
        "source_run": source.run_name,
        "source_method": source.source_method or source.group.attrs.get("method") or "unknown",
        "source_channels": [component_name],
        "source_crop_run": source.crop_run,
        "source_mask_surface_path": source.mask_surface_path,
        **source.source_crop_snapshot,
    }
    label_schema_id = source.group.attrs.get("label_schema_id")
    if label_schema_id is not None:
        source_payload["source_label_schema_id"] = str(label_schema_id)
    source_review = _component_review_from_refined_source(source, component_name)
    if source_review is not None:
        source_payload["source_review_status"] = source_review
    source_created_at = source.group.attrs.get("created_at_utc") or source.group.attrs.get("created_utc")
    if source_created_at is not None:
        source_payload["source_created_at_utc"] = str(source_created_at)

    upstream_provenance = _get_component_provenance_group(source.group, component_name)
    if upstream_provenance is not None:
        source_payload["upstream_component_provenance"] = _json_safe(dict(upstream_provenance.attrs))

    return RefinedSubjectComponentSeed(
        component_name=component_name,
        masks=np.asarray(source.masks_roi[:, comp_idx], dtype=np.uint8),
        source_payload=source_payload,
    )


def _component_review_from_refined_source(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> dict[str, object] | None:
    reviews = source.group.attrs.get("component_review_statuses")
    if not isinstance(reviews, Mapping):
        return None
    payload = reviews.get(component_name)
    if not isinstance(payload, Mapping):
        return None
    return dict(_json_safe(payload))


def _approved_component_review_from_refined_source(
    source: SourceSubjectMaskRun,
    component_name: str,
    *,
    allow_unapproved_components: bool,
    promote_source_review: bool,
) -> dict[str, object] | None:
    review_payload = _component_review_from_refined_source(source, component_name)
    state = ""
    if review_payload is not None:
        state = str(review_payload.get("state") or "").strip().lower()
    if allow_unapproved_components:
        return review_payload if promote_source_review and state == "approved" else None
    if state != "approved":
        run_name = source.run_name
        display_state = state or "missing"
        raise ValueError(
            f"refined_subject_masks_runs/{run_name} component {component_name!r} is not approved "
            f"(review state: {display_state}). Pass allow_unapproved_components=True "
            "or --allow-unapproved-components for draft/QA assembly."
        )
    return review_payload if promote_source_review else None


def _add_component_seed(
    seeds: dict[str, RefinedSubjectComponentSeed],
    review_overrides: dict[str, dict[str, object]],
    component_name: str,
    seed: RefinedSubjectComponentSeed,
    *,
    review_payload: dict[str, object] | None = None,
) -> None:
    if component_name in seeds:
        existing = seeds[component_name].source_payload
        existing_stage = existing.get("source_stage")
        existing_run = existing.get("source_run")
        new_stage = seed.source_payload.get("source_stage")
        new_run = seed.source_payload.get("source_run")
        raise ValueError(
            f"Duplicate source for component {component_name!r}: "
            f"{existing_stage}/{existing_run} and {new_stage}/{new_run}."
        )
    seeds[component_name] = seed
    if review_payload is not None:
        review_overrides[component_name] = review_payload


def _collect_component_seeds(
    *,
    root: zarr.Group,
    subject_source: Optional[SourceSubjectMaskRun],
    body_source: Optional[SourceSubjectMaskRun],
    body_refined_source: Optional[SourceSubjectMaskRun],
    eye_source: Optional[SourceSubjectMaskRun],
    eye_refined_source: Optional[SourceSubjectMaskRun],
    swim_source: Optional[SourceSubjectMaskRun],
    swim_refined_source: Optional[SourceSubjectMaskRun],
    refined_component_sources: Mapping[str, SourceSubjectMaskRun],
    allow_unapproved_components: bool = False,
    promote_source_review: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> tuple[dict[str, RefinedSubjectComponentSeed], list[str], dict[str, dict[str, object]], Optional[dict[str, object]]]:
    seeds: dict[str, RefinedSubjectComponentSeed] = {}
    review_overrides: dict[str, dict[str, object]] = {}
    eyes_union_assignment_summary: Optional[dict[str, object]] = None

    if subject_source is not None:
        for component_name in CANONICAL_COMPONENT_ORDER:
            if not _has_available_component(subject_source, component_name):
                continue
            _add_component_seed(
                seeds,
                review_overrides,
                component_name,
                _component_seed_from_source(subject_source, component_name),
            )

    if body_source is not None:
        _require_available_component(body_source, "subject_body", "subject_mask_runs")
        _add_component_seed(
            seeds,
            review_overrides,
            "subject_body",
            _component_seed_from_source(body_source, "subject_body"),
        )

    if body_refined_source is not None:
        _add_component_seed(
            seeds,
            review_overrides,
            "subject_body",
            _component_seed_from_refined_subject_source(body_refined_source, "subject_body"),
            review_payload=_approved_component_review_from_refined_source(
                body_refined_source,
                "subject_body",
                allow_unapproved_components=allow_unapproved_components,
                promote_source_review=promote_source_review,
            ),
        )

    if eye_source is not None:
        eye_components: list[str] = []
        for component_name in ("eye_left", "eye_right"):
            if component_name in eye_source.mask_labels:
                comp_idx = eye_source.mask_labels.index(component_name)
                if comp_idx < int(eye_source.available_channels.shape[0]) and bool(
                    eye_source.available_channels[comp_idx]
                ):
                    _add_component_seed(
                        seeds,
                        review_overrides,
                        component_name,
                        _component_seed_from_source(eye_source, component_name),
                    )
                    eye_components.append(component_name)
        if not eye_components:
            raise ValueError(
                f"subject_mask_runs/{eye_source.run_name} does not have available eye_left/eye_right channels."
            )

    if eye_refined_source is not None:
        for component_name in _EYE_COMPONENTS:
            _add_component_seed(
                seeds,
                review_overrides,
                component_name,
                _component_seed_from_refined_subject_source(eye_refined_source, component_name),
                review_payload=_approved_component_review_from_refined_source(
                    eye_refined_source,
                    component_name,
                    allow_unapproved_components=allow_unapproved_components,
                    promote_source_review=promote_source_review,
                ),
            )

    if swim_source is not None:
        _require_available_component(swim_source, "swim_bladder", "subject_mask_runs")
        _add_component_seed(
            seeds,
            review_overrides,
            "swim_bladder",
            _component_seed_from_source(swim_source, "swim_bladder"),
        )

    if swim_refined_source is not None:
        _add_component_seed(
            seeds,
            review_overrides,
            "swim_bladder",
            _component_seed_from_refined_subject_source(swim_refined_source, "swim_bladder"),
            review_payload=_approved_component_review_from_refined_source(
                swim_refined_source,
                "swim_bladder",
                allow_unapproved_components=allow_unapproved_components,
                promote_source_review=promote_source_review,
            ),
        )

    for component_name, source in refined_component_sources.items():
        _add_component_seed(
            seeds,
            review_overrides,
            component_name,
            _component_seed_from_refined_subject_source(source, component_name),
            review_payload=_approved_component_review_from_refined_source(
                source,
                component_name,
                allow_unapproved_components=allow_unapproved_components,
                promote_source_review=promote_source_review,
            ),
        )

    if subject_source is not None and _has_available_component(subject_source, _RAW_EYE_UNION_COMPONENT):
        existing_eye_components = set(seeds).intersection(_EYE_COMPONENTS)
        if not existing_eye_components:
            assignment_seeds, eyes_union_assignment_summary = _assign_eyes_union_component_seeds(
                root,
                subject_source,
                assignment_keypoint_group=assignment_keypoint_group,
                assignment_keypoints_run=assignment_keypoints_run,
            )
            for component_name in _EYE_COMPONENTS:
                _add_component_seed(
                    seeds,
                    review_overrides,
                    component_name,
                    assignment_seeds[component_name],
                )
        elif not set(_EYE_COMPONENTS).issubset(existing_eye_components):
            raise ValueError(
                f"subject_mask_runs/{subject_source.run_name} exposes available eyes_union but only partial "
                "eye_left/eye_right seeds are present. Provide a complete LR seed pair or let eyes_union "
                "assignment create both components."
            )

    if (
        subject_source is not None
        and _has_available_component(subject_source, _RAW_EYE_UNION_COMPONENT)
        and not set(_EYE_COMPONENTS).issubset(seeds)
    ):
        raise ValueError(
            f"subject_mask_runs/{subject_source.run_name} exposes available eyes_union but no complete "
            "eye_left/eye_right refined seed and assignment did not produce both components. Provide "
            "eye_left/eye_right via --eye-run or --eye-refined-run, or fix "
            "source keypoint lineage so eyes_union assignment can run."
        )

    component_names = [name for name in CANONICAL_COMPONENT_ORDER if name in seeds]
    if not component_names:
        raise ValueError("At least one source run is required to assemble refined subject masks.")
    return seeds, component_names, review_overrides, eyes_union_assignment_summary


def _apply_component_review_overrides(
    refined_group: zarr.Group,
    *,
    component_names: Sequence[str],
    review_overrides: Mapping[str, Mapping[str, object]],
) -> None:
    if not review_overrides:
        return
    component_reviews = dict(refined_group.attrs.get("component_review_statuses") or {})
    for component_name in component_names:
        override = review_overrides.get(component_name)
        if override is not None:
            component_reviews[str(component_name)] = dict(_json_safe(override))
    refined_group.attrs["component_review_statuses"] = component_reviews
    refined_group.attrs["refined_subject_mask_review_status"] = _review_payload(
        state=_aggregate_run_state(component_reviews),
        method="assembly",
        intended_use="training",
        notes="auto_aggregated_from_component_review_statuses_after_assembly",
    )


def _parse_refined_component_runs(values: Optional[Sequence[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values or ():
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Expected COMPONENT=RUN for refined component source, got {text!r}.")
        raw_component, raw_run = text.split("=", 1)
        component_name = _normalize_component_name(raw_component)
        run_name = raw_run.strip()
        if component_name is None or not run_name:
            raise ValueError(f"Expected COMPONENT=RUN for refined component source, got {text!r}.")
        if component_name in result:
            raise ValueError(f"Duplicate refined component source for {component_name!r}.")
        result[component_name] = run_name
    return result


def assemble_refined_subject_run(
    root: zarr.Group,
    *,
    subject_run: Optional[str] = None,
    body_run: Optional[str] = None,
    body_refined_run: Optional[str] = None,
    eye_run: Optional[str] = None,
    eye_refined_run: Optional[str] = None,
    swim_run: Optional[str] = None,
    swim_refined_run: Optional[str] = None,
    refined_component_runs: Optional[Mapping[str, str]] = None,
    refined_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_unapproved_components: bool = False,
    promote_source_review: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> dict[str, object]:
    if bool(assignment_keypoint_group) != bool(assignment_keypoints_run):
        raise ValueError("Pass both assignment_keypoint_group and assignment_keypoints_run, or neither.")
    if assignment_keypoint_group and assignment_keypoint_group not in _KEYPOINT_GROUP_CHOICES:
        raise ValueError(
            f"Unsupported assignment_keypoint_group {assignment_keypoint_group!r}; "
            f"expected one of {_KEYPOINT_GROUP_CHOICES}."
        )
    if eye_refined_run and eye_run:
        raise ValueError("Pass only one of eye_run or eye_refined_run.")
    subject_source = _load_source_subject_mask_run(root, subject_run) if subject_run else None
    body_source = _load_source_subject_mask_run(root, body_run) if body_run else None
    body_refined_source = _load_refined_subject_mask_source(root, body_refined_run) if body_refined_run else None
    eye_source = _load_source_subject_mask_run(root, eye_run) if eye_run else None
    eye_refined_source = _load_refined_subject_mask_source(root, eye_refined_run) if eye_refined_run else None
    swim_source = _load_source_subject_mask_run(root, swim_run) if swim_run else None
    swim_refined_source = _load_refined_subject_mask_source(root, swim_refined_run) if swim_refined_run else None
    refined_component_runs = dict(refined_component_runs or {})
    refined_component_sources = {
        component_name: _load_refined_subject_mask_source(root, run_name)
        for component_name, run_name in refined_component_runs.items()
    }

    provided_sources = [
        source
        for source in (
            subject_source,
            body_source,
            body_refined_source,
            eye_source,
            eye_refined_source,
            swim_source,
            swim_refined_source,
            *refined_component_sources.values(),
        )
        if source is not None
    ]
    if not provided_sources:
        raise ValueError(
            "At least one of subject_run, body_run, body_refined_run, eye_run, "
            "eye_refined_run, swim_run, swim_refined_run, or refined_component_runs is required."
        )

    reference_source = subject_source or body_source or eye_source or swim_source
    if reference_source is None:
        reference_source = body_refined_source or eye_refined_source or swim_refined_source
    if reference_source is None and refined_component_sources:
        reference_source = next(iter(refined_component_sources.values()))
    assert reference_source is not None
    reference_lineage = _source_lineage_arrays(reference_source)
    row_identity_mode = resolve_row_lineage_identity_mode(
        reference_lineage,
        reference_lineage,
        allow_legacy_positional=True,
    )
    for source in provided_sources:
        if source is reference_source:
            continue
        source_identity_mode = _validate_source_alignment(reference_source, source)
        if source_identity_mode != row_identity_mode:
            raise ValueError(
                "Alignment mismatch: component sources resolved different row identity modes "
                f"({row_identity_mode!r} != {source_identity_mode!r})."
            )

    component_seeds, component_names, review_overrides, eyes_union_assignment_summary = _collect_component_seeds(
        root=root,
        subject_source=subject_source,
        body_source=body_source,
        body_refined_source=body_refined_source,
        eye_source=eye_source,
        eye_refined_source=eye_refined_source,
        swim_source=swim_source,
        swim_refined_source=swim_refined_source,
        refined_component_sources=refined_component_sources,
        allow_unapproved_components=allow_unapproved_components,
        promote_source_review=promote_source_review,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    coarse_subject_source = subject_source or body_source or eye_source or swim_source
    coarse_source_subject_mask_run = (
        coarse_subject_source.run_name
        if coarse_subject_source is not None
        else (
            _shared_group_attr(provided_sources, "source_subject_mask_run")
            or _first_group_attr(provided_sources, "source_subject_mask_run")
        )
    )
    coarse_source_subject_mask_method = (
        coarse_subject_source.source_method
        if coarse_subject_source is not None
        else (
            _shared_group_attr(provided_sources, "source_subject_mask_method")
            or _first_group_attr(provided_sources, "source_subject_mask_method")
        )
    )
    target_run = str(refined_run or _default_refined_run_name())
    refined_parent = root.get("refined_subject_masks_runs")
    target_exists = refined_parent is not None and target_run in refined_parent

    source_component_runs = {
        component_name: str(component_seeds[component_name].source_payload.get("source_run") or "")
        for component_name in component_names
    }
    source_component_sources = {
        component_name: {
            "source_stage": str(component_seeds[component_name].source_payload.get("source_stage") or ""),
            "source_run": str(component_seeds[component_name].source_payload.get("source_run") or ""),
        }
        for component_name in component_names
    }
    summary = {
        "status": "planned" if dry_run else "updated",
        "refined_run": target_run,
        "refined_run_exists": bool(target_exists),
        "would_create_refined_run": not bool(target_exists),
        "mutates_archive": not bool(dry_run),
        "component_names": list(component_names),
        "source_body_subject_mask_run": body_source.run_name if body_source is not None else None,
        "source_body_refined_subject_mask_run": (
            body_refined_source.run_name if body_refined_source is not None else None
        ),
        "source_eye_subject_mask_run": eye_source.run_name if eye_source is not None else None,
        "source_eye_refined_subject_mask_run": eye_refined_source.run_name if eye_refined_source is not None else None,
        "source_swim_subject_mask_run": swim_source.run_name if swim_source is not None else None,
        "source_swim_refined_subject_mask_run": (
            swim_refined_source.run_name if swim_refined_source is not None else None
        ),
        "source_refined_subject_component_runs": dict(refined_component_runs),
        "source_subject_mask_run": coarse_source_subject_mask_run,
        "source_component_runs": source_component_runs,
        "source_subject_mask_runs": source_component_runs,
        "source_component_sources": source_component_sources,
        "source_crop_run": reference_source.crop_run,
        "row_identity_mode": row_identity_mode,
        "row_identity_mode_schema": ROW_IDENTITY_MODE_SCHEMA,
        "promote_source_review": bool(promote_source_review),
        "source_input_subject_mask_run": subject_source.run_name if subject_source is not None else None,
        "eyes_union_assignment_summary": eyes_union_assignment_summary,
        **reference_source.source_crop_snapshot,
        "roi_count": int(reference_source.masks_roi.shape[0]),
        "label_schema_id": _infer_refined_label_schema_id(component_names),
    }
    assignment_keypoint_attrs: dict[str, object] = {}
    if eyes_union_assignment_summary is not None:
        assignment_keypoint_attrs = build_assignment_keypoint_attrs(
            eyes_union_assignment_summary["keypoint_run"],
            assignment_keypoint_group=eyes_union_assignment_summary["keypoint_group"],
            selection=str(eyes_union_assignment_summary.get("keypoint_source_kind") or "unknown"),
        )
        assignment_keypoint_attrs["assignment_keypoint_success_dataset"] = str(
            eyes_union_assignment_summary["keypoint_success_dataset"]
        )
        summary.update(assignment_keypoint_attrs)
    if dry_run:
        return summary

    refined_parent = require_runs_parent(root, "refined_subject_masks_runs")
    if target_run in refined_parent:
        if not overwrite:
            raise ValueError(
                f"refined_subject_masks_runs/{target_run} already exists. Pass overwrite=True to replace it."
            )
        del refined_parent[target_run]

    primary_component = (
        "subject_body"
        if "subject_body" in component_names
        else ("eye_left" if "eye_left" in component_names else "swim_bladder")
    )
    if subject_source is not None and len(provided_sources) == 1 and not refined_component_runs:
        assembly_semantics = "single_source_subject_run_seed"
    elif subject_source is not None:
        assembly_semantics = "subject_run_plus_component_seed"
    else:
        assembly_semantics = "multi_source_component_seed"

    extra_attrs = {
        "assembly_semantics": assembly_semantics,
        "assembly_primary_source_component": primary_component,
        "row_identity_mode": row_identity_mode,
        "row_identity_mode_schema": ROW_IDENTITY_MODE_SCHEMA,
        "source_component_runs": source_component_runs,
        "source_subject_mask_runs": source_component_runs,
        "source_component_sources": source_component_sources,
    }
    if subject_source is not None:
        extra_attrs["source_input_subject_mask_run"] = subject_source.run_name
    if eyes_union_assignment_summary is not None:
        extra_attrs["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        extra_attrs.update(assignment_keypoint_attrs)
    if body_source is not None:
        extra_attrs["source_body_subject_mask_run"] = body_source.run_name
    if body_refined_source is not None:
        extra_attrs["source_body_refined_subject_mask_run"] = body_refined_source.run_name
    if eye_source is not None:
        extra_attrs["source_eye_subject_mask_run"] = eye_source.run_name
    if eye_refined_source is not None:
        extra_attrs["source_eye_refined_subject_mask_run"] = eye_refined_source.run_name
    if swim_source is not None:
        extra_attrs["source_swim_subject_mask_run"] = swim_source.run_name
    if swim_refined_source is not None:
        extra_attrs["source_swim_refined_subject_mask_run"] = swim_refined_source.run_name
    if refined_component_runs:
        extra_attrs["source_refined_subject_component_runs"] = dict(refined_component_runs)

    shared_keypoints_run = _shared_value(provided_sources, "source_keypoints_run")
    shared_keypoint_group = _shared_value(provided_sources, "source_keypoint_group")
    provenance_inputs = {
        "assembly_semantics": assembly_semantics,
        "row_identity_mode": row_identity_mode,
        "row_identity_mode_schema": ROW_IDENTITY_MODE_SCHEMA,
        "source_component_runs": source_component_runs,
        "source_subject_mask_runs": source_component_runs,
        "source_component_sources": source_component_sources,
    }
    if subject_source is not None:
        provenance_inputs["source_input_subject_mask_run"] = subject_source.run_name
    if eyes_union_assignment_summary is not None:
        provenance_inputs["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        provenance_inputs.update(assignment_keypoint_attrs)
    if body_source is not None:
        provenance_inputs["source_body_subject_mask_run"] = body_source.run_name
    if body_refined_source is not None:
        provenance_inputs["source_body_refined_subject_mask_run"] = body_refined_source.run_name
    if eye_source is not None:
        provenance_inputs["source_eye_subject_mask_run"] = eye_source.run_name
    if eye_refined_source is not None:
        provenance_inputs["source_eye_refined_subject_mask_run"] = eye_refined_source.run_name
    if swim_source is not None:
        provenance_inputs["source_swim_subject_mask_run"] = swim_source.run_name
    if swim_refined_source is not None:
        provenance_inputs["source_swim_refined_subject_mask_run"] = swim_refined_source.run_name
    if refined_component_runs:
        provenance_inputs["source_refined_subject_component_runs"] = dict(refined_component_runs)

    refined = _create_refined_subject_run_from_component_seeds(
        refined_parent=refined_parent,
        target_run=target_run,
        reference_source=reference_source,
        component_names=component_names,
        component_seeds=component_seeds,
        coarse_source_subject_mask_run=coarse_source_subject_mask_run,
        coarse_source_subject_mask_method=coarse_source_subject_mask_method,
        source_keypoints_run=shared_keypoints_run,
        source_keypoint_group=shared_keypoint_group,
        run_method=ASSEMBLE_REFINED_SUBJECT_METHOD,
        stage_command=" ".join(sys.argv) if sys.argv else "unknown",
        extra_attrs=extra_attrs,
        provenance_inputs=provenance_inputs,
    )
    _apply_component_review_overrides(
        refined.group,
        component_names=component_names,
        review_overrides=review_overrides,
    )
    return summary


def assemble_refined_subject_masks(
    zarr_path: str | Path,
    *,
    subject_run: Optional[str] = None,
    body_run: Optional[str] = None,
    body_refined_run: Optional[str] = None,
    eye_run: Optional[str] = None,
    eye_refined_run: Optional[str] = None,
    swim_run: Optional[str] = None,
    swim_refined_run: Optional[str] = None,
    refined_component_runs: Optional[Mapping[str, str]] = None,
    refined_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_unapproved_components: bool = False,
    promote_source_review: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    summary = assemble_refined_subject_run(
        root,
        subject_run=subject_run,
        body_run=body_run,
        body_refined_run=body_refined_run,
        eye_run=eye_run,
        eye_refined_run=eye_refined_run,
        swim_run=swim_run,
        swim_refined_run=swim_refined_run,
        refined_component_runs=refined_component_runs,
        refined_run=refined_run,
        overwrite=overwrite,
        dry_run=dry_run,
        allow_unapproved_components=allow_unapproved_components,
        promote_source_review=promote_source_review,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    summary["zarr_path"] = str(Path(zarr_path))
    if not dry_run:
        refined_parent = root.get("refined_subject_masks_runs")
        resolved_run = str(summary.get("refined_run") or "")
        if refined_parent is not None and resolved_run in refined_parent:
            emit_refined_subject_mask_stage_completion(
                root,
                zarr_path,
                run_group=refined_parent[resolved_run],
                run_name=resolved_run,
                source=_REFINED_SUBJECT_MASKS_STATUS_SOURCE,
                console=None,
                invalidate_on_ok=True,
            )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the Palette zarr archive.")
    parser.add_argument(
        "--subject-run",
        help=(
            "subject_mask_runs/<run> providing one raw multi-component source. "
            "All available canonical components are copied as refined seeds."
        ),
    )
    parser.add_argument("--body-run", help="subject_mask_runs/<run> providing subject_body.")
    parser.add_argument("--body-refined-run", help="refined_subject_masks_runs/<run> providing subject_body.")
    parser.add_argument("--eye-run", help="subject_mask_runs/<run> providing eye_left/eye_right.")
    parser.add_argument("--eye-refined-run", help="refined_subject_masks_runs/<run> providing eye_left/eye_right.")
    parser.add_argument("--swim-run", help="subject_mask_runs/<run> providing swim_bladder.")
    parser.add_argument("--swim-refined-run", help="refined_subject_masks_runs/<run> providing swim_bladder.")
    parser.add_argument(
        "--refined-component-source",
        action="append",
        default=[],
        metavar="COMPONENT=RUN",
        help="Use one refined_subject_masks_runs/<run> source for a specific component; repeatable.",
    )
    parser.add_argument(
        "--refined-run",
        "--run-name",
        dest="refined_run",
        help="Target refined_subject_masks_runs/<run>.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing refined run of the same name.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the assembly without mutating the archive.")
    parser.add_argument(
        "--allow-unapproved-components",
        action="store_true",
        help="Allow draft/QA assembly from pending or missing refined_subject_masks_runs component approvals.",
    )
    parser.add_argument(
        "--promote-source-review",
        action="store_true",
        help=(
            "Explicitly copy approved upstream review payloads onto the assembled refined run. "
            "By default assembly finalizes metrics/QC but leaves target component reviews pending."
        ),
    )
    parser.add_argument(
        "--assignment-keypoint-group",
        choices=_KEYPOINT_GROUP_CHOICES,
        help="Explicit keypoint group to use for eyes_union -> eye_left/eye_right assignment.",
    )
    parser.add_argument(
        "--assignment-keypoint-run",
        help="Explicit keypoint run to use for eyes_union -> eye_left/eye_right assignment.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the result summary as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        refined_component_runs = _parse_refined_component_runs(args.refined_component_source)
    except ValueError as exc:
        parser.error(str(exc))
    summary = assemble_refined_subject_masks(
        args.zarr_path,
        subject_run=args.subject_run,
        body_run=args.body_run,
        body_refined_run=args.body_refined_run,
        eye_run=args.eye_run,
        eye_refined_run=args.eye_refined_run,
        swim_run=args.swim_run,
        swim_refined_run=args.swim_refined_run,
        refined_component_runs=refined_component_runs,
        refined_run=args.refined_run,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        allow_unapproved_components=bool(args.allow_unapproved_components),
        promote_source_review=bool(args.promote_source_review),
        assignment_keypoint_group=args.assignment_keypoint_group,
        assignment_keypoints_run=args.assignment_keypoint_run,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
