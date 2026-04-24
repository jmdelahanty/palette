"""Assemble multi-source refined subject-mask runs and finalize them immediately."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..shared.provenance_attrs import (
    CANONICAL_SOURCE_CROP_SNAPSHOT_ATTRS,
    build_source_crop_snapshot_attrs,
    extract_source_crop_snapshot_attrs,
    resolve_source_keypoints_run,
)
from ..shared.row_lineage import assert_row_lineage_sources_equal
from ..shared.subject_mask_registry_status import emit_refined_subject_mask_stage_completion
from ..tune.refined_subject_mask_review import (
    RefinedSubjectComponentSeed,
    SourceSubjectMaskRun,
    _aggregate_run_state,
    _build_single_source_component_seeds,
    _create_refined_subject_run_from_component_seeds,
    _default_refined_run_name,
    _get_component_provenance_group,
    _infer_refined_label_schema_id,
    _load_refined_eye_mask_source,
    _load_source_subject_mask_run,
    _normalize_component_name,
    _open_existing_refined_subject_run,
    _review_payload,
)
from ..utils.zarr_io import open_zarr_root

ASSEMBLE_REFINED_SUBJECT_METHOD = "assemble_refined_subject_masks_v1"
CANONICAL_COMPONENT_ORDER = ("subject_body", "eye_left", "eye_right", "swim_bladder")
_REFINED_SUBJECT_MASKS_STATUS_SOURCE = "runtime_assemble_refined_subject_masks"
_EYE_COMPONENTS = ("eye_left", "eye_right")


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


def _source_lineage_arrays(source: SourceSubjectMaskRun) -> dict[str, object | None]:
    return {
        "frame_indices": source.frame_indices,
        "frame_counts": source.frame_counts,
        "detection_indices": source.detection_indices,
        "source_refined_row_ids": source.source_refined_row_ids,
        "source_detect_row_index": source.source_detect_row_index,
    }


def _validate_source_alignment(reference: SourceSubjectMaskRun, other: SourceSubjectMaskRun) -> None:
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
    if crop_snapshot_mismatches:
        raise ValueError(
            "Alignment mismatch for crop snapshot fields: "
            + "; ".join(crop_snapshot_mismatches)
            + "."
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
    assert_row_lineage_sources_equal(_source_lineage_arrays(reference), _source_lineage_arrays(other))


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
    _open_existing_refined_subject_run(root, run_name)
    assert parent is not None
    group = parent[run_name]
    masks_arr = group.get("masks_roi")
    if masks_arr is None:
        raise RuntimeError(f"refined_subject_masks_runs/{run_name} missing masks_roi.")
    if len(masks_arr.shape) != 4:
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

    return SourceSubjectMaskRun(
        run_name=str(run_name),
        group=group,
        crop_run=crop_run,
        source_crop_snapshot=source_crop_snapshot,
        masks_roi=masks_arr,
        detection_source=detection_source,
        mask_labels=tuple(str(item) for item in labels_raw),
        available_channels=np.asarray(available[:], dtype=bool),
        frame_indices=_lineage_array(root, group, crop_run=crop_run, name="frame_indices"),
        frame_counts=_lineage_array(root, group, crop_run=crop_run, name="frame_counts"),
        detection_indices=_lineage_array(root, group, crop_run=crop_run, name="detection_indices"),
        source_method=str(group.attrs.get("method")) if group.attrs.get("method") is not None else None,
        source_keypoints_run=resolve_source_keypoints_run(group.attrs),
        source_keypoint_group=(
            str(group.attrs.get("source_keypoint_group"))
            if group.attrs.get("source_keypoint_group") is not None
            else None
        ),
        source_refined_row_ids=_lineage_array(root, group, crop_run=crop_run, name="source_refined_row_ids"),
        source_detect_row_index=_lineage_array(root, group, crop_run=crop_run, name="source_detect_row_index"),
    )


def _require_available_component(source: SourceSubjectMaskRun, component_name: str, stage_group: str) -> int:
    if component_name not in source.mask_labels:
        raise ValueError(f"{stage_group}/{source.run_name} does not expose {component_name}.")
    comp_idx = source.mask_labels.index(component_name)
    if comp_idx >= int(source.available_channels.shape[0]) or not bool(source.available_channels[comp_idx]):
        raise ValueError(f"{stage_group}/{source.run_name} does not have an available {component_name} channel.")
    return int(comp_idx)


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
        **source.source_crop_snapshot,
    }
    label_schema_id = source.group.attrs.get("label_schema_id")
    if label_schema_id is not None:
        source_payload["source_label_schema_id"] = str(label_schema_id)
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


def _component_seed_from_refined_eye_source(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> RefinedSubjectComponentSeed:
    if component_name not in _EYE_COMPONENTS:
        raise ValueError(f"Unsupported refined-eye component {component_name!r}.")
    comp_idx = source.mask_labels.index(component_name)
    source_payload: dict[str, object] = {
        "source_stage": "refined_eye_masks_runs",
        "source_run": source.run_name,
        "source_method": source.source_method or source.group.attrs.get("method") or "unknown",
        "source_channels": [component_name],
        "source_crop_run": source.crop_run,
        **source.source_crop_snapshot,
    }
    raw_eye_run = source.group.attrs.get("source_eye_masks_run")
    if raw_eye_run is not None:
        source_payload["source_eye_masks_run"] = str(raw_eye_run)
    source_created_at = source.group.attrs.get("created_at_utc") or source.group.attrs.get("created_utc")
    if source_created_at is not None:
        source_payload["source_created_at_utc"] = str(source_created_at)
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
) -> dict[str, object] | None:
    review_payload = _component_review_from_refined_source(source, component_name)
    if allow_unapproved_components:
        return review_payload
    state = ""
    if review_payload is not None:
        state = str(review_payload.get("state") or "").strip().lower()
    if state != "approved":
        run_name = source.run_name
        display_state = state or "missing"
        raise ValueError(
            f"refined_subject_masks_runs/{run_name} component {component_name!r} is not approved "
            f"(review state: {display_state}). Pass allow_unapproved_components=True "
            "or --allow-unapproved-components for draft/QA assembly."
        )
    return review_payload


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
    body_source: Optional[SourceSubjectMaskRun],
    body_refined_source: Optional[SourceSubjectMaskRun],
    eye_source: Optional[SourceSubjectMaskRun],
    refined_eye_source: Optional[SourceSubjectMaskRun],
    eye_refined_source: Optional[SourceSubjectMaskRun],
    swim_source: Optional[SourceSubjectMaskRun],
    swim_refined_source: Optional[SourceSubjectMaskRun],
    refined_component_sources: Mapping[str, SourceSubjectMaskRun],
    allow_unapproved_components: bool = False,
) -> tuple[dict[str, RefinedSubjectComponentSeed], list[str], dict[str, dict[str, object]]]:
    seeds: dict[str, RefinedSubjectComponentSeed] = {}
    review_overrides: dict[str, dict[str, object]] = {}

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

    if refined_eye_source is not None:
        for component_name in _EYE_COMPONENTS:
            _add_component_seed(
                seeds,
                review_overrides,
                component_name,
                _component_seed_from_refined_eye_source(refined_eye_source, component_name),
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
            ),
        )

    component_names = [name for name in CANONICAL_COMPONENT_ORDER if name in seeds]
    if not component_names:
        raise ValueError("At least one source run is required to assemble refined subject masks.")
    return seeds, component_names, review_overrides


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
    body_run: Optional[str] = None,
    body_refined_run: Optional[str] = None,
    eye_run: Optional[str] = None,
    refined_eye_run: Optional[str] = None,
    eye_refined_run: Optional[str] = None,
    swim_run: Optional[str] = None,
    swim_refined_run: Optional[str] = None,
    refined_component_runs: Optional[Mapping[str, str]] = None,
    refined_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_unapproved_components: bool = False,
) -> dict[str, object]:
    if eye_run and refined_eye_run:
        raise ValueError("Pass only one of eye_run or refined_eye_run.")
    if eye_refined_run and (eye_run or refined_eye_run):
        raise ValueError("Pass only one of eye_run, refined_eye_run, or eye_refined_run.")
    body_source = _load_source_subject_mask_run(root, body_run) if body_run else None
    body_refined_source = _load_refined_subject_mask_source(root, body_refined_run) if body_refined_run else None
    eye_source = _load_source_subject_mask_run(root, eye_run) if eye_run else None
    refined_eye_source = _load_refined_eye_mask_source(root, refined_eye_run) if refined_eye_run else None
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
            body_source,
            body_refined_source,
            eye_source,
            refined_eye_source,
            eye_refined_source,
            swim_source,
            swim_refined_source,
            *refined_component_sources.values(),
        )
        if source is not None
    ]
    if not provided_sources:
        raise ValueError(
            "At least one of body_run, body_refined_run, eye_run, refined_eye_run, "
            "eye_refined_run, swim_run, swim_refined_run, or refined_component_runs is required."
        )

    reference_source = body_source or eye_source or refined_eye_source or swim_source
    if reference_source is None:
        reference_source = body_refined_source or eye_refined_source or swim_refined_source
    if reference_source is None and refined_component_sources:
        reference_source = next(iter(refined_component_sources.values()))
    assert reference_source is not None
    for source in provided_sources:
        if source is reference_source:
            continue
        _validate_source_alignment(reference_source, source)

    component_seeds, component_names, review_overrides = _collect_component_seeds(
        body_source=body_source,
        body_refined_source=body_refined_source,
        eye_source=eye_source,
        refined_eye_source=refined_eye_source,
        eye_refined_source=eye_refined_source,
        swim_source=swim_source,
        swim_refined_source=swim_refined_source,
        refined_component_sources=refined_component_sources,
        allow_unapproved_components=allow_unapproved_components,
    )
    coarse_subject_source = body_source or eye_source or swim_source
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
        "source_refined_eye_masks_run": refined_eye_source.run_name if refined_eye_source is not None else None,
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
        **reference_source.source_crop_snapshot,
        "roi_count": int(reference_source.masks_roi.shape[0]),
        "label_schema_id": _infer_refined_label_schema_id(component_names),
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

    primary_component = (
        "subject_body"
        if "subject_body" in component_names
        else ("eye_left" if "eye_left" in component_names else "swim_bladder")
    )
    extra_attrs = {
        "assembly_semantics": "multi_source_component_seed",
        "assembly_primary_source_component": primary_component,
        "source_component_runs": source_component_runs,
        "source_subject_mask_runs": source_component_runs,
        "source_component_sources": source_component_sources,
    }
    if body_source is not None:
        extra_attrs["source_body_subject_mask_run"] = body_source.run_name
    if body_refined_source is not None:
        extra_attrs["source_body_refined_subject_mask_run"] = body_refined_source.run_name
    if eye_source is not None:
        extra_attrs["source_eye_subject_mask_run"] = eye_source.run_name
    if refined_eye_source is not None:
        extra_attrs["source_refined_eye_masks_run"] = refined_eye_source.run_name
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
        "assembly_semantics": "multi_source_component_seed",
        "source_component_runs": source_component_runs,
        "source_subject_mask_runs": source_component_runs,
        "source_component_sources": source_component_sources,
    }
    if body_source is not None:
        provenance_inputs["source_body_subject_mask_run"] = body_source.run_name
    if body_refined_source is not None:
        provenance_inputs["source_body_refined_subject_mask_run"] = body_refined_source.run_name
    if eye_source is not None:
        provenance_inputs["source_eye_subject_mask_run"] = eye_source.run_name
    if refined_eye_source is not None:
        provenance_inputs["source_refined_eye_masks_run"] = refined_eye_source.run_name
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
    body_run: Optional[str] = None,
    body_refined_run: Optional[str] = None,
    eye_run: Optional[str] = None,
    refined_eye_run: Optional[str] = None,
    eye_refined_run: Optional[str] = None,
    swim_run: Optional[str] = None,
    swim_refined_run: Optional[str] = None,
    refined_component_runs: Optional[Mapping[str, str]] = None,
    refined_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    allow_unapproved_components: bool = False,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    summary = assemble_refined_subject_run(
        root,
        body_run=body_run,
        body_refined_run=body_refined_run,
        eye_run=eye_run,
        refined_eye_run=refined_eye_run,
        eye_refined_run=eye_refined_run,
        swim_run=swim_run,
        swim_refined_run=swim_refined_run,
        refined_component_runs=refined_component_runs,
        refined_run=refined_run,
        overwrite=overwrite,
        dry_run=dry_run,
        allow_unapproved_components=allow_unapproved_components,
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
    parser.add_argument("--body-run", help="subject_mask_runs/<run> providing subject_body.")
    parser.add_argument("--body-refined-run", help="refined_subject_masks_runs/<run> providing subject_body.")
    parser.add_argument("--eye-run", help="subject_mask_runs/<run> providing eye_left/eye_right.")
    parser.add_argument("--refined-eye-run", help="refined_eye_masks_runs/<run> providing eye_left/eye_right.")
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
        body_run=args.body_run,
        body_refined_run=args.body_refined_run,
        eye_run=args.eye_run,
        refined_eye_run=args.refined_eye_run,
        eye_refined_run=args.eye_refined_run,
        swim_run=args.swim_run,
        swim_refined_run=args.swim_refined_run,
        refined_component_runs=refined_component_runs,
        refined_run=args.refined_run,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        allow_unapproved_components=bool(args.allow_unapproved_components),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
