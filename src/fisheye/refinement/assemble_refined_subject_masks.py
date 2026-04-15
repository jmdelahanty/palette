"""Assemble multi-source refined subject-mask runs and finalize them immediately."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from ..shared.provenance_attrs import CANONICAL_SOURCE_CROP_SNAPSHOT_ATTRS
from ..shared.subject_mask_registry_status import emit_refined_subject_mask_stage_completion
from ..tune.refined_subject_mask_review import (
    RefinedSubjectComponentSeed,
    SourceSubjectMaskRun,
    _build_single_source_component_seeds,
    _create_refined_subject_run_from_component_seeds,
    _default_refined_run_name,
    _infer_refined_label_schema_id,
    _load_source_subject_mask_run,
)
from ..utils.zarr_io import open_zarr_root

ASSEMBLE_REFINED_SUBJECT_METHOD = "assemble_refined_subject_masks_v1"
CANONICAL_COMPONENT_ORDER = ("subject_body", "eye_left", "eye_right", "swim_bladder")
_REFINED_SUBJECT_MASKS_STATUS_SOURCE = "runtime_assemble_refined_subject_masks"


def _required_array_equal(name: str, left: Any, right: Any) -> None:
    left_arr = np.asarray(left[:])
    right_arr = np.asarray(right[:])
    if left_arr.shape != right_arr.shape or not np.array_equal(left_arr, right_arr):
        raise ValueError(f"Alignment mismatch for {name}.")


def _optional_array_equal(name: str, left: Any | None, right: Any | None) -> None:
    if left is None and right is None:
        return
    if left is None or right is None:
        raise ValueError(f"Alignment mismatch for optional {name}: one source is missing it.")
    _required_array_equal(name, left, right)


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
    _optional_array_equal("frame_indices", reference.frame_indices, other.frame_indices)
    _optional_array_equal("frame_counts", reference.frame_counts, other.frame_counts)
    _optional_array_equal("detection_indices", reference.detection_indices, other.detection_indices)


def _shared_value(sources: Sequence[SourceSubjectMaskRun], attr_name: str) -> Optional[str]:
    values = [getattr(source, attr_name) for source in sources if getattr(source, attr_name) is not None]
    if not values:
        return None
    first = str(values[0])
    if all(str(value) == first for value in values[1:]):
        return first
    return None


def _component_seed_from_source(source: SourceSubjectMaskRun, component_name: str) -> RefinedSubjectComponentSeed:
    return _build_single_source_component_seeds(source, (component_name,))[component_name]


def _collect_component_seeds(
    *,
    body_source: Optional[SourceSubjectMaskRun],
    eye_source: Optional[SourceSubjectMaskRun],
    swim_source: Optional[SourceSubjectMaskRun],
) -> tuple[dict[str, RefinedSubjectComponentSeed], list[str]]:
    seeds: dict[str, RefinedSubjectComponentSeed] = {}

    if body_source is not None:
        if "subject_body" not in body_source.mask_labels:
            raise ValueError(f"subject_mask_runs/{body_source.run_name} does not expose subject_body.")
        body_idx = body_source.mask_labels.index("subject_body")
        if body_idx >= int(body_source.available_channels.shape[0]) or not bool(body_source.available_channels[body_idx]):
            raise ValueError(f"subject_mask_runs/{body_source.run_name} does not have an available subject_body channel.")
        seeds["subject_body"] = _component_seed_from_source(body_source, "subject_body")

    if eye_source is not None:
        eye_components: list[str] = []
        for component_name in ("eye_left", "eye_right"):
            if component_name in eye_source.mask_labels:
                comp_idx = eye_source.mask_labels.index(component_name)
                if comp_idx < int(eye_source.available_channels.shape[0]) and bool(eye_source.available_channels[comp_idx]):
                    seeds[component_name] = _component_seed_from_source(eye_source, component_name)
                    eye_components.append(component_name)
        if not eye_components:
            raise ValueError(
                f"subject_mask_runs/{eye_source.run_name} does not have available eye_left/eye_right channels."
            )

    if swim_source is not None:
        if "swim_bladder" not in swim_source.mask_labels:
            raise ValueError(f"subject_mask_runs/{swim_source.run_name} does not expose swim_bladder.")
        swim_idx = swim_source.mask_labels.index("swim_bladder")
        if swim_idx >= int(swim_source.available_channels.shape[0]) or not bool(swim_source.available_channels[swim_idx]):
            raise ValueError(f"subject_mask_runs/{swim_source.run_name} does not have an available swim_bladder channel.")
        seeds["swim_bladder"] = _component_seed_from_source(swim_source, "swim_bladder")

    component_names = [name for name in CANONICAL_COMPONENT_ORDER if name in seeds]
    if not component_names:
        raise ValueError("At least one source run is required to assemble refined subject masks.")
    return seeds, component_names


def assemble_refined_subject_run(
    root: zarr.Group,
    *,
    body_run: Optional[str] = None,
    eye_run: Optional[str] = None,
    swim_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    body_source = _load_source_subject_mask_run(root, body_run) if body_run else None
    eye_source = _load_source_subject_mask_run(root, eye_run) if eye_run else None
    swim_source = _load_source_subject_mask_run(root, swim_run) if swim_run else None

    provided_sources = [source for source in (body_source, eye_source, swim_source) if source is not None]
    if not provided_sources:
        raise ValueError("At least one of body_run, eye_run, or swim_run is required.")

    reference_source = body_source or eye_source or swim_source
    assert reference_source is not None
    for source in provided_sources[1:]:
        _validate_source_alignment(reference_source, source)

    component_seeds, component_names = _collect_component_seeds(
        body_source=body_source,
        eye_source=eye_source,
        swim_source=swim_source,
    )
    target_run = str(refined_run or _default_refined_run_name())
    refined_parent = root.get("refined_subject_masks_runs")
    target_exists = refined_parent is not None and target_run in refined_parent

    source_component_runs = {
        component_name: str(component_seeds[component_name].source_payload.get("source_run") or "")
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
        "source_eye_subject_mask_run": eye_source.run_name if eye_source is not None else None,
        "source_swim_subject_mask_run": swim_source.run_name if swim_source is not None else None,
        "source_subject_mask_run": reference_source.run_name,
        "source_subject_mask_runs": source_component_runs,
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

    primary_component = "subject_body" if body_source is not None else ("eye_left" if eye_source is not None else "swim_bladder")
    extra_attrs = {
        "assembly_semantics": "multi_source_component_seed",
        "assembly_primary_source_component": primary_component,
        "source_subject_mask_runs": source_component_runs,
    }
    if body_source is not None:
        extra_attrs["source_body_subject_mask_run"] = body_source.run_name
    if eye_source is not None:
        extra_attrs["source_eye_subject_mask_run"] = eye_source.run_name
    if swim_source is not None:
        extra_attrs["source_swim_subject_mask_run"] = swim_source.run_name

    shared_keypoints_run = _shared_value(provided_sources, "source_keypoints_run")
    shared_keypoint_group = _shared_value(provided_sources, "source_keypoint_group")
    provenance_inputs = {
        "assembly_semantics": "multi_source_component_seed",
        "source_subject_mask_runs": source_component_runs,
    }
    if body_source is not None:
        provenance_inputs["source_body_subject_mask_run"] = body_source.run_name
    if eye_source is not None:
        provenance_inputs["source_eye_subject_mask_run"] = eye_source.run_name
    if swim_source is not None:
        provenance_inputs["source_swim_subject_mask_run"] = swim_source.run_name

    _create_refined_subject_run_from_component_seeds(
        refined_parent=refined_parent,
        target_run=target_run,
        reference_source=reference_source,
        component_names=component_names,
        component_seeds=component_seeds,
        coarse_source_subject_mask_run=reference_source.run_name,
        coarse_source_subject_mask_method=reference_source.source_method,
        source_keypoints_run=shared_keypoints_run,
        source_keypoint_group=shared_keypoint_group,
        run_method=ASSEMBLE_REFINED_SUBJECT_METHOD,
        stage_command=" ".join(sys.argv) if sys.argv else "unknown",
        extra_attrs=extra_attrs,
        provenance_inputs=provenance_inputs,
    )
    return summary


def assemble_refined_subject_masks(
    zarr_path: str | Path,
    *,
    body_run: Optional[str] = None,
    eye_run: Optional[str] = None,
    swim_run: Optional[str] = None,
    refined_run: Optional[str] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    summary = assemble_refined_subject_run(
        root,
        body_run=body_run,
        eye_run=eye_run,
        swim_run=swim_run,
        refined_run=refined_run,
        overwrite=overwrite,
        dry_run=dry_run,
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
    parser.add_argument("--eye-run", help="subject_mask_runs/<run> providing eye_left/eye_right.")
    parser.add_argument("--swim-run", help="subject_mask_runs/<run> providing swim_bladder.")
    parser.add_argument("--refined-run", "--run-name", dest="refined_run", help="Target refined_subject_masks_runs/<run>.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing refined run of the same name.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the assembly without mutating the archive.")
    parser.add_argument("--json", action="store_true", help="Emit the result summary as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = assemble_refined_subject_masks(
        args.zarr_path,
        body_run=args.body_run,
        eye_run=args.eye_run,
        swim_run=args.swim_run,
        refined_run=args.refined_run,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
