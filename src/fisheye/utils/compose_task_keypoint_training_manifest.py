#!/usr/bin/env python3
"""Compose a task-specific keypoint merge manifest from reviewed source artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.training_leakage_groups import resolve_training_leakage_group
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    refined_keypoint_source_bindings_from_manifest,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.utils.export_keypoint_training_zarr import _resolve_roi_pixel_contract


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _reviewed_dataset_entry(archive: Path) -> tuple[dict[str, Any], str]:
    root = open_zarr_group_direct(archive, mode="r")
    publication = root.attrs.get("reviewed_keypoint_training_artifact")
    if not isinstance(publication, Mapping):
        raise ValueError(f"Reviewed artifact manifest is missing: {archive}")
    if (
        publication.get("schema_id") != "palette.reviewed_keypoint_training_artifact"
        or publication.get("schema_version") != 1
        or root.attrs.get("training_task") != "keypoints"
        or root.attrs.get("training_artifact_status")
        != "reviewed_keypoint_immutable_candidate"
        or root.attrs.get("artifact_mutability") != "immutable_snapshot"
        or root.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            f"Reviewed artifact publication envelope is invalid: {archive}"
        )
    payload = publication.get("payload")
    included = (
        payload.get("included_run_paths") if isinstance(payload, Mapping) else None
    )
    if not isinstance(included, Mapping) or set(included) != {
        "crop",
        "raw_keypoints",
        "keypoint_quality",
        "refined_keypoints",
    }:
        raise ValueError(
            "Reviewed artifact does not contain the exact keypoint source roles."
        )

    crop_path = str(included["crop"])
    raw_path = str(included["raw_keypoints"])
    refined_path = str(included["refined_keypoints"])
    crop = root[crop_path]
    refined = root[refined_path]
    source_bindings_raw = refined.attrs.get("source_bindings")
    if not isinstance(source_bindings_raw, Mapping):
        raise ValueError(
            "Reviewed refined keypoints are missing strict-v2 source bindings."
        )
    bindings = refined_keypoint_source_bindings_from_manifest(source_bindings_raw)
    if (
        raw_path != f"keypoints_runs/{bindings.raw_run_id}"
        or crop_path != f"crop_runs/{bindings.crop_run_id}"
        or str(included["keypoint_quality"])
        != f"keypoint_quality_runs/{bindings.quality_run_id}"
    ):
        raise ValueError(
            "Reviewed artifact roles do not match refined source bindings."
        )

    refined_run = refined_path.split("/", 1)[1]
    roi_contract, roi_contract_name = _resolve_roi_pixel_contract(crop)
    if roi_contract_name is None:
        raise ValueError("Reviewed artifact crop pixel contract cannot be resolved.")
    semantics = dict(bindings.skeleton_semantics)
    labels = list(semantics["keypoint_labels"])
    usable = np.asarray(refined["usable_keypoints"][:], dtype=np.bool_)
    sample_count = int(refined["keypoints_roi"].shape[0])
    if usable.shape != (sample_count,):
        raise ValueError("Reviewed usable_keypoints does not match keypoint row count.")
    review_derivation = refined.attrs.get("review_derivation")
    if not isinstance(review_derivation, Mapping):
        raise ValueError(
            "Reviewed refined keypoints are missing manual-review derivation."
        )

    recording_id = str(root.attrs.get("recording_id") or root.attrs.get("session_uuid"))
    leakage_group_id, leakage_group_source = resolve_training_leakage_group(
        recording_id=recording_id,
        subject_ids=(),
        started_utc=None,
    )
    dataset_id = (
        f"{recording_id}:reviewed_keypoints:{publication['payload_digest'][:12]}"
    )
    entry = {
        "name": str(root.attrs.get("recording_name") or archive.stem),
        "dataset_id": dataset_id,
        "session_uuid": str(root.attrs.get("session_uuid") or recording_id),
        "recording_id": recording_id,
        "leakage_group": {
            "id": leakage_group_id,
            "source": leakage_group_source,
            "subject_ids": [],
            "recording_started_utc": None,
        },
        "zarr_path": str(archive),
        "rig_id": root.attrs.get("rig_id"),
        "dish_design": root.attrs.get("dish_design"),
        "canvas_name": root.attrs.get("canvas_name"),
        "source_type_requested": "refined",
        "source_type_resolved": "refined",
        "source_crop_run": bindings.crop_run_id,
        "source_crop_storage_mode": "materialized",
        "source_roi_image_representation": "uint8_grayscale_roi_v1",
        "source_roi_pixel_contract": roi_contract,
        "source_roi_pixel_contract_name": roi_contract_name,
        "required_roi_pixel_contract_name": roi_contract_name,
        "source_roi_read_mode": "materialized_crop_run",
        "input_format": "gray",
        "keypoint_run_requested": bindings.raw_run_id,
        "keypoint_run_selector": "explicit_reviewed_snapshot",
        "keypoint_run_resolved": bindings.raw_run_id,
        "quality_registry_used": False,
        "quality_registry_refined_run": refined_run,
        "annotation_source_kind": "refined",
        "annotation_source_parent": "refined_keypoints_runs",
        "annotation_source_run": refined_run,
        "refined_keypoint_run": refined_run,
        "keypoints_array_path": f"{refined_path}/keypoints_roi",
        "detection_success_path": f"{refined_path}/usable_keypoints",
        "keypoints_total": sample_count,
        "keypoints_successful": int(usable.sum()),
        "keypoints_success_rate": float(usable.mean()) if sample_count else 0.0,
        "usable_keypoints_total": int(usable.sum()),
        "usable_keypoints_rate": float(usable.mean()) if sample_count else 0.0,
        "skeleton_id": bindings.skeleton_id,
        "kpt_shape": [len(labels), 3],
        "keypoint_labels": labels,
        "skeleton_signature": (
            f"skeleton_id={bindings.skeleton_id}, kpt_shape=[{len(labels)},3]"
        ),
        "keypoint_review_status": {
            "state": "approved",
            "method": "manual_web_review_compaction",
            "intended_use": "training",
            "generation_sha256": review_derivation.get("generation_sha256"),
            "overlay_sha256": review_derivation.get("overlay_sha256"),
        },
        "reviewed_artifact_receipt_digest": publication.get("payload_digest"),
        "warnings": [],
    }
    return entry, roi_contract_name


def _census_dataset_entry(source: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    source_origin = str(source.get("source_origin") or "")
    archive = Path(str(source["zarr_path"])).expanduser().resolve()
    if source_origin == "reviewed_artifact_candidate":
        entry, contract_name = _reviewed_dataset_entry(archive)
        entry["leakage_group"] = dict(source["split_group"])
        return entry, contract_name

    if source_origin != "registry_approved":
        raise ValueError(
            f"Unsupported census source origin {source_origin!r}: {archive}"
        )
    if source.get("safe_for_pose_merge") is not True or source.get("problems") != []:
        raise ValueError(f"Census source is not safe for pose merge: {archive}")
    root = open_zarr_group_direct(archive, mode="r")
    crop_run = str(source["source_crop_run"])
    refined_run = str(source["refined_run"])
    raw_run = str(source["source_keypoint_run"])
    crop_path = f"crop_runs/{crop_run}"
    refined_path = f"refined_keypoints_runs/{refined_run}"
    if crop_path not in root or refined_path not in root:
        raise ValueError(f"Census source run paths changed after census: {archive}")
    crop = root[crop_path]
    refined = root[refined_path]
    keypoints = refined["keypoints_roi"]
    usable = np.asarray(refined["usable_keypoints"][:], dtype=np.bool_)
    shape = tuple(int(value) for value in keypoints.shape)
    expected_shape = tuple(int(value) for value in source["keypoints_shape"])
    if shape != expected_shape or usable.shape != (shape[0],):
        raise ValueError(
            f"Census source keypoint surface changed after census: {archive}"
        )
    roi_contract, roi_contract_name = _resolve_roi_pixel_contract(crop)
    expected_contract = str(source["pixel_contract"]["resolved_name"])
    if roi_contract_name != expected_contract:
        raise ValueError(
            f"Census source pixel contract changed ({roi_contract_name!r} != "
            f"{expected_contract!r}): {archive}"
        )
    source_type = str(crop.attrs.get("detection_source_type") or "").strip().lower()
    if source_type not in {"manual", "refined"}:
        raise ValueError(
            f"Census source has unsupported crop detection_source_type "
            f"{source_type or 'missing'!r}: {archive}"
        )
    labels = [str(value) for value in source["keypoint_labels"]]
    review = dict(source["review"])
    recording_id = str(source["recording_id"])
    entry = {
        "name": archive.stem,
        "dataset_id": str(source["dataset_id"]),
        "session_uuid": recording_id,
        "recording_id": recording_id,
        "leakage_group": dict(source["split_group"]),
        "zarr_path": str(archive),
        "source_type_requested": "refined",
        "source_type_resolved": source_type,
        "source_crop_run": crop_run,
        "source_crop_storage_mode": source.get("crop_storage_mode"),
        "source_roi_image_representation": source["pixel_contract"].get(
            "image_representation"
        ),
        "source_roi_pixel_contract": roi_contract,
        "source_roi_pixel_contract_name": roi_contract_name,
        "required_roi_pixel_contract_name": roi_contract_name,
        "source_roi_read_mode": "materialized_crop_run",
        "input_format": "gray",
        "keypoint_run_requested": raw_run,
        "keypoint_run_selector": "frozen_census_reviewed_run",
        "keypoint_run_resolved": raw_run,
        "quality_registry_used": True,
        "quality_registry_refined_run": refined_run,
        "annotation_source_kind": "refined",
        "annotation_source_parent": "refined_keypoints_runs",
        "annotation_source_run": refined_run,
        "refined_keypoint_run": refined_run,
        "keypoints_array_path": f"{refined_path}/keypoints_roi",
        "detection_success_path": f"{refined_path}/usable_keypoints",
        "keypoints_total": int(shape[0]),
        "keypoints_successful": int(usable.sum()),
        "keypoints_success_rate": float(usable.mean()) if usable.size else 0.0,
        "usable_keypoints_total": int(usable.sum()),
        "usable_keypoints_rate": float(usable.mean()) if usable.size else 0.0,
        "skeleton_id": str(source["skeleton_id"]),
        "kpt_shape": [int(shape[1]), 3],
        "keypoint_labels": labels,
        "skeleton_signature": (
            f"skeleton_id={source['skeleton_id']}, kpt_shape=[{shape[1]},3]"
        ),
        "keypoint_review_status": {
            "state": review.get("state"),
            "intended_use": review.get("intended_use"),
            "method": review.get("method"),
            "reviewer": review.get("reviewer"),
            "timestamp_utc": review.get("timestamp_utc"),
        },
        "warnings": [],
    }
    return entry, roi_contract_name


def compose_manifest_from_census(
    *,
    census_path: Path,
    set_id: str,
    set_name: str,
    set_version: str,
) -> dict[str, Any]:
    census = _load_object(census_path)
    if (
        census.get("schema_id") != "palette.keypoint_training_source_census"
        or census.get("schema_version") != 1
    ):
        raise ValueError("Unsupported keypoint source census schema.")
    selection = census.get("selection")
    summary = census.get("summary")
    sources = census.get("sources")
    if not isinstance(selection, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("Keypoint source census is missing selection or summary.")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Keypoint source census contains no sources.")
    datasets: list[dict[str, Any]] = []
    contract_counts: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError("Keypoint source census contains a non-object source.")
        entry, contract_name = _census_dataset_entry(source)
        datasets.append(entry)
        contract_counts[contract_name] = contract_counts.get(contract_name, 0) + 1

    expected_source_count = int(summary.get("source_count") or -1)
    expected_usable = int(summary.get("usable_pose_rows") or -1)
    observed_usable = sum(
        int(dataset["usable_keypoints_total"]) for dataset in datasets
    )
    if len(datasets) != expected_source_count or observed_usable != expected_usable:
        raise ValueError(
            "Census composition changed while building the source manifest "
            f"(sources={len(datasets)}/{expected_source_count}, "
            f"usable={observed_usable}/{expected_usable})."
        )
    labels = [str(value) for value in selection["expected_keypoint_labels"]]
    skeleton_id = str(selection["skeleton_id"])
    payload: dict[str, Any] = {
        "set_id": str(set_id),
        "set_name": str(set_name),
        "set_version": str(set_version),
        "task": "pose",
        "source_type": "refined",
        "source_type_requested": "refined",
        "input_format": "gray",
        "pose_schema": {
            "skeleton_id": skeleton_id,
            "kpt_shape": [len(labels), 3],
            "keypoint_labels": labels,
            "skeleton_signature": (
                f"skeleton_id={skeleton_id}, kpt_shape=[{len(labels)},3]"
            ),
        },
        "keypoint_labels": labels,
        "datasets": datasets,
        "required_roi_pixel_contract_name": None,
        "keypoint_contract_policy": {
            "schema_version": "palette.keypoint_roi_pixel_contract_policy.v2",
            "status": "explicit_output_canvas_transform",
            "strict_missing_contracts": True,
            "mixed_contracts_require_explicit_compatibility": True,
            "compatible_contracts": sorted(contract_counts),
            "explicit_contracts": sorted(contract_counts),
            "required_roi_pixel_contract_name": None,
            "contract_counts": dict(sorted(contract_counts.items())),
            "compatibility_basis": "zero_pad_without_resize_to_explicit_output_canvas_v1",
        },
        "task_specific_merge": {
            "schema_id": "palette.task_specific_keypoint_training_merge",
            "schema_version": 2,
            "task": "keypoints",
            "source_census_path": str(census_path.expanduser().resolve()),
            "source_census_sha256": canonical_json_sha256(census),
            "source_composition_sha256": summary.get("source_composition_sha256"),
            "mask_dependencies": [],
            "registry_activation": "deferred",
        },
    }
    payload["task_specific_merge"]["composition_digest"] = canonical_json_sha256(
        {
            "set_id": payload["set_id"],
            "set_version": payload["set_version"],
            "dataset_ids": [dataset["dataset_id"] for dataset in datasets],
            "zarr_paths": [dataset["zarr_path"] for dataset in datasets],
            "contract_counts": contract_counts,
            "leakage_groups": [dataset["leakage_group"] for dataset in datasets],
            "source_composition_sha256": summary.get("source_composition_sha256"),
        }
    )
    return payload


def compose_manifest(
    *,
    base_manifest_paths: Sequence[Path],
    reviewed_artifact: Path,
    set_id: str,
    set_name: str,
    set_version: str,
) -> dict[str, Any]:
    if not base_manifest_paths:
        raise ValueError("At least one base manifest is required.")
    bases = [_load_object(path) for path in base_manifest_paths]
    first = bases[0]
    expected_pose_schema = first.get("pose_schema")
    if not isinstance(expected_pose_schema, Mapping):
        raise ValueError("Base manifest is missing pose_schema.")
    expected_skeleton = expected_pose_schema.get("skeleton_id")
    datasets: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    contract_counts: dict[str, int] = {}
    for path, base in zip(base_manifest_paths, bases):
        pose_schema = base.get("pose_schema")
        if (
            not isinstance(pose_schema, Mapping)
            or pose_schema.get("skeleton_id") != expected_skeleton
        ):
            raise ValueError(f"Base manifest skeleton differs: {path}")
        for raw_dataset in base.get("datasets", []):
            if not isinstance(raw_dataset, Mapping):
                raise TypeError(f"Base manifest contains a non-object dataset: {path}")
            dataset = dict(raw_dataset)
            zarr_path = str(dataset.get("zarr_path"))
            if zarr_path in seen_paths:
                raise ValueError(
                    f"Duplicate source Zarr across base manifests: {zarr_path}"
                )
            seen_paths.add(zarr_path)
            datasets.append(dataset)
            contract_name = str(dataset.get("source_roi_pixel_contract_name"))
            contract_counts[contract_name] = contract_counts.get(contract_name, 0) + 1

    reviewed_entry, reviewed_contract = _reviewed_dataset_entry(reviewed_artifact)
    if reviewed_entry["skeleton_id"] != expected_skeleton:
        raise ValueError("Reviewed artifact skeleton differs from the base manifests.")
    reviewed_path = str(reviewed_entry["zarr_path"])
    if reviewed_path in seen_paths:
        raise ValueError("Reviewed artifact is already present in a base manifest.")
    datasets.append(reviewed_entry)
    contract_counts[reviewed_contract] = contract_counts.get(reviewed_contract, 0) + 1

    payload = dict(first)
    payload.update(
        {
            "set_id": str(set_id),
            "set_name": str(set_name),
            "set_version": str(set_version),
            "datasets": datasets,
            "required_roi_pixel_contract_name": None,
            "keypoint_contract_policy": {
                "schema_version": "palette.keypoint_roi_pixel_contract_policy.v2",
                "status": "explicit_output_canvas_transform",
                "strict_missing_contracts": True,
                "mixed_contracts_require_explicit_compatibility": True,
                "compatible_contracts": sorted(contract_counts),
                "explicit_contracts": sorted(contract_counts),
                "required_roi_pixel_contract_name": None,
                "contract_counts": dict(sorted(contract_counts.items())),
                "compatibility_basis": "zero_pad_without_resize_to_explicit_output_canvas_v1",
            },
            "task_specific_merge": {
                "schema_id": "palette.task_specific_keypoint_training_merge",
                "schema_version": 2,
                "task": "keypoints",
                "source_manifest_paths": [
                    str(path.resolve()) for path in base_manifest_paths
                ],
                "source_manifest_sha256": {
                    str(path.resolve()): canonical_json_sha256(base)
                    for path, base in zip(base_manifest_paths, bases)
                },
                "reviewed_artifact": str(reviewed_artifact.resolve()),
                "reviewed_artifact_receipt_digest": reviewed_entry[
                    "reviewed_artifact_receipt_digest"
                ],
                "mask_dependencies": [],
                "registry_activation": "deferred",
            },
        }
    )
    payload["task_specific_merge"]["composition_digest"] = canonical_json_sha256(
        {
            "set_id": payload["set_id"],
            "set_version": payload["set_version"],
            "dataset_ids": [dataset["dataset_id"] for dataset in datasets],
            "zarr_paths": [dataset["zarr_path"] for dataset in datasets],
            "contract_counts": contract_counts,
            "leakage_groups": [dataset.get("leakage_group") for dataset in datasets],
        }
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--base-manifest", type=Path, action="append")
    source_group.add_argument("--source-census", type=Path)
    parser.add_argument("--reviewed-artifact", type=Path)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--set-name", required=True)
    parser.add_argument("--set-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.source_census is not None:
        if args.reviewed_artifact is not None:
            raise ValueError(
                "--source-census already binds reviewed artifacts; omit --reviewed-artifact."
            )
        payload = compose_manifest_from_census(
            census_path=args.source_census.expanduser().resolve(),
            set_id=args.set_id,
            set_name=args.set_name,
            set_version=args.set_version,
        )
    else:
        if args.reviewed_artifact is None:
            raise ValueError("--base-manifest mode requires --reviewed-artifact.")
        payload = compose_manifest(
            base_manifest_paths=args.base_manifest,
            reviewed_artifact=args.reviewed_artifact.expanduser().resolve(),
            set_id=args.set_id,
            set_name=args.set_name,
            set_version=args.set_version,
        )
    write_json_atomic(args.output.expanduser().resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "dataset_count": len(payload["datasets"]),
                "composition_digest": payload["task_specific_merge"][
                    "composition_digest"
                ],
                "registry_activation": "deferred",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
