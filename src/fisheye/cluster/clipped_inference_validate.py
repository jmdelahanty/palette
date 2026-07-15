"""Independently validate exact outputs from a clipped inference DAG target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.cluster.clipped_inference import PLAN_SCHEMA
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.cluster.whole_recording_analysis_validate import (
    _require_complete_run,
    _validate_raw_masks,
    _validate_refined_masks,
)
from fisheye.shared.flat_roi_cache import load_flat_roi_cache_manifest
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.utils.validate_refined_subject_mask_contract import (
    validate_refined_subject_mask_contract,
)


REPORT_SCHEMA = "palette.clipped_inference_target_validation.v1"


def _read_plan_target(plan_path: Path, target_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported clipped inference plan: {plan_path}")
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Clipped inference plan has no targets list.")
    exact = [row for row in targets if isinstance(row, Mapping) and str(row.get("target_id")) == target_id]
    if len(exact) != 1:
        raise ValueError(f"Expected exactly one plan target {target_id!r}; found {len(exact)}.")
    return payload, exact[0]


def _instance_keys(run: Any, *, label: str) -> dict[str, Any]:
    array = run.get("instance_key")
    if array is None:
        raise RuntimeError(f"Modern production run is missing instance_key: {label}")
    values = np.asarray(array[:], dtype=np.uint64).reshape(-1)
    if values.size == 0:
        raise RuntimeError(f"Modern production run has no instance_key rows: {label}")
    unique = int(np.unique(values).size)
    if unique != values.size:
        raise RuntimeError(
            f"Modern production run has duplicate instance_key values: {label} "
            f"({unique}/{values.size} unique)."
        )
    return {
        "row_count": int(values.size),
        "unique_count": unique,
        "dtype": str(values.dtype),
    }


def _refined_instance_keys(run: Any, *, label: str) -> dict[str, Any]:
    instances = run.get("instances")
    if instances is None:
        raise RuntimeError(f"Modern refined detection run is missing instances: {label}")
    return _instance_keys(instances, label=f"{label}/instances")


def _cache_manifest_report(path: Path, *, zarr_path: Path, collection_id: str, clip_id: str) -> dict[str, Any]:
    manifest = load_flat_roi_cache_manifest(path)
    if not bool(manifest.get("cache_complete")):
        raise RuntimeError(f"Cache manifest is not complete: {path}")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError(f"Cache manifest has no source object: {path}")
    source_archive = Path(str(source.get("archive_path") or "")).expanduser().resolve()
    if source_archive != zarr_path:
        raise RuntimeError(f"Cache archive mismatch for {clip_id}: {source_archive} != {zarr_path}")
    if str(source.get("collection_id") or "") != collection_id:
        raise RuntimeError(f"Cache collection mismatch for {clip_id}.")
    selection = source.get("selection")
    selected_clips = (
        [str(value) for value in selection.get("clip_ids", [])]
        if isinstance(selection, Mapping)
        else []
    )
    if selected_clips != [clip_id]:
        raise RuntimeError(
            f"Cache clip selection mismatch for {clip_id}: {selected_clips!r}."
        )
    array = manifest.get("array")
    if not isinstance(array, Mapping):
        raise RuntimeError(f"Cache manifest has no array object: {path}")
    shape = array.get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or int(shape[0]) <= 0:
        raise RuntimeError(f"Cache shape is invalid for {clip_id}: {shape!r}")
    bin_path = Path(str(array.get("bin_path") or ""))
    if not bin_path.is_absolute():
        bin_path = path.parent / bin_path
    row_index = manifest.get("row_index")
    if not isinstance(row_index, Mapping):
        raise RuntimeError(f"Cache manifest has no row_index object: {path}")
    row_path = Path(str(row_index.get("path") or ""))
    if not row_path.is_absolute():
        row_path = path.parent / row_path
    if not bin_path.is_file() or not row_path.is_file():
        raise RuntimeError(f"Cache payload or row index is missing for {clip_id}.")
    if int(row_index.get("row_count") or -1) != int(shape[0]):
        raise RuntimeError(f"Cache row count mismatch for {clip_id}.")
    return {
        "clip_id": clip_id,
        "manifest": str(path),
        "shape": [int(value) for value in shape],
        "payload": str(bin_path.resolve()),
        "row_index": str(row_path.resolve()),
    }


def validate_target(plan_path: Path, *, target_id: str, sample_rows: int = 16) -> dict[str, Any]:
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive.")
    plan_path = plan_path.expanduser().resolve()
    _plan, target = _read_plan_target(plan_path, target_id)
    zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
    root = open_zarr_group_direct(zarr_path, mode="r")
    collection_id = str(target["collection_id"])
    quality_source_run = str(target["detect_quality_source_run"])
    quality_run = str(target["detect_quality_run"])
    quality_source = _require_complete_run(
        root, "detect_collection_sources", quality_source_run
    )
    quality = _require_complete_run(root, "detect_quality_runs", quality_run)
    quality_source_identity = _instance_keys(
        quality_source,
        label=f"detect_collection_sources/{quality_source_run}",
    )
    quality_identity = _instance_keys(
        quality,
        label=f"detect_quality_runs/{quality_run}",
    )
    if quality_source_identity != quality_identity:
        raise RuntimeError(
            "Collection quality identity summary does not match its source: "
            f"{quality_identity!r} != {quality_source_identity!r}."
        )
    expected_quality_source = str(target["detect_quality_source_group_path"])
    if normalize_attr(quality.attrs.get("source_detection_group_path")) != expected_quality_source:
        raise RuntimeError("Collection quality source group path does not match the plan.")
    quality_validation = quality.attrs.get("collection_quality_validation")
    if not isinstance(quality_validation, Mapping) or str(
        quality_validation.get("status")
    ) != "complete":
        raise RuntimeError("Collection quality validation contract is incomplete.")
    collection = root["experiment_index"]["finalized_runs"][collection_id]
    selected = collection.attrs.get("selected_runs")
    if not isinstance(selected, list) or len(selected) != len(target["clips"]):
        raise RuntimeError(
            f"Detection collection selected-run count mismatch: "
            f"{len(selected) if isinstance(selected, list) else None} != {len(target['clips'])}."
        )

    detection_reports: list[dict[str, Any]] = []
    cache_reports: list[dict[str, Any]] = []
    raw_mask_reports: list[dict[str, Any]] = []
    detection_rows = 0
    cache_rows = 0
    for clip in target["clips"]:
        refined_path = str(clip["refined_detect_group_path"])
        refined = root
        for part in Path(refined_path).parts:
            refined = refined[part]
        identity = _refined_instance_keys(refined, label=refined_path)
        detection_rows += int(identity["row_count"])
        detection_reports.append({"clip_id": str(clip["clip_id"]), "instance_key": identity})

        cache = _cache_manifest_report(
            Path(str(clip["cache_manifest"])),
            zarr_path=zarr_path,
            collection_id=collection_id,
            clip_id=str(clip["clip_id"]),
        )
        cache_rows += int(cache["shape"][0])
        cache_reports.append(cache)

        raw = _require_complete_run(
            root,
            "subject_mask_shard_runs",
            str(clip["subject_mask_shard_run"]),
        )
        raw_report = _validate_raw_masks(raw, sample_rows=sample_rows)
        raw_mask_reports.append({"clip_id": str(clip["clip_id"]), **raw_report})

    keypoints = _require_complete_run(root, "keypoints_runs", str(target["keypoint_run"]))
    refined_keypoints = _require_complete_run(
        root, "refined_keypoints_runs", str(target["refined_keypoint_run"])
    )
    keypoint_identity = _instance_keys(keypoints, label=f"keypoints_runs/{target['keypoint_run']}")
    refined_keypoint_identity = _instance_keys(
        refined_keypoints,
        label=f"refined_keypoints_runs/{target['refined_keypoint_run']}",
    )
    expected_rows = int(keypoint_identity["row_count"])
    for label, count in (
        ("refined detections", detection_rows),
        ("flat ROI caches", cache_rows),
        ("refined keypoints", int(refined_keypoint_identity["row_count"])),
    ):
        if count != expected_rows:
            raise RuntimeError(f"{label} row count {count} != keypoint row count {expected_rows}.")

    refined_masks = _require_complete_run(
        root,
        "refined_subject_masks_runs",
        str(target["refined_subject_mask_run"]),
    )
    assignment = (
        normalize_attr(refined_masks.attrs.get("assignment_keypoint_group")),
        normalize_attr(
            refined_masks.attrs.get("assignment_keypoints_run")
            or refined_masks.attrs.get("assignment_keypoint_run")
        ),
    )
    expected_assignment = ("refined_keypoints_runs", str(target["refined_keypoint_run"]))
    if assignment != expected_assignment:
        raise RuntimeError(
            f"Refined-mask keypoint assignment mismatch: {assignment!r} != {expected_assignment!r}."
        )
    refined_report = _validate_refined_masks(refined_masks, sample_rows=max(32, sample_rows))
    if int(refined_report["shape"][0]) != expected_rows:
        raise RuntimeError(
            f"Refined-mask row count {refined_report['shape'][0]} != keypoint row count {expected_rows}."
        )
    contract = validate_refined_subject_mask_contract(
        zarr_path,
        run_name=str(target["refined_subject_mask_run"]),
    )
    if not bool(contract.get("valid")):
        raise RuntimeError(
            "Refined subject-mask contract validation failed: "
            + json.dumps(contract.get("errors") or [], sort_keys=True)
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": "ok",
        "plan_path": str(plan_path),
        "target_id": target_id,
        "analysis_zarr": str(zarr_path),
        "collection_id": collection_id,
        "detect_quality_source_run": quality_source_run,
        "detect_quality_run": quality_run,
        "detect_quality_source": quality_source_identity,
        "detect_quality": quality_identity,
        "clip_count": len(target["clips"]),
        "row_count": expected_rows,
        "detections": detection_reports,
        "caches": cache_reports,
        "keypoints": keypoint_identity,
        "refined_keypoints": refined_keypoint_identity,
        "raw_subject_masks": raw_mask_reports,
        "refined_subject_masks": refined_report,
        "refined_subject_mask_contract": contract,
        "assignment_keypoint_group": assignment[0],
        "assignment_keypoint_run": assignment[1],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--sample-rows", type=int, default=16)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    report = validate_target(
        args.plan,
        target_id=args.target_id,
        sample_rows=int(args.sample_rows),
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "validate_target"]
