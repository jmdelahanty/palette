"""Independently validate exact outputs from a clipped inference DAG target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from fisheye.cluster.clipped_inference import SUPPORTED_PLAN_SCHEMAS
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.cluster.whole_recording_analysis_validate import (
    _require_complete_run,
    _validate_raw_masks,
    _validate_refined_masks,
)
from fisheye.shared.flat_roi_cache import load_flat_roi_cache_manifest
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    validate_subject_mask_bundle_candidate,
)
from fisheye.shared.zarr.subject_mask_schema import (
    derive_subject_mask_frame_row_offsets,
)

REPORT_SCHEMA = "palette.clipped_inference_target_validation.v1"
VALIDATION_ROW_CHUNK = 131_072


def _read_plan_target(
    plan_path: Path, target_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") not in SUPPORTED_PLAN_SCHEMAS
    ):
        raise ValueError(f"Unsupported clipped inference plan: {plan_path}")
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Clipped inference plan has no targets list.")
    exact = [
        row
        for row in targets
        if isinstance(row, Mapping) and str(row.get("target_id")) == target_id
    ]
    if len(exact) != 1:
        raise ValueError(
            f"Expected exactly one plan target {target_id!r}; found {len(exact)}."
        )
    return payload, exact[0]


def _required_vector(run: Any, name: str, *, label: str) -> Any:
    array = run.get(name)
    if array is None:
        raise RuntimeError(f"Modern production run is missing {name}: {label}")
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 1:
        raise RuntimeError(f"{label}/{name} must be a vector, got {shape!r}.")
    return array


def _validate_instance_key_values(
    values: np.ndarray,
    *,
    label: str,
) -> tuple[dict[str, Any], np.ndarray]:
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
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
    }, values


def _instance_key_values(run: Any, *, label: str) -> tuple[dict[str, Any], np.ndarray]:
    array = run.get("instance_key")
    if array is None:
        raise RuntimeError(f"Modern production run is missing instance_key: {label}")
    return _validate_instance_key_values(np.asarray(array[:]), label=label)


def _instance_keys(run: Any, *, label: str) -> dict[str, Any]:
    report, _values = _instance_key_values(run, label=label)
    return report


def _require_exact_instance_key_order(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    label: str,
) -> None:
    observed_values = np.asarray(observed, dtype=np.uint64).reshape(-1)
    expected_values = np.asarray(expected, dtype=np.uint64).reshape(-1)
    if observed_values.shape != expected_values.shape:
        raise RuntimeError(
            f"{label} instance_key row count {int(observed_values.size)} does not match "
            f"the canonical row count {int(expected_values.size)}."
        )
    if not np.array_equal(observed_values, expected_values):
        mismatch = np.flatnonzero(observed_values != expected_values)
        first = int(mismatch[0]) if mismatch.size else 0
        raise RuntimeError(
            f"{label} instance_key order does not exactly match canonical refined detections; "
            f"first mismatch at row {first}."
        )


def _require_exact_vector(
    array: Any,
    expected: np.ndarray,
    *,
    label: str,
    dtype: Any,
) -> None:
    shape = tuple(int(value) for value in array.shape)
    if shape != (int(expected.shape[0]),):
        raise RuntimeError(
            f"{label} shape {shape!r} does not match expected {(int(expected.shape[0]),)!r}."
        )
    for start in range(0, int(expected.shape[0]), VALIDATION_ROW_CHUNK):
        stop = min(int(expected.shape[0]), start + VALIDATION_ROW_CHUNK)
        observed = np.asarray(array[start:stop], dtype=dtype).reshape(-1)
        if not np.array_equal(observed, expected[start:stop]):
            mismatch = np.flatnonzero(observed != expected[start:stop])
            first = start + int(mismatch[0]) if mismatch.size else start
            raise RuntimeError(
                f"{label} differs from its canonical lineage at row {first}."
            )


def _validate_run_frame_counts(
    run: Any,
    *,
    label: str,
    expected_row_count: int,
    expected_counts: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    frames = _required_vector(run, "frame_indices", label=label)
    counts_array = _required_vector(run, "frame_counts", label=label)
    if int(frames.shape[0]) != int(expected_row_count):
        raise RuntimeError(
            f"{label}/frame_indices row count {int(frames.shape[0])} != {int(expected_row_count)}."
        )
    total_frames = int(counts_array.shape[0])
    if total_frames <= 0:
        raise RuntimeError(f"{label}/frame_counts is empty.")
    rebuilt = np.zeros((total_frames,), dtype=np.int64)
    for start in range(0, int(expected_row_count), VALIDATION_ROW_CHUNK):
        stop = min(int(expected_row_count), start + VALIDATION_ROW_CHUNK)
        values = np.asarray(frames[start:stop], dtype=np.int64).reshape(-1)
        if values.size and (int(values.min()) < 0 or int(values.max()) >= total_frames):
            raise RuntimeError(
                f"{label}/frame_indices rows {start}:{stop} fall outside [0, {total_frames})."
            )
        np.add.at(rebuilt, values, 1)
    observed = np.asarray(counts_array[:], dtype=np.int64).reshape(-1)
    if not np.array_equal(observed, rebuilt):
        mismatch = np.flatnonzero(observed != rebuilt)
        first = int(mismatch[0]) if mismatch.size else 0
        raise RuntimeError(
            f"{label}/frame_counts does not equal bincount(frame_indices); first mismatch at frame {first}."
        )
    if expected_counts is not None and not np.array_equal(observed, expected_counts):
        raise RuntimeError(
            f"{label}/frame_counts does not match the canonical crop frame universe."
        )
    return {
        "frame_count": total_frames,
        "row_count": int(expected_row_count),
        "sum": int(observed.sum(dtype=np.int64)),
        "exact_bincount_match": True,
        "exact_canonical_match": expected_counts is not None,
    }, observed


def _refined_instance_keys(run: Any, *, label: str) -> dict[str, Any]:
    instances = run.get("instances")
    if instances is None:
        raise RuntimeError(
            f"Modern refined detection run is missing instances: {label}"
        )
    return _instance_keys(instances, label=f"{label}/instances")


def _refined_instance_key_values(
    run: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], np.ndarray]:
    instances = run.get("instances")
    if instances is None:
        raise RuntimeError(
            f"Modern refined detection run is missing instances: {label}"
        )
    return _instance_key_values(instances, label=f"{label}/instances")


def _cache_manifest_report(
    path: Path,
    *,
    zarr_path: Path,
    collection_id: str,
    clip_id: str,
) -> tuple[dict[str, Any], np.ndarray]:
    manifest = load_flat_roi_cache_manifest(path)
    if not bool(manifest.get("cache_complete")):
        raise RuntimeError(f"Cache manifest is not complete: {path}")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError(f"Cache manifest has no source object: {path}")
    source_archive = Path(str(source.get("archive_path") or "")).expanduser().resolve()
    if source_archive != zarr_path:
        raise RuntimeError(
            f"Cache archive mismatch for {clip_id}: {source_archive} != {zarr_path}"
        )
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
    columns = row_index.get("columns")
    if not isinstance(columns, list) or "instance_key" not in columns:
        raise RuntimeError(f"Cache row index is missing instance_key for {clip_id}.")
    try:
        key_table = pq.read_table(row_path, columns=["instance_key"])
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read cache instance_key row index for {clip_id}: {row_path}"
        ) from exc
    if int(key_table.num_rows) != int(shape[0]):
        raise RuntimeError(
            f"Cache instance_key row count mismatch for {clip_id}: "
            f"{int(key_table.num_rows)} != {int(shape[0])}."
        )
    key_report, instance_keys = _validate_instance_key_values(
        np.asarray(key_table["instance_key"].combine_chunks().to_numpy()),
        label=f"cache row index {clip_id}",
    )
    return {
        "clip_id": clip_id,
        "manifest": str(path),
        "shape": [int(value) for value in shape],
        "payload": str(bin_path.resolve()),
        "row_index": str(row_path.resolve()),
        "instance_key": key_report,
    }, instance_keys


def _cache_validation_mode(*, cleaned_cache_count: int, clip_count: int) -> str:
    cleaned = int(cleaned_cache_count)
    total = int(clip_count)
    if cleaned not in {0, total}:
        raise RuntimeError(
            "Post-cleanup cache validation found a partial cache set: "
            f"{cleaned}/{total} manifests are absent."
        )
    return "post_cleanup_all_absent" if cleaned == total else "live_payloads"


def validate_target(
    plan_path: Path,
    *,
    target_id: str,
    sample_rows: int = 16,
    allow_cleaned_caches: bool = False,
) -> dict[str, Any]:
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive.")
    plan_path = plan_path.expanduser().resolve()
    plan, target = _read_plan_target(plan_path, target_id)
    zarr_path = Path(str(target["analysis_zarr"])).expanduser().resolve()
    root = open_zarr_group_direct(zarr_path, mode="r")
    downstream_mode = str(plan.get("workflow_scope") or "") == "downstream"
    collection_id = str(target["collection_id"])
    quality_source_run = str(target["detect_quality_source_run"])
    quality_run = str(target["detect_quality_run"])
    registered = target.get("registered_dish_geometry")
    registered_mode = (
        str(registered.get("gate_requirement") or "off")
        if isinstance(registered, Mapping)
        else "off"
    )
    modern_registered = registered_mode != "off"
    quality_source_identity: dict[str, Any] | None = None
    quality_identity: dict[str, Any] | None = None
    quality_source_keys: np.ndarray | None = None
    if downstream_mode:
        canonical_refined_run = str(target["finalized_refined_detect_run"])
        bound = bind_refined_detection_crop_source(
            zarr_path,
            run_id=canonical_refined_run,
            allow_selector_ineligible_benchmark=True,
        )
        attrs = bound.run_group.attrs
        gate = attrs.get("registered_detection_gate")
        if (
            bound.run_id != canonical_refined_run
            or attrs.get("status") != "complete"
            or attrs.get("finalized_recording_authority") is not True
            or attrs.get("immutable_snapshot") is not True
            or attrs.get("registered_detection_gate_requirement") != "required"
            or not isinstance(gate, Mapping)
            or gate.get("status") != "applied"
            or gate.get("applied") is not True
            or gate.get("ordered_instance_key_coverage_exact") is not True
        ):
            raise RuntimeError(
                "Downstream refined-detection input is not a complete gated "
                "immutable recording authority."
            )
        instances = bound.instances_group
        canonical_frames = np.asarray(
            instances["frame_indices"][:], dtype=np.int64
        ).reshape(-1)
        canonical_keys = np.asarray(
            instances["instance_key"][:], dtype=np.uint64
        ).reshape(-1)
        if canonical_frames.shape != canonical_keys.shape:
            raise RuntimeError("Canonical refined frame/key coverage differs.")
    elif modern_registered:
        canonical_detect_run = str(target["native_canonical_run_id"])
        quality_source = _require_complete_run(
            root, "detect_runs", canonical_detect_run
        )
        quality_source_table = (
            quality_source["instances"]
            if "instances" in quality_source
            else quality_source
        )
    else:
        quality_source = _require_complete_run(
            root, "detect_collection_sources", quality_source_run
        )
        quality_source_table = quality_source
    if not downstream_mode:
        quality = _require_complete_run(root, "detect_quality_runs", quality_run)
        quality_source_identity, quality_source_keys = _instance_key_values(
            quality_source_table,
            label=(
                f"detect_runs/{canonical_detect_run}/instances"
                if modern_registered
                else f"detect_collection_sources/{quality_source_run}"
            ),
        )
        quality_identity, quality_keys = _instance_key_values(
            quality,
            label=f"detect_quality_runs/{quality_run}",
        )
        if quality_source_identity != quality_identity:
            raise RuntimeError(
                "Collection quality identity summary does not match its source: "
                f"{quality_identity!r} != {quality_source_identity!r}."
            )
        if not np.array_equal(quality_keys, quality_source_keys):
            raise RuntimeError(
                "Collection quality instance_key order does not exactly match its source."
            )
        expected_quality_source = str(target["detect_quality_source_group_path"])
        if (
            normalize_attr(quality.attrs.get("source_detection_group_path"))
            != expected_quality_source
        ):
            raise RuntimeError(
                "Collection quality source group path does not match the plan."
            )
        quality_validation = quality.attrs.get("collection_quality_validation")
        if (
            not isinstance(quality_validation, Mapping)
            or str(quality_validation.get("status")) != "complete"
        ):
            raise RuntimeError("Collection quality validation contract is incomplete.")
        collection = root["experiment_index"]["finalized_runs"][collection_id]
        selected = collection.attrs.get("selected_runs")
        if not isinstance(selected, list) or len(selected) != len(target["clips"]):
            raise RuntimeError(
                f"Detection collection selected-run count mismatch: "
                f"{len(selected) if isinstance(selected, list) else None} != {len(target['clips'])}."
            )
    if modern_registered and not downstream_mode:
        gate = collection.attrs.get("registered_detection_gate")
        if not isinstance(gate, Mapping):
            raise RuntimeError("Registered clipped collection lacks gate consumption.")
        if gate.get("requirement") != registered_mode:
            raise RuntimeError("Registered clipped collection has the wrong gate mode.")
        if registered_mode == "required" and (
            gate.get("status") != "applied" or gate.get("applied") is not True
        ):
            raise RuntimeError("Required clipped collection is not gate-consumed.")
        canonical_refined_run = str(target["canonical_refined_run_id"])
        canonical_refined = _require_complete_run(
            root, "refined_detect_runs", canonical_refined_run
        )
        canonical_instances = canonical_refined.get("instances")
        if canonical_instances is None:
            raise RuntimeError("Canonical refined run has no instances group.")
        canonical_frames = np.asarray(
            canonical_instances["frame_indices"][:], dtype=np.int64
        ).reshape(-1)
        canonical_keys = np.asarray(
            canonical_instances["instance_key"][:], dtype=np.uint64
        ).reshape(-1)
        if canonical_frames.shape != canonical_keys.shape:
            raise RuntimeError("Canonical refined frame/key coverage differs.")

    detection_reports: list[dict[str, Any]] = []
    cache_reports: list[dict[str, Any]] = []
    raw_mask_reports: list[dict[str, Any]] = []
    detection_key_parts: list[np.ndarray] = []
    detection_rows = 0
    cache_rows = 0
    cleaned_cache_count = 0
    for clip in target["clips"]:
        if downstream_mode or modern_registered:
            frame_start = int(clip["frame_start"])
            frame_stop = int(clip["frame_stop"])
            mask = (canonical_frames >= frame_start) & (canonical_frames < frame_stop)
            keys = canonical_keys[mask]
            identity, keys = _validate_instance_key_values(
                keys,
                label=(
                    f"refined_detect_runs/{canonical_refined_run}"
                    f"[{frame_start}:{frame_stop}]"
                ),
            )
            if downstream_mode:
                row_start = int(clip["crop_row_start"])
                row_stop = int(clip["crop_row_stop"])
                if row_start < 0 or row_stop < row_start:
                    raise RuntimeError(
                        f"Invalid downstream crop-row interval for {clip['clip_id']}."
                    )
                if not np.array_equal(
                    keys,
                    canonical_keys[row_start:row_stop],
                ):
                    raise RuntimeError(
                        "Downstream clip crop-row interval does not exactly match "
                        f"canonical refined detections for {clip['clip_id']}."
                    )
        else:
            refined_path = str(clip["refined_detect_group_path"])
            refined = root
            for part in Path(refined_path).parts:
                refined = refined[part]
            identity, keys = _refined_instance_key_values(refined, label=refined_path)
        detection_rows += int(identity["row_count"])
        detection_key_parts.append(keys)
        detection_reports.append(
            {"clip_id": str(clip["clip_id"]), "instance_key": identity}
        )

        if downstream_mode:
            cache_reports.append(
                {
                    "clip_id": str(clip["clip_id"]),
                    "status": "direct_recording_crop_provider",
                    "crop_row_start": int(clip["crop_row_start"]),
                    "crop_row_stop": int(clip["crop_row_stop"]),
                }
            )
        else:
            cache_path = Path(str(clip["cache_manifest"])).expanduser().resolve()
            if not cache_path.is_file() and allow_cleaned_caches:
                cleaned_cache_count += 1
                cache_reports.append(
                    {
                        "clip_id": str(clip["clip_id"]),
                        "manifest": str(cache_path),
                        "status": "intentionally_absent_post_cleanup",
                    }
                )
            else:
                cache, cache_keys = _cache_manifest_report(
                    cache_path,
                    zarr_path=zarr_path,
                    collection_id=collection_id,
                    clip_id=str(clip["clip_id"]),
                )
                _require_exact_instance_key_order(
                    cache_keys,
                    keys,
                    label=f"Flat ROI cache for {clip['clip_id']}",
                )
                cache_rows += int(cache["shape"][0])
                cache_reports.append(cache)

        raw = _require_complete_run(
            root,
            "subject_mask_shard_runs",
            str(clip["subject_mask_shard_run"]),
        )
        raw_report = _validate_raw_masks(
            root,
            raw,
            run_name=str(clip["subject_mask_shard_run"]),
            sample_rows=sample_rows,
        )
        raw_identity, raw_keys = _instance_key_values(
            raw,
            label=f"subject_mask_shard_runs/{clip['subject_mask_shard_run']}",
        )
        _require_exact_instance_key_order(
            raw_keys,
            keys,
            label=f"Raw subject-mask shard for {clip['clip_id']}",
        )
        raw_mask_reports.append(
            {
                "clip_id": str(clip["clip_id"]),
                "instance_key": raw_identity,
                **raw_report,
            }
        )

    detection_keys = np.concatenate(detection_key_parts).astype(np.uint64, copy=False)
    detection_unique = int(np.unique(detection_keys).size)
    if detection_unique != int(detection_keys.size):
        raise RuntimeError(
            "Selected refined detections contain duplicate instance_key values across clips: "
            f"{detection_unique}/{int(detection_keys.size)} unique."
        )
    selected_keys_in_quality: bool | None
    if downstream_mode:
        selected_keys_in_quality = None
    else:
        assert quality_source_keys is not None
        sorted_quality_keys = np.sort(quality_source_keys)
        quality_positions = np.searchsorted(sorted_quality_keys, detection_keys)
        selected_keys_in_quality = bool(
            quality_positions.size == detection_keys.size
            and np.all(quality_positions < sorted_quality_keys.size)
            and np.array_equal(sorted_quality_keys[quality_positions], detection_keys)
        )
        if not selected_keys_in_quality:
            raise RuntimeError(
                "At least one selected refined-detection instance_key is absent "
                "from the quality source."
            )

    crop_run = str(target["merged_proxy_crop_run"])
    crop = root.get(f"crop_runs/{crop_run}")
    if crop is None:
        raise RuntimeError(f"Missing merged proxy crop run crop_runs/{crop_run}.")
    crop_identity, crop_keys = _instance_key_values(
        crop,
        label=f"crop_runs/{crop_run}",
    )
    if not np.array_equal(crop_keys, detection_keys):
        raise RuntimeError(
            "Merged crop proxy instance_key order does not exactly match selected refined detections."
        )

    keypoints = _require_complete_run(
        root, "keypoints_runs", str(target["keypoint_run"])
    )
    refined_keypoints = _require_complete_run(
        root, "refined_keypoints_runs", str(target["refined_keypoint_run"])
    )
    keypoint_identity, keypoint_keys = _instance_key_values(
        keypoints,
        label=f"keypoints_runs/{target['keypoint_run']}",
    )
    refined_keypoint_identity, refined_keypoint_keys = _instance_key_values(
        refined_keypoints,
        label=f"refined_keypoints_runs/{target['refined_keypoint_run']}",
    )
    expected_rows = int(keypoint_identity["row_count"])
    clip_count = len(target["clips"])
    cache_validation_mode = (
        "direct_recording_crop_provider"
        if downstream_mode
        else _cache_validation_mode(
            cleaned_cache_count=cleaned_cache_count,
            clip_count=clip_count,
        )
    )
    row_counts_to_check = [
        ("refined detections", detection_rows),
        ("refined keypoints", int(refined_keypoint_identity["row_count"])),
    ]
    if not downstream_mode and cleaned_cache_count == 0:
        row_counts_to_check.append(("flat ROI caches", cache_rows))
    for label, count in row_counts_to_check:
        if count != expected_rows:
            raise RuntimeError(
                f"{label} row count {count} != keypoint row count {expected_rows}."
            )
    for label, values in (
        ("keypoints", keypoint_keys),
        ("refined keypoints", refined_keypoint_keys),
    ):
        if not np.array_equal(values, detection_keys):
            raise RuntimeError(
                f"{label} instance_key order does not exactly match selected refined detections."
            )

    recording_frame_count = int(root.attrs.get("recording_frame_index_row_count") or 0)
    if recording_frame_count <= 0:
        raise RuntimeError(
            "Root recording_frame_index_row_count is required for clipped collection validation."
        )
    crop_frames_array = _required_vector(
        crop, "frame_indices", label=f"crop_runs/{crop_run}"
    )
    crop_frames = np.asarray(crop_frames_array[:], dtype=np.int64).reshape(-1)
    crop_acquisition_frames = np.asarray(
        _required_vector(
            crop,
            "source_acquisition_frame_index",
            label=f"crop_runs/{crop_run}",
        )[:],
        dtype=np.int64,
    ).reshape(-1)
    if int(crop_frames.size) != expected_rows:
        raise RuntimeError(
            f"Merged crop proxy frame_indices row count {int(crop_frames.size)} != {expected_rows}."
        )
    if crop_frames.size and (
        int(crop_frames.min()) < 0 or int(crop_frames.max()) >= recording_frame_count
    ):
        raise RuntimeError(
            "Merged crop proxy frame_indices fall outside the canonical recording frame universe."
        )
    canonical_frame_counts = np.bincount(
        crop_acquisition_frames,
        minlength=recording_frame_count,
    ).astype(np.int64, copy=False)
    for label, run in (
        (f"keypoints_runs/{target['keypoint_run']}", keypoints),
        (f"refined_keypoints_runs/{target['refined_keypoint_run']}", refined_keypoints),
    ):
        _require_exact_vector(
            _required_vector(run, "frame_indices", label=label),
            crop_frames,
            label=f"{label}/frame_indices",
            dtype=np.int64,
        )

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
    expected_assignment = (
        "refined_keypoints_runs",
        str(target["refined_keypoint_run"]),
    )
    if assignment != expected_assignment:
        raise RuntimeError(
            f"Refined-mask keypoint assignment mismatch: {assignment!r} != {expected_assignment!r}."
        )
    refined_mask_identity, refined_mask_keys = _instance_key_values(
        refined_masks,
        label=f"refined_subject_masks_runs/{target['refined_subject_mask_run']}",
    )
    if not np.array_equal(refined_mask_keys, detection_keys):
        raise RuntimeError(
            "Refined subject-mask instance_key order does not exactly match selected refined detections."
        )
    mask_label = f"refined_subject_masks_runs/{target['refined_subject_mask_run']}"
    _require_exact_vector(
        _required_vector(
            refined_masks,
            "source_acquisition_frame_index",
            label=mask_label,
        ),
        crop_frames,
        label=f"{mask_label}/source_acquisition_frame_index",
        dtype=np.int64,
    )
    frame_count_reports: dict[str, Any] = {}
    for label, run in (
        (f"keypoints_runs/{target['keypoint_run']}", keypoints),
        (f"refined_keypoints_runs/{target['refined_keypoint_run']}", refined_keypoints),
    ):
        report, _counts = _validate_run_frame_counts(
            run,
            label=label,
            expected_row_count=expected_rows,
            expected_counts=canonical_frame_counts,
        )
        frame_count_reports[label] = report
    expected_offsets = derive_subject_mask_frame_row_offsets(
        crop_acquisition_frames,
        n_frames=recording_frame_count,
    )
    refined_offsets = np.asarray(
        _required_vector(refined_masks, "frame_row_offsets", label=mask_label)[:],
        dtype=np.int64,
    )
    if not np.array_equal(refined_offsets, expected_offsets):
        raise RuntimeError(
            f"{mask_label}/frame_row_offsets does not exactly index crop frames."
        )
    frame_count_reports[mask_label] = {
        "frame_count": recording_frame_count,
        "row_count": expected_rows,
        "sum": int(refined_offsets[-1]),
        "exact_csr_match": True,
    }
    refined_report = _validate_refined_masks(
        refined_masks, sample_rows=max(32, sample_rows)
    )
    if int(refined_report["shape"][0]) != expected_rows:
        raise RuntimeError(
            f"Refined-mask row count {refined_report['shape'][0]} != keypoint row count {expected_rows}."
        )
    quality_run_name = str(target["subject_mask_quality_run"])
    _require_complete_run(root, "subject_mask_quality_runs", quality_run_name)
    bundle_id = str(target["subject_mask_bundle_id"])
    bundle = validate_subject_mask_bundle_candidate(
        analysis_zarr=zarr_path,
        bundle_id=bundle_id,
    )
    contract = {"valid": True, "source": "subject_mask_bundle_v1"}
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
        "clip_count": clip_count,
        "row_count": expected_rows,
        "collection_instance_key_unique_count": detection_unique,
        "selected_instance_keys_present_in_quality_source": selected_keys_in_quality,
        "detections": detection_reports,
        "caches": cache_reports,
        "cache_validation_mode": cache_validation_mode,
        "cleaned_cache_count": int(cleaned_cache_count),
        "crop_proxy": crop_identity,
        "keypoints": keypoint_identity,
        "refined_keypoints": refined_keypoint_identity,
        "raw_subject_masks": raw_mask_reports,
        "refined_subject_masks": refined_report,
        "refined_subject_mask_identity": refined_mask_identity,
        "frame_counts": frame_count_reports,
        "refined_subject_mask_contract": contract,
        "subject_mask_quality_run": quality_run_name,
        "subject_mask_bundle_id": bundle_id,
        "subject_mask_bundle": bundle,
        "assignment_keypoint_group": assignment[0],
        "assignment_keypoint_run": assignment[1],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--sample-rows", type=int, default=16)
    parser.add_argument(
        "--allow-cleaned-caches",
        action="store_true",
        help=(
            "Permit all planned ROI cache manifests to be absent after intentional workflow cleanup. "
            "A partial cache set still fails closed."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    report = validate_target(
        args.plan,
        target_id=args.target_id,
        sample_rows=int(args.sample_rows),
        allow_cleaned_caches=bool(args.allow_cleaned_caches),
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "validate_target"]
