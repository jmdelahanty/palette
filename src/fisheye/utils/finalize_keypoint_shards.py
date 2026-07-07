#!/usr/bin/env python3
"""Publish completed keypoint shard runs as one canonical keypoints run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.keypoint_summary import build_frame_keypoint_counts
from fisheye.shared.row_lineage import ROW_IDENTITY_ARRAYS, SOURCE_CROP_ROW_IDS_ARRAY
from fisheye.shared.run_provenance import build_writer_run_provenance, json_ready, stable_json
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_PROVENANCE_ATTR,
    is_run_complete,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)

KEYPOINT_SHARD_PARENT = "keypoint_shard_runs"
KEYPOINT_OUTPUT_PARENT = "keypoints_runs"
FINALIZER_SCHEMA = "palette_keypoint_shard_collection_finalizer_v1"

REQUIRED_ROW_ARRAYS: tuple[str, ...] = (
    "frame_indices",
    SOURCE_CROP_ROW_IDS_ARRAY,
    "detection_indices",
    "keypoints_roi",
    "keypoints_img",
    "keypoints_norm",
    "keypoint_confidences",
    "confidence",
    "detection_success",
    "heading",
    "heading_finite",
    "heading_usable",
    "pose_bbox_xyxy_roi",
    "effective_threshold",
    "effective_se2_radius",
    "detection_source",
)

COUNT_ARRAYS: tuple[str, ...] = ("frame_counts", "n_rois", "n_keypoints")

OPTIONAL_ROW_ARRAYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        name
        for name in (
            *ROW_IDENTITY_ARRAYS,
            "source_frame_indices",
            "source_clip_indices",
            "source_clip_local_frame_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
        )
        if name not in REQUIRED_ROW_ARRAYS
    )
)

GEOMETRY_PRELOAD_ROW_ARRAYS: frozenset[str] = frozenset(
    (
        "frame_indices",
        SOURCE_CROP_ROW_IDS_ARRAY,
        "detection_indices",
        *ROW_IDENTITY_ARRAYS,
        "source_frame_indices",
        "source_clip_indices",
        "source_clip_local_frame_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
        "detection_source",
    )
)

COMPAT_ATTRS: tuple[str, ...] = (
    "keypoint_labels",
    "keypoint_confidence_labels",
    "skeleton_id",
    "kpt_shape",
    "pose_schema",
    "model_kpt_shape",
)

COPY_IF_CONSISTENT_ATTRS: tuple[str, ...] = (
    "method",
    "model_path",
    "model_name",
    "model_names",
    "ultralytics_version",
    "input_mode_requested",
    "input_mode_effective",
    "model_input_transform",
    "model_input_transform_name",
    "model_input_shape_hw",
    "native_roi_shape_hw",
    "source_detect_run",
    "source_refined_run",
    "source_roi_read_mode",
    "roi_cache_policy",
    "source_roi_cache_used",
    "source_roi_cache_backend",
    "source_roi_cache_source_tier",
    "source_roi_cache_staged_to_node_scratch",
    "source_crop_storage_mode",
    "source_crop_signature",
    "source_crop_revision",
    "source_crop_created_at_utc",
)


@dataclass(frozen=True)
class Shard:
    name: str
    group: zarr.Group
    total_rows: int
    source_crop_run: str
    frame_axis_len: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_equal(left: object, right: object) -> bool:
    return stable_json(left) == stable_json(right)


def _as_array(group: zarr.Group, name: str) -> np.ndarray:
    return np.asarray(group[name][:])


def _array_chunks(source: object | None, data: np.ndarray, *, default_row_chunk: int = 2048) -> tuple[int, ...] | None:
    if data.ndim == 0:
        return None
    raw_chunks = getattr(source, "chunks", None)
    if raw_chunks:
        chunks = tuple(int(v) for v in raw_chunks)
    else:
        chunks = (min(max(1, int(data.shape[0])), int(default_row_chunk)), *tuple(max(1, int(v)) for v in data.shape[1:]))
    normalized = []
    for axis, dim in enumerate(data.shape):
        value = chunks[axis] if axis < len(chunks) else chunks[-1]
        normalized.append(max(1, min(int(dim), int(value))))
    return tuple(normalized)


def _write_array_like(
    target: zarr.Group,
    name: str,
    data: np.ndarray,
    *,
    source: object | None = None,
    geometry_preload: bool = False,
) -> None:
    data = np.asarray(data)
    if geometry_preload:
        create_geometry_preload_array(target, name, data=data, overwrite=True)
        return
    kwargs: dict[str, Any] = {"data": data, "overwrite": True}
    chunks = _array_chunks(source, data)
    if chunks is not None:
        kwargs["chunks"] = chunks
    try:
        target.create_array(name, **kwargs)
    except TypeError:
        kwargs.pop("chunks", None)
        target.create_array(name, **kwargs)


def _resolve_shard(root: zarr.Group, shard_name: str) -> Shard:
    shard_parent = root.get(KEYPOINT_SHARD_PARENT)
    if shard_parent is None:
        raise ValueError(f"Missing {KEYPOINT_SHARD_PARENT} group.")
    if shard_name not in shard_parent:
        raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name} not found.")
    group = shard_parent[shard_name]

    status = normalize_attr(group.attrs.get(RUN_COMPLETION_STATUS_ATTR))
    if status is not None and not is_run_complete(group, legacy_default=False):
        raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name} is not complete (status={status!r}).")

    missing = [name for name in (*REQUIRED_ROW_ARRAYS, *COUNT_ARRAYS) if name not in group]
    if missing:
        raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name} missing required arrays: {missing}")

    total_rows = int(group["frame_indices"].shape[0])
    for name in REQUIRED_ROW_ARRAYS:
        rows = int(group[name].shape[0])
        if rows != total_rows:
            raise ValueError(
                f"{KEYPOINT_SHARD_PARENT}/{shard_name}/{name} has {rows} rows; expected {total_rows}."
            )

    source_crop_run = normalize_attr(group.attrs.get("source_crop_run"))
    if not source_crop_run:
        raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name} missing source_crop_run attr.")
    crop_group = root.get(f"crop_runs/{source_crop_run}")
    if crop_group is None:
        raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name} references missing crop_runs/{source_crop_run}.")
    if "frame_indices" in crop_group:
        crop_rows = int(crop_group["frame_indices"].shape[0])
        crop_row_ids = _as_array(group, SOURCE_CROP_ROW_IDS_ARRAY).astype(np.int64, copy=False).reshape(-1)
        if crop_row_ids.size and (int(crop_row_ids.min()) < 0 or int(crop_row_ids.max()) >= crop_rows):
            raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name} has source_crop_row_ids outside crop row range.")
        crop_frames = _as_array(crop_group, "frame_indices").astype(np.int64, copy=False).reshape(-1)
        shard_frames = _as_array(group, "frame_indices").astype(np.int64, copy=False).reshape(-1)
        if not np.array_equal(shard_frames, crop_frames[crop_row_ids]):
            raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name} source_crop_row_ids do not match crop frames.")

    frame_axis_len = 0
    for name in COUNT_ARRAYS:
        array = group[name]
        if array.ndim != 1:
            raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shard_name}/{name} must be 1D, got shape {array.shape}.")
        frame_axis_len = max(frame_axis_len, int(array.shape[0]))
    return Shard(
        name=shard_name,
        group=group,
        total_rows=total_rows,
        source_crop_run=source_crop_run,
        frame_axis_len=frame_axis_len,
    )


def _load_shards(root: zarr.Group, shard_names: Sequence[str]) -> list[Shard]:
    if not shard_names:
        raise ValueError("At least one --shard-run is required.")
    if len(set(shard_names)) != len(shard_names):
        raise ValueError("Duplicate shard run names were supplied.")
    shards = [_resolve_shard(root, name) for name in shard_names]

    source_crop_runs = {shard.source_crop_run for shard in shards}
    if len(source_crop_runs) != 1:
        raise ValueError(
            "Mixed source_crop_run shard finalization is not supported in v1; "
            f"found {sorted(source_crop_runs)!r}."
        )

    first = shards[0].group
    for attr in COMPAT_ATTRS:
        if attr not in first.attrs:
            raise ValueError(f"{KEYPOINT_SHARD_PARENT}/{shards[0].name} missing compatibility attr {attr!r}.")
        expected = first.attrs.get(attr)
        for shard in shards[1:]:
            if attr not in shard.group.attrs or not _json_equal(shard.group.attrs.get(attr), expected):
                raise ValueError(f"Shard attr {attr!r} differs between {shards[0].name} and {shard.name}.")

    optional_presence: dict[str, bool] = {name: name in first for name in OPTIONAL_ROW_ARRAYS}
    for shard in shards[1:]:
        for name, expected_present in optional_presence.items():
            if (name in shard.group) != expected_present:
                raise ValueError(f"Optional row array {name!r} is not present consistently across shards.")

    return shards


def _merged_row_arrays(shards: Sequence[Shard]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    names = list(REQUIRED_ROW_ARRAYS)
    for name in OPTIONAL_ROW_ARRAYS:
        if name in shards[0].group and name not in names:
            names.append(name)

    merged = {name: np.concatenate([_as_array(shard.group, name) for shard in shards], axis=0) for name in names}
    crop_row_ids = np.asarray(merged[SOURCE_CROP_ROW_IDS_ARRAY], dtype=np.int64).reshape(-1)
    unique_crop_rows = np.unique(crop_row_ids)
    if unique_crop_rows.size != crop_row_ids.size:
        raise ValueError("Duplicate source_crop_row_ids found across keypoint shards for the same source_crop_run.")

    order = np.argsort(crop_row_ids, kind="stable")
    sorted_rows = {name: values[order] for name, values in merged.items()}
    return sorted_rows, order


def _merge_count_arrays(shards: Sequence[Shard], name: str) -> np.ndarray:
    frame_axis_len = max((shard.frame_axis_len for shard in shards), default=0)
    out = np.zeros(frame_axis_len, dtype=np.int32)
    for shard in shards:
        data = _as_array(shard.group, name).astype(np.int32, copy=False).reshape(-1)
        out[: data.shape[0]] += data
    return out


def _consistent_attrs(shards: Sequence[Shard]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    first = shards[0].group
    for name in (*COMPAT_ATTRS, *COPY_IF_CONSISTENT_ATTRS):
        if name not in first.attrs:
            continue
        value = first.attrs.get(name)
        if all(name in shard.group.attrs and _json_equal(shard.group.attrs.get(name), value) for shard in shards[1:]):
            attrs[name] = json_ready(value)
    return attrs


def _summary_statistics(row_arrays: Mapping[str, np.ndarray], keypoint_count: int) -> dict[str, Any]:
    total_rows = int(row_arrays["frame_indices"].shape[0])
    success = np.asarray(row_arrays["detection_success"], dtype=bool).reshape(-1)
    successful = int(np.count_nonzero(success))
    failed = int(total_rows - successful)
    confidence = np.asarray(row_arrays["confidence"], dtype=np.float64).reshape(-1)
    finite_conf = confidence[np.isfinite(confidence)]
    return {
        "total_rois": total_rows,
        "keypoint_count": int(keypoint_count),
        "successful_detections": successful,
        "failed_detections": failed,
        "success_rate_percent": round((successful / total_rows * 100.0) if total_rows else 0.0, 2),
        "mean_confidence": float(np.mean(finite_conf)) if finite_conf.size else 0.0,
    }


def _load_shard_names_from_file(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("shard_runs", "keypoint_shard_runs", "runs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
    raise ValueError(f"Could not read shard run list from {path}; expected list or mapping with shard_runs.")


def finalize_keypoint_shards(
    *,
    zarr_path: str | Path,
    shard_runs: Sequence[str],
    output_run: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Finalize completed ``keypoint_shard_runs`` into a normal keypoint run."""

    archive = Path(zarr_path).expanduser().resolve()
    root = open_zarr_root(archive, mode="r" if dry_run else "a")
    shards = _load_shards(root, [str(name) for name in shard_runs])
    row_arrays, sort_order = _merged_row_arrays(shards)
    source_crop_run = shards[0].source_crop_run
    total_rows = int(row_arrays["frame_indices"].shape[0])
    keypoint_count = int(row_arrays["keypoints_roi"].shape[1])
    count_arrays = {name: _merge_count_arrays(shards, name) for name in COUNT_ARRAYS}
    recomputed_keypoints = build_frame_keypoint_counts(
        np.asarray(row_arrays["frame_indices"], dtype=np.int64),
        np.asarray(row_arrays["detection_success"], dtype=bool),
        frame_axis_len=int(count_arrays["n_keypoints"].shape[0]),
        keypoint_count=keypoint_count,
    )
    if not np.array_equal(count_arrays["n_keypoints"], recomputed_keypoints):
        count_arrays["n_keypoints_from_shards"] = count_arrays["n_keypoints"]
        count_arrays["n_keypoints"] = recomputed_keypoints

    created_at = _utc_now()
    resolved_output_run = output_run or f"keypoints_collection_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    source_shard_paths = [f"{KEYPOINT_SHARD_PARENT}/{shard.name}" for shard in shards]
    source_crop_runs = sorted({shard.source_crop_run for shard in shards})
    summary_stats = _summary_statistics(row_arrays, keypoint_count)
    result = {
        "ok": True,
        "status": "would_write" if dry_run else "written",
        "zarr_path": str(archive),
        "output_run": resolved_output_run,
        "output_path": f"{KEYPOINT_OUTPUT_PARENT}/{resolved_output_run}",
        "source_keypoint_shard_runs": [shard.name for shard in shards],
        "source_keypoint_shard_run_paths": source_shard_paths,
        "source_crop_run": source_crop_run,
        "source_crop_runs": source_crop_runs,
        "total_rois": total_rows,
        "sort_policy": "source_crop_row_ids_stable_ascending",
        "sort_changed_order": bool(not np.array_equal(sort_order, np.arange(sort_order.shape[0]))),
        "summary_statistics": summary_stats,
    }
    if dry_run:
        return result

    parent = require_runs_parent(
        root,
        KEYPOINT_OUTPUT_PARENT,
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    if resolved_output_run in parent:
        if not overwrite:
            raise ValueError(f"{KEYPOINT_OUTPUT_PARENT}/{resolved_output_run} already exists. Pass --overwrite to replace it.")
        del parent[resolved_output_run]

    start = time.perf_counter()
    run_group = parent.create_group(resolved_output_run)
    mark_run_started(run_group, run_name=resolved_output_run, stage="keypoints")
    note_pending_latest(parent, resolved_output_run)
    try:
        for name, data in row_arrays.items():
            source_array = shards[0].group.get(name)
            _write_array_like(
                run_group,
                name,
                data,
                source=source_array,
                geometry_preload=name in GEOMETRY_PRELOAD_ROW_ARRAYS,
            )
        for name, data in count_arrays.items():
            create_geometry_preload_array(run_group, name, data=data, overwrite=True)

        attrs = _consistent_attrs(shards)
        attrs.update(
            {
                "collection_finalizer_schema": FINALIZER_SCHEMA,
                "method": "fisheye.utils.finalize_keypoint_shards",
                "run_semantics": "finalized_keypoint_shard_collection",
                "created_at_utc": created_at,
                "keypoints_timestamp_utc": created_at,
                "source_keypoint_shard_runs": [shard.name for shard in shards],
                "source_keypoint_shard_run_paths": source_shard_paths,
                "source_crop_run": source_crop_run,
                "source_crop_runs": source_crop_runs,
                "source_kind": "keypoint_shard_collection_finalizer",
                "finalized_from_keypoint_shards": True,
                "stage_selector_eligible": True,
                "output_parent": KEYPOINT_OUTPUT_PARENT,
                "run_group_parent": KEYPOINT_OUTPUT_PARENT,
                "source_keypoint_shard_count": int(len(shards)),
                "keypoints_processed": total_rows,
                "success_rate": summary_stats["success_rate_percent"],
                "summary_statistics": summary_stats,
                "sort_policy": "source_crop_row_ids_stable_ascending",
                "sort_changed_order": bool(result["sort_changed_order"]),
                "parameters": {
                    "shard_runs": [shard.name for shard in shards],
                    "output_run": resolved_output_run,
                    "overwrite": bool(overwrite),
                },
            }
        )
        run_group.attrs.update(json_ready(attrs))
        run_provenance = build_writer_run_provenance(
            command="fisheye.utils.finalize_keypoint_shards",
            params={
                "zarr_path": str(archive),
                "shard_runs": [shard.name for shard in shards],
                "output_run": resolved_output_run,
                "sort_policy": "source_crop_row_ids_stable_ascending",
            },
            input_run_ids={
                "keypoint_shards": source_shard_paths,
                "crop": source_crop_run,
            },
            cwd=Path.cwd(),
        )
        run_group.attrs[RUN_PROVENANCE_ATTR] = run_provenance
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=resolved_output_run,
            run_provenance=run_provenance,
        )
        root.attrs["current_keypoint_group_path"] = run_group.path
        duration = time.perf_counter() - start
        run_group.attrs["finalization_duration_seconds"] = float(duration)
        result["finalization_duration_seconds"] = float(duration)
        return result
    except Exception as exc:
        mark_run_failed(run_group, error=str(exc))
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette analysis/training Zarr archive.")
    parser.add_argument("--shard-run", action="append", default=[], help="keypoint_shard_runs child to merge. Repeatable.")
    parser.add_argument("--shard-runs-file", type=Path, help="JSON list or mapping containing shard_runs.")
    parser.add_argument("--output-run", help="Destination keypoints_runs child name.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    shard_runs = list(args.shard_run or [])
    if args.shard_runs_file is not None:
        shard_runs.extend(_load_shard_names_from_file(args.shard_runs_file.expanduser()))
    try:
        result = finalize_keypoint_shards(
            zarr_path=args.zarr_path,
            shard_runs=shard_runs,
            output_run=args.output_run,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    else:
        print(f"{result['status']}: {result['output_path']} ({result['total_rois']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
