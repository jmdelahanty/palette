"""Build exact regular/sharded subject-mask finalizer A/B fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_filesystem import describe_filesystem, require_storage_tier
from fisheye.diagnostics.benchmark_subject_mask_probability_sharding import (
    _array_digest,
    _copy_codec_kwargs,
    _storage_stats,
)
from fisheye.refinement.assemble_refined_subject_masks import _resolve_keypoint_success_array
from fisheye.refinement.finalize_subject_masks import (
    _CROP_REBASE_COPY_ARRAYS,
    _CROP_REBASE_IDENTITY_ARRAYS,
    _load_subject_mask_source,
    _resolve_eye_assignment_context,
    finalize_subject_masks,
)
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.tune.refined_subject_mask_review import _load_source_subject_mask_run


_FIXTURE_SCHEMA = "palette.subject_mask_finalizer_ab_fixture.v1"
_FIXTURE_RUN = "subject_masks_finalizer_ab_fixture"
_FIXTURE_CROP_RUN = "crop_finalizer_ab_fixture"
_FIXTURE_KEYPOINT_RUN = "refined_keypoints_finalizer_ab_fixture"


def _iter_ranges(total_rows: int, block_rows: int) -> Iterable[tuple[int, int]]:
    for start in range(0, int(total_rows), max(1, int(block_rows))):
        yield int(start), min(int(total_rows), int(start) + max(1, int(block_rows)))


def _take_rows(array: Any, rows: np.ndarray, *rest: object) -> np.ndarray:
    indexes = np.asarray(rows, dtype=np.int64).reshape(-1)
    if not indexes.size:
        return np.empty((0, *tuple(int(value) for value in array.shape[1 + len(rest) :])), dtype=array.dtype)
    first = int(indexes[0])
    if np.array_equal(indexes, np.arange(first, first + int(indexes.shape[0]), dtype=np.int64)):
        return np.asarray(array[(slice(first, first + int(indexes.shape[0])), *rest)])
    return np.asarray(array.oindex[(indexes, *rest)])


def _selected_rows_digest(
    array: Any,
    ordered_rows: np.ndarray,
    *,
    block_rows: int,
) -> str:
    digest = hashlib.sha256()
    for channel in range(int(array.shape[1])):
        for start, stop in _iter_ranges(int(ordered_rows.shape[0]), int(block_rows)):
            values = _take_rows(array, ordered_rows[start:stop], channel, slice(None), slice(None))
            digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _recompute_frame_counts(frame_indices: np.ndarray) -> np.ndarray:
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    valid = frames >= 0
    maximum = int(frames[valid].max()) if bool(np.any(valid)) else -1
    return np.bincount(frames[valid], minlength=maximum + 1).astype(np.int32, copy=False)


def _keypoint_selected_rows(keypoint_group: zarr.Group, target_crop_rows: np.ndarray) -> np.ndarray:
    source_ids_array = keypoint_group.get("source_crop_row_ids")
    if source_ids_array is None:
        if int(keypoint_group["keypoints_roi"].shape[0]) != int(target_crop_rows.shape[0]):
            raise ValueError("Keypoint run has no source_crop_row_ids and does not match fixture rows.")
        return np.arange(int(target_crop_rows.shape[0]), dtype=np.int64)
    source_ids = np.asarray(source_ids_array[:], dtype=np.int64).reshape(-1)
    targets = np.asarray(target_crop_rows, dtype=np.int64).reshape(-1)
    if targets.size and int(targets.max()) < int(source_ids.shape[0]):
        direct = targets
        if np.array_equal(source_ids[direct], targets):
            return direct
    order = np.argsort(source_ids, kind="stable")
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, targets)
    if bool(np.any(positions >= sorted_ids.shape[0])) or not np.array_equal(sorted_ids[positions], targets):
        raise ValueError("Keypoint source_crop_row_ids do not cover all fixture rows.")
    if np.unique(source_ids).shape[0] != source_ids.shape[0]:
        raise ValueError("Keypoint source_crop_row_ids contain duplicates.")
    return np.asarray(order[positions], dtype=np.int64)


def _copy_context_arrays(
    source_root: zarr.Group,
    target_root: zarr.Group,
    *,
    source: Any,
    source_shard_run: str,
    target_crop_run: str,
    target_crop_rows: np.ndarray,
    ordered_local_rows: np.ndarray,
    assignment_keypoint_group: str,
    assignment_keypoints_run: str,
) -> dict[str, Any]:
    row_count = int(ordered_local_rows.shape[0])
    fixture_rows = np.arange(row_count, dtype=np.int64)

    source_crop = source_root[f"crop_runs/{target_crop_run}"]
    crop_parent = require_runs_parent(
        target_root,
        "crop_runs",
        completion_epoch=COMPLETION_EPOCH_STRICT,
    )
    crop_parent.attrs["latest"] = _FIXTURE_CROP_RUN
    crop = crop_parent.create_group(_FIXTURE_CROP_RUN)
    mark_run_started(crop, run_name=_FIXTURE_CROP_RUN, stage="benchmark_crop_fixture")
    crop.attrs.update(dict(source_crop.attrs))
    crop.attrs.update(
        {
            "benchmark_source_crop_run": str(target_crop_run),
            "benchmark_source_crop_rows_min": int(target_crop_rows.min()) if target_crop_rows.size else None,
            "benchmark_source_crop_rows_max": int(target_crop_rows.max()) if target_crop_rows.size else None,
        }
    )
    crop_names = tuple(dict.fromkeys((*_CROP_REBASE_IDENTITY_ARRAYS, *_CROP_REBASE_COPY_ARRAYS)))
    copied_crop_arrays: list[str] = []
    for name in crop_names:
        if name not in source_crop:
            continue
        values = np.asarray(source_crop[name].oindex[target_crop_rows])
        crop.create_array(name, data=values, overwrite=True)
        copied_crop_arrays.append(name)
    if "frame_indices" not in crop:
        raise ValueError("Fixture crop context requires frame_indices.")
    crop.create_array(
        "frame_counts",
        data=_recompute_frame_counts(np.asarray(crop["frame_indices"][:])),
        overwrite=True,
    )

    source_keypoints = source_root[f"{assignment_keypoint_group}/{assignment_keypoints_run}"]
    selected_keypoint_rows = _keypoint_selected_rows(source_keypoints, target_crop_rows)
    keypoint_parent = require_runs_parent(
        target_root,
        assignment_keypoint_group,
        completion_epoch=COMPLETION_EPOCH_STRICT,
    )
    keypoint_parent.attrs["latest"] = _FIXTURE_KEYPOINT_RUN
    keypoints = keypoint_parent.create_group(_FIXTURE_KEYPOINT_RUN)
    mark_run_started(
        keypoints,
        run_name=_FIXTURE_KEYPOINT_RUN,
        stage="benchmark_keypoint_fixture",
    )
    keypoints.attrs.update(dict(source_keypoints.attrs))
    keypoints.attrs.update(
        {
            "benchmark_source_keypoint_group": str(assignment_keypoint_group),
            "benchmark_source_keypoints_run": str(assignment_keypoints_run),
        }
    )
    success_values, success_name = _resolve_keypoint_success_array(
        source_keypoints,
        assignment_keypoints_run,
    )
    copied_keypoint_arrays: list[str] = []
    for name in ("keypoints_roi", "keypoints_img", "keypoint_scores"):
        array = source_keypoints.get(name)
        if array is None:
            continue
        values = np.asarray(array.oindex[selected_keypoint_rows])
        keypoints.create_array(name, data=values, overwrite=True)
        copied_keypoint_arrays.append(name)
    keypoints.create_array(
        str(success_name),
        data=np.asarray(success_values, dtype=bool)[selected_keypoint_rows],
        overwrite=True,
    )
    keypoints.create_array("source_crop_row_ids", data=fixture_rows, overwrite=True)

    source_shard = source_root[f"subject_mask_shard_runs/{source_shard_run}"]
    subject_parent = require_runs_parent(
        target_root,
        "subject_mask_runs",
        completion_epoch=COMPLETION_EPOCH_STRICT,
    )
    subject_parent.attrs["latest"] = _FIXTURE_RUN
    subject_run = subject_parent.create_group(_FIXTURE_RUN)
    mark_run_started(
        subject_run,
        run_name=_FIXTURE_RUN,
        stage="benchmark_subject_mask_fixture",
    )
    subject_run.attrs.update(dict(source_shard.attrs))
    subject_run.attrs.update(
        {
            "source_crop_run": _FIXTURE_CROP_RUN,
            "source_keypoints_run": _FIXTURE_KEYPOINT_RUN,
            "source_keypoint_run": _FIXTURE_KEYPOINT_RUN,
            "source_keypoint_group": str(assignment_keypoint_group),
            "assignment_keypoint_group": str(assignment_keypoint_group),
            "assignment_keypoints_run": _FIXTURE_KEYPOINT_RUN,
            "benchmark_source_subject_mask_shard_run": str(source_shard_run),
            "benchmark_source_target_crop_run": str(target_crop_run),
        }
    )
    row_array_names = (
        "detection_source",
        "frame_indices",
        "detection_indices",
        "source_frame_indices",
        "source_clip_indices",
        "source_clip_local_frame_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
        "instance_key",
    )
    copied_subject_arrays: list[str] = []
    for name in row_array_names:
        array = source_shard.get(name)
        if array is None:
            continue
        values = _take_rows(array, ordered_local_rows)
        subject_run.create_array(name, data=values, overwrite=True)
        copied_subject_arrays.append(name)
    subject_run.create_array("source_crop_row_ids", data=fixture_rows, overwrite=True)
    if "frame_indices" not in subject_run:
        subject_run.create_array(
            "frame_indices",
            data=np.asarray(crop["frame_indices"][:]),
            overwrite=True,
        )
    subject_run.create_array(
        "frame_counts",
        data=_recompute_frame_counts(np.asarray(subject_run["frame_indices"][:])),
        overwrite=True,
    )
    available = source_shard.get("available_channels")
    if available is not None:
        subject_run.create_array("available_channels", data=np.asarray(available[:]), overwrite=True)

    mark_run_complete(crop, parent_group=crop_parent, run_name=_FIXTURE_CROP_RUN)
    mark_run_complete(
        keypoints,
        parent_group=keypoint_parent,
        run_name=_FIXTURE_KEYPOINT_RUN,
    )
    mark_run_complete(
        subject_run,
        parent_group=subject_parent,
        run_name=_FIXTURE_RUN,
    )

    return {
        "row_count": row_count,
        "copied_crop_arrays": copied_crop_arrays,
        "copied_keypoint_arrays": copied_keypoint_arrays,
        "keypoint_success_dataset": str(success_name),
        "copied_subject_arrays": copied_subject_arrays,
        "target_crop_rows_sha256": hashlib.sha256(
            np.ascontiguousarray(target_crop_rows, dtype=np.int64).view(np.uint8)
        ).hexdigest(),
        "ordered_local_rows_sha256": hashlib.sha256(
            np.ascontiguousarray(ordered_local_rows, dtype=np.int64).view(np.uint8)
        ).hexdigest(),
    }


def _build_variant(
    source_root: zarr.Group,
    *,
    source_zarr_path: Path,
    output_path: Path,
    layout: str,
    source: Any,
    source_shard_run: str,
    target_crop_run: str,
    target_crop_rows: np.ndarray,
    ordered_local_rows: np.ndarray,
    assignment_keypoint_group: str,
    assignment_keypoints_run: str,
    inner_chunk_rows: int,
    shard_rows: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "schema_id": _FIXTURE_SCHEMA,
            "created_utc": utc_now(),
            "layout": str(layout),
            "source_zarr_path": str(source_zarr_path),
            "source_subject_mask_shard_run": str(source_shard_run),
            "source_target_crop_run": str(target_crop_run),
            "fixture_subject_mask_run": _FIXTURE_RUN,
            "fixture_crop_run": _FIXTURE_CROP_RUN,
            "fixture_keypoint_group": str(assignment_keypoint_group),
            "fixture_keypoints_run": _FIXTURE_KEYPOINT_RUN,
        }
    )
    context = _copy_context_arrays(
        source_root,
        root,
        source=source,
        source_shard_run=source_shard_run,
        target_crop_run=target_crop_run,
        target_crop_rows=target_crop_rows,
        ordered_local_rows=ordered_local_rows,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    source_array = source_root[f"subject_mask_shard_runs/{source_shard_run}/mask_probs_roi"]
    target_run = root[f"subject_mask_runs/{_FIXTURE_RUN}"]
    shape = (
        int(ordered_local_rows.shape[0]),
        int(source_array.shape[1]),
        int(source_array.shape[2]),
        int(source_array.shape[3]),
    )
    chunks = (int(inner_chunk_rows), 1, int(source_array.shape[2]), int(source_array.shape[3]))
    create_kwargs: dict[str, Any] = {
        "shape": shape,
        "dtype": source_array.dtype,
        "chunks": chunks,
        "fill_value": source_array.fill_value,
        "overwrite": True,
        **_copy_codec_kwargs(source_array),
    }
    shards = None
    if layout == "sharded":
        shards = (int(shard_rows), 1, int(source_array.shape[2]), int(source_array.shape[3]))
        create_kwargs["shards"] = shards
    destination = target_run.create_array("mask_probs_roi", **create_kwargs)
    destination.attrs.update(dict(source_array.attrs))
    write_block_rows = int(shard_rows) if layout == "sharded" else int(inner_chunk_rows)
    source_digest = hashlib.sha256()
    write_started = time.perf_counter()
    for channel in range(int(source_array.shape[1])):
        for start, stop in _iter_ranges(int(ordered_local_rows.shape[0]), write_block_rows):
            values = _take_rows(
                source_array,
                ordered_local_rows[start:stop],
                channel,
                slice(None),
                slice(None),
            )
            destination[start:stop, channel, :, :] = values
            source_digest.update(np.ascontiguousarray(values).view(np.uint8))
    write_seconds = float(time.perf_counter() - write_started)
    source_sha256 = source_digest.hexdigest()
    destination_sha256 = _array_digest(
        destination,
        start_row=0,
        total_rows=int(shape[0]),
        inner_chunk_rows=int(inner_chunk_rows),
    )
    if source_sha256 != destination_sha256:
        raise RuntimeError(f"Probability digest mismatch for {layout} fixture.")

    assignment = _resolve_eye_assignment_context(
        root,
        source=_load_source_subject_mask_run(root, _FIXTURE_RUN),
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=_FIXTURE_KEYPOINT_RUN,
    )
    dry_run = finalize_subject_masks(
        output_path,
        subject_run=_FIXTURE_RUN,
        components=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=_FIXTURE_KEYPOINT_RUN,
        execution_backend="process_shards",
        num_workers=8,
        dry_run=True,
        defer_registry_status=True,
    )
    storage = _storage_stats(output_path)
    return {
        "layout": str(layout),
        "path": str(output_path),
        "shape": list(shape),
        "chunks": list(chunks),
        "shards": list(shards) if shards is not None else None,
        "context": context,
        "source_sha256": source_sha256,
        "destination_sha256": destination_sha256,
        "exact": True,
        "write_seconds": write_seconds,
        "total_build_seconds": float(time.perf_counter() - started),
        "stored_bytes": int(storage["stored_bytes"]),
        "file_count": int(storage["file_count"]),
        "storage_inventory_seconds": float(storage["inventory_seconds"]),
        "assignment_row_identity": dict(assignment.row_identity_summary),
        "finalizer_dry_run": dry_run,
    }


def build_finalizer_ab_fixture(
    source_zarr: Path | str,
    *,
    source_shard_run: str,
    target_crop_run: str,
    assignment_keypoint_group: str,
    assignment_keypoints_run: str,
    output_root: Path | str,
    inner_chunk_rows: int = 32,
    shard_rows: int = 2048,
    require_output_storage_tier: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if int(shard_rows) % int(inner_chunk_rows) != 0:
        raise ValueError("shard_rows must be an integer multiple of inner_chunk_rows.")
    source_path = Path(source_zarr).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    source_filesystem = describe_filesystem(source_path)
    output_filesystem = describe_filesystem(output_path)
    require_storage_tier(
        output_filesystem,
        require_output_storage_tier,
        label="Finalizer A/B fixture output",
    )
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Fixture output already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    source_root = open_zarr_root(source_path, mode="r")
    source, collection = _load_subject_mask_source(
        source_root,
        subject_run=None,
        subject_shard_runs=[str(source_shard_run)],
        target_crop_run=str(target_crop_run),
    )
    if collection is None:
        raise RuntimeError("Could not resolve source shard as a collection fixture.")
    ordered_local_rows = np.asarray(collection.row_local_indices, dtype=np.int64)
    target_crop_rows = np.asarray(collection.source_crop_row_ids, dtype=np.int64)
    source_array = source_root[f"subject_mask_shard_runs/{source_shard_run}/mask_probs_roi"]
    expected_source_digest = _selected_rows_digest(
        source_array,
        ordered_local_rows,
        block_rows=int(inner_chunk_rows),
    )
    variants = []
    for layout, directory in (("regular", "regular.zarr"), ("sharded", "shard_02048.zarr")):
        variant = _build_variant(
            source_root,
            source_zarr_path=source_path,
            output_path=output_path / directory,
            layout=layout,
            source=source,
            source_shard_run=str(source_shard_run),
            target_crop_run=str(target_crop_run),
            target_crop_rows=target_crop_rows,
            ordered_local_rows=ordered_local_rows,
            assignment_keypoint_group=str(assignment_keypoint_group),
            assignment_keypoints_run=str(assignment_keypoints_run),
            inner_chunk_rows=int(inner_chunk_rows),
            shard_rows=int(shard_rows),
        )
        if str(variant["source_sha256"]) != expected_source_digest:
            raise RuntimeError(f"{layout} fixture source digest differs from the resolved collection order.")
        variants.append(variant)
    context_digests = {
        json.dumps(variant["context"], sort_keys=True, separators=(",", ":")) for variant in variants
    }
    if len(context_digests) != 1:
        raise RuntimeError("Regular and sharded fixture context summaries differ.")
    result = {
        "schema_id": _FIXTURE_SCHEMA,
        "created_utc": utc_now(),
        "source_zarr": str(source_path),
        "source_filesystem": source_filesystem,
        "output_filesystem": output_filesystem,
        "source_shard_run": str(source_shard_run),
        "target_crop_run": str(target_crop_run),
        "assignment_keypoint_group": str(assignment_keypoint_group),
        "assignment_keypoints_run": str(assignment_keypoints_run),
        "fixture_subject_mask_run": _FIXTURE_RUN,
        "fixture_crop_run": _FIXTURE_CROP_RUN,
        "fixture_keypoints_run": _FIXTURE_KEYPOINT_RUN,
        "row_count": int(ordered_local_rows.shape[0]),
        "inner_chunk_rows": int(inner_chunk_rows),
        "shard_rows": int(shard_rows),
        "probability_sha256": expected_source_digest,
        "all_exact": all(bool(variant["exact"]) for variant in variants),
        "variants": variants,
    }
    (output_path / "fixture_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zarr", type=Path)
    parser.add_argument("--source-shard-run", required=True)
    parser.add_argument("--target-crop-run", required=True)
    parser.add_argument(
        "--assignment-keypoint-group",
        choices=("refined_keypoints_runs", "keypoints_runs"),
        required=True,
    )
    parser.add_argument("--assignment-keypoints-run", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inner-chunk-rows", type=int, default=32)
    parser.add_argument("--shard-rows", type=int, default=2048)
    parser.add_argument(
        "--require-output-storage-tier",
        choices=("prfs", "network", "local"),
        help="Fail unless output-root resolves to this storage tier.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = build_finalizer_ab_fixture(
        args.source_zarr,
        source_shard_run=str(args.source_shard_run),
        target_crop_run=str(args.target_crop_run),
        assignment_keypoint_group=str(args.assignment_keypoint_group),
        assignment_keypoints_run=str(args.assignment_keypoints_run),
        output_root=args.output_root,
        inner_chunk_rows=int(args.inner_chunk_rows),
        shard_rows=int(args.shard_rows),
        require_output_storage_tier=args.require_output_storage_tier,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"rows={result['row_count']} sha256={result['probability_sha256']} "
            f"exact={result['all_exact']}"
        )
        for variant in result["variants"]:
            print(
                f"{variant['layout']}: files={variant['file_count']} "
                f"stored_mib={variant['stored_bytes'] / (1024.0 * 1024.0):.2f} "
                f"write_s={variant['write_seconds']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
