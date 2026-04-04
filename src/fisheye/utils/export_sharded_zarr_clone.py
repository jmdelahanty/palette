#!/usr/bin/env python3
"""Rewrite a Palette Zarr archive into a benchmark-only sharded clone.

Default mode is dry-run. Use --apply to write a new destination .zarr and a
sidecar manifest describing source/destination array layouts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.subject_mask_chunks import (
    refined_subject_mask_storage_chunks,
    subject_mask_storage_chunks,
)

POLICY_CHOICES = ("raw_only", "raw_and_crops", "dense_readmostly_v1", "dense_readmostly_rechunk_v1")
RAW_VIDEO_IMAGE_PATHS = {"raw_video/images_full", "raw_video/images_ds", "raw_video/images_ds_color"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_path_for(dest_zarr: Path) -> Path:
    return dest_zarr.with_name(f"{dest_zarr.name}.shard-manifest.json")


def _copy_attrs(src_attrs: Any, dest_attrs: Any) -> None:
    dest_attrs.update(dict(src_attrs))


def _iter_chunk_slices(shape: Sequence[int], chunks: Sequence[int] | None) -> Iterable[tuple[slice, ...]]:
    if not chunks:
        yield tuple(slice(0, int(dim)) for dim in shape)
        return
    chunk_dims: list[int] = []
    for axis, dim in enumerate(shape):
        chunk = int(chunks[axis] if axis < len(chunks) else chunks[-1])
        if chunk <= 0:
            chunk = int(dim)
        chunk_dims.append(chunk)
    grid = [int(math.ceil(int(dim) / int(chunk))) for dim, chunk in zip(shape, chunk_dims)]
    for idx in np.ndindex(*grid):
        slices: list[slice] = []
        for axis, chunk_idx in enumerate(idx):
            start = int(chunk_idx) * int(chunk_dims[axis])
            stop = min(start + int(chunk_dims[axis]), int(shape[axis]))
            slices.append(slice(start, stop))
        yield tuple(slices)


def _iter_arrays(group: zarr.Group, prefix: str = "") -> Iterable[tuple[str, zarr.Array]]:
    for name in group.keys():
        item = group[name]
        path = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(item, zarr.Array):
            yield path, item
        elif isinstance(item, zarr.Group):
            yield from _iter_arrays(item, prefix=path)


def _is_crop_roi_images(path: str) -> bool:
    parts = path.split("/")
    return len(parts) == 3 and parts[0] == "crop_runs" and parts[2] == "roi_images"


def _is_subject_mask_dense(path: str) -> bool:
    parts = path.split("/")
    return (
        len(parts) == 3
        and parts[0] in {"subject_mask_runs", "refined_subject_masks_runs"}
        and parts[2] in {"masks_roi", "mask_probs_roi"}
    )


def _policy_selects_path(path: str, policy: str) -> bool:
    if policy == "raw_only":
        return path in RAW_VIDEO_IMAGE_PATHS
    if policy == "raw_and_crops":
        return path in RAW_VIDEO_IMAGE_PATHS or _is_crop_roi_images(path)
    if policy in {"dense_readmostly_v1", "dense_readmostly_rechunk_v1"}:
        return path in RAW_VIDEO_IMAGE_PATHS or _is_crop_roi_images(path) or _is_subject_mask_dense(path)
    raise ValueError(f"Unsupported policy {policy!r}.")


def _array_data_type(arr: zarr.Array) -> Any:
    metadata = getattr(arr, "metadata", None)
    data_type = getattr(metadata, "data_type", None)
    return data_type if data_type is not None else arr.dtype


def _array_itemsize(arr: zarr.Array) -> Optional[int]:
    try:
        return int(np.dtype(arr.dtype).itemsize)
    except Exception:
        return None


def _array_is_shard_eligible(arr: zarr.Array) -> bool:
    if getattr(arr, "chunks", None) is None:
        return False
    if int(arr.ndim) <= 0:
        return False
    dtype = arr.dtype
    kind = getattr(dtype, "kind", None)
    dtype_text = str(dtype).lower()
    metadata = getattr(arr, "metadata", None)
    data_type = getattr(metadata, "data_type", None)
    data_type_text = str(data_type).lower()
    if kind in {"O", "U", "S"}:
        return False
    if "utf8" in dtype_text or "utf8" in data_type_text or "variablelengthutf8" in data_type_text:
        return False
    return _array_itemsize(arr) is not None


def _dest_chunks_for_policy(path: str, arr: zarr.Array, policy: str) -> tuple[int, ...] | None:
    chunks = getattr(arr, "chunks", None)
    if chunks is None:
        return None
    source_chunks = tuple(int(v) for v in chunks)
    if policy != "dense_readmostly_rechunk_v1":
        return source_chunks
    if _is_subject_mask_dense(path):
        if int(arr.ndim) != 4:
            return source_chunks
        if path.startswith("refined_subject_masks_runs/"):
            return refined_subject_mask_storage_chunks(
                total_rows=int(arr.shape[0]),
                height=int(arr.shape[2]),
                width=int(arr.shape[3]),
            )
        return subject_mask_storage_chunks(
            total_rows=int(arr.shape[0]),
            height=int(arr.shape[2]),
            width=int(arr.shape[3]),
        )
    return source_chunks


def _compute_shards_for_layout(
    shape: Sequence[int],
    chunk_shape: Sequence[int] | None,
    *,
    itemsize: int | None,
    target_mb: int,
) -> tuple[int, ...] | None:
    if chunk_shape is None:
        return None
    chunk_shape = tuple(int(v) for v in chunk_shape)
    if not chunk_shape or chunk_shape[0] <= 0:
        return None
    if itemsize is None or itemsize <= 0:
        return None
    chunk_bytes = int(itemsize * math.prod(chunk_shape))
    if chunk_bytes <= 0:
        return None
    target_bytes = max(1, int(target_mb)) * 1024 * 1024
    chunks_per_shard = max(1, target_bytes // chunk_bytes)
    desired_shard0 = int(chunk_shape[0] * chunks_per_shard)
    if int(shape[0]) >= int(chunk_shape[0]):
        max_full_multiple = int(shape[0] // chunk_shape[0]) * int(chunk_shape[0])
        shard0 = max(int(chunk_shape[0]), min(desired_shard0, max_full_multiple))
    else:
        shard0 = int(chunk_shape[0])
    return (int(shard0), *chunk_shape[1:])


@dataclass(frozen=True)
class ArrayClonePlan:
    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: Optional[tuple[int, ...]]
    dest_chunks: Optional[tuple[int, ...]]
    source_shards: Optional[tuple[int, ...]]
    dest_shards: Optional[tuple[int, ...]]
    policy_selected: bool
    action: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class ExportPlan:
    source_zarr: str
    dest_zarr: str
    policy: str
    target_mb: int
    array_plans: tuple[ArrayClonePlan, ...]


def build_export_plan(source_zarr: Path | str, dest_zarr: Path | str, *, policy: str, target_mb: int) -> ExportPlan:
    source_path = Path(source_zarr).expanduser()
    dest_path = Path(dest_zarr).expanduser()
    if policy not in POLICY_CHOICES:
        raise ValueError(f"Unsupported policy {policy!r}.")
    if int(target_mb) <= 0:
        raise ValueError("target_mb must be positive.")

    root = zarr.open_group(str(source_path), mode="r")
    array_plans: list[ArrayClonePlan] = []
    for path, arr in _iter_arrays(root):
        source_shards = tuple(int(v) for v in getattr(arr, "shards", None)) if getattr(arr, "shards", None) else None
        chunks = tuple(int(v) for v in getattr(arr, "chunks", None)) if getattr(arr, "chunks", None) else None
        dest_chunks = _dest_chunks_for_policy(path, arr, policy)
        selected = _policy_selects_path(path, policy)
        if source_shards is not None and dest_chunks == chunks:
            action = "preserve_existing_shards"
            dest_shards = source_shards
            reason = "source already sharded"
        elif selected and _array_is_shard_eligible(arr):
            dest_shards = _compute_shards_for_layout(arr.shape, dest_chunks, itemsize=_array_itemsize(arr), target_mb=target_mb)
            if dest_shards is None:
                action = "rechunk_only" if dest_chunks != chunks else "keep_chunked"
                reason = "could not compute shard shape"
            else:
                action = "rechunk_and_add_shards" if dest_chunks != chunks else "add_shards"
                reason = f"policy={policy}"
        elif selected:
            action = "rechunk_only" if dest_chunks != chunks else "keep_chunked"
            dest_shards = None
            reason = "selected path is not shard-eligible" if dest_chunks == chunks else f"policy={policy}"
        else:
            action = "keep_chunked"
            dest_shards = None
            reason = None
        array_plans.append(
            ArrayClonePlan(
                path=path,
                shape=tuple(int(v) for v in arr.shape),
                dtype=str(arr.dtype),
                chunks=chunks,
                dest_chunks=dest_chunks,
                source_shards=source_shards,
                dest_shards=dest_shards,
                policy_selected=selected,
                action=action,
                reason=reason,
            )
        )

    return ExportPlan(
        source_zarr=str(source_path),
        dest_zarr=str(dest_path),
        policy=policy,
        target_mb=int(target_mb),
        array_plans=tuple(array_plans),
    )


def _array_plan_map(plan: ExportPlan) -> dict[str, ArrayClonePlan]:
    return {row.path: row for row in plan.array_plans}


def _copy_array(
    src: zarr.Array,
    dest_group: zarr.Group,
    name: str,
    *,
    dest_chunks: tuple[int, ...] | None,
    dest_shards: tuple[int, ...] | None,
) -> zarr.Array:
    kwargs: dict[str, Any] = {
        "shape": src.shape,
        "dtype": _array_data_type(src),
        "overwrite": True,
    }
    source_chunks = getattr(src, "chunks", None)
    if dest_chunks is not None:
        kwargs["chunks"] = tuple(int(v) for v in dest_chunks)
    if dest_shards is not None:
        kwargs["shards"] = tuple(int(v) for v in dest_shards)
    fill_value = getattr(src, "fill_value", None)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    compressors = getattr(src, "compressors", None)
    if compressors:
        kwargs["compressors"] = compressors
    filters = getattr(src, "filters", None)
    if filters:
        kwargs["filters"] = filters
    serializer = getattr(src, "serializer", None)
    if serializer is not None:
        kwargs["serializer"] = serializer

    dest = dest_group.create_array(name, **kwargs)
    _copy_attrs(src.attrs, dest.attrs)

    if source_chunks is None:
        dest[...] = src[...]
        return dest

    for slc in _iter_chunk_slices(tuple(int(v) for v in src.shape), tuple(int(v) for v in source_chunks)):
        dest[slc] = src[slc]
    return dest


def _clone_group(src_group: zarr.Group, dest_group: zarr.Group, *, prefix: str, plan_map: dict[str, ArrayClonePlan]) -> None:
    _copy_attrs(src_group.attrs, dest_group.attrs)
    for name in src_group.keys():
        item = src_group[name]
        path = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(item, zarr.Group):
            child = dest_group.require_group(name)
            _clone_group(item, child, prefix=path, plan_map=plan_map)
            continue
        if isinstance(item, zarr.Array):
            array_plan = plan_map[path]
            _copy_array(
                item,
                dest_group,
                name,
                dest_chunks=array_plan.dest_chunks,
                dest_shards=array_plan.dest_shards,
            )


def _write_manifest(plan: ExportPlan, dest_zarr: Path) -> Path:
    manifest_path = _manifest_path_for(dest_zarr)
    manifest = {
        "version": 1,
        "created_utc": _utc_now_iso(),
        "source_zarr": plan.source_zarr,
        "dest_zarr": plan.dest_zarr,
        "policy": plan.policy,
        "target_mb": plan.target_mb,
        "arrays_total": len(plan.array_plans),
        "arrays_with_dest_shards": sum(1 for row in plan.array_plans if row.dest_shards is not None),
        "arrays_added_shards": sum(1 for row in plan.array_plans if row.action == "add_shards"),
        "arrays_rechunked": sum(1 for row in plan.array_plans if row.dest_chunks != row.chunks),
        "arrays_rechunked_and_sharded": sum(1 for row in plan.array_plans if row.action == "rechunk_and_add_shards"),
        "arrays_preserved_existing_shards": sum(
            1 for row in plan.array_plans if row.action == "preserve_existing_shards"
        ),
        "array_plans": [asdict(row) for row in plan.array_plans],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def export_sharded_zarr_clone(
    source_zarr: Path | str,
    *,
    dest_zarr: Path | str,
    policy: str,
    target_mb: int = 128,
    overwrite: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    plan = build_export_plan(source_zarr, dest_zarr, policy=policy, target_mb=target_mb)
    dest_path = Path(dest_zarr).expanduser()
    manifest_path = _manifest_path_for(dest_path)
    summary = {
        "source_zarr": plan.source_zarr,
        "dest_zarr": plan.dest_zarr,
        "policy": plan.policy,
        "target_mb": plan.target_mb,
        "status": "planned",
        "arrays_total": len(plan.array_plans),
        "arrays_added_shards": sum(1 for row in plan.array_plans if row.action == "add_shards"),
        "arrays_rechunked": sum(1 for row in plan.array_plans if row.dest_chunks != row.chunks),
        "arrays_rechunked_and_sharded": sum(1 for row in plan.array_plans if row.action == "rechunk_and_add_shards"),
        "arrays_preserved_existing_shards": sum(
            1 for row in plan.array_plans if row.action == "preserve_existing_shards"
        ),
        "manifest_path": str(manifest_path),
        "array_plans": [asdict(row) for row in plan.array_plans],
    }

    if dest_path.exists() and not overwrite:
        summary["status"] = "skipped_existing"
        summary["reason"] = f"{dest_path} already exists"
        return summary

    if not apply:
        return summary

    if dest_path.exists() and overwrite:
        if dest_path.is_dir():
            shutil.rmtree(dest_path)
        else:
            dest_path.unlink()
    if manifest_path.exists() and overwrite:
        manifest_path.unlink()

    src_root = zarr.open_group(str(Path(source_zarr).expanduser()), mode="r")
    dest_root = zarr.open_group(str(dest_path), mode="w", zarr_format=3)
    _clone_group(src_root, dest_root, prefix="", plan_map=_array_plan_map(plan))
    manifest_path = _write_manifest(plan, dest_path)

    summary["status"] = "updated"
    summary["manifest_path"] = str(manifest_path)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "{status} {dest} policy={policy} arrays={arrays} add_shards={add} rechunk={rechunk} preserve_existing={preserve}".format(
            status=summary["status"],
            dest=summary["dest_zarr"],
            policy=summary["policy"],
            arrays=summary["arrays_total"],
            add=summary["arrays_added_shards"],
            rechunk=summary["arrays_rechunked"],
            preserve=summary["arrays_preserved_existing_shards"],
        )
    )
    if summary.get("reason"):
        print(f"  reason: {summary['reason']}")
    print(f"  manifest: {summary['manifest_path']}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zarr", type=Path, help="Source .zarr archive to clone.")
    parser.add_argument("--dest", type=Path, required=True, help="Destination .zarr path for the clone.")
    parser.add_argument("--policy", choices=list(POLICY_CHOICES), required=True, help="Benchmark sharding policy.")
    parser.add_argument(
        "--target-mb",
        type=int,
        default=128,
        help="Approximate target shard size in MB along axis 0 (default: 128).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination clone.")
    parser.add_argument("--apply", action="store_true", help="Write the destination clone and manifest.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    summary = export_sharded_zarr_clone(
        args.source_zarr,
        dest_zarr=args.dest,
        policy=str(args.policy),
        target_mb=int(args.target_mb),
        overwrite=bool(args.overwrite),
        apply=bool(args.apply),
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
        if not args.apply:
            print("Dry run: add --apply to write the sharded clone and manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
