"""Deterministically copy a completed row-aligned Zarr run into indexed shards.

Each process owns one complete, non-overlapping outer shard for one array.  The
copy therefore never lets two workers perform read-modify-write operations on
the same physical Zarr object.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr


SHARDED_COPY_SCHEMA_ID = "palette.zarr_sharded_run_copy.v1"
_SOURCE_ROOT: zarr.Group | None = None
_DESTINATION_ROOT: zarr.Group | None = None


@dataclass(frozen=True)
class ShardedArrayPlan:
    path: str
    shape: tuple[int, ...]
    dtype: str
    inner_chunks: tuple[int, ...]
    row_aligned: bool
    requested_shard_rows: int | None
    effective_shard_rows: int | None
    outer_shards: tuple[int, ...] | None


def _iter_arrays(group: zarr.Group, prefix: str = "") -> Iterable[tuple[str, zarr.Array]]:
    for name, array in sorted(group.arrays(), key=lambda item: item[0]):
        path = f"{prefix}/{name}" if prefix else str(name)
        yield path, array
    for name, child in sorted(group.groups(), key=lambda item: item[0]):
        path = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_arrays(child, path)


def _iter_groups(group: zarr.Group, prefix: str = "") -> Iterable[tuple[str, zarr.Group]]:
    yield prefix, group
    for name, child in sorted(group.groups(), key=lambda item: item[0]):
        path = f"{prefix}/{name}" if prefix else str(name)
        yield from _iter_groups(child, path)


def _require_group(root: zarr.Group, path: str) -> zarr.Group:
    current = root
    for part in (piece for piece in path.split("/") if piece):
        current = current.require_group(part)
    return current


def _array_parent(root: zarr.Group, path: str) -> tuple[zarr.Group, str]:
    parts = path.split("/")
    return _require_group(root, "/".join(parts[:-1])), parts[-1]


def _data_type(array: zarr.Array) -> Any:
    metadata = getattr(array, "metadata", None)
    return getattr(metadata, "data_type", None) or array.dtype


def _effective_shard_rows(requested: int, inner_rows: int) -> int:
    if int(requested) <= 0 or int(inner_rows) <= 0:
        raise ValueError("Shard and inner-chunk row counts must be positive.")
    return int(math.ceil(int(requested) / int(inner_rows)) * int(inner_rows))


def build_sharded_copy_plan(
    source_run: str | Path,
    *,
    row_count_array: str,
    shard_rows: int,
) -> tuple[ShardedArrayPlan, ...]:
    source = zarr.open_group(str(Path(source_run).expanduser()), mode="r", use_consolidated=False)
    if str(source.attrs.get("palette_run_completion_status", "")) != "complete":
        raise ValueError("Sharded copy requires a completed immutable source run.")
    row_node = source.get(str(row_count_array))
    if not isinstance(row_node, zarr.Array) or int(row_node.ndim) < 1:
        raise ValueError(f"Row-count array {row_count_array!r} is missing or invalid.")
    row_count = int(row_node.shape[0])
    plans: list[ShardedArrayPlan] = []
    for path, array in _iter_arrays(source):
        chunks = getattr(array, "chunks", None)
        if not chunks:
            raise ValueError(f"Array {path!r} has no logical chunk contract.")
        inner = tuple(int(value) for value in chunks)
        row_aligned = int(array.ndim) >= 1 and int(array.shape[0]) == row_count
        effective = _effective_shard_rows(int(shard_rows), inner[0]) if row_aligned else None
        plans.append(
            ShardedArrayPlan(
                path=path,
                shape=tuple(int(value) for value in array.shape),
                dtype=str(array.dtype),
                inner_chunks=inner,
                row_aligned=row_aligned,
                requested_shard_rows=int(shard_rows) if row_aligned else None,
                effective_shard_rows=effective,
                outer_shards=(effective, *inner[1:]) if row_aligned else None,
            )
        )
    if not plans:
        raise ValueError("Completed source run contains no arrays.")
    return tuple(plans)


def _create_destination_array(
    source: zarr.Array,
    destination_root: zarr.Group,
    plan: ShardedArrayPlan,
) -> zarr.Array:
    parent, name = _array_parent(destination_root, plan.path)
    kwargs: dict[str, Any] = {
        "shape": source.shape,
        "dtype": _data_type(source),
        "chunks": plan.inner_chunks,
        "overwrite": True,
    }
    if plan.outer_shards is not None:
        kwargs["shards"] = plan.outer_shards
    fill_value = getattr(source, "fill_value", None)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    compressors = getattr(source, "compressors", None)
    if compressors:
        kwargs["compressors"] = compressors
    filters = getattr(source, "filters", None)
    if filters:
        kwargs["filters"] = filters
    serializer = getattr(source, "serializer", None)
    if serializer is not None:
        kwargs["serializer"] = serializer
    destination = parent.create_array(name, **kwargs)
    destination.attrs.update(dict(source.attrs))
    return destination


def _decoded_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    if contiguous.dtype.hasobject:
        raise TypeError("Object arrays are not supported by exact sharded copy validation.")
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _init_worker(source_path: str, destination_path: str) -> None:
    global _SOURCE_ROOT, _DESTINATION_ROOT
    _SOURCE_ROOT = zarr.open_group(source_path, mode="r", use_consolidated=False)
    _DESTINATION_ROOT = zarr.open_group(destination_path, mode="r+", use_consolidated=False)


def _copy_shard_task(task: tuple[str, int, int]) -> dict[str, Any]:
    if _SOURCE_ROOT is None or _DESTINATION_ROOT is None:
        raise RuntimeError("Sharded-copy worker was not initialized.")
    path, start, stop = task
    source = _SOURCE_ROOT[path]
    destination = _DESTINATION_ROOT[path]
    trailing = (slice(None),) * (int(source.ndim) - 1)
    selection = (slice(int(start), int(stop)), *trailing)
    values = np.ascontiguousarray(source[selection])
    source_digest = _decoded_digest(values)
    destination[selection] = values
    destination_digest = _decoded_digest(np.asarray(destination[selection]))
    if destination_digest != source_digest:
        raise RuntimeError(f"Decoded shard validation failed for {path}[{start}:{stop}].")
    return {
        "path": path,
        "start_row": int(start),
        "stop_row": int(stop),
        "decoded_bytes": int(values.nbytes),
        "decoded_sha256": source_digest,
    }


def copy_completed_run_to_sharded(
    source_run: str | Path,
    destination_run: str | Path,
    *,
    row_count_array: str,
    shard_rows: int,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a completed run, validating every decoded outer shard after write."""

    source_path = Path(source_run).expanduser().resolve()
    destination_path = Path(destination_run).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source run not found: {source_path}")
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination run already exists: {destination_path}")
        shutil.rmtree(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    plans = build_sharded_copy_plan(
        source_path,
        row_count_array=row_count_array,
        shard_rows=int(shard_rows),
    )
    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    destination = zarr.open_group(str(destination_path), mode="w", zarr_format=3)
    for group_path, source_group in _iter_groups(source):
        _require_group(destination, group_path).attrs.update(dict(source_group.attrs))

    tasks: list[tuple[str, int, int]] = []
    static_results: list[dict[str, Any]] = []
    for plan in plans:
        source_array = source[plan.path]
        destination_array = _create_destination_array(source_array, destination, plan)
        if plan.row_aligned:
            step = int(plan.effective_shard_rows or 0)
            for start in range(0, int(plan.shape[0]), step):
                tasks.append((plan.path, start, min(start + step, int(plan.shape[0]))))
            continue
        values = np.ascontiguousarray(source_array[...])
        digest = _decoded_digest(values)
        destination_array[...] = values
        if _decoded_digest(np.asarray(destination_array[...])) != digest:
            raise RuntimeError(f"Decoded static-array validation failed for {plan.path}.")
        static_results.append(
            {
                "path": plan.path,
                "decoded_bytes": int(values.nbytes),
                "decoded_sha256": digest,
            }
        )

    worker_count = max(1, int(workers))
    shard_results: list[dict[str, Any]] = []
    if worker_count == 1:
        _init_worker(str(source_path), str(destination_path))
        shard_results = [_copy_shard_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(str(source_path), str(destination_path)),
        ) as executor:
            shard_results = list(executor.map(_copy_shard_task, tasks, chunksize=1))

    effective_values = sorted(
        {int(plan.effective_shard_rows) for plan in plans if plan.effective_shard_rows is not None}
    )
    decoded_bytes_copied = sum(
        int(item["decoded_bytes"]) for item in (*static_results, *shard_results)
    )
    duration_seconds = float(time.perf_counter() - started)
    report = {
        "schema_id": SHARDED_COPY_SCHEMA_ID,
        "status": "complete",
        "source_run": str(source_path),
        "destination_run": str(destination_path),
        "row_count_array": str(row_count_array),
        "requested_shard_rows": int(shard_rows),
        "effective_shard_rows": effective_values,
        "worker_count": worker_count,
        "worker_task_count": len(tasks),
        "worker_ownership": "one_complete_nonoverlapping_outer_row_shard_per_array_task",
        "array_count": len(plans),
        "row_aligned_array_count": sum(1 for plan in plans if plan.row_aligned),
        "static_array_count": sum(1 for plan in plans if not plan.row_aligned),
        "decoded_bytes_copied": decoded_bytes_copied,
        "duration_seconds": duration_seconds,
        "decoded_mib_per_second": (
            float(decoded_bytes_copied / (1024.0**2) / duration_seconds)
            if duration_seconds > 0.0
            else None
        ),
        "exact_decoded_validation": True,
        "arrays": [asdict(plan) for plan in plans],
        "shards": sorted(shard_results, key=lambda item: (item["path"], item["start_row"])),
        "static_arrays": sorted(static_results, key=lambda item: item["path"]),
    }
    destination = zarr.open_group(str(destination_path), mode="r+", use_consolidated=False)
    destination.attrs["physical_storage_layout"] = {
        "schema_id": SHARDED_COPY_SCHEMA_ID,
        "layout": "zarr_v3_indexed_sharding",
        "requested_outer_shard_rows": int(shard_rows),
        "effective_outer_shard_rows": effective_values,
        "worker_ownership": report["worker_ownership"],
        "exact_decoded_validation": True,
    }
    return report
