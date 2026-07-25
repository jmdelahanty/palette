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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr


SHARDED_COPY_SCHEMA_ID = "palette.zarr_sharded_run_copy.v1"
SHARD_POLICY_ALL_ROW_ALIGNED = "all_row_aligned"
SHARD_POLICY_MULTI_CHUNK_CAPPED = "multi_chunk_capped"
SHARD_POLICIES = (
    SHARD_POLICY_ALL_ROW_ALIGNED,
    SHARD_POLICY_MULTI_CHUNK_CAPPED,
)
STRUCTURED_DTYPE_SINGLE_CHUNK_LAYOUT = (
    "structured_dtype_single_chunk_zarr_v3_sharding_codec_workaround_v1"
)
_SOURCE_ROOT: zarr.Group | None = None
_DESTINATION_ROOT: zarr.Group | None = None


@dataclass(frozen=True)
class ShardedArrayLayout:
    """Optional target chunk/shard grid for one copied array.

    The logical values are unchanged.  ``inner_chunks`` and ``outer_shards``
    describe only the physical Zarr-v3 layout, and every requested outer shard
    dimension is rounded up to the target inner-chunk grid.
    """

    inner_chunks: tuple[int, ...] | None = None
    outer_shards: tuple[int, ...] | None = None
    layout_profile: str | None = None


@dataclass(frozen=True)
class ShardedArrayPlan:
    path: str
    shape: tuple[int, ...]
    dtype: str
    source_chunks: tuple[int, ...]
    requested_inner_chunks: tuple[int, ...]
    inner_chunks: tuple[int, ...]
    row_aligned: bool
    requested_shard_rows: int | None
    effective_shard_rows: int | None
    requested_outer_shards: tuple[int, ...] | None
    outer_shards: tuple[int, ...] | None
    layout_profile: str | None
    shard_policy: str


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


def _normalize_grid(
    requested: Sequence[int],
    *,
    shape: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in requested)
    dimensions = tuple(int(value) for value in shape)
    if len(values) != len(dimensions):
        raise ValueError(
            f"{label} has {len(values)} dimensions for an array with "
            f"{len(dimensions)} dimensions."
        )
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} dimensions must all be positive.")
    return tuple(
        min(value, max(1, dimension))
        for value, dimension in zip(values, dimensions)
    )


def _aligned_outer_grid(
    requested: Sequence[int],
    *,
    inner_chunks: Sequence[int],
    shape: Sequence[int],
) -> tuple[int, ...]:
    requested_grid = tuple(int(value) for value in requested)
    if len(requested_grid) != len(tuple(shape)):
        raise ValueError(
            "Requested outer-shard grid has "
            f"{len(requested_grid)} dimensions for an array with {len(tuple(shape))} dimensions."
        )
    if any(value <= 0 for value in requested_grid):
        raise ValueError("Requested outer-shard dimensions must all be positive.")
    return tuple(
        int(math.ceil(requested_value / inner_value) * inner_value)
        for requested_value, inner_value in zip(requested_grid, inner_chunks)
    )


def build_sharded_copy_plan(
    source_run: str | Path,
    *,
    row_count_array: str | None,
    shard_rows: int,
    array_layouts: Mapping[str, ShardedArrayLayout] | None = None,
    shard_policy: str = SHARD_POLICY_ALL_ROW_ALIGNED,
) -> tuple[ShardedArrayPlan, ...]:
    source = zarr.open_group(str(Path(source_run).expanduser()), mode="r", use_consolidated=False)
    if str(source.attrs.get("palette_run_completion_status", "")) != "complete":
        raise ValueError("Sharded copy requires a completed immutable source run.")
    policy = str(shard_policy)
    if policy not in SHARD_POLICIES:
        raise ValueError(
            f"Unsupported shard_policy={policy!r}; expected one of {SHARD_POLICIES!r}."
        )
    row_count: int | None = None
    if row_count_array is not None:
        row_node = source.get(str(row_count_array))
        if not isinstance(row_node, zarr.Array) or int(row_node.ndim) < 1:
            raise ValueError(f"Row-count array {row_count_array!r} is missing or invalid.")
        row_count = int(row_node.shape[0])
    plans: list[ShardedArrayPlan] = []
    layouts = dict(array_layouts or {})
    observed_paths: set[str] = set()
    for path, array in _iter_arrays(source):
        observed_paths.add(path)
        chunks = getattr(array, "chunks", None)
        if not chunks:
            raise ValueError(f"Array {path!r} has no logical chunk contract.")
        source_chunks = tuple(int(value) for value in chunks)
        layout = layouts.get(path)
        structured_dtype = np.dtype(array.dtype).kind == "V"
        if structured_dtype and layout is not None and layout.outer_shards is not None:
            raise ValueError(
                f"Array {path!r} has a structured dtype that Zarr v3 cannot "
                "safely write through its sharding codec."
            )
        requested_inner = (
            tuple(max(1, int(value)) for value in array.shape)
            if structured_dtype
            else (
                source_chunks
                if layout is None or layout.inner_chunks is None
                else tuple(int(value) for value in layout.inner_chunks)
            )
        )
        inner = _normalize_grid(
            requested_inner,
            shape=array.shape,
            label=f"Requested inner-chunk grid for {path!r}",
        )
        row_aligned = int(array.ndim) >= 1 and (
            row_count is None or int(array.shape[0]) == row_count
        )
        if layout is not None and layout.outer_shards is not None and not row_aligned:
            raise ValueError(
                f"Array {path!r} has an outer-shard override but is not row aligned."
            )
        requested_outer: tuple[int, ...] | None = None
        outer: tuple[int, ...] | None = None
        if row_aligned and not structured_dtype:
            if layout is not None and layout.outer_shards is not None:
                requested_outer = tuple(int(value) for value in layout.outer_shards)
                outer = _aligned_outer_grid(
                    requested_outer,
                    inner_chunks=inner,
                    shape=array.shape,
                )
            elif policy == SHARD_POLICY_ALL_ROW_ALIGNED:
                requested_outer = (int(shard_rows), *inner[1:])
                outer = _aligned_outer_grid(
                    requested_outer,
                    inner_chunks=inner,
                    shape=array.shape,
                )
            else:
                logical_row_chunks = (
                    int(math.ceil(int(array.shape[0]) / int(inner[0])))
                    if int(array.shape[0]) > 0
                    else 0
                )
                if logical_row_chunks > 1:
                    requested_outer = (int(shard_rows), *inner[1:])
                    aligned = _aligned_outer_grid(
                        requested_outer,
                        inner_chunks=inner,
                        shape=array.shape,
                    )
                    maximum_useful_rows = logical_row_chunks * int(inner[0])
                    outer = (
                        min(int(aligned[0]), int(maximum_useful_rows)),
                        *aligned[1:],
                    )
        effective = outer[0] if outer is not None else None
        plans.append(
            ShardedArrayPlan(
                path=path,
                shape=tuple(int(value) for value in array.shape),
                dtype=str(array.dtype),
                source_chunks=source_chunks,
                requested_inner_chunks=requested_inner,
                inner_chunks=inner,
                row_aligned=row_aligned,
                requested_shard_rows=(requested_outer[0] if requested_outer else None),
                effective_shard_rows=effective,
                requested_outer_shards=requested_outer,
                outer_shards=outer,
                layout_profile=(
                    STRUCTURED_DTYPE_SINGLE_CHUNK_LAYOUT
                    if structured_dtype
                    else None if layout is None else layout.layout_profile
                ),
                shard_policy=policy,
            )
        )
    if not plans:
        raise ValueError("Completed source run contains no arrays.")
    unknown_layout_paths = sorted(set(layouts) - observed_paths)
    if unknown_layout_paths:
        raise ValueError(
            "Array-layout overrides reference missing source arrays: "
            f"{unknown_layout_paths}."
        )
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


def _hash_destination_array_task(path: str) -> dict[str, Any]:
    """Hash one complete decoded destination array in bounded row blocks."""

    if _DESTINATION_ROOT is None:
        raise RuntimeError("Sharded-copy hash worker was not initialized.")
    array = _DESTINATION_ROOT[path]
    dtype = np.dtype(array.dtype)
    if dtype.hasobject:
        raise TypeError("Object arrays have no stable decoded-content hash.")
    digest = hashlib.sha256()
    decoded_bytes = 0
    if int(array.ndim) == 0:
        blocks = (np.asarray(array[...]),)
    else:
        outer = getattr(array, "shards", None)
        chunks = getattr(array, "chunks", None)
        block_rows = (
            max(1, int(outer[0]))
            if outer is not None and len(outer) >= 1
            else (
                max(1, int(chunks[0]))
                if chunks is not None and len(chunks) >= 1
                else max(1, int(array.shape[0]))
            )
        )
        trailing = (slice(None),) * (int(array.ndim) - 1)
        blocks = (
            np.asarray(
                array[
                    (
                        slice(start, min(start + block_rows, int(array.shape[0]))),
                        *trailing,
                    )
                ]
            )
            for start in range(0, int(array.shape[0]), block_rows)
        )
    for values in blocks:
        if values.dtype != dtype:
            raise TypeError(f"Decoded dtype changed while hashing {path!r}.")
        payload = np.ascontiguousarray(values).tobytes(order="C")
        digest.update(payload)
        decoded_bytes += len(payload)
    return {
        "path": str(path),
        "decoded_content_bytes": int(decoded_bytes),
        "decoded_content_sha256": digest.hexdigest(),
    }


def copy_completed_run_to_sharded(
    source_run: str | Path,
    destination_run: str | Path,
    *,
    row_count_array: str | None,
    shard_rows: int,
    array_layouts: Mapping[str, ShardedArrayLayout] | None = None,
    shard_policy: str = SHARD_POLICY_ALL_ROW_ALIGNED,
    workers: int = 1,
    compute_full_decoded_content_hashes: bool = False,
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
        array_layouts=array_layouts,
        shard_policy=shard_policy,
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
        if plan.outer_shards is not None:
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

    plan_paths = [str(plan.path) for plan in plans]
    content_hashes: dict[str, dict[str, Any]] = {}
    content_hash_seconds: float | None = None
    if compute_full_decoded_content_hashes:
        content_hash_started = time.perf_counter()
        if worker_count == 1:
            _init_worker(str(source_path), str(destination_path))
            content_hash_results = [
                _hash_destination_array_task(path) for path in plan_paths
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(str(source_path), str(destination_path)),
            ) as executor:
                content_hash_results = list(
                    executor.map(
                        _hash_destination_array_task,
                        plan_paths,
                        chunksize=1,
                    )
                )
        content_hash_seconds = float(time.perf_counter() - content_hash_started)
        content_hashes = {
            str(record["path"]): record for record in content_hash_results
        }
        if set(content_hashes) != set(plan_paths):
            raise RuntimeError("Decoded-content hash inventory differs from copy plan.")

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
        "row_count_array": None if row_count_array is None else str(row_count_array),
        "requested_shard_rows": int(shard_rows),
        "shard_policy": str(shard_policy),
        "effective_shard_rows": effective_values,
        "worker_count": worker_count,
        "worker_task_count": len(tasks),
        "worker_ownership": "one_complete_nonoverlapping_outer_row_shard_per_array_task",
        "array_count": len(plans),
        "row_aligned_array_count": sum(1 for plan in plans if plan.row_aligned),
        "sharded_array_count": sum(
            1 for plan in plans if plan.outer_shards is not None
        ),
        "regular_array_count": sum(
            1 for plan in plans if plan.outer_shards is None
        ),
        "static_array_count": sum(1 for plan in plans if not plan.row_aligned),
        "decoded_bytes_copied": decoded_bytes_copied,
        "duration_seconds": duration_seconds,
        "decoded_mib_per_second": (
            float(decoded_bytes_copied / (1024.0**2) / duration_seconds)
            if duration_seconds > 0.0
            else None
        ),
        "exact_decoded_validation": True,
        "exact_full_decoded_content_hashes": bool(
            compute_full_decoded_content_hashes
        ),
        "full_decoded_content_hash_seconds": content_hash_seconds,
        "full_decoded_content_hash_worker_count": (
            worker_count if compute_full_decoded_content_hashes else 0
        ),
        "array_layout_override_count": len(array_layouts or {}),
        "arrays": [
            (
                {
                    **asdict(plan),
                    "decoded_content_bytes": int(
                        content_hashes[plan.path]["decoded_content_bytes"]
                    ),
                    "decoded_content_sha256": str(
                        content_hashes[plan.path]["decoded_content_sha256"]
                    ),
                }
                if compute_full_decoded_content_hashes
                else asdict(plan)
            )
            for plan in plans
        ],
        "shards": sorted(shard_results, key=lambda item: (item["path"], item["start_row"])),
        "static_arrays": sorted(static_results, key=lambda item: item["path"]),
    }
    destination = zarr.open_group(str(destination_path), mode="r+", use_consolidated=False)
    destination.attrs["physical_storage_layout"] = {
        "schema_id": SHARDED_COPY_SCHEMA_ID,
        "layout": "zarr_v3_indexed_sharding",
        "requested_outer_shard_rows": int(shard_rows),
        "shard_policy": str(shard_policy),
        "effective_outer_shard_rows": effective_values,
        "eligibility": (
            (
                "all_arrays_with_multiple_logical_row_chunks"
                if row_count_array is None
                else f"multiple_logical_row_chunks_and_first_axis_matches:{row_count_array}"
            )
            if str(shard_policy) == SHARD_POLICY_MULTI_CHUNK_CAPPED
            else (
                "all_arrays_with_a_first_axis"
                if row_count_array is None
                else f"first_axis_matches:{row_count_array}"
            )
        ),
        "worker_ownership": report["worker_ownership"],
        "exact_decoded_validation": True,
        "array_layout_overrides": {
            str(path): {
                "inner_chunks": (
                    None if layout.inner_chunks is None else list(layout.inner_chunks)
                ),
                "outer_shards": (
                    None if layout.outer_shards is None else list(layout.outer_shards)
                ),
                "layout_profile": layout.layout_profile,
            }
            for path, layout in sorted((array_layouts or {}).items())
        },
        "effective_overridden_array_layouts": {
            plan.path: {
                "source_chunks": list(plan.source_chunks),
                "requested_inner_chunks": list(plan.requested_inner_chunks),
                "effective_inner_chunks": list(plan.inner_chunks),
                "requested_outer_shards": (
                    None
                    if plan.requested_outer_shards is None
                    else list(plan.requested_outer_shards)
                ),
                "effective_outer_shards": (
                    None if plan.outer_shards is None else list(plan.outer_shards)
                ),
                "layout_profile": plan.layout_profile,
            }
            for plan in plans
            if plan.layout_profile is not None
        },
    }
    return report
