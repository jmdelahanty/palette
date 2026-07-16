#!/usr/bin/env python3
"""Benchmark aligned outer sharding for a completed columnar Zarr run.

The source is opened read-only. Each candidate is a disposable, serially
written clone that preserves the source's logical chunks while applying the
same row-shard policy as :mod:`fisheye.shared.zarr.columnar`. Validation is
bounded: metadata, shape, dtype, logical chunks, and representative values are
checked without performing a second full decoded-value comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import (
    COLUMNAR_SHARD_ALIGNMENT_POLICY,
    DEFAULT_COLUMNAR_SHARD_ROWS,
    pick_shards,
)


REPORT_SCHEMA = "palette.columnar_zarr_sharding_benchmark.v1"
DEFAULT_CANDIDATE_SHARD_ROWS = (65_536, 131_072, 262_144, 524_288)
_METADATA_NAMES = frozenset({"zarr.json", ".zarray", ".zattrs", ".zgroup"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _storage_stats(path: Path) -> dict[str, Any]:
    payload_sizes: list[int] = []
    totals: dict[str, Any] = {
        "file_count": 0,
        "metadata_file_count": 0,
        "payload_file_count": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
    }
    for root, _directories, filenames in os.walk(path):
        for filename in filenames:
            stat_result = (Path(root) / filename).stat()
            size = int(stat_result.st_size)
            totals["file_count"] += 1
            if filename in _METADATA_NAMES:
                totals["metadata_file_count"] += 1
            else:
                totals["payload_file_count"] += 1
                payload_sizes.append(size)
            totals["apparent_bytes"] += size
            totals["allocated_bytes"] += int(getattr(stat_result, "st_blocks", 0)) * 512
    if payload_sizes:
        ordered = sorted(payload_sizes)
        p95_index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
        totals["payload_size_bytes"] = {
            "minimum": int(ordered[0]),
            "median": int(statistics.median(ordered)),
            "p95": int(ordered[p95_index]),
            "maximum": int(ordered[-1]),
        }
    else:
        totals["payload_size_bytes"] = None
    return totals


def _metadata_digest(path: Path) -> str:
    digest = hashlib.sha256()
    metadata_paths = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name in _METADATA_NAMES
    )
    for candidate in metadata_paths:
        digest.update(str(candidate.relative_to(path)).encode("utf-8"))
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _decoded_nbytes(array: zarr.Array) -> int:
    return int(math.prod(int(value) for value in array.shape)) * int(np.dtype(array.dtype).itemsize)


def _copy_group_attrs(source: zarr.Group, destination: zarr.Group) -> None:
    for path, source_group in _iter_groups(source):
        target = _require_group(destination, path)
        target.attrs.update(dict(source_group.attrs))


def _sample_selections(array: zarr.Array) -> tuple[Any, ...]:
    if int(array.ndim) == 0:
        return (Ellipsis,)
    row_count = int(array.shape[0])
    if row_count <= 0:
        return ()
    rows = tuple(dict.fromkeys((0, row_count // 2, row_count - 1)))
    trailing = (slice(None),) * (int(array.ndim) - 1)
    return tuple((int(row), *trailing) for row in rows)


def _exact_values_match(left: Any, right: Any) -> bool:
    left_array = np.ascontiguousarray(left)
    right_array = np.ascontiguousarray(right)
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes() == right_array.tobytes()
    )


def _create_candidate(
    source_path: Path,
    destination_path: Path,
    *,
    shard_rows: int | None,
) -> dict[str, Any]:
    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    destination = zarr.open_group(str(destination_path), mode="w", zarr_format=3)
    _copy_group_attrs(source, destination)
    destination.attrs.update(
        {
            "benchmark_only": True,
            "benchmark_schema": REPORT_SCHEMA,
            "benchmark_source_group": str(source_path),
            "benchmark_created_at_utc": _utc_now(),
            "benchmark_shard_rows_requested": shard_rows,
            "benchmark_shard_alignment_policy": COLUMNAR_SHARD_ALIGNMENT_POLICY,
        }
    )

    started = time.perf_counter()
    decoded_bytes = 0
    sharded_arrays = 0
    regular_arrays = 0
    layout_rows: list[dict[str, Any]] = []
    for path, source_array in _iter_arrays(source):
        chunks = tuple(int(value) for value in source_array.chunks)
        shape = tuple(int(value) for value in source_array.shape)
        shards = pick_shards(shape, chunks, shard_rows=shard_rows)
        values = np.asarray(source_array[...])
        parent, name = _array_parent(destination, path)
        kwargs: dict[str, Any] = {
            "data": values,
            "chunks": chunks,
            "overwrite": True,
        }
        if shards is not None:
            kwargs["shards"] = shards
            sharded_arrays += 1
        else:
            regular_arrays += 1
        target = parent.create_array(name, **kwargs)
        target.attrs.update(dict(source_array.attrs))
        decoded_bytes += int(values.nbytes)
        layout_rows.append(
            {
                "path": path,
                "shape": list(shape),
                "logical_chunks": list(chunks),
                "outer_shards": list(shards) if shards is not None else None,
            }
        )
    elapsed = float(time.perf_counter() - started)
    return {
        "write_seconds": elapsed,
        "decoded_bytes_copied": int(decoded_bytes),
        "decoded_mib_per_second": (
            float(decoded_bytes / (1024.0**2) / elapsed) if elapsed > 0 else None
        ),
        "writer_concurrency": "serial_whole_array",
        "sharded_array_count": int(sharded_arrays),
        "regular_array_count": int(regular_arrays),
        "arrays": layout_rows,
    }


def _validate_candidate(source: zarr.Group, candidate: zarr.Group) -> dict[str, Any]:
    failures: list[str] = []
    checked_values = 0
    source_arrays = {path: array for path, array in _iter_arrays(source)}
    candidate_arrays = {path: array for path, array in _iter_arrays(candidate)}
    if set(source_arrays) != set(candidate_arrays):
        failures.append("array_path_set_mismatch")
    for path in sorted(set(source_arrays) & set(candidate_arrays)):
        source_array = source_arrays[path]
        candidate_array = candidate_arrays[path]
        if tuple(source_array.shape) != tuple(candidate_array.shape):
            failures.append(f"{path}:shape")
        if np.dtype(source_array.dtype) != np.dtype(candidate_array.dtype):
            failures.append(f"{path}:dtype")
        if tuple(source_array.chunks) != tuple(candidate_array.chunks):
            failures.append(f"{path}:logical_chunks")
        for selection in _sample_selections(source_array):
            if not _exact_values_match(source_array[selection], candidate_array[selection]):
                failures.append(f"{path}:sample:{selection!r}")
            checked_values += 1
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "sample_selection_count": int(checked_values),
        "validation_scope": "paths_shape_dtype_logical_chunks_and_first_middle_last_rows",
    }


def _read_scan(root: zarr.Group, *, block_rows: int) -> dict[str, Any]:
    started = time.perf_counter()
    decoded_bytes = 0
    read_operations = 0
    for _path, array in _iter_arrays(root):
        if int(array.ndim) == 0:
            values = np.asarray(array[...])
            decoded_bytes += int(values.nbytes)
            read_operations += 1
            continue
        row_count = int(array.shape[0])
        trailing = (slice(None),) * (int(array.ndim) - 1)
        for start in range(0, row_count, max(1, int(block_rows))):
            values = np.asarray(
                array[(slice(start, min(start + int(block_rows), row_count)), *trailing)]
            )
            decoded_bytes += int(values.nbytes)
            read_operations += 1
    elapsed = float(time.perf_counter() - started)
    return {
        "seconds": elapsed,
        "decoded_bytes": int(decoded_bytes),
        "read_operations": int(read_operations),
        "decoded_mib_per_second": (
            float(decoded_bytes / (1024.0**2) / elapsed) if elapsed > 0 else None
        ),
        "block_rows": int(block_rows),
        "cache_semantics": "node_local_warm_or_mixed",
    }


def _read_windows(root: zarr.Group, *, window_rows: int) -> dict[str, Any]:
    started = time.perf_counter()
    decoded_bytes = 0
    read_operations = 0
    array_count = 0
    for _path, array in _iter_arrays(root):
        if int(array.ndim) == 0 or int(array.shape[0]) <= int(array.chunks[0]):
            continue
        row_count = int(array.shape[0])
        width = min(max(1, int(window_rows)), row_count)
        starts = tuple(
            dict.fromkeys((0, max(0, (row_count - width) // 2), row_count - width))
        )
        trailing = (slice(None),) * (int(array.ndim) - 1)
        array_count += 1
        for start in starts:
            values = np.asarray(array[(slice(start, start + width), *trailing)])
            decoded_bytes += int(values.nbytes)
            read_operations += 1
    elapsed = float(time.perf_counter() - started)
    return {
        "seconds": elapsed,
        "decoded_bytes": int(decoded_bytes),
        "read_operations": int(read_operations),
        "array_count": int(array_count),
        "decoded_mib_per_second": (
            float(decoded_bytes / (1024.0**2) / elapsed) if elapsed > 0 else None
        ),
        "window_rows": int(window_rows),
        "windows_per_array": "first_middle_last",
        "cache_semantics": "node_local_warm_or_mixed",
    }


def run_benchmark(
    source_group: Path | str,
    *,
    output_root: Path | str,
    shard_rows: Sequence[int] = DEFAULT_CANDIDATE_SHARD_ROWS,
    scan_rows: int = 65_536,
    window_rows: int = 1_024,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_group).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    if output_path == source_path or source_path in output_path.parents:
        raise ValueError("Benchmark output_root must not be inside the read-only source group.")
    candidates = tuple(dict.fromkeys(int(value) for value in shard_rows))
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("At least one positive candidate shard-row value is required.")
    if int(scan_rows) <= 0:
        raise ValueError("scan_rows must be positive.")
    if int(window_rows) <= 0:
        raise ValueError("window_rows must be positive.")

    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    completion = str(source.attrs.get("palette_run_completion_status", ""))
    if completion != "complete":
        raise ValueError(
            "Columnar sharding benchmark requires a completed immutable run; "
            f"found palette_run_completion_status={completion!r}."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    source_metadata_before = _metadata_digest(source_path)
    source_storage = _storage_stats(source_path)

    variants: list[dict[str, Any]] = []
    for requested in (None, *candidates):
        label = "regular" if requested is None else f"shard_rows_{requested}"
        destination_path = output_path / f"{source_path.name}__{label}.zarr"
        if destination_path.exists():
            if not overwrite:
                raise FileExistsError(f"Benchmark candidate already exists: {destination_path}")
            shutil.rmtree(destination_path)
        write = _create_candidate(
            source_path,
            destination_path,
            shard_rows=requested,
        )
        candidate = zarr.open_group(
            str(destination_path), mode="r", use_consolidated=False
        )
        validation = _validate_candidate(source, candidate)
        if not validation["passed"]:
            raise RuntimeError(
                f"Candidate validation failed for {label}: {validation['failures'][:5]!r}"
            )
        variants.append(
            {
                "label": label,
                "requested_shard_rows": requested,
                "path": str(destination_path),
                "write": write,
                "storage": _storage_stats(destination_path),
                "validation": validation,
                "bounded_windows": _read_windows(
                    candidate, window_rows=int(window_rows)
                ),
                "full_scan": _read_scan(candidate, block_rows=int(scan_rows)),
            }
        )

    source_metadata_after = _metadata_digest(source_path)
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": _utc_now(),
        "source_group": str(source_path),
        "source_open_mode": "read_only",
        "source_completion_status": completion,
        "source_storage": source_storage,
        "source_metadata_sha256_before": source_metadata_before,
        "source_metadata_sha256_after": source_metadata_after,
        "source_metadata_unchanged": source_metadata_before == source_metadata_after,
        "alignment_policy": COLUMNAR_SHARD_ALIGNMENT_POLICY,
        "production_default_shard_rows": DEFAULT_COLUMNAR_SHARD_ROWS,
        "scan_rows": int(scan_rows),
        "window_rows": int(window_rows),
        "variants": variants,
    }
    report_path = output_path / f"{source_path.name}.benchmark.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_group", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--shard-rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_SHARD_ROWS),
    )
    parser.add_argument("--scan-rows", type=int, default=65_536)
    parser.add_argument("--window-rows", type=int, default=1_024)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(
        args.source_group,
        output_root=args.output_root,
        shard_rows=args.shard_rows,
        scan_rows=args.scan_rows,
        window_rows=args.window_rows,
        overwrite=bool(args.overwrite),
    )
    summary = {
        "report_path": report["report_path"],
        "source_metadata_unchanged": report["source_metadata_unchanged"],
        "variants": [
            {
                "label": row["label"],
                "payload_file_count": row["storage"]["payload_file_count"],
                "apparent_bytes": row["storage"]["apparent_bytes"],
                "write_seconds": row["write"]["write_seconds"],
                "bounded_window_seconds": row["bounded_windows"]["seconds"],
                "full_scan_seconds": row["full_scan"]["seconds"],
            }
            for row in report["variants"]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
