#!/usr/bin/env python3
"""Benchmark physical shard sizes for an immutable tail-kinematics run.

The source run is opened read-only. Candidate layouts are standalone Zarr v3
clones intended for disposable node-local scratch; they preserve every logical
chunk and decoded value while changing only the outer row-shard span of arrays
whose first axis is the run's frame/ROI row axis.

The driver alone creates groups, arrays, attributes, and reports. Copy workers
own complete, non-overlapping outer row-shard stripes in each destination
array. This is both a benchmark and an executable proof of the Palette parallel
Zarr write-ownership contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr


DEFAULT_SHARD_ROWS = (16_384, 65_536, 131_072, 262_144)
DEFAULT_READ_ARRAYS = (
    "frame_index",
    "valid",
    "tail_tip_angle_deg",
    "tail_angle_deg",
    "tail_angle_sample_xy",
)
REPORT_SCHEMA = "palette.tail_kinematics_sharding_benchmark.v1"
_METADATA_NAMES = frozenset({"zarr.json", ".zarray", ".zattrs", ".zgroup"})
_COPY_SOURCE_ROOT: zarr.Group | None = None
_COPY_DESTINATION_ROOT: zarr.Group | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_type(array: zarr.Array) -> Any:
    metadata = getattr(array, "metadata", None)
    return getattr(metadata, "data_type", None) or array.dtype


def _effective_shard_rows(requested: int, inner_rows: int) -> int:
    if int(requested) <= 0:
        raise ValueError("Requested shard rows must be positive.")
    if int(inner_rows) <= 0:
        raise ValueError("Inner chunk rows must be positive.")
    return int(math.ceil(int(requested) / int(inner_rows)) * int(inner_rows))


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


def _decoded_nbytes(array: zarr.Array) -> int:
    return int(math.prod(int(value) for value in array.shape)) * int(np.dtype(array.dtype).itemsize)


def _digest_array(array: zarr.Array, *, row_step: int) -> str:
    digest = hashlib.sha256()
    if int(array.ndim) == 0:
        values = np.ascontiguousarray(array[...])
        digest.update(values.view(np.uint8))
        return digest.hexdigest()
    for start in range(0, int(array.shape[0]), max(1, int(row_step))):
        stop = min(start + max(1, int(row_step)), int(array.shape[0]))
        values = np.ascontiguousarray(array[start:stop, ...])
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class ArrayPlan:
    path: str
    shape: tuple[int, ...]
    dtype: str
    inner_chunks: tuple[int, ...]
    row_aligned: bool
    requested_shard_rows: int | None
    effective_shard_rows: int | None
    outer_shards: tuple[int, ...] | None
    decoded_nbytes: int


def _resolve_row_count(source: zarr.Group) -> int:
    for name in ("frame_index", "valid"):
        node = source.get(name)
        if isinstance(node, zarr.Array) and int(node.ndim) >= 1:
            return int(node.shape[0])
    raise ValueError("Tail-kinematics source must contain frame_index or valid.")


def build_plan(
    source_group: Path | str,
    *,
    shard_rows: int,
) -> tuple[ArrayPlan, ...]:
    source = zarr.open_group(
        str(Path(source_group).expanduser()),
        mode="r",
        use_consolidated=False,
    )
    completion = str(source.attrs.get("palette_run_completion_status", ""))
    if completion != "complete":
        raise ValueError(
            "Tail-kinematics sharding benchmark requires a completed immutable run; "
            f"found palette_run_completion_status={completion!r}."
        )
    row_count = _resolve_row_count(source)
    plans: list[ArrayPlan] = []
    for path, array in _iter_arrays(source):
        chunks = getattr(array, "chunks", None)
        if not chunks:
            raise ValueError(f"Array {path!r} has no logical chunk contract.")
        inner_chunks = tuple(int(value) for value in chunks)
        row_aligned = int(array.ndim) >= 1 and int(array.shape[0]) == int(row_count)
        effective_rows = (
            _effective_shard_rows(int(shard_rows), int(inner_chunks[0]))
            if row_aligned
            else None
        )
        plans.append(
            ArrayPlan(
                path=str(path),
                shape=tuple(int(value) for value in array.shape),
                dtype=str(array.dtype),
                inner_chunks=inner_chunks,
                row_aligned=bool(row_aligned),
                requested_shard_rows=int(shard_rows) if row_aligned else None,
                effective_shard_rows=effective_rows,
                outer_shards=(effective_rows, *inner_chunks[1:]) if row_aligned else None,
                decoded_nbytes=_decoded_nbytes(array),
            )
        )
    if not plans:
        raise ValueError("Tail-kinematics source contains no arrays.")
    return tuple(plans)


def _copy_group_attrs(source: zarr.Group, destination: zarr.Group) -> None:
    for path, source_group in _iter_groups(source):
        target = _require_group(destination, path)
        target.attrs.update(dict(source_group.attrs))


def _create_destination_array(
    source: zarr.Array,
    destination_root: zarr.Group,
    plan: ArrayPlan,
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


def _init_copy_worker(source_path: str, destination_path: str) -> None:
    global _COPY_SOURCE_ROOT, _COPY_DESTINATION_ROOT
    _COPY_SOURCE_ROOT = zarr.open_group(source_path, mode="r", use_consolidated=False)
    _COPY_DESTINATION_ROOT = zarr.open_group(
        destination_path,
        mode="r+",
        use_consolidated=False,
    )


def _copy_row_shard_task(task: tuple[str, int, int]) -> int:
    if _COPY_SOURCE_ROOT is None or _COPY_DESTINATION_ROOT is None:
        raise RuntimeError("Copy worker was not initialized.")
    path, start, stop = task
    source = _COPY_SOURCE_ROOT[path]
    destination = _COPY_DESTINATION_ROOT[path]
    trailing = (slice(None),) * (int(source.ndim) - 1)
    selection = (slice(int(start), int(stop)), *trailing)
    values = np.ascontiguousarray(source[selection])
    destination[selection] = values
    return int(values.nbytes)


def _copy_static_array(source: zarr.Array, destination: zarr.Array) -> int:
    values = np.ascontiguousarray(source[...])
    destination[...] = values
    return int(values.nbytes)


def _build_candidate(
    source_path: Path,
    destination_path: Path,
    *,
    plans: Sequence[ArrayPlan],
    workers: int,
) -> dict[str, Any]:
    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    destination = zarr.open_group(str(destination_path), mode="w", zarr_format=3)
    started = time.perf_counter()
    _copy_group_attrs(source, destination)
    destination.attrs.update(
        {
            "benchmark_only": True,
            "benchmark_schema": REPORT_SCHEMA,
            "benchmark_source_group": str(source_path),
            "benchmark_created_at_utc": _utc_now(),
        }
    )

    static_bytes = 0
    tasks: list[tuple[str, int, int]] = []
    for plan in plans:
        source_array = source[plan.path]
        destination_array = _create_destination_array(source_array, destination, plan)
        if not plan.row_aligned:
            static_bytes += _copy_static_array(source_array, destination_array)
            continue
        shard_rows = int(plan.effective_shard_rows or 0)
        for start in range(0, int(plan.shape[0]), shard_rows):
            tasks.append((plan.path, start, min(start + shard_rows, int(plan.shape[0]))))

    copied_bytes = int(static_bytes)
    worker_count = max(1, int(workers))
    if worker_count == 1:
        _init_copy_worker(str(source_path), str(destination_path))
        for task in tasks:
            copied_bytes += _copy_row_shard_task(task)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_copy_worker,
            initargs=(str(source_path), str(destination_path)),
        ) as executor:
            for result in executor.map(_copy_row_shard_task, tasks, chunksize=1):
                copied_bytes += int(result)
    elapsed = float(time.perf_counter() - started)
    return {
        "write_seconds": elapsed,
        "worker_count": worker_count,
        "worker_task_count": len(tasks),
        "worker_ownership": "one_complete_nonoverlapping_outer_row_shard_stripe_per_task",
        "decoded_bytes_copied": copied_bytes,
        "decoded_mib_per_second": (
            float(copied_bytes / (1024.0**2) / elapsed) if elapsed > 0 else None
        ),
    }


def _selected_read_arrays(root: zarr.Group, requested: Sequence[str]) -> tuple[str, ...]:
    available = {path for path, _array in _iter_arrays(root)}
    selected = tuple(path for path in requested if path in available)
    if not selected:
        raise ValueError(
            "None of the requested benchmark read arrays are present: "
            f"requested={list(requested)!r}."
        )
    return selected


def _hash_values(digest: Any, values: Any) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(contiguous.view(np.uint8))


def _read_pattern_once(
    root: zarr.Group,
    arrays: Sequence[str],
    *,
    pattern: str,
    random_rows: int,
    window_rows: int,
    window_count: int,
    scan_rows: int,
    seed: int,
) -> tuple[float, str, int]:
    digest = hashlib.sha256()
    decoded_bytes = 0
    started = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    for path in arrays:
        array = root[path]
        row_count = int(array.shape[0])
        if pattern == "random_rows":
            rows = rng.integers(0, row_count, size=min(int(random_rows), row_count))
            for row in rows:
                values = np.ascontiguousarray(array[int(row), ...])
                _hash_values(digest, values)
                decoded_bytes += int(values.nbytes)
        elif pattern == "contiguous_windows":
            width = min(max(1, int(window_rows)), row_count)
            starts = rng.integers(0, max(1, row_count - width + 1), size=max(1, int(window_count)))
            for start in starts:
                values = np.ascontiguousarray(array[int(start) : int(start) + width, ...])
                _hash_values(digest, values)
                decoded_bytes += int(values.nbytes)
        elif pattern == "full_scan":
            for start in range(0, row_count, max(1, int(scan_rows))):
                values = np.ascontiguousarray(
                    array[start : min(start + max(1, int(scan_rows)), row_count), ...]
                )
                _hash_values(digest, values)
                decoded_bytes += int(values.nbytes)
        else:
            raise ValueError(f"Unsupported read pattern: {pattern}")
    return float(time.perf_counter() - started), digest.hexdigest(), decoded_bytes


def _read_benchmark(
    root: zarr.Group,
    *,
    requested_arrays: Sequence[str],
    repeats: int,
    random_rows: int,
    window_rows: int,
    window_count: int,
    scan_rows: int,
    seed: int,
) -> dict[str, Any]:
    arrays = _selected_read_arrays(root, requested_arrays)
    patterns: dict[str, Any] = {}
    for pattern_index, pattern in enumerate(("random_rows", "contiguous_windows", "full_scan")):
        seconds: list[float] = []
        digests: list[str] = []
        decoded_bytes = 0
        for repeat in range(max(1, int(repeats))):
            elapsed, digest, trial_bytes = _read_pattern_once(
                root,
                arrays,
                pattern=pattern,
                random_rows=int(random_rows),
                window_rows=int(window_rows),
                window_count=int(window_count),
                scan_rows=int(scan_rows),
                seed=int(seed) + pattern_index * 1000 + repeat,
            )
            seconds.append(elapsed)
            digests.append(digest)
            decoded_bytes = int(trial_bytes)
        median_seconds = float(statistics.median(seconds))
        patterns[pattern] = {
            "seconds": seconds,
            "median_seconds": median_seconds,
            "digests": digests,
            "decoded_bytes_per_trial": decoded_bytes,
            "median_decoded_mib_per_second": (
                float(decoded_bytes / (1024.0**2) / median_seconds)
                if median_seconds > 0
                else None
            ),
        }
    return {
        "arrays": list(arrays),
        "repeats": max(1, int(repeats)),
        "cache_semantics": "node_local_cache_warm_or_mixed; do not interpret as a cold-cache benchmark",
        "random_row_count_per_array": int(random_rows),
        "contiguous_window_rows": int(window_rows),
        "contiguous_window_count_per_array": int(window_count),
        "full_scan_block_rows": int(scan_rows),
        "patterns": patterns,
    }


def _validate_candidate(
    source_digests: dict[str, str],
    destination: zarr.Group,
    plans: Sequence[ArrayPlan],
    *,
    digest_rows: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for plan in plans:
        array = destination[plan.path]
        digest = _digest_array(array, row_step=int(digest_rows))
        expected = source_digests[plan.path]
        rows.append(
            {
                "path": plan.path,
                "source_sha256": expected,
                "destination_sha256": digest,
                "exact_match": digest == expected,
            }
        )
    return {
        "all_arrays_exact": all(bool(row["exact_match"]) for row in rows),
        "array_results": rows,
    }


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)


def _transfer_candidate(
    source_path: Path,
    destination_path: Path,
    *,
    remove_after_validation: bool,
) -> dict[str, Any]:
    if destination_path.exists():
        raise FileExistsError(f"Transfer benchmark destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_storage = _storage_stats(source_path)
    started = time.perf_counter()
    subprocess.run(
        ["rsync", "-a", "--", f"{source_path}/", f"{destination_path}/"],
        check=True,
    )
    transfer_seconds = float(time.perf_counter() - started)
    destination_storage = _storage_stats(destination_path)

    verification_started = time.perf_counter()
    verification = subprocess.run(
        [
            "rsync",
            "-a",
            "-n",
            "-c",
            "--delete",
            "--itemize-changes",
            "--",
            f"{source_path}/",
            f"{destination_path}/",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    verification_seconds = float(time.perf_counter() - verification_started)
    physical_changes = [line for line in verification.stdout.splitlines() if line.strip()]
    storage_matches = all(
        int(source_storage[name]) == int(destination_storage[name])
        for name in ("file_count", "payload_file_count", "apparent_bytes")
    )
    if physical_changes or not storage_matches:
        raise RuntimeError(
            "Transfer benchmark checksum verification failed: "
            f"changes={physical_changes[:5]!r}, storage_matches={storage_matches}."
        )

    result = {
        "source": str(source_path),
        "destination": str(destination_path),
        "method": "rsync_archive_then_checksum_dry_run",
        "transfer_seconds": transfer_seconds,
        "verification_seconds": verification_seconds,
        "physical_files_exact": True,
        "source_storage": source_storage,
        "destination_storage": destination_storage,
        "apparent_mib_per_second": (
            float(destination_storage["apparent_bytes"] / (1024.0**2) / transfer_seconds)
            if transfer_seconds > 0
            else None
        ),
        "files_per_second": (
            float(destination_storage["file_count"] / transfer_seconds)
            if transfer_seconds > 0
            else None
        ),
        "removed_after_validation": bool(remove_after_validation),
    }
    if remove_after_validation:
        shutil.rmtree(destination_path)
    return result


def run_benchmark(
    source_group: Path | str,
    *,
    output_root: Path | str,
    shard_rows: Sequence[int] = DEFAULT_SHARD_ROWS,
    workers: int = 8,
    read_repeats: int = 3,
    read_arrays: Sequence[str] = DEFAULT_READ_ARRAYS,
    random_rows: int = 32,
    window_rows: int = 1024,
    window_count: int = 8,
    scan_rows: int = 16_384,
    digest_rows: int = 16_384,
    report_path: Path | str | None = None,
    transfer_root: Path | str | None = None,
    remove_transfer_copies: bool = True,
    apply: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_group).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    candidates = tuple(dict.fromkeys(int(value) for value in shard_rows))
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("At least one positive candidate shard-row value is required.")
    if int(workers) <= 0:
        raise ValueError("workers must be positive.")
    plans_by_candidate = {
        candidate: build_plan(source_path, shard_rows=candidate) for candidate in candidates
    }
    chosen_report_path = (
        Path(report_path).expanduser()
        if report_path is not None
        else output_path.with_name(f"{output_path.name}.benchmark.json")
    )
    chosen_transfer_root = (
        Path(transfer_root).expanduser().resolve() if transfer_root is not None else None
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": "planned",
        "source_group": str(source_path),
        "output_root": str(output_path),
        "report_path": str(chosen_report_path),
        "candidate_shard_rows": list(candidates),
        "worker_count_requested": int(workers),
        "source_mutation_policy": "read_only",
        "candidate_storage_policy": "disposable_node_local_only",
        "transfer_benchmark_root": (
            str(chosen_transfer_root) if chosen_transfer_root is not None else None
        ),
        "transfer_copy_retention": (
            "remove_after_checksum_validation"
            if chosen_transfer_root is not None and remove_transfer_copies
            else "retain"
            if chosen_transfer_root is not None
            else "not_requested"
        ),
        "array_plans": {
            str(candidate): [asdict(plan) for plan in plans]
            for candidate, plans in plans_by_candidate.items()
        },
    }
    if not apply:
        return report
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Benchmark output root already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    report["status"] = "running"
    _write_report(report, chosen_report_path)

    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    source_plans = plans_by_candidate[candidates[0]]
    digest_started = time.perf_counter()
    source_digests = {
        plan.path: _digest_array(source[plan.path], row_step=int(digest_rows))
        for plan in source_plans
    }
    report["source_digest_seconds"] = float(time.perf_counter() - digest_started)
    report["source_storage"] = _storage_stats(source_path)
    report["source_array_digests"] = source_digests
    report["source_read_benchmark"] = _read_benchmark(
        source,
        requested_arrays=read_arrays,
        repeats=int(read_repeats),
        random_rows=int(random_rows),
        window_rows=int(window_rows),
        window_count=int(window_count),
        scan_rows=int(scan_rows),
        seed=20260715,
    )
    report["variants"] = []
    _write_report(report, chosen_report_path)

    try:
        for candidate in candidates:
            destination_path = output_path / f"shard_rows_{candidate}.zarr"
            plans = plans_by_candidate[candidate]
            build_result = _build_candidate(
                source_path,
                destination_path,
                plans=plans,
                workers=int(workers),
            )
            destination = zarr.open_group(
                str(destination_path),
                mode="r",
                use_consolidated=False,
            )
            validation_started = time.perf_counter()
            validation = _validate_candidate(
                source_digests,
                destination,
                plans,
                digest_rows=int(digest_rows),
            )
            validation_seconds = float(time.perf_counter() - validation_started)
            if not validation["all_arrays_exact"]:
                raise RuntimeError(f"Decoded digest mismatch for candidate shard_rows={candidate}.")
            storage = _storage_stats(destination_path)
            read_result = _read_benchmark(
                destination,
                requested_arrays=read_arrays,
                repeats=int(read_repeats),
                random_rows=int(random_rows),
                window_rows=int(window_rows),
                window_count=int(window_count),
                scan_rows=int(scan_rows),
                seed=20260715,
            )
            source_patterns = report["source_read_benchmark"]["patterns"]
            read_ratios = {
                name: (
                    float(values["median_seconds"] / source_patterns[name]["median_seconds"])
                    if float(source_patterns[name]["median_seconds"]) > 0
                    else None
                )
                for name, values in read_result["patterns"].items()
            }
            report["variants"].append(
                {
                    "requested_shard_rows": int(candidate),
                    "destination": str(destination_path),
                    **build_result,
                    "validation_seconds": validation_seconds,
                    **validation,
                    "storage": storage,
                    "read_benchmark": read_result,
                    "read_median_seconds_over_staged_source": read_ratios,
                }
            )
            _write_report(report, chosen_report_path)

        if chosen_transfer_root is not None:
            if chosen_transfer_root.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Transfer benchmark root already exists: {chosen_transfer_root}"
                    )
                shutil.rmtree(chosen_transfer_root)
            chosen_transfer_root.mkdir(parents=True, exist_ok=False)
            report["transfer_benchmark_status"] = "running"
            _write_report(report, chosen_report_path)
            for variant in report["variants"]:
                candidate = int(variant["requested_shard_rows"])
                variant["transfer_benchmark"] = _transfer_candidate(
                    Path(variant["destination"]),
                    chosen_transfer_root / f"shard_rows_{candidate}.zarr",
                    remove_after_validation=bool(remove_transfer_copies),
                )
                _write_report(report, chosen_report_path)
            if remove_transfer_copies:
                chosen_transfer_root.rmdir()
            report["transfer_benchmark_status"] = "complete"
            _write_report(report, chosen_report_path)
    except BaseException as exc:
        report["status"] = "failed"
        report["failed_at_utc"] = _utc_now()
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(report, chosen_report_path)
        raise

    report["status"] = "complete"
    report["completed_at_utc"] = _utc_now()
    report["all_variants_exact"] = all(
        bool(variant["all_arrays_exact"]) for variant in report["variants"]
    )
    _write_report(report, chosen_report_path)
    return report


def _summary(report: dict[str, Any]) -> str:
    if report["status"] != "complete":
        return (
            f"{report['status']} candidates={report['candidate_shard_rows']} "
            f"output={report['output_root']}"
        )
    rows = []
    for variant in report["variants"]:
        rows.append(
            "shard_rows={requested_shard_rows} files={files} payload={payload} "
            "write_s={seconds:.3f} full_scan_s={scan:.3f}".format(
                requested_shard_rows=variant["requested_shard_rows"],
                files=variant["storage"]["file_count"],
                payload=variant["storage"]["payload_file_count"],
                seconds=variant["write_seconds"],
                scan=variant["read_benchmark"]["patterns"]["full_scan"]["median_seconds"],
            )
        )
    return "complete exact={}\n{}".format(report["all_variants_exact"], "\n".join(rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_group", type=Path, help="Completed tail-kinematics run-group path.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--shard-rows",
        type=int,
        action="append",
        dest="shard_rows",
        help="Candidate outer row-shard span; repeat for multiple candidates.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--read-repeats", type=int, default=3)
    parser.add_argument("--read-array", action="append", dest="read_arrays")
    parser.add_argument("--random-rows", type=int, default=32)
    parser.add_argument("--window-rows", type=int, default=1024)
    parser.add_argument("--window-count", type=int, default=8)
    parser.add_argument("--scan-rows", type=int, default=16_384)
    parser.add_argument("--digest-rows", type=int, default=16_384)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--transfer-root",
        type=Path,
        help="Optional destination root for timed, checksum-validated rsync publication trials.",
    )
    parser.add_argument(
        "--keep-transfer-copies",
        action="store_true",
        help="Retain transfer trial copies instead of removing each after validation.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.source_group,
        output_root=args.output_root,
        shard_rows=args.shard_rows or DEFAULT_SHARD_ROWS,
        workers=int(args.workers),
        read_repeats=int(args.read_repeats),
        read_arrays=args.read_arrays or DEFAULT_READ_ARRAYS,
        random_rows=int(args.random_rows),
        window_rows=int(args.window_rows),
        window_count=int(args.window_count),
        scan_rows=int(args.scan_rows),
        digest_rows=int(args.digest_rows),
        report_path=args.report,
        transfer_root=args.transfer_root,
        remove_transfer_copies=not bool(args.keep_transfer_copies),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
