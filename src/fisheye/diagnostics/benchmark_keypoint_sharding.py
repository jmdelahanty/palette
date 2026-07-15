#!/usr/bin/env python3
"""Clone one immutable keypoint run into aligned Zarr v3 shards and benchmark it.

The source group is never modified. Default mode is dry-run; ``--apply`` writes
the standalone benchmark clone and an adjacent JSON report.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr


DEFAULT_ROI_SHARD_ROWS = 131_072
DEFAULT_FRAME_SHARD_ROWS = 131_072
REPORT_SCHEMA = "palette.keypoint_sharding_canary.v1"
_METADATA_NAMES = frozenset({"zarr.json", ".zarray", ".zattrs", ".zgroup"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.benchmark.json")


def _data_type(array: zarr.Array) -> Any:
    metadata = getattr(array, "metadata", None)
    return getattr(metadata, "data_type", None) or array.dtype


def _effective_shard_rows(requested: int, inner_rows: int) -> int:
    if requested <= 0:
        raise ValueError("Requested shard rows must be positive.")
    if inner_rows <= 0:
        raise ValueError("Inner chunk rows must be positive.")
    return int(math.ceil(int(requested) / int(inner_rows)) * int(inner_rows))


def _digest_array(array: zarr.Array, *, row_step: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, int(array.shape[0]), int(row_step)):
        stop = min(start + int(row_step), int(array.shape[0]))
        values = np.ascontiguousarray(array[start:stop, ...])
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _storage_stats(path: Path) -> dict[str, int]:
    totals = {
        "file_count": 0,
        "metadata_file_count": 0,
        "payload_file_count": 0,
        "apparent_bytes": 0,
        "allocated_bytes": 0,
    }
    for root, _directories, filenames in os.walk(path):
        for filename in filenames:
            stat_result = (Path(root) / filename).stat()
            totals["file_count"] += 1
            if filename in _METADATA_NAMES:
                totals["metadata_file_count"] += 1
            else:
                totals["payload_file_count"] += 1
            totals["apparent_bytes"] += int(stat_result.st_size)
            totals["allocated_bytes"] += int(getattr(stat_result, "st_blocks", 0)) * 512
    return totals


def _payload_file_count(path: Path) -> int:
    return int(_storage_stats(path)["payload_file_count"])


@dataclass(frozen=True)
class ArrayPlan:
    name: str
    domain: str
    shape: tuple[int, ...]
    dtype: str
    inner_chunks: tuple[int, ...]
    outer_shards: tuple[int, ...]


def build_plan(
    source_group: Path | str,
    *,
    roi_shard_rows: int = DEFAULT_ROI_SHARD_ROWS,
    frame_shard_rows: int = DEFAULT_FRAME_SHARD_ROWS,
) -> tuple[ArrayPlan, ...]:
    source = zarr.open_group(str(Path(source_group).expanduser()), mode="r", use_consolidated=False)
    child_groups = [name for name, _group in source.groups()]
    if child_groups:
        raise ValueError(f"Keypoint canary expects a flat run group; found child groups {child_groups}.")
    if "keypoints_roi" not in source:
        raise ValueError("Source group does not contain keypoints_roi.")
    roi_rows = int(source["keypoints_roi"].shape[0])
    plans: list[ArrayPlan] = []
    for name, array in sorted(source.arrays(), key=lambda item: item[0]):
        if int(array.ndim) < 1:
            raise ValueError(f"Array {name!r} has no row axis.")
        chunks = getattr(array, "chunks", None)
        if not chunks:
            raise ValueError(f"Array {name!r} has no inner chunk contract.")
        inner_chunks = tuple(int(value) for value in chunks)
        domain = "roi" if int(array.shape[0]) == roi_rows else "frame"
        requested_rows = int(roi_shard_rows if domain == "roi" else frame_shard_rows)
        shard_rows = _effective_shard_rows(requested_rows, inner_chunks[0])
        plans.append(
            ArrayPlan(
                name=str(name),
                domain=domain,
                shape=tuple(int(value) for value in array.shape),
                dtype=str(array.dtype),
                inner_chunks=inner_chunks,
                outer_shards=(shard_rows, *inner_chunks[1:]),
            )
        )
    return tuple(plans)


def _create_destination_array(
    source: zarr.Array,
    destination_group: zarr.Group,
    plan: ArrayPlan,
) -> zarr.Array:
    kwargs: dict[str, Any] = {
        "shape": source.shape,
        "dtype": _data_type(source),
        "chunks": plan.inner_chunks,
        "shards": plan.outer_shards,
        "overwrite": True,
    }
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
    destination = destination_group.create_array(plan.name, **kwargs)
    destination.attrs.update(dict(source.attrs))
    return destination


def _copy_array_by_complete_shard(
    source: zarr.Array,
    destination: zarr.Array,
    plan: ArrayPlan,
) -> tuple[str, float]:
    digest = hashlib.sha256()
    started = time.perf_counter()
    shard_rows = int(plan.outer_shards[0])
    trailing = (slice(None),) * (int(source.ndim) - 1)
    for start in range(0, int(source.shape[0]), shard_rows):
        stop = min(start + shard_rows, int(source.shape[0]))
        selection = (slice(start, stop), *trailing)
        values = np.ascontiguousarray(source[selection])
        digest.update(values.view(np.uint8))
        destination[selection] = values
    return digest.hexdigest(), float(time.perf_counter() - started)


def _timed(callable_) -> tuple[float, float]:
    started = time.perf_counter()
    value = np.asarray(callable_())
    elapsed = float(time.perf_counter() - started)
    checksum = float(np.nan_to_num(value, copy=False).sum(dtype=np.float64))
    return elapsed, checksum


def _read_trials(array: zarr.Array, *, repeats: int, seed: int) -> dict[str, Any]:
    row_count = int(array.shape[0])
    random_rows = np.random.default_rng(seed).integers(0, row_count, size=min(256, row_count))
    range_rows = min(1024, row_count)
    range_start = max(0, (row_count - range_rows) // 2)
    trials: dict[str, list[float]] = {"random_rows": [], "range_1024": [], "full_scan": []}
    checksums: dict[str, list[float]] = {key: [] for key in trials}
    for _repeat in range(max(1, int(repeats))):
        elapsed, checksum = _timed(
            lambda: np.stack([np.asarray(array[int(row), ...]) for row in random_rows])
        )
        trials["random_rows"].append(elapsed)
        checksums["random_rows"].append(checksum)
        elapsed, checksum = _timed(lambda: array[range_start : range_start + range_rows, ...])
        trials["range_1024"].append(elapsed)
        checksums["range_1024"].append(checksum)
        elapsed, checksum = _timed(lambda: array[:])
        trials["full_scan"].append(elapsed)
        checksums["full_scan"].append(checksum)
    return {
        name: {
            "seconds": values,
            "median_seconds": float(statistics.median(values)),
            "checksums": checksums[name],
        }
        for name, values in trials.items()
    }


def _read_benchmark(
    source: zarr.Group,
    destination: zarr.Group,
    *,
    repeats: int,
) -> dict[str, Any]:
    source_trials = _read_trials(source["keypoints_roi"], repeats=repeats, seed=20260712)
    destination_trials = _read_trials(destination["keypoints_roi"], repeats=repeats, seed=20260712)
    comparisons: dict[str, Any] = {}
    for name in source_trials:
        regular = float(source_trials[name]["median_seconds"])
        sharded = float(destination_trials[name]["median_seconds"])
        comparisons[name] = {
            "source_median_seconds": regular,
            "sharded_median_seconds": sharded,
            "sharded_over_source": float(sharded / regular) if regular > 0 else None,
            "checksums_match": source_trials[name]["checksums"] == destination_trials[name]["checksums"],
        }
    return {
        "array": "keypoints_roi",
        "repeats": int(repeats),
        "source": source_trials,
        "sharded": destination_trials,
        "comparisons": comparisons,
    }


def run_canary(
    source_group: Path | str,
    *,
    destination: Path | str,
    roi_shard_rows: int = DEFAULT_ROI_SHARD_ROWS,
    frame_shard_rows: int = DEFAULT_FRAME_SHARD_ROWS,
    read_repeats: int = 5,
    apply: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_group).expanduser()
    destination_path = Path(destination).expanduser()
    plans = build_plan(
        source_path,
        roi_shard_rows=roi_shard_rows,
        frame_shard_rows=frame_shard_rows,
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": "planned",
        "source_group": str(source_path),
        "destination": str(destination_path),
        "roi_shard_rows_requested": int(roi_shard_rows),
        "frame_shard_rows_requested": int(frame_shard_rows),
        "array_plans": [asdict(plan) for plan in plans],
    }
    if not apply:
        return report
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        shutil.rmtree(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    destination_group = zarr.open_group(str(destination_path), mode="w", zarr_format=3)
    destination_group.attrs.update(dict(source.attrs))
    destination_group.attrs.update(
        {
            "benchmark_only": True,
            "benchmark_schema": REPORT_SCHEMA,
            "benchmark_source_group": str(source_path),
            "benchmark_created_at_utc": report["created_at_utc"],
        }
    )

    array_results: list[dict[str, Any]] = []
    clone_started = time.perf_counter()
    for plan in plans:
        source_array = source[plan.name]
        destination_array = _create_destination_array(source_array, destination_group, plan)
        source_digest, copy_seconds = _copy_array_by_complete_shard(
            source_array,
            destination_array,
            plan,
        )
        validation_started = time.perf_counter()
        destination_digest = _digest_array(
            destination_array,
            row_step=int(plan.outer_shards[0]),
        )
        validation_seconds = float(time.perf_counter() - validation_started)
        if source_digest != destination_digest:
            raise RuntimeError(
                f"Decoded digest mismatch for {plan.name}: "
                f"source={source_digest} destination={destination_digest}"
            )
        array_results.append(
            {
                **asdict(plan),
                "source_sha256": source_digest,
                "destination_sha256": destination_digest,
                "exact_match": True,
                "copy_seconds": copy_seconds,
                "validation_seconds": validation_seconds,
                "source_payload_files": _payload_file_count(source_path / plan.name),
                "destination_payload_files": _payload_file_count(destination_path / plan.name),
            }
        )
    clone_seconds = float(time.perf_counter() - clone_started)

    report.update(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "clone_and_validate_seconds": clone_seconds,
            "all_arrays_exact": all(bool(row["exact_match"]) for row in array_results),
            "array_results": array_results,
            "source_storage": _storage_stats(source_path),
            "destination_storage": _storage_stats(destination_path),
            "read_benchmark": _read_benchmark(
                source,
                destination_group,
                repeats=max(1, int(read_repeats)),
            ),
        }
    )
    report_path = _report_path(destination_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _summary(report: dict[str, Any]) -> str:
    if report["status"] != "complete":
        return (
            f"planned arrays={len(report['array_plans'])} destination={report['destination']} "
            "(add --apply to write)"
        )
    source = report["source_storage"]
    destination = report["destination_storage"]
    return (
        f"complete exact={report['all_arrays_exact']} arrays={len(report['array_results'])} "
        f"files={source['file_count']}->{destination['file_count']} "
        f"payload_files={source['payload_file_count']}->{destination['payload_file_count']} "
        f"bytes={source['apparent_bytes']}->{destination['apparent_bytes']} "
        f"seconds={report['clone_and_validate_seconds']:.3f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_group", type=Path, help="Completed keypoint run-group directory.")
    parser.add_argument("--destination", required=True, type=Path, help="Standalone benchmark Zarr path.")
    parser.add_argument("--roi-shard-rows", type=int, default=DEFAULT_ROI_SHARD_ROWS)
    parser.add_argument("--frame-shard-rows", type=int, default=DEFAULT_FRAME_SHARD_ROWS)
    parser.add_argument("--read-repeats", type=int, default=5)
    parser.add_argument("--apply", action="store_true", help="Write and validate the benchmark clone.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing benchmark destination.")
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON.")
    args = parser.parse_args(argv)
    report = run_canary(
        args.source_group,
        destination=args.destination,
        roi_shard_rows=int(args.roi_shard_rows),
        frame_shard_rows=int(args.frame_shard_rows),
        read_repeats=int(args.read_repeats),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
