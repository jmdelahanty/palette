#!/usr/bin/env python3
"""Benchmark bounded subject-shape computation without mutating source runs.

The benchmark reads a contiguous row window from an immutable refined subject-
mask run and writes the real subject-shape products into disposable standalone
Zarr groups.  Variants may change worker count, logical compute-block size,
native-library thread limits, and the exactness-preserving foreground-crop
prototype used by centerline extraction.

Each process owns whole, non-overlapping 256-row physical chunks in the
disposable output.  Source groups are always opened read-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Sequence

import cv2
from threadpoolctl import threadpool_info, threadpool_limits
import zarr

from fisheye.analysis import subject_shape_runs as shape
from fisheye.diagnostics.benchmark_tail_kinematics_sharding import (
    _digest_array,
    _iter_arrays,
    _storage_stats,
)


REPORT_SCHEMA = "palette.subject_shape_compute_benchmark.v1"
OUTPUT_SCHEMA = "palette.subject_shape_compute_benchmark_output.v1"
OUTPUT_LOGICAL_ROW_CHUNK = 256
PROCESS_START_METHOD = "spawn"


@dataclass(frozen=True)
class ComputeVariant:
    name: str
    workers: int
    block_rows: int
    native_threads: int
    persistent_worker_inputs: bool = True
    centerline_crop_to_foreground: bool = False


_WORKER_SOURCE_PATH: str | None = None
_WORKER_OUTPUT_PATH: str | None = None
_WORKER_REFINED_RUN: str | None = None
_WORKER_COMPONENT_INDICES: tuple[tuple[str, int], ...] = ()
_WORKER_SOURCE_ROOT: zarr.Group | None = None
_WORKER_REFINED_GROUP: zarr.Group | None = None
_WORKER_MASK_STORE: shape.MaskStore | None = None
_WORKER_OUTPUT_GROUP: zarr.Group | None = None
_WORKER_NATIVE_LIMITER: Any = None
_WORKER_NATIVE_THREADS = 1
_WORKER_PERSISTENT_INPUTS = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_variant(variant: ComputeVariant) -> None:
    if not variant.name or any(value in variant.name for value in ("/", "\\", "..")):
        raise ValueError(f"Unsafe benchmark variant name: {variant.name!r}.")
    if int(variant.workers) <= 0:
        raise ValueError(f"Variant {variant.name!r} workers must be positive.")
    if int(variant.block_rows) <= 0:
        raise ValueError(f"Variant {variant.name!r} block_rows must be positive.")
    if int(variant.block_rows) % OUTPUT_LOGICAL_ROW_CHUNK != 0:
        raise ValueError(
            f"Variant {variant.name!r} block_rows must be a multiple of "
            f"{OUTPUT_LOGICAL_ROW_CHUNK} for whole-chunk output ownership."
        )
    if int(variant.native_threads) <= 0:
        raise ValueError(f"Variant {variant.name!r} native_threads must be positive.")


def parse_variant(value: str) -> ComputeVariant:
    """Parse NAME:WORKERS:BLOCK_ROWS:NATIVE_THREADS[:FLAGS]."""

    parts = str(value).split(":", maxsplit=4)
    if len(parts) < 4:
        raise argparse.ArgumentTypeError(
            "Variant must be NAME:WORKERS:BLOCK_ROWS:NATIVE_THREADS[:FLAGS]."
        )
    name, workers, block_rows, native_threads = parts[:4]
    flags = set(filter(None, parts[4].split(","))) if len(parts) == 5 else set()
    supported = {"crop", "per-task-open"}
    unknown = flags - supported
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown variant flags: {sorted(unknown)!r}.")
    try:
        variant = ComputeVariant(
            name=name,
            workers=int(workers),
            block_rows=int(block_rows),
            native_threads=int(native_threads),
            persistent_worker_inputs="per-task-open" not in flags,
            centerline_crop_to_foreground="crop" in flags,
        )
        _validate_variant(variant)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return variant


def _resolve_source(
    source_zarr: Path,
    *,
    refined_run: str | None,
    components: Sequence[str] | None,
) -> tuple[str, tuple[tuple[str, int], ...], int]:
    root = shape.open_zarr_root(source_zarr, mode="r")
    resolved_run, _refined_group = shape._resolve_refined_run(root, refined_run)
    tables = shape.load_refined_subject_masks_run_tables(
        root,
        run_name=resolved_run,
        component_names=components,
        include_masks_roi=True,
        include_metrics=False,
        include_components=False,
        include_relations=False,
    )
    component_indices = shape._resolve_components_from_refined_tables(tables, components)
    return resolved_run, tuple(component_indices), int(tables.require_mask_store().n_rows)


def _prepare_output(
    path: Path,
    *,
    row_count: int,
    component_indices: Sequence[tuple[str, int]],
    variant: ComputeVariant,
    source_zarr: Path,
    refined_run: str,
    source_start_row: int,
) -> None:
    output = zarr.open_group(str(path), mode="w", zarr_format=3)
    output.attrs.update(
        {
            "schema_id": OUTPUT_SCHEMA,
            "benchmark_only": True,
            "source_mutation_policy": "read_only",
            "source_zarr": str(source_zarr),
            "source_refined_subject_masks_run": str(refined_run),
            "source_start_row": int(source_start_row),
            "source_stop_row": int(source_start_row) + int(row_count),
            "row_count": int(row_count),
            "variant": asdict(variant),
            "process_start_method": PROCESS_START_METHOD,
            "created_at_utc": _utc_now(),
        }
    )
    component_names = tuple(name for name, _index in component_indices)
    for component_name in component_names:
        shape._prepare_component_group(output, component_name, total_rows=int(row_count))
    shape._prepare_relation_groups(output, component_names, total_rows=int(row_count))
    if set(shape.BODY_FRAME_COMPONENTS).issubset(component_names):
        shape._prepare_body_frame_group(output, total_rows=int(row_count))


def _open_worker_inputs() -> tuple[zarr.Group, shape.MaskStore, zarr.Group]:
    if _WORKER_SOURCE_PATH is None or _WORKER_OUTPUT_PATH is None or _WORKER_REFINED_RUN is None:
        raise RuntimeError("Subject-shape benchmark worker was not initialized.")
    source_root = shape.open_zarr_root(_WORKER_SOURCE_PATH, mode="r")
    refined_group = source_root["refined_subject_masks_runs"][_WORKER_REFINED_RUN]
    mask_store = shape.open_mask_store(
        refined_group,
        source_path=f"refined_subject_masks_runs/{_WORKER_REFINED_RUN}",
        prefer="dense",
    )
    output_group = zarr.open_group(
        _WORKER_OUTPUT_PATH,
        mode="r+",
        use_consolidated=False,
    )
    return refined_group, mask_store, output_group


def _init_worker(
    source_path: str,
    output_path: str,
    refined_run: str,
    component_indices: tuple[tuple[str, int], ...],
    native_threads: int,
    persistent_worker_inputs: bool,
) -> None:
    global _WORKER_SOURCE_PATH
    global _WORKER_OUTPUT_PATH
    global _WORKER_REFINED_RUN
    global _WORKER_COMPONENT_INDICES
    global _WORKER_SOURCE_ROOT
    global _WORKER_REFINED_GROUP
    global _WORKER_MASK_STORE
    global _WORKER_OUTPUT_GROUP
    global _WORKER_NATIVE_LIMITER
    global _WORKER_NATIVE_THREADS
    global _WORKER_PERSISTENT_INPUTS

    _WORKER_SOURCE_PATH = str(source_path)
    _WORKER_OUTPUT_PATH = str(output_path)
    _WORKER_REFINED_RUN = str(refined_run)
    _WORKER_COMPONENT_INDICES = tuple(component_indices)
    _WORKER_NATIVE_THREADS = int(native_threads)
    _WORKER_PERSISTENT_INPUTS = bool(persistent_worker_inputs)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(native_threads))
    _WORKER_NATIVE_LIMITER = threadpool_limits(limits=int(native_threads))
    cv2.setNumThreads(int(native_threads))
    if _WORKER_PERSISTENT_INPUTS:
        _WORKER_SOURCE_ROOT = shape.open_zarr_root(str(source_path), mode="r")
        _WORKER_REFINED_GROUP = _WORKER_SOURCE_ROOT["refined_subject_masks_runs"][str(refined_run)]
        _WORKER_MASK_STORE = shape.open_mask_store(
            _WORKER_REFINED_GROUP,
            source_path=f"refined_subject_masks_runs/{refined_run}",
            prefer="dense",
        )
        _WORKER_OUTPUT_GROUP = zarr.open_group(
            str(output_path),
            mode="r+",
            use_consolidated=False,
        )


def _run_worker_task(task: tuple[int, int, int, int, bool]) -> dict[str, Any]:
    source_start, source_stop, output_start, chunk_index, crop_to_foreground = task
    if _WORKER_PERSISTENT_INPUTS:
        if _WORKER_REFINED_GROUP is None or _WORKER_MASK_STORE is None or _WORKER_OUTPUT_GROUP is None:
            raise RuntimeError("Persistent benchmark worker inputs were not initialized.")
        refined_group = _WORKER_REFINED_GROUP
        mask_store = _WORKER_MASK_STORE
        output_group = _WORKER_OUTPUT_GROUP
    else:
        refined_group, mask_store, output_group = _open_worker_inputs()
    result = shape._process_and_write_subject_shape_chunk_groups(
        refined_group,
        output_group,
        mask_store=mask_store,
        component_indices=_WORKER_COMPONENT_INDICES,
        start_row=int(source_start),
        stop_row=int(source_stop),
        output_start_row=int(output_start),
        chunk_index=int(chunk_index),
        execution_backend="benchmark_process_pool",
        centerline_crop_to_foreground=bool(crop_to_foreground),
    )
    result["native_thread_control"] = (
        {
            "requested_threads": int(_WORKER_NATIVE_THREADS),
            "opencv_threads": int(cv2.getNumThreads()),
            "threadpools": [
                {
                    key: item.get(key)
                    for key in ("user_api", "internal_api", "prefix", "num_threads")
                }
                for item in threadpool_info()
            ],
        }
        if int(chunk_index) == 0
        else None
    )
    return result


def _tasks(
    *,
    source_start_row: int,
    row_count: int,
    block_rows: int,
    crop_to_foreground: bool,
) -> list[tuple[int, int, int, int, bool]]:
    rows: list[tuple[int, int, int, int, bool]] = []
    for chunk_index, output_start in enumerate(range(0, int(row_count), int(block_rows))):
        width = min(int(block_rows), int(row_count) - int(output_start))
        rows.append(
            (
                int(source_start_row) + int(output_start),
                int(source_start_row) + int(output_start) + int(width),
                int(output_start),
                int(chunk_index),
                bool(crop_to_foreground),
            )
        )
    return rows


def _numeric_timing_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    timings = [dict(result["chunk_timing"]) for result in results]
    totals = [float(item["total_seconds"]) for item in timings]
    numeric_keys = sorted(
        {
            key
            for item in timings
            for key, value in item.items()
            if key.endswith("_seconds") and isinstance(value, (int, float))
        }
    )
    ordered = sorted(totals)
    p95_index = min(len(ordered) - 1, max(0, int(math.ceil(0.95 * len(ordered))) - 1))
    return {
        "chunk_count": len(timings),
        "mean_chunk_seconds": float(statistics.mean(totals)),
        "median_chunk_seconds": float(statistics.median(totals)),
        "p95_chunk_seconds": float(ordered[p95_index]),
        "summed_timing_seconds": {
            key: float(sum(float(item.get(key, 0.0)) for item in timings))
            for key in numeric_keys
        },
    }


def _array_digests(group: zarr.Group, *, row_step: int = 256) -> dict[str, str]:
    return {
        path: _digest_array(array, row_step=max(1, int(row_step)))
        for path, array in _iter_arrays(group)
    }


def _compare_digests(reference: dict[str, str], candidate: dict[str, str]) -> dict[str, Any]:
    reference_paths = set(reference)
    candidate_paths = set(candidate)
    mismatches = sorted(
        path
        for path in reference_paths & candidate_paths
        if reference[path] != candidate[path]
    )
    return {
        "reference_only": sorted(reference_paths - candidate_paths),
        "candidate_only": sorted(candidate_paths - reference_paths),
        "digest_mismatches": mismatches,
        "array_count": len(candidate_paths),
        "all_arrays_exact": not mismatches and reference_paths == candidate_paths,
    }


def run_benchmark(
    source_zarr: Path | str,
    *,
    output_root: Path | str,
    variants: Sequence[ComputeVariant],
    refined_run: str | None = None,
    components: Sequence[str] | None = None,
    source_start_row: int = 0,
    row_count: int = 32_768,
    report_path: Path | str | None = None,
    apply: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_zarr).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    chosen_variants = tuple(variants)
    if not chosen_variants:
        raise ValueError("At least one compute variant is required.")
    for variant in chosen_variants:
        _validate_variant(variant)
    names = [variant.name for variant in chosen_variants]
    if len(names) != len(set(names)):
        raise ValueError("Compute variant names must be unique.")
    resolved_run, component_indices, source_rows = _resolve_source(
        source_path,
        refined_run=refined_run,
        components=components,
    )
    start = int(source_start_row)
    count = int(row_count)
    if start < 0 or count <= 0 or start + count > source_rows:
        raise ValueError(
            f"Requested source window [{start}, {start + count}) is outside [0, {source_rows})."
        )
    if start % OUTPUT_LOGICAL_ROW_CHUNK != 0:
        raise ValueError(
            f"source_start_row must align to {OUTPUT_LOGICAL_ROW_CHUNK} rows."
        )
    report_file = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else output_path.with_name(f"{output_path.name}.json")
    )
    report: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA,
        "status": "planned",
        "created_at_utc": _utc_now(),
        "source_zarr": str(source_path),
        "source_mutation_policy": "read_only",
        "source_refined_subject_masks_run": resolved_run,
        "source_total_rows": int(source_rows),
        "source_start_row": start,
        "source_stop_row": start + count,
        "row_count": count,
        "component_indices": [[name, int(index)] for name, index in component_indices],
        "output_root": str(output_path),
        "report_path": str(report_file),
        "variants": [asdict(variant) for variant in chosen_variants],
        "process_start_method": PROCESS_START_METHOD,
        "mutates_source": False,
        "mutates_disposable_output": bool(apply),
    }
    if not apply:
        return report
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Benchmark output root already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    report["status"] = "running"
    report["results"] = []
    _write_report(report, report_file)
    reference_digests: dict[str, str] | None = None
    try:
        for variant in chosen_variants:
            variant_path = output_path / f"{variant.name}.zarr"
            _prepare_output(
                variant_path,
                row_count=count,
                component_indices=component_indices,
                variant=variant,
                source_zarr=source_path,
                refined_run=resolved_run,
                source_start_row=start,
            )
            work = _tasks(
                source_start_row=start,
                row_count=count,
                block_rows=int(variant.block_rows),
                crop_to_foreground=bool(variant.centerline_crop_to_foreground),
            )
            started = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=int(variant.workers),
                mp_context=multiprocessing.get_context(PROCESS_START_METHOD),
                initializer=_init_worker,
                initargs=(
                    str(source_path),
                    str(variant_path),
                    resolved_run,
                    tuple(component_indices),
                    int(variant.native_threads),
                    bool(variant.persistent_worker_inputs),
                ),
            ) as executor:
                results = list(executor.map(_run_worker_task, work, chunksize=1))
            wall_seconds = float(time.perf_counter() - started)
            output_group = zarr.open_group(
                str(variant_path),
                mode="r+",
                use_consolidated=False,
            )
            digest_started = time.perf_counter()
            digests = _array_digests(output_group)
            digest_seconds = float(time.perf_counter() - digest_started)
            if reference_digests is None:
                reference_digests = dict(digests)
            exactness = _compare_digests(reference_digests, digests)
            rows_with_component: dict[str, int] = {}
            for result in results:
                for name, value in dict(result.get("rows_with_component") or {}).items():
                    rows_with_component[str(name)] = rows_with_component.get(str(name), 0) + int(value)
            variant_result = {
                "variant": asdict(variant),
                "output": str(variant_path),
                "task_count": len(work),
                "wall_seconds": wall_seconds,
                "rows_per_second": float(count / wall_seconds) if wall_seconds > 0 else None,
                "timings": _numeric_timing_summary(results),
                "rows_with_component": rows_with_component,
                "native_thread_control": results[0].get("native_thread_control") if results else None,
                "digest_seconds": digest_seconds,
                "array_digests": digests,
                "exactness_vs_first_variant": exactness,
                "storage": _storage_stats(variant_path),
            }
            output_group.attrs["benchmark_completed_at_utc"] = _utc_now()
            output_group.attrs["benchmark_wall_seconds"] = wall_seconds
            output_group.attrs["benchmark_all_arrays_exact"] = bool(exactness["all_arrays_exact"])
            report["results"].append(variant_result)
            _write_report(report, report_file)
    except BaseException as exc:
        report["status"] = "failed"
        report["failed_at_utc"] = _utc_now()
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(report, report_file)
        raise
    report["status"] = "complete"
    report["completed_at_utc"] = _utc_now()
    report["all_variants_exact"] = all(
        bool(result["exactness_vs_first_variant"]["all_arrays_exact"])
        for result in report["results"]
    )
    _write_report(report, report_file)
    return report


def _summary(report: dict[str, Any]) -> str:
    if report["status"] != "complete":
        return f"{report['status']} rows={report['row_count']} variants={len(report['variants'])}"
    lines = [f"complete exact={report['all_variants_exact']} rows={report['row_count']}"]
    for result in report["results"]:
        lines.append(
            "{name}: workers={workers} block={block} native={native} crop={crop} "
            "seconds={seconds:.3f} rows_per_second={rate:.3f} exact={exact}".format(
                name=result["variant"]["name"],
                workers=result["variant"]["workers"],
                block=result["variant"]["block_rows"],
                native=result["variant"]["native_threads"],
                crop=result["variant"]["centerline_crop_to_foreground"],
                seconds=result["wall_seconds"],
                rate=result["rows_per_second"],
                exact=result["exactness_vs_first_variant"]["all_arrays_exact"],
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zarr", type=Path)
    parser.add_argument("--refined-run")
    parser.add_argument("--component", action="append", choices=shape.COMPONENT_ORDER)
    parser.add_argument("--source-start-row", type=int, default=0)
    parser.add_argument("--row-count", type=int, default=32_768)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.source_zarr,
        output_root=args.output_root,
        variants=args.variant,
        refined_run=args.refined_run,
        components=args.component,
        source_start_row=int(args.source_start_row),
        row_count=int(args.row_count),
        report_path=args.report,
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
