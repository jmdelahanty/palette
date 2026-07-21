#!/usr/bin/env python3
"""Benchmark complete eye-angle layouts in one disposable node allocation.

The authoritative recording is opened read-only and its exact eye-angle input
surface is staged once to node-local storage.  The real scientific writer then
runs in A/B/B/A order against that same staged source.  Every trial is
validated, copied into its production sharded layout, validated again, and
compared by name-normalized decoded digests before its disposable outputs are
removed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.analysis import eye_angle_analysis as eye_writer
from fisheye.analysis_workflows.materializers.eye_angles import (
    _iter_arrays,
    _sealed_output_identity_digests,
    _validate_eye_angle_run,
    audit_eye_angle_source_revision,
    build_eye_angle_materialization_plan,
    stage_eye_angle_sources,
)
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.run_provenance import git_identity, runtime_context, scheduler_context
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_sharded_copy import (
    ShardedArrayLayout,
    copy_completed_run_to_sharded,
)


REPORT_SCHEMA_ID = "palette.eye_angle_materialization_layout_benchmark.v1"
DEFAULT_ORDER = ("all_columns", "semantic_16", "semantic_16", "all_columns")
INDEXED_TABLES = {
    "roi_angles": "angle_channel_index",
    "frame_angles": "angle_channel_index",
    "roi_qa": "qa_channel_index",
    "frame_qa": "qa_channel_index",
    "roi_vectors": "vector_channel_index",
    "frame_vectors": "vector_channel_index",
}
INDEX_GROUPS = frozenset(INDEXED_TABLES.values())


@dataclass(frozen=True)
class LayoutVariant:
    name: str
    angle_chunk_rows: int
    angle_chunk_columns: int
    output_shard_rows: int
    angle_shard_columns: int


VARIANTS = {
    "all_columns": LayoutVariant(
        name="all_columns",
        angle_chunk_rows=8_192,
        angle_chunk_columns=141,
        output_shard_rows=262_144,
        angle_shard_columns=141,
    ),
    "semantic_16": LayoutVariant(
        name="semantic_16",
        angle_chunk_rows=4_096,
        angle_chunk_columns=16,
        output_shard_rows=131_072,
        angle_shard_columns=32,
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_attr_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _decode_names(group: zarr.Group, index_path: str) -> tuple[str, ...]:
    index = group.get(index_path)
    if not isinstance(index, zarr.Group) or "name" not in index:
        raise ValueError(f"Missing channel-name index: {index_path}/name")
    values = np.asarray(index["name"][:])
    names = tuple(str(decode_null_terminated_text(value)) for value in values)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Invalid or duplicate channel names in {index_path}/name")
    return names


def _update_array_digest(
    digest: Any,
    array: zarr.Array,
    *,
    row_step: int,
    column_order: Sequence[int] | None = None,
) -> None:
    digest.update(str(np.dtype(array.dtype)).encode("utf-8"))
    digest.update(json.dumps([int(value) for value in array.shape]).encode("ascii"))
    if int(array.ndim) == 0:
        digest.update(np.ascontiguousarray(array[...]).tobytes())
        return
    for start in range(0, int(array.shape[0]), max(1, int(row_step))):
        values = np.asarray(array[start : min(start + row_step, int(array.shape[0]))])
        if column_order is not None:
            values = values[:, list(column_order), ...]
        digest.update(np.ascontiguousarray(values).tobytes())


def logical_run_digests(
    run_path: str | Path,
    *,
    row_step: int = 8_192,
) -> dict[str, str]:
    """Hash decoded products while normalizing named table column order."""

    run = open_zarr_root(run_path, mode="r")
    names_by_index = {
        index_path: _decode_names(run, index_path)
        for index_path in INDEX_GROUPS
        if index_path in run
    }
    digests: dict[str, str] = {}
    for array_path, array in _iter_arrays(run):
        if any(array_path.startswith(f"{group}/") for group in INDEX_GROUPS):
            continue
        digest = hashlib.sha256()
        digest.update(array_path.encode("utf-8"))
        index_path = INDEXED_TABLES.get(array_path)
        column_order: tuple[int, ...] | None = None
        if index_path is not None:
            names = names_by_index[index_path]
            if int(array.ndim) < 2 or int(array.shape[1]) != len(names):
                raise ValueError(f"Named table shape/index mismatch: {array_path}")
            canonical_names = tuple(sorted(names))
            digest.update("\0".join(canonical_names).encode("utf-8"))
            column_order = tuple(names.index(name) for name in canonical_names)
        _update_array_digest(
            digest,
            array,
            row_step=row_step,
            column_order=column_order,
        )
        digests[array_path] = digest.hexdigest()
    return digests


def _compare_digests(
    reference: Mapping[str, str],
    candidate: Mapping[str, str],
) -> dict[str, Any]:
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
        "all_arrays_exact": (
            not mismatches and reference_paths == candidate_paths
        ),
    }


def _storage_stats(path: Path) -> dict[str, int]:
    file_count = 0
    apparent_bytes = 0
    allocated_bytes = 0
    for directory, _children, filenames in os.walk(path):
        for filename in filenames:
            stat = (Path(directory) / filename).stat()
            file_count += 1
            apparent_bytes += int(stat.st_size)
            allocated_bytes += int(getattr(stat, "st_blocks", 0)) * 512
    return {
        "file_count": file_count,
        "apparent_bytes": apparent_bytes,
        "allocated_bytes": allocated_bytes,
    }


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(
        own.ru_utime
        + own.ru_stime
        + children.ru_utime
        + children.ru_stime
    )


def _validation_errors(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(error)
        for error in payload.get("errors", [])
        if str(error) != "node_local_materialization provenance is missing"
    ]


def _summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant_name in VARIANTS:
        trials = [item for item in results if item["variant"]["name"] == variant_name]
        if not trials:
            continue
        phase_names = sorted(
            {
                name
                for item in trials
                for name in item["writer_timing_summary"].get("phase_seconds", {})
            }
        )
        summary[variant_name] = {
            "trial_count": len(trials),
            "median_writer_wall_seconds": float(
                statistics.median(float(item["writer_wall_seconds"]) for item in trials)
            ),
            "median_writer_cpu_seconds": float(
                statistics.median(float(item["writer_cpu_seconds"]) for item in trials)
            ),
            "median_sharding_seconds": float(
                statistics.median(float(item["sharding"]["duration_seconds"]) for item in trials)
            ),
            "median_phase_seconds": {
                name: float(
                    statistics.median(
                        float(item["writer_timing_summary"]["phase_seconds"].get(name, 0.0))
                        for item in trials
                    )
                )
                for name in phase_names
            },
        }
    return summary


def run_benchmark(
    source_zarr: str | Path,
    *,
    output_root: str | Path,
    report_path: str | Path,
    benchmark_id: str,
    subject_shape_run: str | None,
    keypoint_run: str | None,
    order: Sequence[str] = DEFAULT_ORDER,
    chunk_rows: int = 8_192,
    num_workers: int = 8,
    shard_workers: int = 8,
    native_threads: int = 1,
    copy_backend: str = "rsync",
) -> dict[str, Any]:
    """Execute a full-duration, same-source A/B/B/A benchmark."""

    if not benchmark_id or any(value in benchmark_id for value in ("/", "\\", "..")):
        raise ValueError(f"Unsafe benchmark ID: {benchmark_id!r}")
    unknown = [name for name in order if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown layout variants: {unknown}")
    output = Path(output_root).expanduser().resolve()
    report = Path(report_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing benchmark output root: {output}")
    if str(output).startswith("/groups/"):
        raise ValueError("Benchmark output root must use node-local storage, not /groups.")

    probe_name = f"eye_layout_benchmark_{benchmark_id}"
    plan = build_eye_angle_materialization_plan(
        source_zarr,
        scratch_root=output,
        subject_shape_run=subject_shape_run,
        keypoint_run=keypoint_run,
        run_name=probe_name,
        chunk_rows=chunk_rows,
        angle_chunk_rows=VARIANTS["semantic_16"].angle_chunk_rows,
        angle_chunk_columns=VARIANTS["semantic_16"].angle_chunk_columns,
        output_shard_rows=VARIANTS["semantic_16"].output_shard_rows,
        angle_shard_columns=VARIANTS["semantic_16"].angle_shard_columns,
        num_workers=num_workers,
        shard_workers=shard_workers,
        native_threads=native_threads,
    )
    sealed_identity = _sealed_output_identity_digests(
        plan.staged_input_integrity_receipt
    )
    source_before = audit_eye_angle_source_revision(plan)
    if source_before["status"] != "current":
        raise RuntimeError(f"Source revision is not current: {source_before}")
    staging = stage_eye_angle_sources(
        plan,
        copy_backend=copy_backend,
        check_capacity=True,
    )
    result: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "status": "running",
        "benchmark_id": benchmark_id,
        "created_at_utc": _utc_now(),
        "source_zarr": str(plan.source_zarr),
        "source_access": "read_only",
        "mutates_source": False,
        "staged_zarr": str(plan.staged_zarr),
        "staging": staging,
        "subject_shape_run": plan.subject_shape_run,
        "keypoint_run": plan.keypoint_run,
        "row_count": plan.row_count,
        "frame_count": plan.frame_count,
        "chunk_rows": int(chunk_rows),
        "num_workers": int(num_workers),
        "shard_workers": int(shard_workers),
        "native_threads": int(native_threads),
        "order": list(order),
        "runtime": runtime_context(),
        "scheduler": scheduler_context(),
        "git": git_identity(),
        "trials": [],
    }
    _write_report(report, result)
    reference_digests: dict[str, str] | None = None
    try:
        for trial_index, variant_name in enumerate(order, start=1):
            variant = VARIANTS[variant_name]
            run_name = f"benchmark_{trial_index:02d}_{variant.name}"
            run_path = (
                plan.staged_zarr / "analysis" / "eye_angle_runs" / run_name
            )
            sharded_path = output / f"trial-{trial_index:02d}-sharded"
            writer_argv = [
                str(plan.staged_zarr),
                "--subject-shape-run",
                plan.subject_shape_run,
                "--keypoint-run",
                plan.keypoint_run,
                "--run-name",
                run_name,
                "--chunk-size",
                str(chunk_rows),
                "--dense-chunk-rows",
                str(variant.angle_chunk_rows),
                "--dense-chunk-columns",
                str(variant.angle_chunk_columns),
                "--execution-backend",
                eye_writer.DASK_WORKER_EXECUTION_BACKEND,
                "--scheduler",
                "processes",
                "--num-workers",
                str(num_workers),
                "--layout",
                eye_writer.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
                "--quiet",
            ]
            if plan.fps is not None:
                writer_argv.extend(("--fps", str(plan.fps)))

            writer_cpu_started = _cpu_seconds()
            writer_started = time.perf_counter()
            eye_writer.main(
                writer_argv,
                _staged_input_integrity_receipt=(
                    plan.staged_input_integrity_receipt
                ),
            )
            writer_wall_seconds = float(time.perf_counter() - writer_started)
            writer_cpu_seconds = float(_cpu_seconds() - writer_cpu_started)

            run_group = open_zarr_root(run_path, mode="a")
            timing_summary = dict(run_group.attrs.get("eye_angle_timing_summary", {}))
            run_group.attrs["node_local_materialization"] = {
                "schema_id": REPORT_SCHEMA_ID,
                "benchmark_only": True,
                "trial_index": trial_index,
                "variant": asdict(variant),
            }
            validation_started = time.perf_counter()
            regular_validation = _validate_eye_angle_run(
                run_path,
                row_count=plan.row_count,
                frame_count=plan.frame_count,
                expected_source_contract_sha256=plan.source_contract_sha256,
                expected_instance_key_sha256=sealed_identity["instance_key"],
                expected_acquisition_frame_index_sha256=sealed_identity[
                    "source_acquisition_frame_index"
                ],
                require_sharded=False,
                expected_angle_chunk_rows=variant.angle_chunk_rows,
                expected_angle_chunk_columns=variant.angle_chunk_columns,
                expected_angle_shard_rows=variant.output_shard_rows,
                expected_angle_shard_columns=variant.angle_shard_columns,
            )
            regular_validation_seconds = float(
                time.perf_counter() - validation_started
            )
            regular_errors = _validation_errors(regular_validation)
            if regular_errors:
                raise RuntimeError(f"Regular validation failed: {regular_errors}")

            digest_started = time.perf_counter()
            digests = logical_run_digests(run_path)
            digest_seconds = float(time.perf_counter() - digest_started)
            if reference_digests is None:
                reference_digests = digests
            exactness = _compare_digests(reference_digests, digests)
            if not exactness["all_arrays_exact"]:
                raise RuntimeError(f"Logical output mismatch: {exactness}")

            sharding = copy_completed_run_to_sharded(
                run_path,
                sharded_path,
                row_count_array=None,
                shard_rows=variant.output_shard_rows,
                array_layouts={
                    array_name: ShardedArrayLayout(
                        inner_chunks=(
                            variant.angle_chunk_rows,
                            variant.angle_chunk_columns,
                        ),
                        outer_shards=(
                            variant.output_shard_rows,
                            variant.angle_shard_columns,
                        ),
                        layout_profile=eye_writer.EYE_ANGLE_COLUMN_ORDER_PROFILE,
                    )
                    for array_name in ("roi_angles", "frame_angles")
                },
                workers=shard_workers,
            )
            sharded_validation_started = time.perf_counter()
            sharded_validation = _validate_eye_angle_run(
                sharded_path,
                row_count=plan.row_count,
                frame_count=plan.frame_count,
                expected_source_contract_sha256=plan.source_contract_sha256,
                expected_instance_key_sha256=sealed_identity["instance_key"],
                expected_acquisition_frame_index_sha256=sealed_identity[
                    "source_acquisition_frame_index"
                ],
                require_sharded=True,
                expected_angle_chunk_rows=variant.angle_chunk_rows,
                expected_angle_chunk_columns=variant.angle_chunk_columns,
                expected_angle_shard_rows=variant.output_shard_rows,
                expected_angle_shard_columns=variant.angle_shard_columns,
            )
            sharded_validation_seconds = float(
                time.perf_counter() - sharded_validation_started
            )
            sharded_errors = _validation_errors(sharded_validation)
            if sharded_errors:
                raise RuntimeError(f"Sharded validation failed: {sharded_errors}")

            sharding_summary = {
                key: value
                for key, value in sharding.items()
                if key not in {"arrays", "shards", "static_arrays"}
            }
            sharding_summary["angle_array_layouts"] = [
                item
                for item in sharding["arrays"]
                if item["path"] in {"roi_angles", "frame_angles"}
            ]
            trial = {
                "trial_index": trial_index,
                "variant": asdict(variant),
                "run_name": run_name,
                "writer_wall_seconds": writer_wall_seconds,
                "writer_cpu_seconds": writer_cpu_seconds,
                "writer_timing_summary": timing_summary,
                "regular_validation_seconds": regular_validation_seconds,
                "logical_digest_seconds": digest_seconds,
                "logical_exactness_vs_first_trial": exactness,
                "regular_storage": _storage_stats(run_path),
                "sharding": sharding_summary,
                "sharded_validation_seconds": sharded_validation_seconds,
                "sharded_storage": _storage_stats(sharded_path),
            }
            result["trials"].append(trial)
            result["summary"] = _summary(result["trials"])
            _write_report(report, result)
            del run_group
            shutil.rmtree(run_path)
            shutil.rmtree(sharded_path)

        source_after = audit_eye_angle_source_revision(plan)
        if source_after["status"] != "current":
            raise RuntimeError(f"Source changed during benchmark: {source_after}")
        result.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "all_trials_logically_exact": True,
                "source_revision_after": source_after,
                "summary": _summary(result["trials"]),
            }
        )
        _write_report(report, result)
        return result
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_report(report, result)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zarr", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--subject-shape-run", required=True)
    parser.add_argument("--keypoint-run", required=True)
    parser.add_argument("--order", default=",".join(DEFAULT_ORDER))
    parser.add_argument("--chunk-rows", type=int, default=8_192)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shard-workers", type=int, default=8)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.apply:
        raise SystemExit("Refusing to execute without --apply.")
    if not os.environ.get("LSB_JOBID"):
        raise SystemExit("Refusing benchmark execution outside an LSF allocation.")
    report = run_benchmark(
        args.source_zarr,
        output_root=args.output_root,
        report_path=args.report,
        benchmark_id=args.benchmark_id,
        subject_shape_run=args.subject_shape_run,
        keypoint_run=args.keypoint_run,
        order=tuple(value.strip() for value in args.order.split(",") if value.strip()),
        chunk_rows=args.chunk_rows,
        num_workers=args.num_workers,
        shard_workers=args.shard_workers,
        native_threads=args.native_threads,
        copy_backend=args.copy_backend,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(Path(args.report).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
