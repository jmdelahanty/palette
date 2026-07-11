"""Measure read-only process-worker initialization for a subject-mask collection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import resource
import socket
import time
from typing import Any, Sequence

import numpy as np

from fisheye.refinement.finalize_subject_masks import (
    _SubjectMaskCollectionWorkerPlan,
    _build_collection_worker_plan,
    _collection_worker_plan_summary,
    _load_subject_mask_source,
    _resolve_eye_assignment_context,
)
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.zarr_io import open_zarr_root


def _proc_memory_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator or key not in {"VmRSS", "VmHWM", "VmSize", "VmSwap"}:
                continue
            fields = raw_value.strip().split()
            if fields:
                values[str(key)] = int(fields[0])
    values["ru_maxrss"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return values


def _worker_initialize_collection(
    zarr_path: str,
    *,
    subject_shard_runs: tuple[str, ...],
    target_crop_run: str,
    collection_worker_plan: _SubjectMaskCollectionWorkerPlan,
    assignment_keypoint_group: str | None,
    assignment_keypoints_run: str | None,
    sample_start: int,
    sample_rows: int,
    hold_seconds: float,
    worker_index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_before = _proc_memory_kib()
    root = open_zarr_root(zarr_path, mode="r")
    source, collection = _load_subject_mask_source(
        root,
        subject_run=None,
        subject_shard_runs=subject_shard_runs,
        target_crop_run=target_crop_run,
        collection_worker_plan=collection_worker_plan,
    )
    source_loaded_seconds = float(time.perf_counter() - started)
    memory_after_source = _proc_memory_kib()

    assignment_summary: dict[str, Any] | None = None
    assignment_context = None
    if assignment_keypoint_group and assignment_keypoints_run:
        assignment_started = time.perf_counter()
        assignment_context = _resolve_eye_assignment_context(
            root,
            source,
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
        )
        assignment_summary = {
            "seconds": float(time.perf_counter() - assignment_started),
            "row_identity": dict(assignment_context.row_identity_summary),
            "keypoint_success_rows": int(assignment_context.keypoint_success.shape[0]),
        }
    memory_after_assignment = _proc_memory_kib()

    stop = min(int(source.masks_roi.shape[0]), int(sample_start) + max(1, int(sample_rows)))
    sample_started = time.perf_counter()
    masks = np.asarray(source.masks_roi[int(sample_start) : int(stop)])
    mask_checksum = int(masks.sum(dtype=np.uint64))
    keypoint_checksum = None
    if assignment_context is not None:
        keypoints = np.asarray(assignment_context.keypoints_roi[int(sample_start) : int(stop)])
        keypoint_checksum = float(np.nansum(keypoints, dtype=np.float64))
    sample_seconds = float(time.perf_counter() - sample_started)
    memory_after_sample = _proc_memory_kib()

    if float(hold_seconds) > 0:
        time.sleep(float(hold_seconds))
    memory_after_hold = _proc_memory_kib()
    if collection is None:
        raise RuntimeError("Worker did not resolve a subject-mask shard collection.")
    return {
        "worker_index": int(worker_index),
        "pid": int(os.getpid()),
        "hostname": socket.gethostname(),
        "sample_start": int(sample_start),
        "sample_stop": int(stop),
        "sample_rows": int(stop - int(sample_start)),
        "source_loaded_seconds": source_loaded_seconds,
        "assignment": assignment_summary,
        "sample_seconds": sample_seconds,
        "mask_checksum": mask_checksum,
        "keypoint_checksum": keypoint_checksum,
        "hold_seconds": float(hold_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "memory_kib": {
            "before": memory_before,
            "after_source": memory_after_source,
            "after_assignment": memory_after_assignment,
            "after_sample": memory_after_sample,
            "after_hold": memory_after_hold,
        },
        "collection_worker_plan": _collection_worker_plan_summary(collection_worker_plan),
    }


def benchmark_collection_worker_initialization(
    zarr_path: Path | str,
    *,
    source_refined_run: str,
    num_workers: int = 8,
    sample_rows: int = 1,
    hold_seconds: float = 15.0,
    output_json: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(zarr_path).expanduser().resolve()
    parent_started = time.perf_counter()
    parent_memory_before = _proc_memory_kib()
    root = open_zarr_root(path, mode="r")
    source_refined = root.get(f"refined_subject_masks_runs/{source_refined_run}")
    if source_refined is None:
        raise ValueError(f"refined_subject_masks_runs/{source_refined_run} not found.")
    attrs = source_refined.attrs
    shard_runs = tuple(str(value) for value in attrs.get("source_subject_mask_shard_runs") or ())
    target_crop_run = str(
        attrs.get("source_crop_rebase_target_run") or attrs.get("source_crop_run") or ""
    )
    assignment_keypoint_group = str(attrs.get("assignment_keypoint_group") or "") or None
    assignment_keypoints_run = str(attrs.get("assignment_keypoints_run") or "") or None
    if not shard_runs or not target_crop_run:
        raise ValueError("Source refined run does not identify its shard collection and target crop run.")

    source, collection = _load_subject_mask_source(
        root,
        subject_run=None,
        subject_shard_runs=shard_runs,
        target_crop_run=target_crop_run,
    )
    if collection is None:
        raise RuntimeError("Parent did not resolve a subject-mask shard collection.")
    plan = _build_collection_worker_plan(collection)
    plan_summary = _collection_worker_plan_summary(plan)
    parent_plan_seconds = float(time.perf_counter() - parent_started)
    parent_memory_after_plan = _proc_memory_kib()
    total_rows = int(source.masks_roi.shape[0])
    worker_count = max(1, min(int(num_workers), total_rows))
    starts = [(total_rows * worker_index) // worker_count for worker_index in range(worker_count)]
    print(
        f"parent_plan rows={total_rows} bytes={plan_summary['array_bytes'] if plan_summary else None} "
        f"seconds={parent_plan_seconds:.3f} workers={worker_count}",
        flush=True,
    )

    workers: list[dict[str, Any]] = []
    worker_phase_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _worker_initialize_collection,
                str(path),
                subject_shard_runs=shard_runs,
                target_crop_run=target_crop_run,
                collection_worker_plan=plan,
                assignment_keypoint_group=assignment_keypoint_group,
                assignment_keypoints_run=assignment_keypoints_run,
                sample_start=int(start),
                sample_rows=int(sample_rows),
                hold_seconds=float(hold_seconds),
                worker_index=int(worker_index),
            )
            for worker_index, start in enumerate(starts)
        ]
        for future in as_completed(futures):
            payload = dict(future.result())
            workers.append(payload)
            print(
                f"worker={payload['worker_index']} pid={payload['pid']} "
                f"rss_kib={payload['memory_kib']['after_sample'].get('VmRSS')} "
                f"seconds={payload['total_seconds']:.3f}",
                flush=True,
            )
    worker_phase_seconds = float(time.perf_counter() - worker_phase_started)
    workers.sort(key=lambda item: int(item["worker_index"]))
    current_rss = [int(item["memory_kib"]["after_sample"].get("VmRSS") or 0) for item in workers]
    peak_rss = [int(item["memory_kib"]["after_hold"].get("VmHWM") or 0) for item in workers]
    result = {
        "schema_id": "palette.subject_mask_collection_worker_init_benchmark.v1",
        "created_utc": utc_now(),
        "zarr_path": str(path),
        "source_refined_run": str(source_refined_run),
        "subject_shard_runs": list(shard_runs),
        "target_crop_run": target_crop_run,
        "assignment_keypoint_group": assignment_keypoint_group,
        "assignment_keypoints_run": assignment_keypoints_run,
        "total_rows": total_rows,
        "worker_count": worker_count,
        "sample_rows_per_worker": int(sample_rows),
        "hold_seconds": float(hold_seconds),
        "parent_plan_seconds": parent_plan_seconds,
        "parent_memory_kib": {
            "before": parent_memory_before,
            "after_plan": parent_memory_after_plan,
            "after_workers": _proc_memory_kib(),
        },
        "collection_worker_plan": plan_summary,
        "worker_phase_seconds": worker_phase_seconds,
        "worker_current_rss_sum_kib": int(sum(current_rss)),
        "worker_current_rss_max_kib": int(max(current_rss, default=0)),
        "worker_peak_rss_sum_kib": int(sum(peak_rss)),
        "worker_peak_rss_max_kib": int(max(peak_rss, default=0)),
        "workers": workers,
    }
    if output_json is not None:
        output_path = Path(output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--source-refined-run", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-rows", type=int, default=1)
    parser.add_argument("--hold-seconds", type=float, default=15.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = benchmark_collection_worker_initialization(
        args.zarr_path,
        source_refined_run=str(args.source_refined_run),
        num_workers=int(args.num_workers),
        sample_rows=int(args.sample_rows),
        hold_seconds=float(args.hold_seconds),
        output_json=args.output_json,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"workers={result['worker_count']} parent_plan_s={result['parent_plan_seconds']:.3f} "
            f"worker_rss_sum_mib={result['worker_current_rss_sum_kib'] / 1024.0:.1f} "
            f"worker_rss_max_mib={result['worker_current_rss_max_kib'] / 1024.0:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
