"""Benchmark subject-mask finalizer schedulers on real zarr probability chunks.

This diagnostic does not write refined outputs. It reads real
``subject_mask_runs/<run>/mask_probs_roi`` rows, runs the pure spatial
finalization helpers, optionally performs ``eyes_union`` left/right assignment,
and compares per-component output hashes across scheduler modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import dask
from dask import delayed
import numpy as np

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:  # pragma: no cover - depends on optional dependency
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    HAVE_DISTRIBUTED = False

from ..refinement.assemble_refined_subject_masks import (
    _resolve_eye_keypoint_indices,
    _resolve_keypoint_success_array,
    _resolve_subject_keypoint_group,
)
from ..refinement.subject_eye_assignment import assign_eyes_union_to_lr
from ..refinement.subject_mask_finalization import _default_policy_for_component, finalize_component_mask
from ..shared.mask_probability_encoding import decode_probability_values
from ..tune.refined_subject_mask_review import (
    _load_source_subject_mask_run,
    _normalize_component_name,
    _probability_encoding_for_group,
    _probability_thresholds_for_labels,
)
from ..shared.zarr_io import open_zarr_root

_SUPPORTED_SCHEDULERS = ("single-threaded", "threads", "processes", "distributed")
_DEFAULT_COMPONENTS = ("subject_body", "eyes_union", "swim_bladder")


@dataclass(frozen=True)
class ChunkResult:
    start_row: int
    stop_row: int
    row_count: int
    component_hashes: dict[str, str]
    reason_hashes: dict[str, str]
    review_counts: dict[str, dict[str, int]]
    timing_seconds: dict[str, float]


def _normalize_scheduler(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"single_thread", "single-thread"}:
        text = "single-threaded"
    if text not in _SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler {value!r}; expected one of {', '.join(_SUPPORTED_SCHEDULERS)}."
        )
    return text


def _row_chunks(*, start_row: int, roi_count: int, chunk_size: int) -> list[tuple[int, int]]:
    start = max(0, int(start_row))
    count = max(0, int(roi_count))
    size = max(1, int(chunk_size))
    return [
        (row_start, min(start + count, row_start + size))
        for row_start in range(start, start + count, size)
    ]


def _decode_probabilities(values: np.ndarray, *, encoding: Optional[str], source_path: str) -> np.ndarray:
    return decode_probability_values(values, encoding=encoding, source_path=source_path)


def _hash_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _update_text_hash(digest: "hashlib._Hash", value: object) -> None:
    digest.update(str(value).encode("utf-8"))
    digest.update(b"\0")


def _combine_hashes(chunk_hashes: Sequence[tuple[int, str]]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for start_row, value in sorted((int(row), str(hash_value)) for row, hash_value in chunk_hashes):
        digest.update(str(start_row).encode("utf-8"))
        digest.update(b":")
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _component_indices(labels: Sequence[str], components: Sequence[str]) -> dict[str, int]:
    normalized_labels = [_normalize_component_name(label) or str(label) for label in labels]
    indices: dict[str, int] = {}
    for raw_component in components:
        component = _normalize_component_name(raw_component) or str(raw_component)
        if component not in normalized_labels:
            continue
        indices[component] = normalized_labels.index(component)
    return indices


def _increment_review_count(counts: dict[str, dict[str, int]], component: str, review: str) -> None:
    component_counts = counts.setdefault(str(component), {})
    key = str(review or "pending")
    component_counts[key] = int(component_counts.get(key, 0)) + 1


def _finalize_chunk(
    zarr_path: str,
    *,
    source_run: str,
    start_row: int,
    stop_row: int,
    components: Sequence[str],
) -> dict[str, Any]:
    timing: dict[str, float] = {}
    chunk_start = time.perf_counter()
    root = open_zarr_root(zarr_path, mode="r")
    source = _load_source_subject_mask_run(root, source_run)
    group = source.group
    probabilities_arr = group.get("mask_probs_roi")
    if probabilities_arr is None:
        raise RuntimeError(f"subject_mask_runs/{source_run} missing mask_probs_roi.")

    labels = tuple(str(label) for label in source.mask_labels)
    component_indices = _component_indices(labels, components)
    if not component_indices:
        raise RuntimeError(f"No requested components are available in subject_mask_runs/{source_run}.")
    source_path = f"subject_mask_runs/{source_run}/mask_probs_roi"
    encoding = _probability_encoding_for_group(
        group,
        source_path=source_path,
        observed_dtype=probabilities_arr.dtype,
    )
    thresholds = _probability_thresholds_for_labels(group, labels)

    read_start = time.perf_counter()
    raw_probs = np.asarray(probabilities_arr[int(start_row) : int(stop_row), :, :, :])
    probabilities = _decode_probabilities(raw_probs, encoding=encoding, source_path=source_path)
    timing["read_decode"] = time.perf_counter() - read_start

    finalize_start = time.perf_counter()
    component_hashes: dict[str, str] = {}
    reason_hashes: dict[str, str] = {}
    review_counts: dict[str, dict[str, int]] = {}
    finalized_masks: dict[str, np.ndarray] = {}

    for component, channel_idx in component_indices.items():
        threshold = float(thresholds[channel_idx]) if channel_idx < len(thresholds) else 0.5
        policy = replace(_default_policy_for_component(component), threshold=threshold)
        masks: list[np.ndarray] = []
        mask_digest = hashlib.blake2b(digest_size=16)
        reason_digest = hashlib.blake2b(digest_size=16)
        for row_offset in range(int(probabilities.shape[0])):
            result = finalize_component_mask(
                component,
                probabilities[row_offset, channel_idx],
                policy=policy,
                surface_is_probability=True,
            )
            # Record the effective threshold even when the default policy is used.
            _update_text_hash(reason_digest, f"threshold={threshold:g}")
            masks.append(np.asarray(result.mask, dtype=np.uint8))
            mask_digest.update(_hash_array(result.mask).encode("utf-8"))
            _update_text_hash(reason_digest, "|".join(result.reason_tags))
            _increment_review_count(review_counts, component, result.review_recommendation)
        finalized = np.stack(masks, axis=0) if masks else np.zeros((0,), dtype=np.uint8)
        finalized_masks[component] = finalized
        component_hashes[component] = mask_digest.hexdigest()
        reason_hashes[component] = reason_digest.hexdigest()
    timing["spatial_finalize"] = time.perf_counter() - finalize_start

    assignment_start = time.perf_counter()
    if "eyes_union" in finalized_masks:
        kp_group, keypoint_run_name, _keypoint_group_name, _source_kind = _resolve_subject_keypoint_group(root, source)
        keypoints_roi = kp_group.get("keypoints_roi")
        if keypoints_roi is None:
            raise RuntimeError(f"Keypoint run {keypoint_run_name!r} missing keypoints_roi.")
        keypoint_success, _success_dataset = _resolve_keypoint_success_array(kp_group, keypoint_run_name)
        eye_keypoint_indices = _resolve_eye_keypoint_indices(kp_group, keypoint_run_name)
        assignment = assign_eyes_union_to_lr(
            finalized_masks["eyes_union"],
            keypoints_roi=np.asarray(keypoints_roi[int(start_row) : int(stop_row)], dtype=np.float32),
            keypoint_success=np.asarray(keypoint_success[int(start_row) : int(stop_row)], dtype=bool),
            eye_keypoint_indices=eye_keypoint_indices,
        )
        for component in ("eye_left", "eye_right"):
            masks = np.asarray(assignment.masks[component], dtype=np.uint8)
            reasons = np.asarray(assignment.reason_labels[component], dtype=object)
            component_hashes[component] = _hash_array(masks)
            reason_digest = hashlib.blake2b(digest_size=16)
            for reason in reasons:
                _update_text_hash(reason_digest, reason)
            reason_hashes[component] = reason_digest.hexdigest()
            assigned_rows = int(np.count_nonzero(np.any(masks > 0, axis=(1, 2))))
            failed_rows = int(masks.shape[0] - assigned_rows)
            review_counts[component] = {
                "pending": assigned_rows,
                "needs_review": failed_rows,
            }
    timing["eye_assignment"] = time.perf_counter() - assignment_start
    timing["total"] = time.perf_counter() - chunk_start

    return {
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "row_count": int(stop_row) - int(start_row),
        "component_hashes": component_hashes,
        "reason_hashes": reason_hashes,
        "review_counts": review_counts,
        "timing_seconds": timing,
    }


def _aggregate_chunk_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    component_chunks: dict[str, list[tuple[int, str]]] = {}
    reason_chunks: dict[str, list[tuple[int, str]]] = {}
    review_counts: dict[str, dict[str, int]] = {}
    timing: dict[str, float] = {}
    row_count = 0
    for result in results:
        start_row = int(result["start_row"])
        row_count += int(result["row_count"])
        for component, value in dict(result["component_hashes"]).items():
            component_chunks.setdefault(str(component), []).append((start_row, str(value)))
        for component, value in dict(result["reason_hashes"]).items():
            reason_chunks.setdefault(str(component), []).append((start_row, str(value)))
        for component, counts in dict(result["review_counts"]).items():
            target = review_counts.setdefault(str(component), {})
            for key, value in dict(counts).items():
                target[str(key)] = int(target.get(str(key), 0)) + int(value)
        for key, value in dict(result["timing_seconds"]).items():
            timing[str(key)] = float(timing.get(str(key), 0.0)) + float(value)
    return {
        "row_count": int(row_count),
        "component_hashes": {
            component: _combine_hashes(chunks)
            for component, chunks in sorted(component_chunks.items())
        },
        "reason_hashes": {
            component: _combine_hashes(chunks)
            for component, chunks in sorted(reason_chunks.items())
        },
        "review_counts": review_counts,
        "chunk_timing_seconds_sum": timing,
    }


def _run_scheduler(
    *,
    zarr_path: str,
    source_run: str,
    scheduler: str,
    chunks: Sequence[tuple[int, int]],
    components: Sequence[str],
    num_workers: Optional[int],
) -> dict[str, Any]:
    scheduler_key = _normalize_scheduler(scheduler)
    tasks = [
        delayed(_finalize_chunk)(
            zarr_path,
            source_run=source_run,
            start_row=start_row,
            stop_row=stop_row,
            components=tuple(components),
        )
        for start_row, stop_row in chunks
    ]
    start = time.perf_counter()
    cluster = None
    client = None
    try:
        if scheduler_key == "distributed":
            if not HAVE_DISTRIBUTED:
                raise RuntimeError("Dask distributed is not available.")
            cluster_kwargs: dict[str, object] = {}
            if num_workers is not None:
                cluster_kwargs["n_workers"] = int(num_workers)
            cluster = LocalCluster(**cluster_kwargs)
            client = Client(cluster)
            results = list(client.gather(client.compute(tasks)))
        else:
            compute_kwargs: dict[str, object] = {"scheduler": scheduler_key}
            if num_workers is not None and scheduler_key != "single-threaded":
                compute_kwargs["num_workers"] = int(num_workers)
            results = list(dask.compute(*tasks, **compute_kwargs))
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
    wall = time.perf_counter() - start
    aggregate = _aggregate_chunk_results(results)
    rows = int(aggregate["row_count"])
    return {
        "scheduler": scheduler_key,
        "status": "ok",
        "wall_seconds": float(wall),
        "rows_per_second": float(rows / wall) if wall > 0 else 0.0,
        "chunk_count": int(len(chunks)),
        **aggregate,
    }


def benchmark_subject_mask_finalizer_schedulers(
    zarr_path: str | Path,
    *,
    source_run: str,
    schedulers: Sequence[str] = _SUPPORTED_SCHEDULERS,
    start_row: int = 0,
    roi_count: int = 512,
    chunk_size: int = 128,
    components: Sequence[str] = _DEFAULT_COMPONENTS,
    num_workers: Optional[int] = None,
) -> dict[str, Any]:
    zarr_path = str(Path(zarr_path))
    scheduler_keys = [_normalize_scheduler(scheduler) for scheduler in schedulers]
    chunks = _row_chunks(start_row=start_row, roi_count=roi_count, chunk_size=chunk_size)
    results: list[dict[str, Any]] = []
    baseline: Optional[dict[str, Any]] = None
    for scheduler in scheduler_keys:
        result = _run_scheduler(
            zarr_path=zarr_path,
            source_run=source_run,
            scheduler=scheduler,
            chunks=chunks,
            components=components,
            num_workers=num_workers,
        )
        if baseline is None:
            baseline = result
            result["matches_baseline"] = True
        else:
            result["matches_baseline"] = (
                result["component_hashes"] == baseline["component_hashes"]
                and result["reason_hashes"] == baseline["reason_hashes"]
                and result["review_counts"] == baseline["review_counts"]
            )
        results.append(result)
    return {
        "zarr_path": zarr_path,
        "source_run": str(source_run),
        "start_row": int(start_row),
        "roi_count": int(roi_count),
        "chunk_size": int(chunk_size),
        "components": [str(component) for component in components],
        "schedulers": scheduler_keys,
        "chunks": [{"start": int(start), "stop": int(stop)} for start, stop in chunks],
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the Palette zarr archive.")
    parser.add_argument("--source-run", required=True, help="subject_mask_runs/<run> to benchmark.")
    parser.add_argument(
        "--scheduler",
        action="append",
        dest="schedulers",
        choices=_SUPPORTED_SCHEDULERS,
        help="Scheduler to benchmark. Repeatable. Defaults to all supported schedulers.",
    )
    parser.add_argument("--start-row", type=int, default=0, help="First ROI row to benchmark.")
    parser.add_argument("--roi-count", type=int, default=512, help="Number of contiguous ROI rows to benchmark.")
    parser.add_argument("--chunk-size", type=int, default=128, help="Rows per Dask chunk.")
    parser.add_argument(
        "--component",
        action="append",
        dest="components",
        help="Raw source component to finalize. Repeatable. Defaults to body, eyes_union, swim_bladder.",
    )
    parser.add_argument("--num-workers", type=int, help="Optional Dask worker count.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = benchmark_subject_mask_finalizer_schedulers(
        args.zarr_path,
        source_run=args.source_run,
        schedulers=args.schedulers or _SUPPORTED_SCHEDULERS,
        start_row=int(args.start_row),
        roi_count=int(args.roi_count),
        chunk_size=int(args.chunk_size),
        components=args.components or _DEFAULT_COMPONENTS,
        num_workers=args.num_workers,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
