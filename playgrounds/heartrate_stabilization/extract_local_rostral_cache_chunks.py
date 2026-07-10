from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np

from _common import cfg_path, load_config, read_crop_meta, row_float
from extract_reliable_local_rostral_heartrate import (
    _timestamp_scale,
    load_dataset,
    save_dataset,
)


FRAME_ARRAY_NAMES = (
    "frame_indices",
    "traces",
    "pixel_valid",
    "frame_valid",
    "source_xy",
    "bilinear_weights",
    "body_occupancy",
    "eye_occupancy",
    "gradient_magnitude",
    "motion_prediction",
    "nuisance_values",
    "transform_uncertainty",
)


def _part_paths(prefix: Path) -> tuple[Path, Path, Path]:
    return (
        prefix.with_suffix(".local_pixel_matrix.npz"),
        prefix.with_suffix(".local_pixel_matrix.frames.csv"),
        prefix.with_suffix(".local_pixel_matrix.summary.json"),
    )


def _part_is_complete(
    prefix: Path,
    *,
    extraction_start: int,
    extraction_count: int,
) -> bool:
    dataset_path, frames_path, summary_path = _part_paths(prefix)
    if not (dataset_path.is_file() and frames_path.is_file() and summary_path.is_file()):
        return False
    try:
        with np.load(dataset_path, allow_pickle=False) as data:
            indices = np.asarray(data["frame_indices"], dtype=np.int64)
        return bool(
            indices.size == int(extraction_count)
            and int(indices[0]) == int(extraction_start)
            and int(indices[-1]) == int(extraction_start + extraction_count - 1)
        )
    except (OSError, ValueError, KeyError):
        return False


def _run_part(
    *,
    repo_root: Path,
    extractor: Path,
    config: Path,
    roi_json: Path,
    mask_npz: Path,
    status_csv: Path,
    prefix: Path,
    extraction_start: int,
    extraction_count: int,
    reference_anterior_xy: str,
    reference_posterior_xy: str,
    timestamp_column: str,
    timestamp_unit: str,
    mask_read_cache_rows: int,
    resume: bool,
) -> dict[str, Any]:
    if resume and _part_is_complete(
        prefix,
        extraction_start=extraction_start,
        extraction_count=extraction_count,
    ):
        return {"prefix": str(prefix), "status": "reused"}
    command = [
        str(repo_root / "scripts/py"),
        str(extractor),
        "--config",
        str(config),
        "--roi-json",
        str(roi_json),
        "--mask-npz",
        str(mask_npz),
        "--status-csv",
        str(status_csv),
        "--frame-start",
        str(extraction_start),
        "--frame-count",
        str(extraction_count),
        "--reference-anterior-xy",
        str(reference_anterior_xy),
        "--reference-posterior-xy",
        str(reference_posterior_xy),
        "--timestamp-column",
        str(timestamp_column),
        "--timestamp-unit",
        str(timestamp_unit),
        "--mask-read-cache-rows",
        str(mask_read_cache_rows),
        "--output-prefix",
        str(prefix),
        "--extract-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = prefix.with_suffix(".extract.log")
    log_path.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-20:])
        raise RuntimeError(
            f"chunk {prefix.name} failed with exit {completed.returncode}:\n{tail}"
        )
    if not _part_is_complete(
        prefix,
        extraction_start=extraction_start,
        extraction_count=extraction_count,
    ):
        raise RuntimeError(f"chunk {prefix.name} completed without valid output")
    return {"prefix": str(prefix), "status": "extracted"}


def _validate_static_contract(reference: Any, candidate: Any, *, prefix: Path) -> None:
    exact_names = (
        "pixel_xy",
        "image_shape_hw",
        "administrative_boundary_distance_px",
        "physical_boundary_distance_px",
    )
    for name in exact_names:
        if not np.array_equal(np.asarray(getattr(reference, name)), np.asarray(getattr(candidate, name))):
            raise ValueError(f"{prefix}: static dataset field {name} differs")
    if tuple(reference.nuisance_names) != tuple(candidate.nuisance_names):
        raise ValueError(f"{prefix}: nuisance schema differs")
    metadata_keys = (
        "pixel_contract",
        "source_image_shape_hw",
        "crop_video",
        "crop_meta_csv",
        "zarr_path",
        "keypoint_group",
        "frame_id_column",
        "mask_parent",
        "mask_run",
        "roi_mask_npz",
        "reference_anchors",
        "local_correction_limits",
        "frame_occupancy_thresholds",
    )
    for key in metadata_keys:
        if reference.metadata.get(key) != candidate.metadata.get(key):
            raise ValueError(f"{prefix}: metadata contract {key} differs")


def _timestamps_for_indices(
    *,
    config: Path,
    frame_indices: np.ndarray,
    timestamp_column: str,
    timestamp_unit: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    parsed = load_config(config)
    crop_meta = read_crop_meta(cfg_path(parsed, "inputs", "crop_meta_csv"))
    raw = np.asarray(
        [row_float(crop_meta[int(index)], timestamp_column) for index in frame_indices],
        dtype=np.float64,
    )
    if not np.isfinite(raw).all():
        raise ValueError("merged cache requires finite acquisition timestamps")
    scale, resolved_unit = _timestamp_scale(timestamp_unit, raw)
    timestamps = (raw - raw[0]) * scale
    differences = np.diff(timestamps)
    if differences.size == 0 or np.any(~np.isfinite(differences)) or np.any(differences <= 0.0):
        raise ValueError("merged timestamps are not strictly increasing")
    median_dt = float(np.median(differences))
    frame_steps = np.diff(frame_indices)
    jitter = np.abs(differences - median_dt)
    return timestamps, {
        "source": str(timestamp_column),
        "resolved_unit": str(resolved_unit),
        "sample_count": int(timestamps.size),
        "elapsed_seconds": float(timestamps[-1] - timestamps[0]),
        "median_dt_seconds": median_dt,
        "effective_fps": 1.0 / median_dt,
        "dt_min_seconds": float(np.min(differences)),
        "dt_p95_seconds": float(np.quantile(differences, 0.95)),
        "dt_max_seconds": float(np.max(differences)),
        "jitter_p99_seconds": float(np.quantile(jitter, 0.99)),
        "non_unit_frame_steps": int(np.count_nonzero(frame_steps != 1)),
        "frame_step": 1,
    }


def _merge_parts(
    *,
    part_specs: Sequence[dict[str, Any]],
    config: Path,
    output_prefix: Path,
    timestamp_column: str,
    timestamp_unit: str,
    workers: int,
    chunk_frames: int,
    mask_read_cache_rows: int,
) -> dict[str, Any]:
    datasets = []
    retained_slices: list[slice] = []
    frame_rows: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    for part_index, spec in enumerate(part_specs):
        prefix = Path(spec["prefix"])
        dataset_path, frames_path, _summary_path = _part_paths(prefix)
        dataset = load_dataset(dataset_path)
        if datasets:
            _validate_static_contract(datasets[0], dataset, prefix=prefix)
        drop = 0 if part_index == 0 else 1
        retained = slice(drop, dataset.frame_count)
        expected_start = int(spec["logical_start"])
        expected_stop = expected_start + int(spec["logical_count"])
        retained_indices = np.asarray(dataset.frame_indices)[retained]
        if (
            retained_indices.size != int(spec["logical_count"])
            or int(retained_indices[0]) != expected_start
            or int(retained_indices[-1]) != expected_stop - 1
        ):
            raise ValueError(f"{prefix}: retained logical frame range is incorrect")
        datasets.append(dataset)
        retained_slices.append(retained)
        with frames_path.open() as handle:
            rows = list(csv.DictReader(handle))[drop:]
        if len(rows) != int(spec["logical_count"]):
            raise ValueError(f"{prefix}: frame CSV row count differs from retained cache")
        for row in rows:
            row["row"] = str(len(frame_rows))
            frame_rows.append(row)
            reason_counts[str(row["reason"])] += 1
    first = datasets[0]
    merged_values = {
        name: np.concatenate(
            [np.asarray(getattr(dataset, name))[retained] for dataset, retained in zip(datasets, retained_slices)],
            axis=0,
        )
        for name in FRAME_ARRAY_NAMES
    }
    frame_indices = np.asarray(merged_values.pop("frame_indices"), dtype=np.int64)
    if np.any(np.diff(frame_indices) != 1):
        raise ValueError("merged frame indices are not consecutive")
    timestamps, timebase = _timestamps_for_indices(
        config=config,
        frame_indices=frame_indices,
        timestamp_column=timestamp_column,
        timestamp_unit=timestamp_unit,
    )
    metadata = dict(first.metadata)
    metadata["timebase"] = timebase
    cache_components = {
        "body": [dataset.metadata.get("mask_read_cache", {}).get("body") for dataset in datasets],
        "swim": [dataset.metadata.get("mask_read_cache", {}).get("swim") for dataset in datasets],
    }
    eye_names = sorted(
        {
            str(name)
            for dataset in datasets
            for name in dataset.metadata.get("mask_read_cache", {}).get("eyes", {})
        }
    )
    aggregated_cache: dict[str, Any] = {
        "requested_rows": int(mask_read_cache_rows),
        "aggregate_across_parts": True,
    }
    for name, summaries in cache_components.items():
        finite = [summary for summary in summaries if summary is not None]
        aggregated_cache[name] = (
            {
                "block_rows": int(finite[0]["block_rows"]),
                "hits": int(sum(int(summary["hits"]) for summary in finite)),
                "misses": int(sum(int(summary["misses"]) for summary in finite)),
            }
            if finite
            else None
        )
    aggregated_cache["eyes"] = {}
    for eye_name in eye_names:
        finite = [
            dataset.metadata.get("mask_read_cache", {}).get("eyes", {}).get(eye_name)
            for dataset in datasets
        ]
        finite = [summary for summary in finite if summary is not None]
        aggregated_cache["eyes"][eye_name] = (
            {
                "block_rows": int(finite[0]["block_rows"]),
                "hits": int(sum(int(summary["hits"]) for summary in finite)),
                "misses": int(sum(int(summary["misses"]) for summary in finite)),
            }
            if finite
            else None
        )
    metadata["mask_read_cache"] = aggregated_cache
    metadata["chunked_extraction"] = {
        "chunk_frames": int(chunk_frames),
        "worker_count": int(workers),
        "mask_read_cache_rows": int(mask_read_cache_rows),
        "part_count": int(len(part_specs)),
        "one_frame_overlap": True,
        "merge_contract": "validated static pixel/schema/anchor contracts; acquisition timestamps rebuilt from crop metadata",
        "parts": [
            {
                "prefix": str(spec["prefix"]),
                "logical_start": int(spec["logical_start"]),
                "logical_count": int(spec["logical_count"]),
                "extraction_start": int(spec["extraction_start"]),
                "extraction_count": int(spec["extraction_count"]),
            }
            for spec in part_specs
        ],
    }
    merged = replace(
        first,
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        metadata=metadata,
        **merged_values,
    ).validated()
    dataset_path = output_prefix.with_suffix(".local_pixel_matrix.npz")
    frames_path = output_prefix.with_suffix(".local_pixel_matrix.frames.csv")
    summary_path = output_prefix.with_suffix(".local_pixel_matrix.summary.json")
    partial_dataset = dataset_path.with_name(dataset_path.name + ".partial.npz")
    partial_frames = frames_path.with_name(frames_path.name + ".partial")
    partial_summary = summary_path.with_name(summary_path.name + ".partial")
    save_dataset(partial_dataset, merged)
    with partial_frames.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    summary = {
        "frame_count": int(merged.frame_count),
        "valid_frame_count": int(np.count_nonzero(merged.frame_valid)),
        "valid_frame_fraction": float(np.mean(merged.frame_valid)),
        "pixel_count": int(merged.pixel_count),
        "timebase": timebase,
        "reason_counts": dict(sorted(reason_counts.items())),
        "metadata": metadata,
    }
    partial_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(partial_dataset, dataset_path)
    os.replace(partial_frames, frames_path)
    os.replace(partial_summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a resumable parallel local-rostral cache and merge it fail-closed."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--chunk-frames", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mask-read-cache-rows", type=int, default=256)
    parser.add_argument("--reference-anterior-xy", type=str, required=True)
    parser.add_argument("--reference-posterior-xy", type=str, required=True)
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--timestamp-unit", choices=("auto", "s", "ms", "us", "ns"), default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if int(args.frame_start) < 0 or int(args.frame_count) < 2:
        raise ValueError("frame range must be nonnegative and contain at least two frames")
    if (
        int(args.chunk_frames) < 2
        or int(args.workers) < 1
        or int(args.mask_read_cache_rows) < 0
    ):
        raise ValueError(
            "chunk-frames must be at least two; workers must be positive; "
            "mask-read-cache-rows cannot be negative"
        )
    repo_root = Path(__file__).resolve().parents[2]
    extractor = Path(__file__).with_name("extract_reliable_local_rostral_heartrate.py")
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_prefix.parent / f"{output_prefix.name}.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    stop = int(args.frame_start) + int(args.frame_count)
    part_specs: list[dict[str, Any]] = []
    for part_index, logical_start in enumerate(
        range(int(args.frame_start), stop, int(args.chunk_frames))
    ):
        logical_count = min(int(args.chunk_frames), stop - logical_start)
        overlap = 0 if part_index == 0 else 1
        extraction_start = logical_start - overlap
        extraction_count = logical_count + overlap
        prefix = chunk_dir / f"part_{part_index:04d}_{logical_start:06d}_{logical_count:06d}"
        part_specs.append(
            {
                "prefix": str(prefix),
                "logical_start": int(logical_start),
                "logical_count": int(logical_count),
                "extraction_start": int(extraction_start),
                "extraction_count": int(extraction_count),
            }
        )
    print(
        f"chunk_plan: parts={len(part_specs)} workers={int(args.workers)} "
        f"logical_frames={int(args.frame_count)}",
        flush=True,
    )
    failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        future_to_spec = {
            executor.submit(
                _run_part,
                repo_root=repo_root,
                extractor=extractor,
                config=Path(args.config).resolve(),
                roi_json=Path(args.roi_json).resolve(),
                mask_npz=Path(args.mask_npz).resolve(),
                status_csv=Path(args.status_csv).resolve(),
                prefix=Path(spec["prefix"]),
                extraction_start=int(spec["extraction_start"]),
                extraction_count=int(spec["extraction_count"]),
                reference_anterior_xy=str(args.reference_anterior_xy),
                reference_posterior_xy=str(args.reference_posterior_xy),
                timestamp_column=str(args.timestamp_column),
                timestamp_unit=str(args.timestamp_unit),
                mask_read_cache_rows=int(args.mask_read_cache_rows),
                resume=not bool(args.no_resume),
            ): spec
            for spec in part_specs
        }
        completed_count = 0
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
                completed_count += 1
                print(
                    f"chunk_complete: {completed_count}/{len(part_specs)} "
                    f"{Path(spec['prefix']).name} status={result['status']}",
                    flush=True,
                )
            except BaseException as error:
                failures.append(error)
                print(f"chunk_failed: {Path(spec['prefix']).name}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} chunk extractions failed; merge was not attempted")
    summary = _merge_parts(
        part_specs=part_specs,
        config=Path(args.config).resolve(),
        output_prefix=output_prefix,
        timestamp_column=str(args.timestamp_column),
        timestamp_unit=str(args.timestamp_unit),
        workers=int(args.workers),
        chunk_frames=int(args.chunk_frames),
        mask_read_cache_rows=int(args.mask_read_cache_rows),
    )
    print(f"merged_dataset: {output_prefix.with_suffix('.local_pixel_matrix.npz')}")
    print(f"merged_frames: {summary['frame_count']}")
    print(f"merged_valid_frames: {summary['valid_frame_count']}")
    print(f"merged_valid_fraction: {summary['valid_frame_fraction']:.6f}")


if __name__ == "__main__":
    main()
