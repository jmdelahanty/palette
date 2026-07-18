#!/usr/bin/env python3
"""Benchmark Phase-1 keyed crop copy-forward on a deterministic local Zarr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import shutil
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.tracking.incremental_crop import (
    materialize_composite_incremental_crop_run,
    materialize_incremental_crop_run,
)


def _source(
    root: Any,
    name: str,
    *,
    keys: np.ndarray,
    frame_indices: np.ndarray,
    boxes: np.ndarray,
) -> Any:
    group = root.require_group("refined_detect_runs").create_group(name)
    group.create_array("instance_key", data=keys, chunks=(min(16_384, keys.shape[0]),))
    group.create_array(
        "frame_indices",
        data=frame_indices,
        chunks=(min(16_384, keys.shape[0]),),
    )
    group.create_array(
        "bbox_norm_coords",
        data=boxes,
        chunks=(min(16_384, keys.shape[0]), 4),
    )
    group.attrs["edit_revision"] = 1
    return group


def _stored_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _timed_reads(
    source: CropImageSource,
    *,
    random_rows: np.ndarray,
    repetitions: int,
) -> dict[str, Any]:
    sequential_seconds: list[float] = []
    random_seconds: list[float] = []
    checksum = 0
    for _ in range(int(repetitions)):
        start = time.perf_counter()
        sequential = source.read_slice(0, source.total_rois)
        sequential_seconds.append(time.perf_counter() - start)
        checksum ^= int(np.bitwise_xor.reduce(sequential.reshape(-1), initial=np.uint8(0)))
        start = time.perf_counter()
        random = source.read_indices(random_rows)
        random_seconds.append(time.perf_counter() - start)
        checksum ^= int(np.bitwise_xor.reduce(random.reshape(-1), initial=np.uint8(0)))
    return {
        "repetitions": int(repetitions),
        "random_row_count": int(random_rows.shape[0]),
        "sequential_seconds": sequential_seconds,
        "sequential_median_seconds": float(np.median(sequential_seconds)),
        "random_seconds": random_seconds,
        "random_median_seconds": float(np.median(random_seconds)),
        "checksum": checksum,
    }


def run_benchmark(
    *,
    workdir: Path,
    row_count: int,
    roi_size: int,
    frame_size: int,
    delta_fraction: float,
    read_repetitions: int = 3,
    random_read_rows: int = 256,
) -> dict[str, Any]:
    if row_count < 128:
        raise ValueError("row_count must be at least 128")
    if roi_size <= 0 or frame_size < roi_size:
        raise ValueError("frame_size must be at least roi_size")
    if not 0.0 < delta_fraction < 0.25:
        raise ValueError("delta_fraction must be between 0 and 0.25")
    archive = workdir / "benchmark.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    rows_per_frame = 32
    frame_count = (row_count + rows_per_frame - 1) // rows_per_frame
    frame_values = np.arange(frame_count, dtype=np.uint16)[:, None, None]
    y_values = np.arange(frame_size, dtype=np.uint16)[None, :, None]
    x_values = np.arange(frame_size, dtype=np.uint16)[None, None, :]
    frames = ((frame_values + y_values + x_values) % 256).astype(np.uint8)
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=frames,
        chunks=(1, frame_size, frame_size),
    )
    keys = np.arange(1, row_count + 1, dtype=np.uint64)
    frame_indices = (np.arange(row_count) // rows_per_frame).astype(np.int64)
    boxes = np.full((row_count, 4), [0.5, 0.5, 0.2, 0.2], dtype=np.float32)
    source_a = _source(
        root,
        "source_a",
        keys=keys,
        frame_indices=frame_indices,
        boxes=boxes,
    )
    provenance = build_writer_run_provenance(
        command="benchmark_incremental_crop",
        params={
            "row_count": row_count,
            "roi_size": roi_size,
            "frame_size": frame_size,
            "delta_fraction": delta_fraction,
        },
        input_run_ids={"source": "synthetic"},
    )
    start = time.perf_counter()
    full = materialize_incremental_crop_run(
        root,
        source_group=source_a,
        source_path="refined_detect_runs/source_a",
        frame_source=raw["images_full"],
        source_pixel_fingerprint="synthetic-pixels-v1",
        roi_size=(roi_size, roi_size),
        run_name="crop_full",
        run_provenance=provenance,
        roi_chunk_rows=32,
    )
    full_seconds = time.perf_counter() - start

    delta_rows = max(1, int(round(row_count * delta_fraction)))
    target_keys = np.roll(keys, row_count // 3).copy()
    target_frames = np.roll(frame_indices, row_count // 3).copy()
    target_boxes = np.roll(boxes, row_count // 3, axis=0).copy()
    target_boxes[:delta_rows, 0] = np.float32(0.6)
    target_keys[-delta_rows:] = np.arange(
        row_count + 1,
        row_count + delta_rows + 1,
        dtype=np.uint64,
    )
    target_frames[-delta_rows:] = np.arange(delta_rows, dtype=np.int64) % frame_count
    source_b = _source(
        root,
        "source_b",
        keys=target_keys,
        frame_indices=target_frames,
        boxes=target_boxes,
    )
    start = time.perf_counter()
    delta = materialize_incremental_crop_run(
        root,
        source_group=source_b,
        source_path="refined_detect_runs/source_b",
        frame_source=raw["images_full"],
        source_pixel_fingerprint="synthetic-pixels-v1",
        roi_size=(roi_size, roi_size),
        run_name="crop_delta",
        run_provenance=provenance,
        base_run_name="crop_full",
        roi_chunk_rows=32,
    )
    delta_seconds = time.perf_counter() - start
    start = time.perf_counter()
    composite = materialize_composite_incremental_crop_run(
        root,
        source_group=source_b,
        source_path="refined_detect_runs/source_b",
        frame_source=raw["images_full"],
        source_pixel_fingerprint="synthetic-pixels-v1",
        roi_size=(roi_size, roi_size),
        run_name="crop_composite",
        run_provenance=provenance,
        base_run_name="crop_full",
        roi_chunk_rows=32,
        promote=False,
    )
    composite_seconds = time.perf_counter() - start
    standalone_source = CropImageSource.open(root, crop_run="crop_delta")
    composite_source = CropImageSource.open(root, crop_run="crop_composite")
    random_count = min(max(1, int(random_read_rows)), row_count)
    random_rows = np.random.default_rng(0).choice(
        row_count,
        size=random_count,
        replace=False,
    ).astype(np.int64, copy=False)
    standalone_read = _timed_reads(
        standalone_source,
        random_rows=random_rows,
        repetitions=read_repetitions,
    )
    composite_read = _timed_reads(
        composite_source,
        random_rows=random_rows,
        repetitions=read_repetitions,
    )
    parity = np.array_equal(
        standalone_source.read_slice(0, row_count),
        composite_source.read_slice(0, row_count),
    )
    standalone_source.close()
    composite_source.close()
    crop_root = archive / "crop_runs"
    return {
        "schema_id": "palette.incremental_crop_benchmark",
        "schema_version": 2,
        "row_count": row_count,
        "roi_size": [roi_size, roi_size],
        "frame_size": [frame_size, frame_size],
        "delta_fraction_requested": delta_fraction,
        "full": {"wall_seconds": full_seconds, **full.to_dict()},
        "delta": {"wall_seconds": delta_seconds, **delta.to_dict()},
        "composite": {"wall_seconds": composite_seconds, **composite.to_dict()},
        "delta_speedup_over_full": full_seconds / delta_seconds,
        "composite_speedup_over_standalone_delta": delta_seconds / composite_seconds,
        "composite_logical_parity": bool(parity),
        "stored_bytes_by_crop_run": {
            name: _stored_bytes(crop_root / name)
            for name in ("crop_full", "crop_delta", "crop_composite")
        },
        "reads": {
            "standalone_delta": standalone_read,
            "composite": composite_read,
            "composite_sequential_slowdown": (
                composite_read["sequential_median_seconds"]
                / standalone_read["sequential_median_seconds"]
            ),
            "composite_random_slowdown": (
                composite_read["random_median_seconds"]
                / standalone_read["random_median_seconds"]
            ),
        },
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "archive_stored_bytes": _stored_bytes(archive),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--delta-fraction", type=float, default=0.01)
    parser.add_argument("--read-repetitions", type=int, default=3)
    parser.add_argument("--random-read-rows", type=int, default=256)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--keep", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    temporary = args.workdir is None
    workdir = (
        Path(tempfile.mkdtemp(prefix="palette-incremental-crop-benchmark-"))
        if temporary
        else args.workdir.expanduser().resolve()
    )
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        report = run_benchmark(
            workdir=workdir,
            row_count=args.rows,
            roi_size=args.roi_size,
            frame_size=args.frame_size,
            delta_fraction=args.delta_fraction,
            read_repetitions=args.read_repetitions,
            random_read_rows=args.random_read_rows,
        )
        report["workdir"] = str(workdir)
        if args.output_json is not None:
            write_json_atomic(args.output_json.expanduser().resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    finally:
        if temporary and not args.keep:
            shutil.rmtree(workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
