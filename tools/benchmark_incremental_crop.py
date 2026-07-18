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

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.tracking.incremental_crop import materialize_incremental_crop_run


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


def run_benchmark(
    *,
    workdir: Path,
    row_count: int,
    roi_size: int,
    frame_size: int,
    delta_fraction: float,
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
    return {
        "schema_id": "palette.incremental_crop_benchmark",
        "schema_version": 1,
        "row_count": row_count,
        "roi_size": [roi_size, roi_size],
        "frame_size": [frame_size, frame_size],
        "delta_fraction_requested": delta_fraction,
        "full": {"wall_seconds": full_seconds, **full.to_dict()},
        "delta": {"wall_seconds": delta_seconds, **delta.to_dict()},
        "delta_speedup_over_full": full_seconds / delta_seconds,
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "archive_stored_bytes": _stored_bytes(archive),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--delta-fraction", type=float, default=0.01)
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
