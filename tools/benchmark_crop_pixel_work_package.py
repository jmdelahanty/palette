#!/usr/bin/env python3
"""Synthetic bounded-I/O benchmark for keyed crop pixel work packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    build_crop_pixel_work_package_from_source,
)
from fisheye.shared.roi_pixel_contract import crop_run_pixel_contract
from fisheye.shared.row_source_signature import build_row_source_signatures


def _source(rows: int, roi_size: int) -> tuple[object, np.ndarray]:
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    crop = root.require_group("crop_runs").create_group("crop_benchmark")
    keys = np.arange(1, rows + 1, dtype=np.uint64)
    frames = np.arange(rows, dtype=np.int64)
    coords = np.column_stack(
        (np.arange(rows, dtype=np.int32) % 256,) * 2
    ).astype(np.int32)
    rng = np.random.default_rng(20260718)
    pixels = rng.integers(
        0, 256, size=(rows, roi_size, roi_size), dtype=np.uint8
    )
    signatures = build_row_source_signatures(
        stage="crop",
        instance_keys=keys,
        content_components={"frame_indices": frames, "roi_coordinates_full": coords},
        compatibility_context={"benchmark": True, "roi_size": [roi_size, roi_size]},
    )
    crop.create_array(
        "roi_images",
        data=pixels,
        chunks=(min(128, rows), roi_size, roi_size),
    )
    crop.create_array("instance_key", data=keys, chunks=(min(1024, rows),))
    crop.create_array("frame_indices", data=frames, chunks=(min(1024, rows),))
    crop.create_array("roi_coordinates_full", data=coords, chunks=(min(1024, rows), 2))
    crop.create_array(
        "source_row_signature",
        data=signatures.signatures,
        chunks=(min(1024, rows), 32),
    )
    crop.attrs.update(signatures.spec.to_attrs())
    crop.attrs.update(
        {
            "crop_storage_mode": "materialized",
            "roi_size": [roi_size, roi_size],
            "crop_revision": 1,
            "crop_signature": {"benchmark": True},
            "source_pixel_fingerprint": "benchmark-source-pixels",
            "roi_pixel_contract": crop_run_pixel_contract(
                crop_storage_mode="materialized",
                video_source_type="synthetic",
                acceleration="cpu",
            ),
        }
    )
    return root, pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--roi-size", type=int, default=64)
    parser.add_argument("--delta-fraction", type=float, default=0.01)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.roi_size <= 0 or args.repetitions <= 0:
        raise ValueError("rows, roi-size, and repetitions must be positive.")
    if not 0 < args.delta_fraction <= 1:
        raise ValueError("delta-fraction must be in (0,1].")

    root, pixels = _source(int(args.rows), int(args.roi_size))
    delta_rows = max(1, int(round(args.rows * args.delta_fraction)))
    selected = np.linspace(0, args.rows - 1, delta_rows, dtype=np.int64)
    source = CropImageSource.open(root, crop_run="crop_benchmark")
    results: list[dict[str, object]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="palette-crop-package-benchmark-") as temp:
            work = Path(temp)
            for repetition in range(int(args.repetitions)):
                for label, rows in (
                    ("complete", np.arange(args.rows, dtype=np.int64)),
                    ("delta", selected),
                ):
                    manifest_path = work / f"{label}-{repetition}.json"
                    started = time.perf_counter()
                    manifest = build_crop_pixel_work_package_from_source(
                        source,
                        target_crop_rows=rows,
                        manifest_path=manifest_path,
                        archive_path=work / "benchmark.analysis.zarr",
                        batch_rows=256,
                    )
                    elapsed = time.perf_counter() - started
                    results.append(
                        {
                            "repetition": repetition,
                            "layout": label,
                            "rows": int(rows.shape[0]),
                            "payload_bytes": int(manifest["array"]["total_bytes"]),
                            "seconds": elapsed,
                        }
                    )
    finally:
        source.close()

    summary: dict[str, object] = {
        "schema": "palette.crop_pixel_work_package_benchmark.v1",
        "rows": int(args.rows),
        "roi_size": int(args.roi_size),
        "delta_fraction": float(args.delta_fraction),
        "raw_complete_bytes": int(pixels.nbytes),
        "results": results,
    }
    for label in ("complete", "delta"):
        seconds = [float(item["seconds"]) for item in results if item["layout"] == label]
        summary[f"{label}_median_seconds"] = float(np.median(seconds))
    summary["payload_reduction_ratio"] = float(
        int(pixels.nbytes)
        / int(next(item["payload_bytes"] for item in results if item["layout"] == "delta"))
    )
    output = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json is not None:
        args.output_json.expanduser().write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
