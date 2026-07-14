#!/usr/bin/env python3
"""Validate sampled decoded parity between v1 and encoded-v2 mask imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr


def validate_canary(
    *,
    zarr_path: Path,
    baseline_run: str,
    encoded_run: str,
    sample_row_chunks: int = 16,
) -> dict[str, Any]:
    zarr_path = zarr_path.expanduser().resolve()
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["refined_subject_masks_runs"]
    baseline = parent[baseline_run]
    encoded = parent[encoded_run]
    if baseline.attrs.get("palette_run_completion_status") != "complete":
        raise ValueError(f"Baseline run is not complete: {baseline_run}")
    if encoded.attrs.get("palette_run_completion_status") != "complete":
        raise ValueError(f"Encoded run is not complete: {encoded_run}")
    baseline_ids = np.asarray(baseline["source_crop_row_ids"][:], dtype=np.int64)
    encoded_ids = np.asarray(encoded["source_crop_row_ids"][:], dtype=np.int64)
    if not np.array_equal(baseline_ids, encoded_ids):
        raise ValueError("v1 and v2 source_crop_row_ids differ.")
    baseline_masks = baseline["masks_roi"]
    encoded_masks = encoded["masks_roi"]
    if tuple(baseline_masks.shape) != tuple(encoded_masks.shape):
        raise ValueError(f"v1 and v2 mask shapes differ: {baseline_masks.shape} != {encoded_masks.shape}")
    row_chunk = int(encoded_masks.chunks[0])
    chunk_count = (int(encoded_masks.shape[0]) + row_chunk - 1) // row_chunk
    publication = dict(encoded.attrs.get("encoded_mask_publication") or {})
    boundary_indices = [int(value) for value in publication.get("boundary_row_chunk_indices") or []]
    deterministic = (
        np.linspace(0, max(0, chunk_count - 1), num=min(max(1, sample_row_chunks), chunk_count), dtype=np.int64)
        if chunk_count
        else np.empty((0,), dtype=np.int64)
    )
    selected = sorted({0, max(0, chunk_count - 1), *boundary_indices, *deterministic.tolist()}) if chunk_count else []
    rows_checked = 0
    for chunk_idx in selected:
        start = int(chunk_idx) * row_chunk
        stop = min(start + row_chunk, int(encoded_masks.shape[0]))
        expected = np.asarray(baseline_masks[start:stop])
        observed = np.asarray(encoded_masks[start:stop])
        if not np.array_equal(expected, observed):
            mismatch = np.argwhere(expected != observed)[0]
            raise ValueError(
                f"Mask parity failed in row chunk {chunk_idx} at relative index {mismatch.tolist()}."
            )
        rows_checked += stop - start
    for path in ("frame_indices", "detection_indices", "available_channels"):
        if path in baseline or path in encoded:
            if path not in baseline or path not in encoded:
                raise ValueError(f"v1/v2 array presence differs for {path}.")
            if not np.array_equal(np.asarray(baseline[path][:]), np.asarray(encoded[path][:])):
                raise ValueError(f"v1/v2 array values differ for {path}.")
    return {
        "status": "ok",
        "zarr_path": str(zarr_path),
        "baseline_run": baseline_run,
        "encoded_run": encoded_run,
        "row_count": int(encoded_masks.shape[0]),
        "row_chunk": row_chunk,
        "row_chunks_total": chunk_count,
        "row_chunk_indices_checked": selected,
        "rows_checked": int(rows_checked),
        "boundary_row_chunk_indices": boundary_indices,
        "baseline_duration_seconds": baseline.attrs.get("duration_seconds"),
        "encoded_duration_seconds": encoded.attrs.get("duration_seconds"),
        "encoded_mask_publication": publication,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--encoded-run", required=True)
    parser.add_argument("--sample-row-chunks", type=int, default=16)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = validate_canary(
        zarr_path=args.zarr,
        baseline_run=args.baseline_run,
        encoded_run=args.encoded_run,
        sample_row_chunks=int(args.sample_row_chunks),
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
