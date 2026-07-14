#!/usr/bin/env python3
"""Validate sampled decoded parity between v1 and encoded-v2 mask imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.utils.import_refined_subject_mask_clip_packages import _get_node, _iter_array_paths


def _arrays_equal(expected: np.ndarray, observed: np.ndarray) -> bool:
    if expected.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(expected, observed, equal_nan=True))
    return bool(np.array_equal(expected, observed))


def _validate_non_mask_arrays(baseline: zarr.Group, encoded: zarr.Group) -> dict[str, Any]:
    baseline_paths = set(_iter_array_paths(baseline)) - {"masks_roi"}
    encoded_paths = set(_iter_array_paths(encoded)) - {"masks_roi"}
    if baseline_paths != encoded_paths:
        raise ValueError(
            "v1/v2 non-mask array paths differ: "
            f"missing={sorted(baseline_paths - encoded_paths)!r}, "
            f"extra={sorted(encoded_paths - baseline_paths)!r}."
        )
    elements_checked = 0
    for path in sorted(baseline_paths):
        expected_array = _get_node(baseline, path)
        observed_array = _get_node(encoded, path)
        expected_shape = tuple(int(value) for value in expected_array.shape)
        observed_shape = tuple(int(value) for value in observed_array.shape)
        if expected_shape != observed_shape or np.dtype(expected_array.dtype) != np.dtype(observed_array.dtype):
            raise ValueError(
                f"v1/v2 array contract differs for {path}: "
                f"shape {expected_shape} != {observed_shape}, "
                f"dtype {expected_array.dtype} != {observed_array.dtype}."
            )
        if not expected_shape:
            expected = np.asarray(expected_array[...])
            observed = np.asarray(observed_array[...])
            if not _arrays_equal(expected, observed):
                raise ValueError(f"v1/v2 scalar array values differ for {path}.")
            elements_checked += 1
            continue
        row_count = int(expected_shape[0])
        row_chunk = max(1, int(getattr(expected_array, "chunks", (4096,))[0] or 4096))
        for start in range(0, row_count, row_chunk):
            stop = min(start + row_chunk, row_count)
            expected = np.asarray(expected_array[start:stop])
            observed = np.asarray(observed_array[start:stop])
            if not _arrays_equal(expected, observed):
                raise ValueError(f"v1/v2 array values differ for {path} in rows [{start}, {stop}).")
            elements_checked += int(expected.size)
    return {
        "array_count": int(len(baseline_paths)),
        "elements_checked": int(elements_checked),
        "array_paths": sorted(baseline_paths),
        "comparison": "exact_all_values_equal_nan_for_float",
    }


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
    non_mask_arrays = _validate_non_mask_arrays(baseline, encoded)
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
        "non_mask_arrays": non_mask_arrays,
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
