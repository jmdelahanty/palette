#!/usr/bin/env python3
"""Benchmark destination-side Zarr open/read timings for one archive layout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zarr

from fisheye.utils.zarr_io import open_zarr_root


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text or None


def _resolve_latest_run_name(parent: zarr.Group | None) -> Optional[str]:
    if parent is None:
        return None
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest:
        return latest
    keys = sorted(str(key) for key in parent.keys())
    return keys[-1] if keys else None


def _benchmark_index(total_rows: int, requested_index: Optional[int]) -> int:
    if total_rows <= 0:
        return 0
    if requested_index is None:
        return int(total_rows // 2)
    return max(0, min(int(requested_index), int(total_rows) - 1))


def _shape_list(value: Any) -> list[int]:
    return [int(v) for v in value]


def _array_layout(arr: zarr.Array) -> dict[str, object]:
    layout: dict[str, object] = {
        "shape": _shape_list(arr.shape),
        "dtype": str(arr.dtype),
    }
    chunks = getattr(arr, "chunks", None)
    if chunks is not None:
        layout["chunks"] = _shape_list(chunks)
    shards = getattr(arr, "shards", None)
    if shards is not None:
        layout["shards"] = _shape_list(shards)
    return layout


def _timed_array_read(arr: zarr.Array, row_index: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    block = np.asarray(arr[row_index])
    elapsed = time.perf_counter() - start
    return block, float(elapsed)


def _summarize_read(arr: zarr.Array, path: str, row_index: int) -> dict[str, object]:
    block, read_seconds = _timed_array_read(arr, row_index)
    return {
        "path": str(path),
        "row_index": int(row_index),
        "read_seconds": float(read_seconds),
        "block_shape": _shape_list(block.shape),
        "block_nbytes": int(block.nbytes),
        **_array_layout(arr),
    }


def benchmark_open_group_reads(
    root: zarr.Group,
    *,
    zarr_path: Path | str,
    variant: Optional[str] = None,
    row_index: Optional[int] = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "zarr_path": str(Path(zarr_path).expanduser()),
        "variant": _normalize_text(variant),
        "open_root_seconds": None,
        "reads": {},
    }
    reads = summary["reads"]
    assert isinstance(reads, dict)

    raw_arr = root.get("raw_video/images_full")
    if isinstance(raw_arr, zarr.Array) and int(raw_arr.shape[0]) > 0:
        idx = _benchmark_index(int(raw_arr.shape[0]), row_index)
        reads["raw_video/images_full"] = _summarize_read(raw_arr, "raw_video/images_full", idx)
    else:
        reads["raw_video/images_full"] = {"status": "missing"}

    crop_parent = root.get("crop_runs")
    crop_run = _resolve_latest_run_name(crop_parent if isinstance(crop_parent, zarr.Group) else None)
    crop_path = f"crop_runs/{crop_run}/roi_images" if crop_run else None
    crop_arr = root.get(crop_path) if crop_path else None
    if isinstance(crop_arr, zarr.Array) and int(crop_arr.shape[0]) > 0:
        idx = _benchmark_index(int(crop_arr.shape[0]), row_index)
        reads["crop_runs/latest/roi_images"] = _summarize_read(crop_arr, crop_path or "", idx)
        reads["crop_runs/latest/roi_images"]["selected_run"] = str(crop_run)
    else:
        reads["crop_runs/latest/roi_images"] = {"status": "missing", "selected_run": crop_run}

    subject_parent = root.get("subject_mask_runs")
    subject_run = _resolve_latest_run_name(subject_parent if isinstance(subject_parent, zarr.Group) else None)
    subject_path = f"subject_mask_runs/{subject_run}/masks_roi" if subject_run else None
    subject_arr = root.get(subject_path) if subject_path else None
    if isinstance(subject_arr, zarr.Array) and int(subject_arr.shape[0]) > 0:
        idx = _benchmark_index(int(subject_arr.shape[0]), row_index)
        reads["subject_mask_runs/latest/masks_roi"] = _summarize_read(subject_arr, subject_path or "", idx)
        reads["subject_mask_runs/latest/masks_roi"]["selected_run"] = str(subject_run)
    else:
        reads["subject_mask_runs/latest/masks_roi"] = {"status": "missing", "selected_run": subject_run}

    return summary


def benchmark_zarr_destination_reads(
    zarr_path: Path | str,
    *,
    variant: Optional[str] = None,
    row_index: Optional[int] = None,
) -> dict[str, object]:
    zarr_path = Path(zarr_path).expanduser()
    start = time.perf_counter()
    root = open_zarr_root(zarr_path, mode="r")
    open_seconds = time.perf_counter() - start
    summary = benchmark_open_group_reads(root, zarr_path=zarr_path, variant=variant, row_index=row_index)
    summary["open_root_seconds"] = float(open_seconds)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to the destination .zarr archive to benchmark.")
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional label for the archive variant (for example raw, tar_unpacked, sharded_dense).",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Optional row index to read; defaults to the midpoint row of each array.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the benchmark summary as JSON.",
    )
    return parser


def _format_read_line(label: str, payload: dict[str, object]) -> str:
    if payload.get("status") == "missing":
        selected = payload.get("selected_run")
        if selected:
            return f"  {label}: missing (selected_run={selected})"
        return f"  {label}: missing"
    return (
        f"  {label}: {payload['read_seconds']:.4f}s "
        f"path={payload['path']} row={payload['row_index']} "
        f"shape={tuple(payload['block_shape'])}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    summary = benchmark_zarr_destination_reads(
        args.zarr_path,
        variant=args.variant,
        row_index=args.row_index,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"Zarr destination read benchmark: {summary['zarr_path']}")
    if summary.get("variant"):
        print(f"  variant: {summary['variant']}")
    print(f"  open_root_seconds: {summary['open_root_seconds']:.4f}")
    reads = summary["reads"]
    assert isinstance(reads, dict)
    for key in (
        "raw_video/images_full",
        "crop_runs/latest/roi_images",
        "subject_mask_runs/latest/masks_roi",
    ):
        payload = reads.get(key, {"status": "missing"})
        assert isinstance(payload, dict)
        print(_format_read_line(key, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
