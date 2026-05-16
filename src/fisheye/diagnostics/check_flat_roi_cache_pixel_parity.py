"""Compare flat ROI cache pixels against the canonical CropImageSource path."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.flat_roi_cache import load_flat_roi_cache_manifest


def _parse_rows(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    rows: list[int] = []
    for part in value.replace(",", " ").split():
        rows.append(int(part))
    return rows


def _resolve_crop_run(manifest: dict[str, Any], crop_run: str | None) -> str | None:
    if crop_run:
        return crop_run
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    name = source.get("crop_run_name")
    return str(name) if name else None


def _boundary_rows(source: CropImageSource, limit: int) -> list[int]:
    if limit <= 0 or source.frame_shape is None:
        return []
    roi_h, roi_w = source.roi_shape
    frame_h, frame_w = source.frame_shape
    coords = np.asarray(source.roi_coordinates_full, dtype=np.int64)
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    outside = np.flatnonzero((x1 < 0) | (y1 < 0) | (x1 + roi_w > frame_w) | (y1 + roi_h > frame_h))
    return [int(v) for v in outside[:limit]]


def _sample_rows(
    *,
    total_rois: int,
    sample_count: int,
    seed: int,
    explicit_rows: Sequence[int] | None,
    boundary_rows: Sequence[int],
) -> list[int]:
    rows: list[int] = []
    if explicit_rows is not None:
        rows.extend(int(v) for v in explicit_rows)
    else:
        anchors = [0, total_rois // 2, total_rois - 1] if total_rois > 0 else []
        rows.extend(anchors)
        rows.extend(int(v) for v in boundary_rows)
        remaining = max(0, int(sample_count) - len(set(rows)))
        if remaining > 0 and total_rois > 0:
            rng = np.random.default_rng(int(seed))
            random_rows = rng.choice(total_rois, size=min(remaining, total_rois), replace=False)
            rows.extend(int(v) for v in random_rows.tolist())

    unique = sorted(set(rows))
    for row in unique:
        if row < 0 or row >= total_rois:
            raise IndexError(f"ROI row {row} out of range for total_rois={total_rois}")
    return unique


def _batch_iter(rows: Sequence[int], batch_size: int) -> Iterable[np.ndarray]:
    batch = max(1, int(batch_size))
    for start in range(0, len(rows), batch):
        yield np.asarray(rows[start : start + batch], dtype=np.int64)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _diff_summary(
    *,
    rows: Sequence[int],
    reference_batches: list[np.ndarray],
    cache_batches: list[np.ndarray],
    top_mismatches: int,
) -> dict[str, Any]:
    if not reference_batches:
        return {
            "rows_compared": 0,
            "pixels_compared": 0,
            "exact_rows": 0,
            "mismatched_rows": 0,
            "byte_equal": True,
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "p95_abs_diff": 0.0,
            "top_mismatches": [],
        }

    reference = np.concatenate(reference_batches, axis=0)
    cache = np.concatenate(cache_batches, axis=0)
    if reference.shape != cache.shape:
        raise ValueError(f"Shape mismatch: reference={reference.shape}, cache={cache.shape}")

    diff = np.abs(reference.astype(np.int16) - cache.astype(np.int16))
    row_max = diff.reshape(diff.shape[0], -1).max(axis=1)
    row_mean = diff.reshape(diff.shape[0], -1).mean(axis=1)
    exact_mask = row_max == 0
    mismatch_order = np.argsort(-row_max)
    top: list[dict[str, Any]] = []
    for local_idx in mismatch_order[: max(0, int(top_mismatches))]:
        if row_max[int(local_idx)] == 0:
            break
        top.append(
            {
                "row": int(rows[int(local_idx)]),
                "max_abs_diff": int(row_max[int(local_idx)]),
                "mean_abs_diff": float(row_mean[int(local_idx)]),
            }
        )

    return {
        "rows_compared": int(reference.shape[0]),
        "pixels_compared": int(diff.size),
        "exact_rows": int(exact_mask.sum()),
        "mismatched_rows": int((~exact_mask).sum()),
        "byte_equal": bool(np.array_equal(reference, cache)),
        "max_abs_diff": int(diff.max(initial=0)),
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "p95_abs_diff": _safe_percentile(diff.reshape(-1), 95.0),
        "top_mismatches": top,
    }


def check_flat_roi_cache_pixel_parity(
    *,
    zarr_path: str | Path,
    roi_cache_manifest: str | Path,
    crop_run: str | None = None,
    rows: Sequence[int] | None = None,
    sample_count: int = 32,
    seed: int = 0,
    boundary_sample_count: int = 4,
    reference_roi_live_acceleration: str = "gpu",
    reference_roi_live_gpu_chunk_frames: int = 32,
    batch_size: int = 16,
    max_abs_diff: int = 0,
    max_mean_abs_diff: float = 0.0,
    max_p95_abs_diff: float = 0.0,
    top_mismatches: int = 10,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    manifest_path = Path(roi_cache_manifest).expanduser().resolve()
    manifest = load_flat_roi_cache_manifest(manifest_path)
    resolved_crop_run = _resolve_crop_run(manifest, crop_run)

    root = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    reference = CropImageSource.open(
        root,
        crop_run=resolved_crop_run,
        zarr_path=archive_path,
        roi_cache_policy="never",
        roi_live_acceleration=reference_roi_live_acceleration,
        roi_live_gpu_chunk_frames=reference_roi_live_gpu_chunk_frames,
    )
    cache = CropImageSource.open(
        root,
        crop_run=resolved_crop_run,
        zarr_path=archive_path,
        roi_cache_policy="never",
        roi_cache_manifest=manifest_path,
    )
    try:
        if reference.shape != cache.shape:
            raise ValueError(f"Source shape mismatch: reference={reference.shape}, cache={cache.shape}")

        selected_rows = _sample_rows(
            total_rois=reference.total_rois,
            sample_count=sample_count,
            seed=seed,
            explicit_rows=rows,
            boundary_rows=_boundary_rows(reference, boundary_sample_count),
        )

        ref_batches: list[np.ndarray] = []
        cache_batches: list[np.ndarray] = []
        reference_read_seconds = 0.0
        cache_read_seconds = 0.0
        for row_batch in _batch_iter(selected_rows, batch_size):
            started = time.perf_counter()
            ref_batches.append(reference.read_indices(row_batch))
            reference_read_seconds += time.perf_counter() - started

            started = time.perf_counter()
            cache_batches.append(cache.read_indices(row_batch))
            cache_read_seconds += time.perf_counter() - started

        diff = _diff_summary(
            rows=selected_rows,
            reference_batches=ref_batches,
            cache_batches=cache_batches,
            top_mismatches=top_mismatches,
        )
        thresholds = {
            "max_abs_diff": int(max_abs_diff),
            "max_mean_abs_diff": float(max_mean_abs_diff),
            "max_p95_abs_diff": float(max_p95_abs_diff),
        }
        failures: list[str] = []
        if int(diff["max_abs_diff"]) > int(max_abs_diff):
            failures.append(
                f"max_abs_diff {diff['max_abs_diff']} exceeds threshold {int(max_abs_diff)}"
            )
        if float(diff["mean_abs_diff"]) > float(max_mean_abs_diff):
            failures.append(
                "mean_abs_diff "
                f"{float(diff['mean_abs_diff']):.6f} exceeds threshold {float(max_mean_abs_diff):.6f}"
            )
        if float(diff["p95_abs_diff"]) > float(max_p95_abs_diff):
            failures.append(
                f"p95_abs_diff {float(diff['p95_abs_diff']):.6f} exceeds threshold "
                f"{float(max_p95_abs_diff):.6f}"
            )

        source_payload = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        builder_payload = manifest.get("builder") if isinstance(manifest.get("builder"), dict) else {}
        return {
            "status": "ok" if not failures else "fail",
            "failures": failures,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "hostname": socket.gethostname(),
                "LSB_JOBID": os.environ.get("LSB_JOBID"),
                "LSB_QUEUE": os.environ.get("LSB_QUEUE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "inputs": {
                "zarr_path": str(archive_path),
                "roi_cache_manifest": str(manifest_path),
                "crop_run": resolved_crop_run,
                "sample_count_requested": int(sample_count),
                "sample_seed": int(seed),
                "explicit_rows": [int(v) for v in rows] if rows is not None else None,
                "boundary_sample_count": int(boundary_sample_count),
                "batch_size": int(batch_size),
                "reference_roi_live_acceleration_requested": str(reference_roi_live_acceleration),
                "reference_roi_live_gpu_chunk_frames": int(reference_roi_live_gpu_chunk_frames),
            },
            "source": {
                "shape": list(reference.shape),
                "roi_shape": list(reference.roi_shape),
                "total_rois": int(reference.total_rois),
                "storage_mode": reference.storage_mode,
                "reference_roi_read_mode": reference.roi_read_mode,
                "reference_roi_live_acceleration_effective": reference.roi_live_acceleration_effective,
                "reference_roi_live_acceleration_fallback_reason": (
                    reference.roi_live_acceleration_fallback_reason
                ),
                "cache_roi_read_mode": cache.roi_read_mode,
                "cache_backend": cache.roi_cache_backend,
                "cache_manifest_source": source_payload,
                "cache_manifest_builder": {
                    "decode_backend_requested": builder_payload.get("decode_backend_requested"),
                    "decode_backend_effective": builder_payload.get("decode_backend_effective"),
                    "pixel_contract": builder_payload.get("pixel_contract"),
                    "timing": builder_payload.get("timing"),
                },
            },
            "rows": selected_rows,
            "timing": {
                "reference_read_seconds": float(reference_read_seconds),
                "cache_read_seconds": float(cache_read_seconds),
                "reference_rows_per_second": (
                    float(len(selected_rows) / reference_read_seconds)
                    if reference_read_seconds > 0
                    else None
                ),
                "cache_rows_per_second": (
                    float(len(selected_rows) / cache_read_seconds) if cache_read_seconds > 0 else None
                ),
            },
            "thresholds": thresholds,
            "diff": diff,
        }
    finally:
        reference.close()
        cache.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis/training Zarr archive.")
    parser.add_argument("--roi-cache-manifest", type=Path, required=True, help="flat_bin_v1 cache manifest.")
    parser.add_argument("--crop-run", default=None, help="Crop run override; defaults to manifest source.")
    parser.add_argument("--rows", default=None, help="Comma/space separated explicit ROI rows to compare.")
    parser.add_argument("--sample-count", type=int, default=32, help="Number of rows to sample when --rows is absent.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Random sample seed.")
    parser.add_argument(
        "--boundary-sample-count",
        type=int,
        default=4,
        help="Number of near-boundary padded rows to include when available.",
    )
    parser.add_argument(
        "--reference-roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="gpu",
        help="Reference CropImageSource acceleration for geometry-only live reads.",
    )
    parser.add_argument(
        "--reference-roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame chunk size for GPU reference live reads.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Rows per comparison batch.")
    parser.add_argument("--max-abs-diff", type=int, default=0, help="Fail if max absolute byte diff exceeds this.")
    parser.add_argument(
        "--max-mean-abs-diff",
        type=float,
        default=0.0,
        help="Fail if mean absolute byte diff exceeds this.",
    )
    parser.add_argument(
        "--max-p95-abs-diff",
        type=float,
        default=0.0,
        help="Fail if p95 absolute byte diff exceeds this.",
    )
    parser.add_argument("--top-mismatches", type=int, default=10, help="Number of row mismatches to report.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    payload = check_flat_roi_cache_pixel_parity(
        zarr_path=args.zarr_path,
        roi_cache_manifest=args.roi_cache_manifest,
        crop_run=args.crop_run,
        rows=_parse_rows(args.rows),
        sample_count=args.sample_count,
        seed=args.sample_seed,
        boundary_sample_count=args.boundary_sample_count,
        reference_roi_live_acceleration=args.reference_roi_live_acceleration,
        reference_roi_live_gpu_chunk_frames=args.reference_roi_live_gpu_chunk_frames,
        batch_size=args.batch_size,
        max_abs_diff=args.max_abs_diff,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_p95_abs_diff=args.max_p95_abs_diff,
        top_mismatches=args.top_mismatches,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        diff = payload["diff"]
        print(f"status: {payload['status']}")
        print(f"crop_run: {payload['inputs']['crop_run']}")
        print(f"rows_compared: {diff['rows_compared']}")
        print(f"byte_equal: {diff['byte_equal']}")
        print(f"max_abs_diff: {diff['max_abs_diff']}")
        print(f"mean_abs_diff: {float(diff['mean_abs_diff']):.6f}")
        print(f"p95_abs_diff: {float(diff['p95_abs_diff']):.6f}")
        print(
            "reference: "
            f"mode={payload['source']['reference_roi_read_mode']} "
            f"accel={payload['source']['reference_roi_live_acceleration_effective']}"
        )
        print(
            "cache: "
            f"mode={payload['source']['cache_roi_read_mode']} "
            f"backend={payload['source']['cache_manifest_builder']['decode_backend_effective']} "
            f"pixel_contract={payload['source']['cache_manifest_builder']['pixel_contract']}"
        )
        if payload["failures"]:
            print("failures:")
            for failure in payload["failures"]:
                print(f"- {failure}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
