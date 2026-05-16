"""Build a flat binary ROI cache for a crop run.

This CLI is intended for workflow-local caches, especially cluster pipelines
where downstream pose/segmentation stages should avoid re-decoding video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fisheye.shared.flat_roi_cache import build_flat_roi_cache


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis/training Zarr archive containing crop_runs.")
    parser.add_argument("--crop-run", default=None, help="Crop run to cache (default: latest_any/latest).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the .json manifest and .bin payload should be written.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Explicit output manifest path. The payload is written next to it with .bin suffix.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="ROI rows written per batch.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing matching manifest/payload.")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Compute and record a payload sha256. This adds a full streaming hash pass during writing.",
    )
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration when the source crop run is geometry-only.",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full manifest JSON.")
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Write JSONL progress telemetry while the cache is building.",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=30.0,
        help="Emit progress at least this often in seconds; <=0 disables time-based emission.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=0,
        help="Emit progress every N batches; 0 disables count-based emission.",
    )
    parser.add_argument(
        "--progress-stderr",
        action="store_true",
        help="Also print compact progress summaries to stderr.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None and args.manifest_path is None:
        parser.error("Provide --output-dir or --manifest-path.")

    progress_handle = None
    if args.progress_jsonl is not None:
        args.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        progress_handle = args.progress_jsonl.open("a", encoding="utf-8", buffering=1)

    def emit_progress(event: dict) -> None:
        if progress_handle is not None:
            progress_handle.write(json.dumps(event, sort_keys=True) + "\n")
            progress_handle.flush()
        if args.progress_stderr:
            print(_format_progress_event(event), file=sys.stderr, flush=True)

    try:
        manifest = build_flat_roi_cache(
            zarr_path=args.zarr_path,
            crop_run=args.crop_run,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            compute_sha256=args.sha256,
            roi_live_acceleration=args.roi_live_acceleration,
            roi_live_gpu_chunk_frames=args.roi_live_gpu_chunk_frames,
            progress_callback=emit_progress if progress_handle is not None or args.progress_stderr else None,
            progress_interval_seconds=float(args.progress_interval_s),
            progress_every_batches=int(args.progress_every_batches),
        )
    finally:
        if progress_handle is not None:
            progress_handle.close()
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        array = manifest.get("array") if isinstance(manifest.get("array"), dict) else {}
        print(f"manifest={manifest.get('manifest_path')}")
        print(f"payload={array.get('bin_path')}")
        print(f"crop_run={manifest.get('source', {}).get('crop_run_name')}")
        print(f"shape={array.get('shape')} dtype={array.get('dtype')} layout={manifest.get('layout')}")
    return 0


def _format_progress_event(event: dict) -> str:
    name = event.get("event")
    if name == "start":
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        return (
            "flat_roi_cache_progress=start "
            f"total_rois={event.get('total_rois')} "
            f"total_mib={_mib(event.get('total_bytes')):.1f} "
            f"batch_size={event.get('batch_size')} "
            f"storage={source.get('storage_mode')} "
            f"read_mode={source.get('roi_read_mode')} "
            f"accel={source.get('roi_live_acceleration_effective')}"
        )
    if name == "batch":
        progress = event.get("progress") if isinstance(event.get("progress"), dict) else {}
        batch = event.get("batch") if isinstance(event.get("batch"), dict) else {}
        return (
            "flat_roi_cache_progress=batch "
            f"batch={batch.get('index')} "
            f"rows={progress.get('rows_written')}/{progress.get('rows_total')} "
            f"mib={progress.get('mib_written'):.1f}/{progress.get('mib_total'):.1f} "
            f"elapsed_s={progress.get('elapsed_seconds'):.1f} "
            f"eta_s={_format_optional_float(progress.get('eta_seconds'))} "
            f"overall_roi_s={progress.get('rows_per_second'):.1f} "
            f"read_roi_s={progress.get('read_rows_per_second'):.1f} "
            f"write_mib_s={progress.get('write_mib_per_second'):.1f} "
            f"batch_read_s={batch.get('read_seconds'):.3f} "
            f"batch_write_s={batch.get('write_seconds'):.3f}"
        )
    if name == "complete":
        progress = event.get("progress") if isinstance(event.get("progress"), dict) else {}
        timing = event.get("timing") if isinstance(event.get("timing"), dict) else {}
        return (
            "flat_roi_cache_progress=complete "
            f"rows={progress.get('rows_written')}/{progress.get('rows_total')} "
            f"elapsed_s={progress.get('elapsed_seconds'):.1f} "
            f"overall_roi_s={timing.get('rows_per_second'):.1f} "
            f"read_roi_s={timing.get('read_rows_per_second'):.1f} "
            f"write_mib_s={timing.get('write_mib_per_second'):.1f}"
        )
    if name == "reuse_existing":
        return f"flat_roi_cache_progress=reuse_existing manifest={event.get('manifest_path')}"
    return f"flat_roi_cache_progress={name}"


def _mib(value: object) -> float:
    try:
        return float(value) / (1024 * 1024)
    except (TypeError, ValueError):
        return 0.0


def _format_optional_float(value: object) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
