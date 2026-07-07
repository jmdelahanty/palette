"""Build a flat binary ROI cache for a finalized clipped refined-detect collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fisheye.shared.clipped_collection_flat_roi_cache import build_clipped_collection_flat_roi_cache


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr containing a finalized clipped collection.")
    parser.add_argument(
        "--collection-id",
        default=None,
        help="Finalized collection id. Defaults to refined_detect_runs.latest_collection.",
    )
    parser.add_argument(
        "--recording-frame-index",
        type=Path,
        default=None,
        help="Override recording_frame_index.parquet path.",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        default=None,
        help="Restrict cache generation to one finalized collection clip id. Repeatable.",
    )
    parser.add_argument(
        "--work-unit-id",
        action="append",
        default=None,
        help="Restrict cache generation to one finalized collection work_unit_id. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for manifest/bin/rows outputs.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Explicit output manifest path. Payload and row index are written next to it.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="ROI size in Palette order. Defaults to preferred zarr crop ROI size.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Debug/smoke limit on total ROI rows materialized.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--sha256", action="store_true", help="Compute and record payload sha256.")
    parser.add_argument(
        "--gpu-chunk-frames",
        type=int,
        default=32,
        help="Sequential PyNvVideoCodec frame batch size per clip.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full manifest JSON.")
    parser.add_argument("--progress-jsonl", type=Path, default=None, help="Write JSONL progress telemetry.")
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
        help="Emit progress every N ROI-producing frame batches; 0 disables count-based emission.",
    )
    parser.add_argument("--progress-stderr", action="store_true", help="Print compact progress to stderr.")
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
        manifest = build_clipped_collection_flat_roi_cache(
            zarr_path=args.zarr_path,
            collection_id=args.collection_id,
            recording_frame_index=args.recording_frame_index,
            clip_ids=args.clip_id,
            work_unit_ids=args.work_unit_id,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
            roi_size=args.roi_size,
            limit_rows=args.limit_rows,
            overwrite=args.overwrite,
            compute_sha256=args.sha256,
            gpu_chunk_frames=args.gpu_chunk_frames,
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
        row_index = manifest.get("row_index") if isinstance(manifest.get("row_index"), dict) else {}
        source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        print(f"manifest={manifest.get('manifest_path')}")
        print(f"payload={array.get('bin_path')}")
        print(f"row_index={row_index.get('path')}")
        print(f"collection_id={source.get('collection_id')}")
        print(f"shape={array.get('shape')} dtype={array.get('dtype')} layout={manifest.get('layout')}")
    return 0


def _format_progress_event(event: dict) -> str:
    name = event.get("event")
    if name == "start":
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        return (
            "clipped_collection_flat_roi_cache_progress=start "
            f"collection={source.get('collection_id')} "
            f"selected_runs={source.get('selected_run_count')} "
            f"total_rois={event.get('total_rois')} "
            f"total_mib={_mib(event.get('total_bytes')):.1f} "
            f"gpu_chunk_frames={event.get('gpu_chunk_frames')}"
        )
    if name == "batch":
        progress = event.get("progress") if isinstance(event.get("progress"), dict) else {}
        batch = event.get("batch") if isinstance(event.get("batch"), dict) else {}
        return (
            "clipped_collection_flat_roi_cache_progress=batch "
            f"batch={batch.get('index')} "
            f"frame={batch.get('frame_index')} "
            f"rows={progress.get('rows_written')}/{progress.get('rows_total')} "
            f"mib={progress.get('mib_written'):.1f}/{progress.get('mib_total'):.1f} "
            f"elapsed_s={progress.get('elapsed_seconds'):.1f} "
            f"eta_s={_format_optional_float(progress.get('eta_seconds'))} "
            f"overall_roi_s={progress.get('rows_per_second'):.1f} "
            f"decode_s={batch.get('decode_seconds'):.3f} "
            f"crop_s={batch.get('crop_seconds'):.3f} "
            f"write_s={batch.get('write_seconds'):.3f}"
        )
    if name == "complete":
        progress = event.get("progress") if isinstance(event.get("progress"), dict) else {}
        timing = event.get("timing") if isinstance(event.get("timing"), dict) else {}
        return (
            "clipped_collection_flat_roi_cache_progress=complete "
            f"rows={progress.get('rows_written')}/{progress.get('rows_total')} "
            f"elapsed_s={progress.get('elapsed_seconds'):.1f} "
            f"overall_roi_s={timing.get('rows_per_second'):.1f} "
            f"decode_s={timing.get('decode_seconds_total'):.1f} "
            f"crop_s={timing.get('crop_seconds_total'):.1f} "
            f"write_s={timing.get('write_seconds_total'):.1f}"
        )
    if name == "reuse_existing":
        return f"clipped_collection_flat_roi_cache_progress=reuse_existing manifest={event.get('manifest_path')}"
    return f"clipped_collection_flat_roi_cache_progress={name}"


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
