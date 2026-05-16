"""Build a flat binary ROI cache for a crop run.

This CLI is intended for workflow-local caches, especially cluster pipelines
where downstream pose/segmentation stages should avoid re-decoding video.
"""

from __future__ import annotations

import argparse
import json
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None and args.manifest_path is None:
        parser.error("Provide --output-dir or --manifest-path.")

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
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        array = manifest.get("array") if isinstance(manifest.get("array"), dict) else {}
        print(f"manifest={manifest.get('manifest_path')}")
        print(f"payload={array.get('bin_path')}")
        print(f"crop_run={manifest.get('source', {}).get('crop_run_name')}")
        print(f"shape={array.get('shape')} dtype={array.get('dtype')} layout={manifest.get('layout')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
