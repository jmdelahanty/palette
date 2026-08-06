#!/usr/bin/env python3
"""Publish one node-local sampled training base without registry activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.training_base_publication import (
    publish_sampled_training_base,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--source-frame-count", type=int, required=True)
    parser.add_argument("--frame-step", type=int, required=True)
    parser.add_argument("--skip-tail-frames", type=int, default=0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = publish_sampled_training_base(
        destination=args.destination,
        scratch_root=args.scratch_root,
        video_path=args.video_path,
        source_frame_count=int(args.source_frame_count),
        frame_step=int(args.frame_step),
        skip_tail_frames=int(args.skip_tail_frames),
        config_path=args.config,
        camera_id=str(args.camera_id),
        recording_dir=args.recording_dir,
        h5_path=args.h5_path,
        gpu_id=int(args.gpu_id),
        require_cuda=True,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
