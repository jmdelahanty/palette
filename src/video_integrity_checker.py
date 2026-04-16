#!/usr/bin/env python3
"""Compatibility wrapper for the unified fisheye.diagnostics.video report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from fisheye.diagnostics.video import build_video_report
from fisheye.diagnostics.video.render import render_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check video integrity and detect missing frames.")
    parser.add_argument("video_file", help="Path to video file")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to analyze")
    parser.add_argument("--save-plot", help="Ignored; plotting is not available in the unified tool.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check against a small frame sample instead of a full frame scan.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video_file).expanduser()
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return 1
    if args.save_plot:
        print("Plot output is not implemented in fisheye.diagnostics.video and will be ignored.")

    if args.quick:
        full_scan = False
        sample_frames = int(args.max_frames or 120)
    elif args.max_frames is not None:
        full_scan = False
        sample_frames = int(args.max_frames)
    else:
        full_scan = True
        sample_frames = 120

    report = build_video_report(
        video_path,
        include_probe=True,
        include_timing=True,
        include_gop=True,
        include_decode=True,
        full_scan=full_scan,
        sample_frames=sample_frames,
        decode_backend="all",
    )
    print(render_report(report))
    return 0 if report.overall_status != "fail" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
