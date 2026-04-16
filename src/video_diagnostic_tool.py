#!/usr/bin/env python3
"""Compatibility wrapper for the unified fisheye.diagnostics.video report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from fisheye.diagnostics.video import build_video_report
from fisheye.diagnostics.video.render import render_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose video encoding issues.")
    parser.add_argument("video_file", help="Path to video file")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Not implemented in the unified tool yet; suggested commands will still be shown.",
    )
    parser.add_argument(
        "--test-segment",
        action="store_true",
        help="Not implemented in the unified tool yet.",
    )
    parser.add_argument(
        "--output-format",
        choices=["h264", "hevc_nvenc"],
        default="h264",
        help="Reserved for future --fix support.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video_file).expanduser()
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return 1
    if args.fix:
        print("Automatic repair is not implemented in fisheye.diagnostics.video yet; showing suggestions only.")
    if args.test_segment:
        print("Test-segment extraction is not implemented in fisheye.diagnostics.video yet.")

    report = build_video_report(
        video_path,
        include_probe=True,
        include_timing=True,
        include_gop=True,
        include_decode=True,
        full_scan=False,
        sample_frames=120,
        decode_backend="all",
    )
    print(render_report(report))
    return 0 if report.overall_status != "fail" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
