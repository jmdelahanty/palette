from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from . import build_video_report
from .batch import build_batch_report
from .render import render_batch_jsonl, render_batch_report, render_report

DEFAULT_SAMPLE_FRAMES = 120
DEFAULT_DECODE_FRAMES = 30
DEFAULT_SEEK_SAMPLES = 10


def _add_common_video_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video_file", type=Path, help="Path to the video file.")


def _add_scan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Inspect the full frame stream instead of the default quick sample.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_SAMPLE_FRAMES,
        help=f"Frame sample count for quick inspections (default: {DEFAULT_SAMPLE_FRAMES}).",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _write_text_output(path: Path, content: str) -> None:
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect raw-video health and decode behavior.")
    subparsers = parser.add_subparsers(dest="command")

    report = subparsers.add_parser("report", help="Run the combined video health report.")
    _add_common_video_argument(report)
    _add_scan_arguments(report)
    report.add_argument(
        "--backend",
        choices=["opencv", "decord", "all"],
        default="all",
        help="Decode backend selection for decode checks (default: all).",
    )
    _add_output_arguments(report)

    probe = subparsers.add_parser("probe", help="Inspect stream/container metadata only.")
    _add_common_video_argument(probe)
    _add_output_arguments(probe)

    timing = subparsers.add_parser("timing", help="Inspect timestamp continuity and gaps.")
    _add_common_video_argument(timing)
    _add_scan_arguments(timing)
    _add_output_arguments(timing)

    gop = subparsers.add_parser("gop", help="Inspect keyframe and GOP structure.")
    _add_common_video_argument(gop)
    _add_scan_arguments(gop)
    _add_output_arguments(gop)

    decode = subparsers.add_parser("decode", help="Run backend decode smoke tests.")
    _add_common_video_argument(decode)
    decode.add_argument(
        "--backend",
        choices=["opencv", "decord", "all"],
        default="all",
        help="Decode backend selection (default: all).",
    )
    _add_output_arguments(decode)

    batch = subparsers.add_parser("batch", help="Scan one or more directories or files and summarize video health.")
    batch.add_argument("paths", nargs="+", type=Path, help="Video files or directories to scan.")
    _add_scan_arguments(batch)
    batch.add_argument(
        "--backend",
        choices=["opencv", "decord", "all"],
        default="all",
        help="Decode backend selection for decode checks (default: all).",
    )
    batch.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not recurse into subdirectories when scanning directories.",
    )
    batch.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of videos to inspect.",
    )
    batch.add_argument(
        "--source",
        choices=["all", "cams", "raw", "other"],
        default="all",
        help="Limit batch discovery to a specific source bucket (default: all).",
    )
    batch.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Write one JSON object per inspected video to this file.",
    )
    batch.set_defaults(recursive=True)
    _add_output_arguments(batch)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "report"
    if command == "batch":
        kwargs = {
            "full_scan": bool(getattr(args, "full_scan", False)),
            "sample_frames": int(getattr(args, "max_frames", DEFAULT_SAMPLE_FRAMES)),
            "decode_backend": str(getattr(args, "backend", "all")),
            "decode_frames": DEFAULT_DECODE_FRAMES,
            "seek_samples": DEFAULT_SEEK_SAMPLES,
            "include_probe": True,
            "include_timing": True,
            "include_gop": True,
            "include_decode": True,
        }
        report = build_batch_report(
            getattr(args, "paths", []),
            recursive=bool(getattr(args, "recursive", True)),
            limit=getattr(args, "limit", None),
            source=str(getattr(args, "source", "all")),
            **kwargs,
        )
        jsonl_path = getattr(args, "jsonl", None)
        if jsonl_path is not None:
            _write_text_output(Path(jsonl_path), render_batch_jsonl(report))
        print(render_batch_report(report, as_json=bool(getattr(args, "json", False))))
        if report.summary.scanned == 0:
            return 1
        return 0 if report.overall_status != "fail" else 2

    video_path = Path(getattr(args, "video_file", "")).expanduser()
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return 1

    kwargs = {
        "full_scan": bool(getattr(args, "full_scan", False)),
        "sample_frames": int(getattr(args, "max_frames", DEFAULT_SAMPLE_FRAMES)),
        "decode_backend": str(getattr(args, "backend", "all")),
        "decode_frames": DEFAULT_DECODE_FRAMES,
        "seek_samples": DEFAULT_SEEK_SAMPLES,
    }
    if command == "probe":
        kwargs.update(include_container=True, include_timing=False, include_gop=False, include_decode=False)
    elif command == "timing":
        kwargs.update(include_container=False, include_probe=True, include_gop=False, include_decode=False)
    elif command == "gop":
        kwargs.update(include_container=False, include_probe=True, include_timing=False, include_decode=False)
    elif command == "decode":
        kwargs.update(include_container=False, include_probe=True, include_timing=False, include_gop=False)
    else:
        kwargs.update(include_container=True, include_probe=True, include_timing=True, include_gop=True, include_decode=True)

    report = build_video_report(video_path, **kwargs)
    print(render_report(report, as_json=bool(getattr(args, "json", False))))
    return 0 if report.overall_status != "fail" else 2
