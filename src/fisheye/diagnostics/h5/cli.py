from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from . import build_h5_report
from .batch import build_batch_report
from .render import render_batch_jsonl, render_batch_report, render_report


def _add_common_input_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path", type=Path, help="Path to an H5 file or recording directory.")


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=["palette-import", "citrus-contract"],
        default="palette-import",
        help="Validation profile (default: palette-import).",
    )


def _write_text_output(path: Path, content: str) -> None:
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect raw Citrus H5 files for Palette ingest readiness.")
    subparsers = parser.add_subparsers(dest="command")

    report = subparsers.add_parser("report", help="Run the combined H5 health report.")
    _add_common_input_argument(report)
    _add_profile_argument(report)
    _add_output_arguments(report)

    batch = subparsers.add_parser("batch", help="Scan one or more files or directories and summarize H5 health.")
    batch.add_argument("paths", nargs="+", type=Path, help="H5 files or directories to scan.")
    _add_profile_argument(batch)
    batch.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not recurse into subdirectories when scanning directories.",
    )
    batch.add_argument("--limit", type=int, default=None, help="Optional maximum number of H5 files to inspect.")
    batch.add_argument("--jsonl", type=Path, default=None, help="Write one JSON object per inspected H5 to this file.")
    batch.set_defaults(recursive=True)
    _add_output_arguments(batch)

    events = subparsers.add_parser("events", help="Inspect /events only.")
    _add_common_input_argument(events)
    _add_profile_argument(events)
    _add_output_arguments(events)

    frame_metadata = subparsers.add_parser("frame-metadata", help="Inspect /video_metadata/frame_metadata only.")
    _add_common_input_argument(frame_metadata)
    _add_profile_argument(frame_metadata)
    _add_output_arguments(frame_metadata)

    tracking = subparsers.add_parser("tracking", help="Inspect optional /tracking_data datasets.")
    _add_common_input_argument(tracking)
    _add_profile_argument(tracking)
    _add_output_arguments(tracking)

    snapshots = subparsers.add_parser("snapshots", help="Inspect optional metadata and snapshot groups.")
    _add_common_input_argument(snapshots)
    _add_profile_argument(snapshots)
    _add_output_arguments(snapshots)

    enums = subparsers.add_parser("enums", help="Inspect optional /enums datasets.")
    _add_common_input_argument(enums)
    _add_profile_argument(enums)
    _add_output_arguments(enums)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "report"

    if command == "batch":
        report = build_batch_report(
            getattr(args, "paths", []),
            recursive=bool(getattr(args, "recursive", True)),
            limit=getattr(args, "limit", None),
            profile=str(getattr(args, "profile", "palette-import")),
        )
        jsonl_path = getattr(args, "jsonl", None)
        if jsonl_path is not None:
            _write_text_output(Path(jsonl_path), render_batch_jsonl(report))
        print(render_batch_report(report, as_json=bool(getattr(args, "json", False))))
        if report.summary.scanned == 0:
            return 1
        return 0 if report.overall_status not in {"fail", "error"} else 2

    input_path = Path(getattr(args, "input_path", "")).expanduser()
    kwargs = {"profile": str(getattr(args, "profile", "palette-import"))}
    if command == "events":
        kwargs.update(include_frame_metadata=False, include_tracking=False, include_snapshots=False, include_enums=False)
    elif command == "frame-metadata":
        kwargs.update(include_events=False, include_tracking=False, include_snapshots=False, include_enums=False)
    elif command == "tracking":
        kwargs.update(include_events=False, include_frame_metadata=False, include_tracking=True, include_snapshots=False, include_enums=False)
    elif command == "snapshots":
        kwargs.update(include_events=False, include_frame_metadata=False, include_tracking=False, include_snapshots=True, include_enums=False)
    elif command == "enums":
        kwargs.update(include_events=False, include_frame_metadata=False, include_tracking=False, include_snapshots=False, include_enums=True)

    report = build_h5_report(input_path, **kwargs)
    print(render_report(report, as_json=bool(getattr(args, "json", False))))
    return 0 if report.overall_status not in {"fail", "error"} else 2
