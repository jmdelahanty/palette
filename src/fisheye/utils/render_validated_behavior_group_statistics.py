"""Render a receipt-bound static report from exact grouped statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.group_statistics.validated_behavior_report import (
    render_validated_behavior_statistics_report,
)
from fisheye.group_statistics.validated_behavior_views import (
    VIEW_DEFINITIONS,
    ValidatedBehaviorStatisticsViewSource,
    available_statistics_views,
    build_statistics_view_payload,
)


def _csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("--views must contain at least one view ID")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        help="Exact grouped-statistics generation containing manifest.json.",
    )
    parser.add_argument("--report-run-id")
    parser.add_argument(
        "--views",
        help="Comma-separated view IDs. Default: every available view.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New report directory. Existing paths are never overwritten.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically write the selector-ineligible static report.",
    )
    parser.add_argument(
        "--list-views",
        action="store_true",
        help="List registered views and exit before opening statistics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_views:
        for definition in VIEW_DEFINITIONS:
            print(f"{definition.view_id}\t{definition.label}")
        return 0

    missing = [
        flag
        for flag, value in (
            ("--statistics-dir", args.statistics_dir),
            ("--report-run-id", args.report_run_id),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.apply and args.output_dir is None:
        parser.error("--apply requires --output-dir")

    source = ValidatedBehaviorStatisticsViewSource.open(args.statistics_dir)
    available = {item.view_id: item for item in available_statistics_views(source)}
    selected = _csv(args.views) or tuple(available)
    if len(set(selected)) != len(selected):
        parser.error("--views must contain unique view IDs")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        parser.error(f"views are unavailable in this generation: {unknown}")

    print(f"statistics_run_id\t{source.statistics_run_id}")
    print(f"statistics_manifest_sha256\t{source.cache_identity}")
    print(f"report_run_id\t{args.report_run_id}")
    print(f"dpi\t{args.dpi}")
    for view_id in selected:
        payload = build_statistics_view_payload(source, view_id)
        print(
            "view\t"
            f"{view_id}\t{payload['payload_sha256']}\t"
            f"recording={len(payload['recording_rows'])}\t"
            f"descriptive={len(payload['descriptive_rows'])}\t"
            f"contrasts={len(payload['contrast_rows'])}"
        )
    if not args.apply:
        print("dry_run\ttrue")
        print("pass --apply with --output-dir to render the atomic report")
        return 0

    assert args.output_dir is not None
    manifest = render_validated_behavior_statistics_report(
        source,
        report_run_id=str(args.report_run_id),
        output_dir=args.output_dir,
        view_ids=selected,
        dpi=int(args.dpi),
    )
    print(f"manifest\t{manifest['manifest_path']}")
    print(f"record_sha256\t{manifest['record_sha256']}")
    print(f"index\t{args.output_dir.expanduser().resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
