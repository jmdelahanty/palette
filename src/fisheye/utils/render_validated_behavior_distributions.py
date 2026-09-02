"""Render PNG/HTML views from one exact behavior-distribution generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.group_statistics.validated_behavior_distribution_report import (
    render_validated_behavior_distribution_report,
)
from fisheye.group_statistics.validated_behavior_distribution_views import (
    ValidatedBehaviorDistributionViewSource,
    available_distribution_metrics,
)


def _csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("--metrics must contain a metric ID")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distribution-dir",
        type=Path,
        required=True,
        help="Exact immutable distribution-generation directory.",
    )
    parser.add_argument("--report-run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New immutable report directory; existing paths are not overwritten.",
    )
    parser.add_argument("--metrics", help="Optional comma-separated exact metric IDs.")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List metrics in the exact distribution and exit.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write the immutable PNG/HTML report."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source = ValidatedBehaviorDistributionViewSource.open(args.distribution_dir)
    metrics = available_distribution_metrics(source)
    if args.list_metrics:
        for metric in metrics:
            print(
                f"{metric['metric_id']}\t{','.join(metric['weighting_ids'])}\t"
                f"{metric['interpretation']}"
            )
        return 0
    selected = _csv(args.metrics)
    print(f"distribution_run_id\t{source.distribution_run_id}")
    print(f"distribution_manifest_sha256\t{source.cache_identity}")
    print(f"selected_metric_count\t{len(selected or metrics)}")
    if not args.apply:
        print("dry_run\ttrue")
        print("pass --apply with --output-dir to write the report")
        return 0
    if args.output_dir is None:
        raise ValueError("--apply requires --output-dir")
    manifest = render_validated_behavior_distribution_report(
        source,
        report_run_id=args.report_run_id,
        output_dir=args.output_dir,
        metric_ids=selected,
        dpi=args.dpi,
    )
    print(f"manifest\t{manifest['manifest_path']}")
    print(f"record_sha256\t{manifest['record_sha256']}")
    print(f"index\t{args.output_dir.expanduser().resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
