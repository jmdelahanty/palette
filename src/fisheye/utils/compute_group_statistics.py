"""Compute cross-recording group statistics from Palette analytics exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.group_statistics.goodcopbadcop import (
    GoodCopBadCopStatisticsConfig,
    DESCRIPTIVE_TABLE,
    SUMMARY_TABLE,
    compute_goodcopbadcop_outputs,
    contrast_definitions,
    metric_specs_for_families,
    utc_run_id,
    write_goodcopbadcop_statistics,
)


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="chaser",
        choices=("chaser",),
        help="Statistics profile to compute.",
    )
    parser.add_argument(
        "--source-export-run-id",
        required=True,
        help="Cross-recording export run id to read.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        required=True,
        help="Analytics export root, e.g. /nvme1/exports/palette_analytics.",
    )
    parser.add_argument(
        "--stats-run-id",
        help="Statistics export run id. Defaults to stats_<UTC timestamp>.",
    )
    parser.add_argument(
        "--metrics",
        help=(
            "Comma-separated metric families. Default: all chaser-analysis "
            "families. Known: chaser_distance, spatial_occupancy, "
            "epoch_behavior, cra_primary_endpoint, cra_near_field, egocentric_alignment."
        ),
    )
    parser.add_argument(
        "--contrasts",
        help="Comma-separated contrasts. Default: training-pre,post-pre,post-training.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--permutation-iterations", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-recordings", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--apply", action="store_true", help="Write statistics Parquet/manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing stats run id.")
    parser.add_argument(
        "--allow-legacy-export-layout",
        action="store_true",
        help="Explicitly read a historical export without publication-v1.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = metric_specs_for_families(_parse_csv(args.metrics))
    contrasts = contrast_definitions(_parse_csv(args.contrasts))
    stats_run_id = str(args.stats_run_id or utc_run_id())
    config = GoodCopBadCopStatisticsConfig(
        export_root=Path(args.export_root),
        source_export_run_id=str(args.source_export_run_id),
        stats_run_id=stats_run_id,
        metrics=metrics,
        contrasts=contrasts,
        bootstrap_iterations=int(args.bootstrap_iterations),
        permutation_iterations=int(args.permutation_iterations),
        confidence_level=float(args.confidence_level),
        minimum_recordings=int(args.minimum_recordings),
        random_seed=int(args.random_seed),
        overwrite=bool(args.overwrite),
        allow_legacy_export_layout=bool(args.allow_legacy_export_layout),
    )
    rows, descriptive_rows, manifest = compute_goodcopbadcop_outputs(config)
    status_counts = manifest.get("status_counts", {})
    print(f"stats_run_id\t{stats_run_id}")
    print(f"source_export_run_id\t{args.source_export_run_id}")
    print(f"rows\t{SUMMARY_TABLE}\t{len(rows)}")
    print(f"rows\t{DESCRIPTIVE_TABLE}\t{len(descriptive_rows)}")
    if isinstance(status_counts, dict):
        for status, count in sorted(status_counts.items()):
            print(f"status\t{status}\t{count}")
    if not args.apply:
        print("dry_run\ttrue")
        print("pass --apply to write statistics Parquet and manifest")
        return 0

    written = write_goodcopbadcop_statistics(
        rows,
        manifest,
        export_root=Path(args.export_root),
        stats_run_id=stats_run_id,
        descriptive_rows=descriptive_rows,
        overwrite=bool(args.overwrite),
    )
    print(f"manifest\t{written['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
