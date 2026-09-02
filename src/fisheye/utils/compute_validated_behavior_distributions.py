"""Compute a compact distribution successor from one exact behavior export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.group_statistics.validated_behavior_distribution_specs import (
    distribution_metric_family_ids,
    distribution_metric_specs_for_families,
)
from fisheye.group_statistics.validated_behavior_distributions import (
    ValidatedBehaviorDistributionConfig,
    compute_validated_behavior_distributions,
    write_validated_behavior_distributions,
)


def _csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root",
        type=Path,
        help="Exact validated-behavior publication root.",
    )
    parser.add_argument(
        "--source-export-run-id",
        help="Exact parent export run ID; latest discovery is prohibited.",
    )
    parser.add_argument("--distribution-run-id")
    parser.add_argument(
        "--families",
        help="Comma-separated metric families. Default: every registered family.",
    )
    parser.add_argument(
        "--heading-match-atol-deg",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the mandatory sealed epoch-heading crosscheck.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New immutable output directory; existing paths are never overwritten.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the immutable successor after successful computation.",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List registered distribution families and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_families:
        for family in distribution_metric_family_ids():
            print(family)
        return 0
    missing = [
        flag
        for flag, value in (
            ("--export-root", args.export_root),
            ("--source-export-run-id", args.source_export_run_id),
            ("--distribution-run-id", args.distribution_run_id),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.apply and args.output_dir is None:
        parser.error("--apply requires --output-dir")

    specs = distribution_metric_specs_for_families(_csv(args.families))
    dataset = ValidatedBehaviorExportDataset.open(
        args.export_root,
        str(args.source_export_run_id),
        validate=True,
        full_part_hashes=False,
    )
    config = ValidatedBehaviorDistributionConfig(
        distribution_run_id=str(args.distribution_run_id),
        metric_specs=specs,
        heading_match_atol_deg=float(args.heading_match_atol_deg),
    )
    result = compute_validated_behavior_distributions(
        dataset,
        config,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(f"distribution_run_id\t{config.distribution_run_id}")
    print(f"source_export_run_id\t{dataset.export_run_id}")
    print(f"source_export_manifest_sha256\t{dataset.cache_identity}")
    print(f"metric_specs\t{len(specs)}")
    for key, value in result.cohort_summary.items():
        print(f"{key}\t{value}")
    if not args.apply:
        print("dry_run\ttrue")
        print("pass --apply with --output-dir to write the immutable successor")
        return 0
    assert args.output_dir is not None
    manifest = write_validated_behavior_distributions(result, args.output_dir)
    print(f"manifest\t{manifest['manifest_path']}")
    print(f"record_sha256\t{manifest['record_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
