"""Compute recording-scoped statistics from one validated-behavior export."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.group_statistics.validated_behavior import (
    ValidatedBehaviorGroupStatisticsConfig,
    compute_validated_behavior_group_statistics,
    write_validated_behavior_group_statistics_sandbox,
)
from fisheye.group_statistics.validated_behavior_specs import (
    histogram_specs_for_families,
    metric_specs_for_families,
    validated_behavior_family_ids,
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
        help="Validated-behavior publication root containing validated_behavior/v1.",
    )
    parser.add_argument(
        "--source-export-run-id",
        help="Exact validated-behavior export run ID; latest discovery is prohibited.",
    )
    parser.add_argument("--statistics-run-id")
    parser.add_argument(
        "--families",
        help="Comma-separated metric families. Default: every registered family.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-iterations", type=int, default=5000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-recordings", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New sandbox output directory. Existing paths are never overwritten.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically write selector-ineligible sandbox Parquet and manifest files.",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List registered metric families and exit before opening an export.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_families:
        for family in validated_behavior_family_ids():
            print(family)
        return 0
    missing = [
        flag
        for flag, value in (
            ("--export-root", args.export_root),
            ("--source-export-run-id", args.source_export_run_id),
            ("--statistics-run-id", args.statistics_run_id),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.apply and args.output_dir is None:
        parser.error("--apply requires --output-dir")

    families = _csv(args.families)
    specs = metric_specs_for_families(families)
    histogram_specs = histogram_specs_for_families(families)
    dataset = ValidatedBehaviorExportDataset.open(
        args.export_root,
        str(args.source_export_run_id),
        validate=True,
        full_part_hashes=False,
    )
    config = ValidatedBehaviorGroupStatisticsConfig(
        statistics_run_id=str(args.statistics_run_id),
        metric_specs=specs,
        histogram_specs=histogram_specs,
        bootstrap_iterations=int(args.bootstrap_iterations),
        permutation_iterations=int(args.permutation_iterations),
        confidence_level=float(args.confidence_level),
        minimum_recordings=int(args.minimum_recordings),
        random_seed=int(args.random_seed),
    )
    result = compute_validated_behavior_group_statistics(dataset, config)
    print(f"statistics_run_id\t{config.statistics_run_id}")
    print(f"source_export_run_id\t{dataset.export_run_id}")
    print(f"source_manifest_sha256\t{dataset.cache_identity}")
    print("parent_recordings\t" f"{result.cohort_summary['parent_recording_count']}")
    print(f"metric_specs\t{len(config.metric_specs)}")
    print(f"histogram_specs\t{len(config.histogram_specs)}")
    print(f"recording_metric_values\t{len(result.recording_values)}")
    print(f"descriptive_statistics\t{len(result.descriptive_statistics)}")
    print(f"paired_contrasts\t{len(result.paired_contrasts)}")
    print(f"recording_histogram_bins\t{len(result.recording_histogram_bins)}")
    print(
        "histogram_descriptive_statistics\t"
        f"{len(result.histogram_descriptive_statistics)}"
    )
    print("analysis_status\texploratory")
    print("acquisition_batch_adjustment\tnot_performed_identity_unavailable")
    if not args.apply:
        print("dry_run\ttrue")
        print("pass --apply with --output-dir to write the sandbox generation")
        return 0

    assert args.output_dir is not None
    manifest = write_validated_behavior_group_statistics_sandbox(
        result,
        args.output_dir,
    )
    print(f"manifest\t{manifest['manifest_path']}")
    print(f"record_sha256\t{manifest['record_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
