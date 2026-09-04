"""Compute a compact distribution successor from one exact behavior export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.analytics_exports.validated_behavior_product_catalog import (
    BEHAVIOR_DISTRIBUTION,
    ValidatedBehaviorProductCatalogError,
    canonical_validated_behavior_product_dir,
    register_validated_behavior_product,
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
        help=(
            "Optional exact output directory. When omitted with --apply, publish "
            "under the source export's co-located product namespace and append its "
            "product catalog. Existing paths are never overwritten."
        ),
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
    canonical_output: Path | None
    try:
        canonical_output = canonical_validated_behavior_product_dir(
            dataset.root,
            BEHAVIOR_DISTRIBUTION,
            config.distribution_run_id,
        )
    except ValidatedBehaviorProductCatalogError:
        if args.output_dir is None:
            raise
        canonical_output = None
    if not args.apply:
        if canonical_output is not None:
            print(f"canonical_output_dir\t{canonical_output}")
        if args.output_dir is not None:
            print(f"explicit_output_dir\t{args.output_dir.expanduser().resolve()}")
        print("dry_run\ttrue")
        print(
            "pass --apply to write and catalog the co-located immutable successor; "
            "--output-dir retains an explicit uncataloged override"
        )
        return 0
    assert canonical_output is not None or args.output_dir is not None
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else canonical_output
    )
    assert output_dir is not None
    manifest = write_validated_behavior_distributions(result, output_dir)
    print(f"manifest\t{manifest['manifest_path']}")
    print(f"record_sha256\t{manifest['record_sha256']}")
    if canonical_output is not None and output_dir == canonical_output:
        catalog = register_validated_behavior_product(
            dataset,
            product_kind=BEHAVIOR_DISTRIBUTION,
            product_root=output_dir,
        )
        print(f"catalog_manifest\t{catalog['catalog_manifest_path']}")
        print(f"catalog_record_sha256\t{catalog['record_sha256']}")
        print(f"catalog_generation_id\t{catalog['catalog_generation_id']}")
        print(f"catalog_registered\ttrue")
    else:
        print("catalog_registered\tfalse")
        print("catalog_reason\texplicit_noncanonical_output_dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
