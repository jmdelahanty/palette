"""Render PNG/HTML views from one exact behavior-distribution generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.analytics_exports.validated_behavior_product_catalog import (
    BEHAVIOR_DISTRIBUTION,
    BEHAVIOR_DISTRIBUTION_REPORT,
    ValidatedBehaviorProductCatalogError,
    canonical_validated_behavior_product_dir,
    register_validated_behavior_product,
    resolve_validated_behavior_product,
)
from fisheye.group_statistics.validated_behavior_distribution_report import (
    render_validated_behavior_distribution_report,
)
from fisheye.group_statistics.validated_behavior_distribution_views import (
    DEFAULT_DISPLAY_RANGE,
    DISPLAY_RANGE_LABELS,
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
        help=(
            "Optional exact report directory. When omitted with --apply, publish "
            "under the source export's co-located product namespace and append its "
            "product catalog. Existing paths are not overwritten."
        ),
    )
    parser.add_argument("--metrics", help="Optional comma-separated exact metric IDs.")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--display-range",
        choices=tuple(DISPLAY_RANGE_LABELS),
        default=DEFAULT_DISPLAY_RANGE,
        help=(
            "Display-only x-axis range. central_99 retains whole bins covering at "
            "least 99%% of every equal-recording series; all evidence remains sealed."
        ),
    )
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
    print(f"display_range\t{args.display_range}")
    source_export = source.manifest["source_export"]
    export_root = Path(str(source_export["path"]))
    export_run_id = str(source_export["export_run_id"])
    canonical_output: Path | None
    try:
        canonical_output = canonical_validated_behavior_product_dir(
            export_root,
            BEHAVIOR_DISTRIBUTION_REPORT,
            args.report_run_id,
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
            "pass --apply to write and catalog the co-located immutable report; "
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
    dataset: ValidatedBehaviorExportDataset | None = None
    if canonical_output is not None and output_dir == canonical_output:
        dataset = ValidatedBehaviorExportDataset.open(
            export_root,
            export_run_id,
            validate=True,
            full_part_hashes=False,
        )
        parent = resolve_validated_behavior_product(
            export_root,
            export_run_id,
            product_kind=BEHAVIOR_DISTRIBUTION,
            product_run_id=source.distribution_run_id,
        )
        if (
            parent.root != source.root
            or parent.manifest_record_sha256 != source.cache_identity
        ):
            raise ValueError(
                "The report source is not the exact catalog-selected co-located "
                "distribution product"
            )
    manifest = render_validated_behavior_distribution_report(
        source,
        report_run_id=args.report_run_id,
        output_dir=output_dir,
        metric_ids=selected,
        dpi=args.dpi,
        display_range_id=args.display_range,
    )
    print(f"manifest\t{manifest['manifest_path']}")
    print(f"record_sha256\t{manifest['record_sha256']}")
    print(f"index\t{output_dir / 'index.html'}")
    if dataset is not None:
        catalog = register_validated_behavior_product(
            dataset,
            product_kind=BEHAVIOR_DISTRIBUTION_REPORT,
            product_root=output_dir,
        )
        print(f"catalog_manifest\t{catalog['catalog_manifest_path']}")
        print(f"catalog_record_sha256\t{catalog['record_sha256']}")
        print(f"catalog_generation_id\t{catalog['catalog_generation_id']}")
        print("catalog_registered\ttrue")
    else:
        print("catalog_registered\tfalse")
        print("catalog_reason\texplicit_noncanonical_output_dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
