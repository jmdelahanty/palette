"""Adopt an exact validated-behavior product into its source dataset package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.analytics_exports.validated_behavior_product_catalog import (
    adopt_validated_behavior_product,
    canonical_validated_behavior_product_dir,
    inspect_validated_behavior_product,
    validated_behavior_product_kinds,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root",
        type=Path,
        required=True,
        help="Exact validated-behavior publication root named 'publication'.",
    )
    parser.add_argument(
        "--source-export-run-id",
        required=True,
        help="Exact parent export run ID; selector discovery is prohibited.",
    )
    parser.add_argument(
        "--product-kind",
        required=True,
        choices=validated_behavior_product_kinds(),
    )
    parser.add_argument(
        "--source-product-dir",
        type=Path,
        required=True,
        help="Existing exact immutable product directory to validate and adopt.",
    )
    parser.add_argument(
        "--catalog-generation-id",
        help="Optional exact catalog-generation ID; otherwise a UUID is generated.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Copy exact validated bytes when needed and append the product to a new "
            "catalog generation. The source export and source product are unchanged."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset = ValidatedBehaviorExportDataset.open(
        args.export_root,
        str(args.source_export_run_id),
        validate=True,
        full_part_hashes=False,
    )
    inspection = inspect_validated_behavior_product(
        str(args.product_kind), args.source_product_dir
    )
    target = canonical_validated_behavior_product_dir(
        dataset.root,
        str(args.product_kind),
        str(inspection["product_run_id"]),
    )
    print(f"source_export_run_id\t{dataset.export_run_id}")
    print(f"source_export_manifest_sha256\t{dataset.cache_identity}")
    print(f"product_kind\t{inspection['product_kind']}")
    print(f"product_run_id\t{inspection['product_run_id']}")
    print(f"product_manifest_sha256\t{inspection['manifest_record_sha256']}")
    print(f"source_product_dir\t{inspection['product_root']}")
    print(f"canonical_product_dir\t{target}")
    if not args.apply:
        print("dry_run\ttrue")
        print("pass --apply to copy when needed and publish the catalog successor")
        return 0

    result = adopt_validated_behavior_product(
        dataset,
        product_kind=str(args.product_kind),
        source_product_root=args.source_product_dir,
        catalog_generation_id=args.catalog_generation_id,
    )
    for key in (
        "copied",
        "catalog_reused",
        "catalog_generation_id",
        "catalog_record_sha256",
        "catalog_manifest_path",
        "product_root",
    ):
        print(f"{key}\t{result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
