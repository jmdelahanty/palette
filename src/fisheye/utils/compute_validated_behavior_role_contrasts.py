"""Compute aggressive-vs-inert chaser role contrasts from a validated export.

Reads one exact receipt-validated behavior export (never ``latest``), computes
paired aggressive-minus-inert recording-level contrasts within each chaser
epoch, and writes ``role_contrasts.parquet`` + ``manifest.json`` into a fresh
output directory.  The chaser_training rows are the primary contrast; the
chaser_pre/chaser_post rows are secondary and carry
``park_position_asymmetry = true`` because the two chasers park at different
corners outside training.

Example:
    python -m fisheye.utils.compute_validated_behavior_role_contrasts \\
        --export-root /path/to/publication \\
        --run-id goodbatbadbat-validated-behavior-phase-b-... \\
        --output-dir /tmp/role-contrasts-v001
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
)
from fisheye.group_statistics.validated_behavior_role_contrasts import (
    PRIMARY_EPOCH_ROLE,
    RoleContrastParameters,
    SECONDARY_EPOCH_ROLES,
    compute_role_contrast_rows,
    format_contrast_table,
    load_role_contrast_frames,
    write_role_contrasts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--export-root",
        required=True,
        help="Publication root of the validated-behavior cohort export",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Exact export run id to open (never a 'latest' alias)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write role_contrasts.parquet + manifest.json into",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a non-empty output directory instead of refusing",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10_000,
        help="Bootstrap resamples for the mean-difference CI (default 10000)",
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=10_000,
        help="Random sign-flip draws when n is too large for exact (default 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20_260_901,
        help="Base seed for the deterministic per-contrast generators",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Bootstrap CI confidence level (default 0.95)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parameters = RoleContrastParameters(
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )

    dataset = ValidatedBehaviorExportDataset.open(
        args.export_root,
        args.run_id,
        validate=True,
        full_part_hashes=False,
    )
    frames = load_role_contrast_frames(dataset)
    results = compute_role_contrast_rows(frames, parameters=parameters)
    output_dir = write_role_contrasts(
        Path(args.output_dir),
        results,
        source_export_run_id=dataset.export_run_id,
        source_export_manifest_sha256=dataset.cache_identity,
        source_export_root=str(dataset.root),
        parameters=parameters,
        overwrite=bool(args.overwrite),
    )

    print(f"Wrote {results.height} contrast rows to {output_dir}")
    print(f"Source export: {dataset.export_run_id}")
    print(f"Source manifest sha256: {dataset.cache_identity}")
    print()
    print(f"PRIMARY role contrast (aggressive - inert) within {PRIMARY_EPOCH_ROLE}:")
    print(format_contrast_table(results, epoch_role=PRIMARY_EPOCH_ROLE))
    for epoch_role in SECONDARY_EPOCH_ROLES:
        print()
        print(
            f"SECONDARY role contrast within {epoch_role} "
            "(park_position_asymmetry=true; chasers park at different corners):"
        )
        print(format_contrast_table(results, epoch_role=epoch_role))
    return 0


if __name__ == "__main__":
    sys.exit(main())
