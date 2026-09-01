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
import polars as pl

from fisheye.group_statistics.validated_behavior_role_contrasts import (
    BOUT_ASSOCIATION_PARQUET_NAME,
    BOUT_ASSOCIATION_SPEC_VERSION,
    DEFAULT_MIN_ASSOCIATION_BOUTS,
    DEFAULT_MIN_BOUTS_PER_CELL,
    DEFAULT_MIN_IBIS_PER_CELL,
    DEFAULT_MIN_PAIRED,
    DISTANCE_BIN_PARQUET_NAME,
    DISTANCE_BIN_SPEC_VERSION,
    IBI_CELLS_PARQUET_NAME,
    IBI_SHAPE_PARQUET_NAME,
    IBI_SHAPE_SPEC_VERSION,
    PRIMARY_EPOCH_ROLE,
    QUANTILE_SHAPE_PARQUET_NAME,
    QUANTILE_SHAPE_SPEC_VERSION,
    RoleContrastParameters,
    SECONDARY_EPOCH_ROLES,
    build_ibi_cells,
    compute_bout_association_contrast_rows,
    compute_distance_bin_contrast_rows,
    compute_ibi_shape_contrast_rows,
    compute_quantile_shape_contrast_rows,
    compute_role_contrast_rows,
    format_contrast_table,
    format_stats_table,
    load_bout_association_frame,
    load_distance_bin_frame,
    load_ibi_inputs,
    load_quantile_shape_frames,
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
    parser.add_argument(
        "--min-paired",
        type=int,
        default=DEFAULT_MIN_PAIRED,
        help=(
            "Minimum paired recordings for a distance-bin/association/quantile "
            "contrast; below this the row is emitted as status=skipped "
            f"(default {DEFAULT_MIN_PAIRED})"
        ),
    )
    parser.add_argument(
        "--min-association-bouts",
        type=int,
        default=DEFAULT_MIN_ASSOCIATION_BOUTS,
        help=(
            "Minimum qualifying bouts per recording x role cell in the "
            f"bout-association contrasts (default {DEFAULT_MIN_ASSOCIATION_BOUTS})"
        ),
    )
    parser.add_argument(
        "--min-bouts-per-cell",
        type=int,
        default=DEFAULT_MIN_BOUTS_PER_CELL,
        help=(
            "Minimum finite bout values per recording x epoch cell in the "
            f"quantile shape contrasts (default {DEFAULT_MIN_BOUTS_PER_CELL})"
        ),
    )
    parser.add_argument(
        "--min-ibis-per-cell",
        type=int,
        default=DEFAULT_MIN_IBIS_PER_CELL,
        help=(
            "Minimum usable inter-bout intervals per recording x epoch x "
            f"censoring cell (default {DEFAULT_MIN_IBIS_PER_CELL})"
        ),
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
    distance_bins = compute_distance_bin_contrast_rows(
        load_distance_bin_frame(dataset),
        parameters=parameters,
        min_paired=args.min_paired,
    )
    associations = compute_bout_association_contrast_rows(
        load_bout_association_frame(dataset),
        parameters=parameters,
        min_bouts=args.min_association_bouts,
        min_paired=args.min_paired,
    )
    bouts, epochs = load_quantile_shape_frames(dataset)
    quantile_shape = compute_quantile_shape_contrast_rows(
        bouts,
        epochs,
        parameters=parameters,
        min_bouts_per_cell=args.min_bouts_per_cell,
        min_paired=args.min_paired,
    )
    ibi_bouts, ibi_epochs, fps_by_recording, valid_frames = load_ibi_inputs(dataset)
    ibi_cells = build_ibi_cells(
        ibi_bouts,
        ibi_epochs,
        fps_by_recording=fps_by_recording,
        valid_frames_by_recording=valid_frames,
        min_ibis_per_cell=args.min_ibis_per_cell,
    )
    ibi_contrasts = compute_ibi_shape_contrast_rows(
        ibi_cells,
        parameters=parameters,
        min_paired=args.min_paired,
    )
    output_dir = write_role_contrasts(
        Path(args.output_dir),
        results,
        source_export_run_id=dataset.export_run_id,
        source_export_manifest_sha256=dataset.cache_identity,
        source_export_root=str(dataset.root),
        parameters=parameters,
        overwrite=bool(args.overwrite),
        extra_tables={
            DISTANCE_BIN_PARQUET_NAME: (distance_bins, DISTANCE_BIN_SPEC_VERSION),
            BOUT_ASSOCIATION_PARQUET_NAME: (
                associations,
                BOUT_ASSOCIATION_SPEC_VERSION,
            ),
            QUANTILE_SHAPE_PARQUET_NAME: (
                quantile_shape,
                QUANTILE_SHAPE_SPEC_VERSION,
            ),
            IBI_SHAPE_PARQUET_NAME: (ibi_contrasts, IBI_SHAPE_SPEC_VERSION),
            IBI_CELLS_PARQUET_NAME: (ibi_cells, IBI_SHAPE_SPEC_VERSION),
        },
        thresholds={
            "min_paired": args.min_paired,
            "min_association_bouts": args.min_association_bouts,
            "min_bouts_per_cell": args.min_bouts_per_cell,
            "min_ibis_per_cell": args.min_ibis_per_cell,
        },
    )

    total_rows = (
        results.height
        + distance_bins.height
        + associations.height
        + quantile_shape.height
        + ibi_contrasts.height
        + ibi_cells.height
    )
    print(f"Wrote {total_rows} contrast rows to {output_dir}")
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

    print()
    print(
        "DISTANCE-BIN bout-response role contrast (aggressive - inert) "
        f"within {PRIMARY_EPOCH_ROLE} "
        "(pre/post secondaries carry park_position_asymmetry=true in the parquet):"
    )
    print(
        format_stats_table(
            distance_bins.filter(
                pl.col("epoch_role") == PRIMARY_EPOCH_ROLE
            ).sort("metric", "distance_bin_index"),
            ["metric", "distance_bin_start_mm", "distance_bin_end_mm"],
        )
    )

    print()
    print(
        "BOUT-ASSOCIATION role contrast (aggressive - inert) "
        f"within {PRIMARY_EPOCH_ROLE}:"
    )
    print(format_stats_table(associations.sort("metric"), ["metric"]))

    print()
    print(
        "QUANTILE SHAPE contrasts (epoch_b - epoch_a per recording; "
        "no role axis; canonical swim bouts):"
    )
    print(
        format_stats_table(
            quantile_shape.sort("metric", "contrast", "quantile"),
            ["metric", "contrast", "quantile"],
        )
    )

    print()
    print(
        "IBI SHAPE contrasts (epoch_b - epoch_a per recording; "
        "censoring=valid_span_required is primary, censoring=none is the "
        "dropout-sensitivity comparison):"
    )
    print(
        format_stats_table(
            ibi_contrasts.sort("censoring", "metric", "contrast"),
            ["censoring", "metric", "contrast"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
