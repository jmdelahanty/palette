"""Compute strategy-state analysis from one validated-behavior cohort export.

Cross-branch derived products (twin-excess corner nulls, IBI cell statistics)
are consumed as explicit input Parquet paths with fail-closed column
validation — never as code imports from unmerged branches.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fisheye.group_statistics.validated_behavior_strategy_states import (
    StrategyStatesConfig,
    compute_strategy_states,
    gather_export_inputs,
    load_ibi_cell_features,
    load_twin_excess_features,
    sha256_file,
    write_strategy_states_outputs,
)


def _parse_window_range(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--post-window-range must look like '<low>-<high>', e.g. '0-2'"
        )
    low, high = int(parts[0]), int(parts[1])
    if low < 0 or high < low:
        raise argparse.ArgumentTypeError(
            "--post-window-range must satisfy 0 <= low <= high"
        )
    return low, high


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root",
        type=Path,
        required=True,
        help="Validated-behavior export publication root.",
    )
    parser.add_argument(
        "--export-run-id",
        required=True,
        help="Exact validated-behavior export run id (never 'latest').",
    )
    parser.add_argument(
        "--twin-excess-parquet",
        type=Path,
        required=True,
        help=(
            "Twin-excess corner-null parquet (derived product). Required "
            "columns: recording_id, provider_role, epoch_role, behavior_role, "
            "near_zone_fraction_valid_excess, distance_p50_mm_excess."
        ),
    )
    parser.add_argument(
        "--ibi-cells-parquet",
        type=Path,
        required=True,
        help=(
            "IBI cell-statistics parquet (derived product). Required columns: "
            "recording_id, epoch_role, censoring, cell_valid, frac_gt_2s."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs in --output-dir.",
    )
    parser.add_argument(
        "--post-window-range",
        type=_parse_window_range,
        default=None,
        help=(
            "Optional inclusive post-epoch window-index range for a second "
            "decoder run, e.g. '0-2' for post minutes 1-3."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=20260901)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-ari-refits", type=int, default=100)
    parser.add_argument("--gmm-n-init", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = StrategyStatesConfig(
        random_seed=args.random_seed,
        permutation_iterations=args.permutation_iterations,
        bootstrap_ari_refits=args.bootstrap_ari_refits,
        gmm_n_init=args.gmm_n_init,
        k_max=args.k_max,
    )

    from fisheye.analytics_exports.validated_behavior_dataset import (
        ValidatedBehaviorExportDataset,
    )

    dataset = ValidatedBehaviorExportDataset.open(
        args.export_root,
        args.export_run_id,
        validate=True,
        full_part_hashes=False,
    )
    twin = load_twin_excess_features(args.twin_excess_parquet)
    ibi = load_ibi_cell_features(args.ibi_cells_parquet)
    export_inputs = gather_export_inputs(dataset, config)

    outputs = compute_strategy_states(
        twin=twin,
        ibi=ibi,
        export_inputs=export_inputs,
        config=config,
        post_window_range=args.post_window_range,
    )
    manifest_path = write_strategy_states_outputs(
        outputs,
        output_dir=args.output_dir,
        config=config,
        source_export={
            "export_root": str(Path(args.export_root).resolve()),
            "export_run_id": args.export_run_id,
            "export_manifest_record_sha256": dataset.cache_identity,
        },
        input_parquets={
            "twin_excess": {
                "path": str(Path(args.twin_excess_parquet).resolve()),
                "sha256": sha256_file(args.twin_excess_parquet),
            },
            "ibi_cells": {
                "path": str(Path(args.ibi_cells_parquet).resolve()),
                "sha256": sha256_file(args.ibi_cells_parquet),
            },
        },
        overwrite=args.overwrite,
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
