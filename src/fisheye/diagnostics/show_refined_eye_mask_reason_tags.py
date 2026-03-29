#!/usr/bin/env python3
"""Show reason-tag counts for a refined eye-mask run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import zarr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print reason tag counts for a refined eye-mask run."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the analysis zarr.")
    parser.add_argument(
        "run_name", help="Name of the run under refined_eye_masks_runs."
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print headline run attrs before the reason-tag JSON payload.",
    )
    return parser


def show_refined_eye_mask_reason_tags(
    zarr_path: Path,
    run_name: str,
    *,
    show_summary: bool = False,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    refined_parent = root.get("refined_eye_masks_runs")
    if refined_parent is None:
        raise ValueError(f"No refined_eye_masks_runs group found in {zarr_path}.")
    if run_name not in refined_parent:
        raise ValueError(
            f"Run '{run_name}' not found under refined_eye_masks_runs in {zarr_path}."
        )

    run = refined_parent[run_name]
    metrics_summary = run.attrs.get("metrics_summary", {})
    reason_tag_counts = metrics_summary.get("reason_tag_counts", {})

    if show_summary:
        print(f"run_name: {run_name}")
        print(f"successful_roi_pairs: {run.attrs.get('successful_roi_pairs')}")
        print(f"successful_roi_pair_rate: {run.attrs.get('successful_roi_pair_rate')}")
        print(
            f"mask_probability_threshold: {run.attrs.get('mask_probability_threshold')}"
        )
        print(
            "mask_probability_threshold_source: "
            f"{run.attrs.get('mask_probability_threshold_source')}"
        )
        print(f"success_min_eye_area_px: {run.attrs.get('success_min_eye_area_px')}")
        print()

    print(json.dumps(reason_tag_counts, indent=2, sort_keys=True))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    show_refined_eye_mask_reason_tags(
        args.zarr_path,
        args.run_name,
        show_summary=bool(args.show_summary),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
