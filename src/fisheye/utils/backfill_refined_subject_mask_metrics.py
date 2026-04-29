#!/usr/bin/env python3
"""Refresh refined subject-mask component metrics and generated QC reason tags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from fisheye.refinement.finalize_subject_masks import (
    _EXECUTION_BACKENDS,
    _METRIC_LEVELS,
    _SERIAL_EXECUTION_BACKEND,
    _SUPPORTED_SCHEDULERS,
    _normalize_scheduler,
    refresh_refined_subject_mask_metrics,
)


def _parse_components(values: Optional[Sequence[Sequence[str]]], single_values: Optional[Sequence[str]]) -> list[str] | None:
    merged: list[str] = []
    for group in values or ():
        merged.extend(str(value) for value in group)
    for value in single_values or ():
        merged.append(str(value))
    return merged or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette zarr archive.")
    parser.add_argument(
        "--refined-run",
        help="Target refined_subject_masks_runs/<run>. Defaults to latest.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        action="append",
        help="Optional refined components to refresh.",
    )
    parser.add_argument(
        "--component",
        action="append",
        dest="component_values",
        help="Optional single refined component selector. Repeat to add more components.",
    )
    parser.add_argument(
        "--metric-level",
        choices=_METRIC_LEVELS,
        default="cheap",
        help="Metric depth to compute. cheap writes topology; full also writes slower shape-QC metrics.",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of ROI rows per metric chunk.")
    parser.add_argument(
        "--no-refresh-reason-tags",
        action="store_true",
        help="Only recompute numeric metrics; do not refresh generated needs_review_metric_* reason tags.",
    )
    parser.add_argument(
        "--write-eye-geometry",
        action="store_true",
        help="Also refresh refined-subject eye ellipse/eye-pair geometry when both eye components are selected.",
    )
    parser.add_argument(
        "--write-component-contours",
        action="store_true",
        help="Also refresh body/swim component contour caches for selected components.",
    )
    parser.add_argument(
        "--execution-backend",
        choices=_EXECUTION_BACKENDS,
        default=_SERIAL_EXECUTION_BACKEND,
        help="Execution backend. Use dask_worker_chunks to let Dask workers write disjoint metric chunks.",
    )
    parser.add_argument(
        "--scheduler",
        type=_normalize_scheduler,
        choices=_SUPPORTED_SCHEDULERS,
        default="single-threaded",
        help=(
            "Dask scheduler used when --execution-backend=dask_worker_chunks; recorded as instrumentation "
            "for serial_driver."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Worker-count setting for Dask-backed metric refresh.",
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args(argv)

    summary = refresh_refined_subject_mask_metrics(
        args.zarr_path,
        refined_run=args.refined_run,
        components=_parse_components(args.components, args.component_values),
        metric_level=args.metric_level,
        chunk_size=int(args.chunk_size),
        refresh_reason_tags=not bool(args.no_refresh_reason_tags),
        write_eye_geometry=bool(args.write_eye_geometry),
        write_component_contours=bool(args.write_component_contours),
        execution_backend=args.execution_backend,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
