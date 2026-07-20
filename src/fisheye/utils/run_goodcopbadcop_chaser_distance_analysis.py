"""Run GoodCopBadCop offline fish-to-chaser distance analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional, Sequence

import numpy as np

from fisheye.analysis.chaser_distance_runs import (
    build_chaser_distance_result,
    write_chaser_distance_run,
)
from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import json_attr_safe


DEFAULT_RUN = "goodcopbadcop_chaser_distance_v1_20260617"


def _query_targets(
    registry: Path,
    *,
    recording_like: str,
    coverage_min: float,
    limit: Optional[int],
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            dpl.recording_id,
            dcc.zarr_path,
            dpl.coverage_percent,
            dpl.detect_run,
            dpl.model_name,
            dq.refined_run
        FROM detect_performance_latest dpl
        JOIN dataset_context_current dcc ON dcc.dataset_id = dpl.dataset_id
        LEFT JOIN refined_detect_review_current dq ON dq.dataset_id = dpl.dataset_id
        WHERE dpl.recording_id LIKE ?
          AND dpl.coverage_percent >= ?
          AND dcc.zarr_use = 'analysis'
          AND dcc.dataset_status = 'active'
        ORDER BY dpl.coverage_percent DESC, dpl.recording_id
    """
    params: list[Any] = [recording_like, float(coverage_min)]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    conn = sqlite3.connect(str(registry))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def run_for_targets(
    targets: Sequence[dict[str, Any]],
    *,
    run_name: str,
    threshold_mm: float,
    distribution_bin_width_mm: float,
    apply: bool,
    overwrite: bool,
    no_png: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target in targets:
        zarr_path = Path(str(target["zarr_path"]))
        result = build_chaser_distance_result(
            zarr_path,
            run_name=run_name,
            threshold_mm=threshold_mm,
            distribution_bin_width_mm=distribution_bin_width_mm,
        )
        applied_path = None
        if apply:
            applied_path = write_chaser_distance_run(
                zarr_path,
                result,
                overwrite=overwrite,
                write_png=not no_png,
            )
        results.append(
            {
                "recording_id": target.get("recording_id"),
                "zarr_path": str(zarr_path),
                "detect_coverage_percent": target.get("coverage_percent"),
                "detect_run": target.get("detect_run"),
                "refined_run": target.get("refined_run"),
                "chaser_distance_run": run_name,
                "distribution_bin_width_mm": float(distribution_bin_width_mm),
                "chaser_distance_path": applied_path,
                "source_detection_path": result.source_detection_path,
                "source_stimulus_run": result.source_stimulus_run,
                "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
                "chaser_indices": result.chaser_indices.astype(int).tolist(),
                "fish_valid_frame_count": int(result.fish_valid.sum()),
                "distance_valid_frame_count": np.sum(np.isfinite(result.distance_mm), axis=0).astype(int).tolist(),
                "windows": [
                    {
                        "label": window.label,
                        "start_frame": window.start_frame,
                        "end_frame": window.end_frame,
                        "p50_distance_mm": result.epoch_p50_distance_mm[i].tolist(),
                        "mean_distance_mm": result.epoch_mean_distance_mm[i].tolist(),
                        "valid_frame_count": result.epoch_valid_frame_count[i].astype(int).tolist(),
                    }
                    for i, window in enumerate(result.windows)
                ],
            }
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=None, help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config.")
    parser.add_argument(
        "--recording-like",
        default="2026-06-14%GoodCopBadCop%",
        help="SQL LIKE filter for target recordings.",
    )
    parser.add_argument("--coverage-min", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=12, help="Limit to the top N recordings by detect coverage.")
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--threshold-mm", type=float, default=20.0)
    parser.add_argument("--distribution-bin-width-mm", type=float, default=2.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Deprecated compatibility option. This canonical batch command "
            "rejects replacement; use a new --run-name."
        ),
    )
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    targets = _query_targets(
        registry,
        recording_like=str(args.recording_like),
        coverage_min=float(args.coverage_min),
        limit=args.limit,
    )
    results = run_for_targets(
        targets,
        run_name=str(args.run_name),
        threshold_mm=float(args.threshold_mm),
        distribution_bin_width_mm=float(args.distribution_bin_width_mm),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        no_png=bool(args.no_png),
    )
    payload = {
        "apply": bool(args.apply),
        "registry": str(registry),
        "recording_like": str(args.recording_like),
        "coverage_min": float(args.coverage_min),
        "limit": args.limit,
        "target_count": len(targets),
        "results": results,
    }
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"target_count: {len(targets)}")
        for row in results:
            print(
                f"{row['recording_id']}: detect_coverage={float(row['detect_coverage_percent']):.3f}% "
                f"run={row['chaser_distance_path'] or row['chaser_distance_run']}"
            )
            for window in row["windows"]:
                p50 = ", ".join(f"{float(v):.2f}" for v in window["p50_distance_mm"])
                print(f"  {window['label']}: p50_distance_mm=[{p50}]")
        if not args.apply:
            print("dry_run: pass --apply to write zarr analysis runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
