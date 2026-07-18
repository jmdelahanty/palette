"""Run generic chaser quadrant occupancy components across recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional, Sequence

from fisheye.analysis.chaser_quadrant_occupancy import (
    DEFAULT_COMPONENT_NAME,
    build_chaser_quadrant_occupancy_result,
    write_chaser_quadrant_occupancy_component,
)
from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import json_attr_safe


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


def _explicit_zarr_targets(paths: Sequence[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        archive = path.expanduser().resolve()
        recording_id = archive.name.removesuffix(".zarr").removesuffix("_analysis")
        out.append(
            {
                "recording_id": recording_id,
                "zarr_path": str(archive),
                "coverage_percent": None,
                "detect_run": None,
                "model_name": None,
                "refined_run": None,
            }
        )
    return out


def run_for_targets(
    targets: Sequence[dict[str, Any]],
    *,
    chaser_distance_run: str,
    component_name: str,
    dropout_warning_fraction: float,
    dropout_exclusion_fraction: float | None,
    static_object_drift_warning_mm: float,
    apply: bool,
    overwrite: bool,
    no_png: bool,
    no_interactive_spec: bool,
    no_run_level_interactive_spec: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target in targets:
        zarr_path = Path(str(target["zarr_path"]))
        applied_path = None
        status = "dry_run"
        error = None
        summary: dict[str, Any] | None = None
        chaser_run_name = None
        try:
            result = build_chaser_quadrant_occupancy_result(
                zarr_path,
                chaser_distance_run=chaser_distance_run,
                component_name=component_name,
                dropout_warning_fraction=dropout_warning_fraction,
                dropout_exclusion_fraction=dropout_exclusion_fraction,
                static_object_drift_warning_mm=static_object_drift_warning_mm,
            )
            chaser_run_name = result.chaser_distance_run_name
            summary = result.summary
            status = result.endpoint_status
            if apply:
                applied_path = write_chaser_quadrant_occupancy_component(
                    zarr_path,
                    result,
                    overwrite=overwrite,
                    write_png=not no_png,
                    write_interactive_spec=not no_interactive_spec,
                    mirror_run_level_interactive_spec=not no_run_level_interactive_spec,
                )
        except Exception as exc:
            status = "failed"
            error = str(exc)
        results.append(
            {
                "recording_id": target.get("recording_id"),
                "zarr_path": str(zarr_path),
                "detect_coverage_percent": target.get("coverage_percent"),
                "detect_run": target.get("detect_run"),
                "refined_run": target.get("refined_run"),
                "chaser_distance_run": chaser_run_name or chaser_distance_run,
                "component_name": component_name,
                "chaser_quadrant_occupancy_path": applied_path,
                "status": status,
                "error": error,
                "summary": summary,
            }
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=None, help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config.")
    parser.add_argument("--zarr", action="append", type=Path, help="Explicit analysis zarr path. May be repeated.")
    parser.add_argument(
        "--recording-like",
        default="%chaser%",
        help="SQL LIKE filter for registry-selected target recordings.",
    )
    parser.add_argument("--coverage-min", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=12, help="Limit registry-selected targets.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--dropout-warning-fraction", type=float, default=0.20)
    parser.add_argument("--dropout-exclusion-fraction", type=float)
    parser.add_argument("--static-object-drift-warning-mm", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-interactive-spec", action="store_true")
    parser.add_argument("--no-run-level-interactive-spec", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.zarr:
        targets = _explicit_zarr_targets(args.zarr)
        registry = None
    else:
        registry = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
        targets = _query_targets(
            registry,
            recording_like=str(args.recording_like),
            coverage_min=float(args.coverage_min),
            limit=args.limit,
        )
    results = run_for_targets(
        targets,
        chaser_distance_run=str(args.chaser_distance_run),
        component_name=str(args.component_name),
        dropout_warning_fraction=float(args.dropout_warning_fraction),
        dropout_exclusion_fraction=args.dropout_exclusion_fraction,
        static_object_drift_warning_mm=float(args.static_object_drift_warning_mm),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        no_png=bool(args.no_png),
        no_interactive_spec=bool(args.no_interactive_spec),
        no_run_level_interactive_spec=bool(args.no_run_level_interactive_spec),
    )
    payload = {
        "apply": bool(args.apply),
        "registry": str(registry) if registry is not None else None,
        "recording_like": None if args.zarr else str(args.recording_like),
        "coverage_min": None if args.zarr else float(args.coverage_min),
        "limit": None if args.zarr else args.limit,
        "target_count": len(targets),
        "results": results,
    }
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"target_count: {len(targets)}")
        for row in results:
            status = row["status"]
            path_or_run = row["chaser_quadrant_occupancy_path"] or row["component_name"]
            print(f"{row['recording_id']}: status={status} component={path_or_run}")
            if row.get("error"):
                print(f"  error: {row['error']}")
            summary = row.get("summary")
            if isinstance(summary, dict):
                print(
                    "  delta_agg={:.3f} delta_occ_agg={:.3f} specificity_distance={:.3f}".format(
                        float(summary.get("delta_agg") or 0.0),
                        float(summary.get("delta_occ_agg") or 0.0),
                        float(summary.get("specificity_distance") or 0.0),
                    )
                )
        if not args.apply:
            print("dry_run: pass --apply to write chaser quadrant occupancy components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
