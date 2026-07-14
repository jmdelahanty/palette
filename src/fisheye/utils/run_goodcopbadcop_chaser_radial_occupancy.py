"""Run chaser-relative radial occupancy components across recordings."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional, Sequence

from fisheye.analysis.chaser_radial_occupancy import (
    DEFAULT_AREA_CACHE_STEP_MM,
    DEFAULT_CDF_THRESHOLDS_MM,
    DEFAULT_COMPONENT_NAME,
    DEFAULT_MIN_EXPECTED_COUNT,
    DEFAULT_MOTION_SPREAD_THRESHOLD_MM,
    DEFAULT_PERIMETER_BAND_MM,
    DEFAULT_RADIAL_BIN_WIDTH_MM,
    DEFAULT_R_ZONE_MM,
    build_chaser_radial_occupancy_result,
    write_chaser_radial_occupancy_component,
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


def _recording_like_to_glob(pattern: str) -> str:
    return str(pattern).replace("%", "*").replace("_", "?")


def _filesystem_targets(
    roots: Sequence[Path],
    *,
    recording_like: str,
    limit: Optional[int],
    recursive: bool,
) -> list[dict[str, Any]]:
    glob_pattern = _recording_like_to_glob(recording_like)
    targets: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        base = root.expanduser()
        candidates = base.rglob("*_analysis.zarr") if recursive else base.glob("*/zarr/*_analysis.zarr")
        for archive in sorted(candidates):
            resolved = archive.resolve()
            if resolved in seen:
                continue
            recording_id = archive.name.removesuffix(".zarr").removesuffix("_analysis")
            if not fnmatch.fnmatch(recording_id, glob_pattern):
                continue
            seen.add(resolved)
            targets.append(
                {
                    "recording_id": recording_id,
                    "zarr_path": str(resolved),
                    "coverage_percent": None,
                    "detect_run": None,
                    "model_name": None,
                    "refined_run": None,
                }
            )
            if limit is not None and len(targets) >= int(limit):
                return targets
    return targets


def _parse_float_list(value: str | Sequence[float] | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(item) for item in value]


def run_for_targets(
    targets: Sequence[dict[str, Any]],
    *,
    chaser_distance_run: str,
    component_name: str,
    radial_bin_width_mm: float,
    radial_bin_edges_mm: Sequence[float] | None,
    cdf_thresholds_mm: Sequence[float],
    r_zone_mm: float,
    perimeter_band_mm: float,
    motion_spread_threshold_mm: float,
    area_cache_step_mm: float,
    min_expected_count: float,
    settle_trim_s: float | None,
    apply: bool,
    overwrite: bool,
    no_png: bool,
    no_interactive_spec: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target in targets:
        zarr_path = Path(str(target["zarr_path"]))
        applied_path = None
        status = "dry_run"
        error = None
        summary: dict[str, Any] | None = None
        chaser_run_name = None
        geometry_status = None
        qc_warnings: list[str] = []
        try:
            result = build_chaser_radial_occupancy_result(
                zarr_path,
                chaser_distance_run=chaser_distance_run,
                component_name=component_name,
                radial_bin_width_mm=float(radial_bin_width_mm),
                radial_bin_edges_mm=radial_bin_edges_mm,
                cdf_thresholds_mm=cdf_thresholds_mm,
                r_zone_mm=float(r_zone_mm),
                perimeter_band_mm=float(perimeter_band_mm),
                motion_spread_threshold_mm=float(motion_spread_threshold_mm),
                area_cache_step_mm=float(area_cache_step_mm),
                min_expected_count=float(min_expected_count),
                settle_trim_s=settle_trim_s,
            )
            chaser_run_name = result.chaser_distance_run_name
            geometry_status = result.geometry.status
            summary = result.summary
            qc_warnings = list(result.qc_warnings)
            status = result.status
            if apply:
                applied_path = write_chaser_radial_occupancy_component(
                    zarr_path,
                    result,
                    overwrite=overwrite,
                    write_png=not no_png,
                    write_interactive_spec=not no_interactive_spec,
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
                "chaser_radial_occupancy_path": applied_path,
                "geometry_status": geometry_status,
                "status": status,
                "error": error,
                "qc_warnings": qc_warnings,
                "summary": summary,
            }
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=None, help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config.")
    parser.add_argument("--zarr", action="append", type=Path, help="Explicit analysis zarr path. May be repeated.")
    parser.add_argument(
        "--recordings-root",
        action="append",
        type=Path,
        help="Filesystem recording root to scan for <recording>/zarr/*_analysis.zarr. May be repeated.",
    )
    parser.add_argument(
        "--recursive-recordings-root",
        action="store_true",
        help="Use recursive *_analysis.zarr discovery under --recordings-root.",
    )
    parser.add_argument(
        "--recording-like",
        default="%GoodCopBadCop%",
        help="SQL LIKE filter for registry-selected target recordings.",
    )
    parser.add_argument("--coverage-min", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=12, help="Limit registry-selected targets.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--radial-bin-width-mm", type=float, default=DEFAULT_RADIAL_BIN_WIDTH_MM)
    parser.add_argument("--radial-bin-edges-mm", default=None, help="Comma-separated edges. Defaults to arena-spanning bins.")
    parser.add_argument("--cdf-thresholds-mm", default=",".join(f"{value:g}" for value in DEFAULT_CDF_THRESHOLDS_MM))
    parser.add_argument("--r-zone-mm", type=float, default=DEFAULT_R_ZONE_MM)
    parser.add_argument("--perimeter-band-mm", type=float, default=DEFAULT_PERIMETER_BAND_MM)
    parser.add_argument("--motion-spread-threshold-mm", type=float, default=DEFAULT_MOTION_SPREAD_THRESHOLD_MM)
    parser.add_argument("--area-cache-step-mm", type=float, default=DEFAULT_AREA_CACHE_STEP_MM)
    parser.add_argument("--min-expected-count", type=float, default=DEFAULT_MIN_EXPECTED_COUNT)
    parser.add_argument(
        "--settle-trim-s",
        type=float,
        default=None,
        help="Object-repositioning window trimmed from static-configuration epochs. "
             "Defaults to the protocol position_transition_duration_s.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-interactive-spec", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.zarr:
        targets = _explicit_zarr_targets(args.zarr)
        registry = None
        source = "explicit_zarr"
    elif args.recordings_root:
        registry = None
        targets = _filesystem_targets(
            args.recordings_root,
            recording_like=str(args.recording_like),
            limit=args.limit,
            recursive=bool(args.recursive_recordings_root),
        )
        source = "recordings_root"
    else:
        registry = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
        targets = _query_targets(
            registry,
            recording_like=str(args.recording_like),
            coverage_min=float(args.coverage_min),
            limit=args.limit,
        )
        source = "registry"
    radial_edges = _parse_float_list(args.radial_bin_edges_mm)
    cdf_thresholds = _parse_float_list(str(args.cdf_thresholds_mm)) or list(DEFAULT_CDF_THRESHOLDS_MM)
    results = run_for_targets(
        targets,
        chaser_distance_run=str(args.chaser_distance_run),
        component_name=str(args.component_name),
        radial_bin_width_mm=float(args.radial_bin_width_mm),
        radial_bin_edges_mm=radial_edges,
        cdf_thresholds_mm=cdf_thresholds,
        r_zone_mm=float(args.r_zone_mm),
        perimeter_band_mm=float(args.perimeter_band_mm),
        motion_spread_threshold_mm=float(args.motion_spread_threshold_mm),
        area_cache_step_mm=float(args.area_cache_step_mm),
        min_expected_count=float(args.min_expected_count),
        settle_trim_s=None if args.settle_trim_s is None else float(args.settle_trim_s),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        no_png=bool(args.no_png),
        no_interactive_spec=bool(args.no_interactive_spec),
    )
    payload = {
        "apply": bool(args.apply),
        "registry": str(registry) if registry is not None else None,
        "source": source,
        "recordings_root": [str(path) for path in args.recordings_root] if args.recordings_root else None,
        "recording_like": None if args.zarr else str(args.recording_like),
        "coverage_min": None if args.zarr or args.recordings_root else float(args.coverage_min),
        "limit": None if args.zarr else args.limit,
        "target_count": len(targets),
        "parameters": {
            "radial_bin_width_mm": float(args.radial_bin_width_mm),
            "radial_bin_edges_mm": radial_edges,
            "cdf_thresholds_mm": cdf_thresholds,
            "r_zone_mm": float(args.r_zone_mm),
            "perimeter_band_mm": float(args.perimeter_band_mm),
            "motion_spread_threshold_mm": float(args.motion_spread_threshold_mm),
            "area_cache_step_mm": float(args.area_cache_step_mm),
            "min_expected_count": float(args.min_expected_count),
            "settle_trim_s": args.settle_trim_s,
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"target_count: {len(targets)}")
        for row in results:
            path_or_component = row["chaser_radial_occupancy_path"] or row["component_name"]
            print(
                f"{row['recording_id']}: status={row['status']} "
                f"geometry={row.get('geometry_status') or '-'} component={path_or_component}"
            )
            if row.get("error"):
                print(f"  error: {row['error']}")
            summary = row.get("summary")
            if isinstance(summary, dict):
                for key, value in sorted(summary.items()):
                    if key.startswith("nearzone_enrichment_") and value is not None:
                        print(f"  {key[len('nearzone_enrichment_'):]}: near-zone enrichment {float(value):.2f}x")
            pursuing = [w for w in row.get("qc_warnings") or [] if w.startswith("closed_loop_null")]
            if pursuing:
                print(f"  note: chaser pursuing in {len(pursuing)} epoch/chaser pair(s); geometric null only")
        if not args.apply:
            print("dry_run: pass --apply to write chaser radial-occupancy components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
