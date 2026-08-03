"""Run chaser near-field occupancy components across recordings."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Optional, Sequence

from fisheye.analysis.chaser_component_publication import (
    build_chaser_component_handle,
    load_chaser_component_handle_json,
)
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.analysis.chaser_near_field_occupancy import (
    DEFAULT_CDF_THRESHOLDS_MM,
    DEFAULT_COMPONENT_NAME,
    DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S,
    DEFAULT_PERCENTILES,
    DEFAULT_PERIMETER_BAND_MM,
    DEFAULT_R_IN_MM,
    DEFAULT_R_OUT_MM,
    DEFAULT_R_ZONE_MM,
    build_chaser_near_field_occupancy_result,
    write_chaser_near_field_occupancy_component,
)
from fisheye.analysis.chaser_quadrant_occupancy import (
    COMPONENT_PARENT_NAME as QUADRANT_OCCUPANCY_PARENT,
)
from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr_io import open_zarr_root


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


def _build_explicit_quadrant_dependency_handle(
    zarr_path: Path,
    *,
    chaser_distance_run: str,
    quadrant_occupancy_component: str,
) -> Mapping[str, Any]:
    component_name = str(quadrant_occupancy_component).strip()
    if not component_name or component_name == "latest":
        raise ValueError(
            "Maintained near-field batch execution requires an exact "
            "--quadrant-occupancy-component name or a dependency-handle JSON; "
            "latest discovery requires the explicit legacy compatibility flag."
        )
    root = open_zarr_root(zarr_path, mode="r")
    snapshot = load_chaser_distance_run(
        root,
        run_name=str(chaser_distance_run).strip() or "latest",
    )
    relative_path = f"{QUADRANT_OCCUPANCY_PARENT}/{component_name}"
    component_path = f"{snapshot.run_path}/{relative_path}"
    try:
        component = root[component_path]
    except Exception as exc:
        raise ValueError(
            f"Exact quadrant-occupancy component {component_path!r} is unavailable."
        ) from exc
    return build_chaser_component_handle(
        component,
        snapshot=snapshot,
        relative_path=relative_path,
    )


def run_for_targets(
    targets: Sequence[dict[str, Any]],
    *,
    chaser_distance_run: str,
    quadrant_occupancy_component: str,
    quadrant_occupancy_dependency_handle: Mapping[str, Any] | None = None,
    legacy_quadrant_occupancy_component_compatibility: bool = False,
    component_name: str,
    r_zone_mm: float,
    r_in_mm: float,
    r_out_mm: float,
    percentile_values: Sequence[float],
    radial_bin_edges_mm: Sequence[float] | None,
    cdf_thresholds_mm: Sequence[float],
    perimeter_band_mm: float,
    immobility_speed_threshold_mm_s: float,
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
        source_quadrant_component = None
        source_quadrant_manifest_sha256 = None
        geometry_status = None
        try:
            dependency_handle = quadrant_occupancy_dependency_handle
            if dependency_handle is None and not (
                legacy_quadrant_occupancy_component_compatibility
            ):
                dependency_handle = _build_explicit_quadrant_dependency_handle(
                    zarr_path,
                    chaser_distance_run=chaser_distance_run,
                    quadrant_occupancy_component=quadrant_occupancy_component,
                )
            result = build_chaser_near_field_occupancy_result(
                zarr_path,
                chaser_distance_run=chaser_distance_run,
                quadrant_occupancy_component=quadrant_occupancy_component,
                quadrant_occupancy_dependency_handle=dependency_handle,
                legacy_quadrant_occupancy_component_compatibility=(
                    legacy_quadrant_occupancy_component_compatibility
                ),
                component_name=component_name,
                r_zone_mm=float(r_zone_mm),
                r_in_mm=float(r_in_mm),
                r_out_mm=float(r_out_mm),
                percentile_values=percentile_values,
                radial_bin_edges_mm=radial_bin_edges_mm,
                cdf_thresholds_mm=cdf_thresholds_mm,
                perimeter_band_mm=float(perimeter_band_mm),
                immobility_speed_threshold_mm_s=float(immobility_speed_threshold_mm_s),
            )
            chaser_run_name = result.chaser_distance_run_name
            source_quadrant_component = result.source_quadrant_occupancy_component
            source_quadrant_manifest_sha256 = (
                result.source_quadrant_occupancy_manifest_sha256
            )
            geometry_status = result.geometry_status
            summary = result.summary
            status = result.endpoint_status
            if apply:
                applied_path = write_chaser_near_field_occupancy_component(
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
                "quadrant_occupancy_component": source_quadrant_component
                or quadrant_occupancy_component,
                "quadrant_occupancy_manifest_sha256": (
                    source_quadrant_manifest_sha256
                ),
                "component_name": component_name,
                "chaser_near_field_occupancy_path": applied_path,
                "geometry_status": geometry_status,
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
        default="%chaser%",
        help="SQL LIKE filter for registry-selected target recordings.",
    )
    parser.add_argument("--coverage-min", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=12, help="Limit registry-selected targets.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--quadrant-occupancy-component", default="latest")
    parser.add_argument(
        "--quadrant-occupancy-dependency-handle-json",
        type=Path,
        help=(
            "Strict JSON handle for one exact quadrant-occupancy candidate. "
            "Without this option, provide an exact component name."
        ),
    )
    parser.add_argument(
        "--legacy-quadrant-occupancy-component-compatibility",
        action="store_true",
        help=(
            "Explicitly permit historical component name/latest discovery; "
            "exact manifest-digested dependencies remain the default."
        ),
    )
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--r-zone-mm", type=float, default=DEFAULT_R_ZONE_MM)
    parser.add_argument("--r-in-mm", type=float, default=DEFAULT_R_IN_MM)
    parser.add_argument("--r-out-mm", type=float, default=DEFAULT_R_OUT_MM)
    parser.add_argument("--percentiles", default=",".join(f"{value:g}" for value in DEFAULT_PERCENTILES))
    parser.add_argument("--radial-bin-edges-mm", default=None, help="Comma-separated edges. Defaults to data-driven bins.")
    parser.add_argument("--cdf-thresholds-mm", default=",".join(f"{value:g}" for value in DEFAULT_CDF_THRESHOLDS_MM))
    parser.add_argument("--perimeter-band-mm", type=float, default=DEFAULT_PERIMETER_BAND_MM)
    parser.add_argument("--immobility-speed-threshold-mm-s", type=float, default=DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S)
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
    percentiles = _parse_float_list(str(args.percentiles)) or list(DEFAULT_PERCENTILES)
    radial_edges = _parse_float_list(args.radial_bin_edges_mm)
    cdf_thresholds = _parse_float_list(str(args.cdf_thresholds_mm)) or list(DEFAULT_CDF_THRESHOLDS_MM)
    quadrant_dependency_handle = (
        load_chaser_component_handle_json(
            args.quadrant_occupancy_dependency_handle_json
        )
        if args.quadrant_occupancy_dependency_handle_json is not None
        else None
    )
    results = run_for_targets(
        targets,
        chaser_distance_run=str(args.chaser_distance_run),
        quadrant_occupancy_component=str(args.quadrant_occupancy_component),
        quadrant_occupancy_dependency_handle=quadrant_dependency_handle,
        legacy_quadrant_occupancy_component_compatibility=bool(
            args.legacy_quadrant_occupancy_component_compatibility
        ),
        component_name=str(args.component_name),
        r_zone_mm=float(args.r_zone_mm),
        r_in_mm=float(args.r_in_mm),
        r_out_mm=float(args.r_out_mm),
        percentile_values=percentiles,
        radial_bin_edges_mm=radial_edges,
        cdf_thresholds_mm=cdf_thresholds,
        perimeter_band_mm=float(args.perimeter_band_mm),
        immobility_speed_threshold_mm_s=float(args.immobility_speed_threshold_mm_s),
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
            "quadrant_dependency_mode": (
                "explicit_handle_json"
                if quadrant_dependency_handle is not None
                else (
                    "legacy_component_discovery"
                    if args.legacy_quadrant_occupancy_component_compatibility
                    else "exact_named_component_handle"
                )
            ),
            "r_zone_mm": float(args.r_zone_mm),
            "r_in_mm": float(args.r_in_mm),
            "r_out_mm": float(args.r_out_mm),
            "percentile_values": percentiles,
            "radial_bin_edges_mm": radial_edges,
            "cdf_thresholds_mm": cdf_thresholds,
            "perimeter_band_mm": float(args.perimeter_band_mm),
            "immobility_speed_threshold_mm_s": float(args.immobility_speed_threshold_mm_s),
        },
        "results": results,
    }
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"target_count: {len(targets)}")
        for row in results:
            status = row["status"]
            path_or_component = (
                row["chaser_near_field_occupancy_path"] or row["component_name"]
            )
            print(
                f"{row['recording_id']}: status={status} "
                f"geometry={row.get('geometry_status') or '-'} component={path_or_component}"
            )
            if row.get("error"):
                print(f"  error: {row['error']}")
            summary = row.get("summary")
            if isinstance(summary, dict):
                print(
                    "  approach_p05_delta_agg={:.3f} nearzone_occ_delta_agg={:.6f} "
                    "nearzone_entry_rate_delta_agg={:.3f}".format(
                        float(summary.get("approach_p05_delta_agg") or 0.0),
                        float(summary.get("nearzone_occ_delta_agg") or 0.0),
                        float(summary.get("nearzone_entry_rate_delta_agg") or 0.0),
                    )
                )
        if not args.apply:
            print(
                "dry_run: pass --apply to write Chaser near-field occupancy components"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
