"""Run GoodCopBadCop epoch behavior summaries across recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from fisheye.analysis.goodcopbadcop_epoch_behavior_summary import (
    DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM,
    DEFAULT_COMPONENT_NAME,
    DEFAULT_WALL_BAND_MM,
    REQUIRED_TRACK_SCOPE,
    build_goodcopbadcop_epoch_behavior_summary_result,
    write_goodcopbadcop_epoch_behavior_summary_component,
)
from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import json_attr_safe
from fisheye.utils.run_goodcopbadcop_cra_near_field import (
    _explicit_zarr_targets,
    _filesystem_targets,
    _query_targets,
)


def run_for_targets(
    targets: Sequence[dict[str, Any]],
    *,
    chaser_distance_run: str,
    component_name: str,
    swim_bout_run: str | None,
    track_kinematics_run: str | None,
    track_kinematics_scope: str,
    track_id: int | None,
    speed_level: str | None,
    center_distance_bin_width_mm: float = DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM,
    wall_band_mm: float = DEFAULT_WALL_BAND_MM,
    apply: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target in targets:
        zarr_path = Path(str(target["zarr_path"]))
        applied_path = None
        status = "dry_run"
        error = None
        summary: dict[str, Any] | None = None
        chaser_run_name = None
        source_swim_bout_run = None
        source_track_kinematics_run = None
        try:
            result = build_goodcopbadcop_epoch_behavior_summary_result(
                zarr_path,
                chaser_distance_run=chaser_distance_run,
                component_name=component_name,
                swim_bout_run=swim_bout_run,
                track_kinematics_run=track_kinematics_run,
                track_kinematics_scope=track_kinematics_scope,
                track_id=track_id,
                speed_level=speed_level,
                center_distance_bin_width_mm=float(center_distance_bin_width_mm),
                wall_band_mm=float(wall_band_mm),
            )
            chaser_run_name = result.chaser_distance_run_name
            source_swim_bout_run = result.source_swim_bout_run
            source_track_kinematics_run = result.source_track_kinematics_run
            summary = {
                "epoch_count": int(result.per_epoch_fish.shape[0]),
                "chaser_epoch_count": int(result.per_epoch_chaser.shape[0]),
                "per_epoch_bout_count": int(result.per_epoch_bouts.shape[0]),
                "per_epoch_bout_histogram_count": int(
                    getattr(result, "per_epoch_bout_histograms", []).shape[0]
                    if hasattr(getattr(result, "per_epoch_bout_histograms", None), "shape")
                    else 0
                ),
                "per_epoch_inter_bout_interval_histogram_count": int(
                    getattr(result, "per_epoch_inter_bout_interval_histograms", []).shape[0]
                    if hasattr(getattr(result, "per_epoch_inter_bout_interval_histograms", None), "shape")
                    else 0
                ),
                "center_distance_histogram_count": int(result.center_distance_histogram.shape[0]),
                "bout_count": result.per_epoch_fish["bout_count"].astype(int).tolist(),
                "inter_bout_interval_count": result.per_epoch_fish[
                    "inter_bout_interval_count"
                ].astype(int).tolist(),
                "warnings": list(result.warnings),
            }
            if apply:
                applied_path = write_goodcopbadcop_epoch_behavior_summary_component(
                    zarr_path,
                    result,
                    overwrite=overwrite,
                )
                status = "complete"
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
                "source_swim_bout_run": source_swim_bout_run,
                "source_track_kinematics_run": source_track_kinematics_run,
                "component_name": component_name,
                "epoch_behavior_summary_path": applied_path,
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
        default="%GoodCopBadCop%",
        help="SQL LIKE filter for registry-selected targets; glob-like after --recordings-root.",
    )
    parser.add_argument("--coverage-min", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=12, help="Limit selected targets.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--swim-bout-run", default="latest")
    parser.add_argument("--track-kinematics-run")
    parser.add_argument("--track-kinematics-scope", default=REQUIRED_TRACK_SCOPE)
    parser.add_argument("--track-id", type=int)
    parser.add_argument("--speed-level")
    parser.add_argument("--center-distance-bin-width-mm", type=float, default=DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM)
    parser.add_argument("--wall-band-mm", type=float, default=DEFAULT_WALL_BAND_MM)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    registry = None
    if args.zarr:
        targets = _explicit_zarr_targets(args.zarr)
    elif args.recordings_root:
        targets = _filesystem_targets(
            args.recordings_root,
            recording_like=str(args.recording_like),
            limit=args.limit,
            recursive=bool(args.recursive_recordings_root),
        )
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
        swim_bout_run=args.swim_bout_run,
        track_kinematics_run=args.track_kinematics_run,
        track_kinematics_scope=str(args.track_kinematics_scope),
        track_id=args.track_id,
        speed_level=args.speed_level,
        center_distance_bin_width_mm=float(args.center_distance_bin_width_mm),
        wall_band_mm=float(args.wall_band_mm),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
    )
    payload = {
        "apply": bool(args.apply),
        "registry": str(registry) if registry is not None else None,
        "recording_like": str(args.recording_like),
        "coverage_min": None if args.zarr or args.recordings_root else float(args.coverage_min),
        "limit": args.limit,
        "target_count": len(targets),
        "results": results,
    }
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"target_count: {len(targets)}")
        for row in results:
            status = row["status"]
            path_or_component = row["epoch_behavior_summary_path"] or row["component_name"]
            print(f"{row['recording_id']}: status={status} component={path_or_component}")
            if row.get("error"):
                print(f"  error: {row['error']}")
            summary = row.get("summary")
            if isinstance(summary, dict):
                print(
                    "  epochs={epoch_count} chaser_epochs={chaser_epoch_count} warnings={warnings}".format(
                        epoch_count=summary.get("epoch_count"),
                        chaser_epoch_count=summary.get("chaser_epoch_count"),
                        warnings=len(summary.get("warnings") or []),
                    )
                )
        if not args.apply:
            print("dry_run: pass --apply to write epoch behavior summary components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
