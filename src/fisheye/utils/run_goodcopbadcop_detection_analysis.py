"""Run GoodCopBadCop stimulus-epoch and detection-occupancy analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional, Sequence

from fisheye.analysis.detection_occupancy_runs import (
    OccupancyWindow,
    build_detection_occupancy_result,
    write_detection_occupancy_run,
)
from fisheye.analysis.stimulus_epoch_runs import (
    build_stimulus_epoch_result,
    write_stimulus_epoch_run,
)
from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import json_attr_safe


DEFAULT_EPOCH_RUN = "goodcopbadcop_stimulus_epochs_v1_20260617"
DEFAULT_OCCUPANCY_RUN = "goodcopbadcop_detection_occupancy_v1_20260617"


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
    epoch_run_name: str,
    occupancy_run_name: str,
    bin_size: int,
    smooth_sigma: float,
    apply: bool,
    overwrite: bool,
    no_png: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target in targets:
        zarr_path = Path(str(target["zarr_path"]))
        epoch = build_stimulus_epoch_result(zarr_path, run_name=epoch_run_name)
        epoch_windows = tuple(
            OccupancyWindow(
                window_id=window.window_id,
                label=window.label,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                start_time_s=window.start_time_s,
                end_time_s=window.end_time_s,
                duration_s=window.duration_s,
            )
            for window in epoch.windows
        )
        epoch_path = None
        if apply:
            epoch_path = write_stimulus_epoch_run(zarr_path, epoch, overwrite=overwrite)
        occupancy = build_detection_occupancy_result(
            zarr_path,
            run_name=occupancy_run_name,
            stimulus_epoch_run=epoch_run_name,
            epoch_windows=None if apply else epoch_windows,
            source="active",
            bin_size=bin_size,
            smooth_sigma=smooth_sigma,
        )
        occupancy_path = None
        if apply:
            occupancy_path = write_detection_occupancy_run(
                zarr_path,
                occupancy,
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
                "stimulus_epoch_run": epoch_run_name,
                "stimulus_epoch_path": epoch_path,
                "detection_occupancy_run": occupancy_run_name,
                "detection_occupancy_path": occupancy_path,
                "windows": [
                    {
                        "label": window.label,
                        "start_frame": window.start_frame,
                        "end_frame": window.end_frame,
                        "coverage_pct": float(occupancy.coverage_pct[i]),
                        "detection_count": int(occupancy.detection_count[i]),
                    }
                    for i, window in enumerate(occupancy.windows)
                ],
                "spatial_occupancy": [
                    {
                        "zone_set_id": zone_set.zone_set_id,
                        "zone_set_source": zone_set.zone_set_source,
                        "zone_id": list(zone_set.zone_id),
                        "frame_count": zone_set.frame_count.tolist(),
                        "time_s": zone_set.time_s.tolist(),
                        "fraction_of_epoch": zone_set.fraction_of_epoch.tolist(),
                        "fraction_of_detected": zone_set.fraction_of_detected.tolist(),
                        "detected_frame_count": zone_set.detected_frame_count.tolist(),
                        "missing_frame_count": zone_set.missing_frame_count.tolist(),
                        "coverage_pct": zone_set.coverage_pct.tolist(),
                    }
                    for zone_set in occupancy.spatial_occupancy
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
    parser.add_argument("--coverage-min", type=float, default=98.0)
    parser.add_argument("--limit", type=int, default=6, help="Limit to the top N recordings by detect coverage.")
    parser.add_argument("--epoch-run-name", default=DEFAULT_EPOCH_RUN)
    parser.add_argument("--occupancy-run-name", default=DEFAULT_OCCUPANCY_RUN)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
        epoch_run_name=str(args.epoch_run_name),
        occupancy_run_name=str(args.occupancy_run_name),
        bin_size=int(args.bin_size),
        smooth_sigma=float(args.smooth_sigma),
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
                f"epoch={row['stimulus_epoch_path'] or row['stimulus_epoch_run']} "
                f"occupancy={row['detection_occupancy_path'] or row['detection_occupancy_run']}"
            )
            for window in row["windows"]:
                print(
                    f"  {window['label']}: detections={window['detection_count']} "
                    f"coverage={window['coverage_pct']:.1f}%"
                )
            for zone_set in row.get("spatial_occupancy", []):
                zones = [str(value) for value in zone_set.get("zone_id", [])]
                print(f"  spatial_occupancy/{zone_set['zone_set_id']}: zones={', '.join(zones)}")
                for window, counts in zip(row["windows"], zone_set.get("frame_count", [])):
                    rendered = ", ".join(
                        f"{zone}={int(count)}" for zone, count in zip(zones, counts)
                    )
                    print(f"    {window['label']}: {rendered}")
        if not args.apply:
            print("dry_run: pass --apply to write zarr analysis runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
