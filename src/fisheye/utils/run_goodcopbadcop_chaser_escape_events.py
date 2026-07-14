"""Run chaser escape-events components across recordings.

Requires a materialized `chaser_bout_response` on each target -- it supplies the bout table
and the virtual references. Run that first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from fisheye.analysis.chaser_escape_events import (
    DEFAULT_COMPONENT_NAME,
    DEFAULT_PEAK_SPEED_THRESHOLD_MM_S,
    build_chaser_escape_events_result,
    write_chaser_escape_events_component,
)
from fisheye.registry.db import RegistryPaths
from fisheye.shared.json_safety import json_attr_safe
from fisheye.utils.run_goodcopbadcop_chaser_bout_response import (
    _explicit_zarr_targets,
    _filesystem_targets,
    _query_targets,
)


def run_for_targets(
    targets: Sequence[dict[str, Any]],
    *,
    chaser_distance_run: str,
    bout_response_component: str,
    component_name: str,
    peak_speed_threshold_mm_s: float,
    apply: bool,
    overwrite: bool,
    no_png: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target in targets:
        zarr_path = Path(str(target["zarr_path"]))
        applied_path = None
        status = "dry_run"
        error = None
        summary: dict[str, Any] | None = None
        qc_warnings: list[str] = []
        try:
            result = build_chaser_escape_events_result(
                zarr_path,
                chaser_distance_run=chaser_distance_run,
                bout_response_component=bout_response_component,
                component_name=component_name,
                peak_speed_threshold_mm_s=peak_speed_threshold_mm_s,
            )
            summary = result.summary
            qc_warnings = list(result.qc_warnings)
            status = result.status
            if apply:
                applied_path = write_chaser_escape_events_component(
                    zarr_path, result, overwrite=overwrite, write_png=not no_png
                )
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "recording_id": target.get("recording_id"),
                "zarr_path": str(zarr_path),
                "component_name": component_name,
                "chaser_escape_events_path": applied_path,
                "status": status,
                "error": error,
                "qc_warnings": qc_warnings,
                "summary": summary,
            }
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--zarr", action="append", type=Path)
    p.add_argument("--recordings-root", action="append", type=Path)
    p.add_argument("--recursive-recordings-root", action="store_true")
    p.add_argument("--recording-like", default="%GoodCopBadCop%")
    p.add_argument("--coverage-min", type=float, default=90.0)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--chaser-distance-run", default="latest")
    p.add_argument("--bout-response-component", default="latest")
    p.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    p.add_argument("--peak-speed-threshold-mm-s", type=float, default=DEFAULT_PEAK_SPEED_THRESHOLD_MM_S)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.zarr:
        targets = _explicit_zarr_targets(args.zarr)
        source = "explicit_zarr"
    elif args.recordings_root:
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

    results = run_for_targets(
        targets,
        chaser_distance_run=str(args.chaser_distance_run),
        bout_response_component=str(args.bout_response_component),
        component_name=str(args.component_name),
        peak_speed_threshold_mm_s=float(args.peak_speed_threshold_mm_s),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        no_png=bool(args.no_png),
    )
    payload = {"apply": bool(args.apply), "source": source, "target_count": len(targets), "results": results}
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
        return 0

    print(f"target_count: {len(targets)}")
    ok = 0
    for row in results:
        if row["status"] == "failed":
            print(f"  FAILED {row['recording_id']}: {row['error']}")
            continue
        ok += 1
        s = row["summary"] or {}
        rates = " ".join(
            f"{k.removeprefix('escape_rate_per_valid_min_')}={v:.1f}/min"
            for k, v in s.items()
            if k.startswith("escape_rate_per_valid_min_")
        )
        print(f"  {row['recording_id']}: {s.get('escape_event_count', 0)} events  {rates}")
    print(f"  ok: {ok}/{len(results)}")
    if not args.apply:
        print("dry_run: pass --apply to write chaser escape-events components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
