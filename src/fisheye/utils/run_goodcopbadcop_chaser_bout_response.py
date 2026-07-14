"""Run chaser bout-response (object-relative bout kinematics) components across recordings."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sqlite3
from typing import Any, Optional, Sequence

from fisheye.analysis.chaser_bout_response import (
    DEFAULT_COMPONENT_NAME,
    build_chaser_bout_response_result,
    write_chaser_bout_response_component,
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
    swim_bout_run: str,
    component_name: str,
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
        qc_warnings: list[str] = []
        try:
            result = build_chaser_bout_response_result(
                zarr_path,
                chaser_distance_run=chaser_distance_run,
                swim_bout_run=swim_bout_run,
                component_name=component_name,
            )
            summary = result.summary
            qc_warnings = list(result.qc_warnings)
            status = result.status
            if apply:
                applied_path = write_chaser_bout_response_component(
                    zarr_path, result, overwrite=overwrite,
                    write_png=not no_png, write_interactive_spec=not no_interactive_spec,
                )
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "recording_id": target.get("recording_id"),
                "zarr_path": str(zarr_path),
                "component_name": component_name,
                "chaser_bout_response_path": applied_path,
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
    parser.add_argument("--swim-bout-run", default="latest")
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
    results = run_for_targets(
        targets,
        chaser_distance_run=str(args.chaser_distance_run),
        swim_bout_run=str(args.swim_bout_run),
        component_name=str(args.component_name),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        no_png=bool(args.no_png),
        no_interactive_spec=bool(args.no_interactive_spec),
    )
    payload = {"apply": bool(args.apply), "source": source, "target_count": len(targets), "results": results}
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"target_count: {len(targets)}")
        ok=sum(1 for r in results if r["status"]=="computed")
        for row in results:
            if row["status"]=="failed":
                print(f"  FAILED {row['recording_id']}: {row['error']}")
        print(f"  computed: {ok}/{len(results)}")
        if not args.apply:
            print("dry_run: pass --apply to write chaser bout-response components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
