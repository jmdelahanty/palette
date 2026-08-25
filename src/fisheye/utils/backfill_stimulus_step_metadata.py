#!/usr/bin/env python3
"""Backfill canonical stimulus step metadata from recording H5 snapshots.

Default mode is dry-run. Use ``--apply`` to write updates. The intended batch
target is ``/nvme1/recordings``; each existing ``analysis/stimulus_runs/<run>``
is enriched from its ``source_h5`` attr or the affiliated ``raw/*.h5`` file.
"""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from fisheye.analysis.import_stimulus_to_zarr import backfill_stimulus_step_metadata


def _resolve_roots(paths: Sequence[Path]) -> list[Path]:
    if paths:
        return [Path(path).expanduser() for path in paths]
    return [Path("/nvme1/recordings")]


def _status_counts(summaries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"zarr_scanned": len(summaries), "runs_scanned": 0, "errors": 0}
    for summary in summaries:
        counts["runs_scanned"] += int(summary.get("runs_scanned", 0))
        for detail in summary.get("details", []):
            status = str(detail.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            if status == "error":
                counts["errors"] += 1
    return counts


def _print_detail(summary: dict, *, verbose: bool) -> None:
    noisy = {
        "would_backfill",
        "would_overwrite",
        "backfilled",
        "overwritten",
        "requires_immutable_successor",
        "skipped_missing_h5",
        "skipped_ambiguous_h5",
        "skipped_no_step_events",
        "error",
    }
    for detail in summary.get("details", []):
        status = str(detail.get("status", "unknown"))
        if not verbose and status not in noisy:
            continue
        parts = [status, str(summary.get("zarr_path", ""))]
        run_name = detail.get("run_name")
        if run_name:
            parts.append(f"run={run_name}")
        source_h5 = detail.get("source_h5")
        if source_h5:
            parts.append(f"h5={source_h5}")
        semantic_status = detail.get("protocol_semantic_source_status")
        if semantic_status:
            parts.append(f"semantic_status={semantic_status}")
        semantic_hash = detail.get("protocol_semantic_hash")
        if semantic_hash:
            parts.append(f"semantic_hash={semantic_hash}")
        recipe = detail.get("protocol_recipe_label")
        if recipe:
            parts.append(f"recipe={recipe!r}")
        reason = detail.get("reason")
        if reason:
            parts.append(str(reason))
        print(": ".join(parts[:2]) + (" " + " ".join(parts[2:]) if len(parts) > 2 else ""))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or analysis Zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .zarr archives.")
    parser.add_argument("--stimulus-run", help="Only backfill one analysis/stimulus_runs/<run>.")
    parser.add_argument("--source-h5", type=Path, help="Explicit H5 path. Intended for single-archive repairs.")
    parser.add_argument(
        "--camera-to-projector-offset-deg",
        type=float,
        default=0.0,
        help=(
            "Angular correction applied to MOVING_GRATING directions when materializing "
            "canonical step metadata. Use 180.0 for known inverted-projector recordings."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing steps/stimulus_coordinates.")
    parser.add_argument("--apply", action="store_true", help="Write updates (default: dry-run).")
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help="After each write, refresh Zarr consolidated metadata for strict readers.",
    )
    parser.add_argument("--json-report", type=Path, help="Optional path to write the per-archive report JSON.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero when any archive errors.")
    parser.add_argument("--verbose", action="store_true", help="Print skipped archives as well as candidates/errors.")
    args = parser.parse_args(argv)

    if args.consolidate_metadata and not args.apply:
        parser.error("--consolidate-metadata requires --apply")
    if args.source_h5 is not None and len(args.paths) != 1:
        parser.error("--source-h5 requires exactly one zarr path")

    roots = _resolve_roots(args.paths)
    summaries: list[dict] = []
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        try:
            summary = backfill_stimulus_step_metadata(
                zarr_path,
                stimulus_run=args.stimulus_run,
                source_h5=args.source_h5,
                moving_grating_camera_offset_deg=float(args.camera_to_projector_offset_deg),
                overwrite=bool(args.overwrite),
                apply=bool(args.apply),
                consolidate_metadata=bool(args.consolidate_metadata),
                console=None,
            )
        except Exception as exc:
            summary = {
                "zarr_path": str(zarr_path),
                "runs_scanned": 0,
                "details": [{"status": "error", "reason": str(exc)}],
            }
        summaries.append(summary)
        _print_detail(summary, verbose=bool(args.verbose))

    if not summaries:
        print("No zarr files found.")
        return 1

    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")

    counts = _status_counts(summaries)
    mode = "Applied" if args.apply else "Dry run"
    summary_keys = [
        "zarr_scanned",
        "runs_scanned",
        "would_backfill",
        "would_overwrite",
        "backfilled",
        "overwritten",
        "requires_immutable_successor",
        "skipped_existing",
        "skipped_missing_h5",
        "skipped_ambiguous_h5",
        "skipped_no_step_events",
        "error",
        "errors",
    ]
    summary_text = " ".join(f"{key}={counts.get(key, 0)}" for key in summary_keys)
    print(f"Stimulus step metadata backfill {mode}: {summary_text}")
    return 1 if args.fail_on_error and counts.get("errors", 0) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
