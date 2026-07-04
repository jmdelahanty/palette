#!/usr/bin/env python3
"""Audit crop write backend usage across Palette Zarr archives."""

from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use
from functools import partial
_infer_zarr_use = partial(infer_zarr_use, default="unknown")
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import zarr


@dataclass
class CropWriteBackendRow:
    zarr_path: str
    zarr_use: str
    crop_run: str
    status: Optional[str]
    video_source_type: Optional[str]
    requested_backend: Optional[str]
    effective_backend: Optional[str]
    fallback_reason: Optional[str]
    roi_storage: Optional[str]
    created_at_utc: Optional[str]


def _resolve_roots(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _collect_rows(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    latest_only: bool,
) -> List[CropWriteBackendRow]:
    rows: List[CropWriteBackendRow] = []

    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open_group(str(zarr_path), mode="r")
        except Exception:
            continue

        zarr_use = _infer_zarr_use(root, zarr_path)
        if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
            continue

        crop_parent = root.get("crop_runs")
        if crop_parent is None:
            continue

        if latest_only:
            latest = crop_parent.attrs.get("latest")
            run_names = [str(latest).strip()] if latest else []
        else:
            run_names = sorted(crop_parent.group_keys())

        for run_name in run_names:
            if not run_name or run_name not in crop_parent:
                continue
            run_group = crop_parent[run_name]
            requested = run_group.attrs.get("write_backend_requested")
            effective = run_group.attrs.get("write_backend_effective") or run_group.attrs.get("write_backend")
            if requested is None:
                if effective == "kvikio_gds":
                    requested = "kvikio"
                elif effective == "standard_zarr":
                    requested = "standard"

            rows.append(
                CropWriteBackendRow(
                    zarr_path=str(zarr_path),
                    zarr_use=zarr_use,
                    crop_run=run_name,
                    status=run_group.attrs.get("status"),
                    video_source_type=run_group.attrs.get("video_source_type"),
                    requested_backend=requested,
                    effective_backend=effective,
                    fallback_reason=run_group.attrs.get("write_backend_fallback_reason"),
                    roi_storage=run_group.attrs.get("roi_storage"),
                    created_at_utc=run_group.attrs.get("created_at_utc"),
                )
            )

    rows.sort(key=lambda row: (row.zarr_path, row.crop_run))
    return rows


def _summarize(rows: List[CropWriteBackendRow]) -> Dict[str, object]:
    by_requested: Dict[str, int] = {}
    by_effective: Dict[str, int] = {}
    fallback_reasons: Dict[str, int] = {}
    external_runs = 0
    fallback_total = 0

    for row in rows:
        requested = str(row.requested_backend or "<none>")
        effective = str(row.effective_backend or "<none>")
        by_requested[requested] = by_requested.get(requested, 0) + 1
        by_effective[effective] = by_effective.get(effective, 0) + 1
        if row.video_source_type == "external":
            external_runs += 1
        if row.fallback_reason:
            fallback_total += 1
            key = str(row.fallback_reason)
            fallback_reasons[key] = fallback_reasons.get(key, 0) + 1

    return {
        "total_runs": len(rows),
        "external_runs": external_runs,
        "requested_backend_counts": dict(sorted(by_requested.items())),
        "effective_backend_counts": dict(sorted(by_effective.items())),
        "fallback_total": fallback_total,
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
    }


def _write_tsv(rows: List[CropWriteBackendRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "zarr_path\tzarr_use\tcrop_run\tstatus\tvideo_source_type\trequested_backend\teffective_backend\tfallback_reason\troi_storage\tcreated_at_utc"
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.zarr_path,
                    row.zarr_use,
                    row.crop_run,
                    row.status or "",
                    row.video_source_type or "",
                    row.requested_backend or "",
                    row.effective_backend or "",
                    row.fallback_reason or "",
                    row.roi_storage or "",
                    row.created_at_utc or "",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit crop write backend usage across zarr archives.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter by zarr use (default: any).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Inspect all crop runs (default: latest run only).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON payload.")
    parser.add_argument("--details", type=Path, help="Optional TSV output of per-run rows.")
    args = parser.parse_args(argv)

    rows = _collect_rows(
        _resolve_roots(args.paths),
        recursive=bool(args.recursive),
        zarr_use_filter=str(args.zarr_use),
        latest_only=not bool(args.all_runs),
    )
    summary = _summarize(rows)
    if args.details is not None:
        _write_tsv(rows, args.details)

    if args.json:
        payload = {"summary": summary, "rows": [asdict(row) for row in rows]}
        if args.details is not None:
            payload["details_path"] = str(args.details)
        print(json.dumps(payload, indent=2))
        return 0

    print(f"total_runs: {summary['total_runs']}")
    print(f"external_runs: {summary['external_runs']}")
    print(f"requested_backend_counts: {summary['requested_backend_counts']}")
    print(f"effective_backend_counts: {summary['effective_backend_counts']}")
    print(f"fallback_total: {summary['fallback_total']}")
    print(f"fallback_reason_counts: {summary['fallback_reason_counts']}")
    if args.details is not None:
        print(f"details: {args.details}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
