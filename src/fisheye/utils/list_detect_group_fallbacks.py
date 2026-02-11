from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from fisheye.registry.db import Registry
from fisheye.utils import check_recording_steps as status_mod


def _collect_group_rows(
    *,
    roots: List[Path],
    recursive: bool,
    requested_use: str,
    target_group: str,
    registry: Optional[Registry],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for h5_path in status_mod._iter_h5(roots, recursive):  # noqa: SLF001
        recording_dir = h5_path.parent.parent
        camera_id = status_mod._read_camera_id(h5_path)  # noqa: SLF001
        recording_id = status_mod._read_recording_id(h5_path)  # noqa: SLF001
        zarr_candidates = status_mod._resolve_zarr_candidates(  # noqa: SLF001
            recording_dir=recording_dir,
            recording_id=recording_id,
            requested_use=requested_use,
            registry=registry,
        )
        for zarr_path, zarr_use in zarr_candidates:
            info = status_mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001
            if not info["zarr_exists"]:
                continue
            resolved_group = str(info.get("refined_detect_resolved_group") or "").strip().lower()
            if resolved_group != target_group:
                continue
            rows.append(
                {
                    "recording_dir": str(recording_dir),
                    "recording_id": recording_id,
                    "camera_id": camera_id,
                    "zarr_path": str(zarr_path),
                    "zarr_use": zarr_use,
                    "zarr_purpose": info.get("zarr_purpose"),
                    "detect_present": bool(info.get("detect_present")),
                    "detect_coverage": info.get("detect_coverage"),
                    "detect_quality_present": bool(info.get("detect_quality_present")),
                    "detect_quality_grade": info.get("detect_quality_grade"),
                    "detect_quality_clean_percent": info.get("detect_quality_clean_percent"),
                    "resolved_detect_group": resolved_group,
                }
            )
    rows.sort(key=lambda row: (str(row.get("recording_id") or ""), str(row.get("zarr_path") or "")))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List recordings whose resolved refined-detect source group matches a target "
            "(default: raw fallback)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording root(s) to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for recordings.")
    parser.add_argument(
        "--zarr-use",
        choices=("all", "training", "analysis"),
        default="analysis",
        help="Filter rows by zarr use type (default: analysis).",
    )
    parser.add_argument(
        "--group",
        choices=("raw", "manual", "interpolated", "filtered"),
        default="raw",
        help="Resolved detect group to list (default: raw).",
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON array.")
    args = parser.parse_args(argv)

    roots = status_mod._resolve_root(args.paths)  # noqa: SLF001
    registry: Optional[Registry] = None
    try:
        if args.registry is not None:
            registry_path = args.registry.expanduser().resolve()
            if not registry_path.exists():
                print(f"Registry not found: {registry_path}")
                return 1
            registry = Registry(registry_path)

        rows = _collect_group_rows(
            roots=roots,
            recursive=bool(args.recursive),
            requested_use=str(args.zarr_use),
            target_group=str(args.group).lower(),
            registry=registry,
        )
    finally:
        if registry is not None:
            registry.close()

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print(f"No recordings with resolved detect group '{args.group}'.")
        return 0

    for row in rows:
        print(
            "\t".join(
                [
                    str(row["recording_id"] or "unknown"),
                    str(row["camera_id"] or "unknown"),
                    str(row["zarr_path"]),
                    f"quality={row['detect_quality_grade'] or 'none'}",
                ]
            )
        )
    print(f"\nTotal: {len(rows)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
