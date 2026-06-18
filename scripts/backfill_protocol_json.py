"""One-time backfill: patch protocol_json into existing zarr stimulus runs.

The import function previously looked for the wrong H5 key
(protocol_snapshot/protocol_json instead of protocol_snapshot/protocol_definition_json),
so no zarr stimulus runs have protocol_json attrs. This script reads the correct
key from each source H5 and writes it to the corresponding zarr stimulus run group.

Usage:
    python scripts/backfill_protocol_json.py [--registry PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import h5py
import zarr


def backfill_protocol_json(
    registry_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    db = sqlite3.connect(str(registry_path))
    db.row_factory = sqlite3.Row

    rows = db.execute(
        "SELECT zarr_path FROM datasets WHERE status = 'active' AND zarr_use = 'analysis'"
    ).fetchall()
    db.close()

    stats = {
        "zarrs_checked": 0,
        "runs_checked": 0,
        "already_present": 0,
        "patched": 0,
        "h5_missing": 0,
        "h5_no_protocol": 0,
        "errors": 0,
    }

    for row in rows:
        zarr_path = Path(row["zarr_path"])
        if not zarr_path.exists():
            continue
        stats["zarrs_checked"] += 1

        try:
            root = zarr.open(str(zarr_path), mode="r+" if not dry_run else "r")
        except Exception as exc:
            print(f"  ERROR opening {zarr_path}: {exc}")
            stats["errors"] += 1
            continue

        if "analysis" not in root or "stimulus_runs" not in root["analysis"]:
            continue

        sr = root["analysis"]["stimulus_runs"]
        child_names = list(sr.group_keys()) if hasattr(sr, "group_keys") else []

        for child_name in child_names:
            stats["runs_checked"] += 1
            child = sr[child_name]

            if "protocol_json" in child.attrs:
                stats["already_present"] += 1
                continue

            source_h5 = child.attrs.get("source_h5")
            if not source_h5 or not Path(source_h5).exists():
                stats["h5_missing"] += 1
                continue

            try:
                with h5py.File(source_h5, "r") as f:
                    proto_key = None
                    if "/protocol_snapshot/protocol_definition_json" in f:
                        proto_key = "/protocol_snapshot/protocol_definition_json"
                    elif "/protocol_snapshot/protocol_json" in f:
                        proto_key = "/protocol_snapshot/protocol_json"

                    if proto_key is None:
                        stats["h5_no_protocol"] += 1
                        continue

                    proto_bytes = f[proto_key][()]
                    if isinstance(proto_bytes, bytes):
                        proto_str = proto_bytes.decode("utf-8")
                    else:
                        proto_str = str(proto_bytes)

                    # Validate it's parseable JSON
                    payload = json.loads(proto_str)
                    proto_name = payload.get("protocol_name", "<unknown>")

                    if dry_run:
                        print(f"  [DRY RUN] Would patch {zarr_path.name} / {child_name} -> {proto_name}")
                    else:
                        child.attrs["protocol_json"] = proto_str
                        print(f"  Patched {zarr_path.name} / {child_name} -> {proto_name}")
                    stats["patched"] += 1

            except Exception as exc:
                print(f"  ERROR patching {zarr_path.name} / {child_name}: {exc}")
                stats["errors"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill protocol_json into zarr stimulus runs.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to registry SQLite file. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be patched without writing.",
    )
    args = parser.parse_args()

    registry_path = args.registry
    if registry_path is None:
        # Try common locations
        candidates = [
            Path("/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"),
            Path("/nvme1/palette_registry.sqlite"),
            Path("runs/registry/palette_registry.sqlite"),
        ]
        for c in candidates:
            if c.exists():
                registry_path = c
                break
        if registry_path is None:
            print("ERROR: Could not find registry. Use --registry to specify path.")
            sys.exit(1)

    print(f"Registry: {registry_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    stats = backfill_protocol_json(registry_path, dry_run=args.dry_run)

    print()
    print("=== Summary ===")
    print(f"  Zarrs checked:        {stats['zarrs_checked']}")
    print(f"  Stimulus runs checked:{stats['runs_checked']}")
    print(f"  Already had attr:     {stats['already_present']}")
    print(f"  Patched:              {stats['patched']}")
    print(f"  H5 file missing:      {stats['h5_missing']}")
    print(f"  H5 no protocol data:  {stats['h5_no_protocol']}")
    print(f"  Errors:               {stats['errors']}")

    if args.dry_run and stats["patched"] > 0:
        print()
        print("Re-run without --dry-run to apply patches.")


if __name__ == "__main__":
    main()
