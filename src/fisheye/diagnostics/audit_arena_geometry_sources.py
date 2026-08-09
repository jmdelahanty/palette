"""Read-only cohort audit of the arena geometry selected by Palette.

The audit opens active analysis archives with direct metadata, resolves the exact
authoritative/latest-complete chaser-distance child, and invokes the same shared arena
geometry resolver used by analytics.  It never mutates the registry or an archive.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from fisheye.analysis.chaser_distance_io import resolve_chaser_distance_run_path
from fisheye.shared.arena_geometry import resolve_arena_geometry
from fisheye.shared.zarr_io import open_zarr_root


AUDIT_SCHEMA_ID = "palette.arena_geometry_source_audit"
AUDIT_SCHEMA_VERSION = 1
_CHASER_PARENT_PATH = "analysis/chaser_distance_runs"


def _active_analysis_archives(
    registry_path: Path,
    *,
    protocol_name: str,
) -> list[dict[str, str]]:
    resolved = registry_path.expanduser().resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """
            SELECT dataset_id, recording_id, zarr_path
            FROM dataset_context_current
            WHERE dataset_status = 'active'
              AND zarr_use = 'analysis'
              AND protocol_name = ?
            ORDER BY recording_id, zarr_path
            """,
            (str(protocol_name),),
        ).fetchall()

    archives = [
        {
            "dataset_id": str(dataset_id),
            "recording_id": str(recording_id),
            "zarr_path": str(zarr_path),
        }
        for dataset_id, recording_id, zarr_path in rows
    ]
    counts = Counter(item["recording_id"] for item in archives)
    duplicates = sorted(recording_id for recording_id, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(
            "Expected exactly one active analysis archive per recording; ambiguous "
            f"recordings: {duplicates}"
        )
    return archives


def _positive_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not (result > 0.0) or result == float("inf"):
        raise ValueError(f"{field} must be a positive finite number")
    return result


def _close_root(root: Any) -> None:
    store = getattr(root, "store", None)
    close = getattr(store, "close", None)
    if callable(close):
        close()


def inspect_archive(archive: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = dict(archive)
    root = None
    try:
        root = open_zarr_root(Path(archive["zarr_path"]), mode="r")
        try:
            parent = root[_CHASER_PARENT_PATH]
        except KeyError:
            result.update(
                {
                    "status": "unavailable",
                    "unavailable_reason": "no_chaser_distance_runs_parent",
                }
            )
            return result
        selector = (
            "authoritative_run"
            if parent.attrs.get("authoritative_run") is not None
            else "latest_complete"
        )
        run_name, run_path = resolve_chaser_distance_run_path(root, run_name="latest")
        run_group = root[run_path]
        pixels_per_mm = _positive_float(
            run_group.attrs.get("pixels_per_mm_projector"),
            field="pixels_per_mm_projector",
        )
        geometry, notes = resolve_arena_geometry(
            root,
            run_group,
            pixels_per_mm=pixels_per_mm,
        )
        result.update(
            {
                "status": "ok",
                "selector": selector,
                "run_name": run_name,
                "run_path": run_path,
                "pixels_per_mm_projector": pixels_per_mm,
                "arena_geometry_status": geometry.status,
                "arena_geometry_source": geometry.source,
                "arena_geometry_shape": geometry.shape,
                "notes": list(notes),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        if root is not None:
            _close_root(root)
    return result


def build_audit(
    registry_path: Path,
    *,
    protocol_name: str,
) -> dict[str, Any]:
    archives = _active_analysis_archives(
        registry_path,
        protocol_name=protocol_name,
    )
    records = [inspect_archive(archive) for archive in archives]
    geometry_counts = Counter(
        str(record.get("arena_geometry_status"))
        for record in records
        if record.get("status") == "ok"
    )
    source_counts = Counter(
        str(record.get("arena_geometry_source"))
        for record in records
        if record.get("status") == "ok"
    )
    unavailable = [record for record in records if record.get("status") == "unavailable"]
    errors = [record for record in records if record.get("status") == "error"]
    nominal = [
        record
        for record in records
        if record.get("arena_geometry_status") != "dish_mask"
        and record.get("status") == "ok"
    ]
    return {
        "schema_id": AUDIT_SCHEMA_ID,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path.expanduser().resolve()),
        "registry_open_mode": "sqlite_uri_mode_ro_query_only",
        "zarr_metadata_mode": "direct_use_consolidated_false",
        "protocol_name": str(protocol_name),
        "summary": {
            "recording_count": len(records),
            "auditable_count": sum(record.get("status") == "ok" for record in records),
            "ok_count": sum(record.get("status") == "ok" for record in records),
            "unavailable_count": len(unavailable),
            "error_count": len(errors),
            "nominal_or_non_dish_mask_count": len(nominal),
            "geometry_status_counts": dict(sorted(geometry_counts.items())),
            "geometry_source_counts": dict(sorted(source_counts.items())),
        },
        "records": records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--protocol-name", default="GoodCopBadCop")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    audit = build_audit(args.registry, protocol_name=args.protocol_name)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    if int(audit["summary"]["error_count"]) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
