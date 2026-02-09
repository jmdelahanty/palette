#!/usr/bin/env python3
"""Audit registry dataset rows against on-disk Zarr paths."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from fisheye.registry.db import Registry, RegistryPaths

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Table = None


@dataclass
class AuditRow:
    dataset_id: str
    zarr_path: str
    status: Optional[str]
    fs_state: str
    session_uuid: Optional[str]
    last_seen_utc: Optional[str]


def _norm(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except Exception:
        return candidate.absolute()


def _path_in_scopes(path: Path, scopes: list[Path]) -> bool:
    if not scopes:
        return True
    for root in scopes:
        if path == root:
            return True
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _zarr_fs_state(path: Path) -> str:
    if not path.exists():
        return "missing_path"
    if not path.is_dir():
        return "not_directory"
    if (path / "zarr.json").exists() or (path / ".zgroup").exists():
        return "zarr_root_ok"
    return "missing_zarr_marker"


def _fetch_rows(registry: Registry) -> list[dict]:
    cur = registry.conn.execute(
        """
        SELECT dataset_id, session_uuid, zarr_path, status, last_seen_utc
        FROM datasets
        ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id ASC;
        """
    )
    return [dict(row) for row in cur.fetchall()]


def _audit(registry: Registry, *, scopes: list[Path]) -> list[AuditRow]:
    out: list[AuditRow] = []
    for row in _fetch_rows(registry):
        path = _norm(row["zarr_path"])
        if not _path_in_scopes(path, scopes):
            continue
        out.append(
            AuditRow(
                dataset_id=str(row["dataset_id"]),
                zarr_path=str(path),
                status=row.get("status"),
                fs_state=_zarr_fs_state(path),
                session_uuid=row.get("session_uuid"),
                last_seen_utc=row.get("last_seen_utc"),
            )
        )
    return out


def _print_rich(rows: list[AuditRow], *, limit: int, show_all: bool) -> None:
    console = Console()
    state_counts = Counter(row.fs_state for row in rows)
    status_counts = Counter((row.status or "NULL") for row in rows)

    summary = Table(title="Registry Dataset Path Audit", title_style="bold magenta")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="yellow")
    summary.add_row("Rows audited", str(len(rows)))
    summary.add_row("On-disk OK", str(state_counts.get("zarr_root_ok", 0)))
    summary.add_row("Dangling rows", str(len([r for r in rows if r.fs_state != "zarr_root_ok"])))
    summary.add_row("State counts", ", ".join(f"{k}={v}" for k, v in sorted(state_counts.items())))
    summary.add_row("Status counts", ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    console.print(summary)

    issues = rows if show_all else [r for r in rows if r.fs_state != "zarr_root_ok"]
    if not issues:
        console.print("[green]No dangling dataset rows found.[/green]")
        return

    table = Table(title="Dataset Rows", title_style="bold magenta")
    table.add_column("dataset_id", style="cyan")
    table.add_column("status")
    table.add_column("fs_state")
    table.add_column("session_uuid")
    table.add_column("last_seen_utc")
    table.add_column("zarr_path")
    for row in issues[:limit]:
        table.add_row(
            row.dataset_id,
            row.status or "NULL",
            row.fs_state,
            row.session_uuid or "-",
            row.last_seen_utc or "-",
            row.zarr_path,
        )
    console.print(table)
    if len(issues) > limit:
        console.print(f"... and {len(issues) - limit} more row(s) not shown (use --limit).")


def _print_plain(rows: list[AuditRow], *, limit: int, show_all: bool) -> None:
    state_counts = Counter(row.fs_state for row in rows)
    print(f"Rows audited: {len(rows)}")
    print(f"State counts: {dict(sorted(state_counts.items()))}")
    issues = rows if show_all else [r for r in rows if r.fs_state != "zarr_root_ok"]
    print(f"Rows shown: {min(len(issues), limit)} of {len(issues)}")
    for row in issues[:limit]:
        print(
            "\t".join(
                [
                    row.dataset_id,
                    row.status or "NULL",
                    row.fs_state,
                    row.zarr_path,
                ]
            )
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument(
        "--scope",
        type=Path,
        action="append",
        default=[],
        help="Only audit dataset paths under this root (repeatable).",
    )
    parser.add_argument("--show-all", action="store_true", help="Show all scoped rows, not just dangling rows.")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to display.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument(
        "--mark-missing",
        action="store_true",
        help="Set datasets.status='missing' for dangling rows not already marked missing.",
    )
    parser.add_argument(
        "--delete-missing-rows",
        action="store_true",
        help="Delete dataset rows where status='missing' and path is still dangling.",
    )
    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    scopes = [_norm(path) for path in args.scope]

    registry = Registry(registry_path)
    rows = _audit(registry, scopes=scopes)
    dangling = [r for r in rows if r.fs_state != "zarr_root_ok"]
    to_mark = [r for r in dangling if (r.status or "").lower() != "missing"]
    to_delete = [r for r in dangling if (r.status or "").lower() == "missing"]

    if args.mark_missing and to_mark:
        with registry.conn:
            for row in to_mark:
                registry.conn.execute(
                    "UPDATE datasets SET status = 'missing' WHERE dataset_id = ?;",
                    (row.dataset_id,),
                )
        # Reflect updates for any subsequent delete action in this invocation.
        rows = _audit(registry, scopes=scopes)
        dangling = [r for r in rows if r.fs_state != "zarr_root_ok"]
        to_delete = [r for r in dangling if (r.status or "").lower() == "missing"]

    if args.delete_missing_rows and to_delete:
        with registry.conn:
            for row in to_delete:
                registry.conn.execute(
                    "DELETE FROM datasets WHERE dataset_id = ?;",
                    (row.dataset_id,),
                )
        rows = _audit(registry, scopes=scopes)
        dangling = [r for r in rows if r.fs_state != "zarr_root_ok"]

    if args.json:
        payload = {
            "registry": str(registry_path),
            "scopes": [str(path) for path in scopes],
            "rows_audited": len(rows),
            "dangling_rows": len(dangling),
            "marked_missing": len(to_mark) if args.mark_missing else 0,
            "deleted_rows": len(to_delete) if args.delete_missing_rows else 0,
            "rows": [
                {
                    "dataset_id": row.dataset_id,
                    "status": row.status,
                    "fs_state": row.fs_state,
                    "session_uuid": row.session_uuid,
                    "last_seen_utc": row.last_seen_utc,
                    "zarr_path": row.zarr_path,
                }
                for row in (rows if args.show_all else dangling)
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        if Table is not None and Console is not None:
            _print_rich(rows, limit=max(1, int(args.limit)), show_all=args.show_all)
            if args.mark_missing:
                Console().print(f"[green]Marked missing:[/green] {len(to_mark)} row(s).")
            if args.delete_missing_rows:
                Console().print(f"[green]Deleted rows:[/green] {len(to_delete)} row(s).")
        else:
            _print_plain(rows, limit=max(1, int(args.limit)), show_all=args.show_all)
            if args.mark_missing:
                print(f"Marked missing: {len(to_mark)} row(s).")
            if args.delete_missing_rows:
                print(f"Deleted rows: {len(to_delete)} row(s).")

    registry.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
