#!/usr/bin/env python3
"""List/write analysis zarr archives whose detect review is not approved."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class UnapprovedRow:
    zarr_path: str
    zarr_use: str
    latest_refined_run: Optional[str]
    review_state: Optional[str]
    review_intended_use: Optional[str]
    review_resolved_group: Optional[str]
    reason: str


def _resolve_roots(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _iter_zarr(roots: List[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _read_zarr_attrs(zarr_json_path: Path) -> Dict[str, object]:
    if not zarr_json_path.exists():
        return {}
    try:
        payload = json.loads(zarr_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _infer_zarr_use(zarr_path: Path, root_attrs: Dict[str, object]) -> str:
    purpose = root_attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _collect_unapproved_rows(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    approved_state: str,
) -> List[UnapprovedRow]:
    rows: List[UnapprovedRow] = []
    approved_norm = approved_state.strip().lower()

    for zarr_path in _iter_zarr(roots, recursive):
        root_attrs = _read_zarr_attrs(zarr_path / "zarr.json")
        zarr_use = _infer_zarr_use(zarr_path, root_attrs)
        if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
            continue

        refined_parent = zarr_path / "refined_detect_runs" / "zarr.json"
        parent_attrs = _read_zarr_attrs(refined_parent)
        latest = parent_attrs.get("latest")
        latest_run = str(latest).strip() if latest is not None else None
        latest_run = latest_run or None

        reason = "approved"
        review_state: Optional[str] = None
        review_intended_use: Optional[str] = None
        review_resolved_group: Optional[str] = None

        if latest_run is None:
            reason = "no_latest_refined_run"
        else:
            run_attrs = _read_zarr_attrs(zarr_path / "refined_detect_runs" / latest_run / "zarr.json")
            status = run_attrs.get("detect_review_status")
            if not isinstance(status, dict):
                reason = "no_detect_review_status"
            else:
                raw_state = status.get("state")
                review_state = str(raw_state).strip() if raw_state is not None else None
                review_state = review_state or None

                raw_use = status.get("intended_use")
                review_intended_use = str(raw_use).strip() if raw_use is not None else None
                review_intended_use = review_intended_use or None

                raw_group = status.get("resolved_group")
                review_resolved_group = str(raw_group).strip() if raw_group is not None else None
                review_resolved_group = review_resolved_group or None

                if review_state is None:
                    reason = "review_state_missing"
                elif review_state.strip().lower() != approved_norm:
                    reason = "review_state_not_approved"

        if reason == "approved":
            continue

        rows.append(
            UnapprovedRow(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                latest_refined_run=latest_run,
                review_state=review_state,
                review_intended_use=review_intended_use,
                review_resolved_group=review_resolved_group,
                reason=reason,
            )
        )

    rows.sort(key=lambda row: row.zarr_path)
    return rows


def _object_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1;",
        (str(name),),
    ).fetchone()
    return row is not None


def _preferred_detect_review_view_name(conn: sqlite3.Connection) -> str | None:
    if _object_exists(conn, "refined_detect_review_current"):
        return "refined_detect_review_current"
    if _object_exists(conn, "detect_quality_current"):
        return "detect_quality_current"
    return None


def _collect_unapproved_rows_from_registry(
    registry_path: Path,
    *,
    zarr_use_filter: str,
    approved_state: str,
    path_contains: Optional[str],
) -> List[UnapprovedRow]:
    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    try:
        if not _object_exists(conn, "datasets"):
            raise RuntimeError(f"{registry_path}: missing datasets table")

        detect_review_view = _preferred_detect_review_view_name(conn)
        if detect_review_view is not None:
            sql = [
                "WITH detect_choice AS (",
                "  SELECT",
                "    dqc.dataset_id AS dataset_id,",
                "    dqc.refined_run AS refined_run,",
                "    dqc.review_state AS review_state,",
                "    dqc.review_intended_use AS review_intended_use,",
                "    dqc.review_resolved_group AS review_resolved_group,",
                "    ROW_NUMBER() OVER (",
                "      PARTITION BY dqc.dataset_id",
                "      ORDER BY",
                "        COALESCE(dqc.review_timestamp_utc, dqc.refined_created_utc, dqc.quality_updated_utc) DESC,",
                "        COALESCE(dqc.refined_created_utc, '') DESC,",
                "        dqc.refined_run DESC",
                "    ) AS _rn",
                f"  FROM {detect_review_view} dqc",
                ")",
                "SELECT",
                "  d.zarr_path AS zarr_path,",
                "  d.zarr_use AS zarr_use,",
                "  dc.refined_run AS latest_refined_run,",
                "  dc.review_state AS review_state,",
                "  dc.review_intended_use AS review_intended_use,",
                "  dc.review_resolved_group AS review_resolved_group",
                "FROM datasets d",
                "LEFT JOIN detect_choice dc",
                "  ON dc.dataset_id = d.dataset_id AND dc._rn = 1",
                "WHERE d.zarr_path IS NOT NULL AND TRIM(d.zarr_path) != ''",
                "  AND (d.status IS NULL OR d.status != 'missing')",
            ]
        else:
            sql = [
                "SELECT",
                "  d.zarr_path AS zarr_path,",
                "  d.zarr_use AS zarr_use,",
                "  NULL AS latest_refined_run,",
                "  NULL AS review_state,",
                "  NULL AS review_intended_use,",
                "  NULL AS review_resolved_group",
                "FROM datasets d",
                "WHERE d.zarr_path IS NOT NULL AND TRIM(d.zarr_path) != ''",
                "  AND (d.status IS NULL OR d.status != 'missing')",
            ]

        params: List[object] = []
        if zarr_use_filter != "any":
            sql.append("  AND LOWER(COALESCE(d.zarr_use, '')) = ?")
            params.append(str(zarr_use_filter).strip().lower())
        if path_contains:
            sql.append("  AND d.zarr_path LIKE ?")
            params.append(f"%{str(path_contains)}%")
        sql.append("ORDER BY d.zarr_path")

        rows = conn.execute(" ".join(sql), params).fetchall()
    finally:
        conn.close()

    approved_norm = str(approved_state).strip().lower()
    out: List[UnapprovedRow] = []
    for row in rows:
        zarr_path = str(row["zarr_path"]).strip()
        if not zarr_path:
            continue

        zarr_use = str(row["zarr_use"] or "unknown").strip().lower() or "unknown"
        latest_refined_run = str(row["latest_refined_run"]).strip() if row["latest_refined_run"] is not None else None
        latest_refined_run = latest_refined_run or None
        review_state = str(row["review_state"]).strip() if row["review_state"] is not None else None
        review_state = review_state or None
        review_intended_use = (
            str(row["review_intended_use"]).strip() if row["review_intended_use"] is not None else None
        )
        review_intended_use = review_intended_use or None
        review_resolved_group = (
            str(row["review_resolved_group"]).strip() if row["review_resolved_group"] is not None else None
        )
        review_resolved_group = review_resolved_group or None

        if latest_refined_run is None:
            reason = "no_detect_quality_row"
        elif review_state is None:
            reason = "review_state_missing"
        elif review_state.strip().lower() != approved_norm:
            reason = "review_state_not_approved"
        else:
            continue

        out.append(
            UnapprovedRow(
                zarr_path=zarr_path,
                zarr_use=zarr_use,
                latest_refined_run=latest_refined_run,
                review_state=review_state,
                review_intended_use=review_intended_use,
                review_resolved_group=review_resolved_group,
                reason=reason,
            )
        )
    return out


def _write_text(paths: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(paths)
    if payload:
        payload += "\n"
    output_path.write_text(payload, encoding="utf-8")


def _write_tsv(rows: List[UnapprovedRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["zarr_path\tzarr_use\tlatest_refined_run\treview_state\treview_intended_use\treview_resolved_group\treason"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.zarr_path,
                    row.zarr_use,
                    row.latest_refined_run or "<none>",
                    row.review_state or "<none>",
                    row.review_intended_use or "<none>",
                    row.review_resolved_group or "<none>",
                    row.reason,
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a list of analysis zarr archives that do not have approved detect review status."
    )
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
        default="analysis",
        help="Filter by zarr use (default: analysis).",
    )
    parser.add_argument(
        "--approved-state",
        default="approved",
        help="Review state to consider approved (default: approved).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/analysis_without_approval.txt"),
        help="Output text file with one zarr path per line.",
    )
    parser.add_argument(
        "--details",
        type=Path,
        help="Optional TSV output with detailed status columns.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry sqlite path; when provided, list from registry instead of crawling files.",
    )
    parser.add_argument(
        "--path-contains",
        type=str,
        help="Optional substring filter applied to zarr_path (primarily for registry mode).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON rows to stdout.")
    args = parser.parse_args(argv)

    if args.registry is not None:
        rows = _collect_unapproved_rows_from_registry(
            Path(args.registry),
            zarr_use_filter=str(args.zarr_use),
            approved_state=str(args.approved_state),
            path_contains=args.path_contains,
        )
    else:
        roots = _resolve_roots(args.paths)
        rows = _collect_unapproved_rows(
            roots,
            recursive=bool(args.recursive),
            zarr_use_filter=str(args.zarr_use),
            approved_state=str(args.approved_state),
        )

    _write_text([row.zarr_path for row in rows], args.output)
    if args.details is not None:
        _write_tsv(rows, args.details)

    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(f"wrote: {args.output}")
        print(f"count: {len(rows)}")
        if args.details is not None:
            print(f"details: {args.details}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
