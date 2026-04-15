#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import zarr

from fisheye.shared.refined_detect_resolution import resolve_detect_review_target


def _iter_zarr(roots: Iterable[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_dir() and root.suffix == ".zarr":
            yield root
            continue
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")
            yield from root.glob("*.zarr")


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_detect_runs" in root:
        return root["refined_detect_runs"]
    if "refined_runs" in root:
        return root["refined_runs"]
    return None


def _select_refined_run(parent: zarr.Group) -> Optional[str]:
    latest = parent.attrs.get("latest")
    if latest and latest in parent:
        return str(latest)
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        return None
    return sorted(names)[-1]


def _should_update(status: object, fill_missing: bool) -> bool:
    if not isinstance(status, dict) or not status:
        return True
    if not fill_missing:
        return False
    required = ("state", "method", "intended_use", "resolved_group")
    return any(not status.get(key) for key in required)


def _resolve_review_target(root: zarr.Group, refined_run_name: str, refined_run: zarr.Group) -> tuple[Optional[str], list[str]]:
    resolution = resolve_detect_review_target(
        root,
        refined_run_name=refined_run_name,
        refined_run=refined_run,
        override_group=None,
    )
    return resolution.resolved_group, list(resolution.preference_chain)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill detect_review_status for refined detect runs that are missing it.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarrs.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry-run).")
    parser.add_argument("--state", default="approved", help="Review state to set.")
    parser.add_argument("--method", default="manual", help="Review method to set.")
    parser.add_argument("--intended-use", default="training", help="Intended use label.")
    parser.add_argument("--reviewer", default="delahantyj", help="Reviewer name.")
    parser.add_argument("--notes", help="Optional notes.")
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill missing fields on existing review_status instead of skipping.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing review_status entries.",
    )
    args = parser.parse_args(argv)

    if args.paths:
        roots = args.paths
    else:
        env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
        roots = [Path(env_root)] if env_root else [Path("/nvme1/recordings")]
    seen: set[str] = set()
    candidates = []
    for path in _iter_zarr(roots, args.recursive):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(path)

    if not candidates:
        print("No zarr files found.")
        return 1

    updated = skipped = missing = failed = 0
    for zarr_path in candidates:
        try:
            root = zarr.open_group(str(zarr_path), mode="r" if not args.apply else "a")
        except Exception as exc:
            print(f"ERROR opening {zarr_path}: {exc}")
            failed += 1
            continue

        refined_parent = _pick_refined_parent(root)
        if refined_parent is None:
            skipped += 1
            continue
        refined_run_name = _select_refined_run(refined_parent)
        if not refined_run_name:
            skipped += 1
            continue
        refined_run = refined_parent[refined_run_name]

        existing = refined_run.attrs.get("detect_review_status")
        if not args.force and not _should_update(existing, args.fill_missing):
            skipped += 1
            continue

        resolved_group, preference_chain = _resolve_review_target(root, refined_run_name, refined_run)
        payload: dict[str, object] = {}
        if isinstance(existing, dict) and not args.force:
            payload.update(existing)
        payload.setdefault("state", args.state)
        payload.setdefault("method", args.method)
        payload.setdefault("intended_use", args.intended_use)
        payload.setdefault("resolved_group", resolved_group)
        payload.setdefault("preference_chain", preference_chain)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if args.reviewer:
            payload.setdefault("reviewer", args.reviewer)
        if args.notes:
            payload.setdefault("notes", args.notes)

        if args.apply:
            refined_run.attrs["detect_review_status"] = payload
            refined_parent.attrs["detect_review_status_latest"] = refined_run_name
            print(f"UPDATED {zarr_path} ({refined_run_name})")
            print(f"  review_status: {payload}")
        else:
            print(f"WOULD UPDATE {zarr_path} ({refined_run_name})")
            print(f"  current: {existing if isinstance(existing, dict) else '—'}")
            print(f"  new: {payload}")
        updated += 1

    print("\nSummary:")
    print(f"  updated: {updated}")
    print(f"  skipped: {skipped}")
    print(f"  missing: {missing}")
    print(f"  failed: {failed}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
