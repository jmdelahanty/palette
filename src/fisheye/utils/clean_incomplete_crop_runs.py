from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import zarr


@dataclass
class CropCleanupPlan:
    zarr_path: Path
    delete_runs: list[str]
    latest_before: Optional[str]
    latest_after: Optional[str]


def _iter_zarr(paths: list[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        if path.suffix == ".zarr" and path.exists():
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("zarr/*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")


def _parse_iso_utc(value: object) -> float:
    if value is None:
        return float("-inf")
    text = str(value).strip()
    if not text:
        return float("-inf")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("-inf")


def _pick_latest_run(crop_parent: zarr.Group) -> Optional[str]:
    names = list(crop_parent.group_keys())
    if not names:
        return None
    names.sort(
        key=lambda name: (
            _parse_iso_utc(
                crop_parent[name].attrs.get("created_at_utc")
                or crop_parent[name].attrs.get("started_at_utc")
            ),
            name,
        )
    )
    return names[-1]


def _build_plan(
    zarr_path: Path,
    *,
    remove_non_completed: bool,
    remove_statuses: set[str],
) -> CropCleanupPlan:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return CropCleanupPlan(
            zarr_path=zarr_path,
            delete_runs=[],
            latest_before=None,
            latest_after=None,
        )
    latest_before = crop_parent.attrs.get("latest")
    delete_runs: list[str] = []
    kept_runs: list[str] = []
    for run_name in list(crop_parent.group_keys()):
        status = str(crop_parent[run_name].attrs.get("status", "")).strip().lower()
        should_delete = (status != "completed") if remove_non_completed else (status in remove_statuses)
        if should_delete:
            delete_runs.append(run_name)
        else:
            kept_runs.append(run_name)

    latest_after: Optional[str]
    if latest_before and latest_before in kept_runs:
        latest_after = str(latest_before)
    elif kept_runs:
        latest_after = sorted(kept_runs)[-1]
    else:
        latest_after = None

    return CropCleanupPlan(
        zarr_path=zarr_path,
        delete_runs=sorted(delete_runs),
        latest_before=str(latest_before) if latest_before else None,
        latest_after=latest_after,
    )


def _apply_plan(plan: CropCleanupPlan) -> CropCleanupPlan:
    if not plan.delete_runs:
        return plan
    root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return plan
    for run_name in plan.delete_runs:
        if run_name in crop_parent:
            del crop_parent[run_name]
    latest = crop_parent.attrs.get("latest")
    if not latest or latest not in crop_parent:
        crop_parent.attrs["latest"] = _pick_latest_run(crop_parent)
    return _build_plan(
        plan.zarr_path,
        remove_non_completed=False,
        remove_statuses=set(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove incomplete crop runs (e.g. failed/aborted) and fix crop_runs.latest."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr path(s) or recording root(s).")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument("--apply", action="store_true", help="Apply deletions (default is dry-run).")
    parser.add_argument(
        "--remove-status",
        nargs="+",
        default=["failed", "running"],
        help="Statuses to remove (ignored when --remove-non-completed is set).",
    )
    parser.add_argument(
        "--remove-non-completed",
        action="store_true",
        help="Remove any crop run where status != completed.",
    )
    args = parser.parse_args(argv)

    remove_statuses = {str(value).strip().lower() for value in args.remove_status if str(value).strip()}
    zarr_paths = sorted({path.resolve() for path in _iter_zarr(args.paths, recursive=bool(args.recursive))})
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    plans = [
        _build_plan(
            Path(path),
            remove_non_completed=bool(args.remove_non_completed),
            remove_statuses=remove_statuses,
        )
        for path in zarr_paths
    ]

    if not args.apply:
        print("Planned crop cleanup (dry-run):")
        any_changes = False
        for plan in plans:
            if not plan.delete_runs:
                continue
            any_changes = True
            print(plan.zarr_path)
            print(f"  delete: {len(plan.delete_runs)} run(s)")
            for run_name in plan.delete_runs:
                print(f"    - {run_name}")
            print(f"  latest: {plan.latest_before} -> {plan.latest_after}")
        if not any_changes:
            print("No incomplete crop runs found.")
        print("\nUse --apply to delete these runs.")
        return 0

    changed = 0
    deleted_total = 0
    for plan in plans:
        if not plan.delete_runs:
            continue
        before_delete_count = len(plan.delete_runs)
        after = _apply_plan(plan)
        changed += 1
        deleted_total += before_delete_count
        print(after.zarr_path)
        print(f"  deleted: {before_delete_count}")
        print(f"  latest: {plan.latest_before} -> {after.latest_before}")

    print("\nSummary:")
    print(f"  zarrs_changed: {changed}")
    print(f"  runs_deleted: {deleted_total}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
