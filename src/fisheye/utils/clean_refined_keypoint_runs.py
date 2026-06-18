from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import zarr

from fisheye.shared.zarr_run_completion import (
    RUN_LATEST_COMPLETE_ATTR,
    RUN_LATEST_PENDING_ATTR,
)


PARENT_NAME = "refined_keypoints_runs"


@dataclass(frozen=True)
class CleanupPlan:
    zarr_path: Path
    delete_runs: tuple[str, ...]
    latest_before: Optional[str]
    latest_after: Optional[str]


def _iter_zarr(paths: list[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.suffix == ".zarr" and path.exists():
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("zarr/*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")


def _source_keypoints_run(run_group: zarr.Group) -> Optional[str]:
    value = run_group.attrs.get("source_keypoints_run") or run_group.attrs.get("source_keypoint_run")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_latest(parent: zarr.Group) -> Optional[str]:
    names = sorted(str(name) for name in parent.group_keys())
    return names[-1] if names else None


def _build_plan(
    zarr_path: Path,
    *,
    run_names: set[str],
    source_keypoints_run: Optional[str],
) -> CleanupPlan:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root.get(PARENT_NAME)
    if parent is None:
        return CleanupPlan(zarr_path=zarr_path, delete_runs=(), latest_before=None, latest_after=None)

    latest_before = parent.attrs.get("latest")
    delete_runs: list[str] = []
    kept_runs: list[str] = []
    for run_name in sorted(str(name) for name in parent.group_keys()):
        run_group = parent[run_name]
        matches_name = bool(run_names) and run_name in run_names
        matches_source = (
            source_keypoints_run is not None
            and _source_keypoints_run(run_group) == source_keypoints_run
        )
        if matches_name or matches_source:
            delete_runs.append(run_name)
        else:
            kept_runs.append(run_name)

    if latest_before and str(latest_before) in kept_runs:
        latest_after = str(latest_before)
    elif kept_runs:
        latest_after = sorted(kept_runs)[-1]
    else:
        latest_after = None

    return CleanupPlan(
        zarr_path=zarr_path,
        delete_runs=tuple(delete_runs),
        latest_before=str(latest_before) if latest_before else None,
        latest_after=latest_after,
    )


def _set_or_delete_attr(attrs: object, name: str, value: Optional[str]) -> None:
    if value is None:
        if attrs.get(name) is not None:
            del attrs[name]
        return
    attrs[name] = value


def _apply_plan(plan: CleanupPlan) -> None:
    if not plan.delete_runs:
        return
    root = zarr.open_group(str(plan.zarr_path), mode="a", use_consolidated=False)
    parent = root.get(PARENT_NAME)
    if parent is None:
        return

    for run_name in plan.delete_runs:
        if run_name in parent:
            del parent[run_name]

    latest_after = plan.latest_after
    if latest_after is None:
        latest_after = _pick_latest(parent)

    attrs = parent.attrs
    _set_or_delete_attr(attrs, "latest", latest_after)
    _set_or_delete_attr(attrs, RUN_LATEST_COMPLETE_ATTR, latest_after)
    if attrs.get(RUN_LATEST_PENDING_ATTR) in plan.delete_runs:
        del attrs[RUN_LATEST_PENDING_ATTR]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete selected refined_keypoints_runs children and repair parent latest attrs."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr path(s) or recording root(s).")
    parser.add_argument("--recursive", action="store_true", help="Search roots recursively for zarr archives.")
    parser.add_argument("--run-name", action="append", default=[], help="Refined keypoint run name to delete; repeatable.")
    parser.add_argument("--source-keypoints-run", help="Delete refined runs sourced from this keypoints run.")
    parser.add_argument("--apply", action="store_true", help="Apply deletions; default is dry-run.")
    args = parser.parse_args(argv)

    run_names = {str(value).strip() for value in args.run_name if str(value).strip()}
    source_keypoints_run = str(args.source_keypoints_run).strip() if args.source_keypoints_run else None
    if not run_names and source_keypoints_run is None:
        raise SystemExit("Refusing to delete without --run-name or --source-keypoints-run.")

    zarr_paths = sorted({path.resolve() for path in _iter_zarr(args.paths, bool(args.recursive))})
    if not zarr_paths:
        print("No zarr files found.")
        return 1

    plans = [
        _build_plan(path, run_names=run_names, source_keypoints_run=source_keypoints_run)
        for path in zarr_paths
    ]

    if not args.apply:
        print("Planned refined keypoint cleanup (dry-run):")
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
            print("No matching refined keypoint runs found.")
        print("\nUse --apply to delete these runs.")
        return 0

    changed = 0
    deleted = 0
    for plan in plans:
        if not plan.delete_runs:
            continue
        _apply_plan(plan)
        changed += 1
        deleted += len(plan.delete_runs)
        print(plan.zarr_path)
        print(f"  deleted: {len(plan.delete_runs)}")
        print(f"  latest: {plan.latest_before} -> {plan.latest_after}")

    print("\nSummary:")
    print(f"  zarrs_changed: {changed}")
    print(f"  runs_deleted: {deleted}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
