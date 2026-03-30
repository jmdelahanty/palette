from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import zarr

from fisheye.registry.db import Registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair training zarr refined-keypoint run ties where a migrated "
            "traditional_v2_seed run should supersede an older sibling run "
            "for the same source_keypoints_run."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Path to the Palette registry sqlite database.",
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        help="Filter keypoint_quality_current by review_state.",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        help="Filter keypoint_quality_current by review_intended_use.",
    )
    parser.add_argument(
        "--match",
        default="traditional_v2_seed",
        help="Substring match used to target migrated refined runs for repair.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned repairs without writing zarr attrs.",
    )
    return parser.parse_args()


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _resolve_refined_parent(root: zarr.Group) -> tuple[Optional[zarr.Group], Optional[str]]:
    parent = root.get("refined_keypoints_runs")
    if parent is not None:
        return parent, "refined_keypoints_runs"
    parent = root.get("keypoints_refined_runs")
    if parent is not None:
        return parent, "keypoints_refined_runs"
    return None, None


def main() -> int:
    args = _parse_args()
    registry = Registry(args.registry)

    rows = registry.query_keypoint_quality_current(
        review_state=args.review_state,
        review_intended_use=args.review_intended_use,
    )

    scanned = 0
    updated = 0
    skipped = 0

    for row in rows:
        expected_refined_run = _as_text(row["refined_run"])
        if args.match and args.match not in expected_refined_run:
            continue

        dataset_id = _as_text(row["dataset_id"])
        source_keypoint_run = _as_text(row["source_keypoint_run"])
        zarr_row = registry.conn.execute(
            "SELECT zarr_path FROM dataset_context_current WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        if zarr_row is None:
            skipped += 1
            continue

        zarr_path = Path(_as_text(zarr_row["zarr_path"]))
        scanned += 1

        try:
            root = zarr.open_group(zarr_path, mode="a" if not args.dry_run else "r")
        except Exception as exc:
            print(f"SKIP {zarr_path} open_error={exc}")
            skipped += 1
            continue

        refined_parent, refined_parent_name = _resolve_refined_parent(root)
        if refined_parent is None or refined_parent_name is None:
            print(f"SKIP {zarr_path} missing refined keypoint parent")
            skipped += 1
            continue
        if expected_refined_run not in refined_parent:
            print(f"SKIP {zarr_path} missing expected run {expected_refined_run}")
            skipped += 1
            continue

        run_group = refined_parent[expected_refined_run]
        if _as_text(run_group.attrs.get("source_keypoints_run")) != source_keypoint_run:
            print(
                f"SKIP {zarr_path} source mismatch expected_source={source_keypoint_run} "
                f"run_source={_as_text(run_group.attrs.get('source_keypoints_run'))}"
            )
            skipped += 1
            continue

        migration_created_at = _as_text(run_group.attrs.get("migration_created_at_utc"))
        target_created_at = migration_created_at or _as_text(run_group.attrs.get("created_utc"))
        current_created_at = _as_text(run_group.attrs.get("created_utc"))
        current_latest = _as_text(refined_parent.attrs.get("latest"))

        needs_created = bool(target_created_at) and current_created_at != target_created_at
        needs_latest = current_latest != expected_refined_run

        if not needs_created and not needs_latest:
            print(f"UNCHANGED {zarr_path} run={expected_refined_run}")
            continue

        print(
            f"{'PLAN' if args.dry_run else 'UPDATE'} {zarr_path} "
            f"parent={refined_parent_name} run={expected_refined_run} "
            f"created_utc:{current_created_at or '<missing>'}->{target_created_at or current_created_at or '<missing>'} "
            f"latest:{current_latest or '<missing>'}->{expected_refined_run}"
        )

        if not args.dry_run:
            if needs_created and target_created_at:
                run_group.attrs["created_utc"] = target_created_at
            if needs_latest:
                refined_parent.attrs["latest"] = expected_refined_run
            updated += 1

    print(
        f"Summary scanned={scanned} updated={updated} skipped={skipped} "
        f"dry_run={bool(args.dry_run)} match={args.match}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
