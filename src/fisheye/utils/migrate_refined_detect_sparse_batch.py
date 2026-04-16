#!/usr/bin/env python3
"""Batch sparse-first migration for refined-detect runs.

Default mode is dry-run. Use ``--apply`` to materialize successor sparse
refined runs via raw detect rows plus detect_quality labels.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import zarr

from fisheye.shared.refined_detect_curation import (
    has_curated_refined_source_detections_projection,
    has_sparse_curated_refined_detect_instances_arrays,
)
from fisheye.utils.migrate_refined_detect_sparse import (
    SparseMigrationConflictError,
    _promotion_blocked_reason,
    _open_root,
    _normalize_json,
    _normalize_text,
    apply_sparse_migration,
    build_sparse_migration_plan,
)


REFINED_PARENT_NAMES = ("refined_detect_runs", "refined_runs")


@dataclass
class PlannedMigration:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    source_refined_run: Optional[str] = None
    output_refined_run: Optional[str] = None
    parent_latest_refined_run: Optional[str] = None
    source_is_parent_latest: Optional[bool] = None
    source_detect_run: Optional[str] = None
    source_quality_run: Optional[str] = None
    current_sparse_surface: bool = False
    planned_instances: Optional[int] = None
    planned_source_detections: Optional[int] = None
    planned_multi_instance_frames: Optional[int] = None
    legacy_sparse_groups: Optional[list[str]] = None
    legacy_group_policy: Optional[str] = None
    promotion_requested: bool = False
    promotion_allowed: bool = False
    promotion_blocked_reason: Optional[str] = None


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        candidates: list[Path] = []
        if root.suffix == ".zarr" and (root.is_dir() or root.is_file()):
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(root.rglob("*.zarr"))
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> str:
    for key in ("zarr_use", "zarr_purpose"):
        purpose = root.attrs.get(key)
        if purpose is None:
            continue
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    for parent_name in REFINED_PARENT_NAMES:
        if parent_name in root:
            return root[parent_name]
    return None


def _select_refined_run(parent: zarr.Group, requested: Optional[str]) -> Optional[str]:
    requested_name = _normalize_text(requested)
    if requested_name:
        return requested_name if requested_name in parent else None
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        return None
    return sorted(str(name) for name in names)[-1]


def _already_sparse_migrated(refined_run: zarr.Group) -> bool:
    return bool(
        has_sparse_curated_refined_detect_instances_arrays(refined_run)
        and has_curated_refined_source_detections_projection(refined_run)
    )


def _plan_one(
    zarr_path: Path,
    *,
    zarr_use: str,
    refined_run: Optional[str],
    detect_run: Optional[str],
    quality_run: Optional[str],
    allow_missing_quality: bool,
    ignore_legacy_groups: bool,
    promote_latest_requested: bool,
    force_promote_nonlatest: bool,
    force: bool,
) -> PlannedMigration:
    try:
        root = _open_root(zarr_path, mode="r")
    except Exception as exc:
        return PlannedMigration(zarr_path=zarr_path, status="error", reason=f"open failed: {exc}")

    if zarr_use != "any":
        observed_use = _infer_zarr_use(root, zarr_path)
        if observed_use != zarr_use:
            return PlannedMigration(
                zarr_path=zarr_path,
                status="skipped",
                reason=f"zarr_use mismatch (wanted={zarr_use}, found={observed_use})",
            )

    refined_parent = _pick_refined_parent(root)
    if refined_parent is None:
        return PlannedMigration(zarr_path=zarr_path, status="missing", reason="refined_detect_runs missing")

    refined_run_name = _select_refined_run(refined_parent, refined_run)
    if refined_run_name is None:
        reason = "requested refined run not found" if refined_run else "no refined runs available"
        return PlannedMigration(zarr_path=zarr_path, status="missing", reason=reason)

    run_group = refined_parent[refined_run_name]
    current_sparse_surface = _already_sparse_migrated(run_group)
    if current_sparse_surface and not force:
        return PlannedMigration(
            zarr_path=zarr_path,
            status="skipped",
            reason="already migrated to sparse refined surfaces",
            source_refined_run=refined_run_name,
            parent_latest_refined_run=_normalize_text(refined_parent.attrs.get("latest")),
            source_is_parent_latest=bool(
                _normalize_text(refined_parent.attrs.get("latest")) == refined_run_name
            ),
            current_sparse_surface=True,
            promotion_requested=bool(promote_latest_requested),
        )

    try:
        plan = build_sparse_migration_plan(
            root,
            refined_run_name=refined_run_name,
            detect_run=detect_run,
            quality_run=quality_run,
            require_detect_quality=not bool(allow_missing_quality),
            ignore_legacy_groups=bool(ignore_legacy_groups),
        )
    except SparseMigrationConflictError as exc:
        return PlannedMigration(
            zarr_path=zarr_path,
            status="conflict",
            reason=str(exc),
            source_refined_run=refined_run_name,
            output_refined_run=exc.output_refined_run_name,
            current_sparse_surface=current_sparse_surface,
            legacy_sparse_groups=list(exc.legacy_sparse_groups),
            legacy_group_policy="blocked_requires_ignore_legacy_groups" if exc.legacy_sparse_groups else None,
        )
    except Exception as exc:
        return PlannedMigration(
            zarr_path=zarr_path,
            status="error",
            reason=str(exc),
            source_refined_run=refined_run_name,
            current_sparse_surface=current_sparse_surface,
        )

    promotion_blocked_reason = _promotion_blocked_reason(
        plan,
        promote_latest_requested=bool(promote_latest_requested),
        force_promote_nonlatest=bool(force_promote_nonlatest),
    )

    return PlannedMigration(
        zarr_path=zarr_path,
        status="conflict" if promotion_blocked_reason else "ok",
        reason=promotion_blocked_reason,
        source_refined_run=plan.source_refined_run_name,
        output_refined_run=plan.output_refined_run_name,
        parent_latest_refined_run=plan.parent_latest_refined_run_name,
        source_is_parent_latest=bool(plan.source_is_parent_latest),
        source_detect_run=plan.source_detect_run,
        source_quality_run=plan.source_quality_run,
        current_sparse_surface=current_sparse_surface,
        planned_instances=int(plan.planned_summary.get("total_instances", 0)),
        planned_source_detections=int(plan.planned_summary.get("total_source_detections", 0)),
        planned_multi_instance_frames=int(plan.planned_summary.get("multi_instance_frames", 0)),
        legacy_sparse_groups=list(plan.legacy_sparse_groups),
        legacy_group_policy=plan.legacy_group_policy,
        promotion_requested=bool(promote_latest_requested),
        promotion_allowed=bool(promote_latest_requested and promotion_blocked_reason is None),
        promotion_blocked_reason=promotion_blocked_reason,
    )


def _write_json_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "summary": summary, "rows": rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _row_payload(plan: PlannedMigration, *, applied: bool = False) -> dict[str, Any]:
    return {
        "zarr_path": str(plan.zarr_path),
        "status": plan.status,
        "reason": plan.reason,
        "source_refined_run": plan.source_refined_run,
        "output_refined_run": plan.output_refined_run,
        "parent_latest_refined_run": plan.parent_latest_refined_run,
        "source_is_parent_latest": plan.source_is_parent_latest,
        "source_detect_run": plan.source_detect_run,
        "source_quality_run": plan.source_quality_run,
        "current_sparse_surface": bool(plan.current_sparse_surface),
        "planned_instances": plan.planned_instances,
        "planned_source_detections": plan.planned_source_detections,
        "planned_multi_instance_frames": plan.planned_multi_instance_frames,
        "legacy_sparse_groups": list(plan.legacy_sparse_groups or []),
        "legacy_group_policy": plan.legacy_group_policy,
        "promotion_requested": bool(plan.promotion_requested),
        "promotion_allowed": bool(plan.promotion_allowed),
        "promotion_blocked_reason": plan.promotion_blocked_reason,
        "applied": bool(applied),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or .zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--refined-run", help="Source refined detect run to migrate (default: latest).")
    parser.add_argument("--detect-run", help="Override source detect run.")
    parser.add_argument("--quality-run", help="Override source quality run.")
    parser.add_argument(
        "--allow-missing-quality",
        action="store_true",
        help="Allow migration without a usable detect_quality report.",
    )
    parser.add_argument(
        "--ignore-legacy-groups",
        action="store_true",
        help="Allow migration to proceed when source refined runs have legacy manual/interpolated groups.",
    )
    parser.add_argument(
        "--no-promote-latest",
        action="store_true",
        help="Do not update refined_detect_runs.attrs['latest'] or detect_review_status_latest.",
    )
    parser.add_argument(
        "--force-promote-nonlatest",
        action="store_true",
        help="Allow promotion even when the source refined run is not the current parent latest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run migration even when sparse surfaces already exist.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write migrated sparse surfaces into successor refined runs (default: dry-run).",
    )
    parser.add_argument("--json-report", type=Path, help="Optional path to write a JSON report.")
    args = parser.parse_args(argv)

    roots = _resolve_roots(args.paths)
    promote_latest_requested = not bool(args.no_promote_latest)
    plans = [
        _plan_one(
            zarr_path,
            zarr_use=str(args.zarr_use),
            refined_run=args.refined_run,
            detect_run=args.detect_run,
            quality_run=args.quality_run,
            allow_missing_quality=bool(args.allow_missing_quality),
            ignore_legacy_groups=bool(args.ignore_legacy_groups),
            promote_latest_requested=promote_latest_requested,
            force_promote_nonlatest=bool(args.force_promote_nonlatest),
            force=bool(args.force),
        )
        for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive))
    ]

    if not plans:
        print("No zarr files found.")
        return 1

    rows: list[dict[str, Any]] = []
    ok = skipped = missing = conflict = error = applied_count = failed_apply = 0
    mode = "apply" if args.apply else "dry-run"

    if not args.apply:
        print("Planned refined-detect sparse migration (dry-run):")
        for plan in plans:
            rows.append(_row_payload(plan, applied=False))
            print(plan.zarr_path)
            print(f"  status: {plan.status}")
            if plan.source_refined_run:
                print(f"  source_refined_run: {plan.source_refined_run}")
            if plan.output_refined_run:
                print(f"  output_refined_run: {plan.output_refined_run}")
            if plan.parent_latest_refined_run:
                print(f"  parent_latest_refined_run: {plan.parent_latest_refined_run}")
            if plan.source_detect_run:
                print(f"  source_detect_run: {plan.source_detect_run}")
            if plan.source_quality_run:
                print(f"  source_quality_run: {plan.source_quality_run}")
            if plan.status == "ok":
                print(f"  planned_instances: {plan.planned_instances}")
                print(f"  planned_source_detections: {plan.planned_source_detections}")
                print(f"  planned_multi_instance_frames: {plan.planned_multi_instance_frames}")
                print(f"  current_sparse_surface: {plan.current_sparse_surface}")
                print(f"  legacy_group_policy: {plan.legacy_group_policy}")
                print(f"  legacy_sparse_groups: {plan.legacy_sparse_groups or []}")
                print(f"  promotion_requested: {plan.promotion_requested}")
                print(f"  promotion_allowed: {plan.promotion_allowed}")
            elif plan.status == "conflict":
                print(f"  promotion_requested: {plan.promotion_requested}")
                print(f"  promotion_allowed: {plan.promotion_allowed}")
            if plan.reason:
                print(f"  reason: {plan.reason}")
            if plan.status == "ok":
                ok += 1
            elif plan.status == "skipped":
                skipped += 1
            elif plan.status == "missing":
                missing += 1
            elif plan.status == "conflict":
                conflict += 1
            else:
                error += 1

        summary = {
            "scanned": len(plans),
            "ok": ok,
            "skipped": skipped,
            "missing": missing,
            "conflict": conflict,
            "error": error,
            "applied": 0,
            "failed_apply": 0,
        }
        print("\nSummary:")
        for key in ("scanned", "ok", "skipped", "missing", "conflict", "error"):
            print(f"  {key}: {summary[key]}")
        print("\nUse --apply to run migration.")
        if args.json_report:
            _write_json_report(args.json_report, rows=rows, summary=summary, mode=mode)
        return 0

    print("Applying refined-detect sparse migration:")
    for plan in plans:
        if plan.status != "ok":
            rows.append(_row_payload(plan, applied=False))
            print(plan.zarr_path)
            print(f"  status: {plan.status}")
            if plan.reason:
                print(f"  reason: {plan.reason}")
            if plan.status == "skipped":
                skipped += 1
            elif plan.status == "missing":
                missing += 1
            elif plan.status == "conflict":
                conflict += 1
            else:
                error += 1
            continue

        print(plan.zarr_path)
        print(f"  source_refined_run: {plan.source_refined_run}")
        print(f"  output_refined_run: {plan.output_refined_run}")
        print(f"  parent_latest_refined_run: {plan.parent_latest_refined_run or '-'}")
        print(f"  source_detect_run: {plan.source_detect_run}")
        print(f"  source_quality_run: {plan.source_quality_run or '-'}")
        try:
            root = _open_root(plan.zarr_path, mode="a")
            apply_plan = build_sparse_migration_plan(
                root,
                refined_run_name=plan.source_refined_run,
                output_run_name=plan.output_refined_run,
                detect_run=args.detect_run,
                quality_run=args.quality_run,
                require_detect_quality=not bool(args.allow_missing_quality),
                ignore_legacy_groups=bool(args.ignore_legacy_groups),
            )
            apply_result = apply_sparse_migration(
                root,
                zarr_path=plan.zarr_path.expanduser().resolve(),
                plan=apply_plan,
                command=None,
                promote_latest=not bool(args.no_promote_latest),
                force_promote_nonlatest=bool(args.force_promote_nonlatest),
            )
            applied_count += 1
            row = _row_payload(plan, applied=True)
            row["apply_result"] = _normalize_json(apply_result)
            rows.append(row)
            print("  applied: True")
        except Exception as exc:
            failed_apply += 1
            row = _row_payload(plan, applied=False)
            row["status"] = "apply_failed"
            row["reason"] = str(exc)
            rows.append(row)
            print(f"  applied: False")
            print(f"  reason: {exc}")

    summary = {
        "scanned": len(plans),
        "ok": len([plan for plan in plans if plan.status == "ok"]),
        "skipped": skipped,
        "missing": missing,
        "conflict": conflict,
        "error": error,
        "applied": applied_count,
        "failed_apply": failed_apply,
    }
    print("\nSummary:")
    for key in ("scanned", "ok", "skipped", "missing", "conflict", "error", "applied", "failed_apply"):
        print(f"  {key}: {summary[key]}")
    if args.json_report:
        _write_json_report(args.json_report, rows=rows, summary=summary, mode=mode)
    return 0 if failed_apply == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
