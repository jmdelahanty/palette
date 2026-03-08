#!/usr/bin/env python3
"""Backfill canonical eye-mask keypoint lineage attrs.

This utility migrates legacy eye-mask run attrs from:
  source_keypoint_run -> source_keypoints_run

It can also repair source keypoint lineage when the canonical/legacy run name
exists but the canonical group attr is missing, or when an explicit
--keypoint-run/--keypoint-group override is provided for a specific run.

Default mode is dry-run. Use --apply to write updates.
Default archive scope includes both analysis and training archives.
Use --analysis-only to intentionally scope to analysis archives.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import zarr

from fisheye.shared.provenance_attrs import (
    CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR,
    LEGACY_SOURCE_KEYPOINT_RUN_ATTR,
)


EYE_MASK_RUN_GROUPS = ("eye_masks_runs", "refined_eye_masks_runs")
KEYPOINT_RUN_GROUPS = ("keypoints_runs", "refined_keypoints_runs")
SOURCE_KEYPOINT_GROUP_ATTR = "source_keypoint_group"


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


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


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        purpose = root.attrs.get(key)
        if purpose is not None:
            value = str(purpose).strip().lower()
            if value in {"analysis", "training"}:
                return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _normalize_lineage_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return value


def _iter_eye_mask_runs(root: zarr.Group) -> Iterable[tuple[str, zarr.Group]]:
    for parent_name in EYE_MASK_RUN_GROUPS:
        parent = root.get(parent_name)
        if parent is None:
            continue
        try:
            run_names = sorted(list(parent.group_keys()))
        except Exception:
            run_names = sorted(list(parent.keys()))
        for run_name in run_names:
            if run_name not in parent:
                continue
            run_group = parent[run_name]
            if not hasattr(run_group, "attrs"):
                continue
            yield f"{parent_name}/{run_name}", run_group


def _resolve_existing_keypoint_run(
    root: zarr.Group,
    *,
    run_name: str,
    requested_group: Optional[str],
) -> BackfillResult | tuple[str, str]:
    if requested_group is not None:
        parent = root.get(requested_group)
        if parent is None:
            return BackfillResult(status="conflict", reason=f"missing keypoint group '{requested_group}'")
        if run_name not in parent:
            return BackfillResult(
                status="conflict",
                reason=f"keypoint run '{run_name}' not found in '{requested_group}'",
            )
        return run_name, requested_group

    matches: list[str] = []
    for group_name in KEYPOINT_RUN_GROUPS:
        parent = root.get(group_name)
        if parent is not None and run_name in parent:
            matches.append(group_name)
    if len(matches) == 1:
        return run_name, matches[0]
    if len(matches) > 1:
        return BackfillResult(
            status="conflict",
            reason=f"keypoint run '{run_name}' exists in multiple groups: {', '.join(matches)}",
        )
    return BackfillResult(status="missing_target", reason=f"keypoint run '{run_name}' not found")


def _backfill_run_group(
    root: zarr.Group,
    run_path: str,
    run_group: zarr.Group,
    *,
    apply: bool,
    override_keypoint_run: Optional[str] = None,
    override_keypoint_group: Optional[str] = None,
) -> BackfillResult:
    canonical_value = _normalize_lineage_value(run_group.attrs.get(CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR))
    legacy_value = _normalize_lineage_value(run_group.attrs.get(LEGACY_SOURCE_KEYPOINT_RUN_ATTR))
    group_value = _normalize_lineage_value(run_group.attrs.get(SOURCE_KEYPOINT_GROUP_ATTR))

    if canonical_value is not None and legacy_value is not None and str(canonical_value) != str(legacy_value):
        return BackfillResult(
            status="conflict",
            reason=(
                f"{CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR}={canonical_value!r} "
                f"!= {LEGACY_SOURCE_KEYPOINT_RUN_ATTR}={legacy_value!r}"
            ),
        )

    requested_run = _normalize_lineage_value(override_keypoint_run) if override_keypoint_run is not None else (
        canonical_value if canonical_value is not None else legacy_value
    )
    requested_group = _normalize_lineage_value(override_keypoint_group) if override_keypoint_group is not None else group_value

    if requested_run is None:
        if group_value is not None:
            return BackfillResult(
                status="conflict",
                reason=f"{SOURCE_KEYPOINT_GROUP_ATTR}={group_value!r} is set but keypoint run is missing",
            )
        return BackfillResult(status="missing_both")

    resolved = _resolve_existing_keypoint_run(
        root,
        run_name=str(requested_run),
        requested_group=str(requested_group) if requested_group is not None else None,
    )
    if isinstance(resolved, BackfillResult):
        return resolved

    desired_run, desired_group = resolved
    needs_update = (
        canonical_value != desired_run
        or legacy_value != desired_run
        or group_value != desired_group
    )
    if not needs_update:
        return BackfillResult(status="already_valid")

    if apply:
        run_group.attrs[CANONICAL_SOURCE_KEYPOINTS_RUN_ATTR] = desired_run
        run_group.attrs[LEGACY_SOURCE_KEYPOINT_RUN_ATTR] = desired_run
        run_group.attrs[SOURCE_KEYPOINT_GROUP_ATTR] = desired_group
        return BackfillResult(status="updated")
    return BackfillResult(
        status="would_update",
        reason=f"{run_path}: {desired_group}/{desired_run}",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Archive scope (default: any).",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Opt-out of training archives; equivalent to --zarr-use analysis.",
    )
    parser.add_argument(
        "--run-path",
        action="append",
        default=[],
        help="Optional run path(s) like eye_masks_runs/eye_masks_001 to target explicitly.",
    )
    parser.add_argument("--keypoint-run", type=str, default=None, help="Explicit source keypoint run override.")
    parser.add_argument(
        "--keypoint-group",
        choices=list(KEYPOINT_RUN_GROUPS),
        default=None,
        help="Explicit source keypoint group override.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")
    args = parser.parse_args(argv)

    if args.analysis_only and args.zarr_use != "any":
        parser.error("--analysis-only cannot be combined with --zarr-use.")
    if args.keypoint_group and not args.keypoint_run:
        parser.error("--keypoint-group requires --keypoint-run.")
    if args.keypoint_run and not args.run_path:
        parser.error("--keypoint-run requires at least one --run-path target.")
    zarr_use = "analysis" if args.analysis_only else str(args.zarr_use)
    target_run_paths = {str(value) for value in args.run_path}

    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    counts = {
        "zarr_scanned": 0,
        "run_groups_considered": 0,
        "filtered_zarr_use": 0,
        "missing_eye_mask_runs": 0,
        "updated": 0,
        "would_update": 0,
        "already_valid": 0,
        "missing_both": 0,
        "missing_target": 0,
        "conflict": 0,
        "errors": 0,
    }
    conflict_samples: list[str] = []
    update_samples: list[str] = []

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="a" if args.apply else "r")
            observed_use = _infer_zarr_use(root, zarr_path)
            if zarr_use != "any" and observed_use != zarr_use:
                counts["filtered_zarr_use"] += 1
                continue

            run_info = list(_iter_eye_mask_runs(root))
            if not run_info:
                counts["missing_eye_mask_runs"] += 1
                continue

            for run_path, run_group in run_info:
                if target_run_paths and run_path not in target_run_paths:
                    continue
                counts["run_groups_considered"] += 1
                result = _backfill_run_group(
                    root,
                    run_path,
                    run_group,
                    apply=bool(args.apply),
                    override_keypoint_run=args.keypoint_run,
                    override_keypoint_group=args.keypoint_group,
                )
                counts[result.status] += 1
                if result.status == "conflict" and result.reason:
                    if len(conflict_samples) < 5:
                        conflict_samples.append(f"{zarr_path}:{run_path}: {result.reason}")
                if result.status == "would_update" and result.reason:
                    if len(update_samples) < 5:
                        update_samples.append(f"{zarr_path}:{result.reason}")
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Eye-mask lineage attr backfill: "
        f"scope={zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"run_groups_considered={counts['run_groups_considered']} "
        f"filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_eye_mask_runs={counts['missing_eye_mask_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: updated={counts['updated']} would_update={counts['would_update']} "
        f"already_valid={counts['already_valid']} missing_both={counts['missing_both']} "
        f"missing_target={counts['missing_target']} conflict={counts['conflict']}"
    )
    if update_samples:
        print("Planned updates (first 5):")
        for sample in update_samples:
            print(f"  - {sample}")
    if conflict_samples:
        print("Conflicts (first 5):")
        for sample in conflict_samples:
            print(f"  - {sample}")

    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
