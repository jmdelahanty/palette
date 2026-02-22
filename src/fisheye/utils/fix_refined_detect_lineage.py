#!/usr/bin/env python3
"""Repair stale refined-detect lineage pointers in crop metadata.

Default mode is dry-run. Use --apply to write changes.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import zarr

from fisheye.shared.refined_detect_review import resolve_refined_detect_group


REFINED_PARENT_NAMES = ("refined_detect_runs", "refined_runs")
REQUIRED_BBOX_ARRAYS = ("bbox_norm_coords", "bbox_coords", "bbox")
REFINED_SOURCE_TYPES = {"manual", "interpolated", "filtered"}


@dataclass(frozen=True)
class ResolvedRefinedRun:
    run_name: str
    subgroup_name: str
    source_type: str


@dataclass
class PlannedRepair:
    target_group: zarr.Group
    target_path: str
    field: str
    old_value: Any
    new_value: Any
    reason: str


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


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    return value


def _normalize_str(value: Any) -> Optional[str]:
    value = _normalize_scalar(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _group_keys(group: zarr.Group) -> list[str]:
    try:
        return sorted(list(group.group_keys()))
    except Exception:
        return sorted(list(group.keys()))


def _latest_name(group: zarr.Group) -> Optional[str]:
    latest = _normalize_str(group.attrs.get("latest"))
    if latest and latest in group:
        return latest
    return None


def _refined_parent(root: zarr.Group) -> tuple[Optional[str], Optional[zarr.Group]]:
    for parent_name in REFINED_PARENT_NAMES:
        if parent_name in root:
            return parent_name, root[parent_name]
    return None, None


def _has_detection_arrays(group: zarr.Group) -> bool:
    if "frame_indices" not in group:
        return False
    return any(name in group for name in REQUIRED_BBOX_ARRAYS)


def _review_override(run_group: zarr.Group) -> Optional[str]:
    review_status = run_group.attrs.get("detect_review_status")
    if isinstance(review_status, Mapping):
        for key in ("target_group", "resolved_group"):
            value = _normalize_str(review_status.get(key))
            if value:
                return value
    return None


def _resolved_refined_runs(parent: zarr.Group) -> dict[str, ResolvedRefinedRun]:
    resolved: dict[str, ResolvedRefinedRun] = {}
    for run_name in _group_keys(parent):
        if run_name not in parent:
            continue
        run_group = parent[run_name]
        override = _review_override(run_group)
        selection = resolve_refined_detect_group(run_group, override_group=override)
        if selection.group is None or selection.group not in run_group:
            continue
        subgroup = run_group[selection.group]
        if not _has_detection_arrays(subgroup):
            continue
        source_type = selection.label or selection.group
        resolved[run_name] = ResolvedRefinedRun(
            run_name=run_name,
            subgroup_name=selection.group,
            source_type=source_type,
        )
    return resolved


def _normalize_path(path: Any) -> Optional[str]:
    normalized = _normalize_str(path)
    if not normalized:
        return None
    normalized = normalized.strip("/")
    return normalized or None


def _parse_refined_source_path(path: Optional[str], parent_name: str) -> Optional[tuple[str, str]]:
    if not path:
        return None
    parts = path.split("/")
    if len(parts) != 3:
        return None
    if parts[0] != parent_name:
        return None
    run_name, subgroup_name = parts[1], parts[2]
    if not run_name or not subgroup_name:
        return None
    return run_name, subgroup_name


def _select_crop_runs(
    root: zarr.Group,
    *,
    requested_runs: list[str],
    limit: str,
    warnings: list[str],
) -> list[tuple[str, zarr.Group]]:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return []

    all_names = _group_keys(crop_parent)
    if not all_names:
        return []

    if requested_runs:
        selected: list[tuple[str, zarr.Group]] = []
        for run_name in requested_runs:
            if run_name not in crop_parent:
                warnings.append(f"requested crop run '{run_name}' not found")
                continue
            selected.append((run_name, crop_parent[run_name]))
        return selected

    if limit == "all":
        return [(run_name, crop_parent[run_name]) for run_name in all_names]

    latest = _latest_name(crop_parent)
    if latest and latest in crop_parent:
        return [(latest, crop_parent[latest])]
    return [(all_names[-1], crop_parent[all_names[-1]])]


def _choose_run_from_crop(
    crop_runs: list[tuple[str, zarr.Group]],
    *,
    parent_name: str,
    valid_runs: dict[str, ResolvedRefinedRun],
) -> Optional[str]:
    for _run_name, crop_group in crop_runs:
        source_path = _normalize_path(crop_group.attrs.get("detection_source_path"))
        parsed = _parse_refined_source_path(source_path, parent_name) if source_path else None
        if parsed and parsed[0] in valid_runs:
            return parsed[0]
        source_refined = _normalize_str(crop_group.attrs.get("source_refined_run"))
        if source_refined and source_refined in valid_runs:
            return source_refined
    return None


def _choose_canonical_run(
    *,
    current_latest: Optional[str],
    crop_preferred_run: Optional[str],
    valid_runs: dict[str, ResolvedRefinedRun],
) -> tuple[Optional[str], Optional[str]]:
    if current_latest and current_latest in valid_runs:
        return current_latest, "latest_valid"
    if crop_preferred_run and crop_preferred_run in valid_runs:
        return crop_preferred_run, "crop_source_reference"
    if valid_runs:
        return sorted(valid_runs.keys())[-1], "latest_valid_by_name"
    return None, None


def _build_crop_expected_path(parent_name: str, resolved: ResolvedRefinedRun) -> str:
    return f"{parent_name}/{resolved.run_name}/{resolved.subgroup_name}"


def _is_refined_source_path(path: Optional[str], parent_name: str) -> bool:
    return _parse_refined_source_path(path, parent_name) is not None


def _plan_repairs(
    root: zarr.Group,
    *,
    parent_name: str,
    parent_group: zarr.Group,
    requested_runs: list[str],
    limit: str,
) -> tuple[list[PlannedRepair], list[str]]:
    warnings: list[str] = []
    repairs: list[PlannedRepair] = []

    valid_runs = _resolved_refined_runs(parent_group)
    if not valid_runs:
        warnings.append("no refined runs with valid detection arrays were found")
        return repairs, warnings

    crop_runs = _select_crop_runs(
        root,
        requested_runs=requested_runs,
        limit=limit,
        warnings=warnings,
    )
    crop_preferred_run = _choose_run_from_crop(
        crop_runs,
        parent_name=parent_name,
        valid_runs=valid_runs,
    )
    current_latest = _latest_name(parent_group)
    canonical_run, run_reason = _choose_canonical_run(
        current_latest=current_latest,
        crop_preferred_run=crop_preferred_run,
        valid_runs=valid_runs,
    )
    if canonical_run is None:
        warnings.append("unable to resolve canonical refined run")
        return repairs, warnings

    resolved = valid_runs[canonical_run]
    expected_path = _build_crop_expected_path(parent_name, resolved)
    expected_source_type = resolved.source_type

    if current_latest != canonical_run:
        repairs.append(
            PlannedRepair(
                target_group=parent_group,
                target_path=parent_name,
                field="latest",
                old_value=current_latest,
                new_value=canonical_run,
                reason=f"set canonical refined run ({run_reason})",
            )
        )

    for crop_run_name, crop_group in crop_runs:
        crop_path = f"crop_runs/{crop_run_name}"
        current_path = _normalize_path(crop_group.attrs.get("detection_source_path"))
        current_source_refined = _normalize_str(crop_group.attrs.get("source_refined_run"))
        current_source_type = _normalize_str(crop_group.attrs.get("detection_source_type"))

        references_refined = _is_refined_source_path(current_path, parent_name)
        if current_path is None and current_source_refined:
            references_refined = True

        if references_refined and current_path != expected_path:
            repairs.append(
                PlannedRepair(
                    target_group=crop_group,
                    target_path=crop_path,
                    field="detection_source_path",
                    old_value=current_path,
                    new_value=expected_path,
                    reason="align crop source path with canonical refined run",
                )
            )

        if references_refined and current_source_refined != canonical_run:
            repairs.append(
                PlannedRepair(
                    target_group=crop_group,
                    target_path=crop_path,
                    field="source_refined_run",
                    old_value=current_source_refined,
                    new_value=canonical_run,
                    reason="align source_refined_run with canonical refined run",
                )
            )

        if references_refined and current_source_type != expected_source_type:
            if current_source_type is None or current_source_type in REFINED_SOURCE_TYPES:
                repairs.append(
                    PlannedRepair(
                        target_group=crop_group,
                        target_path=crop_path,
                        field="detection_source_type",
                        old_value=current_source_type,
                        new_value=expected_source_type,
                        reason="align detection_source_type with canonical refined subgroup",
                    )
                )

    return repairs, warnings


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    return repr(_normalize_scalar(value))


def _apply_repairs(repairs: list[PlannedRepair]) -> int:
    applied = 0
    for repair in repairs:
        current = _normalize_scalar(repair.target_group.attrs.get(repair.field))
        desired = _normalize_scalar(repair.new_value)
        if current == desired:
            continue
        repair.target_group.attrs[repair.field] = repair.new_value
        applied += 1
    return applied


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair stale refined-detect lineage pointers by reconciling "
            "refined latest pointers and crop source attrs."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument(
        "--crop-run",
        action="append",
        default=[],
        help="Crop run(s) to repair (default: latest crop run). May be specified multiple times.",
    )
    parser.add_argument(
        "--limit",
        choices=["latest", "all"],
        default="latest",
        help="When --crop-run is not set, repair only latest crop run or all crop runs (default: latest).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes only (default behavior).")
    parser.add_argument("--apply", action="store_true", help="Write repairs to zarr attrs.")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        raise SystemExit("Choose either --apply or --dry-run, not both.")

    apply = bool(args.apply)
    roots = _resolve_roots(list(args.paths))

    scanned = 0
    zarr_with_changes = 0
    planned_total = 0
    applied_total = 0
    errors = 0
    any_zarr = False

    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        try:
            root = zarr.open_group(str(zarr_path), mode="a" if apply else "r")
        except Exception as exc:
            errors += 1
            print(f"error: {zarr_path}: {exc}")
            continue

        if args.zarr_use != "any":
            observed_use = _infer_zarr_use(root, zarr_path)
            if observed_use != args.zarr_use:
                continue

        parent_name, parent_group = _refined_parent(root)
        if parent_name is None or parent_group is None:
            continue

        scanned += 1
        repairs, warnings = _plan_repairs(
            root,
            parent_name=parent_name,
            parent_group=parent_group,
            requested_runs=list(args.crop_run),
            limit=args.limit,
        )
        for warning in warnings:
            print(f"warn: {zarr_path}: {warning}")

        if not repairs:
            continue

        zarr_with_changes += 1
        planned_total += len(repairs)

        mode_label = "apply" if apply else "plan"
        for repair in repairs:
            print(
                f"{mode_label}: {zarr_path}:{repair.target_path} "
                f"{repair.field} {_format_value(repair.old_value)} -> {_format_value(repair.new_value)} "
                f"({repair.reason})"
            )

        if apply:
            applied_total += _apply_repairs(repairs)

    if not any_zarr:
        print("No zarr files found.")
        return 1

    print(f"zarr_scanned: {scanned}")
    print(f"zarr_with_changes: {zarr_with_changes}")
    print(f"planned_repairs: {planned_total}")
    if apply:
        print(f"applied_repairs: {applied_total}")
    if errors:
        print(f"errors: {errors}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
