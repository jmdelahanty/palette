#!/usr/bin/env python3
"""Backfill crop storage metadata for legacy crop runs.

Default mode is dry-run. Use --apply to write changes.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import zarr

from ..shared.crop_signature import build_crop_signature

from fisheye.cli.shared_args import add_log_args
from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.utils.metadata import get_total_frames
from fisheye.shared.environment import resolve_log_dir
from fisheye.shared.environment import resolve_recording_roots


VALID_STORAGE_MODES = {"materialized", "geometry_only"}
MISSING_STRING_VALUES = {"", "unknown", "n/a", "na", "none", "null"}
CANONICAL_GEOMETRY_ARRAYS = ("roi_coordinates_full", "bbox_norm_coords", "frame_indices")
SIGNATURE_CANONICAL_KEYS = (
    "signature_version",
    "detection_source_path",
    "detection_source_type",
    "detection_preferred_policy",
    "source_detect_run",
    "source_refined_run",
    "roi_size",
    "parameter_source",
    "parameters_hash",
)


@dataclass(frozen=True)
class CropRunPlan:
    display_path: str
    storage_mode_before: Optional[str]
    capability_mode: Optional[str]
    storage_mode_update: Optional[str]
    roi_size_update: Optional[list[int]]
    crop_signature_update: Optional[dict[str, Any]]
    frame_counts_update: Optional[np.ndarray]
    detection_indices_update: Optional[np.ndarray]
    issues: tuple[str, ...]

    @property
    def would_modify(self) -> bool:
        return (
            self.storage_mode_update is not None
            or self.roi_size_update is not None
            or self.crop_signature_update is not None
            or self.frame_counts_update is not None
            or self.detection_indices_update is not None
        )


@dataclass(frozen=True)
class CropParentPlan:
    display_path: str
    latest_update: Optional[str]
    latest_materialized_update: Optional[str]
    latest_any_update: Optional[str]
    issues: tuple[str, ...]

    @property
    def would_modify(self) -> bool:
        return (
            self.latest_update is not None
            or self.latest_materialized_update is not None
            or self.latest_any_update is not None
        )


def _normalize_status(value: Any) -> Optional[str]:
    value = _normalize_scalar(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def _resolve_roots(paths: list[Path]) -> list[Path]:
    return resolve_recording_roots(paths)


JsonLogger = SharedJsonLogger
_run_id = make_run_id


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: list[Path]) -> Path:
    return resolve_log_dir(arg_log_dir, roots, log_subdir="backfill_crop_storage_metadata")


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


def _open_crop_parent(
    root: zarr.Group,
    zarr_path: Path,
    *,
    mode: str,
) -> Optional[zarr.Group]:
    crop_runs_path = zarr_path / "crop_runs"
    if crop_runs_path.exists():
        try:
            return zarr.open_group(str(crop_runs_path), mode=mode)
        except Exception:
            pass
    crop_parent = root.get("crop_runs")
    if crop_parent is None or not hasattr(crop_parent, "attrs"):
        return None
    return crop_parent


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_json_compatible(value: Any) -> Any:
    value = _normalize_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_json_compatible(item) for item in value)
    if hasattr(value, "tolist"):
        try:
            return _to_json_compatible(value.tolist())
        except Exception:
            pass
    return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    value = _normalize_scalar(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, Mapping):
            return {str(key): _to_json_compatible(item) for key, item in payload.items()}
    return {}


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> str:
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
    return "unknown"


def _normalize_storage_mode(value: Any) -> Optional[str]:
    value = _normalize_scalar(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in VALID_STORAGE_MODES else None


def _normalize_roi_size(value: Any) -> Optional[list[int]]:
    value = _normalize_scalar(value)
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            try:
                side = int(text)
            except ValueError:
                return None
            return [side, side] if side > 0 else None
        return _normalize_roi_size(payload)
    if hasattr(value, "tolist"):
        try:
            return _normalize_roi_size(value.tolist())
        except Exception:
            return None
    if isinstance(value, (int, np.integer)):
        side = int(value)
        return [side, side] if side > 0 else None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            dims = [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
        if dims[0] > 0 and dims[1] > 0:
            return dims
    return None


def _build_crop_signature(attrs: Mapping[str, Any]) -> dict[str, Any]:
    canonical = build_crop_signature(dict(attrs))
    return {str(key): _to_json_compatible(value) for key, value in canonical.items()}


def _vector_chunks(length: int, limit: int) -> tuple[int]:
    return (max(1, min(limit, max(1, int(length)))),)


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


def _read_array(group: zarr.Group, name: str) -> Optional[np.ndarray]:
    if name not in group:
        return None
    return np.asarray(group[name][:])


def _infer_n_rois(run_group: zarr.Group) -> Optional[int]:
    for name in ("roi_images", "frame_indices", "bbox_norm_coords", "roi_coordinates_full"):
        if name in run_group:
            shape = getattr(run_group[name], "shape", None)
            if shape:
                return int(shape[0])
    return None


def _resolve_roi_size(run_group: zarr.Group, attrs: Mapping[str, Any]) -> tuple[Optional[list[int]], tuple[str, ...]]:
    issues: list[str] = []
    from_attr = _normalize_roi_size(attrs.get("roi_size"))
    from_signature = _normalize_roi_size(_as_mapping(attrs.get("crop_signature")).get("roi_size"))

    if "roi_images" in run_group:
        shape = tuple(int(dim) for dim in run_group["roi_images"].shape)
        if len(shape) >= 3:
            from_images = [shape[1], shape[2]]
            if from_attr is not None and from_attr != from_images:
                issues.append("roi_size_attr_mismatch_roi_images")
            if from_signature is not None and from_signature != from_images:
                issues.append("roi_size_signature_mismatch_roi_images")
            return from_images, tuple(dict.fromkeys(issues))

    if from_attr is not None:
        if from_signature is not None and from_signature != from_attr:
            issues.append("roi_size_signature_mismatch_attr")
        return from_attr, tuple(dict.fromkeys(issues))

    if from_signature is not None:
        return from_signature, tuple(dict.fromkeys(issues))

    issues.append("missing_roi_size")
    return None, tuple(dict.fromkeys(issues))


def _infer_capability_mode(run_group: zarr.Group, roi_size: Optional[list[int]]) -> Optional[str]:
    if "roi_images" in run_group:
        return "materialized"
    if roi_size is not None and all(name in run_group for name in CANONICAL_GEOMETRY_ARRAYS):
        return "geometry_only"
    return None


def _expected_frame_counts(
    root: zarr.Group,
    run_group: zarr.Group,
) -> tuple[Optional[np.ndarray], tuple[str, ...]]:
    issues: list[str] = []
    frame_indices = _read_array(run_group, "frame_indices")
    if frame_indices is None:
        issues.append("missing_frame_indices")
        return None, tuple(issues)

    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.size and int(frame_indices.min()) < 0:
        issues.append("negative_frame_indices")
        return None, tuple(issues)

    total_frames = get_total_frames(root, run_group)
    max_frame_plus_one = int(frame_indices.max()) + 1 if frame_indices.size else 0
    if total_frames is None:
        total_frames = max_frame_plus_one
        if frame_indices.size:
            issues.append("frame_counts_frame_universe_inferred_from_frame_indices")
    elif total_frames < max_frame_plus_one:
        issues.append("frame_indices_exceed_total_frames")
        total_frames = max_frame_plus_one

    counts = np.bincount(frame_indices, minlength=int(total_frames)).astype(np.int32, copy=False)
    return counts, tuple(dict.fromkeys(issues))


def _expected_detection_indices(run_group: zarr.Group) -> tuple[Optional[np.ndarray], tuple[str, ...]]:
    n_rois = _infer_n_rois(run_group)
    if n_rois is None:
        return None, ("missing_n_rois_for_detection_indices",)
    return np.arange(int(n_rois), dtype=np.int32), ()


def _merge_crop_signature(existing: Mapping[str, Any], attrs: Mapping[str, Any]) -> dict[str, Any]:
    merged = {str(key): _to_json_compatible(value) for key, value in existing.items()}
    canonical = _build_crop_signature(attrs)
    for key in SIGNATURE_CANONICAL_KEYS:
        merged[key] = canonical.get(key)
    return merged


def _run_action(plan: CropRunPlan, *, apply: bool) -> str:
    if plan.would_modify:
        return "updated" if apply else "would_modify"
    return "no_change"


def _parent_action(plan: CropParentPlan, *, apply: bool) -> str:
    if plan.would_modify:
        return "updated" if apply else "would_modify"
    return "no_change"


def _log_crop_run(
    logger: Optional[JsonLogger],
    *,
    zarr_path: Path,
    run_name: str,
    run_status: Optional[str],
    plan: CropRunPlan,
    apply: bool,
) -> None:
    if logger is None:
        return
    logger.log(
        "crop_run_checked",
        zarr=str(zarr_path),
        crop_run=run_name,
        action=_run_action(plan, apply=apply),
        run_status=run_status,
        storage_mode_before=plan.storage_mode_before,
        capability_mode=plan.capability_mode,
        storage_mode_update=plan.storage_mode_update,
        roi_size_update=_to_json_compatible(plan.roi_size_update),
        crop_signature_update=plan.crop_signature_update,
        frame_counts_backfill=bool(plan.frame_counts_update is not None),
        detection_indices_backfill=bool(plan.detection_indices_update is not None),
        issues=list(plan.issues),
    )


def _log_skipped_non_completed(
    logger: Optional[JsonLogger],
    *,
    zarr_path: Path,
    run_name: str,
    run_status: Optional[str],
) -> None:
    if logger is None:
        return
    logger.log(
        "crop_run_skipped_non_completed",
        zarr=str(zarr_path),
        crop_run=run_name,
        run_status=run_status,
    )


def _log_crop_parent(
    logger: Optional[JsonLogger],
    *,
    zarr_path: Path,
    crop_parent: zarr.Group,
    plan: CropParentPlan,
    apply: bool,
) -> None:
    if logger is None:
        return
    logger.log(
        "crop_parent_checked",
        zarr=str(zarr_path),
        action=_parent_action(plan, apply=apply),
        latest_before=_normalize_scalar(crop_parent.attrs.get("latest")),
        latest_materialized_before=_normalize_scalar(crop_parent.attrs.get("latest_materialized")),
        latest_any_before=_normalize_scalar(crop_parent.attrs.get("latest_any")),
        latest_update=plan.latest_update,
        latest_materialized_update=plan.latest_materialized_update,
        latest_any_update=plan.latest_any_update,
        issues=list(plan.issues),
    )


def _log_zarr_summary(
    logger: Optional[JsonLogger],
    *,
    zarr_path: Path,
    scanned_runs: int,
    skipped_non_completed_runs: int,
    run_issue_targets: int,
    parent_issue: bool,
    updated_runs: int,
    updated_parent_groups: int,
    would_modify_runs: int,
    would_modify_parent: bool,
) -> None:
    if logger is None:
        return
    logger.log(
        "zarr_checked",
        zarr=str(zarr_path),
        crop_runs_scanned=int(scanned_runs),
        skipped_non_completed_runs=int(skipped_non_completed_runs),
        issue_targets=int(run_issue_targets + int(parent_issue)),
        run_issue_targets=int(run_issue_targets),
        parent_issue=bool(parent_issue),
        updated_runs=int(updated_runs),
        updated_parent_groups=int(updated_parent_groups),
        would_modify_runs=int(would_modify_runs),
        would_modify_parent=bool(would_modify_parent),
        would_modify=bool(would_modify_runs > 0 or would_modify_parent),
    )


def _build_run_plan(
    zarr_path: Path,
    root: zarr.Group,
    run_name: str,
    run_group: zarr.Group,
) -> CropRunPlan:
    attrs = dict(run_group.attrs)
    issues: list[str] = []
    display_path = f"{zarr_path}:crop_runs/{run_name}"

    storage_mode_before = _normalize_storage_mode(attrs.get("crop_storage_mode"))
    if attrs.get("crop_storage_mode") is not None and storage_mode_before is None:
        issues.append("invalid_crop_storage_mode")

    roi_size_value, roi_issues = _resolve_roi_size(run_group, attrs)
    issues.extend(roi_issues)
    current_roi_size = _normalize_roi_size(attrs.get("roi_size"))
    roi_size_update = None
    if roi_size_value is not None and current_roi_size != roi_size_value:
        roi_size_update = roi_size_value

    capability_mode = _infer_capability_mode(run_group, roi_size_value)
    if storage_mode_before == "materialized" and "roi_images" not in run_group:
        issues.append("materialized_mode_missing_roi_images")
    if storage_mode_before == "geometry_only" and "roi_images" in run_group:
        issues.append("geometry_only_mode_has_roi_images")
    if capability_mode is None:
        issues.append("unable_to_infer_crop_storage_mode")
    storage_mode_update = None
    if capability_mode is not None and storage_mode_before != capability_mode:
        storage_mode_update = capability_mode

    expected_frame_counts, frame_count_issues = _expected_frame_counts(root, run_group)
    issues.extend(frame_count_issues)
    existing_frame_counts = _read_array(run_group, "frame_counts")
    frame_counts_update = None
    if existing_frame_counts is None and expected_frame_counts is not None:
        frame_counts_update = expected_frame_counts
    elif existing_frame_counts is not None and expected_frame_counts is not None:
        if not np.array_equal(np.asarray(existing_frame_counts, dtype=np.int64), expected_frame_counts):
            issues.append("frame_counts_mismatch_existing")

    expected_detection_indices, detection_issues = _expected_detection_indices(run_group)
    issues.extend(detection_issues)
    existing_detection_indices = _read_array(run_group, "detection_indices")
    detection_indices_update = None
    if existing_detection_indices is None and expected_detection_indices is not None:
        detection_indices_update = expected_detection_indices
    elif existing_detection_indices is not None and expected_detection_indices is not None:
        if int(existing_detection_indices.shape[0]) != int(expected_detection_indices.shape[0]):
            issues.append("detection_indices_length_mismatch")

    signature_attrs = dict(attrs)
    if roi_size_value is not None:
        signature_attrs["roi_size"] = roi_size_value
    existing_signature = _as_mapping(attrs.get("crop_signature"))
    merged_signature = _merge_crop_signature(existing_signature, signature_attrs)
    crop_signature_update = None
    if merged_signature != existing_signature:
        crop_signature_update = merged_signature

    return CropRunPlan(
        display_path=display_path,
        storage_mode_before=storage_mode_before,
        capability_mode=capability_mode,
        storage_mode_update=storage_mode_update,
        roi_size_update=roi_size_update,
        crop_signature_update=crop_signature_update,
        frame_counts_update=frame_counts_update,
        detection_indices_update=detection_indices_update,
        issues=tuple(dict.fromkeys(issues)),
    )


def _pick_run_names(parent: zarr.Group) -> list[str]:
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    return [str(name) for name in names if hasattr(parent[name], "attrs")]


def _build_parent_plan(
    zarr_path: Path,
    crop_parent: zarr.Group,
    run_plans: Mapping[str, CropRunPlan],
) -> CropParentPlan:
    issues: list[str] = []
    run_names = _pick_run_names(crop_parent)
    display_path = f"{zarr_path}:crop_runs"

    def _sort_key(name: str) -> tuple[float, str]:
        run_group = crop_parent[name]
        return (
            _parse_iso_utc(
                run_group.attrs.get("created_at_utc")
                or run_group.attrs.get("started_at_utc")
            ),
            name,
        )

    ordered = sorted(run_names, key=_sort_key)
    any_target = ordered[-1] if ordered else None

    materialized_names = [
        name
        for name in ordered
        if run_plans.get(name) is not None and run_plans[name].capability_mode == "materialized"
    ]
    materialized_target = materialized_names[-1] if materialized_names else None

    latest_before = crop_parent.attrs.get("latest")
    latest_materialized_before = crop_parent.attrs.get("latest_materialized")
    latest_any_before = crop_parent.attrs.get("latest_any")

    if latest_before is not None and str(latest_before) not in crop_parent:
        issues.append("latest_points_to_missing_run")
    elif latest_before and materialized_target is not None:
        latest_plan = run_plans.get(str(latest_before))
        if latest_plan is None or latest_plan.capability_mode != "materialized":
            issues.append("latest_not_materialized_compatible")
    elif latest_before and materialized_target is None:
        issues.append("no_materialized_run_for_latest")

    latest_update = None
    if materialized_target is not None and str(latest_before) != materialized_target:
        latest_update = materialized_target

    latest_materialized_update = None
    if materialized_target is not None and str(latest_materialized_before) != materialized_target:
        latest_materialized_update = materialized_target

    latest_any_update = None
    if any_target is not None and str(latest_any_before) != any_target:
        latest_any_update = any_target

    return CropParentPlan(
        display_path=display_path,
        latest_update=latest_update,
        latest_materialized_update=latest_materialized_update,
        latest_any_update=latest_any_update,
        issues=tuple(dict.fromkeys(issues)),
    )


def _apply_run_plan(run_group: zarr.Group, plan: CropRunPlan) -> bool:
    changed = False
    if plan.storage_mode_update is not None and run_group.attrs.get("crop_storage_mode") != plan.storage_mode_update:
        run_group.attrs["crop_storage_mode"] = plan.storage_mode_update
        changed = True
    if plan.roi_size_update is not None and _normalize_roi_size(run_group.attrs.get("roi_size")) != plan.roi_size_update:
        run_group.attrs["roi_size"] = list(plan.roi_size_update)
        changed = True
    if plan.crop_signature_update is not None and _as_mapping(run_group.attrs.get("crop_signature")) != plan.crop_signature_update:
        run_group.attrs["crop_signature"] = plan.crop_signature_update
        changed = True
    if plan.frame_counts_update is not None and "frame_counts" not in run_group:
        run_group.create_array(
            "frame_counts",
            data=plan.frame_counts_update,
            chunks=_vector_chunks(len(plan.frame_counts_update), 10000),
            overwrite=True,
        )
        changed = True
    if plan.detection_indices_update is not None and "detection_indices" not in run_group:
        run_group.create_array(
            "detection_indices",
            data=plan.detection_indices_update,
            chunks=_vector_chunks(len(plan.detection_indices_update), 1000),
            overwrite=True,
        )
        changed = True
    return changed


def _apply_parent_plan(crop_parent: zarr.Group, plan: CropParentPlan) -> bool:
    changed = False
    if plan.latest_update is not None and crop_parent.attrs.get("latest") != plan.latest_update:
        crop_parent.attrs["latest"] = plan.latest_update
        changed = True
    if (
        plan.latest_materialized_update is not None
        and crop_parent.attrs.get("latest_materialized") != plan.latest_materialized_update
    ):
        crop_parent.attrs["latest_materialized"] = plan.latest_materialized_update
        changed = True
    if plan.latest_any_update is not None and crop_parent.attrs.get("latest_any") != plan.latest_any_update:
        crop_parent.attrs["latest_any"] = plan.latest_any_update
        changed = True
    return changed


def _print_summary(
    *,
    zarrs_scanned: int,
    crop_parent_groups_scanned: int,
    crop_runs_scanned: int,
    skipped_non_completed_runs: int,
    storage_mode_updates: int,
    frame_counts_backfills: int,
    detection_indices_backfills: int,
    roi_size_updates: int,
    crop_signature_updates: int,
    parent_pointer_updates: int,
    issue_targets: int,
    would_modify_targets: list[str],
    issue_target_paths: list[str],
    skipped_non_completed_paths: list[str],
    updated_runs: Optional[int] = None,
    updated_parent_groups: Optional[int] = None,
) -> None:
    print(f"zarrs_scanned: {zarrs_scanned}")
    print(f"crop_parent_groups_scanned: {crop_parent_groups_scanned}")
    print(f"crop_runs_scanned: {crop_runs_scanned}")
    print(f"skipped_non_completed_runs: {skipped_non_completed_runs}")
    print(f"storage_mode_updates: {storage_mode_updates}")
    print(f"frame_counts_backfills: {frame_counts_backfills}")
    print(f"detection_indices_backfills: {detection_indices_backfills}")
    print(f"roi_size_updates: {roi_size_updates}")
    print(f"crop_signature_updates: {crop_signature_updates}")
    print(f"parent_pointer_updates: {parent_pointer_updates}")
    print(f"issue_targets: {issue_targets}")
    if updated_runs is not None:
        print(f"updated_runs: {updated_runs}")
    if updated_parent_groups is not None:
        print(f"updated_parent_groups: {updated_parent_groups}")
    print("would_modify_targets_first5:")
    if would_modify_targets:
        for path in would_modify_targets:
            print(f"  {path}")
    else:
        print("  (none)")
    print("issue_targets_first5:")
    if issue_target_paths:
        for path in issue_target_paths:
            print(f"  {path}")
    else:
        print("  (none)")
    print("skipped_non_completed_first5:")
    if skipped_non_completed_paths:
        for path in skipped_non_completed_paths:
            print(f"  {path}")
    else:
        print("  (none)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill crop storage metadata for legacy crop runs by classifying "
            "materialized vs geometry-only capability, filling deterministic "
            "lineage fields, and repairing safe latest pointers."
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
        "--include-non-completed",
        action="store_true",
        help="Include crop runs whose status is explicitly not completed (default skips them).",
    )
    add_log_args(
        parser,
        log_dir_help=(
            "Directory for JSONL logs (default: $PALETTE_LOG_ROOT/backfill_crop_storage_metadata "
            "or <root>/logs/backfill_crop_storage_metadata)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes only (default behavior).")
    parser.add_argument("--apply", action="store_true", help="Write changes to zarr attrs/arrays.")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        raise SystemExit("Choose either --apply or --dry-run, not both.")

    apply = bool(args.apply)
    roots = _resolve_roots(list(args.paths))
    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()

    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"backfill_crop_storage_metadata_{run_id}.jsonl"
            logger = JsonLogger(log_path, run_id)
            print(f"Log file: {log_path}")
        except Exception as exc:
            logger = None
            print(f"Warning: logging disabled ({exc})")

    if logger is not None:
        logger.log(
            "run_start",
            roots=[str(root) for root in roots],
            recursive=bool(args.recursive),
            zarr_use=args.zarr_use,
            apply=bool(args.apply),
            dry_run=not bool(args.apply),
            include_non_completed=bool(args.include_non_completed),
        )

    any_zarr = False
    errors = 0
    zarrs_scanned = 0
    crop_parent_groups_scanned = 0
    crop_runs_scanned = 0
    skipped_non_completed_runs = 0
    storage_mode_updates = 0
    frame_counts_backfills = 0
    detection_indices_backfills = 0
    roi_size_updates = 0
    crop_signature_updates = 0
    parent_pointer_updates = 0
    issue_targets = 0
    updated_runs = 0
    updated_parent_groups = 0
    would_modify_targets_first5: list[str] = []
    issue_targets_first5: list[str] = []
    skipped_non_completed_first5: list[str] = []

    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        try:
            root = zarr.open_group(str(zarr_path), mode="a" if apply else "r")
        except Exception as exc:
            errors += 1
            print(f"error: {zarr_path}: {exc}")
            if logger is not None:
                logger.log("zarr_error", zarr=str(zarr_path), error=str(exc))
            continue

        if args.zarr_use != "any":
            observed_use = _infer_zarr_use(root, zarr_path)
            if observed_use != args.zarr_use:
                if logger is not None:
                    logger.log(
                        "zarr_skipped_use_filter",
                        zarr=str(zarr_path),
                        observed_zarr_use=observed_use,
                        requested_zarr_use=args.zarr_use,
                    )
                continue

        zarrs_scanned += 1
        crop_parent = _open_crop_parent(root, zarr_path, mode="a" if apply else "r")
        if crop_parent is None:
            if logger is not None:
                logger.log("zarr_skipped_no_crop_runs", zarr=str(zarr_path))
            continue
        crop_parent_groups_scanned += 1

        run_names = _pick_run_names(crop_parent)
        run_plans: dict[str, CropRunPlan] = {}
        zarr_scanned_runs = 0
        zarr_skipped_non_completed = 0
        zarr_run_issue_targets = 0
        zarr_updated_runs = 0
        zarr_would_modify_runs = 0
        for run_name in run_names:
            run_group = crop_parent[run_name]
            run_status = _normalize_status(run_group.attrs.get("status"))
            if not args.include_non_completed and run_status is not None and run_status != "completed":
                skipped_non_completed_runs += 1
                zarr_skipped_non_completed += 1
                if len(skipped_non_completed_first5) < 5:
                    skipped_non_completed_first5.append(
                        f"{zarr_path}:crop_runs/{run_name} [status={run_status}]"
                    )
                _log_skipped_non_completed(
                    logger,
                    zarr_path=zarr_path,
                    run_name=run_name,
                    run_status=run_status,
                )
                continue
            plan = _build_run_plan(zarr_path, root, run_name, crop_parent[run_name])
            run_plans[run_name] = plan
            crop_runs_scanned += 1
            zarr_scanned_runs += 1
            storage_mode_updates += int(plan.storage_mode_update is not None)
            frame_counts_backfills += int(plan.frame_counts_update is not None)
            detection_indices_backfills += int(plan.detection_indices_update is not None)
            roi_size_updates += int(plan.roi_size_update is not None)
            crop_signature_updates += int(plan.crop_signature_update is not None)
            if plan.issues:
                issue_targets += 1
                zarr_run_issue_targets += 1
                if len(issue_targets_first5) < 5:
                    issue_targets_first5.append(f"{plan.display_path} [{', '.join(plan.issues)}]")
            if plan.would_modify and len(would_modify_targets_first5) < 5:
                would_modify_targets_first5.append(plan.display_path)
            zarr_would_modify_runs += int(plan.would_modify)
            _log_crop_run(
                logger,
                zarr_path=zarr_path,
                run_name=run_name,
                run_status=run_status,
                plan=plan,
                apply=apply,
            )
            if apply and plan.would_modify and _apply_run_plan(crop_parent[run_name], plan):
                updated_runs += 1
                zarr_updated_runs += 1

        parent_plan = _build_parent_plan(zarr_path, crop_parent, run_plans)
        parent_pointer_updates += int(parent_plan.would_modify)
        if parent_plan.issues:
            issue_targets += 1
            if len(issue_targets_first5) < 5:
                issue_targets_first5.append(f"{parent_plan.display_path} [{', '.join(parent_plan.issues)}]")
        if parent_plan.would_modify and len(would_modify_targets_first5) < 5:
            would_modify_targets_first5.append(parent_plan.display_path)
        _log_crop_parent(
            logger,
            zarr_path=zarr_path,
            crop_parent=crop_parent,
            plan=parent_plan,
            apply=apply,
        )
        if apply and parent_plan.would_modify and _apply_parent_plan(crop_parent, parent_plan):
            updated_parent_groups += 1
        _log_zarr_summary(
            logger,
            zarr_path=zarr_path,
            scanned_runs=zarr_scanned_runs,
            skipped_non_completed_runs=zarr_skipped_non_completed,
            run_issue_targets=zarr_run_issue_targets,
            parent_issue=bool(parent_plan.issues),
            updated_runs=zarr_updated_runs,
            updated_parent_groups=int(apply and parent_plan.would_modify),
            would_modify_runs=zarr_would_modify_runs,
            would_modify_parent=bool(parent_plan.would_modify),
        )

    if not any_zarr:
        print("No zarr files found.")
        if logger is not None:
            logger.log("run_end", status="failed", reason="no_zarr_files_found")
            logger.close()
        return 1

    _print_summary(
        zarrs_scanned=zarrs_scanned,
        crop_parent_groups_scanned=crop_parent_groups_scanned,
        crop_runs_scanned=crop_runs_scanned,
        skipped_non_completed_runs=skipped_non_completed_runs,
        storage_mode_updates=storage_mode_updates,
        frame_counts_backfills=frame_counts_backfills,
        detection_indices_backfills=detection_indices_backfills,
        roi_size_updates=roi_size_updates,
        crop_signature_updates=crop_signature_updates,
        parent_pointer_updates=parent_pointer_updates,
        issue_targets=issue_targets,
        would_modify_targets=would_modify_targets_first5,
        issue_target_paths=issue_targets_first5,
        skipped_non_completed_paths=skipped_non_completed_first5,
        updated_runs=updated_runs if apply else None,
        updated_parent_groups=updated_parent_groups if apply else None,
    )
    if logger is not None:
        logger.log(
            "run_end",
            status="ok" if errors == 0 else "failed",
            zarrs_scanned=zarrs_scanned,
            crop_parent_groups_scanned=crop_parent_groups_scanned,
            crop_runs_scanned=crop_runs_scanned,
            skipped_non_completed_runs=skipped_non_completed_runs,
            storage_mode_updates=storage_mode_updates,
            frame_counts_backfills=frame_counts_backfills,
            detection_indices_backfills=detection_indices_backfills,
            roi_size_updates=roi_size_updates,
            crop_signature_updates=crop_signature_updates,
            parent_pointer_updates=parent_pointer_updates,
            issue_targets=issue_targets,
            updated_runs=updated_runs if apply else 0,
            updated_parent_groups=updated_parent_groups if apply else 0,
            errors=errors,
        )
        logger.close()
    if errors:
        print(f"errors: {errors}")
    if not apply:
        print("Dry-run only. Re-run with --apply to write changes.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
