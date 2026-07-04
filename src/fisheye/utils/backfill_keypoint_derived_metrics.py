from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

from fisheye.pose.metric_schema import (
    ensure_derived_metric_storage,
    resolve_metric_schema_for_group,
    update_derived_metric_rows,
)
from fisheye.shared.type_conversions import normalize_attr as _as_text


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None
    finalized_migration: bool = False


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
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


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def _select_run_names(refined_parent: zarr.Group, all_runs: bool) -> list[str]:
    if all_runs:
        try:
            return sorted(list(refined_parent.group_keys()))
        except Exception:
            return sorted(list(refined_parent.keys()))
    latest = refined_parent.attrs.get("latest")
    if latest and str(latest) in refined_parent:
        return [str(latest)]
    try:
        names = sorted(list(refined_parent.group_keys()))
    except Exception:
        names = sorted(list(refined_parent.keys()))
    if not names:
        return []
    return [names[-1]]


def _resolve_roi_diagonal(root: zarr.Group, source_crop_run: Optional[str]) -> Optional[float]:
    if not source_crop_run:
        return None
    crop_group = root.get(f"crop_runs/{source_crop_run}")
    if crop_group is None or "roi_images" not in crop_group:
        return None
    try:
        shape = crop_group["roi_images"].shape
        roi_h = float(shape[1])
        roi_w = float(shape[2])
    except Exception:
        return None
    diagonal = float(np.hypot(roi_h, roi_w))
    return diagonal if np.isfinite(diagonal) and diagonal > 0 else None


def _chunk_len(run_group: zarr.Group, rows: int) -> int:
    kp = run_group.get("keypoints_roi")
    if kp is not None and kp.chunks:
        return int(kp.chunks[0])
    refined_success = run_group.get("refined_success")
    if refined_success is not None and refined_success.chunks:
        return int(refined_success.chunks[0])
    return max(1, min(1024, rows))


def _finalize_migration_if_complete(run_group: zarr.Group, keypoint_labels: list[str]) -> bool:
    status = _as_text(run_group.attrs.get("migration_status"))
    if status != "needs_keypoint_completion":
        return False
    required = run_group.attrs.get("migration_completion_required_keypoints")
    if not isinstance(required, (list, tuple)) or not required:
        return False
    label_map = {str(label): int(idx) for idx, label in enumerate(keypoint_labels)}
    required_indices: list[int] = []
    for label in required:
        idx = label_map.get(str(label))
        if idx is None:
            return False
        required_indices.append(idx)
    points = np.asarray(run_group["keypoints_roi"][:], dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 2:
        return False
    missing_mask = ~np.isfinite(points[:, required_indices, :]).all(axis=2)
    missing_row_count = int(np.any(missing_mask, axis=1).sum())
    run_group.attrs["migration_missing_row_count"] = missing_row_count
    if missing_row_count > 0:
        return False
    run_group.attrs["migration_status"] = "completed"
    run_group.attrs["migration_completion_required_keypoints"] = []
    run_group.attrs["migration_completed_at_utc"] = _utc_now()
    return True


def _backfill_run_group(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    overwrite_existing: bool,
    apply: bool,
    finalize_migration: bool,
) -> BackfillResult:
    if "keypoints_roi" not in run_group:
        return BackfillResult(status="no_keypoints", reason="keypoints_roi missing")
    keypoints = run_group["keypoints_roi"]
    if keypoints.ndim != 3 or keypoints.shape[2] != 2:
        return BackfillResult(status="bad_shape", reason=f"keypoints_roi has shape {keypoints.shape}")
    rows = int(keypoints.shape[0])
    n_keypoints = int(keypoints.shape[1])
    if rows <= 0:
        return BackfillResult(status="no_rows", reason="keypoints_roi empty")

    labels_raw = run_group.attrs.get("keypoint_labels")
    if not isinstance(labels_raw, (list, tuple)) or len(labels_raw) != n_keypoints:
        return BackfillResult(status="missing_labels", reason="keypoint_labels missing or mismatched")
    keypoint_labels = [str(label) for label in labels_raw]

    schema = resolve_metric_schema_for_group(run_group, required=False)
    if schema is None:
        return BackfillResult(status="no_schema", reason="no metric schema resolved from pose_schema")

    has_all_arrays = all(name in run_group for name in ("derived_metric_values", "derived_metric_values_norm", "derived_metric_valid"))
    if has_all_arrays and not overwrite_existing:
        if apply and finalize_migration:
            finalized = _finalize_migration_if_complete(run_group, keypoint_labels)
            return BackfillResult(status="skipped_existing", finalized_migration=finalized)
        return BackfillResult(status="skipped_existing")

    refined_success = (
        np.asarray(run_group["refined_success"][:], dtype=bool)
        if "refined_success" in run_group
        else np.ones((rows,), dtype=bool)
    )
    if refined_success.shape[0] != rows:
        return BackfillResult(
            status="shape_mismatch",
            reason=f"refined_success length {refined_success.shape[0]} != keypoints rows {rows}",
        )

    source_crop_run = _as_text(run_group.attrs.get("source_crop_run"))
    roi_diagonal = _resolve_roi_diagonal(root, source_crop_run)
    keypoints_roi = np.asarray(keypoints[:], dtype=np.float64)
    keypoints_roi[~refined_success] = np.nan

    if apply:
        storage = ensure_derived_metric_storage(
            run_group,
            schema=schema,
            row_count=rows,
            chunk_len=_chunk_len(run_group, rows),
            roi_diagonal=roi_diagonal,
            overwrite=True,
        )
        update_derived_metric_rows(
            storage,
            row_indexer=slice(None),
            keypoints_roi=keypoints_roi,
            keypoint_labels=keypoint_labels,
            roi_diagonal=roi_diagonal,
        )
        finalized = False
        if finalize_migration:
            finalized = _finalize_migration_if_complete(run_group, keypoint_labels)
        return BackfillResult(status="ok", finalized_migration=finalized)
    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill schema-driven derived keypoint metrics on refined keypoint runs."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Backfill all refined keypoint runs (default: latest).")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rewrite derived-metric arrays for runs where they already exist.",
    )
    parser.add_argument(
        "--finalize-migration",
        action="store_true",
        help="Mark migration_status completed when required new keypoints are fully populated.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")
    args = parser.parse_args(argv)

    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "no_keypoints": 0,
        "bad_shape": 0,
        "no_rows": 0,
        "missing_labels": 0,
        "no_schema": 0,
        "shape_mismatch": 0,
        "missing_refined": 0,
        "filtered_zarr_use": 0,
        "errors": 0,
        "migrations_finalized": 0,
    }

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="a" if args.apply else "r")
            if args.zarr_use != "any":
                observed_use = _infer_zarr_use(root, zarr_path)
                if observed_use != args.zarr_use:
                    counts["filtered_zarr_use"] += 1
                    continue
            refined_parent = _pick_refined_parent(root)
            if refined_parent is None:
                counts["missing_refined"] += 1
                continue
            run_names = _select_run_names(refined_parent, all_runs=bool(args.all_runs))
            if not run_names:
                counts["missing_refined"] += 1
                continue
            for run_name in run_names:
                counts["runs_considered"] += 1
                result = _backfill_run_group(
                    root,
                    refined_parent[run_name],
                    overwrite_existing=bool(args.overwrite_existing),
                    apply=bool(args.apply),
                    finalize_migration=bool(args.finalize_migration),
                )
                counts[result.status] += 1
                if result.finalized_migration:
                    counts["migrations_finalized"] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint derived-metric backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_refined={counts['missing_refined']} errors={counts['errors']}"
    )
    print(
        f"{mode} ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_schema={counts['no_schema']} missing_labels={counts['missing_labels']} "
        f"shape_mismatch={counts['shape_mismatch']} migrations_finalized={counts['migrations_finalized']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
