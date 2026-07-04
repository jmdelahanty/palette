from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zarr

from ..cli.shared_args import add_log_args
from ..shared.batch_logging import JsonLogger as SharedJsonLogger
from ..shared.batch_logging import make_run_id
from ..shared.environment import resolve_log_dir as resolve_shared_log_dir
from ..shared.mask_store import MaskStoreError, open_mask_store
from ..shared.refined_subject_eye_geometry import EYE_COMPONENTS, write_refined_subject_eye_geometry
from ..shared.zarr_helpers import _direct_group_names, _group_names, _open_group_direct, _open_mode, _root_fs_path
from .zarr_io import open_zarr_root


JsonLogger = SharedJsonLogger
_run_id = make_run_id


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None
    roi_count: Optional[int] = None
    ellipse_success_count: Optional[int] = None
    pair_success_count: Optional[int] = None
    geometry_existing: bool = False


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


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: list[Path]) -> Path:
    return resolve_shared_log_dir(arg_log_dir, roots, log_subdir="backfill_refined_subject_eye_geometry")


def _iter_run_groups(
    root: zarr.Group,
    all_runs: bool,
    *,
    zarr_path: Optional[Path] = None,
    open_mode: Optional[str] = None,
) -> Iterable[tuple[str, zarr.Group]]:
    root_fs_path = zarr_path.expanduser().resolve() if zarr_path is not None else _root_fs_path(root)
    resolved_open_mode = open_mode or _open_mode(root)
    parent_name = "refined_subject_masks_runs"
    parent_fs_path = (root_fs_path / parent_name) if root_fs_path is not None else None

    parent = root.get(parent_name)
    if parent is None and parent_fs_path is not None and parent_fs_path.is_dir():
        try:
            parent = _open_group_direct(parent_fs_path, mode=resolved_open_mode)
        except Exception:
            parent = None
    if parent is None:
        return

    available_names = sorted(set(_group_names(parent)) | set(_direct_group_names(parent_fs_path)))
    if all_runs:
        run_names = available_names
    else:
        latest = parent.attrs.get("latest")
        latest_name = str(latest) if latest else ""
        if latest_name and latest_name in available_names:
            run_names = [latest_name]
        else:
            run_names = [available_names[-1]] if available_names else []

    for run_name in run_names:
        direct_path = (parent_fs_path / run_name) if parent_fs_path is not None else None
        if direct_path is not None and run_name in available_names:
            try:
                yield f"{parent_name}/{run_name}", _open_group_direct(direct_path, mode=resolved_open_mode)
                continue
            except Exception:
                if run_name not in parent:
                    continue
        if run_name in parent:
            yield f"{parent_name}/{run_name}", parent[run_name]


def _label_index_map(group: zarr.Group) -> dict[str, int]:
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return {}
    return {str(label): idx for idx, label in enumerate(labels_raw)}


def _available_channels(group: zarr.Group) -> Optional[np.ndarray]:
    available_arr = group.get("available_channels")
    if available_arr is None:
        return None
    try:
        return np.asarray(available_arr[:], dtype=bool).reshape(-1)
    except Exception:
        return np.asarray(available_arr, dtype=bool).reshape(-1)


def _group_source_path(group: zarr.Group) -> str:
    return str(getattr(group, "name", "") or "")


def _path_exists(group: zarr.Group, path: str) -> bool:
    try:
        return group.get(path) is not None
    except Exception:
        try:
            group[path]
            return True
        except Exception:
            return False


def _has_canonical_geometry(group: zarr.Group, roi_count: int) -> bool:
    for component in EYE_COMPONENTS:
        geometry_path = f"components/{component}/geometry"
        if not _path_exists(group, f"{geometry_path}/ellipse_params"):
            return False
        if not _path_exists(group, f"{geometry_path}/ellipse_success"):
            return False
        try:
            if tuple(group[f"{geometry_path}/ellipse_params"].shape) != (roi_count, 5):
                return False
            if tuple(group[f"{geometry_path}/ellipse_success"].shape) != (roi_count,):
                return False
        except Exception:
            return False
        if not _path_exists(group, f"components/{component}/contours/ptr"):
            return False
        if not _path_exists(group, f"components/{component}/contours/len"):
            return False
        if not _path_exists(group, f"components/{component}/contours/points_xy"):
            return False
    if not _path_exists(group, "relations/eye_pair/metrics/separation_px"):
        return False
    if not _path_exists(group, "relations/eye_pair/metrics/separation_valid"):
        return False
    try:
        if tuple(group["relations/eye_pair/metrics/separation_px"].shape) != (roi_count,):
            return False
        if tuple(group["relations/eye_pair/metrics/separation_valid"].shape) != (roi_count,):
            return False
    except Exception:
        return False
    return True


def _inspect_run_group(run_group: zarr.Group) -> BackfillResult:
    label_map = _label_index_map(run_group)
    missing_components = [component for component in EYE_COMPONENTS if component not in label_map]
    if missing_components:
        return BackfillResult(status="no_lr_eyes", reason="missing eye_left/eye_right mask labels")

    try:
        mask_store = open_mask_store(run_group, source_path=_group_source_path(run_group), prefer="dense")
    except MaskStoreError as exc:
        return BackfillResult(status="missing_masks_roi", reason=str(exc))

    roi_count = int(mask_store.n_rows)
    channel_count = int(mask_store.n_channels)
    eye_indices = [int(label_map[component]) for component in EYE_COMPONENTS]
    if any(index < 0 or index >= channel_count for index in eye_indices):
        return BackfillResult(
            status="shape_mismatch",
            reason=f"eye label indices {eye_indices} outside masks_roi channel count {channel_count}",
            roi_count=roi_count,
        )

    available = _available_channels(run_group)
    if available is not None:
        unavailable = [
            component
            for component, index in zip(EYE_COMPONENTS, eye_indices)
            if index >= int(available.shape[0]) or not bool(available[index])
        ]
        if unavailable:
            return BackfillResult(
                status="unavailable_eyes",
                reason=f"available_channels marks unavailable components: {', '.join(unavailable)}",
                roi_count=roi_count,
                geometry_existing=_has_canonical_geometry(run_group, roi_count),
            )

    return BackfillResult(status="ok", roi_count=roi_count, geometry_existing=_has_canonical_geometry(run_group, roi_count))


def _backfill_run_group(run_group: zarr.Group, *, apply: bool) -> BackfillResult:
    inspected = _inspect_run_group(run_group)
    if inspected.status != "ok" or not apply:
        return inspected

    summary = write_refined_subject_eye_geometry(run_group)
    if summary.get("status") != "updated":
        return BackfillResult(
            status="writer_skipped",
            reason=str(summary.get("reason") or "writer did not update"),
            roi_count=inspected.roi_count,
            geometry_existing=inspected.geometry_existing,
        )

    return BackfillResult(
        status="ok",
        roi_count=int(summary.get("roi_count", inspected.roi_count or 0)),
        ellipse_success_count=int(summary.get("ellipse_success_count", 0)),
        pair_success_count=int(summary.get("pair_success_count", 0)),
        geometry_existing=inspected.geometry_existing,
    )


def _result_log_fields(result: BackfillResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "reason": result.reason,
        "roi_count": result.roi_count,
        "ellipse_success_count": result.ellipse_success_count,
        "pair_success_count": result.pair_success_count,
        "geometry_existing": result.geometry_existing,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill canonical eye geometry arrays on refined_subject_masks_runs from stored LR eye masks. "
            "Dry-run is the default; use --apply to rewrite the geometry and eye-pair relation surfaces."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by purpose (default: any).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Backfill all run groups (default: latest only).")
    add_log_args(
        parser,
        log_dir_help=(
            "Directory for JSONL logs (default: $PALETTE_LOG_ROOT/backfill_refined_subject_eye_geometry "
            "or <root>/logs/backfill_refined_subject_eye_geometry)."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")

    args = parser.parse_args(argv)
    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    logger: Optional[JsonLogger] = None
    run_id = _run_id()

    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"backfill_refined_subject_eye_geometry_{run_id}.jsonl"
            logger = JsonLogger(log_path, run_id)
            print(f"Log file: {log_path}")
        except Exception as exc:
            logger = None
            print(f"Warning: logging disabled ({exc})")

    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "no_lr_eyes": 0,
        "unavailable_eyes": 0,
        "missing_masks_roi": 0,
        "shape_mismatch": 0,
        "writer_skipped": 0,
        "missing_runs": 0,
        "filtered_zarr_use": 0,
        "errors": 0,
    }

    if logger is not None:
        logger.log(
            "run_start",
            roots=[str(root) for root in roots],
            recursive=bool(args.recursive),
            zarr_use=str(args.zarr_use),
            all_runs=bool(args.all_runs),
            apply=bool(args.apply),
            dry_run=not bool(args.apply),
        )

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            root = open_zarr_root(zarr_path, mode="a" if args.apply else "r")
            if args.zarr_use != "any":
                observed_use = _infer_zarr_use(root, zarr_path)
                if observed_use != args.zarr_use:
                    counts["filtered_zarr_use"] += 1
                    if logger is not None:
                        logger.log(
                            "zarr_skipped_use_filter",
                            zarr=str(zarr_path),
                            observed_zarr_use=observed_use,
                            requested_zarr_use=str(args.zarr_use),
                        )
                    continue
            else:
                observed_use = _infer_zarr_use(root, zarr_path)

            run_info = list(
                _iter_run_groups(
                    root,
                    all_runs=bool(args.all_runs),
                    zarr_path=zarr_path,
                    open_mode="a" if args.apply else "r",
                )
            )
            if not run_info:
                counts["missing_runs"] += 1
                if logger is not None:
                    logger.log("zarr_missing_runs", zarr=str(zarr_path))
                continue

            for run_path, run_group in run_info:
                counts["runs_considered"] += 1
                result = _backfill_run_group(run_group, apply=bool(args.apply))
                counts[result.status] += 1
                if logger is not None:
                    logger.log(
                        "run_group_checked",
                        zarr=str(zarr_path),
                        zarr_use=observed_use,
                        run_path=run_path,
                        apply=bool(args.apply),
                        dry_run=not bool(args.apply),
                        **_result_log_fields(result),
                    )
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")
            if logger is not None:
                logger.log("zarr_error", zarr=str(zarr_path), error=str(exc))

    if not any_zarr:
        print("No zarr files found.")
        if logger is not None:
            logger.log("run_end", status="no_zarr_found", exit_code=1)
            logger.close()
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Refined subject eye-geometry backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_runs={counts['missing_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} no_lr_eyes={counts['no_lr_eyes']} "
        f"unavailable_eyes={counts['unavailable_eyes']} missing_masks_roi={counts['missing_masks_roi']} "
        f"shape_mismatch={counts['shape_mismatch']} writer_skipped={counts['writer_skipped']}"
    )

    exit_code = 0 if counts["errors"] == 0 else 1
    if logger is not None:
        logger.log(
            "run_end",
            status="ok" if exit_code == 0 else "error",
            mode="apply" if args.apply else "dry-run",
            exit_code=exit_code,
            **counts,
        )
        logger.close()
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
