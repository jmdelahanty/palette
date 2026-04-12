from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

from ..tune.keypoint_review import _update_postprocess_summary


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


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


def _iter_run_groups(root: zarr.Group, all_runs: bool) -> Iterable[tuple[str, str, zarr.Group]]:
    candidates = [
        ("keypoints_runs", "detection_success"),
        ("refined_keypoints_runs", "refined_success"),
        ("keypoints_refined_runs", "refined_success"),
    ]
    for parent_name, success_name in candidates:
        parent = root.get(parent_name)
        if parent is None:
            continue
        if all_runs:
            try:
                run_names = sorted(list(parent.group_keys()))
            except Exception:
                run_names = sorted(list(parent.keys()))
        else:
            latest = parent.attrs.get("latest")
            if latest and latest in parent:
                run_names = [str(latest)]
            else:
                try:
                    names = sorted(list(parent.group_keys()))
                except Exception:
                    names = sorted(list(parent.keys()))
                run_names = [names[-1]] if names else []
        for run_name in run_names:
            if run_name in parent:
                yield f"{parent_name}/{run_name}", success_name, parent[run_name]


def _compute_chunks(run_group: zarr.Group, size: int) -> tuple[int, ...]:
    heading = run_group.get("heading")
    if heading is not None and heading.chunks:
        return heading.chunks
    success = run_group.get("detection_success") or run_group.get("refined_success")
    if success is not None and success.chunks:
        return success.chunks
    return (max(1, min(1024, int(size))),)


def _has_temporal_heading_arrays(run_group: zarr.Group) -> bool:
    return all(
        name in run_group
        for name in ("heading_delta_prev_deg", "heading_delta_next_deg", "heading_temporal_outlier")
    )


def _has_temporal_heading_summary(run_group: zarr.Group) -> bool:
    summary_raw = run_group.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        return False
    postprocess = summary_raw.get("postprocess")
    source = postprocess if isinstance(postprocess, dict) else summary_raw
    return all(
        key in source
        for key in (
            "heading_temporal_evaluable",
            "heading_temporal_outlier",
            "heading_temporal_outlier_rate_percent",
        )
    )


def _temporal_heading_status(run_group: zarr.Group) -> Optional[str]:
    summary_raw = run_group.attrs.get("summary_statistics", {})
    if not isinstance(summary_raw, dict):
        return None
    postprocess = summary_raw.get("postprocess")
    source = postprocess if isinstance(postprocess, dict) else summary_raw
    value = source.get("temporal_heading_status")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _backfill_heading_columns(
    run_group: zarr.Group,
    *,
    root: Optional[zarr.Group],
    success_array_name: str,
    overwrite_existing: bool,
    apply: bool,
) -> BackfillResult:
    heading = run_group.get("heading")
    if heading is None:
        return BackfillResult(status="no_heading", reason="heading array missing")
    success_arr = run_group.get(success_array_name)
    if success_arr is None:
        return BackfillResult(status="no_success", reason=f"{success_array_name} array missing")

    heading_vals = np.asarray(heading[:], dtype=np.float64)
    success_vals = np.asarray(success_arr[:], dtype=bool)
    if heading_vals.shape[0] != success_vals.shape[0]:
        return BackfillResult(
            status="shape_mismatch",
            reason=f"heading len {heading_vals.shape[0]} != {success_array_name} len {success_vals.shape[0]}",
        )

    source_arr = run_group.get("detection_source")
    if source_arr is not None:
        source_vals = np.asarray(source_arr[:], dtype=np.int8)
        if source_vals.shape[0] != heading_vals.shape[0]:
            return BackfillResult(
                status="shape_mismatch",
                reason=f"heading len {heading_vals.shape[0]} != detection_source len {source_vals.shape[0]}",
            )
    else:
        source_vals = np.zeros(heading_vals.shape[0], dtype=np.int8)

    is_refined_run = success_array_name == "refined_success"
    has_heading_finite = "heading_finite" in run_group
    has_heading_usable = "heading_usable" in run_group
    has_legacy = "heading_valid" in run_group
    temporal_status = _temporal_heading_status(run_group) if is_refined_run else None
    has_temporal_arrays = _has_temporal_heading_arrays(run_group) if is_refined_run else True
    has_temporal_summary = _has_temporal_heading_summary(run_group) if is_refined_run else True
    temporal_ready = (
        temporal_status == "disabled_sampled_import"
        if is_refined_run and temporal_status == "disabled_sampled_import"
        else has_temporal_arrays and has_temporal_summary and temporal_status == "enabled"
    )
    if (
        has_heading_finite
        and has_heading_usable
        and not has_legacy
        and temporal_ready
        and not overwrite_existing
    ):
        return BackfillResult(status="skipped_existing")

    heading_finite = np.isfinite(heading_vals)
    heading_usable = success_vals & (source_vals == 0) & heading_finite

    if apply:
        if has_legacy:
            del run_group["heading_valid"]
        if is_refined_run:
            try:
                _update_postprocess_summary(run_group, root=root, print_summary=False)
            except Exception as exc:
                return BackfillResult(status="summary_error", reason=str(exc))
        else:
            chunks = _compute_chunks(run_group, int(heading_vals.shape[0]))
            if has_heading_finite:
                run_group["heading_finite"][:] = heading_finite
            else:
                run_group.create_array(
                    "heading_finite",
                    data=heading_finite,
                    chunks=chunks,
                    overwrite=True,
                )
            if has_heading_usable:
                run_group["heading_usable"][:] = heading_usable
            else:
                run_group.create_array(
                    "heading_usable",
                    data=heading_usable,
                    chunks=chunks,
                    overwrite=True,
                )

    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill keypoint heading fields. Raw runs get heading_finite/heading_usable "
            "and legacy heading_valid removal; refined runs also refresh temporal heading "
            "arrays and postprocess summary statistics."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Backfill all run groups (default: latest only).")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute heading_finite/heading_usable even when both are already present.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")

    args = parser.parse_args(argv)
    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]

    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "no_heading": 0,
        "no_success": 0,
        "shape_mismatch": 0,
        "summary_error": 0,
        "missing_runs": 0,
        "filtered_zarr_use": 0,
        "errors": 0,
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
            run_info = list(_iter_run_groups(root, all_runs=bool(args.all_runs)))
            if not run_info:
                counts["missing_runs"] += 1
                continue
            for _, success_name, run_group in run_info:
                counts["runs_considered"] += 1
                result = _backfill_heading_columns(
                    run_group,
                    root=root,
                    success_array_name=success_name,
                    overwrite_existing=bool(args.overwrite_existing),
                    apply=bool(args.apply),
                )
                counts[result.status] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint heading fields backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_runs={counts['missing_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_heading={counts['no_heading']} no_success={counts['no_success']} "
        f"shape_mismatch={counts['shape_mismatch']} summary_error={counts['summary_error']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
