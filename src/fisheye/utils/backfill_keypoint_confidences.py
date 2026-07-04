from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


def _select_keypoint_runs(root: zarr.Group, all_runs: bool) -> list[zarr.Group]:
    parent = root.get("keypoints_runs")
    if parent is None:
        return []
    if all_runs:
        try:
            names = sorted(list(parent.group_keys()))
        except Exception:
            names = sorted(list(parent.keys()))
    else:
        latest = parent.attrs.get("latest")
        if latest and latest in parent:
            names = [str(latest)]
        else:
            try:
                all_names = sorted(list(parent.group_keys()))
            except Exception:
                all_names = sorted(list(parent.keys()))
            names = [all_names[-1]] if all_names else []
    return [parent[name] for name in names if name in parent]


def _infer_chunk_len(run_group: zarr.Group, length: int) -> int:
    source = run_group.get("keypoints_roi")
    if source is not None and source.chunks:
        return int(source.chunks[0])
    source = run_group.get("detection_success")
    if source is not None and source.chunks:
        return int(source.chunks[0])
    return max(1, min(1024, int(length)))


def _backfill_run_group(
    run_group: zarr.Group,
    *,
    confidence_value: float,
    overwrite_existing: bool,
    apply: bool,
) -> BackfillResult:
    success_arr = run_group.get("detection_success")
    if success_arr is None:
        return BackfillResult(status="no_success", reason="detection_success array missing")

    success_vals = np.asarray(success_arr[:], dtype=bool)
    length = int(success_vals.shape[0])
    if length <= 0:
        return BackfillResult(status="no_rows", reason="detection_success is empty")

    existing = run_group.get("keypoint_confidences")
    if existing is not None and not overwrite_existing:
        return BackfillResult(status="skipped_existing")

    conf_values = np.full((length, 3), np.nan, dtype=np.float64)
    conf_values[success_vals, :] = float(confidence_value)

    if apply:
        chunk_len = _infer_chunk_len(run_group, length)
        if existing is not None:
            if existing.shape != conf_values.shape:
                return BackfillResult(
                    status="shape_mismatch",
                    reason=f"existing shape {existing.shape} != expected {conf_values.shape}",
                )
            existing[:] = conf_values
        else:
            run_group.create_array(
                "keypoint_confidences",
                data=conf_values,
                chunks=(chunk_len, 3),
                overwrite=True,
            )
        run_group.attrs["keypoint_confidence_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill keypoint_confidences for YOLO keypoint runs that do not "
            "persist per-keypoint confidences."
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
    parser.add_argument("--all-runs", action="store_true", help="Backfill all keypoint runs (default: latest only).")
    parser.add_argument(
        "--confidence-value",
        type=float,
        default=0.95,
        help="Confidence value to assign for rows with detection_success=true (default: 0.95).",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rewrite keypoint_confidences for runs where the array already exists.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")

    args = parser.parse_args(argv)
    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]

    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "no_success": 0,
        "no_rows": 0,
        "shape_mismatch": 0,
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
            run_groups = _select_keypoint_runs(root, all_runs=bool(args.all_runs))
            if not run_groups:
                counts["missing_runs"] += 1
                continue
            for run_group in run_groups:
                counts["runs_considered"] += 1
                result = _backfill_run_group(
                    run_group,
                    confidence_value=float(args.confidence_value),
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
        "Keypoint confidence backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_runs={counts['missing_runs']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_success={counts['no_success']} no_rows={counts['no_rows']} "
        f"shape_mismatch={counts['shape_mismatch']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
