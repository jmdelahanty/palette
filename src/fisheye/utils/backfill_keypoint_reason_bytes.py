from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


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


def _select_run_names(refined_parent: zarr.Group, all_runs: bool) -> list[str]:
    if all_runs:
        try:
            return sorted(list(refined_parent.group_keys()))
        except Exception:
            return sorted(list(refined_parent.keys()))
    latest = refined_parent.attrs.get("latest")
    if latest and latest in refined_parent:
        return [str(latest)]
    try:
        names = sorted(list(refined_parent.group_keys()))
    except Exception:
        names = sorted(list(refined_parent.keys()))
    if not names:
        return []
    return [names[-1]]


def _compute_chunk_size(group: zarr.Group, labels_count: int) -> int:
    heading = group.get("heading")
    if heading is not None and heading.chunks:
        return int(heading.chunks[0])
    reason = group.get("reason")
    if reason is not None and reason.chunks:
        return int(reason.chunks[0])
    return max(1, min(1024, int(labels_count)))


def _backfill_reason_columns(
    run_group: zarr.Group,
    *,
    overwrite_existing: bool,
    apply: bool,
) -> BackfillResult:
    if "reason_bytes" in run_group and not overwrite_existing:
        return BackfillResult(status="skipped_existing")

    labels = read_reason_labels(run_group)
    if labels is None:
        return BackfillResult(status="no_reason", reason="reason/reason_bytes/detection_source missing")

    labels = np.asarray(labels, dtype=object)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if apply:
        chunk_size = _compute_chunk_size(run_group, int(labels.shape[0]))
        write_reason_columns(
            run_group,
            labels,
            chunk_size,
            include_reason_text=True,
            overwrite=True,
        )
    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill Crimson-compatible keypoint reason_bytes on refined keypoint runs."
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
    parser.add_argument("--overwrite-existing", action="store_true", help="Rewrite reason_bytes even if already present.")
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")

    args = parser.parse_args(argv)
    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]

    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "no_reason": 0,
        "missing_refined": 0,
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
                result = _backfill_reason_columns(
                    refined_parent[run_name],
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
        "Keypoint reason_bytes backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_refined={counts['missing_refined']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_reason={counts['no_reason']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
