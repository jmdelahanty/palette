from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import zarr

from fisheye.pose.schema import resolve_head_triangle_indices, resolve_keypoint_labels_from_attrs
from fisheye.shared.derived_metrics_schema import build_refined_keypoint_derived_metrics_schema
from fisheye.utils.zarr_io import open_zarr_root


REQUIRED_REFINED_KEYPOINT_ARRAYS = (
    "keypoints_roi",
    "triangle_area",
    "triangle_angles",
    "min_angle",
    "geometry_valid",
)


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.suffix == ".zarr" and root.exists():
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


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def _group_names(parent: zarr.Group) -> list[str]:
    try:
        return sorted(str(name) for name in parent.group_keys())
    except Exception:
        return sorted(str(name) for name in parent.keys())


def _select_run_names(refined_parent: zarr.Group, all_runs: bool) -> list[str]:
    names = _group_names(refined_parent)
    if all_runs:
        return names
    latest = refined_parent.attrs.get("latest")
    if latest and str(latest) in refined_parent:
        return [str(latest)]
    return [names[-1]] if names else []


def _missing_required_arrays(run_group: zarr.Group) -> list[str]:
    return [name for name in REQUIRED_REFINED_KEYPOINT_ARRAYS if name not in run_group]


def _resolve_keypoint_labels(run_group: zarr.Group) -> tuple[str, ...]:
    keypoints = run_group["keypoints_roi"]
    shape = tuple(int(v) for v in getattr(keypoints, "shape", ()))
    if len(shape) != 3 or shape[2] != 2:
        raise ValueError(f"keypoints_roi has shape {shape}; expected (N, K, 2).")
    return resolve_keypoint_labels_from_attrs(run_group.attrs, keypoint_count=shape[1])


def _backfill_run_group(
    run_group: zarr.Group,
    *,
    overwrite_existing: bool,
    apply: bool,
) -> BackfillResult:
    missing = _missing_required_arrays(run_group)
    if missing:
        return BackfillResult(status="missing_arrays", reason=", ".join(missing))

    existing = run_group.attrs.get("derived_metrics_schema")
    if existing is not None and not overwrite_existing:
        return BackfillResult(status="skipped_existing")

    try:
        keypoint_labels = _resolve_keypoint_labels(run_group)
    except ValueError as exc:
        return BackfillResult(status="label_mismatch", reason=str(exc))
    if not keypoint_labels:
        return BackfillResult(status="missing_labels", reason="keypoint_labels or pose_schema labels are required")

    try:
        resolve_head_triangle_indices(
            keypoint_labels,
            keypoint_count=len(keypoint_labels),
            allow_legacy_3point_fallback=False,
        )
        schema = build_refined_keypoint_derived_metrics_schema(keypoint_labels=keypoint_labels)
    except ValueError as exc:
        return BackfillResult(status="unsupported_labels", reason=str(exc))

    if apply:
        run_group.attrs["derived_metrics_schema"] = schema
        run_group.attrs["derived_metrics_schema_backfilled_at_utc"] = _utc_now()
    return BackfillResult(status="ok")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill attrs['derived_metrics_schema'] on existing refined keypoint runs. "
            "This is metadata-only and does not recompute predictions or derived arrays."
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
    parser.add_argument("--all-runs", action="store_true", help="Backfill all refined keypoint runs (default: latest).")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rewrite derived_metrics_schema when it is already present.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")
    args = parser.parse_args(argv)

    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "missing_arrays": 0,
        "missing_labels": 0,
        "label_mismatch": 0,
        "unsupported_labels": 0,
        "missing_refined": 0,
        "filtered_zarr_use": 0,
        "errors": 0,
    }

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
                    refined_parent[run_name],
                    overwrite_existing=bool(args.overwrite_existing),
                    apply=bool(args.apply),
                )
                counts[result.status] += 1
                if result.reason:
                    print(f"{result.status}: {zarr_path}: refined_keypoints_runs/{run_name}: {result.reason}")
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint derived-metrics-schema backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_refined={counts['missing_refined']} errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"missing_arrays={counts['missing_arrays']} missing_labels={counts['missing_labels']} "
        f"label_mismatch={counts['label_mismatch']} unsupported_labels={counts['unsupported_labels']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
