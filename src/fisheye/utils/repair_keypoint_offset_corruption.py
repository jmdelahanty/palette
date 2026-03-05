#!/usr/bin/env python3
"""Repair offset-corrupted keypoints rows in a palette/crimson zarr keypoint run.

Dry-run by default. Use --apply to write changes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr


@dataclass
class RunSelection:
    run_group_name: str
    run_name: str
    run: Any


def _is_group(obj: Any) -> bool:
    return hasattr(obj, "group_keys") or hasattr(obj, "keys")


def _group_keys(group: Any) -> List[str]:
    if hasattr(group, "group_keys"):
        return list(group.group_keys())
    out: List[str] = []
    try:
        for k in group.keys():
            try:
                if _is_group(group[k]):
                    out.append(k)
            except Exception:
                continue
    except Exception:
        return []
    return out


def _pick_latest_run(group: Any) -> Optional[str]:
    latest = None
    try:
        latest = group.attrs.get("latest")
    except Exception:
        latest = None
    if isinstance(latest, str) and latest:
        return latest
    names = sorted(_group_keys(group))
    return names[-1] if names else None


def _pick_keypoint_run(root: Any, requested_run: Optional[str]) -> RunSelection:
    checked_groups: List[str] = []
    for group_name in ("refined_keypoints_runs", "keypoints_runs"):
        if group_name not in root:
            continue
        checked_groups.append(group_name)
        run_group = root[group_name]
        if requested_run:
            if requested_run in run_group:
                return RunSelection(group_name, requested_run, run_group[requested_run])
            continue
        run_name = _pick_latest_run(run_group)
        if run_name and run_name in run_group:
            return RunSelection(group_name, run_name, run_group[run_name])
    if requested_run:
        if checked_groups:
            raise RuntimeError(
                f"Requested run '{requested_run}' not found in any of: "
                + ", ".join(checked_groups)
            )
        raise RuntimeError(
            "No keypoint run groups found under 'refined_keypoints_runs' or 'keypoints_runs'."
        )
    raise RuntimeError(
        "Could not locate a keypoint run under 'refined_keypoints_runs' or 'keypoints_runs'."
    )


def _normalize_crop_name(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("crop_runs/") else name


def _load_array(group: Any, name: str) -> Optional[np.ndarray]:
    if name not in group:
        return None
    return np.asarray(group[name][:])


def _infer_roi_size(crop_group: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        roi_size = crop_group.attrs.get("roi_size")
        if (
            isinstance(roi_size, (list, tuple))
            and len(roi_size) >= 2
            and np.isfinite(float(roi_size[0]))
            and np.isfinite(float(roi_size[1]))
        ):
            h = float(roi_size[0])
            w = float(roi_size[1])
            if h > 0 and w > 0:
                return h, w
    except Exception:
        pass

    for candidate in ("roi_images", "images", "crop_images"):
        if candidate not in crop_group:
            continue
        try:
            shape = crop_group[candidate].shape
            if len(shape) >= 3 and shape[1] > 0 and shape[2] > 0:
                return float(shape[1]), float(shape[2])
        except Exception:
            continue
    return None, None


def _load_crop_offsets(
    root: Any, run_attrs: Dict[str, Any], requested_crop_run: Optional[str], roi_count: int
) -> Tuple[str, np.ndarray, float, float]:
    if "crop_runs" not in root:
        raise RuntimeError("No crop_runs group found; cannot repair without ROI offsets.")
    crop_runs = root["crop_runs"]

    candidates: List[str] = []
    if requested_crop_run:
        candidates.append(_normalize_crop_name(requested_crop_run))
    else:
        source_crop = run_attrs.get("source_crop_run")
        if isinstance(source_crop, str) and source_crop:
            candidates.append(_normalize_crop_name(source_crop))
    latest = _pick_latest_run(crop_runs)
    if latest:
        candidates.append(latest)
    for name in _group_keys(crop_runs):
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        if name not in crop_runs:
            continue
        grp = crop_runs[name]
        arr = _load_array(grp, "roi_coordinates_full")
        if arr is None or arr.ndim < 2 or arr.shape[0] != roi_count or arr.shape[1] < 2:
            continue
        offsets = arr[:, :2].astype(np.float64, copy=False)
        roi_h, roi_w = _infer_roi_size(grp)
        if roi_h is None or roi_w is None:
            raise RuntimeError(
                f"Could not infer ROI size for crop run '{name}' (need roi_size or roi_images)."
            )
        return name, offsets, roi_h, roi_w

    raise RuntimeError("Could not find a compatible crop run with roi_coordinates_full.")


def _rows_to_ranges(rows: Iterable[int]) -> List[List[int]]:
    seq = sorted(set(int(x) for x in rows))
    if not seq:
        return []
    out: List[List[int]] = []
    start = prev = seq[0]
    for x in seq[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append([start, prev])
        start = prev = x
    out.append([start, prev])
    return out


def _frame_ranges(frame_indices: np.ndarray, rows: List[int]) -> List[List[int]]:
    return _rows_to_ranges(int(frame_indices[r]) for r in rows)


def _in_roi_bounds_ratio(points: np.ndarray, roi_w: float, roi_h: float, pad_px: float) -> float:
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    if not np.any(finite):
        return 0.0
    p = points[finite]
    in_bounds = (
        (p[:, 0] >= -pad_px)
        & (p[:, 0] <= roi_w + pad_px)
        & (p[:, 1] >= -pad_px)
        & (p[:, 1] <= roi_h + pad_px)
    )
    return float(in_bounds.mean())


def _median_norm(v: np.ndarray) -> float:
    if v.size == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(v, axis=1)))


def repair(
    zarr_path: str,
    run_name: Optional[str],
    crop_run: Optional[str],
    pad_px: float,
    min_fixed_ratio: float,
    max_n: int,
    apply_changes: bool,
) -> Dict[str, Any]:
    if max_n < 1:
        raise ValueError("--max-n must be >= 1")

    mode = "r+" if apply_changes else "r"
    root = zarr.open(zarr_path, mode=mode)
    run_sel = _pick_keypoint_run(root, run_name)
    run = run_sel.run
    run_attrs = dict(getattr(run, "attrs", {}))

    frame_indices = _load_array(run, "frame_indices")
    keypoints_img = _load_array(run, "keypoints_img")
    keypoints_roi = _load_array(run, "keypoints_roi")
    keypoints_norm = _load_array(run, "keypoints_norm")

    if frame_indices is None:
        raise RuntimeError("Run missing frame_indices.")
    if keypoints_img is None or keypoints_roi is None:
        raise RuntimeError("Run must contain both keypoints_img and keypoints_roi.")

    frame_indices = frame_indices.astype(np.int64, copy=False)
    roi_count = int(frame_indices.shape[0])
    if keypoints_img.shape[0] != roi_count or keypoints_roi.shape[0] != roi_count:
        raise RuntimeError("Keypoint arrays are not aligned to frame_indices row count.")
    if keypoints_img.ndim != 3 or keypoints_roi.ndim != 3:
        raise RuntimeError("Expected keypoints_img/keypoints_roi rank-3 arrays.")

    crop_run_name, offsets, roi_h, roi_w = _load_crop_offsets(
        root, run_attrs, crop_run, roi_count
    )

    # Work in float64 for stable math; cast back when writing.
    img = keypoints_img[:, :, :2].astype(np.float64, copy=False)
    roi = keypoints_roi[:, :, :2].astype(np.float64, copy=False)

    proposed_rows: List[int] = []
    proposed_by_n: Counter[int] = Counter()
    preview: List[Dict[str, Any]] = []

    # Copy for edit buffers.
    roi_fixed = np.array(roi, copy=True)
    img_fixed = np.array(img, copy=True)
    norm_fixed: Optional[np.ndarray] = None
    if keypoints_norm is not None:
        if keypoints_norm.shape[0] != roi_count or keypoints_norm.ndim != 3:
            raise RuntimeError("keypoints_norm shape mismatch.")
        norm_fixed = np.array(keypoints_norm[:, :, :2].astype(np.float64, copy=False), copy=True)

    for row in range(roi_count):
        off = offsets[row]
        if not np.all(np.isfinite(off)):
            continue

        roi_row = roi[row]
        img_row = img[row]

        valid = (np.isfinite(roi_row[:, 0]) & np.isfinite(roi_row[:, 1]) &
                 np.isfinite(img_row[:, 0]) & np.isfinite(img_row[:, 1]))
        if not np.any(valid):
            continue

        roi_valid = roi_row[valid]
        img_valid = img_row[valid]
        current_ratio = _in_roi_bounds_ratio(roi_valid, roi_w, roi_h, pad_px)

        # If already mostly valid ROI-local, keep as-is.
        if current_ratio >= min_fixed_ratio:
            continue

        # Only consider rows where img ~= roi + offset (internal consistency pattern).
        internal_err = _median_norm(img_valid - (roi_valid + off[None, :]))
        if not np.isfinite(internal_err) or internal_err > 2.0:
            continue

        best_n = 0
        best_ratio = current_ratio
        best_candidate = roi_valid
        for n in range(1, max_n + 1):
            candidate = roi_valid - n * off[None, :]
            ratio = _in_roi_bounds_ratio(candidate, roi_w, roi_h, pad_px)
            if ratio > best_ratio + 1e-6:
                best_ratio = ratio
                best_n = n
                best_candidate = candidate

        if best_n <= 0 or best_ratio < min_fixed_ratio:
            continue

        proposed_rows.append(row)
        proposed_by_n[best_n] += 1

        # Apply to full row (including potentially non-finite points stays non-finite after arithmetic).
        roi_fixed[row] = roi_row - best_n * off[None, :]
        img_fixed[row] = roi_fixed[row] + off[None, :]
        if norm_fixed is not None:
            norm_fixed[row, :, 0] = roi_fixed[row, :, 0] / roi_w
            norm_fixed[row, :, 1] = roi_fixed[row, :, 1] / roi_h

        if len(preview) < 20:
            preview.append(
                {
                    "row": int(row),
                    "frame": int(frame_indices[row]),
                    "chosen_n": int(best_n),
                    "current_roi_in_bounds_ratio": float(current_ratio),
                    "fixed_roi_in_bounds_ratio": float(best_ratio),
                    "internal_err_img_minus_roi_plus_offset": float(internal_err),
                    "offset_xy": [float(off[0]), float(off[1])],
                }
            )

    report: Dict[str, Any] = {
        "zarr_path": str(Path(zarr_path).resolve()),
        "keypoint_run_group": run_sel.run_group_name,
        "keypoint_run_name": run_sel.run_name,
        "crop_run_name": crop_run_name,
        "roi_count": roi_count,
        "roi_height_px": roi_h,
        "roi_width_px": roi_w,
        "pad_px": pad_px,
        "min_fixed_ratio": min_fixed_ratio,
        "max_n": max_n,
        "proposed_row_count": len(proposed_rows),
        "proposed_row_ranges": _rows_to_ranges(proposed_rows),
        "proposed_frame_ranges": _frame_ranges(frame_indices, proposed_rows),
        "chosen_n_counts": {str(k): int(v) for k, v in sorted(proposed_by_n.items())},
        "preview": preview,
        "applied": bool(apply_changes),
    }

    if apply_changes and proposed_rows:
        kp_img_arr = run["keypoints_img"]
        kp_roi_arr = run["keypoints_roi"]

        full_img = np.asarray(kp_img_arr[:])
        full_roi = np.asarray(kp_roi_arr[:])
        full_img[:, :, :2] = img_fixed
        full_roi[:, :, :2] = roi_fixed
        kp_img_arr[:] = full_img
        kp_roi_arr[:] = full_roi

        if norm_fixed is not None and "keypoints_norm" in run:
            kp_norm_arr = run["keypoints_norm"]
            full_norm = np.asarray(kp_norm_arr[:])
            full_norm[:, :, :2] = norm_fixed
            kp_norm_arr[:] = full_norm

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair offset-corrupted rows in keypoint arrays."
    )
    parser.add_argument("zarr_path", help="Path to zarr archive directory")
    parser.add_argument(
        "--run",
        default=None,
        help="Keypoint run name (default: latest refined_keypoints run, else latest keypoints run)",
    )
    parser.add_argument(
        "--crop-run",
        default=None,
        help="Crop run override (default: source_crop_run, then latest/available)",
    )
    parser.add_argument(
        "--pad-px",
        type=float,
        default=5.0,
        help="ROI bounds slack (default: 5 px)",
    )
    parser.add_argument(
        "--min-fixed-ratio",
        type=float,
        default=0.9,
        help="Required in-bounds ratio after correction to apply row fix (default: 0.9)",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=3,
        help="Max offset multiples to test: roi' = roi - n*offset (default: 3)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write repaired arrays back to zarr (default: dry-run)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write report to JSON path",
    )
    args = parser.parse_args()

    report = repair(
        zarr_path=args.zarr_path,
        run_name=args.run,
        crop_run=args.crop_run,
        pad_px=args.pad_px,
        min_fixed_ratio=args.min_fixed_ratio,
        max_n=args.max_n,
        apply_changes=args.apply,
    )

    print(f"Zarr: {report['zarr_path']}")
    print(f"Keypoint run: {report['keypoint_run_group']}/{report['keypoint_run_name']}")
    print(f"Crop run: {report['crop_run_name']}")
    print(f"Proposed rows: {report['proposed_row_count']}")
    print(f"Row ranges: {report['proposed_row_ranges']}")
    print(f"Frame ranges: {report['proposed_frame_ranges']}")
    print(f"Chosen n counts: {report['chosen_n_counts']}")
    print(f"Applied: {report['applied']}")
    if report["preview"]:
        print("Preview:")
        for row in report["preview"][:10]:
            print(
                "  row={row} frame={frame} n={n}"
                " ratio_before={rb:.3f} ratio_after={ra:.3f}".format(
                    row=row["row"],
                    frame=row["frame"],
                    n=row["chosen_n"],
                    rb=row["current_roi_in_bounds_ratio"],
                    ra=row["fixed_roi_in_bounds_ratio"],
                )
            )

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON written: {out_path}")


if __name__ == "__main__":
    main()
