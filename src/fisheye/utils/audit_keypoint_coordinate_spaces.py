#!/usr/bin/env python3
"""Audit keypoint coordinate-space consistency inside a palette/crimson zarr."""

from __future__ import annotations

import argparse
import json
import math
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
    keys: List[str] = []
    try:
        for k in group.keys():
            try:
                if _is_group(group[k]):
                    keys.append(k)
            except Exception:
                continue
    except Exception:
        return []
    return keys


def _array_keys(group: Any) -> List[str]:
    if hasattr(group, "array_keys"):
        return list(group.array_keys())
    out: List[str] = []
    try:
        for k in group.keys():
            try:
                if not _is_group(group[k]):
                    out.append(k)
            except Exception:
                continue
    except Exception:
        return []
    return out


def _load_array(group: Any, name: str) -> Optional[np.ndarray]:
    if name not in group:
        return None
    return np.asarray(group[name][:])


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
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[float], Optional[float]]:
    if "crop_runs" not in root:
        return None, None, None, None
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
        return name, offsets, roi_h, roi_w

    return None, None, None, None


def _frames_to_ranges(frames: Iterable[int]) -> List[List[int]]:
    seq = sorted(set(int(f) for f in frames))
    if not seq:
        return []
    out: List[List[int]] = []
    start = seq[0]
    prev = seq[0]
    for val in seq[1:]:
        if val == prev + 1:
            prev = val
            continue
        out.append([start, prev])
        start = prev = val
    out.append([start, prev])
    return out


def _finite_xy(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])


def _median_norm(v: np.ndarray) -> float:
    if v.size == 0:
        return math.nan
    return float(np.median(np.linalg.norm(v, axis=1)))


def analyze(
    zarr_path: str,
    run_name: Optional[str],
    crop_run: Optional[str],
    tol_px: float,
    pad_px: float,
) -> Dict[str, Any]:
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
    run_sel = _pick_keypoint_run(root, run_name)
    run = run_sel.run

    run_attrs = dict(getattr(run, "attrs", {}))
    frame_indices = _load_array(run, "frame_indices")
    if frame_indices is None:
        raise RuntimeError("Run is missing 'frame_indices'.")
    frame_indices = frame_indices.astype(np.int64, copy=False)
    roi_count = int(frame_indices.shape[0])

    keypoints_img = _load_array(run, "keypoints_img")
    keypoints_roi = _load_array(run, "keypoints_roi")
    keypoints_norm = _load_array(run, "keypoints_norm")

    if keypoints_img is None and keypoints_roi is None and keypoints_norm is None:
        raise RuntimeError(
            "Run has none of keypoints_img / keypoints_roi / keypoints_norm."
        )

    if keypoints_img is not None and keypoints_img.ndim != 3:
        raise RuntimeError(f"Unexpected keypoints_img shape: {keypoints_img.shape}")
    if keypoints_roi is not None and keypoints_roi.ndim != 3:
        raise RuntimeError(f"Unexpected keypoints_roi shape: {keypoints_roi.shape}")
    if keypoints_norm is not None and keypoints_norm.ndim != 3:
        raise RuntimeError(f"Unexpected keypoints_norm shape: {keypoints_norm.shape}")

    crop_run_name, offsets, roi_h, roi_w = _load_crop_offsets(
        root, run_attrs, crop_run, roi_count
    )

    # Optional image bounds from detect metadata.
    image_w: Optional[float] = None
    image_h: Optional[float] = None
    for detect_group_name in ("refined_detect_runs", "detect_runs"):
        if detect_group_name not in root:
            continue
        detect_runs = root[detect_group_name]
        latest_detect = _pick_latest_run(detect_runs)
        if not latest_detect or latest_detect not in detect_runs:
            continue
        detect_run = detect_runs[latest_detect]
        attrs = dict(getattr(detect_run, "attrs", {}))
        w = attrs.get("image_width")
        h = attrs.get("image_height")
        try:
            if w is not None and h is not None:
                image_w = float(w)
                image_h = float(h)
                if image_w > 0 and image_h > 0:
                    break
        except Exception:
            pass

    # Use dimensions from whichever keypoint array exists.
    sample_kp = keypoints_img if keypoints_img is not None else keypoints_roi
    if sample_kp is None:
        sample_kp = keypoints_norm
    assert sample_kp is not None

    if sample_kp.shape[0] != roi_count:
        raise RuntimeError(
            f"Keypoint rows ({sample_kp.shape[0]}) != frame_indices rows ({roi_count})."
        )
    num_kp = int(sample_kp.shape[1])
    coord_dim = int(sample_kp.shape[2])
    if coord_dim < 2:
        raise RuntimeError(f"coord_dim < 2 in keypoints arrays ({coord_dim}).")

    if keypoints_img is not None:
        keypoints_img = keypoints_img[:, :, :2].astype(np.float64, copy=False)
    if keypoints_roi is not None:
        keypoints_roi = keypoints_roi[:, :, :2].astype(np.float64, copy=False)
    if keypoints_norm is not None:
        keypoints_norm = keypoints_norm[:, :, :2].astype(np.float64, copy=False)

    if offsets is not None and offsets.shape[0] != roi_count:
        raise RuntimeError(
            f"ROI offsets rows ({offsets.shape[0]}) != frame_indices rows ({roi_count})."
        )

    per_row: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    frame_to_statuses: Dict[int, List[str]] = defaultdict(list)
    frame_to_rows: Dict[int, List[int]] = defaultdict(list)

    for row in range(roi_count):
        status = "insufficient_data"
        details: Dict[str, Any] = {
            "row": row,
            "frame": int(frame_indices[row]),
        }

        img = keypoints_img[row] if keypoints_img is not None else None
        roi = keypoints_roi[row] if keypoints_roi is not None else None
        norm = keypoints_norm[row] if keypoints_norm is not None else None
        off = offsets[row] if offsets is not None else None

        if img is not None:
            details["finite_img_points"] = int(_finite_xy(img).sum())
        if roi is not None:
            details["finite_roi_points"] = int(_finite_xy(roi).sum())
        if norm is not None:
            details["finite_norm_points"] = int(_finite_xy(norm).sum())

        if roi is not None and roi_w is not None and roi_h is not None:
            valid_roi = _finite_xy(roi)
            if np.any(valid_roi):
                roi_valid = roi[valid_roi]
                roi_in_bounds = (
                    (roi_valid[:, 0] >= -pad_px)
                    & (roi_valid[:, 0] <= roi_w + pad_px)
                    & (roi_valid[:, 1] >= -pad_px)
                    & (roi_valid[:, 1] <= roi_h + pad_px)
                )
                details["roi_in_bounds_ratio"] = float(roi_in_bounds.mean())

        if img is not None and image_w is not None and image_h is not None:
            valid_img = _finite_xy(img)
            if np.any(valid_img):
                img_valid = img[valid_img]
                img_in_bounds = (
                    (img_valid[:, 0] >= -pad_px)
                    & (img_valid[:, 0] <= image_w + pad_px)
                    & (img_valid[:, 1] >= -pad_px)
                    & (img_valid[:, 1] <= image_h + pad_px)
                )
                details["img_in_bounds_ratio"] = float(img_in_bounds.mean())

        if norm is not None:
            valid_norm = _finite_xy(norm)
            if np.any(valid_norm):
                norm_valid = norm[valid_norm]
                norm_in_bounds = (
                    (norm_valid[:, 0] >= -0.05)
                    & (norm_valid[:, 0] <= 1.05)
                    & (norm_valid[:, 1] >= -0.05)
                    & (norm_valid[:, 1] <= 1.05)
                )
                details["norm_in_bounds_ratio"] = float(norm_in_bounds.mean())

        # Primary check: does keypoints_img ~= keypoints_roi + crop_offset ?
        if img is not None and roi is not None and off is not None and np.all(np.isfinite(off)):
            valid = _finite_xy(img) & _finite_xy(roi)
            if np.any(valid):
                img_valid = img[valid]
                roi_valid = roi[valid]

                err_img_vs_roi_plus_off = img_valid - (roi_valid + off[None, :])
                med_err_img_vs_roi_plus_off = _median_norm(err_img_vs_roi_plus_off)
                details["med_err_img_vs_roi_plus_offset"] = med_err_img_vs_roi_plus_off

                err_img_vs_roi = img_valid - roi_valid
                med_err_img_vs_roi = _median_norm(err_img_vs_roi)
                details["med_err_img_vs_roi"] = med_err_img_vs_roi

                local_from_img = img_valid - off[None, :]
                if roi_w is not None and roi_h is not None:
                    in_bounds = (
                        (local_from_img[:, 0] >= -pad_px)
                        & (local_from_img[:, 0] <= roi_w + pad_px)
                        & (local_from_img[:, 1] >= -pad_px)
                        & (local_from_img[:, 1] <= roi_h + pad_px)
                    )
                    in_bounds_ratio = float(in_bounds.mean())
                    details["img_minus_offset_in_roi_bounds_ratio"] = in_bounds_ratio
                else:
                    in_bounds_ratio = math.nan

                if med_err_img_vs_roi_plus_off <= tol_px:
                    status = "aligned_img_eq_roi_plus_offset"
                elif med_err_img_vs_roi <= tol_px:
                    status = "suspicious_img_eq_roi_possible_double_offset"
                elif not math.isnan(in_bounds_ratio) and in_bounds_ratio < 0.5:
                    status = "misaligned_img_minus_offset_oob"
                else:
                    status = "misaligned_other"

        # Secondary check: keypoints_norm vs keypoints_roi when roi size is known.
        if (
            status == "insufficient_data"
            and norm is not None
            and roi is not None
            and roi_w is not None
            and roi_h is not None
        ):
            valid = _finite_xy(norm) & _finite_xy(roi)
            if np.any(valid):
                norm_to_roi = np.empty_like(norm[valid])
                norm_to_roi[:, 0] = norm[valid][:, 0] * roi_w
                norm_to_roi[:, 1] = norm[valid][:, 1] * roi_h
                med_err = _median_norm(norm_to_roi - roi[valid])
                details["med_err_norm_scaled_vs_roi"] = med_err
                status = (
                    "aligned_norm_scaled_eq_roi"
                    if med_err <= tol_px
                    else "misaligned_norm_vs_roi"
                )

        roi_bounds_ratio = details.get("roi_in_bounds_ratio")
        if isinstance(roi_bounds_ratio, (int, float)) and np.isfinite(roi_bounds_ratio):
            if roi_bounds_ratio < 0.5:
                if status == "aligned_img_eq_roi_plus_offset":
                    status = "internally_aligned_but_roi_out_of_bounds"
                elif status.startswith("aligned_"):
                    status = "aligned_but_roi_out_of_bounds"

        img_bounds_ratio = details.get("img_in_bounds_ratio")
        if isinstance(img_bounds_ratio, (int, float)) and np.isfinite(img_bounds_ratio):
            if img_bounds_ratio < 0.5 and status.startswith("aligned_"):
                status = "aligned_but_img_out_of_bounds"

        details["status"] = status
        per_row.append(details)
        status_counts[status] += 1
        frame = int(frame_indices[row])
        frame_to_statuses[frame].append(status)
        frame_to_rows[frame].append(row)

    frame_priority = {
        "internally_aligned_but_roi_out_of_bounds": 6,
        "aligned_but_roi_out_of_bounds": 6,
        "aligned_but_img_out_of_bounds": 6,
        "suspicious_img_eq_roi_possible_double_offset": 5,
        "misaligned_img_minus_offset_oob": 4,
        "misaligned_other": 3,
        "misaligned_norm_vs_roi": 3,
        "insufficient_data": 2,
        "aligned_norm_scaled_eq_roi": 1,
        "aligned_img_eq_roi_plus_offset": 0,
    }

    frame_status: Dict[int, str] = {}
    for frame, statuses in frame_to_statuses.items():
        frame_status[frame] = max(statuses, key=lambda s: frame_priority.get(s, 0))

    bad_frames = [
        frame
        for frame, st in frame_status.items()
        if st.startswith("misaligned")
        or st.startswith("suspicious")
        or st.endswith("_out_of_bounds")
    ]

    bad_rows = [
        r
        for r in per_row
        if r["status"].startswith("misaligned")
        or r["status"].startswith("suspicious")
        or r["status"].endswith("_out_of_bounds")
    ]

    preview_rows = sorted(
        bad_rows,
        key=lambda r: (r["frame"], r["row"]),
    )[:100]

    out: Dict[str, Any] = {
        "zarr_path": str(Path(zarr_path).resolve()),
        "keypoint_run_group": run_sel.run_group_name,
        "keypoint_run_name": run_sel.run_name,
        "crop_run_name": crop_run_name,
        "roi_count": roi_count,
        "num_keypoints": num_kp,
        "coord_dim": coord_dim,
        "roi_height_px": roi_h,
        "roi_width_px": roi_w,
        "image_height_px": image_h,
        "image_width_px": image_w,
        "has_keypoints_img": keypoints_img is not None,
        "has_keypoints_roi": keypoints_roi is not None,
        "has_keypoints_norm": keypoints_norm is not None,
        "has_roi_offsets": offsets is not None,
        "status_counts": dict(status_counts),
        "bad_frame_count": len(set(bad_frames)),
        "bad_frame_ranges": _frames_to_ranges(bad_frames),
        "bad_row_count": len(bad_rows),
        "bad_row_preview": preview_rows,
        "frame_status_preview": [
            {"frame": int(frame), "status": st, "rows": frame_to_rows[frame][:8]}
            for frame, st in sorted(frame_status.items())[:200]
        ],
    }
    return out


def _print_human_summary(report: Dict[str, Any]) -> None:
    print(f"Zarr: {report['zarr_path']}")
    print(
        f"Keypoint run: {report['keypoint_run_group']}/{report['keypoint_run_name']}"
    )
    print(f"Crop run: {report['crop_run_name']}")
    print(
        "Arrays:"
        f" img={report['has_keypoints_img']}"
        f" roi={report['has_keypoints_roi']}"
        f" norm={report['has_keypoints_norm']}"
        f" offsets={report['has_roi_offsets']}"
    )
    print(
        "ROI size:"
        f" h={report['roi_height_px']}"
        f" w={report['roi_width_px']}"
    )
    print(
        "Image size:"
        f" h={report['image_height_px']}"
        f" w={report['image_width_px']}"
    )
    print(f"ROI rows: {report['roi_count']}")
    print("Status counts:")
    for k, v in sorted(report["status_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    print(f"Bad frames: {report['bad_frame_count']}")
    print(f"Bad frame ranges: {report['bad_frame_ranges']}")
    if report["bad_row_preview"]:
        print("Bad preview (first 10 rows):")
        for row in report["bad_row_preview"][:10]:
            med_a = row.get("med_err_img_vs_roi_plus_offset")
            med_b = row.get("med_err_img_vs_roi")
            print(
                "  row={row} frame={frame} status={status}"
                " med_err(img-(roi+off))={a} med_err(img-roi)={b}".format(
                    row=row["row"],
                    frame=row["frame"],
                    status=row["status"],
                    a=f"{med_a:.3f}" if isinstance(med_a, (int, float)) and np.isfinite(med_a) else "n/a",
                    b=f"{med_b:.3f}" if isinstance(med_b, (int, float)) and np.isfinite(med_b) else "n/a",
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit coordinate-space consistency of keypoint arrays in a zarr."
    )
    parser.add_argument("zarr_path", help="Path to zarr archive directory")
    parser.add_argument(
        "--run",
        default=None,
        help="Keypoint run name (default: latest in refined_keypoints_runs, else keypoints_runs)",
    )
    parser.add_argument(
        "--crop-run",
        default=None,
        help="Crop run name override (default: source_crop_run then latest/available)",
    )
    parser.add_argument(
        "--tol-px",
        type=float,
        default=2.0,
        help="Pixel tolerance for aligned-vs-misaligned checks (default: 2.0)",
    )
    parser.add_argument(
        "--pad-px",
        type=float,
        default=5.0,
        help="Bounds slack when checking ROI-local coordinates (default: 5.0)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write full report to JSON path",
    )
    args = parser.parse_args()

    report = analyze(
        zarr_path=args.zarr_path,
        run_name=args.run,
        crop_run=args.crop_run,
        tol_px=args.tol_px,
        pad_px=args.pad_px,
    )
    _print_human_summary(report)

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON written: {out_path}")


if __name__ == "__main__":
    main()
