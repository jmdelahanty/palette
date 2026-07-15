#!/usr/bin/env python3
"""Analyze overlap between bad keypoint rows and reason/quality/source labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


def _decode_fixed_bytes_row(row: np.ndarray) -> str:
    row_u8 = np.asarray(row).astype(np.uint8, copy=False).ravel()
    if row_u8.size == 0:
        return ""
    if 0 in row_u8:
        row_u8 = row_u8[: int(np.where(row_u8 == 0)[0][0])]
    return bytes(row_u8.tolist()).decode("utf-8", errors="ignore").strip()


def _decode_string_array(arr: np.ndarray) -> List[str]:
    if arr.ndim == 1:
        if arr.dtype.kind == "U":
            return [str(x) for x in arr.tolist()]
        if arr.dtype.kind == "S":
            return [bytes(x).decode("utf-8", errors="ignore").strip("\x00") for x in arr.tolist()]
        if arr.dtype.kind == "O":
            out: List[str] = []
            for x in arr.tolist():
                if isinstance(x, bytes):
                    out.append(x.decode("utf-8", errors="ignore").strip("\x00"))
                else:
                    out.append(str(x))
            return out
    if arr.ndim == 2 and arr.dtype.kind in ("u", "i"):
        return [_decode_fixed_bytes_row(arr[i]) for i in range(arr.shape[0])]
    raise RuntimeError(f"Unsupported string array layout: shape={arr.shape} dtype={arr.dtype}")


def _load_reason_strings(run: Any, roi_count: int) -> Tuple[str, List[str]]:
    if "reason_bytes" in run:
        reason = _decode_string_array(np.asarray(run["reason_bytes"][:]))
        if len(reason) == roi_count:
            return "reason_bytes", reason
    if "reason" in run:
        reason = _decode_string_array(np.asarray(run["reason"][:]))
        if len(reason) == roi_count:
            return "reason", reason
    return "none", [""] * roi_count


def _split_reason_tags(reason: str) -> Set[str]:
    reason = (reason or "").strip()
    if not reason:
        return {"<empty>"}
    tags = {part.strip() for part in reason.split("|") if part.strip()}
    return tags if tags else {"<empty>"}


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


def _frames_to_ranges(frames: Iterable[int]) -> List[List[int]]:
    return _rows_to_ranges(frames)


def _median_norm(v: np.ndarray) -> float:
    if v.size == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(v, axis=1)))


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


def _label_counts(values: Sequence[int], bad_rows: Set[int]) -> List[Dict[str, Any]]:
    total = len(values)
    total_counter = Counter(int(v) for v in values)
    bad_counter = Counter(int(values[i]) for i in bad_rows)
    rows: List[Dict[str, Any]] = []
    for label, tot in sorted(total_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        bad = bad_counter.get(label, 0)
        rows.append(
            {
                "label": int(label),
                "total_rows": int(tot),
                "bad_rows": int(bad),
                "bad_rate_within_label": (float(bad) / float(tot)) if tot > 0 else 0.0,
                "share_of_all_bad_rows": (float(bad) / float(len(bad_rows))) if bad_rows else 0.0,
            }
        )
    return rows


def analyze(
    zarr_path: str,
    run_name: Optional[str],
    crop_run: Optional[str],
    tol_px: float,
    bad_ratio_threshold: float,
    pad_px: float,
) -> Dict[str, Any]:
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
    run_sel = _pick_keypoint_run(root, run_name)
    run = run_sel.run
    run_attrs = dict(getattr(run, "attrs", {}))

    frame_indices = _load_array(run, "frame_indices")
    keypoints_img = _load_array(run, "keypoints_img")
    keypoints_roi = _load_array(run, "keypoints_roi")
    if frame_indices is None or keypoints_img is None or keypoints_roi is None:
        raise RuntimeError("Run must include frame_indices, keypoints_img, keypoints_roi.")

    frame_indices = frame_indices.astype(np.int64, copy=False)
    roi_count = int(frame_indices.shape[0])
    if keypoints_img.shape[0] != roi_count or keypoints_roi.shape[0] != roi_count:
        raise RuntimeError("Keypoint arrays not aligned with frame_indices.")

    crop_run_name, offsets, roi_h, roi_w = _load_crop_offsets(
        root, run_attrs, crop_run, roi_count
    )
    if offsets is None or roi_h is None or roi_w is None:
        raise RuntimeError(
            "Could not load crop offsets + ROI size (needed to classify bad rows)."
        )

    img = keypoints_img[:, :, :2].astype(np.float64, copy=False)
    roi = keypoints_roi[:, :, :2].astype(np.float64, copy=False)

    source_name, reasons = _load_reason_strings(run, roi_count)

    quality_arr = _load_array(run, "quality_labels")
    if quality_arr is None:
        quality = np.full(roi_count, -1, dtype=np.int64)
    else:
        quality = quality_arr.astype(np.int64, copy=False).reshape(-1)
        if quality.shape[0] != roi_count:
            raise RuntimeError("quality_labels row count mismatch.")

    detection_source_arr = _load_array(run, "detection_source")
    if detection_source_arr is None:
        detection_source = np.full(roi_count, -1, dtype=np.int64)
    else:
        detection_source = detection_source_arr.astype(np.int64, copy=False).reshape(-1)
        if detection_source.shape[0] != roi_count:
            raise RuntimeError("detection_source row count mismatch.")

    row_records: List[Dict[str, Any]] = []
    bad_rows: Set[int] = set()

    for row in range(roi_count):
        off = offsets[row]
        if not np.all(np.isfinite(off)):
            continue

        img_row = img[row]
        roi_row = roi[row]
        valid = (
            np.isfinite(img_row[:, 0])
            & np.isfinite(img_row[:, 1])
            & np.isfinite(roi_row[:, 0])
            & np.isfinite(roi_row[:, 1])
        )
        if not np.any(valid):
            continue

        img_valid = img_row[valid]
        roi_valid = roi_row[valid]
        med_err = _median_norm(img_valid - (roi_valid + off[None, :]))
        roi_ratio = _in_roi_bounds_ratio(roi_valid, roi_w, roi_h, pad_px)

        is_bad = bool(np.isfinite(med_err) and med_err <= tol_px and roi_ratio < bad_ratio_threshold)
        if is_bad:
            bad_rows.add(row)

        row_records.append(
            {
                "row": int(row),
                "frame": int(frame_indices[row]),
                "is_bad": is_bad,
                "med_err_img_vs_roi_plus_offset": float(med_err),
                "roi_in_bounds_ratio": float(roi_ratio),
            }
        )

    bad_frames = sorted({int(frame_indices[r]) for r in bad_rows})

    tag_total_rows: Counter[str] = Counter()
    tag_bad_rows: Counter[str] = Counter()
    for row in range(roi_count):
        tags = _split_reason_tags(reasons[row] if row < len(reasons) else "")
        for t in tags:
            tag_total_rows[t] += 1
            if row in bad_rows:
                tag_bad_rows[t] += 1

    tag_stats: List[Dict[str, Any]] = []
    for tag, tot in sorted(tag_total_rows.items(), key=lambda kv: (-kv[1], kv[0])):
        bad = tag_bad_rows.get(tag, 0)
        tag_stats.append(
            {
                "tag": tag,
                "total_rows": int(tot),
                "bad_rows": int(bad),
                "bad_rate_within_tag": (float(bad) / float(tot)) if tot > 0 else 0.0,
                "share_of_all_bad_rows": (float(bad) / float(len(bad_rows))) if bad_rows else 0.0,
            }
        )

    quality_stats = _label_counts(quality.tolist(), bad_rows)
    detection_source_stats = _label_counts(detection_source.tolist(), bad_rows)

    bad_preview: List[Dict[str, Any]] = []
    for rec in sorted((r for r in row_records if r["is_bad"]), key=lambda r: r["row"])[:100]:
        row = rec["row"]
        bad_preview.append(
            {
                **rec,
                "reason": reasons[row] if row < len(reasons) else "",
                "quality_label": int(quality[row]) if row < quality.shape[0] else -1,
                "detection_source": int(detection_source[row]) if row < detection_source.shape[0] else -1,
            }
        )

    report: Dict[str, Any] = {
        "zarr_path": str(Path(zarr_path).resolve()),
        "keypoint_run_group": run_sel.run_group_name,
        "keypoint_run_name": run_sel.run_name,
        "crop_run_name": crop_run_name,
        "reason_source": source_name,
        "roi_count": roi_count,
        "roi_height_px": float(roi_h),
        "roi_width_px": float(roi_w),
        "tol_px": float(tol_px),
        "bad_ratio_threshold": float(bad_ratio_threshold),
        "pad_px": float(pad_px),
        "bad_row_count": int(len(bad_rows)),
        "bad_row_ranges": _rows_to_ranges(sorted(bad_rows)),
        "bad_frame_count": int(len(bad_frames)),
        "bad_frame_ranges": _frames_to_ranges(bad_frames),
        "reason_tag_stats": tag_stats,
        "quality_label_stats": quality_stats,
        "detection_source_stats": detection_source_stats,
        "bad_preview": bad_preview,
    }
    return report


def _print_top_stats(report: Dict[str, Any], top_k: int) -> None:
    print(f"Zarr: {report['zarr_path']}")
    print(
        f"Run: {report['keypoint_run_group']}/{report['keypoint_run_name']} "
        f"(crop={report['crop_run_name']})"
    )
    print(f"Reason source: {report['reason_source']}")
    print(f"Bad rows: {report['bad_row_count']} / {report['roi_count']}")
    print(f"Bad frame ranges: {report['bad_frame_ranges']}")

    print("\nTop reason tags by bad rows:")
    for row in sorted(
        report["reason_tag_stats"],
        key=lambda r: (-r["bad_rows"], -r["bad_rate_within_tag"], r["tag"]),
    )[:top_k]:
        print(
            "  {tag}: bad={bad}/{total} bad_rate={rate:.3f} share_bad={share:.3f}".format(
                tag=row["tag"],
                bad=row["bad_rows"],
                total=row["total_rows"],
                rate=row["bad_rate_within_tag"],
                share=row["share_of_all_bad_rows"],
            )
        )

    print("\nQuality labels by bad rows:")
    for row in sorted(
        report["quality_label_stats"],
        key=lambda r: (-r["bad_rows"], -r["bad_rate_within_label"], r["label"]),
    )[:top_k]:
        print(
            "  label={label}: bad={bad}/{total} bad_rate={rate:.3f} share_bad={share:.3f}".format(
                label=row["label"],
                bad=row["bad_rows"],
                total=row["total_rows"],
                rate=row["bad_rate_within_label"],
                share=row["share_of_all_bad_rows"],
            )
        )

    print("\nDetection source by bad rows:")
    for row in sorted(
        report["detection_source_stats"],
        key=lambda r: (-r["bad_rows"], -r["bad_rate_within_label"], r["label"]),
    )[:top_k]:
        print(
            "  source={label}: bad={bad}/{total} bad_rate={rate:.3f} share_bad={share:.3f}".format(
                label=row["label"],
                bad=row["bad_rows"],
                total=row["total_rows"],
                rate=row["bad_rate_within_label"],
                share=row["share_of_all_bad_rows"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect bad keypoint rows (internally aligned but ROI out-of-bounds) "
            "and summarize overlap with reason tags/quality labels/source."
        )
    )
    parser.add_argument("zarr_path", help="Path to zarr archive directory")
    parser.add_argument(
        "--run",
        default=None,
        help="Keypoint run name (default: latest refined run, else latest keypoints run)",
    )
    parser.add_argument(
        "--crop-run",
        default=None,
        help="Crop run override (default: source_crop_run then latest/available)",
    )
    parser.add_argument(
        "--tol-px",
        type=float,
        default=2.0,
        help="Internal alignment tolerance for img-(roi+offset) (default 2.0)",
    )
    parser.add_argument(
        "--bad-ratio-threshold",
        type=float,
        default=0.5,
        help="Rows with roi_in_bounds_ratio below this are marked bad (default 0.5)",
    )
    parser.add_argument(
        "--pad-px",
        type=float,
        default=5.0,
        help="Bounds slack for ROI-local in-bounds checks (default 5.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many rows to show in each summary table",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write full report JSON to this path",
    )
    args = parser.parse_args()

    report = analyze(
        zarr_path=args.zarr_path,
        run_name=args.run,
        crop_run=args.crop_run,
        tol_px=args.tol_px,
        bad_ratio_threshold=args.bad_ratio_threshold,
        pad_px=args.pad_px,
    )

    _print_top_stats(report, top_k=max(1, int(args.top_k)))

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON written: {out_path}")


if __name__ == "__main__":
    main()
