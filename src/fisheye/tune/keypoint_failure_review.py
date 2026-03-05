"""
Manual review tool for keypoints in a refined keypoint run.

Edits are applied to the refined run only (raw keypoints remain untouched).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict, Any
from datetime import datetime, timezone
import hashlib
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zarr

from ..shared.detect_reason_codec import read_reason_labels, write_reason_columns
from ..shared.keypoint_stale import mark_downstream_eye_mask_runs_stale
from ..refinement.keypoint_quality import compute_geometry_metrics
from ..refinement.refine_keypoints import _compute_heading_from_points


_DEFAULT_LABELS = ("swim_bladder", "eye_left", "eye_right")
_DEFAULT_COLORS = ("#22c55e", "#1a66f3", "#f85151")


def _get_latest_run(root: zarr.Group, group_name: str) -> str:
    parent_name = f"{group_name}_runs"
    if parent_name not in root:
        raise RuntimeError(f"No '{parent_name}' group found in Zarr store.")
    latest = root[parent_name].attrs.get("latest")
    if not latest:
        raise RuntimeError(f"No runs recorded under '{parent_name}'.")
    return latest


def _total_keypoints(refined: zarr.Group) -> int:
    for key in ("keypoints_roi", "keypoints_img", "keypoints_norm"):
        if key in refined:
            return int(refined[key].shape[0])
    for key in ("refined_success", "source_success", "failure_indices"):
        if key in refined:
            return int(refined[key].shape[0])
    return 0


def _as_positive_int(value: object) -> Optional[int]:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _resolve_full_frame_dimensions(root: zarr.Group) -> tuple[int, int]:
    raw = root.get("raw_video")
    if raw is not None and "images_full" in raw:
        full = raw["images_full"]
        return int(full.shape[1]), int(full.shape[2])

    width = (
        _as_positive_int(root.attrs.get("width"))
        or _as_positive_int(root.attrs.get("video_width"))
        or _as_positive_int(root.attrs.get("palette_video_width"))
        or _as_positive_int(root.attrs.get("source_full_width"))
        or _as_positive_int(root.attrs.get("source_video_width"))
    )
    height = (
        _as_positive_int(root.attrs.get("height"))
        or _as_positive_int(root.attrs.get("video_height"))
        or _as_positive_int(root.attrs.get("palette_video_height"))
        or _as_positive_int(root.attrs.get("source_full_height"))
        or _as_positive_int(root.attrs.get("source_video_height"))
    )

    if raw is not None:
        original_resolution = raw.attrs.get("original_resolution")
        if isinstance(original_resolution, (list, tuple, np.ndarray)) and len(original_resolution) == 2:
            res_h = _as_positive_int(original_resolution[0])
            res_w = _as_positive_int(original_resolution[1])
            if height is None and res_h is not None:
                height = res_h
            if width is None and res_w is not None:
                width = res_w

        if "images_ds" in raw and (height is None or width is None):
            ds = raw["images_ds"]
            if height is None:
                height = _as_positive_int(ds.shape[1])
            if width is None:
                width = _as_positive_int(ds.shape[2])

    return int(height or 640), int(width or 640)


def _load_failure_indices(refined: zarr.Group, include_all: bool = False) -> np.ndarray:
    if include_all:
        total = _total_keypoints(refined)
        if total <= 0:
            return np.zeros(0, dtype="i4")
        return np.arange(total, dtype="i4")
    if "refined_success" in refined:
        success = np.asarray(refined["refined_success"][:], dtype=bool)
        failures = np.where(~success)[0].astype("i4", copy=False)
    elif "source_success" in refined:
        success = np.asarray(refined["source_success"][:], dtype=bool)
        failures = np.where(~success)[0].astype("i4", copy=False)
    elif "failure_indices" in refined:
        failures = np.asarray(refined["failure_indices"][:], dtype="i4")
    else:
        return np.zeros(0, dtype="i4")

    if failures.size == 0:
        return failures
    reason_vals = read_reason_labels(refined)
    if reason_vals is None:
        return failures
    reason_vals = np.asarray(reason_vals, dtype=object)
    if reason_vals.size == 0:
        return failures
    keep_mask = []
    for idx in failures:
        try:
            text = str(reason_vals[int(idx)]) if reason_vals[int(idx)] is not None else ""
        except Exception:
            text = ""
        keep_mask.append(
            "fish_present_no_keypoints" not in text
            and "detection_issue" not in text
        )
    return failures[np.array(keep_mask, dtype=bool)]


def _clean_reason(existing: str, new_tags: Sequence[str]) -> str:
    tags = [tag for tag in existing.split("|") if tag and tag != "detection_failed"]
    tags.extend(new_tags)
    unique = sorted(set(tags))
    return "|".join(unique) if unique else "manual_correction"


def _sanitize_reason_array(reason_arr: zarr.Array) -> None:
    try:
        raw = reason_arr[:]
    except Exception:
        return
    if raw.size == 0:
        return

    def coerce(val: object) -> str:
        if val is None:
            return ""
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            if val.size == 1:
                return str(val.item())
            return "|".join(str(item) for item in val.tolist())
        return str(val)

    cleaned = np.array([coerce(v) for v in raw], dtype=object)
    reason_arr[:] = cleaned


def _write_reason_labels(refined: zarr.Group, labels: np.ndarray) -> None:
    heading = refined.get("heading")
    if heading is not None and heading.chunks:
        chunk_size = int(heading.chunks[0])
    else:
        chunk_size = max(1, min(1024, int(np.asarray(labels).shape[0])))
    write_reason_columns(
        refined,
        np.asarray(labels, dtype=object),
        chunk_size,
        include_reason_text=True,
        overwrite=True,
    )


def _load_frame_flags(path: Path) -> Dict[str, list[Dict[str, Optional[int]]]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    parsed: Dict[str, list[Dict[str, Optional[int]]]] = {}
    for key, value in data.items():
        entries: list[Dict[str, Optional[int]]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    frame_val = item.get("frame_idx")
                    roi_val = item.get("roi_idx")
                    try:
                        frame_idx = int(frame_val) if frame_val is not None else None
                    except (TypeError, ValueError):
                        frame_idx = None
                    try:
                        roi_idx = int(roi_val) if roi_val is not None else None
                    except (TypeError, ValueError):
                        roi_idx = None
                    if frame_idx is not None:
                        entries.append({"frame_idx": frame_idx, "roi_idx": roi_idx})
                else:
                    try:
                        frame_idx = int(item)
                    except (TypeError, ValueError):
                        continue
                    entries.append({"frame_idx": frame_idx, "roi_idx": None})
        parsed[str(key)] = entries
    return parsed


def _append_flagged_frame(
    flag_path: Path,
    zarr_path: str,
    frame_idx: int,
    roi_idx: Optional[int],
) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_frame_flags(flag_path)
    entries = data.get(zarr_path, [])
    dedupe = {(entry.get("frame_idx"), entry.get("roi_idx")) for entry in entries}
    key = (int(frame_idx), int(roi_idx) if roi_idx is not None else None)
    if key in dedupe:
        return
    entries.append({"frame_idx": int(frame_idx), "roi_idx": key[1]})
    entries.sort(key=lambda item: (item.get("frame_idx") or 0, item.get("roi_idx") or -1))
    data[zarr_path] = entries
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_detection_frame_flags(path: Path) -> Dict[str, list[int]]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    parsed: Dict[str, list[int]] = {}
    for key, value in data.items():
        frames: list[int] = []
        if isinstance(value, list):
            for item in value:
                try:
                    frames.append(int(item))
                except (TypeError, ValueError):
                    continue
        parsed[str(key)] = sorted(set(frames))
    return parsed


def _append_detection_frame(flag_path: Path, zarr_path: str, frame_idx: int) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_detection_frame_flags(flag_path)
    frames = set(data.get(zarr_path, []))
    frames.add(int(frame_idx))
    data[zarr_path] = sorted(frames)
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _append_flagged_path(flag_path: Path, zarr_path: str) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if flag_path.exists():
        with flag_path.open("r", encoding="utf-8") as handle:
            existing = {line.strip() for line in handle if line.strip()}
    if zarr_path in existing:
        return
    with flag_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{zarr_path}\n")


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_keypoint_signature(attrs: Dict[str, Any]) -> Dict[str, object]:
    params = attrs.get("parameters")
    if not isinstance(params, dict):
        provenance = attrs.get("provenance")
        if isinstance(provenance, dict):
            params = provenance.get("parameters")
    if not isinstance(params, dict):
        params = None

    parameter_source = attrs.get("parameter_source")
    if parameter_source is None and isinstance(params, dict):
        parameter_source = params.get("parameter_source")

    return {
        "signature_version": 1,
        "source_keypoints_run": attrs.get("source_keypoints_run"),
        "source_crop_run": attrs.get("source_crop_run"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "parameter_source": parameter_source,
        "parameters_hash": _hash_parameters(params),
    }


def _apply_review_status(
    refined_parent: zarr.Group,
    refined_run: str,
    refined: zarr.Group,
    *,
    state: str,
    method: str,
    intended_use: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "state": state,
        "method": method,
        "intended_use": intended_use,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes

    refined.attrs["keypoint_review_status"] = payload

    signature = refined.attrs.get("keypoint_signature")
    if not isinstance(signature, dict):
        signature = _build_keypoint_signature(dict(refined.attrs))
        refined.attrs["keypoint_signature"] = signature
    refined.attrs["keypoint_review_signature"] = signature

    refined_parent.attrs["keypoint_review_status_latest"] = refined_run
    return payload


def launch_review(
    zarr_path: str,
    refined_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    include_all: bool = False,
    target_frames: Optional[Sequence[int]] = None,
    target_roi_indices: Optional[Sequence[int]] = None,
    review_state: str = "approved",
    review_method: str = "manual",
    review_intended_use: str = "training",
    reviewer: Optional[str] = None,
    review_notes: Optional[str] = None,
    frame_flag_file: Optional[str] = None,
    detect_flag_file: Optional[str] = None,
    detect_frame_flag_file: Optional[str] = None,
) -> None:
    root = zarr.open_group(zarr_path, mode="a")

    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")

    using_latest = refined_run is None
    if refined_run is None:
        refined_run = refined_parent.attrs.get("latest")
    if not refined_run or refined_run not in refined_parent:
        raise RuntimeError("Refined keypoint run not found.")
    refined = refined_parent[refined_run]

    flag_path = Path(frame_flag_file).expanduser() if frame_flag_file else None
    detect_flag_path = Path(detect_flag_file).expanduser() if detect_flag_file else None
    detect_frame_flag_path = (
        Path(detect_frame_flag_file).expanduser() if detect_frame_flag_file else None
    )
    crop_run = crop_run or refined.attrs.get("source_crop_run") or _get_latest_run(root, "crop")
    crop_group = root[f"crop_runs/{crop_run}"]
    roi_images = crop_group["roi_images"]
    roi_coords = crop_group["roi_coordinates_full"]
    frame_indices = crop_group["frame_indices"][:]

    full_h, full_w = _resolve_full_frame_dimensions(root)
    norm_factor = np.array([full_w, full_h], dtype=np.float64)

    failures = _load_failure_indices(refined, include_all=include_all)
    targeted = False
    if target_frames or target_roi_indices:
        selected: set[int] = set()
        if target_frames:
            target_frames_arr = np.array(sorted(set(int(f) for f in target_frames)), dtype=np.int64)
            frame_hits = np.where(np.isin(frame_indices, target_frames_arr))[0].astype("i4", copy=False)
            selected.update(int(v) for v in frame_hits.tolist())
        if target_roi_indices:
            total_rows = int(frame_indices.shape[0])
            for roi_idx in sorted(set(int(v) for v in target_roi_indices)):
                if 0 <= roi_idx < total_rows:
                    selected.add(int(roi_idx))
        targeted = True
        failures = np.array(sorted(selected), dtype="i4")
    if failures.size == 0:
        if targeted:
            print("No matching keypoints found for requested targets.")
            return
        if include_all:
            print("No keypoints found to review.")
        else:
            print("No failed keypoints to review.")
        return
    if flag_path is not None:
        print(f"Frame flag file: {flag_path.expanduser().resolve(strict=False)}")
    if detect_flag_path is not None:
        print(f"Detection flag file: {detect_flag_path.expanduser().resolve(strict=False)}")
    if detect_frame_flag_path is not None:
        print(f"Detection frame flag file: {detect_frame_flag_path.expanduser().resolve(strict=False)}")

    summary_raw = refined.attrs.get("summary_statistics", {})
    summary = summary_raw.get("refine", summary_raw) if isinstance(summary_raw, dict) else {}
    confidence_threshold = float(summary.get("confidence_threshold", 0.3))
    min_triangle_angle = float(summary.get("min_triangle_angle", 10.0))
    min_triangle_area = float(summary.get("min_triangle_area", 100.0))
    max_tri_val = summary.get("max_triangle_area")
    try:
        max_triangle_area = float(max_tri_val) if max_tri_val is not None else None
    except (TypeError, ValueError):
        max_triangle_area = None

    if "keypoints_roi" not in refined:
        raise RuntimeError("Refined run is missing keypoints_roi.")
    kp_roi_arr = refined["keypoints_roi"]
    kp_img_arr = refined.get("keypoints_img")
    kp_norm_arr = refined.get("keypoints_norm")
    heading_arr = refined.get("heading")
    confidence_arr = refined.get("confidence")
    conf_arr = refined.get("keypoint_confidences")
    triangle_area_arr = refined.get("triangle_area")
    min_angle_arr = refined.get("min_angle")
    triangle_angles_arr = refined.get("triangle_angles")
    refined_success_arr = refined.get("refined_success")
    flip_corrected_arr = refined.get("flip_corrected")
    quality_labels_arr = refined.get("quality_labels")
    confidence_valid_arr = refined.get("confidence_valid")
    geometry_valid_arr = refined.get("geometry_valid")
    usable_arr = refined.get("usable_keypoints")
    reason_arr = refined.get("reason")
    heading_finite_arr = refined.get("heading_finite")
    heading_usable_arr = refined.get("heading_usable")
    detection_source_arr = refined.get("detection_source")
    if reason_arr is None:
        reason_labels = read_reason_labels(refined)
        if reason_labels is not None:
            _write_reason_labels(refined, np.asarray(reason_labels, dtype=object))
            reason_arr = refined.get("reason")
    if reason_arr is not None:
        _sanitize_reason_array(reason_arr)

    labels = list(refined.attrs.get("keypoint_labels", _DEFAULT_LABELS))
    colors = _DEFAULT_COLORS[: len(labels)]

    idx_pos = 0
    active_idx = 0
    points = np.full((len(labels), 2), np.nan, dtype=np.float64)
    show_text = True

    mpl.rcParams["keymap.save"] = []
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)

    def load_current_points() -> None:
        nonlocal points
        roi_idx = int(failures[idx_pos])
        existing = np.asarray(kp_roi_arr[roi_idx], dtype=np.float64)
        if existing.shape == points.shape and np.isfinite(existing).any():
            points = existing.copy()
        else:
            points = np.full_like(points, np.nan)

    def update_display() -> None:
        roi_idx = int(failures[idx_pos])
        roi_img = roi_images[roi_idx]
        frame_idx = int(frame_indices[roi_idx])

        ax.clear()
        ax.imshow(roi_img, cmap="gray")

        for i, (label, color) in enumerate(zip(labels, colors)):
            if np.isfinite(points[i]).all():
                ax.scatter(points[i, 0], points[i, 1], s=60, c=color, edgecolors="black", linewidths=1.0)
                if show_text:
                    ax.text(points[i, 0] + 3, points[i, 1] - 3, label, color=color, fontsize=8, weight="bold")

        if show_text:
            active_label = labels[active_idx] if active_idx < len(labels) else "unknown"
            flag_label = ""
            if reason_arr is not None:
                try:
                    raw_reason = reason_arr[roi_idx]
                except Exception:
                    raw_reason = ""
                reason_text = str(raw_reason) if raw_reason is not None else ""
                if reason_text:
                    tags = [tag.strip() for tag in reason_text.split("|") if tag.strip()]
                    flagged = [tag for tag in tags if tag in {"fish_present_no_keypoints", "detection_issue"}]
                    if flagged:
                        flag_label = "Flagged: " + ", ".join(flagged)
            ax.set_title(
                f"ROI {roi_idx} | Frame {frame_idx} | {idx_pos + 1}/{len(failures)} "
                f"| Active: {active_label}",
                fontsize=10,
            )
            if flag_label:
                ax.text(
                    0.02,
                    0.02,
                    flag_label,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#f97316",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2),
                )
        else:
            ax.set_title("")
        ax.set_axis_off()
        fig.canvas.draw_idle()

    def save_current() -> None:
        nonlocal active_idx, idx_pos
        roi_idx = int(failures[idx_pos])
        frame_idx = int(frame_indices[roi_idx])
        if not np.isfinite(points).all():
            print("Set all three keypoints before saving.")
            return

        kp_roi_arr[roi_idx] = points
        full_points = points + roi_coords[roi_idx]
        if kp_img_arr is not None:
            kp_img_arr[roi_idx] = full_points
        if kp_norm_arr is not None:
            kp_norm_arr[roi_idx] = full_points / norm_factor

        heading_val = _compute_heading_from_points(points[0], points[1], points[2])
        if heading_arr is not None:
            heading_arr[roi_idx] = heading_val

        metrics = compute_geometry_metrics(points)
        max_ok = max_triangle_area is None or metrics.area <= max_triangle_area
        geom_ok = bool(
            np.isfinite(metrics.min_angle)
            and np.isfinite(metrics.area)
            and metrics.min_angle >= min_triangle_angle
            and metrics.area >= min_triangle_area
            and max_ok
        )

        if triangle_area_arr is not None:
            triangle_area_arr[roi_idx] = metrics.area
        if min_angle_arr is not None:
            min_angle_arr[roi_idx] = metrics.min_angle
        if triangle_angles_arr is not None:
            triangle_angles_arr[roi_idx] = metrics.angles

        conf_ok = True
        if conf_arr is not None:
            conf_vals = np.ones(len(labels), dtype=np.float64)
            conf_arr[roi_idx] = conf_vals
            conf_ok = bool(np.all(conf_vals >= confidence_threshold))

        if confidence_arr is not None:
            confidence_arr[roi_idx] = 1.0

        refined_success_val = True
        if refined_success_arr is not None:
            refined_success_arr[roi_idx] = refined_success_val
        if flip_corrected_arr is not None:
            flip_corrected_arr[roi_idx] = False
        if quality_labels_arr is not None:
            quality_labels_arr[roi_idx] = 0
        if confidence_valid_arr is not None:
            confidence_valid_arr[roi_idx] = conf_ok
        if geometry_valid_arr is not None:
            geometry_valid_arr[roi_idx] = geom_ok
        if usable_arr is not None:
            usable_arr[roi_idx] = conf_ok and geom_ok
        heading_is_finite = bool(np.isfinite(heading_val))
        if heading_finite_arr is not None:
            heading_finite_arr[roi_idx] = heading_is_finite
        if heading_usable_arr is not None:
            det_src = int(detection_source_arr[roi_idx]) if detection_source_arr is not None else 0
            heading_usable_arr[roi_idx] = refined_success_val and det_src == 0 and heading_is_finite
        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            existing_tags = [tag for tag in existing.split("|") if tag]
            drop_tags = {
                "detection_failed",
                "low_confidence",
                "confidence_missing",
                "fish_present_no_keypoints",
                "detection_issue",
            }
            kept_tags = [tag for tag in existing_tags if tag not in drop_tags and tag != "manual_correction"]
            tags = kept_tags + ["manual_correction"]
            if not geom_ok:
                tags.append("geometry_issue")
            unique: list[str] = []
            seen = set()
            for tag in tags:
                if not tag or tag in seen:
                    continue
                unique.append(tag)
                seen.add(tag)
            reason_value = "|".join(unique) if unique else "manual_correction"
            reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)

        stale_touched = mark_downstream_eye_mask_runs_stale(
            root,
            source_keypoint_group="refined_keypoints_runs",
            source_keypoints_run=str(refined_run),
            roi_indices=[roi_idx],
            frame_indices=[frame_idx],
            reason="keypoint_manual_correction",
        )
        print(f"Saved manual correction for ROI {roi_idx}.")
        if stale_touched:
            print(f"Marked {stale_touched} downstream eye-mask run(s) stale.")

        active_idx = 0
        if idx_pos < len(failures) - 1:
            idx_pos += 1
            load_current_points()
            update_display()

    def mark_no_keypoints() -> None:
        nonlocal active_idx, idx_pos, failures
        roi_idx = int(failures[idx_pos])
        frame_idx = int(frame_indices[roi_idx])

        if kp_roi_arr is not None:
            kp_roi_arr[roi_idx] = np.nan
        if kp_img_arr is not None:
            kp_img_arr[roi_idx] = np.nan
        if kp_norm_arr is not None:
            kp_norm_arr[roi_idx] = np.nan
        if heading_arr is not None:
            heading_arr[roi_idx] = np.nan
        if confidence_arr is not None:
            confidence_arr[roi_idx] = np.nan
        if conf_arr is not None:
            conf_arr[roi_idx] = np.nan
        if triangle_area_arr is not None:
            triangle_area_arr[roi_idx] = np.nan
        if min_angle_arr is not None:
            min_angle_arr[roi_idx] = np.nan
        if triangle_angles_arr is not None:
            triangle_angles_arr[roi_idx] = np.nan
        if refined_success_arr is not None:
            refined_success_arr[roi_idx] = False
        if flip_corrected_arr is not None:
            flip_corrected_arr[roi_idx] = False
        if quality_labels_arr is not None:
            quality_labels_arr[roi_idx] = 0
        if confidence_valid_arr is not None:
            confidence_valid_arr[roi_idx] = False
        if geometry_valid_arr is not None:
            geometry_valid_arr[roi_idx] = False
        if usable_arr is not None:
            usable_arr[roi_idx] = False
        if heading_finite_arr is not None:
            heading_finite_arr[roi_idx] = False
        if heading_usable_arr is not None:
            heading_usable_arr[roi_idx] = False
        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            reason_value = str(_clean_reason(existing, ["fish_present_no_keypoints"]))
            reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)

        stale_touched = mark_downstream_eye_mask_runs_stale(
            root,
            source_keypoint_group="refined_keypoints_runs",
            source_keypoints_run=str(refined_run),
            roi_indices=[roi_idx],
            frame_indices=[frame_idx],
            reason="keypoint_mark_no_keypoints",
        )
        print(f"Marked fish-present/no-keypoints for ROI {roi_idx} (frame {frame_idx}).")
        if stale_touched:
            print(f"Marked {stale_touched} downstream eye-mask run(s) stale.")

        failures = np.delete(failures, idx_pos)
        if failures.size == 0:
            plt.close(fig)
            return
        idx_pos = min(idx_pos, len(failures) - 1)
        active_idx = 0
        load_current_points()
        update_display()

    def mark_detection_issue() -> None:
        nonlocal active_idx, idx_pos, failures
        roi_idx = int(failures[idx_pos])
        frame_idx = int(frame_indices[roi_idx])

        if detect_frame_flag_path is not None:
            try:
                _append_detection_frame(detect_frame_flag_path, zarr_path, frame_idx)
            except Exception as exc:
                print(f"Failed to flag detection frame: {exc}")
        if detect_flag_path is not None:
            try:
                _append_flagged_path(detect_flag_path, zarr_path)
            except Exception as exc:
                print(f"Failed to flag detection path: {exc}")

        if kp_roi_arr is not None:
            kp_roi_arr[roi_idx] = np.nan
        if kp_img_arr is not None:
            kp_img_arr[roi_idx] = np.nan
        if kp_norm_arr is not None:
            kp_norm_arr[roi_idx] = np.nan
        if heading_arr is not None:
            heading_arr[roi_idx] = np.nan
        if confidence_arr is not None:
            confidence_arr[roi_idx] = np.nan
        if conf_arr is not None:
            conf_arr[roi_idx] = np.nan
        if triangle_area_arr is not None:
            triangle_area_arr[roi_idx] = np.nan
        if min_angle_arr is not None:
            min_angle_arr[roi_idx] = np.nan
        if triangle_angles_arr is not None:
            triangle_angles_arr[roi_idx] = np.nan
        if refined_success_arr is not None:
            refined_success_arr[roi_idx] = False
        if flip_corrected_arr is not None:
            flip_corrected_arr[roi_idx] = False
        if quality_labels_arr is not None:
            quality_labels_arr[roi_idx] = 0
        if confidence_valid_arr is not None:
            confidence_valid_arr[roi_idx] = False
        if geometry_valid_arr is not None:
            geometry_valid_arr[roi_idx] = False
        if usable_arr is not None:
            usable_arr[roi_idx] = False
        if heading_finite_arr is not None:
            heading_finite_arr[roi_idx] = False
        if heading_usable_arr is not None:
            heading_usable_arr[roi_idx] = False
        if reason_arr is not None:
            existing = str(reason_arr[roi_idx]) if reason_arr[roi_idx] is not None else ""
            reason_value = str(_clean_reason(existing, ["detection_issue"]))
            reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)

        stale_touched = mark_downstream_eye_mask_runs_stale(
            root,
            source_keypoint_group="refined_keypoints_runs",
            source_keypoints_run=str(refined_run),
            roi_indices=[roi_idx],
            frame_indices=[frame_idx],
            reason="keypoint_mark_detection_issue",
        )
        print(f"Marked detection issue for ROI {roi_idx} (frame {frame_idx}).")
        if stale_touched:
            print(f"Marked {stale_touched} downstream eye-mask run(s) stale.")

        failures = np.delete(failures, idx_pos)
        if failures.size == 0:
            plt.close(fig)
            return
        idx_pos = min(idx_pos, len(failures) - 1)
        active_idx = 0
        load_current_points()
        update_display()

    def next_failure() -> None:
        nonlocal idx_pos
        if idx_pos < len(failures) - 1:
            idx_pos += 1
            load_current_points()
            update_display()

    def prev_failure() -> None:
        nonlocal idx_pos
        if idx_pos > 0:
            idx_pos -= 1
            load_current_points()
            update_display()

    def reset_points() -> None:
        load_current_points()
        update_display()

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        points[active_idx] = [event.xdata, event.ydata]
        update_display()

    def on_key(event) -> None:
        nonlocal active_idx, show_text
        def apply_state(state: str) -> None:
            payload = _apply_review_status(
                refined_parent,
                refined_run,
                refined,
                state=state,
                method=review_method,
                intended_use=review_intended_use,
                reviewer=reviewer or os.environ.get("USER"),
                notes=review_notes,
            )
            print(f"✓ Keypoint review set: {payload.get('state')} ({payload.get('method')}/{payload.get('intended_use')})")

        if event.key in {"1", "2", "3"}:
            active_idx = int(event.key) - 1
            update_display()
        elif event.key == "n":
            next_failure()
        elif event.key == "p":
            prev_failure()
        elif event.key == "r":
            reset_points()
        elif event.key == "t":
            show_text = not show_text
            update_display()
        elif event.key == "s":
            save_current()
        elif event.key == "b":
            if flag_path is None:
                print("No frame flag file configured. Pass --frame-flag-file to enable frame flagging.")
            else:
                roi_idx = int(failures[idx_pos])
                frame_idx = int(frame_indices[roi_idx])
                try:
                    _append_flagged_frame(flag_path, zarr_path, frame_idx, roi_idx)
                    print(f"Flagged frame {frame_idx} (ROI {roi_idx}) for keypoint follow-up.")
                    print(f"Frame flag file: {flag_path}")
                except Exception as exc:
                    print(f"Failed to flag frame: {exc}")
        elif event.key == "d":
            mark_detection_issue()
        elif event.key == "x":
            mark_no_keypoints()
        elif event.key == "a":
            apply_state(review_state)
        elif event.key == "R":
            apply_state("rejected")
        elif event.key == "N":
            apply_state("needs_review")
        elif event.key == "P":
            apply_state("pending")
        elif event.key == "q":
            plt.close(fig)

    print("\nKeypoint Review")
    print(f"  Zarr: {zarr_path}")
    refined_label = f"{refined_run} (latest)" if using_latest else refined_run
    print(f"  Refined run: {refined_label}")
    print(f"  Crop run: {crop_run}")
    if targeted:
        print(f"  ROIs to review (target selection): {len(failures)}")
    elif include_all:
        print(f"  ROIs to review: {len(failures)}")
    else:
        print(f"  Failures to review: {len(failures)}")
    print("\nControls:")
    print("  Click: set active keypoint")
    print("  1/2/3: select bladder/left/right")
    print("  s: save correction")
    print("  t: toggle text overlays")
    print("  b: flag frame for follow-up (writes --frame-flag-file)")
    print("  d: flag detection issue (writes retune flags)")
    print("  x: mark fish present but no keypoints")
    print("  r: reset points from current data")
    print("  n/p: next/previous failure")
    print("  a: approve keypoints")
    print("  N: mark needs_review")
    print("  R: mark rejected")
    print("  P: mark pending")
    print("  q: quit")

    load_current_points()
    update_display()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if reason_arr is not None:
        _write_reason_labels(refined, np.asarray(reason_arr[:], dtype=object))


def main(argv: Optional[Sequence[str]] = None) -> None:
    raise SystemExit(
        "The keypoint_failure_review entrypoint has been removed. "
        "Use `python -m fisheye.tune.keypoint_review --manual`."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
