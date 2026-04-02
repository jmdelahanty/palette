#!/usr/bin/env python3
"""Interactive per-eye patch viewer for refined eye masks.

Displays, for each eye channel:
1. A local crop around the eye center.
2. The refined binary mask patch.
3. The crop with refined mask overlay + fitted ellipse.
4. The full ROI with that eye's mask + ellipse overlay.

Intended for quick visual QA of refined-eye-mask geometry.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Keep viewer resource usage bounded on shared workstations.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OPENCV_FOR_THREADS_NUM"] = "8"

import cv2
import numpy as np
import zarr

from ..shared.crop_image_source import CropImageSource
from ..shared.detect_reason_codec import decode_reason_bytes, write_reason_columns
from ..shared.mask_source import load_mask_bundle
from ..shared.provenance_attrs import resolve_source_keypoints_run
from ..utils.refined_eye_masks_compat import (
    refined_eye_masks_compat_context,
    refined_eye_masks_redirect_hint,
)

cv2.setNumThreads(8)

MAX_PADDING = 192
DEFAULT_PADDING = 24
PADDING_STEP = 2
MAX_SCALE_PERCENT = 600
DEFAULT_SCALE_PERCENT = 220
MAX_EDIT_ZOOM = 24
DEFAULT_EDIT_ZOOM = 6
BRUSH_MIN = 1
BRUSH_MAX = 40
DEFAULT_BRUSH = 6
BRUSH_STEP = 1
PANEL_LABEL_HEIGHT = 18
MASK_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 220, 255),   # left eye: amber
    (255, 128, 0),   # right eye: blue
    (140, 230, 120),
    (220, 120, 220),
)
MAJOR_AXIS_COLOR = (255, 255, 0)  # cyan
MINOR_AXIS_COLOR = (0, 0, 255)    # red
KEYPOINT_COLOR = (255, 0, 255)    # magenta
DEFAULT_RECOMMENDED_PROBABILITY_THRESHOLD = 0.45
PROBABILITY_TRACKBAR_MAX = 100
PROBABILITY_TRACKBAR_NAME = "Thr x100"
RECOMMENDED_PROBABILITY_THRESHOLD_ATTR = "recommended_probability_threshold"
CENTER_SOURCE_LABELS: Dict[str, str] = {
    "keypoint": "KP",
    "ellipse": "ELL",
    "mask_centroid": "MASK",
    "roi_center": "ROI",
}
REJECT_REASON_PRIORITY: Tuple[str, ...] = (
    "keypoint_fail",
    "no_region",
    "non_circular",
    "incomplete",
    "left_empty",
    "right_empty",
    "overlap",
    "too_close",
    "too_far",
)


def _load_frame_flags(path: Path) -> dict[str, list[dict[str, object]]]:
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
    parsed: dict[str, list[dict[str, object]]] = {}
    for key, value in data.items():
        entries: list[dict[str, object]] = []
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
                        payload: dict[str, object] = {"frame_idx": frame_idx, "roi_idx": roi_idx}
                        for extra_key in ("action", "preserve_eye_masks", "note", "requested_by"):
                            if extra_key not in item:
                                continue
                            extra_value = item.get(extra_key)
                            if isinstance(extra_value, (str, bool, int, float)) or extra_value is None:
                                payload[extra_key] = extra_value
                        entries.append(payload)
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
    *,
    extra_fields: Optional[dict[str, object]] = None,
) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_frame_flags(flag_path)
    entries = data.get(zarr_path, [])
    dedupe = {(entry.get("frame_idx"), entry.get("roi_idx")) for entry in entries}
    key = (int(frame_idx), int(roi_idx) if roi_idx is not None else None)
    if key in dedupe:
        return
    payload: dict[str, object] = {"frame_idx": int(frame_idx), "roi_idx": key[1]}
    if extra_fields:
        for extra_key, extra_value in extra_fields.items():
            if extra_key in {"frame_idx", "roi_idx"}:
                continue
            if isinstance(extra_value, (str, bool, int, float)) or extra_value is None:
                payload[extra_key] = extra_value
    entries.append(payload)
    entries.sort(key=lambda item: (item.get("frame_idx") or 0, item.get("roi_idx") or -1))
    data[zarr_path] = entries
    flag_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _derived_compat_write_guard(
    refined: zarr.Group,
    *,
    refined_run: Optional[str] = None,
    zarr_path: Optional[Path] = None,
) -> None:
    context = refined_eye_masks_compat_context(refined)
    if not bool(context.get("is_derived_compat")):
        return

    resolved_run = refined_run
    if not resolved_run:
        group_path = str(getattr(refined, "path", "") or "")
        if group_path:
            resolved_run = group_path.split("/")[-1]

    raise RuntimeError(
        refined_eye_masks_redirect_hint(
            refined_run=resolved_run,
            source=refined,
            zarr_path=zarr_path,
        )
    )


def _apply_eye_mask_review_status(
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
    _derived_compat_write_guard(refined, refined_run=refined_run)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, object] = {
        "state": state,
        "method": method,
        "intended_use": intended_use,
        "timestamp_utc": timestamp_utc,
        "timestamp": timestamp_utc,
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes

    for key in ("source_eye_masks_run", "source_keypoints_run", "source_keypoint_group"):
        value = refined.attrs.get(key)
        if value is not None:
            payload[key] = value

    refined.attrs["eye_mask_review_status"] = payload
    refined_parent.attrs["eye_mask_review_status_latest"] = refined_run
    return payload


def _format_review_status(status: object) -> str:
    if not isinstance(status, dict):
        return "none"
    state = str(status.get("state") or "").strip() or "review"
    method = str(status.get("method") or "").strip()
    intended_use = str(status.get("intended_use") or "").strip()
    suffix_parts = [part for part in (method, intended_use) if part]
    if suffix_parts:
        return f"{state} ({'/'.join(suffix_parts)})"
    return state


def _coerce_probability_threshold(value: object, *, default: float = DEFAULT_RECOMMENDED_PROBABILITY_THRESHOLD) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not 0.0 <= threshold <= 1.0:
        return float(default)
    return threshold


def _to_numpy_roi_slice(arr: object, roi_idx: int) -> np.ndarray:
    if arr is None:
        raise ValueError("Probability array is missing.")
    roi = arr[roi_idx]
    compute = getattr(roi, "compute", None)
    if callable(compute):
        roi = compute()
    return np.asarray(roi)


def _select_probability_preview_channel(
    probs_row: np.ndarray,
    *,
    eye_idx: int,
) -> Tuple[np.ndarray, str]:
    probs_row = np.asarray(probs_row, dtype=np.float32)
    if probs_row.ndim == 2:
        return probs_row, "Prob"
    if probs_row.ndim != 3:
        raise ValueError(f"Expected probability ROI slice with 2 or 3 dims, got shape {probs_row.shape}.")
    if probs_row.shape[0] == 1:
        return probs_row[0], "Union Prob"
    if eye_idx < probs_row.shape[0]:
        return probs_row[eye_idx], "Prob"
    return probs_row[0], "Prob"


def _apply_recommended_probability_threshold(
    source_run: zarr.Group,
    *,
    threshold: float,
    reviewer: Optional[str],
    notes: Optional[str],
    source_refined_run: Optional[str],
) -> Dict[str, object]:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, object] = {
        "threshold": float(threshold),
        "timestamp_utc": timestamp_utc,
        "timestamp": timestamp_utc,
        "source": "visualize_eye_mask_patches",
    }
    if reviewer:
        payload["reviewer"] = reviewer
    if notes:
        payload["notes"] = notes
    if source_refined_run:
        payload["source_refined_eye_masks_run"] = source_refined_run

    source_run.attrs[RECOMMENDED_PROBABILITY_THRESHOLD_ATTR] = float(threshold)
    source_run.attrs["recommended_probability_threshold_review"] = payload
    return payload


def _resolve_registry_dataset_row(
    registry: object,
    *,
    zarr_path: Path,
) -> Optional[Mapping[str, object]]:
    normalized = zarr_path.expanduser().resolve(strict=False)
    candidates = tuple(dict.fromkeys((str(zarr_path), str(zarr_path.expanduser()), str(normalized))))
    placeholders = ",".join("?" for _ in candidates)
    rows = registry.conn.execute(  # type: ignore[attr-defined]
        f"""
        SELECT dataset_id, recording_id, zarr_use, zarr_path
        FROM datasets
        WHERE zarr_path IN ({placeholders})
        ORDER BY CASE WHEN recording_id IS NULL THEN 1 ELSE 0 END, dataset_id;
        """,
        candidates,
    ).fetchall()
    if rows:
        return rows[0]

    # Fallback for path normalization mismatches (symlinked roots, etc.).
    fallback_rows = registry.conn.execute(  # type: ignore[attr-defined]
        """
        SELECT dataset_id, recording_id, zarr_use, zarr_path
        FROM datasets
        WHERE zarr_path IS NOT NULL;
        """
    ).fetchall()
    for row in fallback_rows:
        try:
            db_path = Path(str(row["zarr_path"])).expanduser().resolve(strict=False)
        except Exception:
            continue
        if db_path == normalized:
            return row
    return None


def _sync_registry_for_zarr(
    *,
    registry_path: Path,
    zarr_path: Path,
) -> Tuple[bool, str]:
    from ..registry.db import Registry
    from ..registry.maintenance import _backfill_recording_step_status

    registry = Registry(registry_path)
    try:
        row = _resolve_registry_dataset_row(registry, zarr_path=zarr_path)
        if row is None:
            return False, f"no dataset row found for {zarr_path}"

        dataset_id = str(row["dataset_id"])
        recording_id_raw = row["recording_id"]
        recording_id = str(recording_id_raw) if recording_id_raw else None
        zarr_use_raw = row["zarr_use"]
        zarr_use = str(zarr_use_raw).strip().lower() if zarr_use_raw else None
        if zarr_use not in {"analysis", "training"}:
            zarr_use = None

        eye_rows = registry.refresh_eye_mask_performance_for_dataset(
            dataset_id,
            zarr_path=zarr_path.expanduser().resolve(strict=False),
            recording_id=recording_id,
            zarr_use=zarr_use,
        )

        step_rows_updated = 0
        if recording_id:
            step_summary = _backfill_recording_step_status(
                registry,
                dry_run=False,
                scope_paths=[zarr_path],
                recording_ids=[recording_id],
                zarr_use_filter=zarr_use or "all",
            )
            step_rows_updated = int(step_summary.get("rows_inserted", 0)) + int(step_summary.get("rows_updated", 0))

        return True, (
            f"dataset_id={dataset_id} eye_mask_rows={eye_rows} "
            f"recording_step_rows_written={step_rows_updated}"
        )
    except Exception as exc:
        return False, str(exc)
    finally:
        registry.close()


def _refresh_refined_eye_mask_metrics(root: zarr.Group, refined: zarr.Group) -> Dict[str, object]:
    from ..tune.eye_mask_review import _update_postprocess_summary

    return _update_postprocess_summary(root, refined, print_summary=False)


def _friendly_eye_label(label: Optional[str], idx: int) -> str:
    if label is None:
        return f"Eye {idx + 1}"
    value = str(label).strip().lower()
    if value in {"eye_left", "left", "left_eye"}:
        return "Left"
    if value in {"eye_right", "right", "right_eye"}:
        return "Right"
    if value.startswith("eye_"):
        suffix = value[4:]
        if suffix.isdigit():
            return f"Eye {int(suffix) + 1}"
    return str(label)


def _extract_patch_bounds(
    roi_shape: Tuple[int, int],
    center_xy: Tuple[float, float],
    padding: int,
) -> Tuple[int, int, int, int]:
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    pad = max(1, int(padding))

    x0 = max(0, int(round(cx)) - pad)
    x1 = min(roi_w, int(round(cx)) + pad + 1)
    y0 = max(0, int(round(cy)) - pad)
    y1 = min(roi_h, int(round(cy)) + pad + 1)
    return x0, x1, y0, y1


def _mask_centroid_xy(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _resolve_eye_center(
    eye_idx: int,
    keypoints_row: Optional[np.ndarray],
    ellipse_row: np.ndarray,
    ellipse_ok: bool,
    mask_row: np.ndarray,
    roi_shape: Tuple[int, int],
) -> Tuple[float, float]:
    center_xy, _source = _resolve_eye_center_with_source(
        eye_idx,
        keypoints_row,
        ellipse_row,
        ellipse_ok,
        mask_row,
        roi_shape,
    )
    return center_xy


def _resolve_eye_center_with_source(
    eye_idx: int,
    keypoints_row: Optional[np.ndarray],
    ellipse_row: np.ndarray,
    ellipse_ok: bool,
    mask_row: np.ndarray,
    roi_shape: Tuple[int, int],
) -> Tuple[Tuple[float, float], str]:
    # Prefer keypoint centers when available (same framing intent as tuner).
    if keypoints_row is not None and keypoints_row.ndim == 2:
        kp_idx = 1 + int(eye_idx)
        if kp_idx < keypoints_row.shape[0]:
            kp = np.asarray(keypoints_row[kp_idx], dtype=np.float32)
            if kp.shape[0] >= 2 and np.all(np.isfinite(kp[:2])):
                return (float(kp[0]), float(kp[1])), "keypoint"

    if ellipse_ok:
        cx, cy = float(ellipse_row[0]), float(ellipse_row[1])
        if np.isfinite(cx) and np.isfinite(cy):
            return (cx, cy), "ellipse"

    centroid = _mask_centroid_xy(mask_row)
    if centroid is not None:
        return centroid, "mask_centroid"

    # Final fallback: ROI center.
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    return (float(roi_w) / 2.0, float(roi_h) / 2.0), "roi_center"


def _extract_eye_keypoint_xy(
    keypoints_row: Optional[np.ndarray],
    eye_idx: int,
) -> Optional[Tuple[float, float]]:
    if keypoints_row is None or keypoints_row.ndim != 2:
        return None
    kp_idx = 1 + int(eye_idx)
    if kp_idx < 0 or kp_idx >= int(keypoints_row.shape[0]):
        return None
    kp = np.asarray(keypoints_row[kp_idx], dtype=np.float32)
    if kp.shape[0] < 2 or not np.all(np.isfinite(kp[:2])):
        return None
    return float(kp[0]), float(kp[1])


def _draw_ellipse_overlay(
    panel: np.ndarray,
    ellipse_row: np.ndarray,
    *,
    x_offset: int,
    y_offset: int,
) -> None:
    if ellipse_row.shape[0] < 5:
        return
    cx, cy, major, minor, theta = (
        float(ellipse_row[0]),
        float(ellipse_row[1]),
        float(ellipse_row[2]),
        float(ellipse_row[3]),
        float(ellipse_row[4]),
    )
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(major) and np.isfinite(minor) and np.isfinite(theta)):
        return
    if major <= 0.0 or minor <= 0.0:
        return

    local_cx = cx - float(x_offset)
    local_cy = cy - float(y_offset)
    center = (int(round(local_cx)), int(round(local_cy)))
    axes = (max(1, int(round(major / 2.0))), max(1, int(round(minor / 2.0))))
    angle = float(theta)

    cv2.ellipse(panel, center, axes, angle, 0.0, 360.0, (0, 255, 0), 1, cv2.LINE_AA)

    theta_rad = np.deg2rad(theta)
    major_dx = np.cos(theta_rad) * (major / 2.0)
    major_dy = np.sin(theta_rad) * (major / 2.0)
    major_p1 = (int(round(local_cx - major_dx)), int(round(local_cy - major_dy)))
    major_p2 = (int(round(local_cx + major_dx)), int(round(local_cy + major_dy)))
    cv2.line(panel, major_p1, major_p2, MAJOR_AXIS_COLOR, 1, cv2.LINE_AA)

    theta_minor = theta_rad + (np.pi / 2.0)
    minor_dx = np.cos(theta_minor) * (minor / 2.0)
    minor_dy = np.sin(theta_minor) * (minor / 2.0)
    minor_p1 = (int(round(local_cx - minor_dx)), int(round(local_cy - minor_dy)))
    minor_p2 = (int(round(local_cx + minor_dx)), int(round(local_cy + minor_dy)))
    cv2.line(panel, minor_p1, minor_p2, MINOR_AXIS_COLOR, 1, cv2.LINE_AA)

    cv2.circle(panel, center, 2, MAJOR_AXIS_COLOR, -1, cv2.LINE_AA)


def _labeled_panel(image_bgr: np.ndarray, label: str) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    canvas = np.zeros((h + PANEL_LABEL_HEIGHT, w, 3), dtype=np.uint8)
    canvas[PANEL_LABEL_HEIGHT:, :] = image_bgr
    cv2.putText(
        canvas,
        label,
        (4, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 255, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _stack_h(panels: Sequence[np.ndarray], gap: int = 6) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(panel.shape[0] for panel in panels)
    padded: List[np.ndarray] = []
    for panel in panels:
        pad_bottom = max(0, h - panel.shape[0])
        if pad_bottom:
            panel = np.pad(panel, ((0, pad_bottom), (0, 0), (0, 0)), mode="constant")
        padded.append(panel)
    if gap <= 0:
        return np.hstack(padded)
    spacer = np.zeros((h, gap, 3), dtype=np.uint8)
    out = padded[0]
    for panel in padded[1:]:
        out = np.hstack([out, spacer, panel])
    return out


def _stack_v(panels: Sequence[np.ndarray], gap: int = 8) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    w = max(panel.shape[1] for panel in panels)
    padded: List[np.ndarray] = []
    for panel in panels:
        pad_right = max(0, w - panel.shape[1])
        if pad_right:
            panel = np.pad(panel, ((0, 0), (0, pad_right), (0, 0)), mode="constant")
        padded.append(panel)
    if gap <= 0:
        return np.vstack(padded)
    spacer = np.zeros((gap, w, 3), dtype=np.uint8)
    out = padded[0]
    for panel in padded[1:]:
        out = np.vstack([out, spacer, panel])
    return out


def _merge_reason(existing: str, tags: Sequence[str]) -> str:
    merged: List[str] = []
    seen: set[str] = set()
    for tag in [*existing.split("|"), *[str(value) for value in tags]]:
        cleaned = str(tag).strip()
        if not cleaned or cleaned in seen:
            continue
        merged.append(cleaned)
        seen.add(cleaned)
    return "|".join(merged) if merged else "clean"


def _read_reason_value(
    *,
    reason_arr: Optional[zarr.Array],
    reason_bytes_arr: Optional[zarr.Array],
    roi_idx: int,
) -> str:
    if isinstance(reason_arr, zarr.Array):
        raw_value = reason_arr[roi_idx]
        return "" if raw_value is None else str(raw_value)
    if isinstance(reason_bytes_arr, zarr.Array):
        encoded = np.asarray(reason_bytes_arr[roi_idx:roi_idx + 1], dtype=np.uint8)
        if encoded.shape[0] == 1:
            return str(decode_reason_bytes(encoded)[0])
    return ""


def _write_reason_value(
    *,
    reason_group: Optional[zarr.Group],
    reason_arr: Optional[zarr.Array],
    reason_bytes_arr: Optional[zarr.Array],
    roi_idx: int,
    reason_value: str,
) -> None:
    if not isinstance(reason_bytes_arr, zarr.Array):
        if isinstance(reason_arr, zarr.Array):
            reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)
        return
    if not isinstance(reason_group, zarr.Group):
        raise RuntimeError("reason_group is required to rewrite synchronized reason_bytes columns.")

    if isinstance(reason_arr, zarr.Array):
        labels = np.asarray(reason_arr[:], dtype=object)
    else:
        labels = decode_reason_bytes(np.asarray(reason_bytes_arr[:], dtype=np.uint8))
    labels = np.asarray(labels, dtype=object)
    labels[roi_idx] = reason_value

    chunk_size = int(labels.shape[0]) or 1
    row_chunks = getattr(reason_bytes_arr, "chunks", None)
    if isinstance(row_chunks, tuple) and row_chunks:
        chunk_size = max(1, int(row_chunks[0]))
    elif isinstance(reason_arr, zarr.Array):
        reason_chunks = getattr(reason_arr, "chunks", None)
        if isinstance(reason_chunks, tuple) and reason_chunks:
            chunk_size = max(1, int(reason_chunks[0]))
        elif isinstance(reason_chunks, int):
            chunk_size = max(1, int(reason_chunks))

    write_reason_columns(
        reason_group,
        labels,
        chunk_size=chunk_size,
        include_reason_text=isinstance(reason_arr, zarr.Array),
        overwrite=True,
    )


def _reason_tags_from_value(reason_value: object) -> List[str]:
    if reason_value is None:
        return []
    if isinstance(reason_value, bytes):
        text = reason_value.decode("utf-8", errors="ignore")
    else:
        text = str(reason_value)
    parts = [part.strip() for part in text.split("|") if part and part.strip()]
    return parts


def _extract_primary_reject_reason(reason_tags: Sequence[str]) -> Optional[str]:
    tag_set = set(str(tag) for tag in reason_tags)
    for tag in REJECT_REASON_PRIORITY:
        if tag in tag_set:
            return tag
    return None


def _format_reason_tags_compact(reason_tags: Sequence[str], max_items: int = 4) -> str:
    tags = [str(tag) for tag in reason_tags if str(tag)]
    if not tags:
        return "none"
    if len(tags) <= max_items:
        return "|".join(tags)
    return f"{'|'.join(tags[:max_items])}|+{len(tags) - max_items}"


def _get_sep_limits(root: zarr.Group, refined: zarr.Group) -> tuple[Optional[float], Optional[float]]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        tuning = analysis.attrs.get("eye_mask_tuning")
        if isinstance(tuning, dict):
            params = tuning.get("tuned_parameters", {})
            min_sep = params.get("min_eye_separation")
            max_sep = params.get("max_eye_separation")
            return (
                float(min_sep) if min_sep is not None else None,
                float(max_sep) if max_sep is not None else None,
            )

    source_run = refined.attrs.get("source_eye_masks_run")
    if source_run and "eye_masks_runs" in root and source_run in root["eye_masks_runs"]:
        src = root["eye_masks_runs"][source_run]
        min_sep = src.attrs.get("min_eye_separation")
        max_sep = src.attrs.get("max_eye_separation")
        return (
            float(min_sep) if min_sep is not None else None,
            float(max_sep) if max_sep is not None else None,
        )
    return None, None


def _fit_ellipse_from_mask(
    mask: np.ndarray,
) -> Tuple[np.ndarray, bool, Optional[np.ndarray], Optional[Tuple[float, float]]]:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask_u8)
    if ys.size == 0:
        return np.full(5, np.nan, dtype=np.float32), False, None, None

    centroid = (float(xs.mean()), float(ys.mean()))
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.full(5, np.nan, dtype=np.float32), False, None, centroid

    contour = max(contours, key=cv2.contourArea)
    contour_xy = contour.reshape(-1, 2).astype(np.float32)
    if contour_xy.shape[0] < 5:
        return np.full(5, np.nan, dtype=np.float32), False, contour_xy, centroid

    (xc, yc), (axis_a, axis_b), angle = cv2.fitEllipse(contour_xy)
    major = float(axis_a)
    minor = float(axis_b)
    theta = float(angle)
    if major < minor:
        major, minor = minor, major
        theta += 90.0
    theta = float((theta + 180.0) % 180.0)

    params = np.array([float(xc), float(yc), major, minor, theta], dtype=np.float32)
    if not np.all(np.isfinite(params[:4])) or major <= 0.0 or minor <= 0.0:
        return np.full(5, np.nan, dtype=np.float32), False, contour_xy, centroid
    return params, True, contour_xy, centroid


def _update_contour_arrays(
    refined: zarr.Group,
    roi_idx: int,
    contour: Optional[np.ndarray],
    *,
    side: str,
) -> None:
    ptr_name = f"contour_{side}_ptr"
    len_name = f"contour_{side}_len"
    cont_name = f"contours_{side}"

    if ptr_name not in refined or len_name not in refined or cont_name not in refined:
        return

    ptr_arr = refined[ptr_name]
    len_arr = refined[len_name]
    cont_arr = refined[cont_name]

    if contour is None or contour.size == 0:
        ptr_arr[roi_idx] = -1
        len_arr[roi_idx] = 0
        return

    contour = np.asarray(contour, dtype=np.float32)
    n_points = contour.shape[0]
    existing_ptr = int(ptr_arr[roi_idx])
    existing_len = int(len_arr[roi_idx])

    if existing_ptr >= 0 and existing_len >= n_points:
        cont_arr[existing_ptr:existing_ptr + n_points] = contour
        len_arr[roi_idx] = n_points
        return

    current_len = cont_arr.shape[0]
    new_len = current_len + n_points
    try:
        cont_arr.resize((new_len, 2))
    except Exception:
        old = cont_arr[:]
        del refined[cont_name]
        cont_arr = refined.create_array(
            cont_name,
            shape=(new_len, 2),
            chunks=(max(1, min(4096, new_len)), 2),
            dtype=old.dtype,
            overwrite=True,
        )
        cont_arr[:old.shape[0]] = old
    cont_arr[current_len:new_len] = contour
    ptr_arr[roi_idx] = current_len
    len_arr[roi_idx] = n_points


def _resolve_keypoint_group(
    root: zarr.Group,
    refined_group: zarr.Group,
    *,
    explicit_run: Optional[str],
    explicit_group: Optional[str],
) -> Tuple[Optional[zarr.Group], Optional[str], Optional[str]]:
    if explicit_group and explicit_run:
        parent = root.get(explicit_group)
        if isinstance(parent, zarr.Group) and explicit_run in parent:
            return parent[explicit_run], explicit_group, explicit_run
        return None, None, None

    if explicit_run and not explicit_group:
        for group_name in ("refined_keypoints_runs", "keypoints_runs"):
            parent = root.get(group_name)
            if isinstance(parent, zarr.Group) and explicit_run in parent:
                return parent[explicit_run], group_name, explicit_run
        return None, None, None

    source_group = refined_group.attrs.get("source_keypoint_group")
    source_run = resolve_source_keypoints_run(refined_group.attrs)
    if isinstance(source_group, str) and isinstance(source_run, str):
        parent = root.get(source_group)
        if isinstance(parent, zarr.Group) and source_run in parent:
            return parent[source_run], source_group, source_run

    for group_name in ("refined_keypoints_runs", "keypoints_runs"):
        parent = root.get(group_name)
        if isinstance(parent, zarr.Group):
            latest = parent.attrs.get("latest")
            if isinstance(latest, str) and latest in parent:
                return parent[latest], group_name, latest
    return None, None, None


def _get_latest_run(root: zarr.Group, parent_name: str) -> str:
    parent = root.get(parent_name)
    if not isinstance(parent, zarr.Group):
        raise RuntimeError(f"'{parent_name}' not found in archive.")
    latest = parent.attrs.get("latest")
    if isinstance(latest, str) and latest in parent:
        return latest
    keys = sorted(parent.group_keys()) if hasattr(parent, "group_keys") else sorted(parent.keys())
    if not keys:
        raise RuntimeError(f"No runs found under '{parent_name}'.")
    return str(keys[-1])


def _probability_heatmap_panel(prob_patch: np.ndarray) -> np.ndarray:
    prob_patch = np.clip(np.asarray(prob_patch, dtype=np.float32), 0.0, 1.0)
    prob_u8 = np.round(prob_patch * 255.0).astype(np.uint8)
    return cv2.applyColorMap(prob_u8, cv2.COLORMAP_TURBO)


def _build_eye_row(
    roi_img: np.ndarray,
    mask_eye: np.ndarray,
    ellipse_row: np.ndarray,
    ellipse_ok: bool,
    center_xy: Tuple[float, float],
    *,
    padding: int,
    eye_label: str,
    eye_idx: int,
    show_ellipse_overlay: bool = True,
    keypoint_xy: Optional[Tuple[float, float]] = None,
    show_keypoint_overlay: bool = False,
    probability_eye: Optional[np.ndarray] = None,
    probability_label: Optional[str] = None,
    probability_threshold: float = DEFAULT_RECOMMENDED_PROBABILITY_THRESHOLD,
) -> np.ndarray:
    x0, x1, y0, y1 = _extract_patch_bounds(roi_img.shape, center_xy, padding)
    crop_patch = np.asarray(roi_img[y0:y1, x0:x1], dtype=np.uint8)
    mask_patch = (np.asarray(mask_eye[y0:y1, x0:x1]) > 0).astype(np.uint8)

    crop_panel = cv2.cvtColor(crop_patch, cv2.COLOR_GRAY2BGR)
    mask_panel = cv2.cvtColor(mask_patch * 255, cv2.COLOR_GRAY2BGR)
    prob_panel = None
    threshold_panel = None

    if probability_eye is not None:
        prob_patch = np.asarray(probability_eye[y0:y1, x0:x1], dtype=np.float32)
        prob_panel = _probability_heatmap_panel(prob_patch)
        threshold_patch = (prob_patch >= float(probability_threshold)).astype(np.uint8)
        threshold_panel = cv2.cvtColor(threshold_patch * 255, cv2.COLOR_GRAY2BGR)

    fit_panel = crop_panel.copy()
    color = MASK_COLORS_BGR[eye_idx % len(MASK_COLORS_BGR)]
    overlay = fit_panel.copy()
    overlay[mask_patch > 0] = color
    fit_panel = cv2.addWeighted(overlay, 0.40, fit_panel, 0.60, 0.0)
    if ellipse_ok and show_ellipse_overlay:
        _draw_ellipse_overlay(fit_panel, ellipse_row, x_offset=x0, y_offset=y0)

    cx_local = int(round(float(center_xy[0]) - x0))
    cy_local = int(round(float(center_xy[1]) - y0))
    cv2.drawMarker(crop_panel, (cx_local, cy_local), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
    cv2.drawMarker(fit_panel, (cx_local, cy_local), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
    if show_keypoint_overlay and keypoint_xy is not None:
        kp_local = (
            int(round(float(keypoint_xy[0]) - x0)),
            int(round(float(keypoint_xy[1]) - y0)),
        )
        cv2.drawMarker(crop_panel, kp_local, KEYPOINT_COLOR, cv2.MARKER_TILTED_CROSS, 10, 1)
        cv2.drawMarker(fit_panel, kp_local, KEYPOINT_COLOR, cv2.MARKER_TILTED_CROSS, 10, 1)

    full_panel = cv2.cvtColor(np.asarray(roi_img, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    full_overlay = full_panel.copy()
    full_mask = np.asarray(mask_eye) > 0
    full_overlay[full_mask] = color
    full_panel = cv2.addWeighted(full_overlay, 0.40, full_panel, 0.60, 0.0)
    if ellipse_ok and show_ellipse_overlay:
        _draw_ellipse_overlay(full_panel, ellipse_row, x_offset=0, y_offset=0)
    cv2.drawMarker(
        full_panel,
        (int(round(float(center_xy[0]))), int(round(float(center_xy[1])))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        8,
        1,
    )
    if show_keypoint_overlay and keypoint_xy is not None:
        cv2.drawMarker(
            full_panel,
            (int(round(float(keypoint_xy[0]))), int(round(float(keypoint_xy[1])))),
            KEYPOINT_COLOR,
            cv2.MARKER_TILTED_CROSS,
            10,
            1,
        )

    panel_crop = _labeled_panel(crop_panel, f"{eye_label} Crop")
    panel_mask = _labeled_panel(mask_panel, f"{eye_label} Mask")
    panel_fit = _labeled_panel(fit_panel, f"{eye_label} Fit")
    panel_full = _labeled_panel(full_panel, f"{eye_label} Full ROI")
    panels = [panel_crop, panel_mask]
    if prob_panel is not None and threshold_panel is not None:
        panels.append(_labeled_panel(prob_panel, f"{eye_label} {probability_label or 'Prob'}"))
        panels.append(
            _labeled_panel(
                threshold_panel,
                f"{eye_label} Thr@{float(probability_threshold):.2f}",
            )
        )
    panels.extend([panel_fit, panel_full])
    return _stack_h(panels, gap=6)


def _build_eye_row_with_editor(
    roi_img: np.ndarray,
    mask_eye: np.ndarray,
    ellipse_row: np.ndarray,
    ellipse_ok: bool,
    center_xy: Tuple[float, float],
    *,
    padding: int,
    eye_label: str,
    eye_idx: int,
    edit_zoom: int,
    active_eye: bool,
    brush_radius: int,
    cursor_patch_xy: Optional[Tuple[int, int]],
    show_ellipse_overlay: bool = True,
    keypoint_xy: Optional[Tuple[float, float]] = None,
    show_keypoint_overlay: bool = False,
    probability_eye: Optional[np.ndarray] = None,
    probability_label: Optional[str] = None,
    probability_threshold: float = DEFAULT_RECOMMENDED_PROBABILITY_THRESHOLD,
) -> Tuple[np.ndarray, Dict[str, int]]:
    x0, x1, y0, y1 = _extract_patch_bounds(roi_img.shape, center_xy, padding)
    crop_patch = np.asarray(roi_img[y0:y1, x0:x1], dtype=np.uint8)
    mask_patch = (np.asarray(mask_eye[y0:y1, x0:x1]) > 0).astype(np.uint8)

    crop_panel = cv2.cvtColor(crop_patch, cv2.COLOR_GRAY2BGR)
    mask_panel = cv2.cvtColor(mask_patch * 255, cv2.COLOR_GRAY2BGR)
    prob_panel = None
    threshold_panel = None

    if probability_eye is not None:
        prob_patch = np.asarray(probability_eye[y0:y1, x0:x1], dtype=np.float32)
        prob_panel = _probability_heatmap_panel(prob_patch)
        threshold_patch = (prob_patch >= float(probability_threshold)).astype(np.uint8)
        threshold_panel = cv2.cvtColor(threshold_patch * 255, cv2.COLOR_GRAY2BGR)

    fit_panel = crop_panel.copy()
    color = MASK_COLORS_BGR[eye_idx % len(MASK_COLORS_BGR)]
    overlay = fit_panel.copy()
    overlay[mask_patch > 0] = color
    fit_panel = cv2.addWeighted(overlay, 0.40, fit_panel, 0.60, 0.0)
    if ellipse_ok and show_ellipse_overlay:
        _draw_ellipse_overlay(fit_panel, ellipse_row, x_offset=x0, y_offset=y0)

    edit_panel = fit_panel.copy()
    zoom = max(1, int(edit_zoom))
    if zoom > 1:
        edit_panel = cv2.resize(
            edit_panel,
            None,
            fx=float(zoom),
            fy=float(zoom),
            interpolation=cv2.INTER_NEAREST,
        )

    if cursor_patch_xy is not None and active_eye:
        cx, cy = cursor_patch_xy
        cv2.circle(
            edit_panel,
            (int(round(cx * zoom)), int(round(cy * zoom))),
            max(1, int(round(brush_radius * zoom))),
            color,
            1,
            cv2.LINE_AA,
        )

    cx_local = int(round(float(center_xy[0]) - x0))
    cy_local = int(round(float(center_xy[1]) - y0))
    cv2.drawMarker(crop_panel, (cx_local, cy_local), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
    cv2.drawMarker(fit_panel, (cx_local, cy_local), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
    if show_keypoint_overlay and keypoint_xy is not None:
        kp_local = (
            int(round(float(keypoint_xy[0]) - x0)),
            int(round(float(keypoint_xy[1]) - y0)),
        )
        cv2.drawMarker(crop_panel, kp_local, KEYPOINT_COLOR, cv2.MARKER_TILTED_CROSS, 10, 1)
        cv2.drawMarker(fit_panel, kp_local, KEYPOINT_COLOR, cv2.MARKER_TILTED_CROSS, 10, 1)
        cv2.drawMarker(
            edit_panel,
            (int(round(kp_local[0] * zoom)), int(round(kp_local[1] * zoom))),
            KEYPOINT_COLOR,
            cv2.MARKER_TILTED_CROSS,
            max(10, int(round(10 * zoom))),
            max(1, int(round(zoom / 2))),
        )

    full_panel = cv2.cvtColor(np.asarray(roi_img, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    full_overlay = full_panel.copy()
    full_mask = np.asarray(mask_eye) > 0
    full_overlay[full_mask] = color
    full_panel = cv2.addWeighted(full_overlay, 0.40, full_panel, 0.60, 0.0)
    if ellipse_ok and show_ellipse_overlay:
        _draw_ellipse_overlay(full_panel, ellipse_row, x_offset=0, y_offset=0)
    cv2.drawMarker(
        full_panel,
        (int(round(float(center_xy[0]))), int(round(float(center_xy[1])))),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        8,
        1,
    )
    if show_keypoint_overlay and keypoint_xy is not None:
        cv2.drawMarker(
            full_panel,
            (int(round(float(keypoint_xy[0]))), int(round(float(keypoint_xy[1])))),
            KEYPOINT_COLOR,
            cv2.MARKER_TILTED_CROSS,
            10,
            1,
        )

    panel_crop = _labeled_panel(crop_panel, f"{eye_label} Crop")
    panel_mask = _labeled_panel(mask_panel, f"{eye_label} Mask")
    panel_fit = _labeled_panel(fit_panel, f"{eye_label} Fit")
    panel_edit = _labeled_panel(edit_panel, f"{eye_label} Edit")
    panel_full = _labeled_panel(full_panel, f"{eye_label} Full ROI")

    if active_eye:
        cv2.rectangle(
            panel_edit,
            (0, 0),
            (panel_edit.shape[1] - 1, panel_edit.shape[0] - 1),
            color,
            2,
            cv2.LINE_AA,
        )

    panels = [panel_crop, panel_mask]
    if prob_panel is not None and threshold_panel is not None:
        panels.append(_labeled_panel(prob_panel, f"{eye_label} {probability_label or 'Prob'}"))
        panels.append(
            _labeled_panel(
                threshold_panel,
                f"{eye_label} Thr@{float(probability_threshold):.2f}",
            )
        )
    panels.extend([panel_fit, panel_edit, panel_full])
    row = _stack_h(panels, gap=6)

    widths = [panel.shape[1] for panel in panels]
    edit_panel_index = len(panels) - 2
    edit_x = sum(widths[:edit_panel_index]) + (6 * edit_panel_index)
    edit_meta = {
        "eye_idx": int(eye_idx),
        "x": int(edit_x),
        "y": 0,
        "w": int(panel_edit.shape[1]),
        "h": int(panel_edit.shape[0]),
        "label_h": int(PANEL_LABEL_HEIGHT),
        "patch_x0": int(x0),
        "patch_y0": int(y0),
        "patch_w": int(crop_patch.shape[1]),
        "patch_h": int(crop_patch.shape[0]),
        "zoom": int(zoom),
    }
    return row, edit_meta


def _compose_rows_with_edit_meta(
    rows: Sequence[np.ndarray],
    metas: Sequence[Dict[str, int]],
    *,
    gap: int = 8,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    if not rows:
        return np.zeros((1, 1, 3), dtype=np.uint8), []

    width = max(row.shape[1] for row in rows)
    total_h = sum(row.shape[0] for row in rows) + max(0, len(rows) - 1) * max(0, int(gap))
    canvas = np.zeros((total_h, width, 3), dtype=np.uint8)

    y = 0
    global_meta: List[Dict[str, int]] = []
    for row, meta in zip(rows, metas):
        h, w = row.shape[:2]
        canvas[y:y + h, :w] = row
        out = dict(meta)
        out["y"] = int(meta["y"] + y)
        global_meta.append(out)
        y += h + max(0, int(gap))

    return canvas, global_meta


def _save_roi_mask_edits(
    *,
    root: zarr.Group,
    refined: zarr.Group,
    roi_idx: int,
    masks_row: np.ndarray,
    masks_arr: zarr.Array,
    edit_applied_arr: Optional[zarr.Array],
    ellipse_params_arr: zarr.Array,
    ellipse_success_arr: zarr.Array,
    eye_separation_arr: Optional[zarr.Array],
    reason_arr: Optional[zarr.Array],
    reason_bytes_arr: Optional[zarr.Array],
    reason_group: Optional[zarr.Group] = None,
) -> Dict[str, object]:
    _derived_compat_write_guard(refined)
    existing_masks = np.asarray(masks_arr[roi_idx], dtype=np.uint8)
    channel_count = int(masks_row.shape[0])
    new_params = np.full((channel_count, 5), np.nan, dtype=np.float32)
    new_success = np.zeros((channel_count,), dtype=bool)

    for eye_idx in range(channel_count):
        params, success, contour_xy, centroid = _fit_ellipse_from_mask(masks_row[eye_idx])
        new_params[eye_idx] = params
        new_success[eye_idx] = bool(success)
        if eye_idx == 0:
            _update_contour_arrays(refined, roi_idx, contour_xy, side="left")
        elif eye_idx == 1:
            _update_contour_arrays(refined, roi_idx, contour_xy, side="right")

    separation = float("nan")
    if (
        channel_count >= 2
        and bool(new_success[0])
        and bool(new_success[1])
        and np.all(np.isfinite(new_params[0, :2]))
        and np.all(np.isfinite(new_params[1, :2]))
    ):
        separation = float(
            np.hypot(
                float(new_params[0, 0]) - float(new_params[1, 0]),
                float(new_params[0, 1]) - float(new_params[1, 1]),
            )
        )

    reject_reason: Optional[str] = None
    min_sep, max_sep = _get_sep_limits(root, refined)
    if channel_count >= 2:
        left_success = bool(new_success[0])
        right_success = bool(new_success[1])
        if not left_success and not right_success:
            reject_reason = "incomplete"
        elif not left_success:
            reject_reason = "left_empty"
        elif not right_success:
            reject_reason = "right_empty"
        else:
            overlap = np.logical_and(masks_row[0] > 0, masks_row[1] > 0).any()
            if overlap:
                reject_reason = "overlap"
            elif np.isfinite(separation):
                if min_sep is not None and separation < min_sep:
                    reject_reason = "too_close"
                elif max_sep is not None and separation > max_sep:
                    reject_reason = "too_far"

    next_masks = np.asarray(masks_row, dtype=np.uint8)
    changed_channels = np.any(existing_masks != next_masks, axis=(1, 2))
    masks_arr[roi_idx] = next_masks
    if edit_applied_arr is not None and np.any(changed_channels):
        prior = np.asarray(edit_applied_arr[roi_idx], dtype=bool)
        edit_applied_arr[roi_idx] = np.logical_or(prior, changed_channels)
    ellipse_params_arr[roi_idx] = new_params
    ellipse_success_arr[roi_idx] = new_success
    if eye_separation_arr is not None and channel_count >= 2:
        eye_separation_arr[roi_idx] = separation

    if reason_arr is not None or reason_bytes_arr is not None:
        existing = _read_reason_value(
            reason_arr=reason_arr,
            reason_bytes_arr=reason_bytes_arr,
            roi_idx=roi_idx,
        )
        tags = ["manual_correction", "patch_viewer_edit"]
        if reject_reason:
            tags.append(reject_reason)
        _write_reason_value(
            reason_group=reason_group,
            reason_arr=reason_arr,
            reason_bytes_arr=reason_bytes_arr,
            roi_idx=roi_idx,
            reason_value=_merge_reason(existing, tags),
        )

    return {
        "channel_count": channel_count,
        "successful_eyes": int(np.sum(new_success)),
        "eye_separation": separation,
        "reject_reason": reject_reason,
    }


def _mouse_modifier_state(flags: int) -> Tuple[bool, bool, bool]:
    ctrl_down = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
    shift_down = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
    left_down = bool(flags & cv2.EVENT_FLAG_LBUTTON)
    return ctrl_down, shift_down, left_down


def _resolve_erase_mode(base_erase_mode: bool, shift_down: bool) -> bool:
    # Shift acts as a temporary inverse while drawing.
    return (not base_erase_mode) if shift_down else base_erase_mode


def _step_clamped(value: int, delta: int, minimum: int, maximum: int) -> int:
    return max(int(minimum), min(int(maximum), int(value) + int(delta)))


def create_viewer(
    zarr_path: Path,
    *,
    registry_path: Optional[Path],
    refined_run: Optional[str],
    crop_run: Optional[str],
    keypoint_run: Optional[str],
    keypoint_group: Optional[str],
    start_roi: int,
    padding: int,
    scale_percent: int,
    edit_zoom: int,
    frame_flag_file: Optional[str],
    keypoint_nudge_flag_file: Optional[str],
    review_state: str,
    review_method: str,
    review_intended_use: str,
    reviewer: Optional[str],
    review_notes: Optional[str],
) -> None:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)

    if refined_run is None:
        refined_run = _get_latest_run(root, "refined_eye_masks_runs")
    refined_parent = root.get("refined_eye_masks_runs")
    if not isinstance(refined_parent, zarr.Group) or refined_run not in refined_parent:
        raise RuntimeError(f"Refined run '{refined_run}' not found.")
    refined = refined_parent[refined_run]
    compat_context = refined_eye_masks_compat_context(refined)
    derived_compat_read_only = bool(compat_context.get("is_derived_compat"))
    compat_redirect_hint = (
        refined_eye_masks_redirect_hint(
            refined_run=refined_run,
            source=refined,
            zarr_path=zarr_path,
        )
        if derived_compat_read_only
        else None
    )

    source_eye_run_name = refined.attrs.get("source_eye_masks_run")
    source_eye_parent = root.get("eye_masks_runs")
    source_eye_run: Optional[zarr.Group] = None
    probability_data: Optional[object] = None
    probability_dataset_names: List[str] = []
    if isinstance(source_eye_parent, zarr.Group) and isinstance(source_eye_run_name, str) and source_eye_run_name in source_eye_parent:
        source_eye_run = source_eye_parent[source_eye_run_name]
        try:
            probability_bundle = load_mask_bundle(
                source_eye_run,
                threshold=DEFAULT_RECOMMENDED_PROBABILITY_THRESHOLD,
                prefer_probs=True,
                materialize=False,
                lazy=True,
            )
            if probability_bundle.probs is not None:
                probability_data = probability_bundle.probs
                probability_dataset_names = [
                    str(name)
                    for name in probability_bundle.provenance.get("source", [])
                    if str(name) in {"mask_probs_roi", "mask_probs_roi_refined"}
                ]
                if int(probability_data.shape[0]) != int(refined["masks_roi"].shape[0]):
                    probability_data = None
                    probability_dataset_names = []
        except Exception:
            probability_data = None
            probability_dataset_names = []

    resolved_crop = crop_run or refined.attrs.get("source_crop_run")
    if not isinstance(resolved_crop, str):
        resolved_crop = _get_latest_run(root, "crop_runs")
    crop_parent = root.get("crop_runs")
    if not isinstance(crop_parent, zarr.Group) or resolved_crop not in crop_parent:
        raise RuntimeError(f"Crop run '{resolved_crop}' not found.")
    crop_group = crop_parent[resolved_crop]
    crop_source = CropImageSource.open(root, crop_run=resolved_crop, zarr_path=zarr_path)
    frame_indices_arr = crop_group.get("frame_indices")
    masks_roi = refined["masks_roi"]
    edit_applied = refined.get("edit_applied")
    ellipse_params = refined["ellipse_params"]
    ellipse_success = refined["ellipse_success"]
    eye_separation = refined.get("eye_separation")
    metrics_group = refined.get("metrics")
    reason_group: Optional[zarr.Group] = metrics_group if isinstance(metrics_group, zarr.Group) else None
    reason_arr = reason_group.get("reason") if isinstance(reason_group, zarr.Group) else None
    reason_bytes_arr = reason_group.get("reason_bytes") if isinstance(reason_group, zarr.Group) else None
    if not isinstance(reason_arr, zarr.Array) and not isinstance(reason_bytes_arr, zarr.Array):
        reason_group = refined
        reason_arr = refined.get("reason")
        reason_bytes_arr = refined.get("reason_bytes")

    total_rois = int(crop_source.total_rois)
    if total_rois <= 0:
        raise RuntimeError("No ROIs found in crop run.")
    if int(masks_roi.shape[0]) != total_rois:
        raise RuntimeError("masks_roi rows do not match crop ROI rows.")
    if tuple(ellipse_params.shape[:2]) != tuple(masks_roi.shape[:2]):
        raise RuntimeError("ellipse_params shape does not align with masks_roi.")
    if tuple(ellipse_success.shape[:2]) != tuple(masks_roi.shape[:2]):
        raise RuntimeError("ellipse_success shape does not align with masks_roi.")
    frame_indices: Optional[np.ndarray] = None
    if frame_indices_arr is not None:
        candidate = np.asarray(frame_indices_arr[:], dtype=np.int64)
        if candidate.shape[0] == total_rois:
            frame_indices = candidate

    flag_path = Path(frame_flag_file).expanduser() if frame_flag_file else None
    nudge_flag_path = Path(keypoint_nudge_flag_file).expanduser() if keypoint_nudge_flag_file else None
    review_label = _format_review_status(refined.attrs.get("eye_mask_review_status"))

    kp_group, kp_group_name, kp_run_name = _resolve_keypoint_group(
        root,
        refined,
        explicit_run=keypoint_run,
        explicit_group=keypoint_group,
    )
    keypoints_roi = None
    if kp_group is not None and "keypoints_roi" in kp_group:
        candidate = np.asarray(kp_group["keypoints_roi"][:], dtype=np.float32)
        if candidate.shape[0] == total_rois:
            keypoints_roi = candidate

    eye_labels_attr = refined.attrs.get("eye_labels")
    if isinstance(eye_labels_attr, (list, tuple)):
        eye_labels = [str(v) for v in eye_labels_attr]
    else:
        eye_labels = [f"eye_{i}" for i in range(int(masks_roi.shape[1]))]

    window_name = "Eye Mask Patch Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    recommended_threshold = _coerce_probability_threshold(
        source_eye_run.attrs.get(RECOMMENDED_PROBABILITY_THRESHOLD_ATTR) if source_eye_run is not None else None,
        default=_coerce_probability_threshold(
            source_eye_run.attrs.get("mask_probability_threshold") if source_eye_run is not None else None,
            default=DEFAULT_RECOMMENDED_PROBABILITY_THRESHOLD,
        ),
    )

    def _noop(_val: int) -> None:
        return

    cv2.createTrackbar("ROI", window_name, max(0, min(int(start_roi), total_rois - 1)), total_rois - 1, _noop)
    cv2.createTrackbar("Padding", window_name, max(1, min(int(padding), MAX_PADDING)), MAX_PADDING, _noop)
    cv2.createTrackbar("Brush", window_name, DEFAULT_BRUSH, BRUSH_MAX, _noop)
    cv2.createTrackbar("Edit Zoom", window_name, max(1, min(int(edit_zoom), MAX_EDIT_ZOOM)), MAX_EDIT_ZOOM, _noop)
    cv2.createTrackbar(
        "Scale %",
        window_name,
        max(25, min(int(scale_percent), MAX_SCALE_PERCENT)),
        MAX_SCALE_PERCENT,
        _noop,
    )
    if probability_data is not None:
        cv2.createTrackbar(
            PROBABILITY_TRACKBAR_NAME,
            window_name,
            int(round(recommended_threshold * PROBABILITY_TRACKBAR_MAX)),
            PROBABILITY_TRACKBAR_MAX,
            _noop,
        )

    state: Dict[str, object] = {
        "loaded_roi_idx": -1,
        "roi_img": None,
        "edit_masks_row": None,
        "active_eye": 0,
        "dirty": False,
        "drawing": False,
        "erase_mode": False,
        "show_ellipse_overlay": True,
        "show_keypoint_overlay": False,
        "panel_regions": [],
        "header_h": 56,
        "display_scale": 1.0,
        "cursor_eye": None,
        "cursor_patch_xy": None,
        "source_reason_compact": "n/a",
        "source_reject_reason": None,
        "probability_preview_enabled": bool(probability_data is not None),
        "probability_row": None,
        "recommended_probability_threshold": float(recommended_threshold),
    }

    def _refresh_source_reason_state(roi_idx: int) -> None:
        raw_value = _read_reason_value(
            reason_arr=reason_arr if isinstance(reason_arr, zarr.Array) else None,
            reason_bytes_arr=reason_bytes_arr if isinstance(reason_bytes_arr, zarr.Array) else None,
            roi_idx=roi_idx,
        )
        if raw_value:
            reason_tags = _reason_tags_from_value(raw_value)
            source_reject = _extract_primary_reject_reason(reason_tags)
            state["source_reason_compact"] = _format_reason_tags_compact(reason_tags)
            state["source_reject_reason"] = source_reject
            return
        state["source_reason_compact"] = "n/a"
        state["source_reject_reason"] = None

    def _adjust_brush_size(delta: int) -> int:
        current = max(BRUSH_MIN, int(cv2.getTrackbarPos("Brush", window_name)))
        updated = _step_clamped(current, delta, BRUSH_MIN, BRUSH_MAX)
        if updated != current:
            cv2.setTrackbarPos("Brush", window_name, updated)
        return updated

    def _adjust_padding_size(delta: int) -> int:
        current = max(1, int(cv2.getTrackbarPos("Padding", window_name)))
        updated = _step_clamped(current, delta, 1, MAX_PADDING)
        if updated != current:
            cv2.setTrackbarPos("Padding", window_name, updated)
        return updated

    def _load_roi(roi_idx: int) -> None:
        state["loaded_roi_idx"] = int(roi_idx)
        state["roi_img"] = np.asarray(crop_source.read_slice(int(roi_idx), int(roi_idx) + 1)[0], dtype=np.uint8)
        state["edit_masks_row"] = np.asarray(masks_roi[roi_idx], dtype=np.uint8).copy()
        if probability_data is not None:
            state["probability_row"] = _to_numpy_roi_slice(probability_data, int(roi_idx))
        else:
            state["probability_row"] = None
        state["dirty"] = False
        state["cursor_eye"] = None
        state["cursor_patch_xy"] = None
        active = int(state.get("active_eye", 0))
        channels = int(np.asarray(state["edit_masks_row"]).shape[0])
        if active >= channels:
            state["active_eye"] = max(0, channels - 1)
        _refresh_source_reason_state(int(roi_idx))

    def _map_point_to_eye_patch(raw_x: int, raw_y: int) -> Optional[Tuple[int, int, int]]:
        for region in state["panel_regions"]:
            panel_x = int(region["x"])
            panel_y = int(region["y"])
            panel_w = int(region["w"])
            panel_h = int(region["h"])
            if not (panel_x <= raw_x < panel_x + panel_w and panel_y <= raw_y < panel_y + panel_h):
                continue
            local_x = raw_x - panel_x
            local_y = raw_y - panel_y
            label_h = int(region["label_h"])
            if local_y < label_h:
                return int(region["eye_idx"]), -1, -1

            zoom = max(1, int(region["zoom"]))
            patch_x = int(local_x / zoom)
            patch_y = int((local_y - label_h) / zoom)
            patch_w = int(region["patch_w"])
            patch_h = int(region["patch_h"])
            if 0 <= patch_x < patch_w and 0 <= patch_y < patch_h:
                return int(region["eye_idx"]), patch_x, patch_y
            return int(region["eye_idx"]), -1, -1
        return None

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        scale = float(state.get("display_scale", 1.0))
        raw_x = int(x / scale) if scale > 0 else int(x)
        raw_y = int(y / scale) if scale > 0 else int(y)
        ctrl_down, shift_down, left_down = _mouse_modifier_state(int(flags))
        mapped = _map_point_to_eye_patch(raw_x, raw_y)

        if mapped is None:
            if event == cv2.EVENT_LBUTTONUP or not left_down:
                state["drawing"] = False
            state["cursor_eye"] = None
            state["cursor_patch_xy"] = None
            return

        eye_idx, patch_x, patch_y = mapped
        state["active_eye"] = int(eye_idx)
        if patch_x >= 0 and patch_y >= 0:
            state["cursor_eye"] = int(eye_idx)
            state["cursor_patch_xy"] = (int(patch_x), int(patch_y))
        else:
            state["cursor_eye"] = None
            state["cursor_patch_xy"] = None

        if derived_compat_read_only:
            state["drawing"] = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Brush edits are gated behind Ctrl to avoid accidental draws while navigating.
            if ctrl_down:
                state["drawing"] = True
            else:
                state["drawing"] = False
        elif event == cv2.EVENT_MOUSEMOVE and bool(state.get("drawing")):
            if not (ctrl_down and left_down):
                state["drawing"] = False
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False

        if not state.get("drawing") or patch_x < 0 or patch_y < 0:
            return

        masks_row = state.get("edit_masks_row")
        if masks_row is None:
            return

        target_region = None
        for region in state["panel_regions"]:
            if int(region["eye_idx"]) == int(eye_idx):
                target_region = region
                break
        if target_region is None:
            return

        roi_x = int(target_region["patch_x0"]) + int(patch_x)
        roi_y = int(target_region["patch_y0"]) + int(patch_y)
        if roi_x < 0 or roi_y < 0:
            return

        mask_eye = np.asarray(masks_row)[int(eye_idx)]
        if roi_y >= mask_eye.shape[0] or roi_x >= mask_eye.shape[1]:
            return
        brush = max(BRUSH_MIN, int(cv2.getTrackbarPos("Brush", window_name)))
        erase_mode = _resolve_erase_mode(bool(state.get("erase_mode")), shift_down)
        value = 0 if erase_mode else 1
        cv2.circle(mask_eye, (roi_x, roi_y), brush, value, -1)
        state["dirty"] = True

    cv2.setMouseCallback(window_name, on_mouse)

    print("\n=== Eye Mask Patch Viewer ===")
    print(f"Refined run: {compat_context['stage_label']}/{refined_run}")
    print(f"Crop run: crop_runs/{resolved_crop}")
    if derived_compat_read_only:
        canonical_run = compat_context.get("source_refined_subject_masks_run")
        if canonical_run:
            print(f"Canonical subject run: refined_subject_masks_runs/{canonical_run}")
        print("Mode: read-only legacy compat view; zarr writes are disabled.")
        print(compat_redirect_hint)
    if kp_group_name and kp_run_name:
        print(f"Keypoints: {kp_group_name}/{kp_run_name}")
    else:
        print("Keypoints: none (using ellipse/mask centers)")
    print("Controls:")
    print("  left/right or p/n: prev/next ROI")
    if derived_compat_read_only:
        print("  Mouse on Eye Edit panel: inspection only in read-only compat mode")
    else:
        print("  Mouse on Eye Edit panel: hold Ctrl+LMB to paint")
    print("  ]/[ or +/-: increase/decrease brush size")
    print("  . / ,: increase/decrease patch size")
    print("  x: toggle brush mode (paint/erase)")
    print("  e: toggle ellipse/axis overlay")
    print("  h: toggle keypoint crosshair overlay")
    if probability_data is not None:
        print("  t: toggle probability preview panels")
        print("  w: save recommended probability threshold metadata for refinement")
    if not derived_compat_read_only:
        print("  Mouse while drawing: hold Shift to temporarily invert brush mode")
    print("  1/2: choose active eye")
    if derived_compat_read_only:
        print("  c: disabled in read-only compat mode")
        print("  s: disabled in read-only compat mode")
    else:
        print("  c: clear all eye masks for current ROI (unsaved until 's')")
        print("  s: save edits to refined run")
    print("  r: reset edits for current ROI")
    print("  b: flag current frame/ROI for cleanup")
    print("  k: flag current frame/ROI for keypoint nudge (keep masks unchanged)")
    if derived_compat_read_only:
        print("  a: disabled in read-only compat mode")
    else:
        print("  a: approve refinement (write eye_mask_review_status)")
    print("  q/ESC: quit")
    if flag_path is not None:
        print(f"Frame flag file: {flag_path.expanduser().resolve(strict=False)}")
    if nudge_flag_path is not None:
        print(f"Keypoint nudge flag file: {nudge_flag_path.expanduser().resolve(strict=False)}")
    if registry_path is not None:
        print(f"Registry auto-sync: enabled ({registry_path.expanduser().resolve(strict=False)})")
    else:
        print("Registry auto-sync: disabled")
    print(f"Current review status: {review_label}")
    if probability_data is not None:
        dataset_label = ", ".join(probability_dataset_names) if probability_dataset_names else "probability masks"
        print(
            "Probability preview source: "
            f"eye_masks_runs/{source_eye_run_name}/{dataset_label} "
            f"(recommended_threshold={recommended_threshold:.2f})"
        )

    try:
        while True:
            roi_idx = cv2.getTrackbarPos("ROI", window_name)
            pad = max(1, cv2.getTrackbarPos("Padding", window_name))
            scale = max(25, cv2.getTrackbarPos("Scale %", window_name))
            zoom = max(1, cv2.getTrackbarPos("Edit Zoom", window_name))
            brush = max(BRUSH_MIN, cv2.getTrackbarPos("Brush", window_name))
            probability_threshold = (
                float(cv2.getTrackbarPos(PROBABILITY_TRACKBAR_NAME, window_name)) / float(PROBABILITY_TRACKBAR_MAX)
                if probability_data is not None
                else float(state.get("recommended_probability_threshold", DEFAULT_RECOMMENDED_PROBABILITY_THRESHOLD))
            )
            state["display_scale"] = float(scale) / 100.0
            show_ellipse_overlay = bool(state.get("show_ellipse_overlay", True))
            show_keypoint_overlay = bool(state.get("show_keypoint_overlay", False))
            probability_preview_enabled = bool(state.get("probability_preview_enabled", False)) and probability_data is not None

            loaded_roi_idx = int(state.get("loaded_roi_idx", -1))
            if loaded_roi_idx != roi_idx:
                if bool(state.get("dirty")) and loaded_roi_idx >= 0:
                    print(f"Discarded unsaved edits for ROI {loaded_roi_idx}.")
                _load_roi(roi_idx)

            roi_img = np.asarray(state["roi_img"], dtype=np.uint8)
            masks_row = np.asarray(state["edit_masks_row"], dtype=np.uint8)
            probability_row = state.get("probability_row")
            ellipse_row = np.asarray(ellipse_params[roi_idx], dtype=np.float32)
            ellipse_ok_row = np.asarray(ellipse_success[roi_idx], dtype=bool)
            kp_row = np.asarray(keypoints_roi[roi_idx], dtype=np.float32) if keypoints_roi is not None else None

            channel_count = int(masks_row.shape[0])
            active_eye = int(state.get("active_eye", 0))
            if active_eye >= channel_count:
                active_eye = max(0, channel_count - 1)
                state["active_eye"] = active_eye

            eye_rows: List[np.ndarray] = []
            eye_metas: List[Dict[str, int]] = []
            eye_center_sources: List[str] = []
            for eye_idx in range(channel_count):
                label = _friendly_eye_label(eye_labels[eye_idx] if eye_idx < len(eye_labels) else None, eye_idx)
                center_xy, center_source = _resolve_eye_center_with_source(
                    eye_idx,
                    kp_row,
                    ellipse_row[eye_idx] if eye_idx < ellipse_row.shape[0] else np.full(5, np.nan, dtype=np.float32),
                    bool(eye_idx < ellipse_ok_row.shape[0] and ellipse_ok_row[eye_idx]),
                    masks_row[eye_idx],
                    roi_img.shape,
                )
                center_source_label = CENTER_SOURCE_LABELS.get(center_source, center_source.upper())
                eye_label = f"{label} [{center_source_label}]"
                eye_center_sources.append(center_source)
                keypoint_xy = _extract_eye_keypoint_xy(kp_row, eye_idx)
                probability_eye = None
                probability_label = None
                if probability_preview_enabled and probability_row is not None:
                    probability_eye, probability_label = _select_probability_preview_channel(
                        np.asarray(probability_row),
                        eye_idx=eye_idx,
                    )
                cursor_eye = state.get("cursor_eye")
                cursor_xy = state.get("cursor_patch_xy")
                row, edit_meta = _build_eye_row_with_editor(
                    roi_img,
                    masks_row[eye_idx],
                    ellipse_row[eye_idx] if eye_idx < ellipse_row.shape[0] else np.full(5, np.nan, dtype=np.float32),
                    bool(eye_idx < ellipse_ok_row.shape[0] and ellipse_ok_row[eye_idx]),
                    center_xy,
                    padding=pad,
                    eye_label=eye_label,
                    eye_idx=eye_idx,
                    edit_zoom=zoom,
                    active_eye=bool(eye_idx == active_eye),
                    brush_radius=brush,
                    cursor_patch_xy=cursor_xy if cursor_eye == eye_idx else None,
                    show_ellipse_overlay=show_ellipse_overlay,
                    keypoint_xy=keypoint_xy,
                    show_keypoint_overlay=show_keypoint_overlay,
                    probability_eye=probability_eye,
                    probability_label=probability_label,
                    probability_threshold=probability_threshold,
                )
                eye_rows.append(row)
                eye_metas.append(edit_meta)

            content, panel_regions = _compose_rows_with_edit_meta(eye_rows, eye_metas, gap=8)
            header_h = int(state.get("header_h", 36))
            for region in panel_regions:
                region["y"] = int(region["y"] + header_h)
            state["panel_regions"] = panel_regions

            header = np.zeros((header_h, content.shape[1], 3), dtype=np.uint8)
            dirty_suffix = " *unsaved*" if bool(state.get("dirty")) else ""
            brush_mode = "erase" if bool(state.get("erase_mode")) else "paint"
            ellipse_overlay = "on" if show_ellipse_overlay else "off"
            keypoint_overlay = "on" if show_keypoint_overlay else "off"
            source_reject = str(state.get("source_reject_reason") or "none")
            source_reason_compact = str(state.get("source_reason_compact") or "none")
            fallback_labels = [CENTER_SOURCE_LABELS.get(src, src.upper()) for src in eye_center_sources if src != "keypoint"]
            if fallback_labels:
                uniq = sorted(set(fallback_labels))
                fallback_summary = f"{len(fallback_labels)}/{len(eye_center_sources)} ({','.join(uniq)})"
            else:
                fallback_summary = "0"
            cv2.putText(
                header,
                (
                    f"ROI {roi_idx + 1}/{total_rois}  pad={pad}  brush={brush}px  "
                    f"mode={brush_mode}  ellipse={ellipse_overlay}  keypoints={keypoint_overlay}  edit_zoom={zoom}x  "
                    f"active_eye={active_eye + 1}  fallback={fallback_summary}  "
                    f"review={review_label}{dirty_suffix}"
                    + ("  compat=read-only" if derived_compat_read_only else "")
                ),
                (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                header,
                (
                    f"Source reject={source_reject}  source reason tags={source_reason_compact}"
                    + (
                        f"  prob_preview={'on' if probability_preview_enabled else 'off'} "
                        f"thr={probability_threshold:.2f} "
                        f"saved={float(state.get('recommended_probability_threshold', recommended_threshold)):.2f}"
                        if probability_data is not None
                        else ""
                    )
                ),
                (8, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )
            frame = np.vstack([header, content])

            if scale != 100:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=float(scale) / 100.0,
                    fy=float(scale) / 100.0,
                    interpolation=cv2.INTER_NEAREST,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("n"), 83):  # 'n' or right-arrow
                cv2.setTrackbarPos("ROI", window_name, min(total_rois - 1, roi_idx + 1))
            elif key in (ord("p"), 81):  # 'p' or left-arrow
                cv2.setTrackbarPos("ROI", window_name, max(0, roi_idx - 1))
            elif key in (ord("1"), ord("2")):
                eye_idx = int(chr(key)) - 1
                if 0 <= eye_idx < channel_count:
                    state["active_eye"] = eye_idx
            elif key in (ord("x"),):
                new_mode = not bool(state.get("erase_mode"))
                state["erase_mode"] = new_mode
                print(f"Brush mode: {'erase' if new_mode else 'paint'}")
            elif key in (ord("e"),):
                new_overlay = not bool(state.get("show_ellipse_overlay", True))
                state["show_ellipse_overlay"] = new_overlay
                print(f"Ellipse overlay: {'on' if new_overlay else 'off'}")
            elif key in (ord("h"),):
                new_overlay = not bool(state.get("show_keypoint_overlay", False))
                state["show_keypoint_overlay"] = new_overlay
                print(f"Keypoint overlay: {'on' if new_overlay else 'off'}")
            elif key in (ord("t"),):
                if probability_data is None:
                    print("No probability masks available for preview.")
                    continue
                new_preview = not bool(state.get("probability_preview_enabled", False))
                state["probability_preview_enabled"] = new_preview
                print(f"Probability preview: {'on' if new_preview else 'off'}")
            elif key in (ord("]"), ord("="), ord("+")):
                brush_value = _adjust_brush_size(BRUSH_STEP)
                print(f"Brush size: {brush_value}px")
            elif key in (ord("["), ord("-"), ord("_")):
                brush_value = _adjust_brush_size(-BRUSH_STEP)
                print(f"Brush size: {brush_value}px")
            elif key in (ord("."), ord(">")):
                pad_value = _adjust_padding_size(PADDING_STEP)
                print(f"Patch size (half-width): {pad_value}px")
            elif key in (ord(","), ord("<")):
                pad_value = _adjust_padding_size(-PADDING_STEP)
                print(f"Patch size (half-width): {pad_value}px")
            elif key in (ord("c"),):
                if derived_compat_read_only:
                    print(compat_redirect_hint)
                    continue
                edit_masks_row = state.get("edit_masks_row")
                if edit_masks_row is None:
                    print("No ROI masks loaded to clear.")
                    continue
                edit_masks_arr = np.asarray(edit_masks_row, dtype=np.uint8)
                edit_masks_arr[...] = 0
                state["edit_masks_row"] = edit_masks_arr
                state["dirty"] = True
                print(f"Cleared all eye masks for ROI {roi_idx}. Press 's' to save.")
            elif key in (ord("r"),):
                _load_roi(roi_idx)
                print(f"Reset edits for ROI {roi_idx}.")
            elif key in (ord("s"),):
                if derived_compat_read_only:
                    print(compat_redirect_hint)
                    continue
                result = _save_roi_mask_edits(
                    root=root,
                    refined=refined,
                    roi_idx=roi_idx,
                    masks_row=np.asarray(state["edit_masks_row"], dtype=np.uint8),
                    masks_arr=masks_roi,
                    edit_applied_arr=edit_applied if isinstance(edit_applied, zarr.Array) else None,
                    ellipse_params_arr=ellipse_params,
                    ellipse_success_arr=ellipse_success,
                    eye_separation_arr=eye_separation if isinstance(eye_separation, zarr.Array) else None,
                    reason_arr=reason_arr if isinstance(reason_arr, zarr.Array) else None,
                    reason_bytes_arr=reason_bytes_arr if isinstance(reason_bytes_arr, zarr.Array) else None,
                    reason_group=reason_group if isinstance(reason_group, zarr.Group) else None,
                )
                if isinstance(reason_group, zarr.Group):
                    next_reason_arr = reason_group.get("reason")
                    next_reason_bytes_arr = reason_group.get("reason_bytes")
                    reason_arr = next_reason_arr if isinstance(next_reason_arr, zarr.Array) else None
                    reason_bytes_arr = next_reason_bytes_arr if isinstance(next_reason_bytes_arr, zarr.Array) else None
                _refresh_source_reason_state(int(roi_idx))
                _refresh_refined_eye_mask_metrics(root, refined)
                state["dirty"] = False
                print(
                    f"Saved ROI {roi_idx}: {result['successful_eyes']}/{result['channel_count']} eyes fit "
                    f"(reject_reason={result['reject_reason'] or 'none'})."
                )
                if registry_path is not None:
                    ok, detail = _sync_registry_for_zarr(
                        registry_path=registry_path,
                        zarr_path=zarr_path,
                    )
                    prefix = "Registry sync OK" if ok else "Registry sync FAILED"
                    print(f"{prefix}: {detail}")
            elif key in (ord("b"),):
                if flag_path is None:
                    print("No frame flag file configured. Pass --frame-flag-file to enable cleanup flagging.")
                    continue
                if frame_indices is None:
                    print(f"crop_runs/{resolved_crop} missing frame_indices; cannot flag cleanup frames.")
                    continue
                frame_idx = int(frame_indices[roi_idx])
                try:
                    _append_flagged_frame(flag_path, str(zarr_path), frame_idx, roi_idx)
                    print(f"Flagged cleanup frame {frame_idx} (roi {roi_idx})")
                    print(f"Frame flag file: {flag_path.expanduser().resolve(strict=False)}")
                except Exception as exc:
                    print(f"Failed to flag cleanup frame: {exc}")
            elif key in (ord("k"),):
                if nudge_flag_path is None:
                    print(
                        "No keypoint nudge flag file configured. "
                        "Pass --keypoint-nudge-flag-file to enable keypoint nudge flagging."
                    )
                    continue
                if frame_indices is None:
                    print(f"crop_runs/{resolved_crop} missing frame_indices; cannot flag keypoint nudges.")
                    continue
                frame_idx = int(frame_indices[roi_idx])
                try:
                    _append_flagged_frame(
                        nudge_flag_path,
                        str(zarr_path),
                        frame_idx,
                        roi_idx,
                        extra_fields={
                            "action": "keypoint_nudge",
                            "preserve_eye_masks": True,
                        },
                    )
                    print(f"Flagged keypoint nudge frame {frame_idx} (roi {roi_idx})")
                    print(f"Keypoint nudge flag file: {nudge_flag_path.expanduser().resolve(strict=False)}")
                except Exception as exc:
                    print(f"Failed to flag keypoint nudge frame: {exc}")
            elif key in (ord("a"),):
                if derived_compat_read_only:
                    print(compat_redirect_hint)
                    continue
                try:
                    try:
                        _refresh_refined_eye_mask_metrics(root, refined)
                    except Exception as exc:
                        print(f"Warning: failed to refresh refined metrics before review status write: {exc}")
                    payload = _apply_eye_mask_review_status(
                        refined_parent,
                        refined_run,
                        refined,
                        state=review_state,
                        method=review_method,
                        intended_use=review_intended_use,
                        reviewer=reviewer or os.environ.get("USER") or os.environ.get("USERNAME"),
                        notes=review_notes,
                    )
                    review_label = _format_review_status(payload)
                    print(
                        f"Set eye_mask_review_status: {payload.get('state')} "
                        f"({payload.get('method')}/{payload.get('intended_use')})"
                    )
                    if registry_path is not None:
                        ok, detail = _sync_registry_for_zarr(
                            registry_path=registry_path,
                            zarr_path=zarr_path,
                        )
                        prefix = "Registry sync OK" if ok else "Registry sync FAILED"
                        print(f"{prefix}: {detail}")
                except Exception as exc:
                    print(f"Failed to set eye_mask_review_status: {exc}")
            elif key in (ord("w"),):
                if derived_compat_read_only:
                    print(compat_redirect_hint)
                    continue
                if probability_data is None or source_eye_run is None or not isinstance(source_eye_run_name, str):
                    print("No source probability run available for threshold metadata.")
                    continue
                try:
                    reviewer_name = reviewer or os.environ.get("USER") or os.environ.get("USERNAME")
                    payload = _apply_recommended_probability_threshold(
                        source_eye_run,
                        threshold=probability_threshold,
                        reviewer=reviewer_name,
                        notes=review_notes,
                        source_refined_run=refined_run,
                    )
                    state["recommended_probability_threshold"] = float(probability_threshold)
                    print(
                        "Saved recommended probability threshold "
                        f"{float(probability_threshold):.2f} to eye_masks_runs/{source_eye_run_name}"
                    )
                    print(f"Threshold metadata: {payload}")
                except Exception as exc:
                    print(f"Failed to save recommended probability threshold: {exc}")
    finally:
        crop_source.close()
        cv2.destroyWindow(window_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "Optional registry sqlite path. When provided, pressing 's' or 'a' "
            "auto-refreshes eye-mask performance and recording-step status for this recording."
        ),
    )
    parser.add_argument("--refined-run", help="Refined eye-mask run name (default: latest).")
    parser.add_argument("--crop-run", help="Crop run name override.")
    parser.add_argument("--keypoint-run", help="Keypoint run override for eye centers.")
    parser.add_argument("--keypoint-group", help="Keypoint group override (e.g. refined_keypoints_runs).")
    parser.add_argument("--start-roi", type=int, default=0, help="Starting ROI index.")
    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_PADDING,
        help=f"Eye patch half-width in pixels (default: {DEFAULT_PADDING}).",
    )
    parser.add_argument(
        "--scale-percent",
        type=int,
        default=DEFAULT_SCALE_PERCENT,
        help=f"Display scale percent (default: {DEFAULT_SCALE_PERCENT}).",
    )
    parser.add_argument(
        "--edit-zoom",
        type=int,
        default=DEFAULT_EDIT_ZOOM,
        help=f"Per-eye edit panel zoom factor (default: {DEFAULT_EDIT_ZOOM}).",
    )
    parser.add_argument(
        "--frame-flag-file",
        default="eye_mask_frame_flags.json",
        help="JSON file to append cleanup flags when pressing 'b' (default: eye_mask_frame_flags.json).",
    )
    parser.add_argument(
        "--keypoint-nudge-flag-file",
        default="keypoint_nudge_flags.json",
        help=(
            "JSON file to append keypoint nudge flags when pressing 'k' "
            "(default: keypoint_nudge_flags.json)."
        ),
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to write when pressing 'a' (default: approved).",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method to write when pressing 'a' (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Review intended_use to write when pressing 'a' (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name override (default: $USER).")
    parser.add_argument("--review-notes", help="Optional review notes for status written by 'a'.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    create_viewer(
        args.zarr_path,
        registry_path=args.registry,
        refined_run=args.refined_run,
        crop_run=args.crop_run,
        keypoint_run=args.keypoint_run,
        keypoint_group=args.keypoint_group,
        start_roi=args.start_roi,
        padding=args.padding,
        scale_percent=args.scale_percent,
        edit_zoom=args.edit_zoom,
        frame_flag_file=args.frame_flag_file,
        keypoint_nudge_flag_file=args.keypoint_nudge_flag_file,
        review_state=args.review_state,
        review_method=args.review_method,
        review_intended_use=args.review_intended_use,
        reviewer=args.reviewer,
        review_notes=args.review_notes,
    )


if __name__ == "__main__":
    main()
