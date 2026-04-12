from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import zarr

from .type_conversions import normalize_attr

HEADING_TEMPORAL_OUTLIER_THRESHOLD_DEG = 120.0
HEADING_TEMPORAL_MAX_FRAME_GAP = 3
TEMPORAL_HEADING_STATUS_ENABLED = "enabled"
TEMPORAL_HEADING_STATUS_DISABLED_SAMPLED_IMPORT = "disabled_sampled_import"
TEMPORAL_HEADING_STATUS_UNAVAILABLE = "unavailable_missing_fields"


def _resolve_chunks(run_group: zarr.Group, size: int) -> tuple[int, ...]:
    heading_arr = run_group.get("heading")
    if heading_arr is not None and heading_arr.chunks:
        return tuple(int(chunk) for chunk in heading_arr.chunks)
    success_arr = run_group.get("refined_success") or run_group.get("detection_success")
    if success_arr is not None and success_arr.chunks:
        return tuple(int(chunk) for chunk in success_arr.chunks)
    return (max(1, min(1024, int(size))),)


def _write_or_create_array(
    run_group: zarr.Group,
    name: str,
    data: np.ndarray,
    *,
    chunks: tuple[int, ...],
) -> None:
    if name in run_group:
        run_group[name][:] = data
    else:
        run_group.create_array(name, data=data, chunks=chunks, overwrite=True)


def _drop_temporal_heading_arrays(run_group: zarr.Group) -> None:
    for name in ("heading_delta_prev_deg", "heading_delta_next_deg", "heading_temporal_outlier"):
        if name in run_group:
            del run_group[name]


def _read_sampled_import_meta(root: Optional[zarr.Group]) -> Dict[str, Any]:
    if root is None:
        return {
            "sampled_import": False,
            "import_mode": None,
            "import_purpose": None,
            "frame_step": None,
            "has_original_frame_indices": False,
        }
    raw = root.get("raw_video")
    if raw is None:
        return {
            "sampled_import": False,
            "import_mode": None,
            "import_purpose": None,
            "frame_step": None,
            "has_original_frame_indices": False,
        }

    import_mode = normalize_attr(raw.attrs.get("import_mode"))
    import_purpose = normalize_attr(raw.attrs.get("import_purpose"))
    frame_step_raw = raw.attrs.get("frame_step")
    frame_step: Optional[int]
    try:
        frame_step = int(frame_step_raw) if frame_step_raw is not None else None
    except Exception:
        frame_step = None
    has_mapping = "original_frame_indices" in raw

    sampled_import = False
    if import_mode == "sampled":
        sampled_import = True
    if import_purpose == "training_data":
        sampled_import = True
    if frame_step is not None and frame_step > 1:
        sampled_import = True
    if has_mapping:
        sampled_import = True

    return {
        "sampled_import": sampled_import,
        "import_mode": import_mode,
        "import_purpose": import_purpose,
        "frame_step": frame_step,
        "has_original_frame_indices": bool(has_mapping),
    }


def _temporal_heading_policy(root: Optional[zarr.Group]) -> Dict[str, Any]:
    sampled_meta = _read_sampled_import_meta(root)
    if bool(sampled_meta["sampled_import"]):
        return {
            **sampled_meta,
            "status": TEMPORAL_HEADING_STATUS_DISABLED_SAMPLED_IMPORT,
            "disabled_reason": "sampled_import",
        }
    return {
        **sampled_meta,
        "status": TEMPORAL_HEADING_STATUS_ENABLED,
        "disabled_reason": None,
    }


def _circular_heading_delta_deg(current_deg: float, neighbor_deg: float) -> float:
    delta = (float(current_deg) - float(neighbor_deg) + 180.0) % 360.0 - 180.0
    return float(abs(delta))


def compute_temporal_heading_arrays(
    *,
    frame_indices: np.ndarray,
    heading_values: np.ndarray,
    heading_usable: np.ndarray,
    detection_indices: Optional[np.ndarray] = None,
    threshold_deg: float = HEADING_TEMPORAL_OUTLIER_THRESHOLD_DEG,
    max_gap_frames: int = HEADING_TEMPORAL_MAX_FRAME_GAP,
) -> Dict[str, Any]:
    frame_vals = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    heading_vals = np.asarray(heading_values, dtype=np.float64).reshape(-1)
    usable_vals = np.asarray(heading_usable, dtype=bool).reshape(-1)
    total = int(heading_vals.shape[0])

    if frame_vals.shape[0] != total:
        raise ValueError("frame_indices length does not match heading values")
    if usable_vals.shape[0] != total:
        raise ValueError("heading_usable length does not match heading values")

    if detection_indices is None:
        detection_vals = np.zeros(total, dtype=np.int64)
    else:
        detection_vals = np.asarray(detection_indices, dtype=np.int64).reshape(-1)
        if detection_vals.shape[0] != total:
            raise ValueError("detection_indices length does not match heading values")

    delta_prev = np.full(total, np.nan, dtype=np.float32)
    delta_next = np.full(total, np.nan, dtype=np.float32)

    unique_detection_vals = np.unique(detection_vals)
    for det_idx in unique_detection_vals.tolist():
        group_rows = np.flatnonzero(detection_vals == int(det_idx))
        if group_rows.size < 2:
            continue
        ordered_rows = group_rows[np.argsort(frame_vals[group_rows], kind="stable")]
        ordered_frames = frame_vals[ordered_rows]
        ordered_headings = heading_vals[ordered_rows]
        ordered_usable = usable_vals[ordered_rows]

        for pos, row_idx in enumerate(ordered_rows.tolist()):
            if not ordered_usable[pos]:
                continue

            prev_pos = pos - 1
            while prev_pos >= 0:
                if not ordered_usable[prev_pos]:
                    prev_pos -= 1
                    continue
                frame_gap = int(ordered_frames[pos] - ordered_frames[prev_pos])
                if frame_gap <= 0:
                    prev_pos -= 1
                    continue
                if frame_gap > int(max_gap_frames):
                    break
                delta_prev[row_idx] = np.float32(
                    _circular_heading_delta_deg(ordered_headings[pos], ordered_headings[prev_pos])
                )
                break

            next_pos = pos + 1
            while next_pos < ordered_rows.size:
                if not ordered_usable[next_pos]:
                    next_pos += 1
                    continue
                frame_gap = int(ordered_frames[next_pos] - ordered_frames[pos])
                if frame_gap <= 0:
                    next_pos += 1
                    continue
                if frame_gap > int(max_gap_frames):
                    break
                delta_next[row_idx] = np.float32(
                    _circular_heading_delta_deg(ordered_headings[pos], ordered_headings[next_pos])
                )
                break

    evaluable_mask = np.isfinite(delta_prev) & np.isfinite(delta_next)
    outlier_mask = evaluable_mask & (delta_prev >= float(threshold_deg)) & (delta_next >= float(threshold_deg))
    evaluable_count = int(np.sum(evaluable_mask))
    outlier_count = int(np.sum(outlier_mask))
    outlier_rate = (float(outlier_count) / float(evaluable_count) * 100.0) if evaluable_count else 0.0

    return {
        "heading_delta_prev_deg": delta_prev,
        "heading_delta_next_deg": delta_next,
        "heading_temporal_outlier": outlier_mask.astype(bool, copy=False),
        "heading_temporal_evaluable": evaluable_count,
        "heading_temporal_outlier_count": outlier_count,
        "heading_temporal_outlier_rate_percent": round(outlier_rate, 2),
        "temporal_heading_threshold_deg": float(threshold_deg),
        "temporal_heading_max_frame_gap": int(max_gap_frames),
    }


def refresh_refined_keypoint_heading_fields(
    refined: zarr.Group,
    *,
    root: Optional[zarr.Group] = None,
    threshold_deg: float = HEADING_TEMPORAL_OUTLIER_THRESHOLD_DEG,
    max_gap_frames: int = HEADING_TEMPORAL_MAX_FRAME_GAP,
) -> Dict[str, Any]:
    policy = _temporal_heading_policy(root)
    heading_arr = refined.get("heading")
    frame_indices_arr = refined.get("frame_indices")
    if heading_arr is None or frame_indices_arr is None:
        _drop_temporal_heading_arrays(refined)
        return {
            "available": False,
            "heading_finite": 0,
            "heading_usable": 0,
            "heading_temporal_evaluable": 0,
            "heading_temporal_outlier_count": 0,
            "heading_temporal_outlier_rate_percent": 0.0,
            "temporal_heading_threshold_deg": float(threshold_deg),
            "temporal_heading_max_frame_gap": int(max_gap_frames),
            "temporal_heading_status": TEMPORAL_HEADING_STATUS_UNAVAILABLE,
            "temporal_heading_disabled_reason": "missing_heading_or_frame_indices",
            "temporal_heading_sampled_import": bool(policy["sampled_import"]),
        }

    heading_vals = np.asarray(heading_arr[:], dtype=np.float64).reshape(-1)
    frame_vals = np.asarray(frame_indices_arr[:], dtype=np.int64).reshape(-1)
    total = int(heading_vals.shape[0])
    if frame_vals.shape[0] != total:
        raise ValueError("frame_indices length does not match heading length")

    success_arr = refined.get("refined_success") or refined.get("detection_success")
    if success_arr is None:
        success_vals = np.zeros(total, dtype=bool)
    else:
        success_vals = np.asarray(success_arr[:], dtype=bool).reshape(-1)
        if success_vals.shape[0] != total:
            raise ValueError("success array length does not match heading length")

    detection_source_arr = refined.get("detection_source")
    if detection_source_arr is None:
        source_is_real = np.ones(total, dtype=bool)
    else:
        detection_source_vals = np.asarray(detection_source_arr[:], dtype=np.int8).reshape(-1)
        if detection_source_vals.shape[0] != total:
            raise ValueError("detection_source length does not match heading length")
        source_is_real = detection_source_vals == 0

    heading_finite = np.isfinite(heading_vals)
    heading_usable = success_vals & source_is_real & heading_finite
    chunks = _resolve_chunks(refined, total)

    _write_or_create_array(refined, "heading_finite", heading_finite.astype(bool, copy=False), chunks=chunks)
    _write_or_create_array(refined, "heading_usable", heading_usable.astype(bool, copy=False), chunks=chunks)

    if policy["status"] != TEMPORAL_HEADING_STATUS_ENABLED:
        _drop_temporal_heading_arrays(refined)
        return {
            "available": True,
            "heading_finite": int(np.sum(heading_finite)),
            "heading_usable": int(np.sum(heading_usable)),
            "heading_temporal_evaluable": 0,
            "heading_temporal_outlier_count": 0,
            "heading_temporal_outlier_rate_percent": 0.0,
            "temporal_heading_threshold_deg": float(threshold_deg),
            "temporal_heading_max_frame_gap": int(max_gap_frames),
            "temporal_heading_status": str(policy["status"]),
            "temporal_heading_disabled_reason": policy["disabled_reason"],
            "temporal_heading_sampled_import": bool(policy["sampled_import"]),
        }

    detection_indices_arr = refined.get("detection_indices")
    detection_indices = None
    if detection_indices_arr is not None:
        detection_indices = np.asarray(detection_indices_arr[:], dtype=np.int64).reshape(-1)

    temporal = compute_temporal_heading_arrays(
        frame_indices=frame_vals,
        heading_values=heading_vals,
        heading_usable=heading_usable,
        detection_indices=detection_indices,
        threshold_deg=threshold_deg,
        max_gap_frames=max_gap_frames,
    )
    _write_or_create_array(
        refined,
        "heading_delta_prev_deg",
        np.asarray(temporal["heading_delta_prev_deg"], dtype=np.float32),
        chunks=chunks,
    )
    _write_or_create_array(
        refined,
        "heading_delta_next_deg",
        np.asarray(temporal["heading_delta_next_deg"], dtype=np.float32),
        chunks=chunks,
    )
    _write_or_create_array(
        refined,
        "heading_temporal_outlier",
        np.asarray(temporal["heading_temporal_outlier"], dtype=bool),
        chunks=chunks,
    )

    return {
        "available": True,
        "heading_finite": int(np.sum(heading_finite)),
        "heading_usable": int(np.sum(heading_usable)),
        "temporal_heading_status": str(policy["status"]),
        "temporal_heading_disabled_reason": policy["disabled_reason"],
        "temporal_heading_sampled_import": bool(policy["sampled_import"]),
        **temporal,
    }
