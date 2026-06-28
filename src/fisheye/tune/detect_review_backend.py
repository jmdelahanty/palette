"""Backend primitives for browser-based refined detection review."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name
from fisheye.tune import detect_review as detect_review_mod


@dataclass(frozen=False)
class DetectReviewSession:
    zarr_path: str
    root: zarr.Group
    refined_run: zarr.Group
    refined_run_name: str
    images: zarr.Array
    payload: dict[str, object]
    review_rows: np.ndarray
    total_frames: int
    height: int
    width: int
    source_height: int
    source_width: int
    downsample_preserve_aspect: bool
    manual_score: float
    manual_class_id: int


def _coerce_ints(values: Optional[Sequence[object]]) -> Optional[np.ndarray]:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    if not out:
        return None
    return np.asarray(sorted(set(out)), dtype=np.int32)


def _json_scalar(value: object) -> object:
    try:
        scalar = np.asarray(value).item()
    except Exception:
        return str(value)
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, (bytes, bytearray, memoryview)):
        try:
            return bytes(scalar).decode("utf-8")
        except Exception:
            return str(scalar)
    if isinstance(scalar, float) and not np.isfinite(scalar):
        return None
    if isinstance(scalar, (str, int, float, bool)) or scalar is None:
        return scalar
    return str(scalar)


def _json_float(value: object) -> float | None:
    scalar = _json_scalar(value)
    try:
        if scalar is None:
            return None
        out = float(scalar)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _image_payload(image: np.ndarray) -> dict[str, object]:
    arr = np.asarray(image, dtype=np.uint8)
    return {
        "shape": [int(v) for v in arr.shape],
        "channels": int(arr.shape[-1]) if arr.ndim >= 3 else 1,
        "dtype": str(arr.dtype),
        "encoding": "base64_raw",
        "pixels": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _finite_bbox_or_none(value: object) -> list[float] | None:
    bbox = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bbox)):
        return None
    return [float(v) for v in bbox.tolist()]


def _resolution_hw(value: object) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) < 2:
        return None
    try:
        height = int(value[0])  # type: ignore[index]
        width = int(value[1])  # type: ignore[index]
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def _bbox_display_transform(session: DetectReviewSession) -> dict[str, object]:
    image_height = int(session.height)
    image_width = int(session.width)
    source_height = int(session.source_height)
    source_width = int(session.source_width)
    if session.downsample_preserve_aspect and source_height > 0 and source_width > 0:
        scale = min(image_height / source_height, image_width / source_width)
        content_width = source_width * scale
        content_height = source_height * scale
        content_x = (image_width - content_width) / 2.0
        content_y = (image_height - content_height) / 2.0
        projection = "letterbox"
    else:
        scale = None
        content_x = 0.0
        content_y = 0.0
        content_width = float(image_width)
        content_height = float(image_height)
        projection = "stretch_or_same_aspect"
    return {
        "schema": "palette.detect_bbox_display_transform.v1",
        "bbox_norm_coordinate_space": "source_image_xywhn",
        "image_surface": "raw_video/images_ds",
        "projection": projection,
        "preserve_aspect": bool(session.downsample_preserve_aspect),
        "source_height": source_height,
        "source_width": source_width,
        "image_height": image_height,
        "image_width": image_width,
        "content_x": float(content_x),
        "content_y": float(content_y),
        "content_width": float(content_width),
        "content_height": float(content_height),
        "scale": None if scale is None else float(scale),
    }


def _copy_payload(payload: Mapping[str, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, value in payload.items():
        copied[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return copied


def _reload_payload(session: DetectReviewSession) -> None:
    session.payload = detect_review_mod._load_dense_curated_edit_payload(  # type: ignore[attr-defined]
        session.refined_run,
        total_frames=session.total_frames,
    )


def resolve_review_context(
    zarr_path: str,
    *,
    refined_run: Optional[str] = None,
    include_all: bool = False,
    target_frames: Optional[Sequence[object]] = None,
    max_items: Optional[int] = None,
    manual_score: float = 1.0,
    manual_class_id: int = 0,
) -> DetectReviewSession:
    root = open_zarr_group_direct(zarr_path, mode="a")
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")
    refined_run_name = refined_run or resolve_latest_complete_run_name(
        refined_parent,
    )
    if not refined_run_name or refined_run_name not in refined_parent:
        raise RuntimeError("Refined detect run not found.")
    refined_group = refined_parent[str(refined_run_name)]

    variant = detect_review_mod._pick_variant(refined_group, "refined")  # type: ignore[attr-defined]
    if variant != "refined":
        raise RuntimeError("Detection web review currently supports only canonical refined detect runs.")

    raw_video = root.get("raw_video")
    if raw_video is None or "images_ds" not in raw_video:
        raise RuntimeError("Zarr archive is missing raw_video/images_ds.")
    images = raw_video["images_ds"]
    if len(images.shape) < 3:
        raise RuntimeError(f"raw_video/images_ds must have shape (N,H,W[,C]), got {tuple(images.shape)}.")
    total_frames = int(images.shape[0])
    height = int(images.shape[1])
    width = int(images.shape[2])
    raw_attrs = dict(raw_video.attrs)
    source_height, source_width = _resolution_hw(raw_attrs.get("original_resolution")) or (height, width)
    downsample_preserve_aspect = bool(raw_attrs.get("downsample_preserve_aspect", False))

    payload = detect_review_mod._load_dense_curated_edit_payload(  # type: ignore[attr-defined]
        refined_group,
        total_frames=total_frames,
    )
    review_axis = detect_review_mod._payload_review_axis(payload)  # type: ignore[attr-defined]
    if review_axis != "frame":
        raise RuntimeError(
            "Detection web review MVP supports only frame-axis refined runs; "
            f"got review_axis={review_axis!r}."
        )
    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    if frame_indices.shape[0] != total_frames:
        raise RuntimeError(
            "Detection web review expects one canonical row per frame. "
            f"Found {frame_indices.shape[0]} rows for {total_frames} video frames."
        )
    review_rows = detect_review_mod._select_refined_review_rows(  # type: ignore[attr-defined]
        payload,
        review_all=include_all,
        target_frames=_coerce_ints(target_frames),
        max_items=max_items,
    )
    return DetectReviewSession(
        zarr_path=str(zarr_path),
        root=root,
        refined_run=refined_group,
        refined_run_name=str(refined_run_name),
        images=images,
        payload=dict(payload),
        review_rows=np.asarray(review_rows, dtype=np.int32),
        total_frames=total_frames,
        height=height,
        width=width,
        source_height=source_height,
        source_width=source_width,
        downsample_preserve_aspect=downsample_preserve_aspect,
        manual_score=float(manual_score),
        manual_class_id=int(manual_class_id),
    )


def review_session_summary(session: DetectReviewSession) -> dict[str, object]:
    status_labels = np.asarray(session.payload["status_labels"], dtype=object).reshape(-1)
    manual_flags = np.asarray(session.payload["manual_edit_flags"], dtype=bool).reshape(-1)
    return {
        "zarr_path": session.zarr_path,
        "refined_run": session.refined_run_name,
        "total_frames": session.total_frames,
        "reviewable_frames": int(session.review_rows.shape[0]),
        "present_frames": int(np.sum(status_labels == "present")),
        "missing_or_filtered_frames": int(np.sum(status_labels != "present")),
        "manual_edits": int(np.sum(manual_flags)),
        "width": session.width,
        "height": session.height,
    }


def load_frame_payload(session: DetectReviewSession, position: int) -> Mapping[str, object]:
    if session.review_rows.size == 0:
        raise IndexError("No frames are currently loaded for review.")
    if position < 0 or position >= int(session.review_rows.size):
        raise IndexError("Review position is out of range.")

    row_idx = int(session.review_rows[position])
    frame_idx = int(np.asarray(session.payload["frame_indices"], dtype=np.int32).reshape(-1)[row_idx])
    bbox = _finite_bbox_or_none(np.asarray(session.payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)[row_idx])
    image = np.asarray(session.images[frame_idx])

    return {
        "position": int(position),
        "total": int(session.review_rows.size),
        "row_idx": row_idx,
        "frame_idx": frame_idx,
        "bbox_norm": bbox,
        "bbox_norm_coordinate_space": "source_image_xywhn",
        "bbox_display_transform": _bbox_display_transform(session),
        "status": {
            "status_label": _json_scalar(np.asarray(session.payload["status_labels"], dtype=object).reshape(-1)[row_idx]),
            "source_kind_label": _json_scalar(np.asarray(session.payload["source_kind_labels"], dtype=object).reshape(-1)[row_idx]),
            "reason_label": _json_scalar(np.asarray(session.payload["reason_labels"], dtype=object).reshape(-1)[row_idx]),
            "manual_edit": bool(np.asarray(session.payload["manual_edit_flags"], dtype=bool).reshape(-1)[row_idx]),
            "source_detect_row_index": int(np.asarray(session.payload["source_detect_row_index"], dtype=np.int32).reshape(-1)[row_idx]),
            "confidence_score": _json_float(np.asarray(session.payload["confidence_scores"], dtype=np.float32).reshape(-1)[row_idx]),
            "class_id": int(np.asarray(session.payload["class_ids"], dtype=np.int32).reshape(-1)[row_idx]),
        },
        "frame_image": _image_payload(image),
    }


def _normalize_bbox_or_none(bbox_norm: Optional[Sequence[object]]) -> np.ndarray | None:
    if bbox_norm is None:
        return None
    bbox = np.asarray(bbox_norm, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bbox)):
        raise ValueError("bbox_norm must contain four finite values or be null.")
    bbox[:2] = np.clip(bbox[:2], 0.0, 1.0)
    bbox[2:] = np.clip(bbox[2:], 0.0, 1.0)
    if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
        raise ValueError("bbox_norm width and height must be positive.")
    return bbox


def apply_manual_edit(
    session: DetectReviewSession,
    *,
    position: int,
    bbox_norm: Optional[Sequence[object]],
) -> dict[str, object]:
    if session.review_rows.size == 0:
        raise IndexError("No frames are currently loaded for review.")
    if position < 0 or position >= int(session.review_rows.size):
        raise IndexError("Review position is out of range.")

    row_idx = int(session.review_rows[position])
    frame_idx = int(np.asarray(session.payload["frame_indices"], dtype=np.int32).reshape(-1)[row_idx])
    normalized_bbox = _normalize_bbox_or_none(bbox_norm)
    updated = _copy_payload(session.payload)

    current_source_row_index = int(np.asarray(updated["source_detect_row_index"], dtype=np.int32).reshape(-1)[row_idx])
    source_surface_row_idx = detect_review_mod._resolve_source_surface_row_for_frame(  # type: ignore[attr-defined]
        updated,  # type: ignore[arg-type]
        frame=frame_idx,
        preferred_source_detect_row_index=current_source_row_index,
    )
    source_detect_row_index = np.asarray(updated["source_detect_row_index"], dtype=np.int32).reshape(-1)
    detection_source = np.asarray(updated["detection_source"], dtype=np.int8).reshape(-1)
    bbox_arr = np.asarray(updated["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(updated["confidence_scores"], dtype=np.float32).reshape(-1)
    class_ids = np.asarray(updated["class_ids"], dtype=np.int32).reshape(-1)
    status_labels = np.asarray(updated["status_labels"], dtype=object).reshape(-1)
    source_kind_labels = np.asarray(updated["source_kind_labels"], dtype=object).reshape(-1)
    manual_edit_flags = np.asarray(updated["manual_edit_flags"], dtype=bool).reshape(-1)
    reason_labels = np.asarray(updated["reason_labels"], dtype=object).reshape(-1)

    chosen_source_row_index = (
        int(np.asarray(updated["source_surface_source_detect_row_index"], dtype=np.int32).reshape(-1)[source_surface_row_idx])
        if source_surface_row_idx is not None and "source_surface_source_detect_row_index" in updated
        else -1
    )
    source_detect_row_index[row_idx] = chosen_source_row_index
    detection_source[row_idx] = 0

    if normalized_bbox is None:
        bbox_arr[row_idx] = np.full((4,), np.nan, dtype=np.float64)
        scores[row_idx] = np.float32(np.nan)
        class_ids[row_idx] = np.int32(-1)
        status_labels[row_idx] = "filtered_out"
        source_kind_labels[row_idx] = "none"
        manual_edit_flags[row_idx] = True
        reason_labels[row_idx] = "manual_clear"
        action = "manual_clear"
        if source_surface_row_idx is not None:
            np.asarray(updated["source_surface_decision_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "manual_clear"
            np.asarray(updated["source_surface_reason_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "manual_clear"
        added = 0
        removed = 1
    else:
        bbox_arr[row_idx] = normalized_bbox
        scores[row_idx] = np.float32(session.manual_score)
        class_ids[row_idx] = np.int32(session.manual_class_id)
        status_labels[row_idx] = "present"
        source_kind_labels[row_idx] = "manual"
        manual_edit_flags[row_idx] = True
        reason_labels[row_idx] = "manual_correction"
        action = "manual_correction"
        if source_surface_row_idx is not None:
            np.asarray(updated["source_surface_decision_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "accepted"
            np.asarray(updated["source_surface_reason_labels"], dtype=object).reshape(-1)[source_surface_row_idx] = "manual_correction"
        added = 1
        removed = 0

    detect_review_mod._write_dense_curated_edit_payload(  # type: ignore[attr-defined]
        session.root,
        zarr_path=session.zarr_path,
        refined_run_name=session.refined_run_name,
        payload=updated,  # type: ignore[arg-type]
        row_indices=np.asarray([row_idx], dtype=np.int32),
        command_label="detect_review_web",
        source_context={
            "editor": "detect_review_web",
            "edit_mode": "manual",
            "manual_review_frames": 1,
            "manual_review_added": added,
            "manual_review_removed": removed,
        },
    )
    _reload_payload(session)
    return {
        "action": action,
        "frame_idx": frame_idx,
        "row_idx": row_idx,
        "bbox_norm": _finite_bbox_or_none(np.asarray(session.payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)[row_idx]),
        "status": load_frame_payload(session, position)["status"],
    }
