"""Backend primitives for browser-based refined detection review."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name, set_authoritative_run
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
    review_axis: str = "frame"


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


def _source_detection_payload(refined_run: zarr.Group) -> dict[str, object]:
    source_group = refined_run.get("source_detections")
    if source_group is None:
        return {
            "source_surface_source_detect_row_index": np.empty((0,), dtype=np.int32),
            "source_surface_frame_indices": np.empty((0,), dtype=np.int32),
            "source_surface_bbox_norm_coords": np.empty((0, 4), dtype=np.float64),
            "source_surface_decision_labels": np.empty((0,), dtype=object),
            "source_surface_reason_labels": np.empty((0,), dtype=object),
            "source_surface_confidence_scores": np.empty((0,), dtype=np.float32),
            "source_surface_class_ids": np.empty((0,), dtype=np.int32),
            "source_surface_review_notes": np.empty((0,), dtype=object),
            "source_rows_by_frame": {},
            "source_row_lookup": {},
        }

    source_rows = np.asarray(source_group["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
    source_frames = np.asarray(source_group["frame_indices"][:], dtype=np.int32).reshape(-1)
    decision_codes = np.asarray(source_group["decision_codes"][:], dtype=np.int8).reshape(-1)
    decision_labels = np.asarray(
        [
            {0: "accepted", 1: "filtered", 2: "duplicate", 3: "manual_clear"}.get(
                int(code), "filtered"
            )
            for code in decision_codes.tolist()
        ],
        dtype=object,
    )
    reason_labels = detect_review_mod.read_reason_labels(source_group)
    if reason_labels is None:
        reason_labels = decision_labels.copy()
    rows_by_frame: dict[int, list[int]] = {}
    row_lookup: dict[int, int] = {}
    for idx, (frame, source_row) in enumerate(zip(source_frames.tolist(), source_rows.tolist())):
        rows_by_frame.setdefault(int(frame), []).append(int(idx))
        row_lookup[int(source_row)] = int(idx)
    return {
        "source_surface_source_detect_row_index": source_rows,
        "source_surface_frame_indices": source_frames,
        "source_surface_bbox_norm_coords": np.asarray(
            source_group["bbox_norm_coords"][:], dtype=np.float64
        ).reshape(-1, 4),
        "source_surface_decision_labels": decision_labels,
        "source_surface_reason_labels": np.asarray(reason_labels, dtype=object).reshape(-1),
        "source_surface_confidence_scores": (
            np.asarray(source_group["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in source_group
            else np.full(source_rows.shape[0], np.nan, dtype=np.float32)
        ),
        "source_surface_class_ids": (
            np.asarray(source_group["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in source_group
            else np.full(source_rows.shape[0], -1, dtype=np.int32)
        ),
        "source_surface_review_notes": (
            np.asarray(source_group["review_notes"][:], dtype=object).reshape(-1)
            if "review_notes" in source_group
            else np.full(source_rows.shape[0], "", dtype=object)
        ),
        "source_rows_by_frame": {
            frame: np.asarray(indices, dtype=np.int32) for frame, indices in rows_by_frame.items()
        },
        "source_row_lookup": row_lookup,
    }


def _load_multi_instance_payload(refined_run: zarr.Group) -> dict[str, object]:
    instances = refined_run["instances"]
    if "instance_key" not in instances:
        raise RuntimeError(
            "Multi-detection browser review requires instances/instance_key; ordinal row identity "
            "is not accepted."
        )
    frame_indices = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
    instance_keys = np.asarray(instances["instance_key"][:], dtype=np.uint64).reshape(-1)
    if instance_keys.shape[0] != frame_indices.shape[0]:
        raise RuntimeError("instances/instance_key length does not match frame_indices.")
    if np.unique(instance_keys).shape[0] != instance_keys.shape[0]:
        raise RuntimeError("instances/instance_key must be unique for multi-detection review.")

    source_kind_codes = np.asarray(instances["source_kind_codes"][:], dtype=np.int8).reshape(-1)
    source_kind_labels = np.asarray(
        [
            detect_review_mod._SOURCE_KIND_LABEL_BY_CODE.get(int(code), "unknown")  # type: ignore[attr-defined]
            for code in source_kind_codes.tolist()
        ],
        dtype=object,
    )
    reason_labels = detect_review_mod.read_reason_labels(instances)
    if reason_labels is None:
        reason_labels = np.full(frame_indices.shape[0], "present", dtype=object)
    payload: dict[str, object] = {
        "review_axis": np.asarray(["frame_instances"], dtype=object),
        "frame_indices": frame_indices,
        "bbox_norm_coords": np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4),
        "confidence_scores": (
            np.asarray(instances["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in instances
            else np.full(frame_indices.shape[0], np.nan, dtype=np.float32)
        ),
        "class_ids": (
            np.asarray(instances["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in instances
            else np.full(frame_indices.shape[0], -1, dtype=np.int32)
        ),
        "status_labels": np.full(frame_indices.shape[0], "present", dtype=object),
        "source_kind_labels": source_kind_labels,
        "manual_edit_flags": np.asarray(instances["manual_edit_flags"][:], dtype=bool).reshape(-1),
        "reason_labels": np.asarray(reason_labels, dtype=object).reshape(-1),
        "source_detect_row_index": np.asarray(
            instances["source_detect_row_index"][:], dtype=np.int32
        ).reshape(-1),
        "detection_source": np.where(source_kind_labels == "interpolated", 1, 0).astype(np.int8),
        "refined_row_ids": np.asarray(instances["refined_row_ids"][:], dtype=np.int64).reshape(-1),
        "instance_keys": instance_keys,
        "instance_key_origin_codes": (
            np.asarray(instances["instance_key_origin_codes"][:], dtype=np.int8).reshape(-1)
            if "instance_key_origin_codes" in instances
            else np.full(frame_indices.shape[0], -1, dtype=np.int8)
        ),
    }
    payload.update(_source_detection_payload(refined_run))
    return payload


def _load_review_payload(refined_run: zarr.Group, *, total_frames: int) -> tuple[dict[str, object], str]:
    if detect_review_mod.has_sparse_curated_refined_detect_instances_arrays(refined_run):
        instances = refined_run["instances"]
        if "instance_key" in instances:
            return _load_multi_instance_payload(refined_run), "frame_instances"
    payload = detect_review_mod._load_dense_curated_edit_payload(  # type: ignore[attr-defined]
        refined_run,
        total_frames=total_frames,
    )
    return dict(payload), detect_review_mod._payload_review_axis(payload)  # type: ignore[attr-defined]


def _select_review_frames(
    payload: Mapping[str, object],
    *,
    total_frames: int,
    include_all: bool,
    target_frames: Optional[np.ndarray],
    max_items: Optional[int],
) -> np.ndarray:
    if target_frames is not None:
        frames = np.asarray(
            sorted({int(frame) for frame in target_frames.tolist() if 0 <= int(frame) < total_frames}),
            dtype=np.int32,
        )
    elif include_all:
        frames = np.arange(total_frames, dtype=np.int32)
    else:
        instance_frames = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
        present = np.zeros(total_frames, dtype=bool)
        present[instance_frames[(instance_frames >= 0) & (instance_frames < total_frames)]] = True
        review = ~present
        source_frames = np.asarray(
            payload.get("source_surface_frame_indices", np.empty((0,), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1)
        source_decisions = np.asarray(
            payload.get("source_surface_decision_labels", np.empty((0,), dtype=object)),
            dtype=object,
        ).reshape(-1)
        for frame, decision in zip(source_frames.tolist(), source_decisions.tolist()):
            if 0 <= int(frame) < total_frames and str(decision) != "accepted":
                review[int(frame)] = True
        frames = np.flatnonzero(review).astype(np.int32, copy=False)
    if max_items is not None:
        frames = frames[:max_items]
    return frames


def _reload_payload(session: DetectReviewSession) -> None:
    payload, review_axis = _load_review_payload(session.refined_run, total_frames=session.total_frames)
    session.payload = payload
    session.review_axis = review_axis


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

    payload, review_axis = _load_review_payload(refined_group, total_frames=total_frames)
    if review_axis not in {"frame", "frame_instances"}:
        raise RuntimeError(
            "Detection web review supports frame and frame_instances refined runs; "
            f"got review_axis={review_axis!r}."
        )
    if review_axis == "frame_instances":
        review_rows = _select_review_frames(
            payload,
            total_frames=total_frames,
            include_all=include_all,
            target_frames=_coerce_ints(target_frames),
            max_items=max_items,
        )
    else:
        frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.shape[0] != total_frames:
            raise RuntimeError(
                "Frame-axis detection review expects one canonical row per frame. "
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
        review_axis=review_axis,
    )


def review_session_summary(session: DetectReviewSession) -> dict[str, object]:
    if session.review_axis == "frame_instances":
        frame_indices = np.asarray(session.payload["frame_indices"], dtype=np.int32).reshape(-1)
        manual_flags = np.asarray(session.payload["manual_edit_flags"], dtype=bool).reshape(-1)
        counts = np.bincount(frame_indices, minlength=session.total_frames).astype(np.int64, copy=False)
        return {
            "zarr_path": session.zarr_path,
            "refined_run": session.refined_run_name,
            "review_axis": session.review_axis,
            "total_frames": session.total_frames,
            "reviewable_frames": int(session.review_rows.shape[0]),
            "present_frames": int(np.count_nonzero(counts)),
            "missing_or_filtered_frames": int(session.total_frames - np.count_nonzero(counts)),
            "total_instances": int(frame_indices.shape[0]),
            "multi_instance_frames": int(np.count_nonzero(counts > 1)),
            "max_instances_per_frame": int(np.max(counts)) if counts.size else 0,
            "manual_edits": int(np.sum(manual_flags)),
            "width": session.width,
            "height": session.height,
        }
    status_labels = np.asarray(session.payload["status_labels"], dtype=object).reshape(-1)
    manual_flags = np.asarray(session.payload["manual_edit_flags"], dtype=bool).reshape(-1)
    return {
        "zarr_path": session.zarr_path,
        "refined_run": session.refined_run_name,
        "review_axis": session.review_axis,
        "total_frames": session.total_frames,
        "reviewable_frames": int(session.review_rows.shape[0]),
        "present_frames": int(np.sum(status_labels == "present")),
        "missing_or_filtered_frames": int(np.sum(status_labels != "present")),
        "manual_edits": int(np.sum(manual_flags)),
        "width": session.width,
        "height": session.height,
    }


def _row_status(payload: Mapping[str, object], row_idx: int) -> dict[str, object]:
    return {
        "status_label": _json_scalar(np.asarray(payload["status_labels"], dtype=object).reshape(-1)[row_idx]),
        "source_kind_label": _json_scalar(
            np.asarray(payload["source_kind_labels"], dtype=object).reshape(-1)[row_idx]
        ),
        "reason_label": _json_scalar(np.asarray(payload["reason_labels"], dtype=object).reshape(-1)[row_idx]),
        "manual_edit": bool(np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)[row_idx]),
        "source_detect_row_index": int(
            np.asarray(payload["source_detect_row_index"], dtype=np.int32).reshape(-1)[row_idx]
        ),
        "confidence_score": _json_float(
            np.asarray(payload["confidence_scores"], dtype=np.float32).reshape(-1)[row_idx]
        ),
        "class_id": int(np.asarray(payload["class_ids"], dtype=np.int32).reshape(-1)[row_idx]),
    }


def _multi_frame_payload(session: DetectReviewSession, *, position: int) -> dict[str, object]:
    frame_idx = int(session.review_rows[position])
    frame_indices = np.asarray(session.payload["frame_indices"], dtype=np.int32).reshape(-1)
    row_indices = np.flatnonzero(frame_indices == frame_idx).astype(np.int32, copy=False)
    bboxes = np.asarray(session.payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    keys = np.asarray(session.payload["instance_keys"], dtype=np.uint64).reshape(-1)
    row_ids = np.asarray(session.payload["refined_row_ids"], dtype=np.int64).reshape(-1)
    detections = [
        {
            "instance_key": str(int(keys[row_idx])),
            "refined_row_id": int(row_ids[row_idx]),
            "row_idx": int(row_idx),
            "bbox_norm": _finite_bbox_or_none(bboxes[row_idx]),
            "status": _row_status(session.payload, int(row_idx)),
        }
        for row_idx in row_indices.tolist()
    ]
    first = detections[0] if detections else None
    empty_status = {
        "status_label": "missing",
        "source_kind_label": "none",
        "reason_label": "missing_detection",
        "manual_edit": False,
        "source_detect_row_index": -1,
        "confidence_score": None,
        "class_id": -1,
    }
    return {
        "position": int(position),
        "total": int(session.review_rows.size),
        "review_axis": session.review_axis,
        "row_idx": first.get("row_idx") if first else None,
        "frame_idx": frame_idx,
        "bbox_norm": first.get("bbox_norm") if first else None,
        "detections": detections,
        "detection_count": len(detections),
        "bbox_norm_coordinate_space": "source_image_xywhn",
        "bbox_display_transform": _bbox_display_transform(session),
        "status": first.get("status") if first else empty_status,
        "frame_image": _image_payload(np.asarray(session.images[frame_idx])),
    }


def load_frame_payload(session: DetectReviewSession, position: int) -> Mapping[str, object]:
    if session.review_rows.size == 0:
        raise IndexError("No frames are currently loaded for review.")
    if position < 0 or position >= int(session.review_rows.size):
        raise IndexError("Review position is out of range.")

    if session.review_axis == "frame_instances":
        return _multi_frame_payload(session, position=position)

    row_idx = int(session.review_rows[position])
    frame_idx = int(np.asarray(session.payload["frame_indices"], dtype=np.int32).reshape(-1)[row_idx])
    bbox = _finite_bbox_or_none(np.asarray(session.payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)[row_idx])
    refined_row_id = (
        int(np.asarray(session.payload["refined_row_ids"], dtype=np.int64).reshape(-1)[row_idx])
        if "refined_row_ids" in session.payload
        else -1
    )
    image = np.asarray(session.images[frame_idx])

    return {
        "position": int(position),
        "total": int(session.review_rows.size),
        "row_idx": row_idx,
        "frame_idx": frame_idx,
        "bbox_norm": bbox,
        "bbox_norm_coordinate_space": "source_image_xywhn",
        "bbox_display_transform": _bbox_display_transform(session),
        "detections": [
            {
                "instance_key": None,
                "refined_row_id": refined_row_id,
                "row_idx": row_idx,
                "bbox_norm": bbox,
                "status": _row_status(session.payload, row_idx),
            }
        ]
        if bbox is not None
        else [],
        "detection_count": 1 if bbox is not None else 0,
        "review_axis": session.review_axis,
        "status": _row_status(session.payload, row_idx),
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


def _parse_instance_key(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if not text.isdecimal():
        raise ValueError("instance_key must be an unsigned decimal string or null for a new detection.")
    parsed = int(text)
    if parsed < 0 or parsed > int(np.iinfo(np.uint64).max):
        raise ValueError("instance_key is outside the uint64 range.")
    return parsed


def _normalized_detection_request(
    value: object,
    *,
    default_class_id: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Each detection must be a JSON object.")
    bbox = _normalize_bbox_or_none(value.get("bbox_norm"))
    if bbox is None:
        raise ValueError("Each detection must contain a non-null bbox_norm.")
    class_value = value.get("class_id")
    class_id = default_class_id if class_value is None else int(class_value)  # type: ignore[arg-type]
    if class_id < 0:
        raise ValueError("class_id must be non-negative for a present detection.")
    return {
        "instance_key": _parse_instance_key(value.get("instance_key")),
        "bbox_norm": bbox,
        "class_id": class_id,
    }


def _set_source_decision(
    payload: dict[str, object],
    *,
    source_detect_row_index: int,
    decision: str,
    reason: str,
) -> None:
    if source_detect_row_index < 0:
        return
    lookup = payload.get("source_row_lookup")
    if not isinstance(lookup, Mapping):
        return
    source_row = lookup.get(int(source_detect_row_index))
    if source_row is None:
        return
    np.asarray(payload["source_surface_decision_labels"], dtype=object).reshape(-1)[int(source_row)] = decision
    np.asarray(payload["source_surface_reason_labels"], dtype=object).reshape(-1)[int(source_row)] = reason


def apply_detection_collection(
    session: DetectReviewSession,
    *,
    position: int,
    detections: Sequence[object],
) -> dict[str, object]:
    """Replace one frame's observation collection using stable instance keys.

    Existing keys may only identify rows already present in the server-selected
    frame. A null key means a new observation. Omitting an existing key deletes
    that observation. The curated writer preserves surviving keys and allocates
    new refined-row and instance identities server-side.
    """

    if session.review_axis != "frame_instances":
        if len(detections) > 1:
            raise ValueError(
                "Legacy frame-axis refined runs cannot store multiple detections; migrate the run "
                "to sparse instances before saving this frame."
            )
        bbox = None
        if detections:
            bbox = _normalized_detection_request(
                detections[0], default_class_id=session.manual_class_id
            )["bbox_norm"]
        return apply_manual_edit(session, position=position, bbox_norm=bbox)  # type: ignore[arg-type]
    if session.review_rows.size == 0:
        raise IndexError("No frames are currently loaded for review.")
    if position < 0 or position >= int(session.review_rows.size):
        raise IndexError("Review position is out of range.")

    frame_idx = int(session.review_rows[position])
    requested = [
        _normalized_detection_request(value, default_class_id=session.manual_class_id)
        for value in detections
    ]
    requested_keys = [item["instance_key"] for item in requested if item["instance_key"] is not None]
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError("detections contains duplicate instance_key values.")

    updated = _copy_payload(session.payload)
    frame_indices = np.asarray(updated["frame_indices"], dtype=np.int32).reshape(-1)
    instance_keys = np.asarray(updated["instance_keys"], dtype=np.uint64).reshape(-1)
    current_rows = np.flatnonzero(frame_indices == frame_idx).astype(np.int32, copy=False)
    current_by_key = {int(instance_keys[row]): int(row) for row in current_rows.tolist()}
    foreign_keys = sorted(
        key for key in requested_keys if key is not None and int(key) not in current_by_key
    )
    if foreign_keys:
        raise ValueError(
            "instance_key is not present in the current server-selected frame: "
            f"{foreign_keys[:5]}."
        )

    old_bbox = np.asarray(updated["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    old_scores = np.asarray(updated["confidence_scores"], dtype=np.float32).reshape(-1)
    old_classes = np.asarray(updated["class_ids"], dtype=np.int32).reshape(-1)
    old_source_kind = np.asarray(updated["source_kind_labels"], dtype=object).reshape(-1)
    old_manual = np.asarray(updated["manual_edit_flags"], dtype=bool).reshape(-1)
    old_reason = np.asarray(updated["reason_labels"], dtype=object).reshape(-1)
    old_source_row = np.asarray(updated["source_detect_row_index"], dtype=np.int32).reshape(-1)
    old_detection_source = np.asarray(updated["detection_source"], dtype=np.int8).reshape(-1)
    old_row_ids = np.asarray(updated["refined_row_ids"], dtype=np.int64).reshape(-1)
    old_origins = np.asarray(updated["instance_key_origin_codes"], dtype=np.int8).reshape(-1)

    kept_rows = np.flatnonzero(frame_indices != frame_idx).astype(np.int32, copy=False)
    columns: dict[str, list[object]] = {
        "frame_indices": [int(frame_indices[row]) for row in kept_rows.tolist()],
        "bbox_norm_coords": [old_bbox[row].copy() for row in kept_rows.tolist()],
        "confidence_scores": [old_scores[row] for row in kept_rows.tolist()],
        "class_ids": [old_classes[row] for row in kept_rows.tolist()],
        "status_labels": ["present" for _row in kept_rows.tolist()],
        "source_kind_labels": [old_source_kind[row] for row in kept_rows.tolist()],
        "manual_edit_flags": [old_manual[row] for row in kept_rows.tolist()],
        "reason_labels": [old_reason[row] for row in kept_rows.tolist()],
        "source_detect_row_index": [old_source_row[row] for row in kept_rows.tolist()],
        "detection_source": [old_detection_source[row] for row in kept_rows.tolist()],
        "refined_row_ids": [old_row_ids[row] for row in kept_rows.tolist()],
        "instance_keys": [instance_keys[row] for row in kept_rows.tolist()],
        "instance_key_origin_codes": [old_origins[row] for row in kept_rows.tolist()],
    }

    updated_count = 0
    added_count = 0
    retained_keys: set[int] = set()
    for item in requested:
        requested_key = item["instance_key"]
        bbox = np.asarray(item["bbox_norm"], dtype=np.float64).reshape(4)
        class_id = int(item["class_id"])
        if requested_key is None:
            added_count += 1
            source_row = -1
            score = np.float32(session.manual_score)
            source_kind = "manual"
            manual = True
            reason = "manual_correction"
            detection_source = np.int8(0)
            row_id = np.int64(-1)
            instance_key = np.uint64(0)
            origin = np.int8(-1)
        else:
            retained_keys.add(int(requested_key))
            row = current_by_key[int(requested_key)]
            changed = not np.array_equal(old_bbox[row].astype(np.float32), bbox.astype(np.float32))
            changed = changed or int(old_classes[row]) != class_id
            source_row = int(old_source_row[row])
            score = np.float32(session.manual_score) if changed else old_scores[row]
            source_kind = "manual" if changed else old_source_kind[row]
            manual = True if changed else bool(old_manual[row])
            reason = "manual_correction" if changed else old_reason[row]
            detection_source = np.int8(0) if changed else old_detection_source[row]
            row_id = old_row_ids[row]
            instance_key = instance_keys[row]
            origin = old_origins[row]
            if changed:
                updated_count += 1
                _set_source_decision(
                    updated,
                    source_detect_row_index=source_row,
                    decision="accepted",
                    reason="manual_correction",
                )

        columns["frame_indices"].append(frame_idx)
        columns["bbox_norm_coords"].append(bbox)
        columns["confidence_scores"].append(score)
        columns["class_ids"].append(np.int32(class_id))
        columns["status_labels"].append("present")
        columns["source_kind_labels"].append(source_kind)
        columns["manual_edit_flags"].append(manual)
        columns["reason_labels"].append(reason)
        columns["source_detect_row_index"].append(np.int32(source_row))
        columns["detection_source"].append(detection_source)
        columns["refined_row_ids"].append(row_id)
        columns["instance_keys"].append(instance_key)
        columns["instance_key_origin_codes"].append(origin)

    deleted_rows = [
        row for key, row in current_by_key.items() if int(key) not in retained_keys
    ]
    for row in deleted_rows:
        _set_source_decision(
            updated,
            source_detect_row_index=int(old_source_row[row]),
            decision="manual_clear",
            reason="manual_clear",
        )

    new_frames = np.asarray(columns["frame_indices"], dtype=np.int32)
    new_row_ids = np.asarray(columns["refined_row_ids"], dtype=np.int64)
    row_id_sort = np.where(new_row_ids >= 0, new_row_ids, np.iinfo(np.int64).max)
    order = np.lexsort((row_id_sort, new_frames))
    updated.update(
        {
            "review_axis": np.asarray(["frame_instances"], dtype=object),
            "frame_indices": new_frames[order],
            "bbox_norm_coords": np.asarray(columns["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)[order],
            "confidence_scores": np.asarray(columns["confidence_scores"], dtype=np.float32)[order],
            "class_ids": np.asarray(columns["class_ids"], dtype=np.int32)[order],
            "status_labels": np.asarray(columns["status_labels"], dtype=object)[order],
            "source_kind_labels": np.asarray(columns["source_kind_labels"], dtype=object)[order],
            "manual_edit_flags": np.asarray(columns["manual_edit_flags"], dtype=bool)[order],
            "reason_labels": np.asarray(columns["reason_labels"], dtype=object)[order],
            "source_detect_row_index": np.asarray(columns["source_detect_row_index"], dtype=np.int32)[order],
            "detection_source": np.asarray(columns["detection_source"], dtype=np.int8)[order],
            "refined_row_ids": new_row_ids[order],
            "instance_keys": np.asarray(columns["instance_keys"], dtype=np.uint64)[order],
            "instance_key_origin_codes": np.asarray(
                columns["instance_key_origin_codes"], dtype=np.int8
            )[order],
        }
    )
    detect_review_mod._write_dense_curated_edit_payload(  # type: ignore[attr-defined]
        session.root,
        zarr_path=session.zarr_path,
        refined_run_name=session.refined_run_name,
        payload=updated,  # type: ignore[arg-type]
        row_indices=np.asarray(current_rows, dtype=np.int32),
        command_label="detect_review_web_multi_instance",
        source_context={
            "editor": "detect_review_web",
            "edit_mode": "frame_instance_collection",
            "manual_review_frames": 1,
            "manual_review_added": added_count,
            "manual_review_updated": updated_count,
            "manual_review_removed": len(deleted_rows),
        },
    )
    _reload_payload(session)
    frame_payload = _multi_frame_payload(session, position=position)
    return {
        "action": "replace_detection_collection",
        "frame_idx": frame_idx,
        "added": added_count,
        "updated": updated_count,
        "removed": len(deleted_rows),
        "detection_count": frame_payload["detection_count"],
        "detections": frame_payload["detections"],
        "status": frame_payload["status"],
    }


def apply_manual_edit(
    session: DetectReviewSession,
    *,
    position: int,
    bbox_norm: Optional[Sequence[object]],
) -> dict[str, object]:
    if session.review_axis == "frame_instances":
        raise ValueError(
            "Multi-detection frames must be saved with the detections collection contract; "
            "single bbox_norm replacement is disabled."
        )
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


def _approve_authoritative_refined_detect(
    session: DetectReviewSession,
    *,
    state: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> dict[str, object]:
    if str(state).strip().lower() != "approved":
        return {"attempted": False, "reason": "review_state_not_approved"}
    zarr_path = Path(session.zarr_path).expanduser()
    if not zarr_path.exists():
        return {"attempted": False, "reason": "zarr_path_unavailable", "zarr_path": str(session.zarr_path)}

    from ..cli.palette import ApproveRequest, approve

    envelope = approve(
        ApproveRequest(
            recording=zarr_path,
            stage="refined_detect",
            run=session.refined_run_name,
            approved_by=reviewer,
            note=notes or "detect review sign-off",
            apply=True,
        )
    )
    return {
        "attempted": True,
        "status": envelope.get("status"),
        "reason_code": envelope.get("reason_code"),
        "run": envelope.get("run"),
        "envelope": envelope,
    }


def _authoritative_approval_ok(payload: Mapping[str, object]) -> bool:
    return bool(payload.get("attempted")) and str(payload.get("status") or "").strip().lower() == "ok"


def _mirror_authoritative_approval(parent: zarr.Group, run_name: str, payload: Mapping[str, object]) -> None:
    envelope = payload.get("envelope")
    approval = envelope.get("approval") if isinstance(envelope, Mapping) else None
    if not isinstance(approval, Mapping):
        approval = {}
    set_authoritative_run(
        parent,
        run_name,
        approved_by=str(approval.get("approved_by") or "unknown"),
        approved_at=str(approval.get("approved_at") or ""),
        git_sha=str(approval.get("git_sha") or ""),
        note=str(approval.get("note") or ""),
    )


def apply_review_status(
    session: DetectReviewSession,
    *,
    state: str = "approved",
    method: str = "manual",
    intended_use: str = "training",
    reviewer: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, object]:
    authoritative_approval = _approve_authoritative_refined_detect(
        session,
        state=state,
        reviewer=reviewer,
        notes=notes,
    )
    if str(state).strip().lower() == "approved" and not _authoritative_approval_ok(authoritative_approval):
        return {
            "action": "apply_review_status",
            "changed": False,
            "review_status": dict(session.refined_run.attrs.get("detect_review_status") or {}),
            "authoritative_approval": authoritative_approval,
        }

    try:
        refined_parent = session.root["refined_detect_runs"]
    except Exception:
        refined_parent = None
    if refined_parent is not None and _authoritative_approval_ok(authoritative_approval):
        _mirror_authoritative_approval(refined_parent, session.refined_run_name, authoritative_approval)
    reviewer_name = reviewer or os.environ.get("USER") or os.environ.get("USERNAME")
    timestamp = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "state": str(state),
        "method": str(method),
        "intended_use": str(intended_use),
        "timestamp": timestamp,
        "timestamp_utc": timestamp,
        "resolved_group": "refined",
        "preference_chain": ["refined"],
        "authoritative_approval": authoritative_approval,
    }
    if reviewer_name:
        payload["reviewer"] = reviewer_name
    if notes:
        payload["notes"] = notes
    session.refined_run.attrs["detect_review_status"] = payload
    return {
        "action": "apply_review_status",
        "changed": True,
        "review_status": payload,
        "authoritative_approval": authoritative_approval,
    }
