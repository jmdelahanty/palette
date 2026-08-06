"""Promote reviewed detection labels from analysis Zarrs into training Zarrs.

This writes the per-recording training store, not a merged/exported
model-training artifact. Traditional recordings and clipped recordings share the
same crop/source-index training surface, while clipped promotion carries extra
clip/frame-map provenance so parent-frame identity is not confused with
clip-local media frame identity.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.refined_detect_curation import (
    REFINED_DETECT_STATUS_CODE_MAP,
    REFINED_REVIEW_STATE_CODE_MAP,
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
    _bbox_norm_to_img_xyxy_with_missing,
    _build_sparse_refined_detect_summary,
    _write_bbox_coordinate_attrs,
    _write_common_array,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_pending,
    mark_run_started,
    require_runs_parent,
    resolve_latest_complete_run_name,
    set_authoritative_run,
)
from fisheye.tune import detect_review as detect_review_mod


PROMOTION_SCHEMA_VERSION = "palette.analysis_detect_to_training_promotion.v1"
DEFAULT_PROMOTED_CROP_RUN = "promoted_detect_manual"
DEFAULT_PROMOTED_REFINED_RUN = "refined_detect_promoted_manual"


@dataclass(frozen=True)
class PromotionOptions:
    refined_run: str | None = None
    target_crop_run: str | None = None
    target_refined_run: str | None = None
    sync_refined_instances: bool = True
    label_origin: str = "manual_review"
    include_negative: bool = True
    allow_unreviewed_negative: bool = False
    target_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class ClippedPromotionFrame:
    parent_frame_index: int
    clip_local_frame_index: int
    refined_group_path: str
    refined_run: str
    collection_id: str
    clip_id: str
    clip_index: int
    camera_serial: str
    source_video_path: str | Path
    recording_frame_id: int | None = None
    entity_id: int = 0


_SOURCE_STRING_FIELDS = (
    "source_analysis_zarr_path",
    "source_refined_detect_run",
    "source_refined_group_path",
    "source_collection_id",
    "source_clip_id",
    "source_camera_serial",
    "source_video_path",
)
_SOURCE_NUMERIC_FIELDS: Mapping[str, tuple[np.dtype | type, int]] = {
    "source_frame_idx": (np.int64, -1),
    "source_entity_id": (np.int32, 0),
    "source_refined_row_ids": (np.int64, -1),
    "source_detect_row_index": (np.int32, -1),
    "source_clip_index": (np.int32, -1),
    "source_clip_local_frame_index": (np.int64, -1),
    "source_parent_frame_index": (np.int64, -1),
    "source_recording_frame_id": (np.int64, -1),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _finite_bbox_or_none(value: object) -> np.ndarray | None:
    bbox = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(bbox)):
        return None
    if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
        return None
    return np.clip(bbox, 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_refined_run(root: zarr.Group, refined_run: str | None) -> tuple[str, zarr.Group]:
    parent = root.get("refined_detect_runs")
    if parent is None:
        raise RuntimeError("Analysis Zarr has no refined_detect_runs group.")
    if parent.attrs.get("latest_collection"):
        raise RuntimeError(
            "Top-level promote_detection_frames cannot resolve clipped collections; use promote_clipped_detection_frames with explicit clip context."
        )
    run_name = str(
        refined_run
        or resolve_latest_complete_run_name(parent)
        or ""
    ).strip()
    if not run_name or run_name not in parent:
        raise RuntimeError(f"Refined detect run not found: {run_name!r}")
    return run_name, parent[run_name]


def _resolve_total_frames(root: zarr.Group, refined_group: zarr.Group, frames: Sequence[int]) -> int:
    raw = root.get("raw_video")
    if raw is not None and "images_ds" in raw:
        return int(raw["images_ds"].shape[0])
    if "instances" in refined_group and "frame_counts" in refined_group["instances"]:
        return int(refined_group["instances"]["frame_counts"].shape[0])
    max_requested = max([int(frame) for frame in frames], default=-1)
    return max_requested + 1


def _load_refined_payload(
    root: zarr.Group,
    *,
    refined_group: zarr.Group,
    frames: Sequence[int],
) -> dict[str, np.ndarray]:
    total_frames = _resolve_total_frames(root, refined_group, frames)
    payload = detect_review_mod._load_dense_curated_edit_payload(  # type: ignore[attr-defined]
        refined_group,
        total_frames=total_frames,
    )
    axis = detect_review_mod._payload_review_axis(payload)  # type: ignore[attr-defined]
    if axis != "frame":
        raise RuntimeError(f"Promotion supports only frame-axis refined detections; got {axis!r}.")
    frame_indices = np.asarray(payload.get("frame_indices"), dtype=np.int64).reshape(-1)
    if frame_indices.size != np.unique(frame_indices).size:
        raise RuntimeError(
            "Frame-axis detection promotion requires exactly one review row per "
            "frame. Multi-instance promotion must use stable instance-key identity; "
            "refusing to collapse duplicate frames by row order."
        )
    return dict(payload)


def _resolve_training_crop_run(root: zarr.Group, requested: str | None) -> str:
    if requested:
        return str(requested)
    crop_parent = root.get("crop_runs")
    latest = resolve_latest_complete_run_name(crop_parent) if crop_parent is not None else None
    return str(latest or DEFAULT_PROMOTED_CROP_RUN)


def _string_values(group: zarr.Group | None, name: str, *, count: int, default: str = "") -> np.ndarray:
    if group is not None and name in group:
        values = np.asarray(group[name][:], dtype=object).reshape(-1)
        if values.shape[0] == count:
            return np.asarray([str(value) for value in values.tolist()], dtype=object)
    return np.full(count, default, dtype=object)


def _numeric_values(
    group: zarr.Group | None,
    name: str,
    *,
    count: int,
    dtype: np.dtype | type,
    default: int | float,
) -> np.ndarray:
    if group is not None and name in group:
        values = np.asarray(group[name][:], dtype=dtype).reshape(-1)
        if values.shape[0] == count:
            return values.copy()
    return np.full(count, default, dtype=dtype)


def _ensure_training_root(path: Path, *, apply: bool) -> zarr.Group | None:
    if path.exists():
        return open_zarr_group_direct(path, mode="a" if apply else "r")
    if not apply:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "created_by": "fisheye.tune.detect_training_promotion_backend",
            "created_at_utc": _utc_now(),
        }
    )
    return root


def _array_chunks_for_images(count: int, height: int, width: int) -> tuple[int, int, int]:
    return (max(1, min(16, max(1, count))), int(height), int(width))


def _ensure_images_ds(
    root: zarr.Group,
    *,
    initial_shape: tuple[int, int],
) -> zarr.Array:
    raw = root.require_group("raw_video")
    if "images_ds" in raw:
        arr = raw["images_ds"]
        if len(arr.shape) != 3:
            raise RuntimeError(f"raw_video/images_ds must be 3D gray images, got shape={arr.shape}")
        return arr
    height, width = int(initial_shape[0]), int(initial_shape[1])
    arr = raw.create_array(
        "images_ds",
        shape=(0, height, width),
        chunks=_array_chunks_for_images(1, height, width),
        dtype=np.uint8,
        fill_value=0,
    )
    arr.attrs.update({"format": "gray", "resolution": [height, width]})
    raw.attrs.setdefault("has_downsampled", True)
    return arr


def _ensure_original_frame_indices(root: zarr.Group, *, count: int) -> np.ndarray:
    raw = root.require_group("raw_video")
    if "original_frame_indices" in raw:
        values = np.asarray(raw["original_frame_indices"][:], dtype=np.int64).reshape(-1)
        if values.shape[0] == count:
            return values.copy()
    return np.full(count, -1, dtype=np.int64)


def _resize_axis0(array: zarr.Array, new_count: int) -> None:
    shape = list(array.shape)
    shape[0] = int(new_count)
    array.resize(tuple(shape))


def _replace_array(group: zarr.Group, name: str, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
    if name in group:
        del group[name]
    kwargs: dict[str, object] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    group.create_array(name, **kwargs)


def _replace_string_array(group: zarr.Group, name: str, values: Sequence[object]) -> None:
    if name in group:
        del group[name]
    labels = np.asarray([str(value) for value in values], dtype=object)
    arr = group.create_array(
        name,
        shape=(int(labels.shape[0]),),
        chunks=(max(1, min(1024, int(labels.shape[0]) or 1)),),
        dtype=VariableLengthUTF8(),
        fill_value="",
        overwrite=True,
    )
    if labels.shape[0]:
        arr[:] = labels


def _read_crop_payload(root: zarr.Group, crop_run: str, *, image_count: int) -> dict[str, np.ndarray]:
    count = int(image_count)
    crop_parent = root.get("crop_runs")
    crop = crop_parent.get(crop_run) if crop_parent is not None else None
    if crop is not None and "bbox_norm_coords" in crop:
        bbox = np.asarray(crop["bbox_norm_coords"][:], dtype=np.float32).reshape(-1, 4)
        if bbox.shape[0] != count:
            raise RuntimeError(
                f"crop_runs/{crop_run}/bbox_norm_coords length {bbox.shape[0]} does not match images {count}."
            )
    else:
        bbox = np.full((count, 4), np.nan, dtype=np.float32)
    if crop is not None and "frame_indices" in crop:
        frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64).reshape(-1)
        if frame_indices.shape[0] != count:
            raise RuntimeError(
                f"crop_runs/{crop_run}/frame_indices length {frame_indices.shape[0]} does not match images {count}."
            )
    else:
        frame_indices = np.arange(count, dtype=np.int64)
    detection_source = _numeric_values(crop, "detection_source", count=count, dtype=np.int8, default=0)
    label_state = _string_values(crop, "label_state", count=count, default="")
    finite = np.all(np.isfinite(bbox), axis=1)
    label_state = np.asarray(
        [
            str(state) if str(state) else ("positive" if bool(is_finite) else "inactive")
            for state, is_finite in zip(label_state.tolist(), finite.tolist())
        ],
        dtype=object,
    )
    label_origin = _string_values(crop, "label_origin", count=count, default="existing")
    return {
        "bbox_norm_coords": bbox,
        "frame_indices": frame_indices,
        "detection_source": detection_source,
        "label_state": label_state,
        "label_origin": label_origin,
    }


def _read_source_index(root: zarr.Group, *, count: int) -> dict[str, np.ndarray]:
    source = root.get("source_index")
    if source is None:
        result: dict[str, np.ndarray] = {name: np.full(count, "", dtype=object) for name in _SOURCE_STRING_FIELDS}
        for name, (dtype, default) in _SOURCE_NUMERIC_FIELDS.items():
            result[name] = np.full(count, default, dtype=dtype)
        return result
    result = {name: _string_values(source, name, count=count) for name in _SOURCE_STRING_FIELDS}
    for name, (dtype, default) in _SOURCE_NUMERIC_FIELDS.items():
        result[name] = _numeric_values(source, name, count=count, dtype=dtype, default=default)
    return result


def _append_row(arr: np.ndarray, value: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    value_arr = np.asarray([value], dtype=dtype if dtype is not None else arr.dtype)
    return np.concatenate([arr, value_arr], axis=0)


def _append_bbox(arr: np.ndarray, value: np.ndarray) -> np.ndarray:
    return np.concatenate([arr, np.asarray(value, dtype=np.float32).reshape(1, 4)], axis=0)


def _source_identity(
    *,
    analysis_zarr: Path,
    refined_run: str,
    frame: int,
) -> tuple[str, str, int, int]:
    return (str(analysis_zarr), str(refined_run), int(frame), 0)


def _find_existing_rows(
    *,
    source_index: Mapping[str, np.ndarray],
    original_frame_indices: np.ndarray,
    identity: tuple[str, str, int, int],
) -> np.ndarray:
    analysis_path, refined_run, frame_idx, entity_id = identity
    source_paths = np.asarray(source_index["source_analysis_zarr_path"], dtype=object)
    source_runs = np.asarray(source_index["source_refined_detect_run"], dtype=object)
    source_frames = np.asarray(source_index["source_frame_idx"], dtype=np.int64)
    source_entities = np.asarray(source_index["source_entity_id"], dtype=np.int32)
    mask = (
        (source_paths == str(analysis_path))
        & (source_runs == str(refined_run))
        & (source_frames == int(frame_idx))
        & (source_entities == int(entity_id))
    )
    matches = np.flatnonzero(mask).astype(np.int64, copy=False)
    if matches.size:
        return matches

    # Legacy compatibility: existing per-recording training Zarrs often only
    # carry raw_video/original_frame_indices. Use this only when unambiguous.
    fallback = np.flatnonzero(np.asarray(original_frame_indices, dtype=np.int64) == int(frame_idx))
    return fallback.astype(np.int64, copy=False) if fallback.size == 1 else np.empty(0, dtype=np.int64)


def _row_matches(
    *,
    crop_payload: Mapping[str, np.ndarray],
    source_index: Mapping[str, np.ndarray],
    row: int,
    identity: tuple[str, str, int, int],
    label_state: str,
    bbox: np.ndarray,
    label_origin: str,
    source_refined_row_id: int,
    source_detect_row_index: int,
) -> bool:
    if str(crop_payload["label_state"][row]) != str(label_state):
        return False
    if str(crop_payload["label_origin"][row]) != str(label_origin):
        return False
    current_bbox = np.asarray(crop_payload["bbox_norm_coords"][row], dtype=np.float32)
    if label_state == "positive":
        if not np.allclose(current_bbox, bbox, atol=1e-6, equal_nan=True):
            return False
    elif np.any(np.isfinite(current_bbox)):
        return False
    analysis_path, refined_run, frame_idx, entity_id = identity
    return (
        str(source_index["source_analysis_zarr_path"][row]) == str(analysis_path)
        and str(source_index["source_refined_detect_run"][row]) == str(refined_run)
        and int(source_index["source_frame_idx"][row]) == int(frame_idx)
        and int(source_index["source_entity_id"][row]) == int(entity_id)
        and int(source_index["source_refined_row_ids"][row]) == int(source_refined_row_id)
        and int(source_index["source_detect_row_index"][row]) == int(source_detect_row_index)
    )


def _resolve_analysis_source_video_path(analysis_zarr: Path, root: zarr.Group) -> Path | None:
    attrs = dict(root.attrs)
    raw = attrs.get("source_video_path") or attrs.get("video_path")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = analysis_zarr.parent / path
        return path.resolve()
    raw = attrs.get("source_video")
    if raw:
        recording_dir = analysis_zarr.parent.parent if analysis_zarr.parent.name == "zarr" else analysis_zarr.parent
        for candidate in (recording_dir / "cams" / str(raw), recording_dir / str(raw)):
            if candidate.exists():
                return candidate.resolve()
        return (recording_dir / "cams" / str(raw)).resolve()
    return None


def _resize_gray(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    target_h, target_w = int(shape_hw[0]), int(shape_hw[1])
    if arr.shape == (target_h, target_w):
        return arr.astype(np.uint8, copy=False)
    import cv2

    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)


def _decode_video_frame_gray_native(video_path: Path, frame_idx: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video for promotion: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not decode frame {frame_idx} from {video_path}")
    finally:
        cap.release()
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.asarray(frame, dtype=np.uint8)


def _decode_video_frames_gray_pynvvc(video_path: Path, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    requested = sorted({int(frame) for frame in frame_indices})
    if not requested:
        return {}

    try:
        import PyNvVideoCodec as nvc  # type: ignore
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyNvVideoCodec import failed: {exc}") from exc

    demuxer = nvc.CreateDemuxer(filename=str(video_path))
    decoder = nvc.CreateDecoder(
        gpuid=0,
        codec=demuxer.GetNvCodecId(),
        usedevicememory=True,
    )
    source_height = int(demuxer.Height())
    wanted = set(requested)
    max_requested = int(requested[-1])
    decoded: dict[int, np.ndarray] = {}
    frame_idx = 0

    for packet in demuxer:
        for frame in decoder.Decode(packet):
            if frame_idx in wanted:
                tensor = torch.from_dlpack(frame)
                y_plane = tensor[:source_height, :].contiguous()
                decoded[frame_idx] = y_plane.cpu().numpy().astype(np.uint8, copy=True)
                del tensor, y_plane
                if len(decoded) == len(wanted):
                    return decoded
            frame_idx += 1
            if frame_idx > max_requested and len(decoded) == len(wanted):
                return decoded
        if frame_idx > max_requested and len(decoded) == len(wanted):
            break

    missing = sorted(wanted - set(decoded))
    if missing:
        raise RuntimeError(f"PyNvVideoCodec did not decode requested frames {missing[:10]} from {video_path}")
    return decoded


def _decode_video_frames_gray_native_batch(
    video_path: Path,
    frame_indices: Sequence[int],
) -> tuple[dict[int, np.ndarray], str]:
    """Decode requested gray frames, preferring one sequential PyNvVC pass."""
    requested = sorted({int(frame) for frame in frame_indices})
    if not requested:
        return {}, "none"
    try:
        return _decode_video_frames_gray_pynvvc(video_path, requested), "pynvvc_luma_sequential"
    except Exception:
        return {
            int(frame_idx): _decode_video_frame_gray_native(video_path, int(frame_idx))
            for frame_idx in requested
        }, "opencv_videocapture_gray_random_access"


def _decode_video_frame_gray(video_path: Path, frame_idx: int, shape_hw: tuple[int, int]) -> np.ndarray:
    return _resize_gray(_decode_video_frame_gray_native(video_path, frame_idx), shape_hw)


def _existing_images_full(root: zarr.Group) -> zarr.Array | None:
    raw = root.get("raw_video")
    if raw is None or "images_full" not in raw:
        return None
    arr = raw["images_full"]
    if len(arr.shape) != 3:
        raise RuntimeError(f"raw_video/images_full must be 3D gray images, got shape={arr.shape}")
    return arr


def _ensure_images_full(root: zarr.Group, *, initial_shape: tuple[int, int]) -> zarr.Array:
    raw = root.require_group("raw_video")
    if "images_full" in raw:
        arr = raw["images_full"]
        if len(arr.shape) != 3:
            raise RuntimeError(f"raw_video/images_full must be 3D gray images, got shape={arr.shape}")
        return arr
    height, width = int(initial_shape[0]), int(initial_shape[1])
    arr = raw.create_array(
        "images_full",
        shape=(0, height, width),
        chunks=_array_chunks_for_images(1, height, width),
        dtype=np.uint8,
        fill_value=0,
    )
    arr.attrs.update(
        {
            "format": "gray",
            "resolution": [height, width],
            "source": "clipped_promotion_source_video_luma",
        }
    )
    return arr


def _load_source_image(
    analysis_zarr: Path,
    analysis_root: zarr.Group,
    *,
    frame_idx: int,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    raw = analysis_root.get("raw_video")
    if raw is not None and "images_ds" in raw and int(raw["images_ds"].shape[0]) > int(frame_idx):
        image = _resize_gray(np.asarray(raw["images_ds"][int(frame_idx)]), shape_hw)
        return image, {
            "image_source": "analysis_raw_video_images_ds",
            "image_source_path": "raw_video/images_ds",
            "decode_method": "zarr_copy",
        }
    video_path = _resolve_analysis_source_video_path(analysis_zarr, analysis_root)
    if video_path is None:
        raise RuntimeError(
            "Cannot append promoted row: analysis Zarr has no raw_video/images_ds and no source video path."
        )
    return _decode_video_frame_gray(video_path, int(frame_idx), shape_hw), {
        "image_source": "analysis_source_video",
        "image_source_path": str(video_path),
        "decode_method": "opencv_videocapture_gray_resize",
    }


def _write_crop_payload(root: zarr.Group, crop_run: str, payload: Mapping[str, np.ndarray]) -> None:
    crop_parent = require_runs_parent(root, "crop_runs")
    crop = crop_parent.require_group(crop_run)
    mark_run_started(crop, run_name=str(crop_run), stage="crop")
    mark_run_pending(crop_parent, str(crop_run))
    count = int(np.asarray(payload["bbox_norm_coords"]).shape[0])
    chunk = max(1, min(8192, count or 1))
    _replace_array(crop, "bbox_norm_coords", np.asarray(payload["bbox_norm_coords"], dtype=np.float32), chunks=(chunk, 4))
    _replace_array(crop, "frame_indices", np.asarray(payload["frame_indices"], dtype=np.int64), chunks=(chunk,))
    _replace_array(crop, "detection_source", np.asarray(payload["detection_source"], dtype=np.int8), chunks=(chunk,))
    _replace_string_array(crop, "label_state", np.asarray(payload["label_state"], dtype=object))
    _replace_string_array(crop, "label_origin", np.asarray(payload["label_origin"], dtype=object))
    crop.attrs.update(
        {
            "promotion_schema_version": PROMOTION_SCHEMA_VERSION,
            "detection_source_type": "manual",
            "includes_interpolated": False,
            "n_rows": count,
            "n_positive": int(np.sum(np.asarray(payload["label_state"], dtype=object) == "positive")),
            "n_negative": int(np.sum(np.asarray(payload["label_state"], dtype=object) == "negative")),
            "updated_at_utc": _utc_now(),
        }
    )
    mark_run_complete(
        crop,
        parent_group=crop_parent,
        run_name=str(crop_run),
        run_provenance=build_writer_run_provenance(
            command="fisheye.tune.detect_training_promotion_backend",
            params={
                "promotion_schema_version": PROMOTION_SCHEMA_VERSION,
                "detection_source_type": "manual",
                "n_rows": count,
            },
            input_run_ids={"crop_run": str(crop_run)},
        ),
    )


def _write_source_index(root: zarr.Group, source_index: Mapping[str, np.ndarray]) -> None:
    group = root.require_group("source_index")
    count = int(np.asarray(source_index["source_frame_idx"]).shape[0])
    chunk = max(1, min(8192, count or 1))
    for name in _SOURCE_STRING_FIELDS:
        _replace_string_array(group, name, source_index[name])
    for name, (dtype, _default) in _SOURCE_NUMERIC_FIELDS.items():
        _replace_array(group, name, np.asarray(source_index[name], dtype=dtype), chunks=(chunk,))
    group.attrs["promotion_schema_version"] = PROMOTION_SCHEMA_VERSION
    group.attrs["row_count"] = count
    group.attrs["updated_at_utc"] = _utc_now()


def _write_original_frame_indices(root: zarr.Group, values: np.ndarray) -> None:
    raw = root.require_group("raw_video")
    count = int(values.shape[0])
    chunk = max(1, min(8192, count or 1))
    _replace_array(raw, "original_frame_indices", np.asarray(values, dtype=np.int64), chunks=(chunk,))


def _resolve_training_refined_run(
    root: zarr.Group,
    requested: str | None,
) -> tuple[str, zarr.Group]:
    parent = require_runs_parent(root, "refined_detect_runs")
    run_name = str(
        requested
        or resolve_latest_complete_run_name(parent)
        or DEFAULT_PROMOTED_REFINED_RUN
    ).strip()
    if not run_name:
        run_name = DEFAULT_PROMOTED_REFINED_RUN
    run = parent.require_group(run_name)
    mark_run_started(run, run_name=run_name, stage="refined_detect")
    mark_run_pending(parent, run_name)
    return run_name, run


def _approve_promoted_refined_detect_authority(
    *,
    training_zarr_path: Path,
    run_name: str,
) -> dict[str, object]:
    """Route promotion authority through the palette approve funnel (fail-closed)."""

    from fisheye.cli.palette import ApproveRequest, approve

    envelope = approve(
        ApproveRequest(
            recording=training_zarr_path,
            stage="refined_detect",
            run=run_name,
            approved_by=None,
            note="detect training promotion",
            apply=True,
        )
    )
    if str(envelope.get("status") or "").strip().lower() != "ok":
        reason = envelope.get("reason_code") or "UNKNOWN"
        hints = envelope.get("next_hints") or []
        raise RuntimeError(
            "promotion could not set authoritative refined detect run "
            f"{run_name!r} for {training_zarr_path}: {reason}; hints={hints}"
        )
    return envelope


def _mirror_authoritative_approval(
    parent: zarr.Group,
    run_name: str,
    envelope: Mapping[str, object],
) -> None:
    approval = envelope.get("approval")
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


def _read_optional_instance_array(
    instances: zarr.Group | None,
    name: str,
    *,
    dtype: np.dtype | type,
) -> np.ndarray | None:
    if instances is None or name not in instances:
        return None
    return np.asarray(instances[name][:], dtype=dtype)


def _read_optional_instance_reasons(instances: zarr.Group | None, count: int) -> np.ndarray:
    if instances is None:
        return np.full(count, "", dtype=object)
    labels = read_reason_labels(instances)
    if labels is None:
        return np.full(count, "", dtype=object)
    labels_arr = np.asarray(labels, dtype=object).reshape(-1)
    if labels_arr.shape[0] != count:
        return np.full(count, "", dtype=object)
    return labels_arr


def _write_empty_source_detections(refined_run: zarr.Group) -> None:
    source = refined_run.require_group("source_detections")
    empty_i32 = np.empty((0,), dtype=np.int32)
    empty_i64 = np.empty((0,), dtype=np.int64)
    empty_bbox = np.empty((0, 4), dtype=np.float64)
    for name, data in (
        ("source_detect_row_index", empty_i32),
        ("frame_indices", empty_i32),
        ("bbox_img_xyxy", empty_bbox),
        ("bbox_norm_coords", empty_bbox),
        ("decision_codes", np.empty((0,), dtype=np.int8)),
        ("resolved_refined_row_id", empty_i64),
    ):
        _write_common_array(source, name, data)
    write_reason_columns(
        source,
        np.empty((0,), dtype=object),
        1,
        overwrite=True,
    )
    source.attrs["decision_code_map"] = dict(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP)
    source.attrs["source_detect_run"] = ""
    source.attrs["decision_projection_policy"] = "empty_promoted_training_manual_rows"


def _drop_inline_consolidated_metadata(group: zarr.Group) -> None:
    """Remove stale inline consolidated child metadata after mutable Zarr v3 writes."""
    try:
        store_root = getattr(group.store_path.store, "root", None)
        group_path = str(getattr(group, "path", "") or "").strip("/")
    except Exception:
        return
    if store_root is None:
        return
    metadata_path = Path(store_root) / group_path / "zarr.json"
    if not metadata_path.is_file():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if "consolidated_metadata" not in payload:
        return
    payload.pop("consolidated_metadata", None)
    tmp = metadata_path.with_name(f".{metadata_path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, metadata_path)


def _reopen_group_direct(group: zarr.Group, *, mode: str = "a") -> zarr.Group:
    try:
        store = group.store_path.store
        path = str(getattr(group, "path", "") or "")
        return zarr.open_group(store=store, path=path, mode=mode, use_consolidated=False)
    except Exception:
        return group


def _sync_training_refined_instances_from_crop(
    root: zarr.Group,
    *,
    training_zarr_path: Path,
    target_crop_run: str,
    target_refined_run: str | None,
    source_index: Mapping[str, np.ndarray],
    image_count: int,
) -> dict[str, Any]:
    """Mirror promoted training labels into the canonical refined instances surface.

    The crop/source-index arrays remain the materialized training-image surface,
    but `refined_detect_runs/<run>/instances` is the canonical label table used
    by refined-source exporters and downstream consumers.
    """
    crop_parent = root.get("crop_runs")
    if crop_parent is None or str(target_crop_run) not in crop_parent:
        raise RuntimeError(f"Cannot sync refined instances: crop run not found: {target_crop_run}")
    crop = crop_parent[str(target_crop_run)]
    crop_payload = _read_crop_payload(root, str(target_crop_run), image_count=int(image_count))
    bbox_norm_all = np.asarray(crop_payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    label_state = np.asarray(crop_payload["label_state"], dtype=object).reshape(-1)
    label_origin = np.asarray(crop_payload["label_origin"], dtype=object).reshape(-1)
    crop_frame_indices = np.asarray(crop_payload["frame_indices"], dtype=np.int64).reshape(-1)
    finite_bbox = np.all(np.isfinite(bbox_norm_all), axis=1)
    positive_mask = (label_state == "positive") & finite_bbox
    positive_rows = np.flatnonzero(positive_mask).astype(np.int64, copy=False)

    run_name, refined_run = _resolve_training_refined_run(root, target_refined_run)
    _drop_inline_consolidated_metadata(refined_run)
    refined_run = _reopen_group_direct(refined_run, mode="a")
    old_instances = refined_run.get("instances")
    if old_instances is not None:
        _drop_inline_consolidated_metadata(old_instances)
        old_instances = _reopen_group_direct(old_instances, mode="a")
    old_count = (
        int(old_instances["frame_indices"].shape[0])
        if old_instances is not None and "frame_indices" in old_instances
        else 0
    )
    old_frame_indices = (
        np.asarray(old_instances["frame_indices"][:], dtype=np.int64).reshape(-1)
        if old_instances is not None and "frame_indices" in old_instances
        else np.empty((0,), dtype=np.int64)
    )
    old_row_by_frame = {
        int(frame): int(idx)
        for idx, frame in enumerate(old_frame_indices.tolist())
    }
    old_refined_row_ids = _read_optional_instance_array(old_instances, "refined_row_ids", dtype=np.int64)
    old_source_kind_codes = _read_optional_instance_array(old_instances, "source_kind_codes", dtype=np.int8)
    old_manual_edit_flags = _read_optional_instance_array(old_instances, "manual_edit_flags", dtype=bool)
    old_source_detect_row_index = _read_optional_instance_array(old_instances, "source_detect_row_index", dtype=np.int32)
    old_confidence_scores = _read_optional_instance_array(old_instances, "confidence_scores", dtype=np.float32)
    old_class_ids = _read_optional_instance_array(old_instances, "class_ids", dtype=np.int32)
    old_reasons = _read_optional_instance_reasons(old_instances, old_count)
    old_review_notes = (
        np.asarray(old_instances["review_notes"][:], dtype=object).reshape(-1)
        if old_instances is not None and "review_notes" in old_instances and int(old_instances["review_notes"].shape[0]) == old_count
        else np.full(old_count, "", dtype=object)
    )

    frame_indices = np.asarray(crop_frame_indices[positive_rows], dtype=np.int32)
    bbox_norm = np.asarray(bbox_norm_all[positive_rows], dtype=np.float64).reshape(-1, 4)
    row_count = int(frame_indices.shape[0])
    refined_row_ids = np.arange(row_count, dtype=np.int64)
    source_kind_codes = np.full(row_count, REFINED_SOURCE_KIND_CODE_MAP["manual"], dtype=np.int8)
    manual_edit_flags = np.ones(row_count, dtype=bool)
    source_detect_row_index = np.full(row_count, -1, dtype=np.int32)
    confidence_scores = np.ones(row_count, dtype=np.float32)
    class_ids = np.zeros(row_count, dtype=np.int32)
    reason_labels = np.full(row_count, "manual_review", dtype=object)
    review_notes = np.full(row_count, "", dtype=object)

    for out_idx, crop_row in enumerate(positive_rows.tolist()):
        frame_idx = int(crop_frame_indices[int(crop_row)])
        is_promoted_manual = str(label_origin[int(crop_row)]) == "manual_review"
        old_idx = old_row_by_frame.get(frame_idx)
        if old_idx is None or is_promoted_manual:
            continue
        if old_refined_row_ids is not None and old_idx < old_refined_row_ids.shape[0]:
            refined_row_ids[out_idx] = int(old_refined_row_ids[old_idx])
        if old_source_kind_codes is not None and old_idx < old_source_kind_codes.shape[0]:
            source_kind_codes[out_idx] = int(old_source_kind_codes[old_idx])
        if old_manual_edit_flags is not None and old_idx < old_manual_edit_flags.shape[0]:
            manual_edit_flags[out_idx] = bool(old_manual_edit_flags[old_idx])
        if old_source_detect_row_index is not None and old_idx < old_source_detect_row_index.shape[0]:
            source_detect_row_index[out_idx] = int(old_source_detect_row_index[old_idx])
        if old_confidence_scores is not None and old_idx < old_confidence_scores.shape[0]:
            confidence_scores[out_idx] = float(old_confidence_scores[old_idx])
        if old_class_ids is not None and old_idx < old_class_ids.shape[0]:
            class_ids[out_idx] = int(old_class_ids[old_idx])
        if old_idx < old_reasons.shape[0] and str(old_reasons[old_idx]).strip():
            reason_labels[out_idx] = str(old_reasons[old_idx])
        if old_idx < old_review_notes.shape[0]:
            review_notes[out_idx] = str(old_review_notes[old_idx])

    # If old IDs were not contiguous with the new promoted rows, normalize them
    # to a stable frame-sorted identity. This preserves source_detections links
    # for the common migrated-training case where old IDs were already 0..N-1.
    if refined_row_ids.size and len(set(refined_row_ids.tolist())) != int(refined_row_ids.shape[0]):
        refined_row_ids = np.arange(row_count, dtype=np.int64)

    raw_group = root.get("raw_video")
    images_full = raw_group.get("images_full") if raw_group is not None else None
    width = int(root.attrs.get("width") or (images_full.shape[2] if images_full is not None else 0))
    height = int(root.attrs.get("height") or (images_full.shape[1] if images_full is not None else 0))
    if width <= 0 or height <= 0:
        width = int(root["raw_video/images_ds"].shape[2])
        height = int(root["raw_video/images_ds"].shape[1])
    bbox_img = _bbox_norm_to_img_xyxy_with_missing(bbox_norm, width=width, height=height)

    order = np.lexsort((refined_row_ids, frame_indices)) if row_count else np.empty((0,), dtype=np.int64)
    frame_indices = frame_indices[order]
    refined_row_ids = refined_row_ids[order]
    bbox_norm = bbox_norm[order]
    bbox_img = bbox_img[order]
    source_kind_codes = source_kind_codes[order]
    manual_edit_flags = manual_edit_flags[order]
    source_detect_row_index = source_detect_row_index[order]
    confidence_scores = confidence_scores[order]
    class_ids = class_ids[order]
    reason_labels = reason_labels[order]
    review_notes = review_notes[order]

    frame_count = int(image_count)
    if frame_indices.size:
        frame_count = max(frame_count, int(np.max(frame_indices)) + 1)
    frame_counts = np.bincount(frame_indices, minlength=frame_count).astype(np.int32, copy=False)
    frame_offsets = np.zeros(frame_count + 1, dtype=np.int64)
    if frame_count:
        frame_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)

    instances = refined_run.require_group("instances")
    for name, data in (
        ("refined_row_ids", refined_row_ids),
        ("frame_indices", frame_indices),
        ("frame_offsets", frame_offsets),
        ("bbox_img_xyxy", bbox_img),
        ("bbox_norm_coords", bbox_norm),
        ("source_kind_codes", source_kind_codes),
        ("manual_edit_flags", manual_edit_flags),
        ("source_detect_row_index", source_detect_row_index),
        ("frame_counts", frame_counts),
        ("confidence_scores", confidence_scores),
        ("class_ids", class_ids),
    ):
        _write_common_array(instances, name, data)
    write_reason_columns(
        instances,
        reason_labels,
        max(1, row_count),
        overwrite=True,
    )
    _replace_string_array(instances, "review_notes", review_notes)
    _write_bbox_coordinate_attrs(
        instances,
        bbox_img_width=width,
        bbox_img_height=height,
        bbox_norm_reference_width=width,
        bbox_norm_reference_height=height,
    )
    instances.attrs["row_sort_order"] = ["frame_indices", "refined_row_ids"]
    instances.attrs["source_kind_code_map"] = dict(REFINED_SOURCE_KIND_CODE_MAP)
    instances.attrs["promotion_source_crop_run"] = str(target_crop_run)
    instances.attrs["promotion_schema_version"] = PROMOTION_SCHEMA_VERSION

    if "source_detections" not in refined_run:
        _write_empty_source_detections(refined_run)
    source_detections = refined_run.get("source_detections")

    refined_run.attrs.update(
        {
            "curated_row_storage": "sparse_instances_v1",
            "curated_primary_surface": "instances",
            "refined_storage_semantics": "sparse_instances_v1",
            "entity_assignment_policy": "local_instance_index_per_frame",
            "coordinate_space": "full_image_xyxy",
            "row_identity_policy": "stable_sparse_refined_row_id",
            "status_code_map": dict(REFINED_DETECT_STATUS_CODE_MAP),
            "source_kind_code_map": dict(REFINED_SOURCE_KIND_CODE_MAP),
            "review_state_code_map": dict(REFINED_REVIEW_STATE_CODE_MAP),
            "bbox_coordinate_contract_version": "refined_detect_bbox_coordinates_v2",
            "promotion_schema_version": PROMOTION_SCHEMA_VERSION,
            "promotion_source_crop_run": str(target_crop_run),
            "updated_at_utc": _utc_now(),
        }
    )
    for key in (
        "bbox_img_xyxy_coordinate_space",
        "bbox_img_xyxy_reference_width",
        "bbox_img_xyxy_reference_height",
        "bbox_norm_coords_format",
        "bbox_norm_coords_coordinate_space",
        "bbox_norm_reference_width",
        "bbox_norm_reference_height",
        "bbox_norm_reference_space",
    ):
        refined_run.attrs[key] = instances.attrs[key]
    if "detect_review_status" not in refined_run.attrs:
        refined_run.attrs["detect_review_status"] = {
            "state": "approved",
            "method": "promotion",
            "intended_use": "training",
            "timestamp": _utc_now(),
            "resolved_group": "refined",
        }
    refined_run.attrs["summary_statistics"] = _build_sparse_refined_detect_summary(
        refined_run,
        total_frames=frame_count,
    )
    crop.attrs["canonical_refined_detect_run"] = run_name
    crop.attrs["canonical_refined_detect_path"] = f"refined_detect_runs/{run_name}/instances"
    _drop_inline_consolidated_metadata(instances)
    if source_detections is not None:
        _drop_inline_consolidated_metadata(source_detections)
    _drop_inline_consolidated_metadata(refined_run)
    refined_parent = root.get("refined_detect_runs")
    authoritative_approval: dict[str, object] | None = None
    if refined_parent is not None:
        _drop_inline_consolidated_metadata(refined_parent)
        mark_run_complete(
            refined_run,
            parent_group=refined_parent,
            run_name=run_name,
            run_provenance=build_writer_run_provenance(
                command="fisheye.tune.detect_training_promotion_backend",
                params={
                    "promotion_schema_version": PROMOTION_SCHEMA_VERSION,
                    "target_refined_run": run_name,
                    "source_instances_path": instances.path,
                },
                input_run_ids={
                    "training_zarr_path": str(training_zarr_path),
                    "source_crop_run": crop.attrs.get("source_crop_run"),
                    "promoted_crop_run": str(target_crop_run),
                },
            ),
        )
        envelope = _approve_promoted_refined_detect_authority(
            training_zarr_path=training_zarr_path,
            run_name=run_name,
        )
        _mirror_authoritative_approval(refined_parent, run_name, envelope)
        authoritative_approval = {
            "status": envelope.get("status"),
            "reason_code": envelope.get("reason_code"),
            "run": envelope.get("run"),
        }
    return {
        "status": "ok",
        "refined_detect_run": run_name,
        "authoritative_approval": authoritative_approval,
        "refined_instances_path": f"refined_detect_runs/{run_name}/instances",
        "rows_instances": row_count,
        "manual_rows": int(np.sum(manual_edit_flags)),
        "raw_rows": int(np.sum(source_kind_codes == REFINED_SOURCE_KIND_CODE_MAP["raw_detect"])),
        "source_detect_indexed_rows": int(np.sum(source_detect_row_index >= 0)),
    }


def _set_row(
    *,
    crop_payload: dict[str, np.ndarray],
    source_index: dict[str, np.ndarray],
    original_frame_indices: np.ndarray,
    row: int,
    identity: tuple[str, str, int, int],
    label_state: str,
    bbox: np.ndarray,
    label_origin: str,
    source_refined_row_id: int,
    source_detect_row_index: int,
) -> None:
    crop_payload["bbox_norm_coords"][row] = bbox
    crop_payload["label_state"][row] = str(label_state)
    crop_payload["label_origin"][row] = str(label_origin)
    crop_payload["detection_source"][row] = np.int8(0)
    crop_payload["frame_indices"][row] = np.int64(row)
    analysis_path, refined_run, frame_idx, entity_id = identity
    source_index["source_analysis_zarr_path"][row] = str(analysis_path)
    source_index["source_refined_detect_run"][row] = str(refined_run)
    source_index["source_frame_idx"][row] = np.int64(frame_idx)
    source_index["source_entity_id"][row] = np.int32(entity_id)
    source_index["source_refined_row_ids"][row] = np.int64(source_refined_row_id)
    source_index["source_detect_row_index"][row] = np.int32(source_detect_row_index)
    source_index["source_refined_group_path"][row] = ""
    source_index["source_collection_id"][row] = ""
    source_index["source_clip_id"][row] = ""
    source_index["source_camera_serial"][row] = ""
    source_index["source_video_path"][row] = ""
    source_index["source_clip_index"][row] = np.int32(-1)
    source_index["source_clip_local_frame_index"][row] = np.int64(-1)
    source_index["source_parent_frame_index"][row] = np.int64(-1)
    source_index["source_recording_frame_id"][row] = np.int64(-1)
    original_frame_indices[row] = np.int64(frame_idx)


def _append_payload_row(
    *,
    crop_payload: dict[str, np.ndarray],
    source_index: dict[str, np.ndarray],
    original_frame_indices: np.ndarray,
    identity: tuple[str, str, int, int],
    label_state: str,
    bbox: np.ndarray,
    label_origin: str,
    source_refined_row_id: int,
    source_detect_row_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    row = int(crop_payload["bbox_norm_coords"].shape[0])
    crop_payload = {key: np.array(value, copy=True) for key, value in crop_payload.items()}
    source_index = {key: np.array(value, copy=True) for key, value in source_index.items()}
    crop_payload["bbox_norm_coords"] = _append_bbox(crop_payload["bbox_norm_coords"], bbox)
    crop_payload["frame_indices"] = _append_row(crop_payload["frame_indices"], row, dtype=np.int64)
    crop_payload["detection_source"] = _append_row(crop_payload["detection_source"], 0, dtype=np.int8)
    crop_payload["label_state"] = _append_row(crop_payload["label_state"], label_state, dtype=object)
    crop_payload["label_origin"] = _append_row(crop_payload["label_origin"], label_origin, dtype=object)
    source_index["source_analysis_zarr_path"] = _append_row(
        source_index["source_analysis_zarr_path"],
        identity[0],
        dtype=object,
    )
    source_index["source_refined_detect_run"] = _append_row(
        source_index["source_refined_detect_run"],
        identity[1],
        dtype=object,
    )
    source_index["source_frame_idx"] = _append_row(source_index["source_frame_idx"], identity[2], dtype=np.int64)
    source_index["source_entity_id"] = _append_row(source_index["source_entity_id"], identity[3], dtype=np.int32)
    source_index["source_refined_row_ids"] = _append_row(
        source_index["source_refined_row_ids"],
        source_refined_row_id,
        dtype=np.int64,
    )
    source_index["source_detect_row_index"] = _append_row(
        source_index["source_detect_row_index"],
        source_detect_row_index,
        dtype=np.int32,
    )
    source_index["source_refined_group_path"] = _append_row(
        source_index["source_refined_group_path"],
        "",
        dtype=object,
    )
    source_index["source_collection_id"] = _append_row(source_index["source_collection_id"], "", dtype=object)
    source_index["source_clip_id"] = _append_row(source_index["source_clip_id"], "", dtype=object)
    source_index["source_camera_serial"] = _append_row(source_index["source_camera_serial"], "", dtype=object)
    source_index["source_video_path"] = _append_row(source_index["source_video_path"], "", dtype=object)
    source_index["source_clip_index"] = _append_row(source_index["source_clip_index"], -1, dtype=np.int32)
    source_index["source_clip_local_frame_index"] = _append_row(
        source_index["source_clip_local_frame_index"],
        -1,
        dtype=np.int64,
    )
    source_index["source_parent_frame_index"] = _append_row(
        source_index["source_parent_frame_index"],
        -1,
        dtype=np.int64,
    )
    source_index["source_recording_frame_id"] = _append_row(
        source_index["source_recording_frame_id"],
        -1,
        dtype=np.int64,
    )
    original_frame_indices = _append_row(original_frame_indices, identity[2], dtype=np.int64)
    return crop_payload, source_index, original_frame_indices, row


def _promotable_label(
    *,
    status: str,
    bbox: np.ndarray | None,
    manual_edit: bool,
    options: PromotionOptions,
) -> tuple[str | None, np.ndarray, str | None]:
    if status == "present" and bbox is not None:
        return "positive", bbox, None
    if not options.include_negative:
        return None, np.full((4,), np.nan, dtype=np.float32), "negative_promotion_disabled"
    if manual_edit or options.allow_unreviewed_negative:
        return "negative", np.full((4,), np.nan, dtype=np.float32), None
    return None, np.full((4,), np.nan, dtype=np.float32), "non_present_without_manual_edit"


def _clipped_refined_group(root: zarr.Group, refined_group_path: str) -> zarr.Group:
    path = str(refined_group_path).strip("/")
    if not path:
        raise RuntimeError("Clipped promotion requires a refined_group_path.")
    try:
        group = root[path]
    except Exception as exc:
        raise RuntimeError(f"Clipped refined group not found: {path}") from exc
    if not isinstance(group, zarr.Group):
        raise RuntimeError(f"Clipped refined path is not a group: {path}")
    return group


def _clipped_identity(
    *,
    analysis_zarr: Path,
    frame: ClippedPromotionFrame,
) -> tuple[str, str, str, str, str, int, int, int]:
    return (
        str(analysis_zarr),
        str(frame.collection_id),
        str(frame.refined_group_path).strip("/"),
        str(frame.clip_id),
        str(frame.camera_serial),
        int(frame.parent_frame_index),
        int(frame.clip_local_frame_index),
        int(frame.entity_id),
    )


def _find_existing_clipped_rows(
    *,
    source_index: Mapping[str, np.ndarray],
    identity: tuple[str, str, str, str, str, int, int, int],
) -> np.ndarray:
    analysis_path, collection_id, refined_group_path, clip_id, camera_serial, parent_idx, local_idx, entity_id = identity
    mask = (
        (np.asarray(source_index["source_analysis_zarr_path"], dtype=object) == str(analysis_path))
        & (np.asarray(source_index["source_collection_id"], dtype=object) == str(collection_id))
        & (np.asarray(source_index["source_refined_group_path"], dtype=object) == str(refined_group_path))
        & (np.asarray(source_index["source_clip_id"], dtype=object) == str(clip_id))
        & (np.asarray(source_index["source_camera_serial"], dtype=object) == str(camera_serial))
        & (np.asarray(source_index["source_parent_frame_index"], dtype=np.int64) == int(parent_idx))
        & (np.asarray(source_index["source_clip_local_frame_index"], dtype=np.int64) == int(local_idx))
        & (np.asarray(source_index["source_entity_id"], dtype=np.int32) == int(entity_id))
    )
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def _clipped_row_matches(
    *,
    crop_payload: Mapping[str, np.ndarray],
    source_index: Mapping[str, np.ndarray],
    row: int,
    identity: tuple[str, str, str, str, str, int, int, int],
    frame: ClippedPromotionFrame,
    label_state: str,
    bbox: np.ndarray,
    label_origin: str,
    source_refined_row_id: int,
    source_detect_row_index: int,
) -> bool:
    if str(crop_payload["label_state"][row]) != str(label_state):
        return False
    if str(crop_payload["label_origin"][row]) != str(label_origin):
        return False
    current_bbox = np.asarray(crop_payload["bbox_norm_coords"][row], dtype=np.float32)
    if label_state == "positive":
        if not np.allclose(current_bbox, bbox, atol=1e-6, equal_nan=True):
            return False
    elif np.any(np.isfinite(current_bbox)):
        return False
    analysis_path, collection_id, refined_group_path, clip_id, camera_serial, parent_idx, local_idx, entity_id = identity
    return (
        str(source_index["source_analysis_zarr_path"][row]) == str(analysis_path)
        and str(source_index["source_collection_id"][row]) == str(collection_id)
        and str(source_index["source_refined_group_path"][row]) == str(refined_group_path)
        and str(source_index["source_clip_id"][row]) == str(clip_id)
        and str(source_index["source_camera_serial"][row]) == str(camera_serial)
        and str(source_index["source_refined_detect_run"][row]) == str(frame.refined_run)
        and str(source_index["source_video_path"][row]) == str(Path(frame.source_video_path))
        and int(source_index["source_frame_idx"][row]) == int(parent_idx)
        and int(source_index["source_parent_frame_index"][row]) == int(parent_idx)
        and int(source_index["source_clip_local_frame_index"][row]) == int(local_idx)
        and int(source_index["source_clip_index"][row]) == int(frame.clip_index)
        and int(source_index["source_recording_frame_id"][row]) == int(frame.recording_frame_id or -1)
        and int(source_index["source_entity_id"][row]) == int(entity_id)
        and int(source_index["source_refined_row_ids"][row]) == int(source_refined_row_id)
        and int(source_index["source_detect_row_index"][row]) == int(source_detect_row_index)
    )


def _set_clipped_row(
    *,
    crop_payload: dict[str, np.ndarray],
    source_index: dict[str, np.ndarray],
    original_frame_indices: np.ndarray,
    row: int,
    identity: tuple[str, str, str, str, str, int, int, int],
    frame: ClippedPromotionFrame,
    label_state: str,
    bbox: np.ndarray,
    label_origin: str,
    source_refined_row_id: int,
    source_detect_row_index: int,
) -> None:
    analysis_path, collection_id, refined_group_path, clip_id, camera_serial, parent_idx, local_idx, entity_id = identity
    crop_payload["bbox_norm_coords"][row] = bbox
    crop_payload["label_state"][row] = str(label_state)
    crop_payload["label_origin"][row] = str(label_origin)
    crop_payload["detection_source"][row] = np.int8(0)
    crop_payload["frame_indices"][row] = np.int64(row)
    source_index["source_analysis_zarr_path"][row] = str(analysis_path)
    source_index["source_refined_detect_run"][row] = str(frame.refined_run)
    source_index["source_refined_group_path"][row] = str(refined_group_path)
    source_index["source_collection_id"][row] = str(collection_id)
    source_index["source_clip_id"][row] = str(clip_id)
    source_index["source_camera_serial"][row] = str(camera_serial)
    source_index["source_video_path"][row] = str(Path(frame.source_video_path))
    source_index["source_frame_idx"][row] = np.int64(parent_idx)
    source_index["source_entity_id"][row] = np.int32(entity_id)
    source_index["source_refined_row_ids"][row] = np.int64(source_refined_row_id)
    source_index["source_detect_row_index"][row] = np.int32(source_detect_row_index)
    source_index["source_clip_index"][row] = np.int32(frame.clip_index)
    source_index["source_clip_local_frame_index"][row] = np.int64(local_idx)
    source_index["source_parent_frame_index"][row] = np.int64(parent_idx)
    source_index["source_recording_frame_id"][row] = np.int64(frame.recording_frame_id or -1)
    original_frame_indices[row] = np.int64(parent_idx)


def _append_clipped_payload_row(
    *,
    crop_payload: dict[str, np.ndarray],
    source_index: dict[str, np.ndarray],
    original_frame_indices: np.ndarray,
    identity: tuple[str, str, str, str, str, int, int, int],
    frame: ClippedPromotionFrame,
    label_state: str,
    bbox: np.ndarray,
    label_origin: str,
    source_refined_row_id: int,
    source_detect_row_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    analysis_path, _collection_id, _refined_group_path, _clip_id, _camera_serial, parent_idx, _local_idx, entity_id = identity
    crop_payload, source_index, original_frame_indices, row = _append_payload_row(
        crop_payload=crop_payload,
        source_index=source_index,
        original_frame_indices=original_frame_indices,
        identity=(analysis_path, frame.refined_run, parent_idx, entity_id),
        label_state=label_state,
        bbox=bbox,
        label_origin=label_origin,
        source_refined_row_id=source_refined_row_id,
        source_detect_row_index=source_detect_row_index,
    )
    _set_clipped_row(
        crop_payload=crop_payload,
        source_index=source_index,
        original_frame_indices=original_frame_indices,
        row=row,
        identity=identity,
        frame=frame,
        label_state=label_state,
        bbox=bbox,
        label_origin=label_origin,
        source_refined_row_id=source_refined_row_id,
        source_detect_row_index=source_detect_row_index,
    )
    return crop_payload, source_index, original_frame_indices, row


def promote_detection_frames(
    analysis_zarr: str | Path,
    training_zarr: str | Path,
    frames: Sequence[int],
    *,
    options: PromotionOptions | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Plan or apply detection-label promotion for selected frames."""
    started = time.perf_counter()
    opts = options or PromotionOptions()
    requested_frames = sorted({int(frame) for frame in frames})
    if not requested_frames:
        raise ValueError("At least one source frame must be requested for promotion.")

    analysis_path = Path(analysis_zarr).expanduser().resolve()
    training_path = Path(training_zarr).expanduser().resolve()
    analysis_root = open_zarr_group_direct(analysis_path, mode="r")
    refined_run_name, refined_group = _resolve_refined_run(analysis_root, opts.refined_run)
    payload = _load_refined_payload(analysis_root, refined_group=refined_group, frames=requested_frames)

    training_root = _ensure_training_root(training_path, apply=apply)
    training_exists = training_root is not None
    target_crop_run = _resolve_training_crop_run(training_root, opts.target_crop_run) if training_root else (
        opts.target_crop_run or DEFAULT_PROMOTED_CROP_RUN
    )

    if training_root is not None and "raw_video" in training_root and "images_ds" in training_root["raw_video"]:
        image_shape = tuple(int(v) for v in training_root["raw_video/images_ds"].shape[1:3])
    elif opts.target_size is not None:
        image_shape = (int(opts.target_size[0]), int(opts.target_size[1]))
    elif "raw_video" in analysis_root and "images_ds" in analysis_root["raw_video"]:
        image_shape = tuple(int(v) for v in analysis_root["raw_video/images_ds"].shape[1:3])
    else:
        image_shape = (640, 640)

    image_count = 0
    images_ds: zarr.Array | None = None
    crop_payload: dict[str, np.ndarray] | None = None
    source_index: dict[str, np.ndarray] | None = None
    original_frame_indices: np.ndarray | None = None
    if training_root is not None:
        images_ds = _ensure_images_ds(training_root, initial_shape=image_shape)
        image_shape = tuple(int(v) for v in images_ds.shape[1:3])
        image_count = int(images_ds.shape[0])
        original_frame_indices = _ensure_original_frame_indices(training_root, count=image_count)
        crop_payload = _read_crop_payload(training_root, target_crop_run, image_count=image_count)
        source_index = _read_source_index(training_root, count=image_count)

    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    bbox_arr = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
    status_labels = np.asarray(payload["status_labels"], dtype=object).reshape(-1)
    manual_flags = np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)
    refined_row_ids = np.asarray(payload.get("refined_row_ids", np.full(frame_indices.shape, -1)), dtype=np.int64).reshape(-1)
    source_detect_row_index = np.asarray(
        payload.get("source_detect_row_index", np.full(frame_indices.shape, -1)),
        dtype=np.int32,
    ).reshape(-1)
    row_by_frame = {int(frame): int(idx) for idx, frame in enumerate(frame_indices.tolist())}

    items: list[dict[str, Any]] = []
    images_appended = 0
    images_updated = 0
    image_sources: list[dict[str, object]] = []
    refined_instances_sync: dict[str, Any] | None = None

    for frame in requested_frames:
        row_idx = row_by_frame.get(int(frame))
        if row_idx is None:
            items.append({"frame": int(frame), "action": "skip", "reason": "frame_not_in_refined_payload"})
            continue
        status = str(status_labels[row_idx])
        bbox = _finite_bbox_or_none(bbox_arr[row_idx])
        label_state, target_bbox, skip_reason = _promotable_label(
            status=status,
            bbox=bbox,
            manual_edit=bool(manual_flags[row_idx]),
            options=opts,
        )
        identity = _source_identity(analysis_zarr=analysis_path, refined_run=refined_run_name, frame=int(frame))
        if label_state is None:
            items.append(
                {
                    "frame": int(frame),
                    "action": "skip",
                    "reason": skip_reason,
                    "status_label": status,
                    "manual_edit": bool(manual_flags[row_idx]),
                }
            )
            continue
        if not training_exists:
            items.append(
                {
                    "frame": int(frame),
                    "action": "append",
                    "label_state": label_state,
                    "target_row": None,
                    "reason": "training_zarr_missing_dry_run",
                }
            )
            continue

        assert training_root is not None
        assert images_ds is not None
        assert crop_payload is not None
        assert source_index is not None
        assert original_frame_indices is not None
        matches = _find_existing_rows(
            source_index=source_index,
            original_frame_indices=original_frame_indices,
            identity=identity,
        )
        if matches.size > 1:
            items.append(
                {
                    "frame": int(frame),
                    "action": "conflict",
                    "reason": "multiple_matching_training_rows",
                    "matching_rows": [int(v) for v in matches.tolist()],
                }
            )
            continue
        source_refined_row_id = int(refined_row_ids[row_idx])
        source_detect_idx = int(source_detect_row_index[row_idx])
        if matches.size == 1:
            target_row = int(matches[0])
            unchanged = _row_matches(
                crop_payload=crop_payload,
                source_index=source_index,
                row=target_row,
                identity=identity,
                label_state=label_state,
                bbox=target_bbox,
                label_origin=opts.label_origin,
                source_refined_row_id=source_refined_row_id,
                source_detect_row_index=source_detect_idx,
            )
            action = "no_change" if unchanged else "update"
            items.append(
                {
                    "frame": int(frame),
                    "action": action,
                    "label_state": label_state,
                    "target_row": target_row,
                    "status_label": status,
                    "manual_edit": bool(manual_flags[row_idx]),
                }
            )
            if apply and action == "update":
                _set_row(
                    crop_payload=crop_payload,
                    source_index=source_index,
                    original_frame_indices=original_frame_indices,
                    row=target_row,
                    identity=identity,
                    label_state=label_state,
                    bbox=target_bbox,
                    label_origin=opts.label_origin,
                    source_refined_row_id=source_refined_row_id,
                    source_detect_row_index=source_detect_idx,
                )
                images_updated += 1
            continue

        target_row = int(crop_payload["bbox_norm_coords"].shape[0])
        items.append(
            {
                "frame": int(frame),
                "action": "append",
                "label_state": label_state,
                "target_row": target_row,
                "status_label": status,
                "manual_edit": bool(manual_flags[row_idx]),
            }
        )
        if apply:
            image, source_meta = _load_source_image(
                analysis_path,
                analysis_root,
                frame_idx=int(frame),
                shape_hw=image_shape,
            )
            _resize_axis0(images_ds, target_row + 1)
            images_ds[target_row] = image
            crop_payload, source_index, original_frame_indices, _ = _append_payload_row(
                crop_payload=crop_payload,
                source_index=source_index,
                original_frame_indices=original_frame_indices,
                identity=identity,
                label_state=label_state,
                bbox=target_bbox,
                label_origin=opts.label_origin,
                source_refined_row_id=source_refined_row_id,
                source_detect_row_index=source_detect_idx,
            )
            images_appended += 1
            image_sources.append({"frame": int(frame), **source_meta})

    actions = {}
    for item in items:
        actions[str(item["action"])] = int(actions.get(str(item["action"]), 0)) + 1
    conflicts = [item for item in items if item.get("action") == "conflict"]
    if apply and conflicts:
        raise RuntimeError(f"Promotion has conflicts; refusing to write: {conflicts[:3]}")

    if apply and training_root is not None and crop_payload is not None and source_index is not None and original_frame_indices is not None:
        _write_crop_payload(training_root, target_crop_run, crop_payload)
        _write_source_index(training_root, source_index)
        _write_original_frame_indices(training_root, original_frame_indices)
        if opts.sync_refined_instances:
            refined_instances_sync = _sync_training_refined_instances_from_crop(
                training_root,
                training_zarr_path=training_path,
                target_crop_run=target_crop_run,
                target_refined_run=opts.target_refined_run,
                source_index=source_index,
                image_count=int(crop_payload["bbox_norm_coords"].shape[0]),
            )
        training_root.attrs["zarr_purpose"] = "training"
        training_root.attrs["last_analysis_detect_promotion"] = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "updated_at_utc": _utc_now(),
            "analysis_zarr": str(analysis_path),
            "refined_detect_run": str(refined_run_name),
            "target_crop_run": str(target_crop_run),
            "frames_requested": [int(frame) for frame in requested_frames],
            "action_counts": actions,
            "label_origin": opts.label_origin,
            "image_sources": image_sources,
            "refined_instances_sync": refined_instances_sync,
        }

    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "ok" if not conflicts else "conflict",
        "apply": bool(apply),
        "analysis_zarr": str(analysis_path),
        "training_zarr": str(training_path),
        "training_zarr_exists": bool(training_exists),
        "refined_detect_run": refined_run_name,
        "target_crop_run": target_crop_run,
        "frames_requested": [int(frame) for frame in requested_frames],
        "action_counts": actions,
        "items": items,
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "images_appended": int(images_appended),
        "images_updated": int(images_updated),
        "image_sources": image_sources,
        "refined_instances_sync": refined_instances_sync,
        "label_origin": opts.label_origin,
        "include_negative": bool(opts.include_negative),
        "allow_unreviewed_negative": bool(opts.allow_unreviewed_negative),
        "created_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "timing": {"total_seconds": time.perf_counter() - started},
    }


def promote_clipped_detection_frames(
    analysis_zarr: str | Path,
    training_zarr: str | Path,
    frames: Sequence[ClippedPromotionFrame],
    *,
    options: PromotionOptions | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Plan or apply clipped detection-label promotion for selected parent frames."""
    started = time.perf_counter()
    opts = options or PromotionOptions()
    requested_frames = sorted(frames, key=lambda frame: (int(frame.parent_frame_index), int(frame.entity_id)))
    if not requested_frames:
        raise ValueError("At least one clipped source frame must be requested for promotion.")

    analysis_path = Path(analysis_zarr).expanduser().resolve()
    training_path = Path(training_zarr).expanduser().resolve()
    analysis_root = open_zarr_group_direct(analysis_path, mode="r")
    training_root = _ensure_training_root(training_path, apply=apply)
    training_exists = training_root is not None
    target_crop_run = _resolve_training_crop_run(training_root, opts.target_crop_run) if training_root else (
        opts.target_crop_run or DEFAULT_PROMOTED_CROP_RUN
    )

    if training_root is not None and "raw_video" in training_root and "images_ds" in training_root["raw_video"]:
        image_shape = tuple(int(v) for v in training_root["raw_video/images_ds"].shape[1:3])
    elif opts.target_size is not None:
        image_shape = (int(opts.target_size[0]), int(opts.target_size[1]))
    else:
        image_shape = (640, 640)

    image_count = 0
    images_ds: zarr.Array | None = None
    images_full: zarr.Array | None = None
    crop_payload: dict[str, np.ndarray] | None = None
    source_index: dict[str, np.ndarray] | None = None
    original_frame_indices: np.ndarray | None = None
    if training_root is not None:
        images_ds = _ensure_images_ds(training_root, initial_shape=image_shape)
        image_shape = tuple(int(v) for v in images_ds.shape[1:3])
        image_count = int(images_ds.shape[0])
        images_full = _existing_images_full(training_root)
        if images_full is not None and int(images_full.shape[0]) != image_count:
            raise RuntimeError(
                f"raw_video/images_full length {images_full.shape[0]} does not match images_ds length {image_count}."
            )
        original_frame_indices = _ensure_original_frame_indices(training_root, count=image_count)
        crop_payload = _read_crop_payload(training_root, target_crop_run, image_count=image_count)
        source_index = _read_source_index(training_root, count=image_count)

    payload_cache: dict[str, dict[str, np.ndarray]] = {}

    def payload_for(frame: ClippedPromotionFrame) -> dict[str, np.ndarray]:
        path = str(frame.refined_group_path).strip("/")
        if path not in payload_cache:
            payload_cache[path] = _load_refined_payload(
                analysis_root,
                refined_group=_clipped_refined_group(analysis_root, path),
                frames=[int(frame.clip_local_frame_index)],
            )
        return payload_cache[path]

    items: list[dict[str, Any]] = []
    images_appended = 0
    images_updated = 0
    image_sources: list[dict[str, object]] = []
    decode_groups: list[dict[str, object]] = []
    append_plans: list[dict[str, Any]] = []
    planned_identities: set[tuple[str, str, str, str, str, int, int, int]] = set()
    refined_instances_sync: dict[str, Any] | None = None
    timing_breakdown: dict[str, float] = {
        "decode_seconds": 0.0,
        "dataset_resize_seconds": 0.0,
        "image_write_seconds": 0.0,
        "payload_append_seconds": 0.0,
        "existing_row_update_seconds": 0.0,
        "zarr_metadata_write_seconds": 0.0,
        "refined_instances_sync_seconds": 0.0,
    }

    for frame in requested_frames:
        payload = payload_for(frame)
        frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
        row_by_frame = {int(value): int(idx) for idx, value in enumerate(frame_indices.tolist())}
        row_idx = row_by_frame.get(int(frame.clip_local_frame_index))
        item_base = {
            "parent_frame": int(frame.parent_frame_index),
            "clip_local_frame": int(frame.clip_local_frame_index),
            "clip_id": str(frame.clip_id),
            "camera_serial": str(frame.camera_serial),
            "refined_group_path": str(frame.refined_group_path).strip("/"),
        }
        if row_idx is None:
            items.append({**item_base, "action": "skip", "reason": "frame_not_in_refined_payload"})
            continue

        bbox_arr = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)
        status_labels = np.asarray(payload["status_labels"], dtype=object).reshape(-1)
        manual_flags = np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1)
        refined_row_ids = np.asarray(
            payload.get("refined_row_ids", np.full(frame_indices.shape, -1)),
            dtype=np.int64,
        ).reshape(-1)
        source_detect_row_index = np.asarray(
            payload.get("source_detect_row_index", np.full(frame_indices.shape, -1)),
            dtype=np.int32,
        ).reshape(-1)

        status = str(status_labels[row_idx])
        bbox = _finite_bbox_or_none(bbox_arr[row_idx])
        label_state, target_bbox, skip_reason = _promotable_label(
            status=status,
            bbox=bbox,
            manual_edit=bool(manual_flags[row_idx]),
            options=opts,
        )
        identity = _clipped_identity(analysis_zarr=analysis_path, frame=frame)
        if label_state is None:
            items.append(
                {
                    **item_base,
                    "action": "skip",
                    "reason": skip_reason,
                    "status_label": status,
                    "manual_edit": bool(manual_flags[row_idx]),
                }
            )
            continue
        if not training_exists:
            items.append(
                {
                    **item_base,
                    "action": "append",
                    "label_state": label_state,
                    "target_row": None,
                    "reason": "training_zarr_missing_dry_run",
                }
            )
            continue

        assert training_root is not None
        assert images_ds is not None
        assert crop_payload is not None
        assert source_index is not None
        assert original_frame_indices is not None
        matches = _find_existing_clipped_rows(source_index=source_index, identity=identity)
        if matches.size > 1:
            items.append(
                {
                    **item_base,
                    "action": "conflict",
                    "reason": "multiple_matching_training_rows",
                    "matching_rows": [int(v) for v in matches.tolist()],
                }
            )
            continue
        if matches.size == 0 and identity in planned_identities:
            items.append(
                {
                    **item_base,
                    "action": "conflict",
                    "reason": "duplicate_requested_source_identity",
                }
            )
            continue
        source_refined_row_id = int(refined_row_ids[row_idx])
        source_detect_idx = int(source_detect_row_index[row_idx])
        if matches.size == 1:
            target_row = int(matches[0])
            unchanged = _clipped_row_matches(
                crop_payload=crop_payload,
                source_index=source_index,
                row=target_row,
                identity=identity,
                frame=frame,
                label_state=label_state,
                bbox=target_bbox,
                label_origin=opts.label_origin,
                source_refined_row_id=source_refined_row_id,
                source_detect_row_index=source_detect_idx,
            )
            action = "no_change" if unchanged else "update"
            items.append(
                {
                    **item_base,
                    "action": action,
                    "label_state": label_state,
                    "target_row": target_row,
                    "status_label": status,
                    "manual_edit": bool(manual_flags[row_idx]),
                }
            )
            if apply and action == "update":
                t_update = time.perf_counter()
                _set_clipped_row(
                    crop_payload=crop_payload,
                    source_index=source_index,
                    original_frame_indices=original_frame_indices,
                    row=target_row,
                    identity=identity,
                    frame=frame,
                    label_state=label_state,
                    bbox=target_bbox,
                    label_origin=opts.label_origin,
                    source_refined_row_id=source_refined_row_id,
                    source_detect_row_index=source_detect_idx,
                )
                timing_breakdown["existing_row_update_seconds"] += time.perf_counter() - t_update
                images_updated += 1
            continue

        target_row = int(crop_payload["bbox_norm_coords"].shape[0])
        if apply:
            target_row += len(append_plans)
        items.append(
            {
                **item_base,
                "action": "append",
                "label_state": label_state,
                "target_row": target_row,
                "status_label": status,
                "manual_edit": bool(manual_flags[row_idx]),
            }
        )
        if apply:
            planned_identities.add(identity)
            append_plans.append(
                {
                    "target_row": int(target_row),
                    "identity": identity,
                    "frame": frame,
                    "label_state": label_state,
                    "bbox": target_bbox,
                    "source_refined_row_id": int(source_refined_row_id),
                    "source_detect_row_index": int(source_detect_idx),
                }
            )

    if apply and append_plans:
        assert images_ds is not None
        assert crop_payload is not None
        assert source_index is not None
        assert original_frame_indices is not None
        initial_count = int(images_ds.shape[0])
        final_count = initial_count + len(append_plans)
        t_resize = time.perf_counter()
        _resize_axis0(images_ds, final_count)
        full_shape = None
        if images_full is not None:
            full_shape = tuple(int(v) for v in images_full.shape[1:3])
            _resize_axis0(images_full, final_count)
        timing_breakdown["dataset_resize_seconds"] += time.perf_counter() - t_resize

        plans_by_video: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for plan in append_plans:
            source_video_path = Path(plan["frame"].source_video_path).expanduser().resolve()
            plans_by_video[source_video_path].append(plan)

        for source_video_path, plans in plans_by_video.items():
            local_frames = [int(plan["frame"].clip_local_frame_index) for plan in plans]
            t_decode_group = time.perf_counter()
            decoded_frames, decode_method = _decode_video_frames_gray_native_batch(source_video_path, local_frames)
            decode_seconds = time.perf_counter() - t_decode_group
            timing_breakdown["decode_seconds"] += decode_seconds
            decode_groups.append(
                {
                    "image_source_path": str(source_video_path),
                    "decode_method": str(decode_method),
                    "requested_frame_count": int(len(set(local_frames))),
                    "min_clip_local_frame": int(min(local_frames)),
                    "max_clip_local_frame": int(max(local_frames)),
                    "seconds": float(decode_seconds),
                }
            )
            for plan in plans:
                frame = plan["frame"]
                local_frame = int(frame.clip_local_frame_index)
                target_row = int(plan["target_row"])
                full_image = decoded_frames[local_frame]
                t_image_write = time.perf_counter()
                images_ds[target_row] = _resize_gray(full_image, image_shape)
                if images_full is None:
                    assert training_root is not None
                    images_full = _ensure_images_full(training_root, initial_shape=tuple(int(v) for v in full_image.shape[:2]))
                    _resize_axis0(images_full, final_count)
                    full_shape = tuple(int(v) for v in images_full.shape[1:3])
                if images_full is not None and full_shape is not None:
                    images_full[target_row] = _resize_gray(full_image, full_shape)
                timing_breakdown["image_write_seconds"] += time.perf_counter() - t_image_write
                t_payload_append = time.perf_counter()
                crop_payload, source_index, original_frame_indices, appended_row = _append_clipped_payload_row(
                    crop_payload=crop_payload,
                    source_index=source_index,
                    original_frame_indices=original_frame_indices,
                    identity=plan["identity"],
                    frame=frame,
                    label_state=str(plan["label_state"]),
                    bbox=np.asarray(plan["bbox"], dtype=np.float32),
                    label_origin=opts.label_origin,
                    source_refined_row_id=int(plan["source_refined_row_id"]),
                    source_detect_row_index=int(plan["source_detect_row_index"]),
                )
                timing_breakdown["payload_append_seconds"] += time.perf_counter() - t_payload_append
                if int(appended_row) != target_row:
                    raise RuntimeError(
                        f"Internal clipped promotion target row mismatch: planned={target_row} appended={appended_row}"
                    )
                images_appended += 1
                image_sources.append(
                    {
                        "parent_frame": int(frame.parent_frame_index),
                        "clip_local_frame": int(local_frame),
                        "image_source": "clip_source_video",
                        "image_source_path": str(source_video_path),
                        "decode_method": decode_method,
                        "wrote_images_full": images_full is not None,
                    }
                )

    actions: dict[str, int] = {}
    for item in items:
        actions[str(item["action"])] = int(actions.get(str(item["action"]), 0)) + 1
    conflicts = [item for item in items if item.get("action") == "conflict"]
    if apply and conflicts:
        raise RuntimeError(f"Promotion has conflicts; refusing to write: {conflicts[:3]}")

    if apply and training_root is not None and crop_payload is not None and source_index is not None and original_frame_indices is not None:
        t_zarr_metadata = time.perf_counter()
        _write_crop_payload(training_root, target_crop_run, crop_payload)
        _write_source_index(training_root, source_index)
        _write_original_frame_indices(training_root, original_frame_indices)
        if opts.sync_refined_instances:
            t_refined_sync = time.perf_counter()
            refined_instances_sync = _sync_training_refined_instances_from_crop(
                training_root,
                training_zarr_path=training_path,
                target_crop_run=target_crop_run,
                target_refined_run=opts.target_refined_run,
                source_index=source_index,
                image_count=int(crop_payload["bbox_norm_coords"].shape[0]),
            )
            timing_breakdown["refined_instances_sync_seconds"] += time.perf_counter() - t_refined_sync
        timing_breakdown["zarr_metadata_write_seconds"] += time.perf_counter() - t_zarr_metadata
        training_root.attrs["zarr_purpose"] = "training"
        training_root.attrs["last_analysis_detect_promotion"] = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "mode": "clipped",
            "updated_at_utc": _utc_now(),
            "analysis_zarr": str(analysis_path),
            "training_zarr": str(training_path),
            "collection_ids": sorted({str(frame.collection_id) for frame in requested_frames}),
            "target_crop_run": str(target_crop_run),
            "parent_frames_requested": [int(frame.parent_frame_index) for frame in requested_frames],
            "action_counts": actions,
            "label_origin": opts.label_origin,
            "image_sources": image_sources,
            "decode_groups": decode_groups,
            "refined_instances_sync": refined_instances_sync,
            "timing_breakdown": timing_breakdown,
        }

    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "ok" if not conflicts else "conflict",
        "mode": "clipped",
        "apply": bool(apply),
        "analysis_zarr": str(analysis_path),
        "training_zarr": str(training_path),
        "training_zarr_exists": bool(training_exists),
        "target_crop_run": target_crop_run,
        "frames_requested": [int(frame.parent_frame_index) for frame in requested_frames],
        "clip_local_frames_requested": [int(frame.clip_local_frame_index) for frame in requested_frames],
        "action_counts": actions,
        "items": items,
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "images_appended": int(images_appended),
        "images_updated": int(images_updated),
        "image_sources": image_sources,
        "decode_groups": decode_groups,
        "refined_instances_sync": refined_instances_sync,
        "label_origin": opts.label_origin,
        "include_negative": bool(opts.include_negative),
        "allow_unreviewed_negative": bool(opts.allow_unreviewed_negative),
        "created_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "timing": {
            "total_seconds": time.perf_counter() - started,
            **{key: float(value) for key, value in timing_breakdown.items()},
            "decode_group_count": int(len(decode_groups)),
        },
    }


def write_promotion_result(path: str | Path, result: Mapping[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, output)
