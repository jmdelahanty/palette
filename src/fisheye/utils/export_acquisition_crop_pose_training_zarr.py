#!/usr/bin/env python3
"""Export pose-training rows from acquisition-time crop videos."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import zarr

from fisheye.diagnostics.compare_realtime_offline_detections import (
    infer_recording_dir_from_zarr,
    resolve_crop_meta_path,
)
from fisheye.shared.crop_geometry import (
    bbox_img_xyxy_to_norm_cxcywh,
    bbox_roi_xyxy_to_img_xyxy,
    resolve_full_frame_shape,
)
from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader
from fisheye.shared.roi_pixel_contract import (
    APPLIED_RANGE_SEMANTICS_ORANGE_MONO_FULL_RANGE,
    CENTER_ROUNDING_NP_ROUND,
    DECODE_BACKEND_PYNVVC_LUMA,
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
    resolve_authoritative_run_name,
)
from fisheye.shared.system_metadata import get_environment_info, get_git_info


SCHEMA_ID = "palette.acquisition_crop_pose_training_export.v1"
DEFAULT_CROP_RUN_PREFIX = "crop_acquisition_crop_video_pose"
DEFAULT_KEYPOINT_RUN_PREFIX = "keypoints_acquisition_crop_video_pose"


@dataclass(frozen=True)
class CropVideoStreamInfo:
    crop_video_path: str
    crop_meta_path: str
    width: int | None
    height: int | None
    codec_name: str | None
    pix_fmt: str | None
    color_range: str | None
    color_space: str | None
    nb_frames: str | None
    frame_format_confirmation_status: str


@dataclass(frozen=True)
class AcquisitionCropPoseReport:
    zarr_path: str
    recording_dir: str
    crop_video: CropVideoStreamInfo
    keypoint_parent: str
    keypoint_run: str
    source_keypoint_rows: int
    usable_keypoint_rows: int
    crop_meta_rows: int
    selected_rows: int
    margin_px: float
    reject_counts: dict[str, int]
    crop_width_stats: dict[str, float | None]
    crop_height_stats: dict[str, float | None]
    crop_video_dim_matches_crop_meta: bool | None
    applied: bool
    out_zarr: str | None = None
    crop_run: str | None = None
    keypoint_export_run: str | None = None


@dataclass(frozen=True)
class _CropMetaTable:
    frame_indices: np.ndarray
    video_frame_indices: np.ndarray
    local_frame_ids: np.ndarray
    row_indices: np.ndarray
    has_detection: np.ndarray
    blank_frame: np.ndarray
    crop_xywh: np.ndarray
    detection_xywh: np.ndarray


@dataclass(frozen=True)
class _KeypointTable:
    parent_name: str
    run_name: str
    group: zarr.Group
    frame_indices: np.ndarray
    keypoints_img: np.ndarray
    success: np.ndarray
    source_row_indices: np.ndarray
    confidences: np.ndarray | None
    keypoint_confidences: np.ndarray | None
    heading: np.ndarray | None
    keypoint_labels: list[str]
    pose_schema: dict[str, Any] | None
    skeleton_id: str | None


@dataclass(frozen=True)
class _Selection:
    source_keypoint_rows: np.ndarray
    source_frames: np.ndarray
    crop_meta_rows: np.ndarray
    crop_video_frame_indices: np.ndarray
    crop_local_frame_ids: np.ndarray
    source_recording_frame_ids: np.ndarray
    source_crop_xywh: np.ndarray
    keypoints_img: np.ndarray
    keypoints_roi: np.ndarray
    keypoints_norm: np.ndarray
    bbox_roi_xyxy: np.ndarray
    bbox_img_xyxy: np.ndarray
    bbox_norm_xywh: np.ndarray
    bbox_crop_norm_xywh: np.ndarray
    realtime_detection_bbox_roi_xyxy: np.ndarray
    success: np.ndarray


def _utc_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _open_root(path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(path), mode=mode, use_consolidated=False)


def _safe_float(text: object) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def _safe_int(text: object, *, default: int = -1) -> int:
    try:
        return int(float(str(text)))
    except Exception:
        return int(default)


def _load_manifest(recording_dir: Path) -> dict[str, Any]:
    path = recording_dir / "recording_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _crop_stream(recording_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest(recording_dir)
    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, dict):
        return {}
    streams = video_streams.get("streams")
    if not isinstance(streams, dict):
        return {}
    crop = streams.get("crop")
    return crop if isinstance(crop, dict) else {}


def resolve_crop_video_path(recording_dir: Path) -> Path:
    crop = _crop_stream(recording_dir)
    video = crop.get("video")
    if video:
        path = Path(str(video))
        if not path.is_absolute():
            path = recording_dir / path
        if path.exists():
            return path
    candidates = sorted((recording_dir / "derived" / "external_crop_recorder").glob("*_crop_external.mp4"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"Multiple acquisition crop videos found under {recording_dir}; pass --crop-video.")
    raise ValueError(f"No acquisition crop video found under {recording_dir}.")


def _ffprobe_stream(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,codec_name,pix_fmt,color_range,color_space,color_transfer,color_primaries,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return {"available": False, "error": "ffprobe not found"}
    if result.returncode != 0:
        return {"available": False, "error": (result.stderr or result.stdout).strip()}
    try:
        payload = json.loads(result.stdout)
    except Exception as exc:
        return {"available": False, "error": f"ffprobe JSON parse failed: {exc}"}
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return {"available": False, "error": "ffprobe returned no video streams"}
    stream = streams[0]
    return stream if isinstance(stream, dict) else {"available": False, "error": "ffprobe stream is not an object"}


def inspect_crop_video_stream(recording_dir: Path, crop_meta_path: Path, crop_video_path: Path) -> CropVideoStreamInfo:
    stream = _ffprobe_stream(crop_video_path)
    return CropVideoStreamInfo(
        crop_video_path=str(crop_video_path),
        crop_meta_path=str(crop_meta_path),
        width=_safe_int(stream.get("width"), default=0) or None,
        height=_safe_int(stream.get("height"), default=0) or None,
        codec_name=str(stream.get("codec_name")) if stream.get("codec_name") is not None else None,
        pix_fmt=str(stream.get("pix_fmt")) if stream.get("pix_fmt") is not None else None,
        color_range=str(stream.get("color_range")) if stream.get("color_range") is not None else None,
        color_space=str(stream.get("color_space")) if stream.get("color_space") is not None else None,
        nb_frames=str(stream.get("nb_frames")) if stream.get("nb_frames") is not None else None,
        frame_format_confirmation_status="pending_orange_confirmation",
    )


def load_crop_meta_table(crop_meta_path: Path) -> _CropMetaTable:
    frame_indices: list[int] = []
    video_frame_indices: list[int] = []
    local_frame_ids: list[int] = []
    row_indices: list[int] = []
    has_detection: list[bool] = []
    blank_frame: list[bool] = []
    crop_xywh: list[tuple[float, float, float, float]] = []
    detection_xywh: list[tuple[float, float, float, float]] = []

    with crop_meta_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"recording_frame_id", "crop_x", "crop_y", "crop_w", "crop_h"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Crop metadata missing required columns: {missing}")
        for row_index, row in enumerate(reader):
            frame = _safe_int(row.get("recording_frame_id"), default=0) - 1
            video_frame = _safe_int(row.get("crop_video_frame_index"), default=row_index)
            local_frame = _safe_int(row.get("local_frame_id"), default=row_index)
            frame_indices.append(frame)
            video_frame_indices.append(video_frame)
            local_frame_ids.append(local_frame)
            row_indices.append(row_index)
            has_detection.append(bool(_safe_int(row.get("has_detection"), default=0)))
            blank_frame.append(bool(_safe_int(row.get("blank_frame"), default=0)))
            crop_x = _safe_float(row.get("crop_x"))
            crop_y = _safe_float(row.get("crop_y"))
            crop_w = _safe_float(row.get("crop_w"))
            crop_h = _safe_float(row.get("crop_h"))
            crop_xywh.append((crop_x, crop_y, crop_w, crop_h))
            det_x = _safe_float(row.get("detection_x"))
            det_y = _safe_float(row.get("detection_y"))
            det_w = _safe_float(row.get("detection_w"))
            det_h = _safe_float(row.get("detection_h"))
            detection_xywh.append((det_x, det_y, det_w, det_h))

    return _CropMetaTable(
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        video_frame_indices=np.asarray(video_frame_indices, dtype=np.int64),
        local_frame_ids=np.asarray(local_frame_ids, dtype=np.int64),
        row_indices=np.asarray(row_indices, dtype=np.int64),
        has_detection=np.asarray(has_detection, dtype=bool),
        blank_frame=np.asarray(blank_frame, dtype=bool),
        crop_xywh=np.asarray(crop_xywh, dtype=np.float64).reshape(-1, 4),
        detection_xywh=np.asarray(detection_xywh, dtype=np.float64).reshape(-1, 4),
    )


def _array_or_none(group: zarr.Group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][:])


def _attr_list_text(attrs: Any, key: str, count: int) -> list[str]:
    value = attrs.get(key)
    if isinstance(value, (list, tuple)) and len(value) == count:
        return [str(item) for item in value]
    pose_schema = attrs.get("pose_schema")
    if isinstance(pose_schema, dict):
        for candidate in ("keypoint_labels", "nodes"):
            raw = pose_schema.get(candidate)
            if isinstance(raw, (list, tuple)) and len(raw) == count:
                return [str(item) for item in raw]
    return [f"kpt_{idx}" for idx in range(count)]


def _resolve_keypoint_table(
    root: zarr.Group,
    *,
    keypoint_run: str | None,
    keypoint_parent: str,
) -> _KeypointTable:
    parent = root.get(keypoint_parent)
    if parent is None:
        raise ValueError(f"Archive has no {keypoint_parent}.")
    resolved_run = str(keypoint_run).strip() if keypoint_run else None
    if not resolved_run:
        resolved_run = resolve_authoritative_run_name(parent)
    if not resolved_run:
        latest = parent.attrs.get("latest")
        resolved_run = str(latest).strip() if latest else None
    if not resolved_run or resolved_run not in parent:
        raise ValueError(f"Could not resolve keypoint run under {keypoint_parent}.")
    group = parent[resolved_run]
    for name in ("frame_indices", "keypoints_img"):
        if name not in group:
            raise ValueError(f"{keypoint_parent}/{resolved_run} missing required array {name!r}.")
    frame_indices = np.asarray(group["frame_indices"][:], dtype=np.int64)
    keypoints_img = np.asarray(group["keypoints_img"][:], dtype=np.float64)
    if keypoints_img.ndim != 3 or int(keypoints_img.shape[2]) != 2:
        raise ValueError(f"keypoints_img must have shape (N,K,2), got {tuple(keypoints_img.shape)}.")

    if "usable_keypoints" in group:
        success = np.asarray(group["usable_keypoints"][:], dtype=bool)
    elif "refined_success" in group:
        success = np.asarray(group["refined_success"][:], dtype=bool)
    elif "detection_success" in group:
        success = np.asarray(group["detection_success"][:], dtype=bool)
    else:
        success = np.isfinite(keypoints_img).all(axis=(1, 2))
    if success.shape[0] != frame_indices.shape[0]:
        raise ValueError("keypoint success array length does not match frame_indices.")

    source_row_indices = (
        np.asarray(group["source_refined_row_ids"][:], dtype=np.int64)
        if "source_refined_row_ids" in group
        else np.arange(frame_indices.shape[0], dtype=np.int64)
    )
    if source_row_indices.shape[0] != frame_indices.shape[0]:
        source_row_indices = np.arange(frame_indices.shape[0], dtype=np.int64)

    confidences = _array_or_none(group, "confidence")
    if confidences is not None:
        confidences = np.asarray(confidences, dtype=np.float64)
    keypoint_confidences = _array_or_none(group, "keypoint_confidences")
    if keypoint_confidences is not None:
        keypoint_confidences = np.asarray(keypoint_confidences, dtype=np.float64)
    heading = _array_or_none(group, "heading")
    if heading is not None:
        heading = np.asarray(heading, dtype=np.float64)

    labels = _attr_list_text(group.attrs, "keypoint_labels", int(keypoints_img.shape[1]))
    pose_schema = group.attrs.get("pose_schema")
    pose_schema_dict = dict(pose_schema) if isinstance(pose_schema, dict) else None
    skeleton_id = group.attrs.get("skeleton_id")
    if skeleton_id is None and pose_schema_dict is not None:
        skeleton_id = pose_schema_dict.get("skeleton_id")

    return _KeypointTable(
        parent_name=keypoint_parent,
        run_name=resolved_run,
        group=group,
        frame_indices=frame_indices,
        keypoints_img=keypoints_img,
        success=success,
        source_row_indices=source_row_indices,
        confidences=confidences,
        keypoint_confidences=keypoint_confidences,
        heading=heading,
        keypoint_labels=labels,
        pose_schema=pose_schema_dict,
        skeleton_id=str(skeleton_id) if skeleton_id is not None else None,
    )


def _stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "median": None, "max": None}
    return {"min": float(np.min(arr)), "median": float(np.median(arr)), "max": float(np.max(arr))}


def _bbox_xyxy_to_norm_xywh(bbox: np.ndarray, *, width: int, height: int) -> np.ndarray:
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    out = np.column_stack(
        [
            ((x1 + x2) * 0.5) / float(width),
            ((y1 + y2) * 0.5) / float(height),
            (x2 - x1) / float(width),
            (y2 - y1) / float(height),
        ]
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _select_rows(
    keypoints: _KeypointTable,
    crop_meta: _CropMetaTable,
    *,
    frame_width: int,
    frame_height: int,
    output_width: int,
    output_height: int,
    margin_px: float,
) -> tuple[_Selection, dict[str, int]]:
    crop_pos_by_frame = {int(frame): idx for idx, frame in enumerate(crop_meta.frame_indices.tolist())}
    selected_keypoint_rows: list[int] = []
    selected_crop_rows: list[int] = []
    reject_counts: dict[str, int] = {
        "source_not_usable": 0,
        "missing_crop_meta_frame": 0,
        "blank_crop_frame": 0,
        "crop_has_no_detection": 0,
        "nonfinite_crop_geometry": 0,
        "nonfinite_keypoints": 0,
        "keypoints_outside_crop_margin": 0,
    }

    keypoints_roi_all: list[np.ndarray] = []
    keypoints_norm_all: list[np.ndarray] = []
    bbox_roi_all: list[tuple[float, float, float, float]] = []
    bbox_img_all: list[np.ndarray] = []
    bbox_norm_all: list[np.ndarray] = []
    bbox_crop_norm_all: list[np.ndarray] = []
    realtime_bbox_roi_all: list[tuple[float, float, float, float]] = []

    for kp_row, frame in enumerate(keypoints.frame_indices.astype(np.int64, copy=False)):
        if not bool(keypoints.success[kp_row]):
            reject_counts["source_not_usable"] += 1
            continue
        crop_row = crop_pos_by_frame.get(int(frame))
        if crop_row is None:
            reject_counts["missing_crop_meta_frame"] += 1
            continue
        if bool(crop_meta.blank_frame[crop_row]):
            reject_counts["blank_crop_frame"] += 1
            continue
        if not bool(crop_meta.has_detection[crop_row]):
            reject_counts["crop_has_no_detection"] += 1
            continue

        crop_x, crop_y, crop_w, crop_h = crop_meta.crop_xywh[crop_row]
        if not np.isfinite([crop_x, crop_y, crop_w, crop_h]).all() or crop_w <= 0.0 or crop_h <= 0.0:
            reject_counts["nonfinite_crop_geometry"] += 1
            continue
        kp_img = keypoints.keypoints_img[kp_row]
        if not np.isfinite(kp_img).all():
            reject_counts["nonfinite_keypoints"] += 1
            continue

        scale_x = float(output_width) / float(crop_w)
        scale_y = float(output_height) / float(crop_h)
        kp_roi = np.empty_like(kp_img, dtype=np.float64)
        kp_roi[:, 0] = (kp_img[:, 0] - crop_x) * scale_x
        kp_roi[:, 1] = (kp_img[:, 1] - crop_y) * scale_y
        inside = (
            (kp_roi[:, 0] >= float(margin_px))
            & (kp_roi[:, 1] >= float(margin_px))
            & (kp_roi[:, 0] <= float(output_width) - float(margin_px))
            & (kp_roi[:, 1] <= float(output_height) - float(margin_px))
        )
        if not bool(np.all(inside)):
            reject_counts["keypoints_outside_crop_margin"] += 1
            continue

        x1 = float(np.min(kp_roi[:, 0]))
        y1 = float(np.min(kp_roi[:, 1]))
        x2 = float(np.max(kp_roi[:, 0]))
        y2 = float(np.max(kp_roi[:, 1]))
        bbox_roi = np.asarray([[x1, y1, x2, y2]], dtype=np.float64)
        bbox_img = bbox_roi_xyxy_to_img_xyxy(
            bbox_roi,
            np.asarray([[crop_x, crop_y, crop_w, crop_h]], dtype=np.float64),
            roi_width=int(output_width),
            roi_height=int(output_height),
        )
        bbox_norm = bbox_img_xyxy_to_norm_cxcywh(
            bbox_img,
            width=int(frame_width),
            height=int(frame_height),
        )[0]
        bbox_crop_norm = _bbox_xyxy_to_norm_xywh(bbox_roi, width=output_width, height=output_height)[0]

        det_x, det_y, det_w, det_h = crop_meta.detection_xywh[crop_row]
        if np.isfinite([det_x, det_y, det_w, det_h]).all() and det_w >= 0.0 and det_h >= 0.0:
            det_bbox = (
                (det_x - crop_x) * scale_x,
                (det_y - crop_y) * scale_y,
                (det_x + det_w - crop_x) * scale_x,
                (det_y + det_h - crop_y) * scale_y,
            )
        else:
            det_bbox = (float("nan"), float("nan"), float("nan"), float("nan"))

        selected_keypoint_rows.append(kp_row)
        selected_crop_rows.append(crop_row)
        keypoints_roi_all.append(kp_roi.astype(np.float64, copy=False))
        keypoints_norm = kp_roi.astype(np.float64, copy=True)
        keypoints_norm[:, 0] /= float(output_width)
        keypoints_norm[:, 1] /= float(output_height)
        keypoints_norm_all.append(keypoints_norm)
        bbox_roi_all.append((x1, y1, x2, y2))
        bbox_img_all.append(bbox_img[0].astype(np.float64, copy=False))
        bbox_norm_all.append(bbox_norm)
        bbox_crop_norm_all.append(bbox_crop_norm)
        realtime_bbox_roi_all.append(det_bbox)

    selected_kp = np.asarray(selected_keypoint_rows, dtype=np.int64)
    selected_crop = np.asarray(selected_crop_rows, dtype=np.int64)
    if selected_kp.size:
        keypoints_roi = np.stack(keypoints_roi_all, axis=0)
        keypoints_norm = np.stack(keypoints_norm_all, axis=0)
        bbox_roi = np.asarray(bbox_roi_all, dtype=np.float32)
        bbox_img = np.asarray(bbox_img_all, dtype=np.float32)
        bbox_norm = np.asarray(bbox_norm_all, dtype=np.float32)
        bbox_crop_norm = np.asarray(bbox_crop_norm_all, dtype=np.float32)
        realtime_bbox_roi = np.asarray(realtime_bbox_roi_all, dtype=np.float32)
    else:
        k_count = int(keypoints.keypoints_img.shape[1])
        keypoints_roi = np.empty((0, k_count, 2), dtype=np.float64)
        keypoints_norm = np.empty((0, k_count, 2), dtype=np.float64)
        bbox_roi = np.empty((0, 4), dtype=np.float32)
        bbox_img = np.empty((0, 4), dtype=np.float32)
        bbox_norm = np.empty((0, 4), dtype=np.float32)
        bbox_crop_norm = np.empty((0, 4), dtype=np.float32)
        realtime_bbox_roi = np.empty((0, 4), dtype=np.float32)

    selection = _Selection(
        source_keypoint_rows=selected_kp,
        source_frames=keypoints.frame_indices[selected_kp].astype(np.int64, copy=False),
        crop_meta_rows=crop_meta.row_indices[selected_crop].astype(np.int64, copy=False),
        crop_video_frame_indices=crop_meta.video_frame_indices[selected_crop].astype(np.int64, copy=False),
        crop_local_frame_ids=crop_meta.local_frame_ids[selected_crop].astype(np.int64, copy=False),
        source_recording_frame_ids=keypoints.frame_indices[selected_kp].astype(np.int64, copy=False) + 1,
        source_crop_xywh=crop_meta.crop_xywh[selected_crop].astype(np.float32, copy=False),
        keypoints_img=keypoints.keypoints_img[selected_kp].astype(np.float64, copy=False),
        keypoints_roi=keypoints_roi,
        keypoints_norm=keypoints_norm,
        bbox_roi_xyxy=bbox_roi,
        bbox_img_xyxy=bbox_img,
        bbox_norm_xywh=bbox_norm,
        bbox_crop_norm_xywh=bbox_crop_norm,
        realtime_detection_bbox_roi_xyxy=realtime_bbox_roi,
        success=np.ones(selected_kp.shape[0], dtype=bool),
    )
    return selection, reject_counts


def inspect_acquisition_crop_pose_training(
    zarr_path: Path,
    *,
    recording_dir: Path | None = None,
    crop_meta_path: Path | None = None,
    crop_video_path: Path | None = None,
    keypoint_parent: str = "refined_keypoints_runs",
    keypoint_run: str | None = None,
    margin_px: float = 4.0,
) -> tuple[AcquisitionCropPoseReport, _Selection, _KeypointTable, _CropMetaTable, Path, Path]:
    zarr_path = Path(zarr_path)
    resolved_recording_dir = Path(recording_dir) if recording_dir is not None else infer_recording_dir_from_zarr(zarr_path)
    resolved_crop_meta = resolve_crop_meta_path(
        zarr_path,
        recording_dir=resolved_recording_dir,
        crop_meta_path=crop_meta_path,
        required=True,
    )
    if resolved_crop_meta is None:
        raise ValueError("Could not resolve acquisition crop metadata.")
    resolved_crop_video = Path(crop_video_path) if crop_video_path is not None else resolve_crop_video_path(resolved_recording_dir)
    if not resolved_crop_video.exists():
        raise FileNotFoundError(f"Crop video not found: {resolved_crop_video}")

    crop_info = inspect_crop_video_stream(resolved_recording_dir, resolved_crop_meta, resolved_crop_video)
    crop_meta = load_crop_meta_table(resolved_crop_meta)
    root = _open_root(zarr_path, mode="r")
    frame_height, frame_width = resolve_full_frame_shape(root)
    keypoints = _resolve_keypoint_table(root, keypoint_run=keypoint_run, keypoint_parent=keypoint_parent)
    output_width = int(crop_info.width or np.nanmedian(crop_meta.crop_xywh[:, 2]))
    output_height = int(crop_info.height or np.nanmedian(crop_meta.crop_xywh[:, 3]))
    selection, reject_counts = _select_rows(
        keypoints,
        crop_meta,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        output_width=output_width,
        output_height=output_height,
        margin_px=float(margin_px),
    )
    width_stats = _stats(crop_meta.crop_xywh[:, 2])
    height_stats = _stats(crop_meta.crop_xywh[:, 3])
    dim_match = None
    if crop_info.width is not None and crop_info.height is not None and width_stats["median"] is not None and height_stats["median"] is not None:
        dim_match = bool(
            int(round(float(width_stats["median"]))) == int(crop_info.width)
            and int(round(float(height_stats["median"]))) == int(crop_info.height)
        )
    report = AcquisitionCropPoseReport(
        zarr_path=str(zarr_path),
        recording_dir=str(resolved_recording_dir),
        crop_video=crop_info,
        keypoint_parent=keypoints.parent_name,
        keypoint_run=keypoints.run_name,
        source_keypoint_rows=int(keypoints.frame_indices.shape[0]),
        usable_keypoint_rows=int(np.count_nonzero(keypoints.success)),
        crop_meta_rows=int(crop_meta.frame_indices.shape[0]),
        selected_rows=int(selection.source_keypoint_rows.shape[0]),
        margin_px=float(margin_px),
        reject_counts=reject_counts,
        crop_width_stats=width_stats,
        crop_height_stats=height_stats,
        crop_video_dim_matches_crop_meta=dim_match,
        applied=False,
    )
    return report, selection, keypoints, crop_meta, resolved_crop_meta, resolved_crop_video


def _create_array(group: zarr.Group, name: str, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
    if name in group:
        del group[name]
    arr = np.asarray(data)
    group.create_array(name, data=arr, chunks=chunks, overwrite=True)


def _read_selected_frames(
    crop_video_path: Path,
    video_frame_indices: np.ndarray,
    *,
    reader_factory: Callable[..., Any] = PynvvcLumaRgbReader,
    gpu_id: int = 0,
    require_cuda: bool = True,
) -> np.ndarray:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "Acquisition crop-video export requires CUDA-enabled torch for PyNvVC decode; "
            "current environment reports torch.cuda.is_available() == False."
        )
    selected = np.asarray(video_frame_indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("No selected crop-video frames to decode.")
    selected_set = set(int(v) for v in selected.tolist())
    max_frame = int(np.max(selected))
    frames: dict[int, np.ndarray] = {}
    reader = reader_factory(crop_video_path, start_frame=0, gpu_id=int(gpu_id))
    try:
        height = int(reader.source_height)
        width = int(reader.source_width)
        with torch.no_grad():
            for frame_idx, frame in enumerate(reader.iter_frames()):
                if frame_idx > max_frame:
                    break
                if frame_idx not in selected_set:
                    continue
                luma = frame[:height, :width].contiguous()
                frames[frame_idx] = luma.to("cpu").numpy().copy()
        missing = [int(v) for v in selected.tolist() if int(v) not in frames]
        if missing:
            raise RuntimeError(f"Crop video ended before selected video frames were decoded: first missing {missing[:5]}")
        return np.stack([frames[int(v)] for v in selected.tolist()], axis=0).astype(np.uint8, copy=False)
    finally:
        reader.close()


def _write_output_zarr(
    *,
    out_zarr: Path,
    overwrite: bool,
    images: np.ndarray,
    selection: _Selection,
    keypoints: _KeypointTable,
    report: AcquisitionCropPoseReport,
    zarr_path: Path,
    crop_meta_path: Path,
    crop_video_path: Path,
    crop_run_name: str,
    keypoint_run_name: str,
    gpu_id: int,
) -> None:
    if out_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"Output zarr already exists: {out_zarr}")
        shutil.rmtree(out_zarr)
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out_zarr), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "zarr_use": "training",
            "schema_id": SCHEMA_ID,
            "training_source_type": "acquisition_crop_video",
            "source_analysis_zarr": str(zarr_path),
            "source_crop_meta_path": str(crop_meta_path),
            "source_crop_video_path": str(crop_video_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    crop_parent = require_runs_parent(root, "crop_runs")
    keypoint_parent = require_runs_parent(root, "keypoints_runs")
    crop_group = crop_parent.create_group(crop_run_name)
    keypoint_group = keypoint_parent.create_group(keypoint_run_name)
    mark_run_started(crop_group, run_name=crop_run_name, stage="crop")
    mark_run_started(keypoint_group, run_name=keypoint_run_name, stage="keypoints")

    row_count = int(images.shape[0])
    height = int(images.shape[1])
    width = int(images.shape[2])
    source_root = _open_root(zarr_path, mode="r")
    frame_height, frame_width = resolve_full_frame_shape(source_root)
    vector_chunks = (max(1, min(8192, row_count)),)
    bbox_chunks = (max(1, min(8192, row_count)), 4)
    image_chunks = (max(1, min(64, row_count)), height, width)
    kpt_chunks = (max(1, min(2048, row_count)), int(selection.keypoints_roi.shape[1]), 2)

    frame_counts = np.bincount(selection.source_frames.astype(np.int64, copy=False), minlength=int(np.max(selection.source_frames)) + 1).astype(np.int32)
    n_keypoints = np.zeros(frame_counts.shape[0], dtype=np.int32)
    n_keypoints[selection.source_frames.astype(np.int64, copy=False)] = int(selection.keypoints_roi.shape[1])

    _create_array(crop_group, "roi_images", images, chunks=image_chunks)
    _create_array(crop_group, "frame_indices", selection.source_frames.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_frame_indices", selection.source_frames.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_recording_frame_ids", selection.source_recording_frame_ids.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_meta_row_indices", selection.crop_meta_rows.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_video_frame_indices", selection.crop_video_frame_indices.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_local_frame_ids", selection.crop_local_frame_ids.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_xywh", selection.source_crop_xywh.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "roi_coordinates_full", selection.source_crop_xywh[:, :2].astype(np.int32), chunks=(bbox_chunks[0], 2))
    _create_array(crop_group, "bbox_roi_xyxy", selection.bbox_roi_xyxy.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "bbox_img_xyxy", selection.bbox_img_xyxy.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "bbox_norm_coords", selection.bbox_norm_xywh.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "bbox_crop_norm_coords", selection.bbox_crop_norm_xywh.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "realtime_detection_bbox_roi_xyxy", selection.realtime_detection_bbox_roi_xyxy.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "detection_indices", np.arange(row_count, dtype=np.int32), chunks=vector_chunks)
    _create_array(crop_group, "frame_counts", frame_counts, chunks=(max(1, min(65536, frame_counts.shape[0])),))
    _create_array(crop_group, "detection_source", np.zeros(row_count, dtype=np.int8), chunks=vector_chunks)

    _create_array(keypoint_group, "frame_indices", selection.source_frames.astype(np.int64), chunks=vector_chunks)
    _create_array(keypoint_group, "frame_counts", frame_counts, chunks=(max(1, min(65536, frame_counts.shape[0])),))
    _create_array(keypoint_group, "detection_indices", np.arange(row_count, dtype=np.int32), chunks=vector_chunks)
    _create_array(keypoint_group, "source_refined_row_ids", keypoints.source_row_indices[selection.source_keypoint_rows].astype(np.int64), chunks=vector_chunks)
    _create_array(keypoint_group, "keypoints_roi", selection.keypoints_roi.astype(np.float64), chunks=kpt_chunks)
    _create_array(keypoint_group, "keypoints_img", selection.keypoints_roi.astype(np.float64), chunks=kpt_chunks)
    _create_array(keypoint_group, "keypoints_norm", selection.keypoints_norm.astype(np.float64), chunks=kpt_chunks)
    _create_array(keypoint_group, "source_keypoints_img", selection.keypoints_img.astype(np.float64), chunks=kpt_chunks)
    _create_array(keypoint_group, "detection_success", selection.success.astype(bool), chunks=vector_chunks)
    _create_array(keypoint_group, "n_keypoints", n_keypoints, chunks=(max(1, min(65536, n_keypoints.shape[0])),))
    if keypoints.confidences is not None:
        _create_array(keypoint_group, "confidence", keypoints.confidences[selection.source_keypoint_rows].astype(np.float64), chunks=vector_chunks)
    else:
        _create_array(keypoint_group, "confidence", np.full(row_count, np.nan, dtype=np.float64), chunks=vector_chunks)
    if keypoints.keypoint_confidences is not None:
        _create_array(
            keypoint_group,
            "keypoint_confidences",
            keypoints.keypoint_confidences[selection.source_keypoint_rows].astype(np.float64),
            chunks=(max(1, min(2048, row_count)), int(selection.keypoints_roi.shape[1])),
        )
    if keypoints.heading is not None:
        _create_array(keypoint_group, "heading", keypoints.heading[selection.source_keypoint_rows].astype(np.float64), chunks=vector_chunks)

    roi_contract = orange_mono_pynvvc_luma_pixel_contract()
    crop_attrs = {
        "schema_id": SCHEMA_ID,
        "crop_storage_mode": "materialized",
        "roi_size": [height, width],
        "roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        "roi_pixel_contract": roi_contract,
        "source_pixels": SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
        "source_pixel_contract": "orange.camera.mono8.full_frame.v1",
        "source_pixel_range": "0_255",
        "decode_backend": DECODE_BACKEND_PYNVVC_LUMA,
        "decode_backend_family": "PyNvVideoCodec",
        "decode_contract_status": "canonical_orange_mono_pynvvc_luma",
        "source_decode_surface": "nv12_y_plane_uint8",
        "applied_range_semantics": APPLIED_RANGE_SEMANTICS_ORANGE_MONO_FULL_RANGE,
        "container_color_range_observed": "tv",
        "container_color_range_handling": roi_contract.get("container_color_range_handling"),
        "center_rounding": CENTER_ROUNDING_NP_ROUND,
        "device": f"cuda:{int(gpu_id)}",
        "source_type": "acquisition_crop_video",
        "source_video_path": str(crop_video_path),
        "source_crop_meta_path": str(crop_meta_path),
        "source_analysis_zarr": str(zarr_path),
        "source_crop_xywh_coordinate_space": "source_image_xywh",
        "roi_coordinates_full_coordinate_space": "source_image_xy",
        "roi_coordinates_full_source": "source_crop_xywh[:, :2]",
        "source_crop_video_frame_indices_semantics": "zero_based_frame_index_in_acquisition_crop_video",
        "source_crop_local_frame_ids_semantics": "orange_acquisition_local_frame_id_not_video_frame_index",
        "bbox_img_xyxy_semantics": "pose_bbox_from_keypoint_extents_xyxy_full_frame_pixels",
        "bbox_norm_coords_semantics": "bbox_xywh_normalized_to_full_frame",
        "bbox_crop_norm_coords_semantics": "pose_bbox_from_keypoint_extents_xywh_normalized_to_crop_video_frame",
        "bbox_roi_xyxy_semantics": "pose_bbox_from_keypoint_extents_crop_video_pixels",
        "bbox_norm_reference_width": int(frame_width),
        "bbox_norm_reference_height": int(frame_height),
        "bbox_norm_reference_space": "source_image",
        "bbox_img_xyxy_reference_width": int(frame_width),
        "bbox_img_xyxy_reference_height": int(frame_height),
        "frame_format_confirmation_status": "pending_orange_confirmation",
        "summary": asdict(report),
    }
    crop_group.attrs.update(crop_attrs)

    keypoint_group.attrs.update(
        {
            "schema_id": SCHEMA_ID,
            "source_crop_run": crop_run_name,
            "keypoint_labels": keypoints.keypoint_labels,
            "pose_schema": keypoints.pose_schema,
            "skeleton_id": keypoints.skeleton_id,
            "keypoint_coordinate_space": "crop_video_frame_px",
            "keypoints_img_coordinate_space": "crop_video_frame_px",
            "source_keypoints_img_coordinate_space": "source_image_px",
            "source_keypoint_parent": keypoints.parent_name,
            "source_keypoint_run": keypoints.run_name,
            "source_analysis_zarr": str(zarr_path),
        }
    )
    git_info = get_git_info(Path(__file__).resolve().parents[3])
    env_info = get_environment_info(include_all_packages=False, disk_path=str(out_zarr), collect_ip=False)
    for stage, group, run_name in (("crop", crop_group, crop_run_name), ("keypoints", keypoint_group, keypoint_run_name)):
        provenance = build_stage_provenance(
            stage=stage,
            command=" ".join(sys.argv),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            version=git_info.get("short_hash") or git_info.get("commit_hash"),
            git=git_info,
            environment=env_info.get("environment"),
            platform=env_info.get("platform"),
            parameters={"margin_px": float(report.margin_px), "gpu_id": int(gpu_id)},
            inputs={
                "analysis_zarr": str(zarr_path),
                "crop_meta": str(crop_meta_path),
                "crop_video": str(crop_video_path),
                "source_keypoints": f"{keypoints.parent_name}/{keypoints.run_name}",
            },
            artifacts={"run_name": run_name, "out_zarr": str(out_zarr)},
        )
        write_stage_provenance(group, provenance)
    mark_run_complete(crop_group, parent_group=crop_parent, run_name=crop_run_name)
    mark_run_complete(keypoint_group, parent_group=keypoint_parent, run_name=keypoint_run_name)


def export_acquisition_crop_pose_training_zarr(
    zarr_path: Path,
    *,
    out_zarr: Path | None = None,
    recording_dir: Path | None = None,
    crop_meta_path: Path | None = None,
    crop_video_path: Path | None = None,
    keypoint_parent: str = "refined_keypoints_runs",
    keypoint_run: str | None = None,
    margin_px: float = 4.0,
    crop_run_name: str | None = None,
    keypoint_export_run_name: str | None = None,
    gpu_id: int = 0,
    overwrite: bool = False,
    apply: bool = False,
    require_cuda: bool = True,
    reader_factory: Callable[..., Any] = PynvvcLumaRgbReader,
) -> AcquisitionCropPoseReport:
    report, selection, keypoints, _crop_meta, resolved_crop_meta, resolved_crop_video = inspect_acquisition_crop_pose_training(
        zarr_path,
        recording_dir=recording_dir,
        crop_meta_path=crop_meta_path,
        crop_video_path=crop_video_path,
        keypoint_parent=keypoint_parent,
        keypoint_run=keypoint_run,
        margin_px=margin_px,
    )
    if not apply:
        return report
    if out_zarr is None:
        raise ValueError("--out-zarr is required with --apply.")
    if selection.source_keypoint_rows.size == 0:
        raise ValueError("No acquisition crop-video pose rows passed the sufficiency gate; refusing empty export.")
    crop_run = crop_run_name or _utc_run_name(DEFAULT_CROP_RUN_PREFIX)
    kp_run = keypoint_export_run_name or _utc_run_name(DEFAULT_KEYPOINT_RUN_PREFIX)
    start = time.perf_counter()
    images = _read_selected_frames(
        resolved_crop_video,
        selection.crop_video_frame_indices,
        reader_factory=reader_factory,
        gpu_id=int(gpu_id),
        require_cuda=require_cuda,
    )
    _write_output_zarr(
        out_zarr=Path(out_zarr),
        overwrite=overwrite,
        images=images,
        selection=selection,
        keypoints=keypoints,
        report=report,
        zarr_path=Path(zarr_path),
        crop_meta_path=resolved_crop_meta,
        crop_video_path=resolved_crop_video,
        crop_run_name=crop_run,
        keypoint_run_name=kp_run,
        gpu_id=int(gpu_id),
    )
    _ = time.perf_counter() - start
    return AcquisitionCropPoseReport(
        zarr_path=report.zarr_path,
        recording_dir=report.recording_dir,
        crop_video=report.crop_video,
        keypoint_parent=report.keypoint_parent,
        keypoint_run=report.keypoint_run,
        source_keypoint_rows=report.source_keypoint_rows,
        usable_keypoint_rows=report.usable_keypoint_rows,
        crop_meta_rows=report.crop_meta_rows,
        selected_rows=report.selected_rows,
        margin_px=report.margin_px,
        reject_counts=report.reject_counts,
        crop_width_stats=report.crop_width_stats,
        crop_height_stats=report.crop_height_stats,
        crop_video_dim_matches_crop_meta=report.crop_video_dim_matches_crop_meta,
        applied=True,
        out_zarr=str(out_zarr),
        crop_run=crop_run,
        keypoint_export_run=kp_run,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Source analysis zarr.")
    parser.add_argument("--out-zarr", type=Path, help="Output crop-video pose training zarr.")
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--crop-meta", type=Path)
    parser.add_argument("--crop-video", type=Path)
    parser.add_argument("--keypoint-parent", default="refined_keypoints_runs")
    parser.add_argument("--keypoint-run")
    parser.add_argument("--margin-px", type=float, default=4.0)
    parser.add_argument("--crop-run-name")
    parser.add_argument("--keypoint-export-run-name")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = export_acquisition_crop_pose_training_zarr(
        args.zarr_path,
        out_zarr=args.out_zarr,
        recording_dir=args.recording_dir,
        crop_meta_path=args.crop_meta,
        crop_video_path=args.crop_video,
        keypoint_parent=args.keypoint_parent,
        keypoint_run=args.keypoint_run,
        margin_px=float(args.margin_px),
        crop_run_name=args.crop_run_name,
        keypoint_export_run_name=args.keypoint_export_run_name,
        gpu_id=int(args.gpu_id),
        overwrite=bool(args.overwrite),
        apply=bool(args.apply),
    )
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"selected_rows: {result.selected_rows}/{result.usable_keypoint_rows} usable keypoint rows")
        print(f"crop_video: {result.crop_video.crop_video_path}")
        print(f"crop_meta: {result.crop_video.crop_meta_path}")
        print(f"keypoints: {result.keypoint_parent}/{result.keypoint_run}")
        print(f"applied: {result.applied}")
        if result.out_zarr:
            print(f"out_zarr: {result.out_zarr}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
