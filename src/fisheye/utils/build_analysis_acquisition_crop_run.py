"""Build geometry-only analysis crop runs from Orange acquisition crop metadata."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.compare_realtime_offline_detections import infer_recording_dir_from_zarr
from fisheye.shared.crop_geometry import bbox_img_xyxy_to_norm_cxcywh
from fisheye.shared.roi_pixel_contract import (
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.json_safety import write_jsonl_atomic
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array, stamp_geometry_preload_attrs
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_failed, mark_run_started, require_runs_parent
from fisheye.utils.import_acquisition_detections_to_detect_run import resolve_source_dimensions
from fisheye.shared.system_metadata import get_environment_info, get_git_info


SCHEMA_ID = "palette.analysis_acquisition_crop_run.v1"
DEFAULT_RUN_PREFIX = "crop_acquisition_crop_video_analysis"
SOURCE_PIXEL_KIND_CODE_MAP = {
    "acquisition_crop_video": 0,
}
CROP_STATE_CODE_MAP = {
    "detected_crop": 0,
}


@dataclass(frozen=True)
class BuildAnalysisAcquisitionCropRunResult:
    zarr_path: str
    recording_dir: str
    crop_meta_path: str
    crop_video_path: str | None
    run_name: str
    run_path: str
    total_crop_meta_rows: int
    selected_rows: int
    rejected_blank_crop_frame: int
    rejected_crop_has_no_detection: int
    rejected_nonfinite_crop_geometry: int
    rejected_nonfinite_detection_geometry: int
    total_frames: int
    frames_with_crops: int
    source_width: int
    source_height: int
    applied: bool
    skipped: bool = False
    skip_reason: str | None = None


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
class _CropRunPayload:
    frame_indices: np.ndarray
    source_recording_frame_ids: np.ndarray
    source_crop_meta_row_indices: np.ndarray
    source_crop_video_frame_indices: np.ndarray
    source_crop_local_frame_ids: np.ndarray
    source_crop_xywh: np.ndarray
    roi_coordinates_full: np.ndarray
    roi_sizes_full: np.ndarray
    selected_live_detection_bbox_img_xyxy: np.ndarray
    selected_live_detection_bbox_roi_xyxy: np.ndarray
    selected_live_detection_bbox_norm_coords: np.ndarray
    selected_live_detection_bbox_crop_norm_coords: np.ndarray
    frame_counts: np.ndarray
    detection_indices: np.ndarray
    detection_success: np.ndarray
    detection_source: np.ndarray
    source_pixel_kind_codes: np.ndarray
    crop_state_codes: np.ndarray
    reject_counts: dict[str, int]
    total_crop_meta_rows: int


def _utc_run_name(prefix: str = DEFAULT_RUN_PREFIX) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value)))
    except Exception:
        return int(default)


def _safe_float(value: object) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_stream(recording_dir: Path, stream_name: str) -> dict[str, Any]:
    manifest = _load_json(recording_dir / "recording_manifest.json")
    streams = (manifest.get("video_streams") or {}).get("streams") if isinstance(manifest.get("video_streams"), dict) else {}
    stream = streams.get(stream_name) if isinstance(streams, dict) else None
    return stream if isinstance(stream, dict) else {}


def _resolve_relative(recording_dir: Path, value: object) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = recording_dir / path
    return path


def _first_existing_or_first(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def resolve_crop_meta_path(recording_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return Path(explicit)
    stream = _manifest_stream(recording_dir, "crop")
    candidates: list[Path] = []
    for key in ("metadata", "metadata_path", "crop_meta", "crop_meta_path"):
        path = _resolve_relative(recording_dir, stream.get(key))
        if path is not None:
            candidates.append(path)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    candidates.extend(sorted(crop_dir.glob("*_crop_meta.csv")))
    resolved = _first_existing_or_first(list(dict.fromkeys(candidates)))
    if resolved is None:
        raise ValueError(f"No acquisition crop metadata CSV found for {recording_dir}.")
    return resolved


def resolve_crop_video_path(recording_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    stream = _manifest_stream(recording_dir, "crop")
    candidates: list[Path] = []
    for key in ("video", "video_path", "path"):
        path = _resolve_relative(recording_dir, stream.get(key))
        if path is not None:
            candidates.append(path)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    candidates.extend(sorted(crop_dir.glob("*_crop_external.mp4")))
    candidates.extend(sorted(crop_dir.glob("*.mp4")))
    return _first_existing_or_first(list(dict.fromkeys(candidates)))


def load_crop_meta_table(crop_meta_path: Path) -> _CropMetaTable:
    frame_indices: list[int] = []
    video_frame_indices: list[int] = []
    local_frame_ids: list[int] = []
    row_indices: list[int] = []
    has_detection: list[bool] = []
    blank_frame: list[bool] = []
    crop_xywh: list[tuple[float, float, float, float]] = []
    detection_xywh: list[tuple[float, float, float, float]] = []

    with Path(crop_meta_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"recording_frame_id", "crop_x", "crop_y", "crop_w", "crop_h"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Crop metadata missing required columns: {missing}")
        for row_index, row in enumerate(reader):
            frame = _safe_int(row.get("recording_frame_id"), default=0) - 1
            frame_indices.append(frame)
            video_frame_indices.append(_safe_int(row.get("crop_video_frame_index"), default=row_index))
            local_frame_ids.append(_safe_int(row.get("local_frame_id"), default=row_index))
            row_indices.append(row_index)
            has_detection.append(bool(_safe_int(row.get("has_detection"), default=0)))
            blank_frame.append(bool(_safe_int(row.get("blank_frame"), default=0)))
            crop_xywh.append(
                (
                    _safe_float(row.get("crop_x")),
                    _safe_float(row.get("crop_y")),
                    _safe_float(row.get("crop_w")),
                    _safe_float(row.get("crop_h")),
                )
            )
            detection_xywh.append(
                (
                    _safe_float(row.get("detection_x")),
                    _safe_float(row.get("detection_y")),
                    _safe_float(row.get("detection_w")),
                    _safe_float(row.get("detection_h")),
                )
            )

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


def _resolve_total_frames(root: zarr.Group, frame_indices: np.ndarray) -> int:
    for attrs in (root.attrs, getattr(root.get("raw_video"), "attrs", {})):
        for key in ("total_frames", "n_frames", "source_video_total_frames"):
            value = attrs.get(key) if attrs is not None else None
            parsed = _safe_int(value, default=0)
            if parsed > 0:
                return max(parsed, int(np.max(frame_indices)) + 1 if frame_indices.size else 0)
    return int(np.max(frame_indices)) + 1 if frame_indices.size else 0


def _build_payload(
    *,
    crop_meta: _CropMetaTable,
    source_width: int,
    source_height: int,
    total_frames: int,
) -> _CropRunPayload:
    reject_counts = {
        "blank_crop_frame": 0,
        "crop_has_no_detection": 0,
        "nonfinite_crop_geometry": 0,
        "nonfinite_detection_geometry": 0,
    }
    selected: list[int] = []
    bbox_img: list[tuple[float, float, float, float]] = []
    bbox_roi: list[tuple[float, float, float, float]] = []

    for idx in range(crop_meta.frame_indices.shape[0]):
        if bool(crop_meta.blank_frame[idx]):
            reject_counts["blank_crop_frame"] += 1
            continue
        if not bool(crop_meta.has_detection[idx]):
            reject_counts["crop_has_no_detection"] += 1
            continue
        crop_x, crop_y, crop_w, crop_h = crop_meta.crop_xywh[idx]
        if not np.isfinite([crop_x, crop_y, crop_w, crop_h]).all() or crop_w <= 0.0 or crop_h <= 0.0:
            reject_counts["nonfinite_crop_geometry"] += 1
            continue
        det_x, det_y, det_w, det_h = crop_meta.detection_xywh[idx]
        if not np.isfinite([det_x, det_y, det_w, det_h]).all() or det_w < 0.0 or det_h < 0.0:
            reject_counts["nonfinite_detection_geometry"] += 1
            continue
        selected.append(idx)
        bbox_img.append((float(det_x), float(det_y), float(det_x + det_w), float(det_y + det_h)))
        bbox_roi.append(
            (
                float(det_x - crop_x),
                float(det_y - crop_y),
                float(det_x + det_w - crop_x),
                float(det_y + det_h - crop_y),
            )
        )

    selected_idx = np.asarray(selected, dtype=np.int64)
    n = int(selected_idx.shape[0])
    source_crop_xywh = crop_meta.crop_xywh[selected_idx].astype(np.float32, copy=False) if n else np.empty((0, 4), dtype=np.float32)
    bbox_img_arr = np.asarray(bbox_img, dtype=np.float32).reshape(-1, 4) if n else np.empty((0, 4), dtype=np.float32)
    bbox_roi_arr = np.asarray(bbox_roi, dtype=np.float32).reshape(-1, 4) if n else np.empty((0, 4), dtype=np.float32)
    bbox_norm = bbox_img_xyxy_to_norm_cxcywh(
        bbox_img_arr.astype(np.float64, copy=False),
        width=int(source_width),
        height=int(source_height),
    ).astype(np.float32, copy=False)
    if n:
        crop_w = source_crop_xywh[:, 2].astype(np.float64, copy=False)
        crop_h = source_crop_xywh[:, 3].astype(np.float64, copy=False)
        bbox_crop_norm = np.column_stack(
            [
                ((bbox_roi_arr[:, 0] + bbox_roi_arr[:, 2]) * 0.5) / crop_w,
                ((bbox_roi_arr[:, 1] + bbox_roi_arr[:, 3]) * 0.5) / crop_h,
                (bbox_roi_arr[:, 2] - bbox_roi_arr[:, 0]) / crop_w,
                (bbox_roi_arr[:, 3] - bbox_roi_arr[:, 1]) / crop_h,
            ]
        ).astype(np.float32, copy=False)
    else:
        bbox_crop_norm = np.empty((0, 4), dtype=np.float32)

    frame_indices = crop_meta.frame_indices[selected_idx].astype(np.int32, copy=False) if n else np.empty((0,), dtype=np.int32)
    frame_counts = np.bincount(frame_indices.astype(np.int64, copy=False), minlength=int(total_frames)).astype(np.int32)
    return _CropRunPayload(
        frame_indices=frame_indices,
        source_recording_frame_ids=frame_indices.astype(np.int64, copy=False) + 1,
        source_crop_meta_row_indices=crop_meta.row_indices[selected_idx].astype(np.int64, copy=False)
        if n
        else np.empty((0,), dtype=np.int64),
        source_crop_video_frame_indices=crop_meta.video_frame_indices[selected_idx].astype(np.int64, copy=False)
        if n
        else np.empty((0,), dtype=np.int64),
        source_crop_local_frame_ids=crop_meta.local_frame_ids[selected_idx].astype(np.int64, copy=False)
        if n
        else np.empty((0,), dtype=np.int64),
        source_crop_xywh=source_crop_xywh,
        roi_coordinates_full=np.rint(source_crop_xywh[:, :2]).astype(np.int32, copy=False) if n else np.empty((0, 2), dtype=np.int32),
        roi_sizes_full=np.rint(source_crop_xywh[:, 2:4]).astype(np.int32, copy=False) if n else np.empty((0, 2), dtype=np.int32),
        selected_live_detection_bbox_img_xyxy=bbox_img_arr,
        selected_live_detection_bbox_roi_xyxy=bbox_roi_arr,
        selected_live_detection_bbox_norm_coords=bbox_norm,
        selected_live_detection_bbox_crop_norm_coords=bbox_crop_norm,
        frame_counts=frame_counts,
        detection_indices=np.arange(n, dtype=np.int32),
        detection_success=np.ones(n, dtype=bool),
        detection_source=np.zeros(n, dtype=np.int8),
        source_pixel_kind_codes=np.full(n, SOURCE_PIXEL_KIND_CODE_MAP["acquisition_crop_video"], dtype=np.int8),
        crop_state_codes=np.full(n, CROP_STATE_CODE_MAP["detected_crop"], dtype=np.int8),
        reject_counts=reject_counts,
        total_crop_meta_rows=int(crop_meta.frame_indices.shape[0]),
    )


def _create_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    create_geometry_preload_array(group, name, data=np.asarray(data), overwrite=True)


def _set_or_clear_attr(group: zarr.Group, name: str, value: Optional[str]) -> None:
    if value is None:
        if name in group.attrs:
            del group.attrs[name]
        return
    group.attrs[name] = value


def _finalize_geometry_only_crop_parent(crop_parent: zarr.Group, *, run_name: str) -> None:
    previous_materialized = crop_parent.attrs.get("latest_materialized")
    previous_latest = crop_parent.attrs.get("latest")
    materialized = None
    for candidate in (previous_materialized, previous_latest):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text and text != run_name and text in crop_parent:
            attrs = crop_parent[text].attrs
            mode = str(attrs.get("crop_storage_mode") or ("materialized" if "roi_images" in crop_parent[text] else "geometry_only"))
            if mode == "materialized":
                materialized = text
                break
    _set_or_clear_attr(crop_parent, "latest_materialized", materialized)
    _set_or_clear_attr(crop_parent, "latest", materialized)
    crop_parent.attrs["latest_any"] = run_name


def _write_crop_run(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_dir: Path,
    crop_meta_path: Path,
    crop_video_path: Optional[Path],
    run_name: str,
    payload: _CropRunPayload,
    source_width: int,
    source_height: int,
    overwrite: bool,
) -> None:
    parent = require_runs_parent(root, "crop_runs")
    if run_name in parent:
        if not overwrite:
            raise ValueError(f"crop_runs/{run_name} already exists in {zarr_path}")
        del parent[run_name]
    group = parent.create_group(run_name)
    mark_run_started(group, run_name=run_name, stage="crop")
    try:
        _create_array(group, "frame_indices", payload.frame_indices)
        _create_array(group, "source_frame_indices", payload.frame_indices.astype(np.int64, copy=False))
        _create_array(group, "source_recording_frame_ids", payload.source_recording_frame_ids)
        _create_array(group, "source_crop_meta_row_indices", payload.source_crop_meta_row_indices)
        _create_array(group, "source_crop_video_frame_indices", payload.source_crop_video_frame_indices)
        _create_array(group, "source_crop_local_frame_ids", payload.source_crop_local_frame_ids)
        _create_array(group, "source_crop_xywh", payload.source_crop_xywh)
        _create_array(group, "roi_coordinates_full", payload.roi_coordinates_full)
        _create_array(group, "roi_sizes_full", payload.roi_sizes_full)
        _create_array(group, "bbox_img_xyxy", payload.selected_live_detection_bbox_img_xyxy)
        _create_array(group, "bbox_norm_coords", payload.selected_live_detection_bbox_norm_coords)
        _create_array(group, "bbox_roi_xyxy", payload.selected_live_detection_bbox_roi_xyxy)
        _create_array(group, "bbox_crop_norm_coords", payload.selected_live_detection_bbox_crop_norm_coords)
        _create_array(group, "selected_live_detection_bbox_img_xyxy", payload.selected_live_detection_bbox_img_xyxy)
        _create_array(group, "selected_live_detection_bbox_norm_coords", payload.selected_live_detection_bbox_norm_coords)
        _create_array(group, "selected_live_detection_bbox_roi_xyxy", payload.selected_live_detection_bbox_roi_xyxy)
        _create_array(group, "selected_live_detection_bbox_crop_norm_coords", payload.selected_live_detection_bbox_crop_norm_coords)
        _create_array(group, "realtime_detection_bbox_roi_xyxy", payload.selected_live_detection_bbox_roi_xyxy)
        _create_array(group, "frame_counts", payload.frame_counts)
        _create_array(group, "detection_indices", payload.detection_indices)
        _create_array(group, "detection_success", payload.detection_success)
        _create_array(group, "detection_source", payload.detection_source)
        _create_array(group, "source_pixel_kind_codes", payload.source_pixel_kind_codes)
        _create_array(group, "crop_state_codes", payload.crop_state_codes)
        stamp_geometry_preload_attrs(group)

        now = datetime.now(timezone.utc).isoformat()
        roi_size: list[int] | None = None
        if payload.roi_sizes_full.shape[0]:
            unique_roi_sizes = np.unique(payload.roi_sizes_full.astype(np.int64, copy=False), axis=0)
            if unique_roi_sizes.shape[0] != 1:
                preview = unique_roi_sizes[:5].tolist()
                raise ValueError(
                    "Acquisition crop-video analysis crop runs require fixed crop-video frame size; "
                    f"found multiple width,height values: {preview}"
                )
            roi_w, roi_h = int(unique_roi_sizes[0, 0]), int(unique_roi_sizes[0, 1])
            roi_size = [roi_h, roi_w]
        summary = {
            "total_crop_meta_rows": int(payload.total_crop_meta_rows),
            "selected_rows": int(payload.frame_indices.shape[0]),
            "frames_with_crops": int(np.count_nonzero(payload.frame_counts)),
            **{f"rejected_{key}": int(value) for key, value in payload.reject_counts.items()},
        }
        group.attrs.update(
            {
                "schema_id": SCHEMA_ID,
                "crop_storage_mode": "geometry_only",
                "source_pixels": "acquisition_crop_video",
                "roi_pixel_provider": "acquisition_crop_video",
                "source_type": "acquisition_crop_video",
                "roi_size": roi_size,
                "roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
                "roi_pixel_contract": orange_mono_pynvvc_luma_pixel_contract(),
                "decode_backend": "pynvvc_luma",
                "detection_source_type": "external_crop_recorder_crop_meta_selected_live_detection",
                "source_video_path": str(crop_video_path) if crop_video_path is not None else None,
                "source_crop_video_path": str(crop_video_path) if crop_video_path is not None else None,
                "source_crop_meta_path": str(crop_meta_path),
                "source_recording_dir": str(recording_dir),
                "source_crop_xywh_coordinate_space": "source_image_xywh",
                "roi_coordinates_full_coordinate_space": "source_image_xy",
                "roi_coordinates_full_source": "source_crop_xywh[:, :2]",
                "source_crop_video_frame_indices_semantics": "zero_based_frame_index_in_acquisition_crop_video",
                "source_crop_local_frame_ids_semantics": "orange_acquisition_local_frame_id_not_video_frame_index",
                "crop_state_code_map": CROP_STATE_CODE_MAP,
                "source_pixel_kind_code_map": SOURCE_PIXEL_KIND_CODE_MAP,
                "crop_state_policy": "detected_crop_rows_only",
                "blank_crop_frames_excluded": True,
                "crop_detection_required": True,
                "bbox_img_xyxy_semantics": "selected_live_detection_bbox_xyxy_full_frame_pixels",
                "bbox_norm_coords_semantics": "bbox_xywh_normalized_to_full_frame",
                "bbox_roi_xyxy_semantics": "selected_live_detection_bbox_xyxy_crop_video_pixels",
                "bbox_crop_norm_coords_semantics": "selected_live_detection_bbox_xywh_normalized_to_crop_video_frame",
                "selected_live_detection_bbox_semantics": "selected_postprocessed_model_detection_used_to_center_crop",
                "source_video_width": int(source_width),
                "source_video_height": int(source_height),
                "bbox_norm_reference_width": int(source_width),
                "bbox_norm_reference_height": int(source_height),
                "bbox_norm_reference_space": "source_image",
                "created_at_utc": now,
                "summary_statistics": summary,
                "status": "completed",
                "completed_at_utc": now,
            }
        )
        git_info = get_git_info(Path(__file__).resolve().parents[3])
        env_info = get_environment_info(include_all_packages=False, disk_path=str(zarr_path), collect_ip=False)
        provenance = build_stage_provenance(
            stage="crop",
            command=" ".join(sys.argv),
            created_at_utc=now,
            version=git_info.get("short_hash") or git_info.get("commit_hash"),
            git=git_info,
            environment=env_info.get("environment"),
            platform=env_info.get("platform"),
            parameters={"run_name": run_name, "crop_storage_mode": "geometry_only"},
            inputs={
                "zarr_path": str(zarr_path),
                "recording_dir": str(recording_dir),
                "crop_meta_path": str(crop_meta_path),
                "crop_video_path": str(crop_video_path) if crop_video_path is not None else None,
            },
            artifacts={"run_path": f"crop_runs/{run_name}", "selected_rows": int(payload.frame_indices.shape[0])},
        )
        write_stage_provenance(group, provenance)
        mark_run_complete(group, parent_group=parent, run_name=run_name)
        _finalize_geometry_only_crop_parent(parent, run_name=run_name)
    except Exception as exc:
        mark_run_failed(group, error=str(exc))
        raise


def build_analysis_acquisition_crop_run(
    zarr_path: Path,
    *,
    recording_dir: Optional[Path] = None,
    crop_meta_path: Optional[Path] = None,
    crop_video_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    overwrite: bool = False,
    apply: bool = False,
) -> BuildAnalysisAcquisitionCropRunResult:
    zarr_path = Path(zarr_path)
    resolved_recording_dir = Path(recording_dir) if recording_dir is not None else infer_recording_dir_from_zarr(zarr_path)
    resolved_crop_meta = resolve_crop_meta_path(resolved_recording_dir, crop_meta_path)
    resolved_crop_video = resolve_crop_video_path(resolved_recording_dir, crop_video_path)
    if resolved_crop_video is None or not Path(resolved_crop_video).exists():
        raise ValueError(f"No acquisition crop video found for {resolved_recording_dir}.")
    root = zarr.open_group(str(zarr_path), mode="a" if apply else "r", use_consolidated=False)
    width, height = resolve_source_dimensions(
        root,
        recording_dir=resolved_recording_dir,
        source_width=source_width,
        source_height=source_height,
    )
    crop_meta = load_crop_meta_table(resolved_crop_meta)
    total_frames = _resolve_total_frames(root, crop_meta.frame_indices)
    payload = _build_payload(crop_meta=crop_meta, source_width=width, source_height=height, total_frames=total_frames)
    resolved_run_name = run_name or _utc_run_name()
    run_path = f"crop_runs/{resolved_run_name}"
    frames_with_crops = int(np.count_nonzero(payload.frame_counts))

    if apply:
        if payload.frame_indices.size == 0:
            raise ValueError("No usable acquisition crop rows; refusing to write an empty analysis crop run.")
        _write_crop_run(
            root,
            zarr_path=zarr_path,
            recording_dir=resolved_recording_dir,
            crop_meta_path=resolved_crop_meta,
            crop_video_path=resolved_crop_video,
            run_name=resolved_run_name,
            payload=payload,
            source_width=width,
            source_height=height,
            overwrite=overwrite,
        )

    return BuildAnalysisAcquisitionCropRunResult(
        zarr_path=str(zarr_path),
        recording_dir=str(resolved_recording_dir),
        crop_meta_path=str(resolved_crop_meta),
        crop_video_path=str(resolved_crop_video) if resolved_crop_video is not None else None,
        run_name=resolved_run_name,
        run_path=run_path,
        total_crop_meta_rows=int(payload.total_crop_meta_rows),
        selected_rows=int(payload.frame_indices.shape[0]),
        rejected_blank_crop_frame=int(payload.reject_counts["blank_crop_frame"]),
        rejected_crop_has_no_detection=int(payload.reject_counts["crop_has_no_detection"]),
        rejected_nonfinite_crop_geometry=int(payload.reject_counts["nonfinite_crop_geometry"]),
        rejected_nonfinite_detection_geometry=int(payload.reject_counts["nonfinite_detection_geometry"]),
        total_frames=int(total_frames),
        frames_with_crops=frames_with_crops,
        source_width=int(width),
        source_height=int(height),
        applied=bool(apply),
    )


def _registry_rows(
    registry: Path,
    *,
    path_contains: Optional[str],
    recording_contains: Optional[str],
    limit: Optional[int],
) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(registry))
    conn.row_factory = sqlite3.Row
    try:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        if not columns:
            raise RuntimeError(f"{registry} has no datasets table")
        fields = ["dataset_id", "recording_id", "zarr_path", "zarr_use", "status"]
        select_cols = [field if field in columns else f"NULL AS {field}" for field in fields]
        clauses = []
        params: list[object] = []
        if "status" in columns:
            clauses.append("status = 'active'")
        if "zarr_use" in columns:
            clauses.append("zarr_use = 'analysis'")
        if path_contains:
            clauses.append("COALESCE(zarr_path, '') LIKE ?")
            params.append(f"%{path_contains}%")
        if recording_contains:
            clauses.append("COALESCE(recording_id, '') LIKE ?")
            params.append(f"%{recording_contains}%")
        sql = f"SELECT {', '.join(select_cols)} FROM datasets"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY zarr_path"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def _batch_from_registry(
    *,
    registry: Path,
    path_contains: Optional[str],
    recording_contains: Optional[str],
    limit: Optional[int],
    run_name: Optional[str],
    run_name_prefix: str,
    overwrite: bool,
    apply: bool,
) -> list[BuildAnalysisAcquisitionCropRunResult]:
    rows = _registry_rows(
        registry,
        path_contains=path_contains,
        recording_contains=recording_contains,
        limit=limit,
    )
    batch_run_name = run_name or _utc_run_name(run_name_prefix)
    results: list[BuildAnalysisAcquisitionCropRunResult] = []
    for row in rows:
        zarr_value = str(row.get("zarr_path") or "").strip()
        if not zarr_value:
            continue
        try:
            result = build_analysis_acquisition_crop_run(
                Path(zarr_value),
                run_name=batch_run_name,
                overwrite=overwrite,
                apply=apply,
            )
        except Exception as exc:
            results.append(
                BuildAnalysisAcquisitionCropRunResult(
                    zarr_path=zarr_value,
                    recording_dir="",
                    crop_meta_path="",
                    crop_video_path=None,
                    run_name=batch_run_name,
                    run_path=f"crop_runs/{batch_run_name}",
                    total_crop_meta_rows=0,
                    selected_rows=0,
                    rejected_blank_crop_frame=0,
                    rejected_crop_has_no_detection=0,
                    rejected_nonfinite_crop_geometry=0,
                    rejected_nonfinite_detection_geometry=0,
                    total_frames=0,
                    frames_with_crops=0,
                    source_width=0,
                    source_height=0,
                    applied=False,
                    skipped=True,
                    skip_reason=str(exc),
                )
            )
            if apply:
                raise
        else:
            results.append(result)
    return results


def _write_acquisition_crop_results_jsonl(path: Path, rows: Sequence[BuildAnalysisAcquisitionCropRunResult]) -> None:
    write_jsonl_atomic(Path(path), [asdict(row) for row in rows])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", nargs="?", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--source", choices=["zarr", "registry"], default="zarr")
    parser.add_argument("--registry", type=Path, help="Registry path for --source registry.")
    parser.add_argument("--path-contains", help="Registry zarr_path substring filter.")
    parser.add_argument("--recording-contains", help="Registry recording_id substring filter.")
    parser.add_argument("--limit", type=int, help="Limit registry rows.")
    parser.add_argument("--recording-dir", type=Path, help="Recording root used to resolve sidecars.")
    parser.add_argument("--crop-meta", type=Path, help="Explicit external crop-recorder *_crop_meta.csv path.")
    parser.add_argument("--crop-video", type=Path, help="Explicit external crop-recorder crop MP4 path.")
    parser.add_argument("--run-name", help="Crop run name. In registry mode, reused for every selected zarr.")
    parser.add_argument("--run-name-prefix", default=DEFAULT_RUN_PREFIX, help="Prefix for generated crop run names.")
    parser.add_argument("--source-width", type=int, help="Full-frame source width.")
    parser.add_argument("--source-height", type=int, help="Full-frame source height.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop run with the same name.")
    parser.add_argument("--apply", action="store_true", help="Write crop_runs/<run>; otherwise dry-run only.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--output-json", type=Path, help="Write result JSON.")
    parser.add_argument("--output-jsonl", type=Path, help="Write registry results as JSONL.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.source == "registry":
        if args.registry is None:
            raise SystemExit("--registry is required with --source registry")
        results = _batch_from_registry(
            registry=args.registry,
            path_contains=args.path_contains,
            recording_contains=args.recording_contains,
            limit=args.limit,
            run_name=args.run_name,
            run_name_prefix=args.run_name_prefix,
            overwrite=bool(args.overwrite),
            apply=bool(args.apply),
        )
        payload: Any = [asdict(result) for result in results]
        if args.output_jsonl:
            _write_acquisition_crop_results_jsonl(args.output_jsonl, results)
        if args.output_json:
            args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"records: {len(results)}")
            print(f"applied: {bool(args.apply)}")
            print(f"selected_rows_total: {sum(result.selected_rows for result in results)}")
            skipped = [result for result in results if result.skipped]
            if skipped:
                print(f"skipped_or_failed: {len(skipped)}")
                for result in skipped[:10]:
                    print(f"  {result.zarr_path}: {result.skip_reason}")
            if not args.apply:
                print("dry_run: pass --apply to write crop_runs/<run>")
        return 0

    if args.zarr_path is None:
        raise SystemExit("zarr_path is required unless --source registry is used")
    result = build_analysis_acquisition_crop_run(
        args.zarr_path,
        recording_dir=args.recording_dir,
        crop_meta_path=args.crop_meta,
        crop_video_path=args.crop_video,
        run_name=args.run_name,
        source_width=args.source_width,
        source_height=args.source_height,
        overwrite=bool(args.overwrite),
        apply=bool(args.apply),
    )
    payload = asdict(result)
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"run_path: {result.run_path}")
        print(f"crop_meta: {result.crop_meta_path}")
        print(f"crop_video: {result.crop_video_path}")
        print(f"selected_rows: {result.selected_rows}/{result.total_crop_meta_rows}")
        print(
            "rejected: "
            f"blank={result.rejected_blank_crop_frame} "
            f"no_detection={result.rejected_crop_has_no_detection} "
            f"bad_crop={result.rejected_nonfinite_crop_geometry} "
            f"bad_detection={result.rejected_nonfinite_detection_geometry}"
        )
        print(f"source_shape: {result.source_width}x{result.source_height}")
        if not result.applied:
            print("dry_run: pass --apply to write crop_runs/<run>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
