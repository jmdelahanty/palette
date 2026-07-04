#!/usr/bin/env python3
"""Append acquisition crop-video samples to an existing training Zarr."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_geometry import bbox_img_xyxy_to_norm_cxcywh, resolve_full_frame_shape
from fisheye.shared.roi_pixel_contract import (
    APPLIED_RANGE_SEMANTICS_ORANGE_MONO_FULL_RANGE,
    CENTER_ROUNDING_NP_ROUND,
    DECODE_BACKEND_PYNVVC_LUMA,
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.utils.export_acquisition_crop_pose_training_zarr import (
    CropVideoStreamInfo,
    _read_selected_frames,
    inspect_crop_video_stream,
    load_crop_meta_table,
    resolve_crop_video_path,
)
from fisheye.shared.system_metadata import get_environment_info, get_git_info


SCHEMA_ID = "palette.acquisition_crop_video_training_append.v1"
DEFAULT_CROP_RUN_PREFIX = "crop_acquisition_crop_video_training"


@dataclass(frozen=True)
class AcquisitionCropVideoAppendReport:
    training_zarr: str
    recording_dir: str
    crop_video: CropVideoStreamInfo
    source_sample_count: int
    selected_rows: int
    reject_counts: dict[str, int]
    run_name: str | None = None
    applied: bool = False


@dataclass(frozen=True)
class _CropSelection:
    source_training_rows: np.ndarray
    source_frames: np.ndarray
    crop_meta_rows: np.ndarray
    crop_video_frame_indices: np.ndarray
    crop_local_frame_ids: np.ndarray
    source_recording_frame_ids: np.ndarray
    source_crop_xywh: np.ndarray
    realtime_detection_bbox_roi_xyxy: np.ndarray
    bbox_img_xyxy: np.ndarray
    bbox_norm_xywh: np.ndarray
    bbox_crop_norm_xywh: np.ndarray


def _utc_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _recording_dir_from_training_zarr(training_zarr: Path) -> Path:
    raw = zarr.open_group(str(training_zarr), mode="r", use_consolidated=False).get("raw_video")
    if raw is not None:
        recording_dir = raw.attrs.get("recording_dir")
        if recording_dir:
            return Path(str(recording_dir))
    if training_zarr.parent.name == "zarr":
        return training_zarr.parent.parent
    raise ValueError("Could not infer recording_dir from training zarr; pass --recording-dir.")


def _resolve_crop_meta_from_recording(recording_dir: Path, crop_meta_path: Path | None) -> Path:
    if crop_meta_path is not None:
        return Path(crop_meta_path)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    candidates = sorted(crop_dir.glob("*_crop_meta.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise ValueError(f"Multiple crop metadata CSV files found under {crop_dir}; pass --crop-meta.")
    raise ValueError(f"No crop metadata CSV found under {crop_dir}.")


def _load_training_frame_indices(root: zarr.Group) -> np.ndarray:
    raw = root.get("raw_video")
    if raw is None:
        raise ValueError("Training zarr missing raw_video group.")
    if "original_frame_indices" not in raw:
        raise ValueError("Training zarr missing raw_video/original_frame_indices.")
    return np.asarray(raw["original_frame_indices"][:], dtype=np.int64)


def _select_crop_rows(
    root: zarr.Group,
    crop_meta_path: Path,
    *,
    crop_frame_width: int | None = None,
    crop_frame_height: int | None = None,
) -> tuple[_CropSelection, dict[str, int]]:
    source_frames = _load_training_frame_indices(root)
    frame_height, frame_width = resolve_full_frame_shape(root)
    crop_meta = load_crop_meta_table(crop_meta_path)
    crop_pos_by_frame = {int(frame): idx for idx, frame in enumerate(crop_meta.frame_indices.tolist())}
    selected_training_rows: list[int] = []
    selected_crop_rows: list[int] = []
    reject_counts = {
        "missing_crop_meta_frame": 0,
        "blank_crop_frame": 0,
        "crop_has_no_detection": 0,
        "nonfinite_crop_geometry": 0,
    }
    bbox_roi: list[tuple[float, float, float, float]] = []
    bbox_img: list[tuple[float, float, float, float]] = []
    bbox_norm: list[np.ndarray] = []
    bbox_crop_norm: list[np.ndarray] = []

    for train_row, source_frame in enumerate(source_frames.tolist()):
        crop_row = crop_pos_by_frame.get(int(source_frame))
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

        det_x, det_y, det_w, det_h = crop_meta.detection_xywh[crop_row]
        if np.isfinite([det_x, det_y, det_w, det_h]).all() and det_w >= 0.0 and det_h >= 0.0:
            output_w = int(crop_frame_width or round(crop_w))
            output_h = int(crop_frame_height or round(crop_h))
            scale_x = float(output_w) / float(crop_w)
            scale_y = float(output_h) / float(crop_h)
            det_bbox = (
                float(det_x - crop_x) * scale_x,
                float(det_y - crop_y) * scale_y,
                float(det_x + det_w - crop_x) * scale_x,
                float(det_y + det_h - crop_y) * scale_y,
            )
            det_bbox_img = (float(det_x), float(det_y), float(det_x + det_w), float(det_y + det_h))
            norm = bbox_img_xyxy_to_norm_cxcywh(
                np.asarray([det_bbox_img], dtype=np.float64),
                width=int(frame_width),
                height=int(frame_height),
            )[0]
            crop_norm = bbox_img_xyxy_to_norm_cxcywh(
                np.asarray([det_bbox], dtype=np.float64),
                width=int(output_w),
                height=int(output_h),
            )[0]
        else:
            det_bbox = (float("nan"), float("nan"), float("nan"), float("nan"))
            det_bbox_img = (float("nan"), float("nan"), float("nan"), float("nan"))
            norm = np.full((4,), np.nan, dtype=np.float32)
            crop_norm = np.full((4,), np.nan, dtype=np.float32)

        selected_training_rows.append(train_row)
        selected_crop_rows.append(crop_row)
        bbox_roi.append(det_bbox)
        bbox_img.append(det_bbox_img)
        bbox_norm.append(norm)
        bbox_crop_norm.append(crop_norm)

    selected_train = np.asarray(selected_training_rows, dtype=np.int64)
    selected_crop = np.asarray(selected_crop_rows, dtype=np.int64)
    if selected_crop.size:
        bbox_roi_arr = np.asarray(bbox_roi, dtype=np.float32)
        bbox_img_arr = np.asarray(bbox_img, dtype=np.float32)
        bbox_norm_arr = np.asarray(bbox_norm, dtype=np.float32)
        bbox_crop_norm_arr = np.asarray(bbox_crop_norm, dtype=np.float32)
    else:
        bbox_roi_arr = np.empty((0, 4), dtype=np.float32)
        bbox_img_arr = np.empty((0, 4), dtype=np.float32)
        bbox_norm_arr = np.empty((0, 4), dtype=np.float32)
        bbox_crop_norm_arr = np.empty((0, 4), dtype=np.float32)

    selection = _CropSelection(
        source_training_rows=selected_train,
        source_frames=source_frames[selected_train].astype(np.int64, copy=False),
        crop_meta_rows=crop_meta.row_indices[selected_crop].astype(np.int64, copy=False),
        crop_video_frame_indices=crop_meta.video_frame_indices[selected_crop].astype(np.int64, copy=False),
        crop_local_frame_ids=crop_meta.local_frame_ids[selected_crop].astype(np.int64, copy=False),
        source_recording_frame_ids=source_frames[selected_train].astype(np.int64, copy=False) + 1,
        source_crop_xywh=crop_meta.crop_xywh[selected_crop].astype(np.float32, copy=False),
        realtime_detection_bbox_roi_xyxy=bbox_roi_arr,
        bbox_img_xyxy=bbox_img_arr,
        bbox_norm_xywh=bbox_norm_arr,
        bbox_crop_norm_xywh=bbox_crop_norm_arr,
    )
    return selection, reject_counts


def _create_array(group: zarr.Group, name: str, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
    if name in group:
        del group[name]
    group.create_array(name, data=np.asarray(data), chunks=chunks, overwrite=True)


def _write_crop_run(
    root: zarr.Group,
    *,
    run_name: str,
    selection: _CropSelection,
    images: np.ndarray,
    report: AcquisitionCropVideoAppendReport,
    training_zarr: Path,
    recording_dir: Path,
    crop_meta_path: Path,
    crop_video_path: Path,
    gpu_id: int,
    overwrite_run: bool,
) -> None:
    crop_parent = require_runs_parent(root, "crop_runs")
    if run_name in crop_parent:
        if not overwrite_run:
            raise FileExistsError(f"crop_runs/{run_name} already exists in {training_zarr}")
        del crop_parent[run_name]
    crop_group = crop_parent.create_group(run_name)
    mark_run_started(crop_group, run_name=run_name, stage="crop")

    row_count = int(images.shape[0])
    height = int(images.shape[1])
    width = int(images.shape[2])
    frame_height, frame_width = resolve_full_frame_shape(root)
    vector_chunks = (max(1, min(8192, row_count)),)
    bbox_chunks = (max(1, min(8192, row_count)), 4)
    image_chunks = (max(1, min(64, row_count)), height, width)
    max_frame = int(np.max(selection.source_frames)) if row_count else -1
    frame_counts = np.zeros((max_frame + 1,), dtype=np.int32)
    if row_count:
        np.add.at(frame_counts, selection.source_frames.astype(np.int64, copy=False), 1)

    _create_array(crop_group, "roi_images", images.astype(np.uint8, copy=False), chunks=image_chunks)
    _create_array(crop_group, "frame_indices", selection.source_frames.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_frame_indices", selection.source_frames.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_training_row_indices", selection.source_training_rows.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_recording_frame_ids", selection.source_recording_frame_ids.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_meta_row_indices", selection.crop_meta_rows.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_video_frame_indices", selection.crop_video_frame_indices.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_local_frame_ids", selection.crop_local_frame_ids.astype(np.int64), chunks=vector_chunks)
    _create_array(crop_group, "source_crop_xywh", selection.source_crop_xywh.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "roi_coordinates_full", selection.source_crop_xywh[:, :2].astype(np.int32), chunks=(bbox_chunks[0], 2))
    _create_array(crop_group, "bbox_roi_xyxy", selection.realtime_detection_bbox_roi_xyxy.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "bbox_img_xyxy", selection.bbox_img_xyxy.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "bbox_norm_coords", selection.bbox_norm_xywh.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "bbox_crop_norm_coords", selection.bbox_crop_norm_xywh.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "realtime_detection_bbox_roi_xyxy", selection.realtime_detection_bbox_roi_xyxy.astype(np.float32), chunks=bbox_chunks)
    _create_array(crop_group, "detection_indices", np.arange(row_count, dtype=np.int32), chunks=vector_chunks)
    _create_array(crop_group, "detection_success", np.ones(row_count, dtype=bool), chunks=vector_chunks)
    _create_array(crop_group, "detection_source", np.zeros(row_count, dtype=np.int8), chunks=vector_chunks)
    _create_array(crop_group, "frame_counts", frame_counts, chunks=(max(1, min(65536, max(1, frame_counts.shape[0]))),))

    roi_contract = orange_mono_pynvvc_luma_pixel_contract()
    crop_group.attrs.update(
        {
            "schema_id": SCHEMA_ID,
            "crop_storage_mode": "materialized",
            "source_pixels": SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
            "source_pixel_contract": "orange.camera.mono8.full_frame.v1",
            "source_pixel_range": "0_255",
            "source_type": "acquisition_crop_video",
            "detection_source_type": "acquisition_crop_video",
            "training_surface": "acquisition_crop_video_samples",
            "roi_size": [height, width],
            "roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
            "roi_pixel_contract": roi_contract,
            "decode_backend": DECODE_BACKEND_PYNVVC_LUMA,
            "decode_backend_family": "PyNvVideoCodec",
            "decode_contract_status": "canonical_orange_mono_pynvvc_luma",
            "source_decode_surface": "nv12_y_plane_uint8",
            "applied_range_semantics": APPLIED_RANGE_SEMANTICS_ORANGE_MONO_FULL_RANGE,
            "container_color_range_observed": "tv",
            "container_color_range_handling": roi_contract.get("container_color_range_handling"),
            "center_rounding": CENTER_ROUNDING_NP_ROUND,
            "device": f"cuda:{int(gpu_id)}",
            "source_video_path": str(crop_video_path),
            "source_crop_meta_path": str(crop_meta_path),
            "source_training_zarr": str(training_zarr),
            "recording_dir": str(recording_dir),
            "source_crop_xywh_coordinate_space": "source_image_xywh",
            "roi_coordinates_full_coordinate_space": "source_image_xy",
            "roi_coordinates_full_source": "source_crop_xywh[:, :2]",
            "source_crop_video_frame_indices_semantics": "zero_based_frame_index_in_acquisition_crop_video",
            "source_crop_local_frame_ids_semantics": "orange_acquisition_local_frame_id_not_video_frame_index",
            "source_sample_count": int(report.source_sample_count),
            "selected_sample_count": int(report.selected_rows),
            "crop_detection_required": True,
            "blank_crop_frames_excluded": True,
            "rejected_missing_crop_meta_frame": int(report.reject_counts.get("missing_crop_meta_frame", 0)),
            "rejected_blank_crop_frame": int(report.reject_counts.get("blank_crop_frame", 0)),
            "rejected_crop_has_no_detection": int(report.reject_counts.get("crop_has_no_detection", 0)),
            "rejected_nonfinite_crop_geometry": int(report.reject_counts.get("nonfinite_crop_geometry", 0)),
            "bbox_img_xyxy_semantics": "realtime_detection_bbox_xyxy_full_frame_pixels",
            "bbox_roi_xyxy_semantics": "realtime_detection_bbox_xyxy_crop_video_pixels",
            "bbox_norm_coords_semantics": "bbox_xywh_normalized_to_full_frame",
            "bbox_crop_norm_coords_semantics": "realtime_detection_bbox_xywh_normalized_to_crop_video_frame",
            "bbox_norm_reference_width": int(frame_width),
            "bbox_norm_reference_height": int(frame_height),
            "bbox_norm_reference_space": "source_image",
            "bbox_img_xyxy_reference_width": int(frame_width),
            "bbox_img_xyxy_reference_height": int(frame_height),
            "frame_format_confirmation_status": "pending_orange_confirmation",
            "summary": asdict(report),
        }
    )
    git_info = get_git_info(Path(__file__).resolve().parents[3])
    env_info = get_environment_info(include_all_packages=False, disk_path=str(training_zarr), collect_ip=False)
    provenance = build_stage_provenance(
        stage="crop",
        command=" ".join(sys.argv),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git=git_info,
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        parameters={"gpu_id": int(gpu_id), "run_name": run_name},
        inputs={
            "training_zarr": str(training_zarr),
            "recording_dir": str(recording_dir),
            "crop_meta": str(crop_meta_path),
            "crop_video": str(crop_video_path),
            "sampling_source": "raw_video/original_frame_indices",
        },
        artifacts={"run_name": run_name, "row_count": row_count},
    )
    write_stage_provenance(crop_group, provenance)
    mark_run_complete(crop_group, parent_group=crop_parent, run_name=run_name)


def append_acquisition_crop_video_training(
    training_zarr: Path,
    *,
    recording_dir: Path | None = None,
    crop_meta_path: Path | None = None,
    crop_video_path: Path | None = None,
    run_name: str | None = None,
    gpu_id: int = 0,
    overwrite_run: bool = False,
    apply: bool = False,
    require_cuda: bool = True,
    reader_factory: Callable[..., Any] | None = None,
) -> AcquisitionCropVideoAppendReport:
    training_zarr = Path(training_zarr)
    resolved_recording_dir = Path(recording_dir) if recording_dir is not None else _recording_dir_from_training_zarr(training_zarr)
    resolved_crop_meta = _resolve_crop_meta_from_recording(resolved_recording_dir, crop_meta_path)
    resolved_crop_video = Path(crop_video_path) if crop_video_path is not None else resolve_crop_video_path(resolved_recording_dir)
    crop_info = inspect_crop_video_stream(resolved_recording_dir, resolved_crop_meta, resolved_crop_video)

    root = zarr.open_group(str(training_zarr), mode="a", use_consolidated=False)
    selection, reject_counts = _select_crop_rows(
        root,
        resolved_crop_meta,
        crop_frame_width=crop_info.width,
        crop_frame_height=crop_info.height,
    )
    source_sample_count = int(_load_training_frame_indices(root).shape[0])
    report = AcquisitionCropVideoAppendReport(
        training_zarr=str(training_zarr),
        recording_dir=str(resolved_recording_dir),
        crop_video=crop_info,
        source_sample_count=source_sample_count,
        selected_rows=int(selection.source_frames.shape[0]),
        reject_counts=reject_counts,
        run_name=run_name,
        applied=False,
    )
    if not apply:
        return report
    if selection.source_frames.size == 0:
        raise ValueError("No sampled training frames have usable acquisition crop-video rows; refusing empty crop run.")
    resolved_run_name = run_name or _utc_run_name(DEFAULT_CROP_RUN_PREFIX)
    images = _read_selected_frames(
        resolved_crop_video,
        selection.crop_video_frame_indices,
        reader_factory=reader_factory or PynvvcLumaRgbReader,
        gpu_id=int(gpu_id),
        require_cuda=require_cuda,
    )
    applied_report = AcquisitionCropVideoAppendReport(
        training_zarr=report.training_zarr,
        recording_dir=report.recording_dir,
        crop_video=report.crop_video,
        source_sample_count=report.source_sample_count,
        selected_rows=report.selected_rows,
        reject_counts=report.reject_counts,
        run_name=resolved_run_name,
        applied=True,
    )
    _write_crop_run(
        root,
        run_name=resolved_run_name,
        selection=selection,
        images=images,
        report=applied_report,
        training_zarr=training_zarr,
        recording_dir=resolved_recording_dir,
        crop_meta_path=resolved_crop_meta,
        crop_video_path=resolved_crop_video,
        gpu_id=int(gpu_id),
        overwrite_run=overwrite_run,
    )
    return applied_report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_zarr", type=Path)
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--crop-meta", type=Path)
    parser.add_argument("--crop-video", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = append_acquisition_crop_video_training(
        args.training_zarr,
        recording_dir=args.recording_dir,
        crop_meta_path=args.crop_meta,
        crop_video_path=args.crop_video,
        run_name=args.run_name,
        gpu_id=int(args.gpu_id),
        overwrite_run=bool(args.overwrite_run),
        apply=bool(args.apply),
    )
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"training_zarr: {result.training_zarr}")
        print(f"crop_video: {result.crop_video.crop_video_path}")
        print(f"source_sample_count: {result.source_sample_count}")
        print(f"selected_rows: {result.selected_rows}")
        print(f"reject_counts: {result.reject_counts}")
        print(f"applied: {result.applied}")
        if result.run_name:
            print(f"run_name: {result.run_name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
