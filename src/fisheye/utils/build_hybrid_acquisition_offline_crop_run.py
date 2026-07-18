"""Build hybrid crop runs backed by acquisition crop video plus offline ROI cache.

The output crop run is geometry-only. Rows from Orange's acquisition crop video
read directly from the crop MP4; frames missed online but recovered by offline
refined detections read from a supplemental flat ROI cache pre-decoded from the
full camera video.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.composite_crop import assert_crop_run_unreferenced
from fisheye.shared.crop_geometry import (
    bbox_img_xyxy_to_norm_cxcywh,
    bbox_norm_cxcywh_to_img_xyxy,
    compute_centered_roi_mapping,
    resolve_full_frame_shape,
)
from fisheye.shared.flat_roi_cache import FLAT_ROI_CACHE_LAYOUT, FLAT_ROI_CACHE_SCHEMA
from fisheye.shared.flat_roi_cache import _crop_pynvvc_luma_frame  # noqa: PLC2701
from fisheye.shared.refined_detect_curation import (
    extract_present_curated_rows,
    resolve_curated_refined_detect_run,
)
from fisheye.shared.roi_pixel_contract import (
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    flat_cache_pixel_contract_for_backend,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array, stamp_geometry_preload_attrs
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_failed, mark_run_started
from fisheye.shared.system_metadata import get_environment_info, get_git_info


SCHEMA_ID = "palette.hybrid_acquisition_offline_crop_run.v1"
DEFAULT_RUN_PREFIX = "crop_hybrid_acquisition_offline"
DECODE_MODES = ("auto", "indexed", "sequential")
SOURCE_PIXEL_KIND_CODE_MAP = {
    "acquisition_crop_video": 0,
    "offline_full_frame_supplemental_flat_cache": 1,
}
CROP_STATE_CODE_MAP = {
    "detected_crop": 0,
    "offline_recovered_crop": 1,
}
DETECTION_SOURCE_CODE_MAP = {
    "acquisition_live_detection": 0,
    "offline_refined_detection": 1,
}


def _utc_run_name(prefix: str = DEFAULT_RUN_PREFIX) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _create_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    create_geometry_preload_array(group, name, data=np.asarray(data), overwrite=True)


def _resolve_crop_run(root: zarr.Group, crop_run: str | None) -> tuple[zarr.Group, zarr.Group, str]:
    parent = root.get("crop_runs")
    if parent is None:
        raise ValueError("Zarr archive is missing crop_runs.")
    if crop_run:
        if crop_run not in parent:
            raise ValueError(f"crop_runs/{crop_run} not found.")
        return parent, parent[crop_run], str(crop_run)
    for attr_name in ("latest_any", "latest", "latest_materialized"):
        candidate = parent.attrs.get(attr_name)
        if candidate and str(candidate) in parent:
            return parent, parent[str(candidate)], str(candidate)
    raise ValueError("No crop run specified and crop_runs has no latest pointer.")


def _resolve_recording_dir(zarr_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent.resolve()
    for parent in zarr_path.parents:
        if (parent / "recording_manifest.json").exists():
            return parent.resolve()
    return zarr_path.parent.resolve()


def _resolve_source_video_path(
    *,
    root: zarr.Group,
    recording_dir: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source video path is not a file: {path}")
        return path

    cams_dir = recording_dir / "cams"
    if cams_dir.is_dir():
        mp4s = sorted(cams_dir.glob("*.mp4"))
        if len(mp4s) == 1:
            return mp4s[0].resolve()
        if len(mp4s) > 1:
            raise ValueError(f"Multiple camera videos found under {cams_dir}; pass --source-video-path.")

    for attr_name in ("source_video_path", "video_path", "source_video"):
        value = root.attrs.get(attr_name)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = recording_dir / path
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Unable to resolve full camera source video. Pass --source-video-path or ensure cams/*.mp4 exists."
    )


def _resolve_roi_shape(crop_group: zarr.Group) -> tuple[int, int]:
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    if "roi_sizes_full" in crop_group:
        sizes = np.asarray(crop_group["roi_sizes_full"][:], dtype=np.int32).reshape(-1, 2)
        valid = np.logical_and(sizes[:, 0] > 0, sizes[:, 1] > 0)
        unique = np.unique(sizes[valid], axis=0)
        if unique.shape[0] == 1:
            width, height = int(unique[0, 0]), int(unique[0, 1])
            return height, width
    raise ValueError("Unable to resolve fixed ROI shape from acquisition crop run.")


def _read_array_or_default(
    group: zarr.Group,
    name: str,
    *,
    rows: int,
    shape_suffix: tuple[int, ...] = (),
    dtype: Any,
    fill: Any,
) -> np.ndarray:
    arr = group.get(name)
    if arr is not None:
        return np.asarray(arr[:], dtype=dtype)
    shape = (int(rows), *shape_suffix)
    return np.full(shape, fill, dtype=dtype)


def _bbox_roi_xyxy(bbox_img_xyxy: np.ndarray, roi_coordinates_full: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox_img_xyxy, dtype=np.float64).reshape(-1, 4)
    offsets = np.asarray(roi_coordinates_full, dtype=np.float64).reshape(-1, 2)
    out = bbox.copy()
    out[:, [0, 2]] -= offsets[:, 0:1]
    out[:, [1, 3]] -= offsets[:, 1:2]
    return out


def _bbox_crop_norm_xywh(bbox_roi_xyxy: np.ndarray, roi_shape: tuple[int, int]) -> np.ndarray:
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    bbox = np.asarray(bbox_roi_xyxy, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(bbox, dtype=np.float64)
    out[:, 0] = ((bbox[:, 0] + bbox[:, 2]) * 0.5) / float(roi_w)
    out[:, 1] = ((bbox[:, 1] + bbox[:, 3]) * 0.5) / float(roi_h)
    out[:, 2] = (bbox[:, 2] - bbox[:, 0]) / float(roi_w)
    out[:, 3] = (bbox[:, 3] - bbox[:, 1]) / float(roi_h)
    return out


def _open_indexed_decoder(video_path: Path) -> Any:
    try:
        import PyNvVideoCodec as nvc  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"PyNvVideoCodec import failed; cannot use indexed decode: {exc}") from exc
    return nvc.SimpleDecoder(
        str(video_path),
        gpu_id=0,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
    )


def _open_sequential_reader(video_path: Path) -> Any:
    from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader

    return PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)


def _decoder_dimensions(decoder: Any) -> tuple[int, int]:
    if hasattr(decoder, "source_height") and hasattr(decoder, "source_width"):
        return int(decoder.source_height), int(decoder.source_width)
    if hasattr(decoder, "get_stream_metadata"):
        metadata = decoder.get_stream_metadata()
        return int(metadata.height), int(metadata.width)
    raise ValueError("Unable to resolve PyNvVideoCodec source dimensions.")


def _close_decoder(decoder: Any | None) -> None:
    if decoder is None:
        return
    try:
        if hasattr(decoder, "close"):
            decoder.close()
        elif hasattr(decoder, "stop"):
            decoder.stop()
        elif hasattr(decoder, "end"):
            decoder.end()
    except Exception:
        pass


def _choose_decode_mode(requested: str, frames: np.ndarray) -> str:
    if requested not in DECODE_MODES:
        raise ValueError(f"Unsupported decode mode: {requested}")
    if requested != "auto":
        return requested
    if frames.size == 0:
        return "indexed"
    span = int(frames.max()) + 1
    return "indexed" if (frames.size / float(max(1, span))) < 0.5 else "sequential"


def _write_cache_rows(
    *,
    handle: Any,
    frame_tensor: Any,
    rows: Sequence[int],
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    row_stride: int,
    rows_written_mask: np.ndarray,
) -> None:
    crops = _crop_pynvvc_luma_frame(
        frame_tensor,
        roi_ids=rows,
        roi_coordinates_full=roi_coordinates_full,
        roi_shape=roi_shape,
        video_shape=video_shape,
    )
    crops_cpu = np.ascontiguousarray(crops.cpu().numpy(), dtype=np.uint8)
    for local_idx, row in enumerate(rows):
        row_int = int(row)
        handle.seek(row_int * row_stride)
        handle.write(crops_cpu[int(local_idx)].tobytes(order="C"))
        rows_written_mask[row_int] = True


def _write_supplemental_cache_indexed(
    *,
    video_path: Path,
    manifest_path: Path,
    bin_path: Path,
    frame_to_rows: Mapping[int, list[int]],
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    decode_chunk_frames: int,
) -> dict[str, Any]:
    import torch

    row_stride = int(roi_shape[0]) * int(roi_shape[1])
    rows_written_mask = np.zeros(int(roi_coordinates_full.shape[0]), dtype=bool)
    timing = {"decode_seconds": 0.0, "crop_write_seconds": 0.0, "indexed_batches": 0, "decoded_frames": 0}
    decoder = None
    started = time.perf_counter()
    try:
        decoder = _open_indexed_decoder(video_path)
        source_height, source_width = _decoder_dimensions(decoder)
        if (source_height, source_width) != tuple(video_shape):
            raise ValueError(
                "PyNvVideoCodec dimensions do not match metadata: "
                f"decoder={source_width}x{source_height}, metadata={video_shape[1]}x{video_shape[0]}."
            )
        with bin_path.open("w+b") as handle:
            handle.truncate(int(row_stride * roi_coordinates_full.shape[0]))
            requested = sorted(int(frame_idx) for frame_idx in frame_to_rows)
            for start in range(0, len(requested), max(1, int(decode_chunk_frames))):
                frame_batch = requested[start : start + max(1, int(decode_chunk_frames))]
                decode_started = time.perf_counter()
                decoded_frames = decoder.get_batch_frames_by_index(frame_batch)
                timing["decode_seconds"] += float(time.perf_counter() - decode_started)
                timing["indexed_batches"] += 1
                timing["decoded_frames"] += int(len(decoded_frames))
                if len(decoded_frames) != len(frame_batch):
                    raise RuntimeError(
                        "PyNvVideoCodec indexed decode returned "
                        f"{len(decoded_frames)} frame(s) for {len(frame_batch)} requested index/indices."
                    )
                for frame_idx, frame in zip(frame_batch, decoded_frames):
                    write_started = time.perf_counter()
                    _write_cache_rows(
                        handle=handle,
                        frame_tensor=torch.from_dlpack(frame),
                        rows=frame_to_rows[int(frame_idx)],
                        roi_coordinates_full=roi_coordinates_full,
                        roi_shape=roi_shape,
                        video_shape=video_shape,
                        row_stride=row_stride,
                        rows_written_mask=rows_written_mask,
                    )
                    timing["crop_write_seconds"] += float(time.perf_counter() - write_started)
        if int(rows_written_mask.sum()) != int(roi_coordinates_full.shape[0]):
            raise RuntimeError(
                "Indexed supplemental cache write missed "
                f"{int(roi_coordinates_full.shape[0]) - int(rows_written_mask.sum())} rows."
            )
        timing["total_seconds"] = float(time.perf_counter() - started)
        return {"decode_mode_effective": "indexed", "timing": timing}
    finally:
        _close_decoder(decoder)


def _write_supplemental_cache_sequential(
    *,
    video_path: Path,
    bin_path: Path,
    frame_to_rows: Mapping[int, list[int]],
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
) -> dict[str, Any]:
    row_stride = int(roi_shape[0]) * int(roi_shape[1])
    rows_written_mask = np.zeros(int(roi_coordinates_full.shape[0]), dtype=bool)
    timing = {"decode_seconds": 0.0, "crop_write_seconds": 0.0, "decoded_frames": 0, "skipped_frames": 0}
    reader = None
    started = time.perf_counter()
    try:
        reader = _open_sequential_reader(video_path)
        source_height, source_width = _decoder_dimensions(reader)
        if (source_height, source_width) != tuple(video_shape):
            raise ValueError(
                "PyNvVideoCodec dimensions do not match metadata: "
                f"decoder={source_width}x{source_height}, metadata={video_shape[1]}x{video_shape[0]}."
            )
        max_frame = int(max(frame_to_rows)) if frame_to_rows else -1
        with bin_path.open("w+b") as handle:
            handle.truncate(int(row_stride * roi_coordinates_full.shape[0]))
            frame_iter = reader.iter_frames()
            frame_idx = 0
            while frame_idx <= max_frame:
                decode_started = time.perf_counter()
                try:
                    frame_tensor = next(frame_iter)
                except StopIteration:
                    break
                timing["decode_seconds"] += float(time.perf_counter() - decode_started)
                timing["decoded_frames"] += 1
                rows = frame_to_rows.get(frame_idx)
                if rows:
                    write_started = time.perf_counter()
                    _write_cache_rows(
                        handle=handle,
                        frame_tensor=frame_tensor,
                        rows=rows,
                        roi_coordinates_full=roi_coordinates_full,
                        roi_shape=roi_shape,
                        video_shape=video_shape,
                        row_stride=row_stride,
                        rows_written_mask=rows_written_mask,
                    )
                    timing["crop_write_seconds"] += float(time.perf_counter() - write_started)
                else:
                    timing["skipped_frames"] += 1
                frame_idx += 1
        if int(rows_written_mask.sum()) != int(roi_coordinates_full.shape[0]):
            raise RuntimeError(
                "Sequential supplemental cache write missed "
                f"{int(roi_coordinates_full.shape[0]) - int(rows_written_mask.sum())} rows."
            )
        timing["total_seconds"] = float(time.perf_counter() - started)
        return {"decode_mode_effective": "sequential", "timing": timing}
    finally:
        _close_decoder(reader)


def _write_supplemental_cache(
    *,
    zarr_path: Path,
    run_name: str,
    video_path: Path,
    manifest_path: Path,
    frame_indices: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    decode_mode: str,
    decode_chunk_frames: int,
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    bin_path = manifest_path.with_suffix(".bin")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Supplemental ROI cache manifest exists: {manifest_path}")
    if bin_path.exists() and not overwrite:
        raise FileExistsError(f"Supplemental ROI cache payload exists: {bin_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    frame_to_rows: dict[int, list[int]] = {}
    for row_idx, frame_idx in enumerate(np.asarray(frame_indices, dtype=np.int64).reshape(-1)):
        frame_to_rows.setdefault(int(frame_idx), []).append(int(row_idx))

    effective_requested = _choose_decode_mode(str(decode_mode), np.unique(frame_indices.astype(np.int64, copy=False)))
    try:
        if effective_requested == "indexed":
            cache_report = _write_supplemental_cache_indexed(
                video_path=video_path,
                manifest_path=manifest_path,
                bin_path=bin_path,
                frame_to_rows=frame_to_rows,
                roi_coordinates_full=roi_coordinates_full,
                roi_shape=roi_shape,
                video_shape=video_shape,
                decode_chunk_frames=decode_chunk_frames,
            )
        else:
            cache_report = _write_supplemental_cache_sequential(
                video_path=video_path,
                bin_path=bin_path,
                frame_to_rows=frame_to_rows,
                roi_coordinates_full=roi_coordinates_full,
                roi_shape=roi_shape,
                video_shape=video_shape,
            )
    except Exception as exc:
        if decode_mode == "indexed":
            raise
        if effective_requested == "indexed":
            cache_report = _write_supplemental_cache_sequential(
                video_path=video_path,
                bin_path=bin_path,
                frame_to_rows=frame_to_rows,
                roi_coordinates_full=roi_coordinates_full,
                roi_shape=roi_shape,
                video_shape=video_shape,
            )
            cache_report["indexed_fallback_error"] = f"{exc.__class__.__name__}: {exc}"
        else:
            raise

    total_bytes = int(roi_coordinates_full.shape[0]) * int(roi_shape[0]) * int(roi_shape[1])
    pixel_contract = flat_cache_pixel_contract_for_backend("pynvvc_luma")
    manifest = {
        "schema": FLAT_ROI_CACHE_SCHEMA,
        "layout": FLAT_ROI_CACHE_LAYOUT,
        "cache_complete": True,
        "cache_key": f"{run_name}:offline_supplement:{roi_coordinates_full.shape[0]}",
        "manifest_path": str(manifest_path),
        "created_at_utc": _utc_now(),
        "source": {
            "archive_path": str(zarr_path),
            "crop_run_name": str(run_name),
            "source_crop_storage_mode": "geometry_only",
            "frame_source_kind": "source_video_path",
            "frame_source_path": str(video_path),
            "frame_source_identity": {
                "source_video_path": str(video_path),
                "frame_index_min": int(frame_indices.min()) if frame_indices.size else None,
                "frame_index_max": int(frame_indices.max()) if frame_indices.size else None,
                "row_count": int(frame_indices.shape[0]),
            },
        },
        "array": {
            "bin_path": bin_path.name,
            "dtype": "uint8",
            "shape": [int(roi_coordinates_full.shape[0]), int(roi_shape[0]), int(roi_shape[1])],
            "order": "C",
            "row_stride_bytes": int(roi_shape[0]) * int(roi_shape[1]),
            "total_bytes": int(total_bytes),
            "sha256": None,
        },
        "builder": {
            "module": __name__,
            "decode_backend_requested": "pynvvc_luma",
            "decode_backend_effective": "pynvvc_luma",
            "decode_mode_requested": str(decode_mode),
            "decode_mode_effective": cache_report["decode_mode_effective"],
            "decode_chunk_frames": int(decode_chunk_frames),
            "pixel_contract": pixel_contract,
            "pixel_contract_name": pixel_contract.get("name"),
            "timing": cache_report["timing"],
        },
    }
    if "indexed_fallback_error" in cache_report:
        manifest["builder"]["indexed_fallback_error"] = cache_report["indexed_fallback_error"]
    tmp_path = manifest_path.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, manifest_path)
    return manifest


def _default_manifest_path(recording_dir: Path, zarr_path: Path, run_name: str) -> Path:
    zarr_stem = zarr_path.name.removesuffix(".zarr")
    return (
        recording_dir
        / "derived"
        / "roi_cache"
        / run_name
        / f"{zarr_stem}__{run_name}.supplemental.flat_roi_cache.json"
    )


def _prepare_hybrid_payload(
    *,
    root: zarr.Group,
    acquisition_group: zarr.Group,
    refined_payload: Mapping[str, np.ndarray],
    frame_width: int,
    frame_height: int,
    roi_shape: tuple[int, int],
) -> dict[str, np.ndarray | dict[str, int]]:
    online_frame_indices = np.asarray(acquisition_group["frame_indices"][:], dtype=np.int64).reshape(-1)
    online_rows = int(online_frame_indices.shape[0])
    online_frame_set = set(int(v) for v in online_frame_indices.tolist())

    refined_frame_indices = np.asarray(refined_payload["frame_indices"], dtype=np.int64).reshape(-1)
    offline_mask = np.asarray([int(frame) not in online_frame_set for frame in refined_frame_indices], dtype=bool)
    if "bbox_img_xyxy" in refined_payload:
        refined_bbox_img = np.asarray(refined_payload["bbox_img_xyxy"], dtype=np.float64).reshape(-1, 4)
    else:
        refined_bbox_img = bbox_norm_cxcywh_to_img_xyxy(
            np.asarray(refined_payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4),
            width=int(frame_width),
            height=int(frame_height),
        )
    finite_bbox = np.isfinite(refined_bbox_img).all(axis=1)
    offline_mask = np.logical_and(offline_mask, finite_bbox)

    offline_frame_indices = refined_frame_indices[offline_mask]
    offline_bbox_img = refined_bbox_img[offline_mask]
    offline_bbox_norm = bbox_img_xyxy_to_norm_cxcywh(
        offline_bbox_img,
        width=int(frame_width),
        height=int(frame_height),
    )
    offline_roi_coordinates, offline_roi_sizes = compute_centered_roi_mapping(
        offline_bbox_img,
        roi_size=roi_shape,
    )
    offline_bbox_roi = _bbox_roi_xyxy(offline_bbox_img, offline_roi_coordinates)
    offline_bbox_crop_norm = _bbox_crop_norm_xywh(offline_bbox_roi, roi_shape)

    online_roi_coordinates = np.asarray(acquisition_group["roi_coordinates_full"][:], dtype=np.int32)
    online_roi_sizes = np.asarray(acquisition_group["roi_sizes_full"][:], dtype=np.int32)
    online_bbox_img = _read_array_or_default(
        acquisition_group,
        "bbox_img_xyxy",
        rows=online_rows,
        shape_suffix=(4,),
        dtype=np.float64,
        fill=np.nan,
    )
    online_bbox_norm = _read_array_or_default(
        acquisition_group,
        "bbox_norm_coords",
        rows=online_rows,
        shape_suffix=(4,),
        dtype=np.float64,
        fill=np.nan,
    )
    online_bbox_roi = _read_array_or_default(
        acquisition_group,
        "bbox_roi_xyxy",
        rows=online_rows,
        shape_suffix=(4,),
        dtype=np.float64,
        fill=np.nan,
    )
    online_bbox_crop_norm = _read_array_or_default(
        acquisition_group,
        "bbox_crop_norm_coords",
        rows=online_rows,
        shape_suffix=(4,),
        dtype=np.float64,
        fill=np.nan,
    )

    n_offline = int(offline_frame_indices.shape[0])
    combined = {
        "frame_indices": np.concatenate([online_frame_indices, offline_frame_indices]).astype(np.int64, copy=False),
        "source_frame_indices": np.concatenate([online_frame_indices, offline_frame_indices]).astype(np.int64, copy=False),
        "source_recording_frame_ids": np.concatenate(
            [
                _read_array_or_default(
                    acquisition_group,
                    "source_recording_frame_ids",
                    rows=online_rows,
                    dtype=np.int64,
                    fill=-1,
                ),
                (offline_frame_indices + 1).astype(np.int64, copy=False),
            ]
        ),
        "source_crop_meta_row_indices": np.concatenate(
            [
                _read_array_or_default(
                    acquisition_group,
                    "source_crop_meta_row_indices",
                    rows=online_rows,
                    dtype=np.int64,
                    fill=-1,
                ),
                np.full(n_offline, -1, dtype=np.int64),
            ]
        ),
        "source_crop_video_frame_indices": np.concatenate(
            [
                _read_array_or_default(
                    acquisition_group,
                    "source_crop_video_frame_indices",
                    rows=online_rows,
                    dtype=np.int64,
                    fill=-1,
                ),
                np.full(n_offline, -1, dtype=np.int64),
            ]
        ),
        "source_crop_local_frame_ids": np.concatenate(
            [
                _read_array_or_default(
                    acquisition_group,
                    "source_crop_local_frame_ids",
                    rows=online_rows,
                    dtype=np.int64,
                    fill=-1,
                ),
                np.full(n_offline, -1, dtype=np.int64),
            ]
        ),
        "source_crop_xywh": np.concatenate(
            [
                _read_array_or_default(
                    acquisition_group,
                    "source_crop_xywh",
                    rows=online_rows,
                    shape_suffix=(4,),
                    dtype=np.float64,
                    fill=np.nan,
                ),
                np.column_stack([offline_roi_coordinates, offline_roi_sizes]).astype(np.float64, copy=False),
            ]
        ),
        "roi_coordinates_full": np.concatenate([online_roi_coordinates, offline_roi_coordinates]).astype(np.int32, copy=False),
        "roi_sizes_full": np.concatenate([online_roi_sizes, offline_roi_sizes]).astype(np.int32, copy=False),
        "bbox_img_xyxy": np.concatenate([online_bbox_img, offline_bbox_img]).astype(np.float64, copy=False),
        "bbox_norm_coords": np.concatenate([online_bbox_norm, offline_bbox_norm]).astype(np.float64, copy=False),
        "bbox_roi_xyxy": np.concatenate([online_bbox_roi, offline_bbox_roi]).astype(np.float64, copy=False),
        "bbox_crop_norm_coords": np.concatenate([online_bbox_crop_norm, offline_bbox_crop_norm]).astype(np.float64, copy=False),
        "detection_success": np.ones(online_rows + n_offline, dtype=bool),
        "detection_source": np.concatenate(
            [
                np.full(online_rows, DETECTION_SOURCE_CODE_MAP["acquisition_live_detection"], dtype=np.int8),
                np.full(n_offline, DETECTION_SOURCE_CODE_MAP["offline_refined_detection"], dtype=np.int8),
            ]
        ),
        "source_pixel_kind_codes": np.concatenate(
            [
                np.full(online_rows, SOURCE_PIXEL_KIND_CODE_MAP["acquisition_crop_video"], dtype=np.int8),
                np.full(
                    n_offline,
                    SOURCE_PIXEL_KIND_CODE_MAP["offline_full_frame_supplemental_flat_cache"],
                    dtype=np.int8,
                ),
            ]
        ),
        "crop_state_codes": np.concatenate(
            [
                np.full(online_rows, CROP_STATE_CODE_MAP["detected_crop"], dtype=np.int8),
                np.full(n_offline, CROP_STATE_CODE_MAP["offline_recovered_crop"], dtype=np.int8),
            ]
        ),
        "supplemental_cache_row_indices": np.concatenate(
            [
                np.full(online_rows, -1, dtype=np.int64),
                np.arange(n_offline, dtype=np.int64),
            ]
        ),
        "source_refined_row_ids": np.concatenate(
            [
                np.full(online_rows, -1, dtype=np.int64),
                np.asarray(
                    refined_payload.get("refined_row_ids", np.arange(refined_frame_indices.shape[0])),
                    dtype=np.int64,
                ).reshape(-1)[offline_mask],
            ]
        ),
        "source_detect_row_index": np.concatenate(
            [
                np.full(online_rows, -1, dtype=np.int64),
                np.asarray(
                    refined_payload.get("source_detect_row_index", np.full(refined_frame_indices.shape[0], -1)),
                    dtype=np.int64,
                ).reshape(-1)[offline_mask],
            ]
        ),
    }
    order = np.argsort(np.asarray(combined["frame_indices"], dtype=np.int64), kind="stable")
    for name, arr in list(combined.items()):
        combined[name] = np.asarray(arr)[order]

    total_frames = max(
        int(root.attrs.get("total_frames") or 0),
        int(np.asarray(combined["frame_indices"]).max()) + 1 if combined["frame_indices"].size else 0,
    )
    frame_counts = np.zeros(total_frames, dtype=np.int32)
    if total_frames:
        np.add.at(frame_counts, np.asarray(combined["frame_indices"], dtype=np.int64), 1)
    combined["frame_counts"] = frame_counts
    combined["detection_indices"] = np.arange(int(combined["frame_indices"].shape[0]), dtype=np.int64)
    combined["summary"] = {
        "online_rows": int(online_rows),
        "offline_refined_rows_available": int(refined_frame_indices.shape[0]),
        "offline_recovered_rows": int(n_offline),
        "offline_rejected_duplicate_online_frame": int(np.count_nonzero(~offline_mask & finite_bbox)),
        "offline_rejected_nonfinite_bbox": int(np.count_nonzero(~finite_bbox)),
        "total_rows": int(combined["frame_indices"].shape[0]),
        "total_frames": int(total_frames),
    }
    return combined


def build_hybrid_acquisition_offline_crop_run(
    zarr_path: str | Path,
    *,
    acquisition_crop_run: str | None = None,
    refined_detect_run: str | None = None,
    run_name: str | None = None,
    recording_dir: str | Path | None = None,
    source_video_path: str | Path | None = None,
    supplemental_manifest_path: str | Path | None = None,
    decode_mode: str = "auto",
    decode_chunk_frames: int = 1,
    overwrite: bool = False,
    set_latest_any: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    resolved_recording_dir = _resolve_recording_dir(
        archive_path,
        Path(recording_dir) if recording_dir is not None else None,
    )
    resolved_run_name = run_name or _utc_run_name()
    manifest_path = (
        Path(supplemental_manifest_path).expanduser()
        if supplemental_manifest_path is not None
        else _default_manifest_path(resolved_recording_dir, archive_path, resolved_run_name)
    )

    root = zarr.open_group(str(archive_path), mode="a" if apply else "r", use_consolidated=False)
    crop_parent, acquisition_group, resolved_acquisition_run = _resolve_crop_run(root, acquisition_crop_run)
    refined_group, resolved_refined_run = resolve_curated_refined_detect_run(root, run_name=refined_detect_run)
    refined_payload = extract_present_curated_rows(refined_group)
    frame_height, frame_width = resolve_full_frame_shape(root)
    roi_shape = _resolve_roi_shape(acquisition_group)
    full_video_path = _resolve_source_video_path(
        root=root,
        recording_dir=resolved_recording_dir,
        explicit=Path(source_video_path) if source_video_path is not None else None,
    )
    payload = _prepare_hybrid_payload(
        root=root,
        acquisition_group=acquisition_group,
        refined_payload=refined_payload,
        frame_width=frame_width,
        frame_height=frame_height,
        roi_shape=roi_shape,
    )
    summary = dict(payload.pop("summary"))  # type: ignore[arg-type]
    offline_rows = np.flatnonzero(
        np.asarray(payload["source_pixel_kind_codes"], dtype=np.int8)
        == SOURCE_PIXEL_KIND_CODE_MAP["offline_full_frame_supplemental_flat_cache"]
    )

    plan = {
        "status": "dry_run" if not apply else "planned",
        "zarr_path": str(archive_path),
        "recording_dir": str(resolved_recording_dir),
        "source_video_path": str(full_video_path),
        "acquisition_crop_run": str(resolved_acquisition_run),
        "refined_detect_run": str(resolved_refined_run),
        "target_crop_run": str(resolved_run_name),
        "supplemental_manifest_path": str(manifest_path),
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "frame_shape": [int(frame_height), int(frame_width)],
        "decode_mode_requested": str(decode_mode),
        "decode_chunk_frames": int(decode_chunk_frames),
        "set_latest_any": bool(set_latest_any),
        "summary": summary,
    }
    if not apply:
        return plan

    if resolved_run_name in crop_parent:
        if not overwrite:
            raise FileExistsError(f"crop_runs/{resolved_run_name} already exists.")
        assert_crop_run_unreferenced(crop_parent, resolved_run_name)
        del crop_parent[resolved_run_name]

    supplemental_manifest: dict[str, Any] | None = None
    if offline_rows.size:
        supplemental_manifest = _write_supplemental_cache(
            zarr_path=archive_path,
            run_name=resolved_run_name,
            video_path=full_video_path,
            manifest_path=manifest_path,
            frame_indices=np.asarray(payload["frame_indices"], dtype=np.int64)[offline_rows],
            roi_coordinates_full=np.asarray(payload["roi_coordinates_full"], dtype=np.int32)[offline_rows],
            roi_shape=roi_shape,
            video_shape=(int(frame_height), int(frame_width)),
            decode_mode=str(decode_mode),
            decode_chunk_frames=int(decode_chunk_frames),
            overwrite=bool(overwrite),
        )

    group = crop_parent.create_group(resolved_run_name)
    mark_run_started(group, run_name=resolved_run_name, stage="crop")
    started = time.perf_counter()
    try:
        for name in (
            "frame_indices",
            "source_frame_indices",
            "source_recording_frame_ids",
            "source_crop_meta_row_indices",
            "source_crop_video_frame_indices",
            "source_crop_local_frame_ids",
            "source_crop_xywh",
            "roi_coordinates_full",
            "roi_sizes_full",
            "bbox_img_xyxy",
            "bbox_norm_coords",
            "bbox_roi_xyxy",
            "bbox_crop_norm_coords",
            "detection_success",
            "detection_source",
            "source_pixel_kind_codes",
            "crop_state_codes",
            "supplemental_cache_row_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
            "frame_counts",
            "detection_indices",
        ):
            _create_array(group, name, np.asarray(payload[name]))
        # Compatibility aliases used by acquisition crop-video readers/reviewers.
        _create_array(group, "selected_live_detection_bbox_img_xyxy", np.asarray(payload["bbox_img_xyxy"]))
        _create_array(group, "selected_live_detection_bbox_norm_coords", np.asarray(payload["bbox_norm_coords"]))
        _create_array(group, "selected_live_detection_bbox_roi_xyxy", np.asarray(payload["bbox_roi_xyxy"]))
        _create_array(group, "selected_live_detection_bbox_crop_norm_coords", np.asarray(payload["bbox_crop_norm_coords"]))
        _create_array(group, "realtime_detection_bbox_roi_xyxy", np.asarray(payload["bbox_roi_xyxy"]))
        stamp_geometry_preload_attrs(group)

        now = _utc_now()
        attrs = {
            "schema_id": SCHEMA_ID,
            "crop_storage_mode": "geometry_only",
            "source_pixels": "hybrid_acquisition_crop_video_offline_supplement",
            "roi_pixel_provider": "hybrid_acquisition_crop_video_offline_supplement",
            "source_type": "hybrid_acquisition_crop_video_offline_supplement",
            "roi_size": [int(roi_shape[0]), int(roi_shape[1])],
            "roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
            "roi_pixel_contract": orange_mono_pynvvc_luma_pixel_contract(),
            "decode_backend": "pynvvc_luma",
            "decode_mode_requested": str(decode_mode),
            "source_video_path": str(full_video_path),
            "source_crop_video_path": acquisition_group.attrs.get("source_crop_video_path")
            or acquisition_group.attrs.get("source_video_path"),
            "source_acquisition_crop_run": str(resolved_acquisition_run),
            "source_refined_detect_run": str(resolved_refined_run),
            "supplemental_roi_cache_manifest": str(manifest_path) if supplemental_manifest else None,
            "source_pixel_kind_code_map": SOURCE_PIXEL_KIND_CODE_MAP,
            "crop_state_code_map": CROP_STATE_CODE_MAP,
            "detection_source_code_map": DETECTION_SOURCE_CODE_MAP,
            "source_crop_video_frame_indices_semantics": "zero_based_frame_index_in_acquisition_crop_video_or_-1_for_supplemental_rows",
            "supplemental_cache_row_indices_semantics": "row_index_in_supplemental_flat_roi_cache_or_-1_for_acquisition_video_rows",
            "roi_coordinates_full_coordinate_space": "source_image_xy",
            "bbox_norm_coords_semantics": "bbox_xywh_normalized_to_full_frame",
            "bbox_img_xyxy_semantics": "bbox_xyxy_full_frame_pixels",
            "summary_statistics": summary,
            "created_at_utc": now,
            "status": "completed",
            "completed_at_utc": now,
            "duration_seconds": float(time.perf_counter() - started),
        }
        if supplemental_manifest is not None:
            attrs["supplemental_roi_cache_manifest_payload"] = supplemental_manifest
        group.attrs.update(attrs)

        git_info = get_git_info(Path(__file__).resolve().parents[3])
        env_info = get_environment_info(include_all_packages=False, disk_path=str(archive_path), collect_ip=False)
        provenance = build_stage_provenance(
            stage="crop",
            command=" ".join(sys.argv),
            created_at_utc=now,
            version=git_info.get("short_hash") or git_info.get("commit_hash"),
            git=git_info,
            environment=env_info.get("environment"),
            platform=env_info.get("platform"),
            parameters={
                "run_name": resolved_run_name,
                "decode_mode": str(decode_mode),
                "set_latest_any": bool(set_latest_any),
            },
            inputs={
                "zarr_path": str(archive_path),
                "source_video_path": str(full_video_path),
                "source_acquisition_crop_run": str(resolved_acquisition_run),
                "source_refined_detect_run": str(resolved_refined_run),
            },
            artifacts={
                "run_path": f"crop_runs/{resolved_run_name}",
                "supplemental_roi_cache_manifest": str(manifest_path) if supplemental_manifest else None,
            },
        )
        write_stage_provenance(group, provenance)
        mark_run_complete(
            group,
            parent_group=crop_parent,
            run_name=resolved_run_name,
            run_provenance=build_run_provenance_from_stage_record(provenance),
        )
        crop_parent.attrs["latest_any"] = resolved_run_name
        if set_latest_any:
            crop_parent.attrs["latest_hybrid_acquisition_offline"] = resolved_run_name
        return {
            **plan,
            "status": "ok",
            "applied": True,
            "host": socket.gethostname(),
            "pid": int(os.getpid()),
            "supplemental_cache": supplemental_manifest,
        }
    except Exception as exc:
        mark_run_failed(group, error=str(exc))
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hybrid crop run that uses acquisition crop-video rows plus a "
            "supplemental flat cache for frames recovered only by offline refined detection."
        )
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--acquisition-crop-run", help="Existing acquisition crop-video crop run.")
    parser.add_argument("--refined-detect-run", help="Refined detect run to recover offline-only frames.")
    parser.add_argument("--run-name", help="Output crop_runs/<run> name.")
    parser.add_argument("--recording-dir", type=Path, help="Recording directory used to resolve cams/*.mp4.")
    parser.add_argument("--source-video-path", type=Path, help="Full camera source MP4 for supplemental rows.")
    parser.add_argument("--supplemental-manifest-path", type=Path, help="Output flat-cache manifest path.")
    parser.add_argument("--decode-mode", choices=DECODE_MODES, default="auto")
    parser.add_argument("--decode-chunk-frames", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set-latest-any", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Write the cache and crop run.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_hybrid_acquisition_offline_crop_run(
        args.zarr_path,
        acquisition_crop_run=args.acquisition_crop_run,
        refined_detect_run=args.refined_detect_run,
        run_name=args.run_name,
        recording_dir=args.recording_dir,
        source_video_path=args.source_video_path,
        supplemental_manifest_path=args.supplemental_manifest_path,
        decode_mode=args.decode_mode,
        decode_chunk_frames=args.decode_chunk_frames,
        overwrite=args.overwrite,
        set_latest_any=args.set_latest_any,
        apply=args.apply,
    )
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.json or args.output_json is None:
        print(text)
    else:
        print(
            f"status: {report['status']}\n"
            f"target_crop_run: {report['target_crop_run']}\n"
            f"supplemental_manifest_path: {report['supplemental_manifest_path']}\n"
            f"offline_recovered_rows: {report['summary']['offline_recovered_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
