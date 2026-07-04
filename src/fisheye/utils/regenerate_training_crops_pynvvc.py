"""Regenerate training crop ROI pixels from Orange mono videos via PyNvVideoCodec.

This utility creates a new materialized ``crop_runs/<target>`` group with the
same row geometry as an existing crop run, but rewrites ``roi_images`` from the
source MP4 using the PyNvVideoCodec NV12 Y/luma plane. It is intended for the
training crop-representation migration and does not change ``crop_runs/latest``
unless explicitly requested.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_roi_layout import (
    DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.flat_roi_cache import _crop_pynvvc_luma_frame
from fisheye.shared.roi_pixel_contract import (
    APPLIED_RANGE_SEMANTICS_ORANGE_MONO_FULL_RANGE,
    CENTER_ROUNDING_NP_ROUND,
    DECODE_BACKEND_PYNVVC_LUMA,
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    orange_mono_pynvvc_luma_pixel_contract,
)


SOURCE_FRAME_INDEX_MODES = ("auto", "direct", "original_frame_indices", "source_frame_index_parquet")
DECODE_MODES = ("auto", "sequential", "indexed")
MODULE_NAME = "fisheye.utils.regenerate_training_crops_pynvvc"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _valid_attr_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "null"}:
        return None
    return text


def _first_attr_text(*values: Any) -> str | None:
    for value in values:
        text = _valid_attr_text(value)
        if text:
            return text
    return None


def _first_positive_int(*values: Any) -> int | None:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _resolve_crop_run(root: Any, crop_run: str | None) -> str:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise KeyError("Zarr archive is missing crop_runs.")
    if crop_run:
        if crop_run not in crop_parent:
            raise KeyError(f"Crop run '{crop_run}' not found.")
        return crop_run
    for attr_name in ("latest_materialized", "latest", "latest_any"):
        candidate = crop_parent.attrs.get(attr_name)
        if candidate and str(candidate) in crop_parent:
            return str(candidate)
    names = sorted(str(name) for name in crop_parent.group_keys())
    if len(names) == 1:
        return names[0]
    raise ValueError("Unable to resolve crop run; pass --source-crop-run.")


def _default_target_run(source_crop_run: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{source_crop_run}_pynvvc_luma_{stamp}"


def _infer_recording_video_path(archive_path: Path | None) -> Path | None:
    if archive_path is None:
        return None
    recording_dir = archive_path.parent.parent
    cams_dir = recording_dir / "cams"
    if not cams_dir.is_dir():
        return None
    candidates = sorted(cams_dir.glob("*.mp4"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_video_path(
    root: Any,
    crop_group: Any,
    explicit: str | Path | None,
    archive_path: Path | None = None,
) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    metadata = root.attrs.get("source_video_metadata")
    metadata_path = metadata.get("source_path") if isinstance(metadata, Mapping) else None
    text = _first_attr_text(
        crop_group.attrs.get("source_video_path"),
        crop_group.attrs.get("video_source_path"),
        root.attrs.get("source_video_path"),
        root.attrs.get("video_source_path"),
        root.attrs.get("source_path"),
        metadata_path,
    )
    if text:
        return Path(text).expanduser()
    inferred = _infer_recording_video_path(archive_path)
    if inferred is not None:
        return inferred
    raise ValueError("Unable to resolve source video path; pass --video-path.")


def _resolve_source_frame_index_path(root: Any, archive_path: Path) -> Path | None:
    raw_video = root.get("raw_video")
    raw_attrs = getattr(raw_video, "attrs", {}) if raw_video is not None else {}
    text = _first_attr_text(
        root.attrs.get("source_frame_index_path"),
        raw_attrs.get("source_frame_index_path"),
    )
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = archive_path / path
    return path


def _load_clipped_source_frame_mapping(
    *,
    root: Any,
    archive_path: Path,
    crop_frame_indices: np.ndarray,
    mode: str,
) -> dict[str, Any] | None:
    if mode not in SOURCE_FRAME_INDEX_MODES:
        raise ValueError(f"Unsupported source frame index mode: {mode}")
    if mode in {"direct", "original_frame_indices"}:
        return None

    index_path = _resolve_source_frame_index_path(root, archive_path)
    if index_path is None:
        if mode == "source_frame_index_parquet":
            raise ValueError("source_frame_index_parquet requested, but no source frame index path is recorded.")
        return None
    if not index_path.exists():
        raise FileNotFoundError(f"source_frame_index.parquet not found: {index_path}")

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - dependency is present in Palette env
        raise RuntimeError(f"pyarrow import failed; cannot read clipped source frame index: {exc}") from exc

    required = {"sample_index", "video_path", "clip_local_frame_index"}
    optional = {"parent_frame_index", "clip_index"}
    schema = pq.read_schema(index_path)
    available = set(schema.names)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"source_frame_index.parquet missing required columns: {missing}")
    table = pq.read_table(index_path, columns=sorted((required | optional) & available)).combine_chunks()

    by_sample: dict[int, Mapping[str, Any]] = {}
    for row in table.to_pylist():
        by_sample[int(row["sample_index"])] = row

    local = np.asarray(crop_frame_indices, dtype=np.int64).reshape(-1)
    source_frame_indices = np.empty(local.shape[0], dtype=np.int64)
    source_clip_local_frame_indices = np.empty(local.shape[0], dtype=np.int64)
    source_clip_indices = np.full(local.shape[0], -1, dtype=np.int64)
    video_frame_to_rows: dict[Path, dict[int, list[int]]] = {}
    video_paths: set[str] = set()

    for roi_idx, sample_index in enumerate(local):
        row = by_sample.get(int(sample_index))
        if row is None:
            raise IndexError(
                "Crop frame index is outside source_frame_index.parquet sample_index values: "
                f"roi_row={roi_idx}, frame_index={int(sample_index)}."
            )
        video_path = Path(str(row["video_path"])).expanduser()
        clip_frame_index = int(row["clip_local_frame_index"])
        parent_frame_index = row.get("parent_frame_index")
        source_frame_indices[roi_idx] = int(parent_frame_index) if parent_frame_index is not None else int(sample_index)
        source_clip_local_frame_indices[roi_idx] = clip_frame_index
        clip_index = row.get("clip_index")
        if clip_index is not None:
            source_clip_indices[roi_idx] = int(clip_index)
        video_paths.add(str(video_path))
        video_frame_to_rows.setdefault(video_path, {}).setdefault(clip_frame_index, []).append(int(roi_idx))

    return {
        "mode": "source_frame_index_parquet",
        "source_frame_index_path": str(index_path),
        "source_frame_indices": source_frame_indices,
        "source_clip_local_frame_indices": source_clip_local_frame_indices,
        "source_clip_indices": source_clip_indices,
        "video_frame_to_rows": video_frame_to_rows,
        "video_paths": sorted(video_paths),
    }


def _shape_from_raw_video(root: Any) -> tuple[int, int] | None:
    raw_video = root.get("raw_video")
    if raw_video is None:
        return None
    for name in ("images_full", "images", "frames"):
        if name not in raw_video:
            continue
        shape = getattr(raw_video[name], "shape", None)
        if shape is None or len(shape) < 3:
            continue
        return int(shape[1]), int(shape[2])
    return None


def _shape_from_video_probe(video_path: Path) -> tuple[int, int] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if height > 0 and width > 0:
            return height, width
        return None
    finally:
        capture.release()


def _resolve_video_shape(root: Any, crop_group: Any, video_path: Path | None = None) -> tuple[int, int]:
    metadata = root.attrs.get("source_video_metadata")
    metadata_height = metadata.get("height") if isinstance(metadata, Mapping) else None
    metadata_width = metadata.get("width") if isinstance(metadata, Mapping) else None
    height = _first_positive_int(
        crop_group.attrs.get("height"),
        crop_group.attrs.get("source_video_height"),
        root.attrs.get("height"),
        root.attrs.get("source_video_height"),
        root.attrs.get("full_height"),
        metadata_height,
    )
    width = _first_positive_int(
        crop_group.attrs.get("width"),
        crop_group.attrs.get("source_video_width"),
        root.attrs.get("width"),
        root.attrs.get("source_video_width"),
        root.attrs.get("full_width"),
        metadata_width,
    )
    if height is not None and width is not None:
        return int(height), int(width)
    raw_shape = _shape_from_raw_video(root)
    if raw_shape is not None:
        return raw_shape
    if video_path is not None:
        probed_shape = _shape_from_video_probe(video_path)
        if probed_shape is not None:
            return probed_shape
    if height is None or width is None:
        raise ValueError("Unable to resolve source video dimensions from crop/root attrs.")
    return int(height), int(width)


def _resolve_roi_shape(crop_group: Any) -> tuple[int, int]:
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    if "roi_images" in crop_group and len(crop_group["roi_images"].shape) >= 3:
        return int(crop_group["roi_images"].shape[1]), int(crop_group["roi_images"].shape[2])
    raise ValueError("Unable to resolve ROI size from crop attrs or roi_images shape.")


def _load_original_frame_indices(root: Any) -> np.ndarray | None:
    raw_video = root.get("raw_video")
    if raw_video is None or "original_frame_indices" not in raw_video:
        return None
    return np.asarray(raw_video["original_frame_indices"][:], dtype=np.int64)


def _should_use_original_frame_indices(
    *,
    root: Any,
    crop_frame_indices: np.ndarray,
    original_frame_indices: np.ndarray | None,
    mode: str,
) -> bool:
    if mode == "direct":
        return False
    if mode == "original_frame_indices":
        if original_frame_indices is None:
            raise ValueError("--source-frame-index-mode original_frame_indices requested, but no mapping exists.")
        return True
    if original_frame_indices is None or crop_frame_indices.size == 0:
        return False
    max_local = int(np.max(crop_frame_indices))
    if max_local >= int(original_frame_indices.shape[0]):
        return False
    purpose = str(root.attrs.get("zarr_purpose") or root.attrs.get("zarr_use") or "").strip().lower()
    source_total = _first_positive_int(root.attrs.get("source_video_total_frames"))
    if purpose == "training":
        return True
    if source_total is not None and source_total != int(original_frame_indices.shape[0]):
        return True
    return False


def _map_source_frame_indices(
    *,
    root: Any,
    crop_frame_indices: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode not in SOURCE_FRAME_INDEX_MODES:
        raise ValueError(f"Unsupported source frame index mode: {mode}")
    original_frame_indices = _load_original_frame_indices(root)
    use_original = _should_use_original_frame_indices(
        root=root,
        crop_frame_indices=crop_frame_indices,
        original_frame_indices=original_frame_indices,
        mode=mode,
    )
    if not use_original:
        return np.asarray(crop_frame_indices, dtype=np.int64), {
            "mode": "direct",
            "original_frame_indices_available": original_frame_indices is not None,
            "original_frame_indices_length": (
                int(original_frame_indices.shape[0]) if original_frame_indices is not None else None
            ),
        }

    assert original_frame_indices is not None
    local = np.asarray(crop_frame_indices, dtype=np.int64)
    bad = np.flatnonzero((local < 0) | (local >= int(original_frame_indices.shape[0])))
    if bad.size:
        row = int(bad[0])
        raise IndexError(
            "Crop frame index is outside raw_video/original_frame_indices: "
            f"row={row}, frame_index={int(local[row])}, "
            f"mapping_length={int(original_frame_indices.shape[0])}."
        )
    mapped = original_frame_indices[local]
    return np.asarray(mapped, dtype=np.int64), {
        "mode": "original_frame_indices",
        "original_frame_indices_available": True,
        "original_frame_indices_length": int(original_frame_indices.shape[0]),
    }


def _frame_to_roi_indices(source_frame_indices: np.ndarray) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for roi_idx, frame_idx in enumerate(np.asarray(source_frame_indices, dtype=np.int64).reshape(-1)):
        mapping.setdefault(int(frame_idx), []).append(int(roi_idx))
    return mapping


def _open_pynvvc_luma_reader(video_path: Path) -> Any:
    from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader

    return PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)


def _open_pynvvc_luma_indexed_decoder(video_path: Path) -> Any:
    try:
        import PyNvVideoCodec as nvc  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"PyNvVideoCodec import failed; cannot use indexed PyNvVideoCodec backend: {exc}"
        ) from exc

    return nvc.SimpleDecoder(
        str(video_path),
        gpu_id=0,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
    )


def _decoder_dimensions(decoder: Any) -> tuple[int, int]:
    if hasattr(decoder, "source_height") and hasattr(decoder, "source_width"):
        return int(decoder.source_height), int(decoder.source_width)
    if hasattr(decoder, "get_stream_metadata"):
        metadata = decoder.get_stream_metadata()
        return int(metadata.height), int(metadata.width)
    raise ValueError("Unable to resolve PyNvVideoCodec source dimensions.")


def _close_reader(reader: Any | None) -> None:
    if reader is None:
        return
    try:
        if hasattr(reader, "close"):
            reader.close()
        elif hasattr(reader, "stop"):
            reader.stop()
        elif hasattr(reader, "end"):
            reader.end()
    except Exception:
        # PyNvVideoCodec's Python SimpleDecoder exposes stop(), but some builds
        # do not implement the underlying bound method. Decoder object
        # destruction is enough for this one-shot tool.
        pass


def _choose_decode_mode(
    *,
    requested: str,
    frame_to_rows: Mapping[int, list[int]],
    max_frame: int,
) -> str:
    if requested not in DECODE_MODES:
        raise ValueError(f"Unsupported decode mode: {requested}")
    if requested != "auto":
        return requested
    frames_with_rois = int(len(frame_to_rows))
    frames_to_scan = int(max_frame + 1) if max_frame >= 0 else 0
    if frames_to_scan <= 0:
        return "sequential"
    # Sparse sampled training Zarrs should not sequentially decode thousands of
    # irrelevant frames. Dense windows keep the low-overhead sequential reader.
    return "indexed" if frames_with_rois / float(frames_to_scan) < 0.5 else "sequential"


def _copy_source_array(source_group: Any, target_group: Any, name: str) -> None:
    source = source_group[name]
    data = np.asarray(source[:])
    chunks = getattr(source, "chunks", None)
    kwargs: dict[str, Any] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    target_group.create_array(name, **kwargs)


def _copy_crop_arrays(source_group: Any, target_group: Any) -> list[str]:
    copied: list[str] = []
    for name in sorted(str(k) for k in source_group.array_keys()):
        if name == "roi_images":
            continue
        _copy_source_array(source_group, target_group, name)
        copied.append(name)
    return copied


def _set_latest_pointers(crop_parent: Any, target_run: str) -> None:
    crop_parent.attrs["latest"] = target_run
    crop_parent.attrs["latest_materialized"] = target_run
    crop_parent.attrs["latest_any"] = target_run


def regenerate_training_crops_pynvvc(
    *,
    zarr_path: str | Path,
    source_crop_run: str | None = None,
    target_crop_run: str | None = None,
    video_path: str | Path | None = None,
    source_frame_index_mode: str = "auto",
    decode_mode: str = "auto",
    decode_chunk_frames: int = 1,
    roi_chunk_len: int = DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    overwrite: bool = False,
    set_latest: bool = False,
    consolidate_metadata: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    started = time.perf_counter()
    root = zarr.open_group(str(archive_path), mode="a", use_consolidated=False)
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise KeyError("Zarr archive is missing crop_runs.")
    resolved_source_crop = _resolve_crop_run(root, source_crop_run)
    source_group = crop_parent[resolved_source_crop]
    if "frame_indices" not in source_group:
        raise ValueError(f"crop_runs/{resolved_source_crop} is missing frame_indices.")
    if "roi_coordinates_full" not in source_group:
        raise ValueError(f"crop_runs/{resolved_source_crop} is missing roi_coordinates_full.")

    resolved_target_crop = target_crop_run or _default_target_run(resolved_source_crop)
    if resolved_target_crop in crop_parent and not overwrite:
        raise FileExistsError(
            f"Target crop run already exists: crop_runs/{resolved_target_crop}. "
            "Pass --overwrite to replace it."
        )

    frame_indices = np.asarray(source_group["frame_indices"][:], dtype=np.int64).reshape(-1)
    roi_coordinates_full = np.asarray(source_group["roi_coordinates_full"][:], dtype=np.int32)
    roi_shape = _resolve_roi_shape(source_group)
    total_rois = int(frame_indices.shape[0])
    if int(roi_coordinates_full.shape[0]) != total_rois:
        raise ValueError(
            "roi_coordinates_full length "
            f"{roi_coordinates_full.shape[0]} does not match frame_indices rows {total_rois}."
        )
    if "roi_images" in source_group and int(source_group["roi_images"].shape[0]) != total_rois:
        raise ValueError(
            f"source roi_images rows {source_group['roi_images'].shape[0]} "
            f"do not match frame_indices rows {total_rois}."
        )

    clipped_mapping = None
    if video_path is None:
        clipped_mapping = _load_clipped_source_frame_mapping(
            root=root,
            archive_path=archive_path,
            crop_frame_indices=frame_indices,
            mode=source_frame_index_mode,
        )

    if clipped_mapping is not None:
        resolved_video_path = None
        first_video = Path(str(clipped_mapping["video_paths"][0]))
        video_shape = _resolve_video_shape(root, source_group, first_video)
        source_frame_indices = np.asarray(clipped_mapping["source_frame_indices"], dtype=np.int64)
        frame_mapping = {
            "mode": "source_frame_index_parquet",
            "source_frame_index_path": clipped_mapping["source_frame_index_path"],
            "video_path_count": len(clipped_mapping["video_paths"]),
            "video_paths_preview": clipped_mapping["video_paths"][:5],
        }
        frame_to_rows: dict[int, list[int]] = {}
        max_frame = max(
            (max(frame_rows) for frame_rows in clipped_mapping["video_frame_to_rows"].values() if frame_rows),
            default=-1,
        )
        decode_mode_effective = "indexed" if decode_mode == "auto" else str(decode_mode)
        if decode_mode_effective == "sequential":
            raise ValueError("Clipped source_frame_index_parquet decoding currently requires indexed decode mode.")
    else:
        resolved_video_path = _resolve_video_path(root, source_group, video_path, archive_path)
        video_shape = _resolve_video_shape(root, source_group, resolved_video_path)
        source_frame_indices, frame_mapping = _map_source_frame_indices(
            root=root,
            crop_frame_indices=frame_indices,
            mode=source_frame_index_mode,
        )
        frame_to_rows = _frame_to_roi_indices(source_frame_indices)
        max_frame = int(max(frame_to_rows)) if frame_to_rows else -1
        decode_mode_effective = _choose_decode_mode(
            requested=str(decode_mode),
            frame_to_rows=frame_to_rows,
            max_frame=max_frame,
        )
    contract = orange_mono_pynvvc_luma_pixel_contract()
    layout = build_canonical_crop_roi_layout(
        total_rois=total_rois,
        preferred_chunk_len=int(roi_chunk_len),
        roi_storage="compressed",
    )

    plan: dict[str, Any] = {
        "status": "dry_run" if dry_run else "planned",
        "zarr_path": str(archive_path),
        "source_crop_run": str(resolved_source_crop),
        "target_crop_run": str(resolved_target_crop),
        "video_path": str(resolved_video_path) if resolved_video_path is not None else None,
        "source_video_paths": clipped_mapping["video_paths"] if clipped_mapping is not None else None,
        "video_shape": [int(video_shape[0]), int(video_shape[1])],
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "total_rois": int(total_rois),
        "source_frame_index_mapping": frame_mapping,
        "source_frame_min": int(source_frame_indices.min()) if source_frame_indices.size else None,
        "source_frame_max": int(source_frame_indices.max()) if source_frame_indices.size else None,
        "decode_mode_requested": str(decode_mode),
        "decode_mode_effective": str(decode_mode_effective),
        "decode_chunk_frames": int(decode_chunk_frames),
        "roi_chunk_len": int(layout.roi_chunk_len),
        "pixel_contract": contract,
        "pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        "set_latest": bool(set_latest),
        "consolidate_metadata": bool(consolidate_metadata),
    }
    if dry_run:
        return plan

    target_group = crop_parent.create_group(resolved_target_crop, overwrite=bool(overwrite))
    target_group.attrs.update(dict(source_group.attrs))
    target_group.attrs.update(
        {
            "status": "running",
            "created_at_utc": _utc_now(),
            "generated_by": MODULE_NAME,
            "source_crop_run": str(resolved_source_crop),
            "source_crop_path": f"crop_runs/{resolved_source_crop}",
            "crop_storage_mode": "materialized",
            "roi_size": [int(roi_shape[0]), int(roi_shape[1])],
            "source_video_path": str(resolved_video_path) if resolved_video_path is not None else "multiple_clips",
            "height": int(video_shape[0]),
            "width": int(video_shape[1]),
            "source_pixels": SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
            "source_pixel_contract": "orange.camera.mono8.full_frame.v1",
            "source_pixel_range": "0_255",
            "decode_backend": DECODE_BACKEND_PYNVVC_LUMA,
            "decode_backend_family": "PyNvVideoCodec",
            "decode_contract_status": "canonical_orange_mono_pynvvc_luma",
            "source_decode_surface": "nv12_y_plane_uint8",
            "applied_range_semantics": APPLIED_RANGE_SEMANTICS_ORANGE_MONO_FULL_RANGE,
            "container_color_range_observed": "tv",
            "container_color_range_handling": contract.get("container_color_range_handling"),
            "center_rounding": CENTER_ROUNDING_NP_ROUND,
            "decode_mode_requested": str(decode_mode),
            "decode_mode_effective": str(decode_mode_effective),
            "crop_pixel_migration_version": "training_orange_mono_pynvvc_luma_v1",
            "roi_image_representation": contract.get("image_representation"),
            "roi_pixel_contract": contract,
            "roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
            "source_frame_index_mapping": frame_mapping,
            "source_frame_index_mode_requested": str(source_frame_index_mode),
            "source_frame_indices_min": int(source_frame_indices.min()) if source_frame_indices.size else None,
            "source_frame_indices_max": int(source_frame_indices.max()) if source_frame_indices.size else None,
        }
    )
    if clipped_mapping is not None:
        target_group.attrs["source_video_paths"] = clipped_mapping["video_paths"]
        target_group.attrs["source_frame_index_path"] = clipped_mapping["source_frame_index_path"]
        target_group.attrs["source_layout"] = "rolling_clips"
    target_group.attrs.update(crop_roi_layout_attrs(layout))

    copied_arrays = _copy_crop_arrays(source_group, target_group)
    target_group.create_array(
        "source_frame_indices",
        data=np.asarray(source_frame_indices, dtype=np.int64),
        chunks=(max(1, min(4096, total_rois)),),
        overwrite=True,
    )
    if clipped_mapping is not None:
        target_group.create_array(
            "source_clip_local_frame_indices",
            data=np.asarray(clipped_mapping["source_clip_local_frame_indices"], dtype=np.int64),
            chunks=(max(1, min(4096, total_rois)),),
            overwrite=True,
        )
        target_group.create_array(
            "source_clip_indices",
            data=np.asarray(clipped_mapping["source_clip_indices"], dtype=np.int64),
            chunks=(max(1, min(4096, total_rois)),),
            overwrite=True,
        )
    roi_images = target_group.create_array(
        "roi_images",
        **build_crop_roi_create_kwargs(
            total_rois=total_rois,
            roi_sz=roi_shape,
            layout=layout,
            overwrite=True,
        ),
    )

    timing: dict[str, Any] = {
        "video_open_seconds": 0.0,
        "decode_seconds": 0.0,
        "crop_seconds": 0.0,
        "contiguous_seconds": 0.0,
        "write_seconds": 0.0,
        "decoded_frames": 0,
        "skipped_frames": 0,
        "frames_with_rois": (
            sum(len(frame_rows) for frame_rows in clipped_mapping["video_frame_to_rows"].values())
            if clipped_mapping is not None
            else int(len(frame_to_rows))
        ),
        "frames_requested": (
            sum(len(frame_rows) for frame_rows in clipped_mapping["video_frame_to_rows"].values())
            if clipped_mapping is not None
            else int(len(frame_to_rows))
        ),
        "decode_mode": str(decode_mode_effective),
        "indexed_batches": 0,
        "rows_written": 0,
        "video_count": len(clipped_mapping["video_paths"]) if clipped_mapping is not None else 1,
    }
    rows_written_mask = np.zeros(total_rois, dtype=bool)
    open_started = time.perf_counter()
    reader: Any | None = None

    def _write_frame_crops(frame_tensor: Any, rows: list[int]) -> None:
        crop_started = time.perf_counter()
        crops = _crop_pynvvc_luma_frame(
            frame_tensor,
            roi_ids=rows,
            roi_coordinates_full=roi_coordinates_full,
            roi_shape=roi_shape,
            video_shape=video_shape,
        )
        try:
            import torch

            if torch.cuda.is_available() and getattr(crops, "is_cuda", False):
                torch.cuda.synchronize()
        except Exception:
            pass
        timing["crop_seconds"] += float(time.perf_counter() - crop_started)

        contiguous_started = time.perf_counter()
        crops_cpu = np.ascontiguousarray(crops.cpu().numpy(), dtype=np.uint8)
        timing["contiguous_seconds"] += float(time.perf_counter() - contiguous_started)

        write_started = time.perf_counter()
        for local_idx, row in enumerate(rows):
            roi_images[int(row)] = crops_cpu[int(local_idx)]
            rows_written_mask[int(row)] = True
        timing["write_seconds"] += float(time.perf_counter() - write_started)
        timing["rows_written"] = int(rows_written_mask.sum())

    def _decode_indexed_frames(video_reader: Any, requested_by_frame: Mapping[int, list[int]]) -> None:
        import torch

        requested_frames = sorted(int(frame_idx) for frame_idx in requested_by_frame)
        for batch_start in range(0, len(requested_frames), int(decode_chunk_frames)):
            batch_frame_indices = requested_frames[batch_start : batch_start + int(decode_chunk_frames)]
            decode_started = time.perf_counter()
            decoded_frames = video_reader.get_batch_frames_by_index(batch_frame_indices)
            timing["decode_seconds"] += float(time.perf_counter() - decode_started)
            timing["decoded_frames"] += int(len(decoded_frames))
            timing["indexed_batches"] += 1

            if len(decoded_frames) != len(batch_frame_indices):
                raise RuntimeError(
                    "PyNvVideoCodec indexed decode returned "
                    f"{len(decoded_frames)} frame(s) for {len(batch_frame_indices)} requested index/indices."
                )

            for decoded_frame_index, frame in zip(batch_frame_indices, decoded_frames):
                frame_tensor = torch.from_dlpack(frame)
                _write_frame_crops(frame_tensor, requested_by_frame[int(decoded_frame_index)])

    try:
        if clipped_mapping is not None:
            expected_height, expected_width = int(video_shape[0]), int(video_shape[1])
            for clip_video_path, clip_frame_to_rows in clipped_mapping["video_frame_to_rows"].items():
                video_open_started = time.perf_counter()
                reader = _open_pynvvc_luma_indexed_decoder(Path(clip_video_path))
                timing["video_open_seconds"] += float(time.perf_counter() - video_open_started)
                source_height, source_width = _decoder_dimensions(reader)
                if (source_height, source_width) != (expected_height, expected_width):
                    raise ValueError(
                        "PyNvVideoCodec dimensions do not match Zarr metadata: "
                        f"video={clip_video_path}, decoder={source_width}x{source_height}, "
                        f"metadata={expected_width}x{expected_height}."
                    )
                _decode_indexed_frames(reader, clip_frame_to_rows)
                _close_reader(reader)
                reader = None
        else:
            assert resolved_video_path is not None
            reader = (
                _open_pynvvc_luma_reader(resolved_video_path)
                if decode_mode_effective == "sequential"
                else _open_pynvvc_luma_indexed_decoder(resolved_video_path)
            )
            timing["video_open_seconds"] = float(time.perf_counter() - open_started)
            source_height, source_width = _decoder_dimensions(reader)
            expected_height, expected_width = int(video_shape[0]), int(video_shape[1])
            if (source_height, source_width) != (expected_height, expected_width):
                raise ValueError(
                    "PyNvVideoCodec dimensions do not match Zarr metadata: "
                    f"decoder={source_width}x{source_height}, metadata={expected_width}x{expected_height}."
                )

            if decode_mode_effective == "sequential":
                frame_iter = reader.iter_frames()
                decoded_frame_index = 0
                while decoded_frame_index <= max_frame:
                    decode_started = time.perf_counter()
                    try:
                        frame_tensor = next(frame_iter)
                    except StopIteration:
                        break
                    timing["decode_seconds"] += float(time.perf_counter() - decode_started)
                    timing["decoded_frames"] += 1

                    rows = frame_to_rows.get(decoded_frame_index)
                    if not rows:
                        timing["skipped_frames"] += 1
                        decoded_frame_index += 1
                        continue
                    _write_frame_crops(frame_tensor, rows)
                    decoded_frame_index += 1
            else:
                _decode_indexed_frames(reader, frame_to_rows)

        rows_written = int(rows_written_mask.sum())
        if rows_written != total_rois:
            missing = int(total_rois - rows_written)
            raise RuntimeError(
                f"PyNvVideoCodec crop regeneration wrote {rows_written}/{total_rois} rows; "
                f"{missing} rows were not produced before decoder EOF."
            )

        duration = float(time.perf_counter() - started)
        timing["total_seconds"] = duration
        timing["decode_fps"] = (
            float(timing["decoded_frames"]) / float(timing["decode_seconds"])
            if float(timing["decode_seconds"]) > 0
            else None
        )
        timing["rows_per_second"] = float(total_rois) / duration if duration > 0 else None
        target_group.attrs["status"] = "completed"
        target_group.attrs["completed_at_utc"] = _utc_now()
        target_group.attrs["duration_seconds"] = duration
        target_group.attrs["summary_statistics"] = {
            "total_rois_cropped": int(total_rois),
            "roi_size": [int(roi_shape[0]), int(roi_shape[1])],
            "roi_pixels_materialized": True,
            "source_crop_run": str(resolved_source_crop),
            "pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        }
        if set_latest:
            _set_latest_pointers(crop_parent, resolved_target_crop)
        if consolidate_metadata:
            consolidate_started = time.perf_counter()
            zarr.consolidate_metadata(str(archive_path))
            timing["consolidate_metadata_seconds"] = float(time.perf_counter() - consolidate_started)
        target_group.attrs["timing"] = _json_safe(timing)

        return {
            **plan,
            "status": "ok",
            "copied_arrays": copied_arrays,
            "timing": _json_safe(timing),
            "host": socket.gethostname(),
            "pid": int(os.getpid()),
        }
    except Exception as exc:
        target_group.attrs["status"] = "failed"
        target_group.attrs["failed_at_utc"] = _utc_now()
        target_group.attrs["error_message"] = str(exc)
        raise
    finally:
        _close_reader(reader)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate a training crop run's roi_images from PyNvVideoCodec luma."
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--source-crop-run", help="Existing crop_runs/<run> to copy geometry from.")
    parser.add_argument("--target-crop-run", help="New crop_runs/<run> name to create.")
    parser.add_argument("--video-path", type=Path, help="Source MP4 path. Overrides zarr metadata.")
    parser.add_argument(
        "--source-frame-index-mode",
        choices=SOURCE_FRAME_INDEX_MODES,
        default="auto",
        help="How crop frame_indices map to source-video frame numbers.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=DECODE_MODES,
        default="auto",
        help="PyNvVideoCodec access pattern. auto uses indexed reads for sparse training samples.",
    )
    parser.add_argument(
        "--decode-chunk-frames",
        type=int,
        default=1,
        help=(
            "Frame indices per indexed PyNvVideoCodec request. Default 1 avoids slow wide-span "
            "indexed batches for sparse long training videos."
        ),
    )
    parser.add_argument("--roi-chunk-len", type=int, default=DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN)
    parser.add_argument("--overwrite", action="store_true", help="Replace target crop run if it already exists.")
    parser.add_argument(
        "--set-latest",
        action="store_true",
        help="Update crop_runs/latest, latest_materialized, and latest_any to the new run.",
    )
    parser.add_argument(
        "--no-consolidate-metadata",
        action="store_true",
        help="Do not refresh consolidated metadata after writing the crop run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print a plan without writing.")
    parser.add_argument("--output-json", type=Path, help="Write the report JSON to this path.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = regenerate_training_crops_pynvvc(
        zarr_path=args.zarr_path,
        source_crop_run=args.source_crop_run,
        target_crop_run=args.target_crop_run,
        video_path=args.video_path,
        source_frame_index_mode=args.source_frame_index_mode,
        decode_mode=args.decode_mode,
        decode_chunk_frames=args.decode_chunk_frames,
        roi_chunk_len=args.roi_chunk_len,
        overwrite=args.overwrite,
        set_latest=args.set_latest,
        consolidate_metadata=not args.no_consolidate_metadata,
        dry_run=args.dry_run,
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
            f"source_crop_run: {report['source_crop_run']}\n"
            f"target_crop_run: {report['target_crop_run']}\n"
            f"total_rois: {report['total_rois']}\n"
            f"pixel_contract: {ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
