"""Materialize training crop ROI pixels from video or a verified flat cache.

This utility creates a new materialized ``crop_runs/<target>`` group with the
same row geometry as an existing crop run, but rewrites ``roi_images`` from the
source MP4 using the PyNvVideoCodec NV12 Y/luma plane, or copies them from an
exact SHA-256-verified flat ROI cache.  Cache bytes are copied into the Zarr;
the resulting training artifact never depends on the cache at read time.  It
does not change ``crop_runs/latest`` unless explicitly requested.
"""

from __future__ import annotations

from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.frame_domains import FrameDomain, FrameDomainError, FrameDomains
from fisheye.shared.crop_roi_layout import (
    DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.flat_roi_cache import (
    _crop_pynvvc_luma_frame,
    load_flat_roi_cache_manifest,
    open_flat_roi_cache,
)
from fisheye.shared.roi_pixel_contract import (
    APPLIED_RANGE_SEMANTICS_ORANGE_MONO_FULL_RANGE,
    CENTER_ROUNDING_NP_ROUND,
    DECODE_BACKEND_PYNVVC_LUMA,
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    SOURCE_PIXELS_ACQUISITION_CROP_VIDEO,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr.training_crop_materialization import (
    TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE,
    TRAINING_CROP_MATERIALIZATION_PROVIDERS,
    TRAINING_CROP_MATERIALIZATION_SCHEMA_ID,
    build_training_crop_materialization_binding,
)
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1


SOURCE_FRAME_INDEX_MODES = ("auto", "direct", "original_frame_indices", "source_frame_index_parquet")
DECODE_MODES = ("auto", "sequential", "indexed")
MODULE_NAME = "fisheye.utils.regenerate_training_crops_pynvvc"
TRAINING_CROP_MATERIALIZATION_SCHEMA = TRAINING_CROP_MATERIALIZATION_SCHEMA_ID
SOURCE_VIDEO_MATERIALIZATION_PROVIDER = "source_video_pynvvc_luma"
VERIFIED_CACHE_MATERIALIZATION_PROVIDER = "verified_flat_roi_cache"


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


def _parse_instance_keys(value: str | None) -> list[int] | None:
    if value is None:
        return None
    keys = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not keys:
        raise argparse.ArgumentTypeError(
            "--source-instance-keys requires at least one comma-separated uint64."
        )
    if any(key < 0 or key > np.iinfo(np.uint64).max for key in keys):
        raise argparse.ArgumentTypeError(
            "--source-instance-keys values must be valid uint64 integers."
        )
    return keys


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


def _map_stored_to_source_frames(
    root: Any,
    local_frame_indices: np.ndarray,
    original_frame_indices: np.ndarray,
) -> np.ndarray:
    try:
        return FrameDomains(root=root).convert(
            local_frame_indices,
            FrameDomain.STORED_ZARR,
            FrameDomain.SOURCE_VIDEO,
        )
    except FrameDomainError:
        return original_frame_indices[local_frame_indices]


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
    mapped = _map_stored_to_source_frames(root, local, original_frame_indices)
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


def _copy_source_array(
    source_group: Any,
    target_group: Any,
    name: str,
    *,
    source_row_ids: np.ndarray,
    source_row_count: int,
    selected_frame_indices: np.ndarray,
) -> None:
    source = source_group[name]
    data = np.asarray(source[:])
    if name == "frame_row_offsets":
        n_frames = int(data.shape[0]) - 1
        counts = np.bincount(
            np.asarray(selected_frame_indices, dtype=np.int64),
            minlength=max(0, n_frames),
        )
        data = np.zeros(n_frames + 1, dtype=np.int64)
        if n_frames:
            data[1:] = np.cumsum(counts[:n_frames], dtype=np.int64)
    elif data.ndim >= 1 and int(data.shape[0]) == int(source_row_count):
        data = data[source_row_ids]
    chunks = getattr(source, "chunks", None)
    kwargs: dict[str, Any] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    target_group.create_array(name, **kwargs)


def _copy_crop_arrays(
    source_group: Any,
    target_group: Any,
    *,
    source_row_ids: np.ndarray,
    source_row_count: int,
    selected_frame_indices: np.ndarray,
) -> list[str]:
    copied: list[str] = []
    for name in sorted(str(k) for k in source_group.array_keys()):
        if name == "roi_images":
            continue
        _copy_source_array(
            source_group,
            target_group,
            name,
            source_row_ids=source_row_ids,
            source_row_count=source_row_count,
            selected_frame_indices=selected_frame_indices,
        )
        copied.append(name)
    return copied


def _set_latest_pointers(crop_parent: Any, target_run: str) -> None:
    crop_parent.attrs["latest"] = target_run
    crop_parent.attrs["latest_materialized"] = target_run
    crop_parent.attrs["latest_any"] = target_run


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _cache_pixel_contract(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    builder = manifest.get("builder")
    if not isinstance(builder, Mapping):
        raise ValueError("Flat ROI cache manifest is missing builder provenance.")
    raw_contract = builder.get("pixel_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("Flat ROI cache manifest is missing builder.pixel_contract.")
    contract = dict(raw_contract)
    name = str(builder.get("pixel_contract_name") or contract.get("name") or "")
    if name != ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME:
        raise ValueError(
            "Training crop materialization requires the canonical Orange mono "
            f"PyNvVC-luma pixel contract; got {name!r}."
        )
    if contract.get("name") != name:
        raise ValueError(
            "Flat ROI cache pixel-contract name disagrees with its contract payload."
        )
    expected_contract = orange_mono_pynvvc_luma_pixel_contract()
    if _json_safe(contract) != _json_safe(expected_contract):
        raise ValueError(
            "Flat ROI cache pixel contract does not exactly match the canonical "
            "Orange mono PyNvVC-luma contract."
        )
    return contract, name


def _source_crop_manifest_binding(source_group: Any) -> dict[str, Any]:
    raw = source_group.attrs.get("run_manifest")
    if not isinstance(raw, Mapping):
        return {
            "available": False,
            "reason": "source_crop_run_has_no_run_manifest",
        }
    envelope = dict(raw)
    digest = envelope.get("payload_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Source crop run_manifest lacks one exact payload_digest.")
    return {
        "available": True,
        "schema_id": envelope.get("schema_id"),
        "schema_version": envelope.get("schema_version"),
        "payload_digest": digest,
    }


def regenerate_training_crops_pynvvc(
    *,
    zarr_path: str | Path,
    source_zarr_path: str | Path | None = None,
    source_crop_run: str | None = None,
    target_crop_run: str | None = None,
    video_path: str | Path | None = None,
    roi_cache_manifest: str | Path | None = None,
    cache_copy_batch_rows: int = 1024,
    source_instance_keys: Sequence[int] | None = None,
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
    if not archive_path.exists():
        raise FileNotFoundError(
            "Training crop materialization extends an existing training Zarr; "
            f"destination does not exist: {archive_path}"
        )
    source_archive_path = (
        Path(source_zarr_path).expanduser().resolve()
        if source_zarr_path is not None
        else archive_path
    )
    cache_manifest_path = (
        Path(roi_cache_manifest).expanduser().resolve()
        if roi_cache_manifest is not None
        else None
    )
    if cache_manifest_path is not None and video_path is not None:
        raise ValueError(
            "--roi-cache-manifest and --video-path are mutually exclusive."
        )
    if type(cache_copy_batch_rows) is not int or cache_copy_batch_rows <= 0:
        raise ValueError("cache_copy_batch_rows must be a positive exact integer.")
    materialization_provider = (
        VERIFIED_CACHE_MATERIALIZATION_PROVIDER
        if cache_manifest_path is not None
        else SOURCE_VIDEO_MATERIALIZATION_PROVIDER
    )
    started = time.perf_counter()
    root = zarr.open_group(
        str(archive_path),
        mode="r" if dry_run else "a",
        use_consolidated=False,
    )
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise ValueError(
            "Training crop materialization requires zarr_purpose='training' on the "
            "destination archive."
        )
    crop_parent = root.get("crop_runs")
    source_root = (
        root
        if source_archive_path == archive_path
        else zarr.open_group(str(source_archive_path), mode="r", use_consolidated=False)
    )
    resolved_source_crop = _resolve_crop_run(source_root, source_crop_run)
    source_crop_parent = source_root.get("crop_runs")
    if source_crop_parent is None:
        raise KeyError("Source Zarr archive is missing crop_runs.")
    source_group = source_crop_parent[resolved_source_crop]
    if "frame_indices" not in source_group:
        raise ValueError(f"crop_runs/{resolved_source_crop} is missing frame_indices.")
    if "roi_coordinates_full" not in source_group:
        raise ValueError(f"crop_runs/{resolved_source_crop} is missing roi_coordinates_full.")

    resolved_target_crop = target_crop_run or _default_target_run(resolved_source_crop)
    if (
        crop_parent is not None
        and resolved_target_crop in crop_parent
        and not overwrite
    ):
        raise FileExistsError(
            f"Target crop run already exists: crop_runs/{resolved_target_crop}. "
            "Pass --overwrite to replace it."
        )

    all_frame_indices = np.asarray(
        source_group["frame_indices"][:], dtype=np.int64
    ).reshape(-1)
    all_roi_coordinates_full = np.asarray(
        source_group["roi_coordinates_full"][:], dtype=np.int32
    )
    roi_shape = _resolve_roi_shape(source_group)
    source_total_rois = int(all_frame_indices.shape[0])
    if int(all_roi_coordinates_full.shape[0]) != source_total_rois:
        raise ValueError(
            "roi_coordinates_full length "
            f"{all_roi_coordinates_full.shape[0]} does not match frame_indices rows {source_total_rois}."
        )
    if "roi_images" in source_group and int(source_group["roi_images"].shape[0]) != source_total_rois:
        raise ValueError(
            f"source roi_images rows {source_group['roi_images'].shape[0]} "
            f"do not match frame_indices rows {source_total_rois}."
        )
    source_row_ids = np.arange(source_total_rois, dtype=np.int64)
    requested_instance_keys: list[int] | None = None
    if source_instance_keys is not None:
        requested_instance_keys = sorted({int(value) for value in source_instance_keys})
        if not requested_instance_keys:
            raise ValueError("source_instance_keys cannot be empty when supplied.")
        if "instance_key" not in source_group:
            raise ValueError(
                "Instance-key training selection requires source crop instance_key."
            )
        available_keys = np.asarray(
            source_group["instance_key"][:], dtype=np.uint64
        ).reshape(-1)
        if (
            available_keys.shape[0] != source_total_rois
            or np.unique(available_keys).shape[0] != available_keys.shape[0]
        ):
            raise ValueError("Source crop instance_key must be complete and unique.")
        row_by_key = {
            int(key): int(row) for row, key in enumerate(available_keys.tolist())
        }
        missing_keys = [key for key in requested_instance_keys if key not in row_by_key]
        if missing_keys:
            raise KeyError(
                f"Requested instance_key values are absent from the crop source: {missing_keys[:10]}."
            )
        source_row_ids = np.asarray(
            sorted(row_by_key[key] for key in requested_instance_keys),
            dtype=np.int64,
        )
    frame_indices = all_frame_indices[source_row_ids]
    roi_coordinates_full = all_roi_coordinates_full[source_row_ids]
    total_rois = int(source_row_ids.shape[0])

    cache_manifest: dict[str, Any] | None = None
    cache_array: Any | None = None
    cache_manifest_sha256: str | None = None
    cache_payload_sha256: str | None = None
    cache_pixel_contract: dict[str, Any] | None = None
    cache_pixel_contract_name: str | None = None
    if cache_manifest_path is not None:
        cache_manifest = load_flat_roi_cache_manifest(cache_manifest_path)
        cache_pixel_contract, cache_pixel_contract_name = _cache_pixel_contract(
            cache_manifest
        )
        cache_array = open_flat_roi_cache(
            cache_manifest_path,
            expected_archive_path=source_archive_path,
            expected_crop_run=resolved_source_crop,
            expected_shape=(source_total_rois, int(roi_shape[0]), int(roi_shape[1])),
            require_payload_sha256=not bool(dry_run),
        )
        cache_manifest_sha256 = _sha256_file(cache_manifest_path)
        raw_cache_array = cache_manifest.get("array")
        if not isinstance(raw_cache_array, Mapping):
            raise ValueError("Flat ROI cache manifest is missing array provenance.")
        cache_payload_sha256 = str(raw_cache_array.get("sha256") or "")

    clipped_mapping = None
    if cache_manifest_path is None and video_path is None:
        clipped_mapping = _load_clipped_source_frame_mapping(
            root=source_root,
            archive_path=source_archive_path,
            crop_frame_indices=frame_indices,
            mode=source_frame_index_mode,
        )

    if cache_manifest_path is not None:
        resolved_video_path = None
        video_shape = _resolve_video_shape(source_root, source_group, None)
        if "source_acquisition_frame_index" in source_group:
            source_frame_indices = np.asarray(
                source_group["source_acquisition_frame_index"][:], dtype=np.int64
            ).reshape(-1)[source_row_ids]
        else:
            source_frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if source_frame_indices.shape[0] != total_rois:
            raise ValueError(
                "Source acquisition-frame identity length does not match crop rows."
            )
        frame_mapping = {
            "mode": "source_crop_acquisition_identity",
            "source_array": (
                "source_acquisition_frame_index"
                if "source_acquisition_frame_index" in source_group
                else "frame_indices"
            ),
            "source_archive_path": str(source_archive_path),
        }
        frame_to_rows = _frame_to_roi_indices(source_frame_indices)
        max_frame = int(max(frame_to_rows)) if frame_to_rows else -1
        decode_mode_effective = "flat_roi_cache"
    elif clipped_mapping is not None:
        resolved_video_path = None
        first_video = Path(str(clipped_mapping["video_paths"][0]))
        video_shape = _resolve_video_shape(source_root, source_group, first_video)
        source_frame_indices = np.asarray(clipped_mapping["source_frame_indices"], dtype=np.int64)
        frame_mapping = {
            "mode": "source_frame_index_parquet",
            "source_frame_index_path": clipped_mapping["source_frame_index_path"],
            "video_path_count": len(clipped_mapping["video_paths"]),
            "video_paths_preview": clipped_mapping["video_paths"][:5],
        }
        frame_to_rows: dict[int, list[int]] = {}
        max_frame = max(
            (
                max(frame_rows)
                for frame_rows in clipped_mapping["video_frame_to_rows"].values()
                if frame_rows
            ),
            default=-1,
        )
        decode_mode_effective = "indexed" if decode_mode == "auto" else str(decode_mode)
        if decode_mode_effective == "sequential":
            raise ValueError(
                "Clipped source_frame_index_parquet decoding currently requires indexed decode mode."
            )
    else:
        resolved_video_path = _resolve_video_path(
            source_root, source_group, video_path, source_archive_path
        )
        video_shape = _resolve_video_shape(
            source_root, source_group, resolved_video_path
        )
        source_frame_indices, frame_mapping = _map_source_frame_indices(
            root=source_root,
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
    contract = (
        dict(cache_pixel_contract)
        if cache_pixel_contract is not None
        else orange_mono_pynvvc_luma_pixel_contract()
    )
    pixel_contract_name = (
        str(cache_pixel_contract_name)
        if cache_pixel_contract_name is not None
        else ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
    )
    layout = build_canonical_crop_roi_layout(
        total_rois=total_rois,
        preferred_chunk_len=int(roi_chunk_len),
        roi_storage="compressed",
    )

    plan: dict[str, Any] = {
        "status": "dry_run" if dry_run else "planned",
        "zarr_path": str(archive_path),
        "source_zarr_path": str(source_archive_path),
        "source_crop_run": str(resolved_source_crop),
        "target_crop_run": str(resolved_target_crop),
        "video_path": str(resolved_video_path) if resolved_video_path is not None else None,
        "source_video_paths": clipped_mapping["video_paths"] if clipped_mapping is not None else None,
        "video_shape": [int(video_shape[0]), int(video_shape[1])],
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "total_rois": int(total_rois),
        "source_total_rois": int(source_total_rois),
        "source_row_selection": {
            "mode": (
                "instance_key_subset" if requested_instance_keys is not None else "all"
            ),
            "row_count": int(total_rois),
            "source_row_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(source_row_ids).view(np.uint8)
            ).hexdigest(),
            "requested_instance_keys": requested_instance_keys,
        },
        "source_frame_index_mapping": frame_mapping,
        "source_frame_min": int(source_frame_indices.min()) if source_frame_indices.size else None,
        "source_frame_max": int(source_frame_indices.max()) if source_frame_indices.size else None,
        "decode_mode_requested": str(decode_mode),
        "decode_mode_effective": str(decode_mode_effective),
        "decode_chunk_frames": int(decode_chunk_frames),
        "roi_chunk_len": int(layout.roi_chunk_len),
        "pixel_contract": contract,
        "pixel_contract_name": pixel_contract_name,
        "materialization_provider": materialization_provider,
        "materialization_provider_contract": list(
            TRAINING_CROP_MATERIALIZATION_PROVIDERS
        ),
        "roi_cache": (
            {
                "manifest_path": str(cache_manifest_path),
                "manifest_sha256": cache_manifest_sha256,
                "payload_sha256": cache_payload_sha256,
                "payload_verification": (
                    "direct_sha256_before_use_v1" if not dry_run else "deferred_apply"
                ),
            }
            if cache_manifest_path is not None
            else None
        ),
        "source_crop_manifest": _source_crop_manifest_binding(source_group),
        "set_latest": bool(set_latest),
        "consolidate_metadata": bool(consolidate_metadata),
    }
    if dry_run:
        if cache_array is not None:
            cache_array.close()
        return plan

    if crop_parent is None:
        crop_parent = root.require_group("crop_runs")
    target_group = crop_parent.create_group(resolved_target_crop, overwrite=bool(overwrite))
    if cache_manifest_path is None and source_archive_path == archive_path:
        target_group.attrs.update(dict(source_group.attrs))
    else:
        # Source crop-v2 manifests and coordinate proofs bind the source archive
        # and exact source path.  Copying those attrs to another archive would
        # create a convincing but invalid authority.  The training surface keeps
        # an explicit source binding and is not relabelled as crop-v2.
        for key in (
            "crop_signature",
            "crop_revision",
            "roi_size",
            "width",
            "height",
            "source_video_path",
            "video_source_path",
            "source_pixels",
            "source_pixel_contract",
        ):
            if key in source_group.attrs:
                target_group.attrs[key] = source_group.attrs[key]
    target_attrs: dict[str, Any] = {
        "status": "running",
        "created_at_utc": _utc_now(),
        "generated_by": MODULE_NAME,
        "training_materialization_schema": TRAINING_CROP_MATERIALIZATION_SCHEMA,
        "training_materialization_provider": materialization_provider,
        "training_materialization_provider_contract": list(
            TRAINING_CROP_MATERIALIZATION_PROVIDERS
        ),
        "stage_selector_eligible": False,
        "source_crop_run": str(resolved_source_crop),
        "source_crop_path": f"crop_runs/{resolved_source_crop}",
        "source_crop_archive_path": str(source_archive_path),
        "source_crop_manifest_binding": _source_crop_manifest_binding(source_group),
        "crop_storage_mode": "materialized",
        "roi_size": [int(roi_shape[0]), int(roi_shape[1])],
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
        "container_color_range_handling": contract.get(
            "container_color_range_handling"
        ),
        "center_rounding": CENTER_ROUNDING_NP_ROUND,
        "decode_mode_requested": str(decode_mode),
        "decode_mode_effective": str(decode_mode_effective),
        "crop_pixel_migration_version": "training_orange_mono_pynvvc_luma_v1",
        "roi_image_representation": contract.get("image_representation"),
        "roi_pixel_contract": contract,
        "roi_pixel_contract_name": pixel_contract_name,
        "source_frame_index_mapping": frame_mapping,
        "source_frame_index_mode_requested": str(source_frame_index_mode),
        "source_frame_indices_min": int(source_frame_indices.min())
        if source_frame_indices.size
        else None,
        "source_frame_indices_max": int(source_frame_indices.max())
        if source_frame_indices.size
        else None,
    }
    if resolved_video_path is not None:
        target_attrs["source_video_path"] = str(resolved_video_path)
    elif clipped_mapping is not None:
        target_attrs["source_video_path"] = "multiple_clips"
    elif cache_manifest is not None:
        cache_source = cache_manifest.get("source")
        if isinstance(cache_source, Mapping):
            cache_frame_source_path = _valid_attr_text(
                cache_source.get("frame_source_path")
            )
            if cache_frame_source_path is not None:
                target_attrs["source_video_path"] = cache_frame_source_path
    target_group.attrs.update(target_attrs)
    if cache_manifest_path is not None:
        target_group.attrs.update(
            {
                "coordinate_contract": "training_materialized_from_crop_v2_v1",
                "source_roi_cache_manifest": str(cache_manifest_path),
                "source_roi_cache_manifest_sha256": cache_manifest_sha256,
                "source_roi_cache_payload_sha256": cache_payload_sha256,
                "source_roi_cache_verified": True,
                "source_roi_cache_backend": "flat_bin_v1",
                "source_roi_cache_independence": (
                    "roi_images_copied_into_training_zarr_no_runtime_cache_dependency"
                ),
            }
        )
    if clipped_mapping is not None:
        target_group.attrs["source_video_paths"] = clipped_mapping["video_paths"]
        target_group.attrs["source_frame_index_path"] = clipped_mapping[
            "source_frame_index_path"
        ]
        target_group.attrs["source_layout"] = "rolling_clips"
    target_group.attrs.update(crop_roi_layout_attrs(layout))

    copied_arrays = _copy_crop_arrays(
        source_group,
        target_group,
        source_row_ids=source_row_ids,
        source_row_count=source_total_rois,
        selected_frame_indices=frame_indices,
    )
    target_group.create_array(
        "source_crop_row_ids",
        data=source_row_ids,
        chunks=(max(1, min(4096, total_rois)),),
        overwrite=True,
    )
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
        "cache_read_seconds": 0.0,
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
                _write_frame_crops(
                    frame_tensor, requested_by_frame[int(decoded_frame_index)]
                )

    try:
        if cache_array is not None:
            for start in range(0, total_rois, int(cache_copy_batch_rows)):
                end = min(start + int(cache_copy_batch_rows), total_rois)
                read_started = time.perf_counter()
                batch = np.ascontiguousarray(
                    cache_array[source_row_ids[start:end]], dtype=np.uint8
                )
                timing["cache_read_seconds"] += float(
                    time.perf_counter() - read_started
                )
                if batch.shape != (end - start, int(roi_shape[0]), int(roi_shape[1])):
                    raise ValueError(
                        "Flat ROI cache returned a batch with an unexpected shape: "
                        f"{batch.shape}."
                    )
                write_started = time.perf_counter()
                roi_images[start:end] = batch
                timing["write_seconds"] += float(time.perf_counter() - write_started)
                rows_written_mask[start:end] = True
                timing["rows_written"] = int(end)
        elif clipped_mapping is not None:
            expected_height, expected_width = int(video_shape[0]), int(video_shape[1])
            for clip_video_path, clip_frame_to_rows in clipped_mapping[
                "video_frame_to_rows"
            ].items():
                video_open_started = time.perf_counter()
                reader = _open_pynvvc_luma_indexed_decoder(Path(clip_video_path))
                timing["video_open_seconds"] += float(
                    time.perf_counter() - video_open_started
                )
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
            "pixel_contract_name": pixel_contract_name,
            "materialization_provider": materialization_provider,
        }
        target_group.attrs["timing"] = _json_safe(timing)
        missing_binding_arrays = sorted(
            set(CROP_GEOMETRY_SCHEMA_V1.binding_paths) - set(target_group.array_keys())
        )
        if missing_binding_arrays:
            target_group.attrs["training_crop_materialization_binding_status"] = (
                "legacy_source_missing_crop_v2_identity"
            )
            target_group.attrs["training_crop_materialization_binding_missing_arrays"] = (
                missing_binding_arrays
            )
        else:
            target_group.attrs[TRAINING_CROP_MATERIALIZATION_BINDING_ATTRIBUTE] = (
                build_training_crop_materialization_binding(target_group)
            )
            target_group.attrs["training_crop_materialization_binding_status"] = (
                "strict_v1"
            )
        if set_latest:
            _set_latest_pointers(crop_parent, resolved_target_crop)
        if consolidate_metadata:
            consolidate_started = time.perf_counter()
            consolidate_metadata_capture_expected_warnings(archive_path)
            timing["consolidate_metadata_seconds"] = float(time.perf_counter() - consolidate_started)

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
        if cache_array is not None:
            cache_array.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a training crop run's roi_images from PyNvVideoCodec "
            "luma or a verified flat ROI cache."
        )
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument(
        "--source-zarr-path",
        type=Path,
        help=(
            "Optional external analysis Zarr that owns the exact source crop run. "
            "Use either its source video or an exactly bound flat ROI cache."
        ),
    )
    parser.add_argument("--source-crop-run", help="Existing crop_runs/<run> to copy geometry from.")
    parser.add_argument("--target-crop-run", help="New crop_runs/<run> name to create.")
    parser.add_argument("--video-path", type=Path, help="Source MP4 path. Overrides zarr metadata.")
    parser.add_argument(
        "--roi-cache-manifest",
        type=Path,
        help=(
            "Verified flat-bin ROI cache to copy into the training crop. This is "
            "mutually exclusive with --video-path."
        ),
    )
    parser.add_argument(
        "--cache-copy-batch-rows",
        type=int,
        default=1024,
        help="Rows per cache-to-Zarr copy batch (default: 1024).",
    )
    parser.add_argument(
        "--source-instance-keys",
        help=(
            "Optional comma-separated stable crop instance_key selection. "
            "Multiple keys from the same frame remain distinct rows."
        ),
    )
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
        source_zarr_path=args.source_zarr_path,
        source_crop_run=args.source_crop_run,
        target_crop_run=args.target_crop_run,
        video_path=args.video_path,
        roi_cache_manifest=args.roi_cache_manifest,
        cache_copy_batch_rows=args.cache_copy_batch_rows,
        source_instance_keys=_parse_instance_keys(args.source_instance_keys),
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
            f"pixel_contract: {report['pixel_contract_name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
