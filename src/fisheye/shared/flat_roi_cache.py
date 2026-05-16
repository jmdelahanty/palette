"""Flat binary ROI cache helpers.

The cache format is intentionally simple: one C-contiguous binary payload plus
one JSON manifest. It is a workflow/runtime cache, not a canonical Zarr surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.roi_pixel_contract import flat_cache_pixel_contract_for_backend

FLAT_ROI_CACHE_SCHEMA = "palette_roi_cache_flat_bin_v1"
FLAT_ROI_CACHE_LAYOUT = "flat_bin_v1"
FLAT_ROI_CACHE_PROGRESS_SCHEMA = "palette_roi_cache_flat_bin_progress_v1"
FLAT_ROI_CACHE_DECODE_BACKENDS = ("auto", "read_slice", "pynvvc_luma")


class FlatRoiCacheArray:
    """Read-only array-like wrapper around a flat ROI binary payload."""

    def __init__(self, *, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.manifest = dict(manifest)
        array = _require_mapping(self.manifest, "array")
        shape = array.get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError("Flat ROI cache manifest array.shape must be [N, H, W].")
        self.shape = tuple(int(v) for v in shape)
        self.ndim = 3
        dtype_name = str(array.get("dtype") or "")
        if dtype_name != "uint8":
            raise ValueError(f"Unsupported flat ROI cache dtype '{dtype_name}'. Expected uint8.")
        self.dtype = np.dtype(np.uint8)
        order = str(array.get("order") or "C")
        if order != "C":
            raise ValueError(f"Unsupported flat ROI cache order '{order}'. Expected C.")
        bin_path = _resolve_payload_path(self.manifest_path, str(array.get("bin_path") or ""))
        expected_bytes = int(np.prod(self.shape, dtype=np.int64)) * int(self.dtype.itemsize)
        if not bin_path.exists():
            raise FileNotFoundError(f"Flat ROI cache payload not found: {bin_path}")
        actual_bytes = bin_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Flat ROI cache payload size mismatch: expected {expected_bytes} bytes, got {actual_bytes}."
            )
        self.bin_path = bin_path
        self._memmap = np.memmap(bin_path, dtype=self.dtype, mode="r", shape=self.shape, order="C")

    def __getitem__(self, key):  # noqa: ANN001
        return self._memmap[key]

    def close(self) -> None:
        mmap = getattr(self._memmap, "_mmap", None)
        if mmap is not None:
            mmap.close()


def load_flat_roi_cache_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != FLAT_ROI_CACHE_SCHEMA:
        raise ValueError(
            f"Unsupported ROI cache schema {manifest.get('schema')!r}; expected {FLAT_ROI_CACHE_SCHEMA!r}."
        )
    if manifest.get("layout") != FLAT_ROI_CACHE_LAYOUT:
        raise ValueError(
            f"Unsupported ROI cache layout {manifest.get('layout')!r}; expected {FLAT_ROI_CACHE_LAYOUT!r}."
        )
    if not bool(manifest.get("cache_complete")):
        raise ValueError(f"Flat ROI cache manifest is not marked complete: {manifest_path}")
    return manifest


def open_flat_roi_cache(
    manifest_path: str | Path,
    *,
    expected_archive_path: str | Path | None = None,
    expected_crop_run: str | None = None,
    expected_shape: Sequence[int] | None = None,
) -> FlatRoiCacheArray:
    path = Path(manifest_path).expanduser()
    manifest = load_flat_roi_cache_manifest(path)
    _validate_manifest_against_expected(
        manifest,
        expected_archive_path=expected_archive_path,
        expected_crop_run=expected_crop_run,
        expected_shape=expected_shape,
    )
    return FlatRoiCacheArray(manifest_path=path, manifest=manifest)


def crop_run_name_from_manifest(manifest_path: str | Path) -> str:
    manifest = load_flat_roi_cache_manifest(manifest_path)
    source = _require_mapping(manifest, "source")
    crop_run = str(source.get("crop_run_name") or "")
    if not crop_run:
        raise ValueError("Flat ROI cache manifest is missing source.crop_run_name.")
    return crop_run


def build_flat_roi_cache(
    *,
    zarr_path: str | Path,
    crop_run: str | None = None,
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    batch_size: int = 1024,
    overwrite: bool = False,
    compute_sha256: bool = False,
    roi_live_acceleration: str = "auto",
    roi_live_gpu_chunk_frames: int = 32,
    roi_decode_backend: str = "auto",
    console: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_seconds: float = 30.0,
    progress_every_batches: int = 0,
) -> dict[str, Any]:
    """Materialize the selected crop run into a flat binary ROI cache."""

    archive_path = Path(zarr_path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {archive_path}")
    if manifest_path is None and output_dir is None:
        raise ValueError("Provide either output_dir or manifest_path.")
    decode_backend_requested = _normalize_decode_backend(roi_decode_backend)

    root = zarr.open(str(archive_path), mode="r", use_consolidated=False)

    # Import lazily to avoid a circular import at module load time.
    from fisheye.shared.crop_image_source import CropImageSource

    source = CropImageSource.open(
        root,
        crop_run=crop_run,
        zarr_path=archive_path,
        roi_cache_policy="never",
        roi_live_acceleration=roi_live_acceleration,
        roi_live_gpu_chunk_frames=roi_live_gpu_chunk_frames,
        console=console,
    )
    try:
        cache_key = source._build_roi_cache_key(archive_path)  # Internal shared cache identity.
        resolved_manifest_path = _default_manifest_path(
            archive_path=archive_path,
            crop_run_name=source.crop_run_name,
            cache_key=cache_key,
            output_dir=output_dir,
            manifest_path=manifest_path,
        )
        resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if resolved_manifest_path.exists() and not overwrite:
            manifest = load_flat_roi_cache_manifest(resolved_manifest_path)
            _validate_manifest_against_expected(
                manifest,
                expected_archive_path=archive_path,
                expected_crop_run=source.crop_run_name,
                expected_shape=source.shape,
            )
            manifest.setdefault("manifest_path", str(resolved_manifest_path))
            _emit_progress(
                progress_callback,
                "reuse_existing",
                manifest_path=str(resolved_manifest_path),
                bin_path=str(resolved_manifest_path.with_suffix(".bin")),
                source=_progress_source_payload(source),
            )
            return manifest

        manifest_tmp = _temporary_path(resolved_manifest_path, suffix=".tmp.json")
        bin_path = resolved_manifest_path.with_suffix(".bin")
        bin_tmp = _temporary_path(bin_path, suffix=".tmp.bin")
        total_bytes = int(np.prod(source.shape, dtype=np.int64))
        total_rois = int(source.total_rois)
        effective_batch_size = max(1, int(batch_size))
        started = time.perf_counter()
        _emit_progress(
            progress_callback,
            "start",
            manifest_path=str(resolved_manifest_path),
            bin_path=str(bin_path),
            tmp_bin_path=str(bin_tmp),
            batch_size=effective_batch_size,
            total_rois=total_rois,
            total_bytes=total_bytes,
            decode_backend_requested=decode_backend_requested,
            source=_progress_source_payload(source),
        )

        try:
            timing, payload_sha256, decode_backend_effective = _write_flat_cache_payload(
                source=source,
                bin_tmp=bin_tmp,
                total_bytes=total_bytes,
                batch_size=effective_batch_size,
                compute_sha256=bool(compute_sha256),
                decode_backend_requested=decode_backend_requested,
                progress_callback=progress_callback,
                progress_interval_seconds=float(progress_interval_seconds),
                progress_every_batches=int(progress_every_batches),
                started=started,
            )
        except Exception:
            try:
                bin_tmp.unlink()
            except FileNotFoundError:
                pass
            raise

        os.replace(bin_tmp, bin_path)
        duration_seconds = float(time.perf_counter() - started)
        timing_summary = _timing_summary(
            timing=timing,
            total_rois=total_rois,
            total_bytes=total_bytes,
            duration_seconds=duration_seconds,
        )
        manifest = _build_manifest(
            source=source,
            archive_path=archive_path,
            manifest_path=resolved_manifest_path,
            bin_path=bin_path,
            cache_key=cache_key,
            batch_size=effective_batch_size,
            duration_seconds=duration_seconds,
            payload_sha256=payload_sha256,
            total_bytes=total_bytes,
            timing_summary=timing_summary,
            decode_backend_requested=decode_backend_requested,
            decode_backend_effective=decode_backend_effective,
        )
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, resolved_manifest_path)
        _emit_progress(
            progress_callback,
            "complete",
            manifest_path=str(resolved_manifest_path),
            bin_path=str(bin_path),
            decode_backend_effective=decode_backend_effective,
            source=_progress_source_payload(source),
            progress=_progress_totals(
                rows_written=int(timing["rows"]),
                bytes_written=int(timing["bytes"]),
                total_rois=total_rois,
                total_bytes=total_bytes,
                elapsed_seconds=duration_seconds,
                timing=timing,
            ),
            timing=timing_summary,
        )
        return manifest
    finally:
        source.close()


def _build_manifest(
    *,
    source: Any,
    archive_path: Path,
    manifest_path: Path,
    bin_path: Path,
    cache_key: str,
    batch_size: int,
    duration_seconds: float,
    payload_sha256: str | None,
    total_bytes: int,
    timing_summary: Mapping[str, Any],
    decode_backend_requested: str,
    decode_backend_effective: str,
) -> dict[str, Any]:
    roi_h, roi_w = source.roi_shape
    return {
        "schema": FLAT_ROI_CACHE_SCHEMA,
        "layout": FLAT_ROI_CACHE_LAYOUT,
        "cache_complete": True,
        "cache_key": cache_key,
        "manifest_path": str(manifest_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive_path": str(archive_path),
            "crop_run_name": source.crop_run_name,
            "source_crop_storage_mode": source.storage_mode,
            "crop_signature": source.crop_group.attrs.get("crop_signature"),
            "crop_revision": source.crop_group.attrs.get("crop_revision"),
            "frame_source_kind": source.frame_source_kind,
            "frame_source_path": source.frame_source_path,
            "frame_source_identity": source._build_frame_source_identity(),
        },
        "array": {
            "bin_path": _relative_or_absolute(bin_path, base=manifest_path.parent),
            "dtype": "uint8",
            "shape": [int(source.total_rois), int(roi_h), int(roi_w)],
            "order": "C",
            "row_stride_bytes": int(roi_h) * int(roi_w),
            "total_bytes": int(total_bytes),
            "sha256": payload_sha256,
        },
        "builder": {
            "batch_size": int(batch_size),
            "duration_seconds": float(duration_seconds),
            "decode_backend_requested": str(decode_backend_requested),
            "decode_backend_effective": str(decode_backend_effective),
            "pixel_contract": flat_cache_pixel_contract_for_backend(decode_backend_effective),
            "timing": dict(timing_summary),
            "format_note": (
                "flat_bin_v1 stores all ROI rows contiguously as raw uint8 bytes. "
                "It is optimized for simple sequential reads and cheap transfer, "
                "not chunked concurrent writes."
            ),
        },
    }

def _write_flat_cache_payload(
    *,
    source: Any,
    bin_tmp: Path,
    total_bytes: int,
    batch_size: int,
    compute_sha256: bool,
    decode_backend_requested: str,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_interval_seconds: float,
    progress_every_batches: int,
    started: float,
) -> tuple[dict[str, Any], str | None, str]:
    if decode_backend_requested in {"auto", "pynvvc_luma"} and _source_supports_pynvvc_luma(source):
        try:
            return _write_flat_cache_payload_pynvvc_luma(
                source=source,
                bin_tmp=bin_tmp,
                total_bytes=total_bytes,
                batch_size=batch_size,
                compute_sha256=compute_sha256,
                progress_callback=progress_callback,
                progress_interval_seconds=progress_interval_seconds,
                progress_every_batches=progress_every_batches,
                started=started,
            )
        except Exception as exc:
            if decode_backend_requested == "pynvvc_luma":
                raise
            _emit_progress(
                progress_callback,
                "backend_fallback",
                requested_backend=decode_backend_requested,
                failed_backend="pynvvc_luma",
                fallback_backend="read_slice",
                error={"type": exc.__class__.__name__, "message": str(exc)},
            )
            try:
                bin_tmp.unlink()
            except FileNotFoundError:
                pass
    elif decode_backend_requested == "pynvvc_luma":
        raise RuntimeError(
            "pynvvc_luma flat ROI cache backend requires a geometry-only crop run "
            "with source_video_path and known frame dimensions."
        )

    return _write_flat_cache_payload_read_slice(
        source=source,
        bin_tmp=bin_tmp,
        total_bytes=total_bytes,
        batch_size=batch_size,
        compute_sha256=compute_sha256,
        progress_callback=progress_callback,
        progress_interval_seconds=progress_interval_seconds,
        progress_every_batches=progress_every_batches,
        started=started,
    )


def _write_flat_cache_payload_read_slice(
    *,
    source: Any,
    bin_tmp: Path,
    total_bytes: int,
    batch_size: int,
    compute_sha256: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_interval_seconds: float,
    progress_every_batches: int,
    started: float,
) -> tuple[dict[str, Any], str | None, str]:
    total_rois = int(source.total_rois)
    timing = _new_timing()
    last_progress = started
    sha = hashlib.sha256() if compute_sha256 else None
    with bin_tmp.open("wb") as handle:
        for batch_index, start in enumerate(range(0, total_rois, batch_size), start=1):
            end = min(start + batch_size, total_rois)
            read_started = time.perf_counter()
            raw_batch = source.read_slice(start, end)
            read_seconds = time.perf_counter() - read_started

            contiguous_started = time.perf_counter()
            batch = np.ascontiguousarray(raw_batch, dtype=np.uint8)
            contiguous_seconds = time.perf_counter() - contiguous_started
            if tuple(batch.shape[1:]) != tuple(source.roi_shape):
                raise ValueError(
                    f"ROI batch shape mismatch: expected (*, {source.roi_shape[0]}, {source.roi_shape[1]}), "
                    f"got {tuple(batch.shape)}."
                )

            serialize_started = time.perf_counter()
            payload = batch.tobytes(order="C")
            serialize_seconds = time.perf_counter() - serialize_started

            write_started = time.perf_counter()
            handle.write(payload)
            write_seconds = time.perf_counter() - write_started

            sha_seconds = 0.0
            if sha is not None:
                sha_started = time.perf_counter()
                sha.update(payload)
                sha_seconds = time.perf_counter() - sha_started

            batch_rows = int(end - start)
            batch_bytes = int(len(payload))
            _record_timing_batch(
                timing,
                batch_rows=batch_rows,
                batch_bytes=batch_bytes,
                read_seconds=read_seconds,
                contiguous_seconds=contiguous_seconds,
                serialize_seconds=serialize_seconds,
                write_seconds=write_seconds,
                sha_seconds=sha_seconds,
                batch_index=batch_index,
            )
            last_progress = _maybe_emit_batch_progress(
                progress_callback=progress_callback,
                batch_index=batch_index,
                batch_payload={
                    "index": int(batch_index),
                    "start": int(start),
                    "end": int(end),
                    "rows": batch_rows,
                    "bytes": batch_bytes,
                    "read_seconds": float(read_seconds),
                    "contiguous_seconds": float(contiguous_seconds),
                    "serialize_seconds": float(serialize_seconds),
                    "write_seconds": float(write_seconds),
                    "sha256_seconds": float(sha_seconds),
                    "read_rows_per_second": _rate(batch_rows, read_seconds),
                    "write_mib_per_second": _rate(batch_bytes / (1024 * 1024), write_seconds),
                    "decode_backend": "read_slice",
                },
                timing=timing,
                total_rois=total_rois,
                total_bytes=total_bytes,
                started=started,
                last_progress=last_progress,
                progress_interval_seconds=progress_interval_seconds,
                progress_every_batches=progress_every_batches,
            )
    return timing, sha.hexdigest() if sha is not None else None, "read_slice"


def _write_flat_cache_payload_pynvvc_luma(
    *,
    source: Any,
    bin_tmp: Path,
    total_bytes: int,
    batch_size: int,
    compute_sha256: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_interval_seconds: float,
    progress_every_batches: int,
    started: float,
) -> tuple[dict[str, Any], str | None, str]:
    import torch

    reader = _open_pynvvc_luma_reader(Path(str(source.frame_source_path)))
    try:
        source_height = int(reader.source_height)
        source_width = int(reader.source_width)
        if source.frame_shape is not None:
            expected_h, expected_w = int(source.frame_shape[0]), int(source.frame_shape[1])
            if expected_h != source_height or expected_w != source_width:
                raise ValueError(
                    "PyNvVideoCodec source dimensions do not match crop metadata: "
                    f"decoder={source_width}x{source_height}, metadata={expected_w}x{expected_h}."
                )

        frame_to_roi = _frame_to_roi_indices(source.frame_indices)
        max_frame = int(max(frame_to_roi)) if frame_to_roi else -1
        row_stride = int(source.roi_shape[0]) * int(source.roi_shape[1])
        timing = _new_timing()
        last_progress = started
        rows_written_mask = np.zeros(int(source.total_rois), dtype=bool)
        batch_index = 0
        decoded_frame_index = 0
        read_batch_frames = max(1, int(getattr(source, "roi_live_gpu_chunk_frames", 32) or 32))

        with bin_tmp.open("w+b") as handle:
            handle.truncate(int(total_bytes))
            while decoded_frame_index <= max_frame:
                decode_count = min(read_batch_frames, max_frame - decoded_frame_index + 1)
                read_started = time.perf_counter()
                frames = reader.decode_next(decode_count)
                read_seconds = time.perf_counter() - read_started
                if not frames:
                    break

                decoded_frame_ids = [decoded_frame_index + offset for offset in range(len(frames))]
                skipped_batch_frames = sum(1 for frame_idx in decoded_frame_ids if frame_idx not in frame_to_roi)
                decode_accounted = False
                for frame_offset, frame_tensor in enumerate(frames):
                    frame_idx = decoded_frame_index + frame_offset
                    roi_ids = frame_to_roi.get(frame_idx)
                    if not roi_ids:
                        continue

                    batch_index += 1
                    crop_started = time.perf_counter()
                    crops = _crop_pynvvc_luma_frame(
                        frame_tensor,
                        roi_ids=roi_ids,
                        roi_coordinates_full=source.roi_coordinates_full,
                        roi_shape=source.roi_shape,
                        video_shape=(source_height, source_width),
                    )
                    if torch.cuda.is_available() and getattr(crops, "is_cuda", False):
                        torch.cuda.synchronize()
                    contiguous_started = time.perf_counter()
                    crops_cpu = np.ascontiguousarray(crops.cpu().numpy(), dtype=np.uint8)
                    contiguous_seconds = time.perf_counter() - contiguous_started
                    crop_seconds = time.perf_counter() - crop_started - contiguous_seconds

                    serialize_started = time.perf_counter()
                    payloads = [
                        np.ascontiguousarray(crops_cpu[local_idx], dtype=np.uint8).tobytes(order="C")
                        for local_idx in range(crops_cpu.shape[0])
                    ]
                    serialize_seconds = time.perf_counter() - serialize_started

                    write_started = time.perf_counter()
                    for roi_id, payload in zip(roi_ids, payloads):
                        handle.seek(int(roi_id) * row_stride)
                        handle.write(payload)
                        rows_written_mask[int(roi_id)] = True
                    write_seconds = time.perf_counter() - write_started

                    batch_rows = int(len(roi_ids))
                    batch_bytes = int(batch_rows * row_stride)
                    account_decode = not decode_accounted
                    decode_accounted = True
                    _record_timing_batch(
                        timing,
                        batch_rows=batch_rows,
                        batch_bytes=batch_bytes,
                        read_seconds=read_seconds if account_decode else 0.0,
                        contiguous_seconds=contiguous_seconds,
                        serialize_seconds=serialize_seconds,
                        write_seconds=write_seconds,
                        sha_seconds=0.0,
                        batch_index=batch_index,
                        decode_seconds=read_seconds if account_decode else 0.0,
                        crop_seconds=crop_seconds,
                        decoded_frames=len(frames) if account_decode else 0,
                        skipped_frames=skipped_batch_frames if account_decode else 0,
                    )
                    last_progress = _maybe_emit_batch_progress(
                        progress_callback=progress_callback,
                        batch_index=batch_index,
                        batch_payload={
                            "index": int(batch_index),
                            "frame_index": int(frame_idx),
                            "rows": batch_rows,
                            "bytes": batch_bytes,
                            "read_seconds": float(read_seconds if account_decode else 0.0),
                            "decode_seconds": float(read_seconds if account_decode else 0.0),
                            "crop_seconds": float(crop_seconds),
                            "contiguous_seconds": float(contiguous_seconds),
                            "serialize_seconds": float(serialize_seconds),
                            "write_seconds": float(write_seconds),
                            "sha256_seconds": 0.0,
                            "read_rows_per_second": _rate(batch_rows, crop_seconds + contiguous_seconds),
                            "write_mib_per_second": _rate(batch_bytes / (1024 * 1024), write_seconds),
                            "decode_backend": "pynvvc_luma",
                        },
                        timing=timing,
                        total_rois=int(source.total_rois),
                        total_bytes=total_bytes,
                        started=started,
                        last_progress=last_progress,
                        progress_interval_seconds=progress_interval_seconds,
                        progress_every_batches=progress_every_batches,
                    )
                    del crops, crops_cpu, payloads

                if not decode_accounted:
                    timing["read_seconds_total"] += float(read_seconds)
                    timing["decode_seconds_total"] += float(read_seconds)
                    timing["decoded_frames"] += int(len(frames))
                    timing["skipped_frames"] += int(len(frames))

                decoded_frame_index += len(frames)
                del frames

        rows_written = int(rows_written_mask.sum())
        if rows_written != int(source.total_rois):
            missing = int(source.total_rois) - rows_written
            raise RuntimeError(
                f"PyNvVideoCodec flat ROI cache wrote {rows_written}/{source.total_rois} rows; "
                f"{missing} rows were not produced before decoder EOF."
            )

        payload_sha256 = _sha256_file(bin_tmp) if compute_sha256 else None
        return timing, payload_sha256, "pynvvc_luma"
    finally:
        reader.close()


def _source_supports_pynvvc_luma(source: Any) -> bool:
    return (
        str(getattr(source, "storage_mode", "")) == "geometry_only"
        and str(getattr(source, "frame_source_kind", "")) == "source_video_path"
        and bool(getattr(source, "frame_source_path", None))
        and getattr(source, "frame_shape", None) is not None
    )


def _open_pynvvc_luma_reader(video_path: Path) -> Any:
    from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader

    return PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)


def _frame_to_roi_indices(frame_indices: np.ndarray) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for roi_idx, frame_idx in enumerate(np.asarray(frame_indices, dtype=np.int64).reshape(-1)):
        mapping.setdefault(int(frame_idx), []).append(int(roi_idx))
    return mapping


def _crop_pynvvc_luma_frame(
    frame_tensor: Any,
    *,
    roi_ids: Sequence[int],
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
) -> Any:
    import torch

    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    height, width = int(video_shape[0]), int(video_shape[1])
    y_plane = frame_tensor[:height, :width].contiguous()
    crops: list[Any] = []
    for roi_id in roi_ids:
        x1 = int(roi_coordinates_full[int(roi_id), 0])
        y1 = int(roi_coordinates_full[int(roi_id), 1])
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        if 0 <= x1 and x2 <= width and 0 <= y1 and y2 <= height:
            crops.append(y_plane[y1:y2, x1:x2].clone())
            continue

        roi = torch.zeros((roi_h, roi_w), dtype=torch.uint8, device=y_plane.device)
        vy1 = max(0, y1)
        vy2 = min(height, y2)
        vx1 = max(0, x1)
        vx2 = min(width, x2)
        if vy2 > vy1 and vx2 > vx1:
            py1 = max(0, -y1)
            px1 = max(0, -x1)
            py2 = py1 + (vy2 - vy1)
            px2 = px1 + (vx2 - vx1)
            roi[py1:py2, px1:px2] = y_plane[vy1:vy2, vx1:vx2]
        crops.append(roi)
    if not crops:
        return torch.empty((0, roi_h, roi_w), dtype=torch.uint8, device=y_plane.device)
    return torch.stack(crops, dim=0)


def _new_timing() -> dict[str, Any]:
    return {
        "read_seconds_total": 0.0,
        "decode_seconds_total": 0.0,
        "crop_seconds_total": 0.0,
        "contiguous_seconds_total": 0.0,
        "serialize_seconds_total": 0.0,
        "write_seconds_total": 0.0,
        "sha256_seconds_total": 0.0,
        "decoded_frames": 0,
        "skipped_frames": 0,
        "batches": 0,
        "rows": 0,
        "bytes": 0,
    }


def _record_timing_batch(
    timing: dict[str, Any],
    *,
    batch_rows: int,
    batch_bytes: int,
    read_seconds: float,
    contiguous_seconds: float,
    serialize_seconds: float,
    write_seconds: float,
    sha_seconds: float,
    batch_index: int,
    decode_seconds: float = 0.0,
    crop_seconds: float = 0.0,
    decoded_frames: int = 0,
    skipped_frames: int = 0,
) -> None:
    timing["read_seconds_total"] += float(read_seconds)
    timing["decode_seconds_total"] += float(decode_seconds)
    timing["crop_seconds_total"] += float(crop_seconds)
    timing["contiguous_seconds_total"] += float(contiguous_seconds)
    timing["serialize_seconds_total"] += float(serialize_seconds)
    timing["write_seconds_total"] += float(write_seconds)
    timing["sha256_seconds_total"] += float(sha_seconds)
    timing["decoded_frames"] += int(decoded_frames)
    timing["skipped_frames"] += int(skipped_frames)
    timing["batches"] = int(batch_index)
    timing["rows"] += int(batch_rows)
    timing["bytes"] += int(batch_bytes)


def _maybe_emit_batch_progress(
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    batch_index: int,
    batch_payload: dict[str, Any],
    timing: Mapping[str, Any],
    total_rois: int,
    total_bytes: int,
    started: float,
    last_progress: float,
    progress_interval_seconds: float,
    progress_every_batches: int,
) -> float:
    now = time.perf_counter()
    if not _should_emit_batch_progress(
        batch_index=batch_index,
        rows_written=int(timing["rows"]),
        total_rois=total_rois,
        now=now,
        last_progress=last_progress,
        progress_interval_seconds=progress_interval_seconds,
        progress_every_batches=progress_every_batches,
    ):
        return last_progress
    elapsed = now - started
    _emit_progress(
        progress_callback,
        "batch",
        batch=batch_payload,
        progress=_progress_totals(
            rows_written=int(timing["rows"]),
            bytes_written=int(timing["bytes"]),
            total_rois=total_rois,
            total_bytes=total_bytes,
            elapsed_seconds=elapsed,
            timing=timing,
        ),
    )
    return now


def _normalize_decode_backend(value: str) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in FLAT_ROI_CACHE_DECODE_BACKENDS:
        choices = ", ".join(FLAT_ROI_CACHE_DECODE_BACKENDS)
        raise ValueError(f"Invalid flat ROI cache decode backend '{value}'. Expected one of: {choices}")
    return backend


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    event: str,
    **payload: Any,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "schema": FLAT_ROI_CACHE_PROGRESS_SCHEMA,
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
    )


def _progress_source_payload(source: Any) -> dict[str, Any]:
    return {
        "crop_run_name": str(source.crop_run_name),
        "storage_mode": str(source.storage_mode),
        "roi_shape": [int(source.roi_shape[0]), int(source.roi_shape[1])],
        "total_rois": int(source.total_rois),
        "frame_source_kind": source.frame_source_kind,
        "frame_source_path": None if source.frame_source_path is None else str(source.frame_source_path),
        "roi_read_mode": getattr(source, "roi_read_mode", None),
        "roi_live_acceleration_requested": getattr(source, "roi_live_acceleration_requested", None),
        "roi_live_acceleration_effective": getattr(source, "roi_live_acceleration_effective", None),
        "roi_live_acceleration_fallback_reason": getattr(
            source, "roi_live_acceleration_fallback_reason", None
        ),
    }


def _should_emit_batch_progress(
    *,
    batch_index: int,
    rows_written: int,
    total_rois: int,
    now: float,
    last_progress: float,
    progress_interval_seconds: float,
    progress_every_batches: int,
) -> bool:
    if batch_index == 1 or rows_written >= total_rois:
        return True
    if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
        return True
    if progress_interval_seconds > 0 and now - last_progress >= progress_interval_seconds:
        return True
    return False


def _progress_totals(
    *,
    rows_written: int,
    bytes_written: int,
    total_rois: int,
    total_bytes: int,
    elapsed_seconds: float,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    remaining_rows = max(0, int(total_rois) - int(rows_written))
    rows_per_second = _rate(rows_written, elapsed_seconds)
    eta_seconds = None if rows_per_second <= 0 else remaining_rows / rows_per_second
    return {
        "rows_written": int(rows_written),
        "rows_total": int(total_rois),
        "rows_fraction": _fraction(rows_written, total_rois),
        "bytes_written": int(bytes_written),
        "bytes_total": int(total_bytes),
        "bytes_fraction": _fraction(bytes_written, total_bytes),
        "mib_written": float(bytes_written / (1024 * 1024)),
        "mib_total": float(total_bytes / (1024 * 1024)),
        "elapsed_seconds": float(elapsed_seconds),
        "eta_seconds": None if eta_seconds is None else float(eta_seconds),
        "rows_per_second": float(rows_per_second),
        "mib_per_second": float(_rate(bytes_written / (1024 * 1024), elapsed_seconds)),
        "read_rows_per_second": float(
            _rate(rows_written, float(timing.get("read_seconds_total", 0.0)))
        ),
        "write_mib_per_second": float(
            _rate(bytes_written / (1024 * 1024), float(timing.get("write_seconds_total", 0.0)))
        ),
        "batches": int(timing.get("batches", 0)),
    }


def _timing_summary(
    *,
    timing: Mapping[str, Any],
    total_rois: int,
    total_bytes: int,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "duration_seconds": float(duration_seconds),
        "batches": int(timing.get("batches", 0)),
        "rows": int(timing.get("rows", 0)),
        "bytes": int(timing.get("bytes", 0)),
        "read_seconds_total": float(timing.get("read_seconds_total", 0.0)),
        "contiguous_seconds_total": float(timing.get("contiguous_seconds_total", 0.0)),
        "serialize_seconds_total": float(timing.get("serialize_seconds_total", 0.0)),
        "write_seconds_total": float(timing.get("write_seconds_total", 0.0)),
        "sha256_seconds_total": float(timing.get("sha256_seconds_total", 0.0)),
        "decode_seconds_total": float(timing.get("decode_seconds_total", 0.0)),
        "crop_seconds_total": float(timing.get("crop_seconds_total", 0.0)),
        "decoded_frames": int(timing.get("decoded_frames", 0)),
        "skipped_frames": int(timing.get("skipped_frames", 0)),
        "rows_per_second": float(_rate(total_rois, duration_seconds)),
        "mib_per_second": float(_rate(total_bytes / (1024 * 1024), duration_seconds)),
        "read_rows_per_second": float(
            _rate(total_rois, float(timing.get("read_seconds_total", 0.0)))
        ),
        "write_mib_per_second": float(
            _rate(total_bytes / (1024 * 1024), float(timing.get("write_seconds_total", 0.0)))
        ),
    }


def _rate(numerator: float, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(numerator) / float(seconds)


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return float(numerator) / float(denominator)


def _default_manifest_path(
    *,
    archive_path: Path,
    crop_run_name: str,
    cache_key: str,
    output_dir: str | Path | None,
    manifest_path: str | Path | None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    assert output_dir is not None
    safe_archive = _safe_component(archive_path.stem)
    safe_crop = _safe_component(crop_run_name)
    return (
        Path(output_dir).expanduser().resolve()
        / f"{safe_archive}__{safe_crop}__{cache_key[:12]}.flat_roi_cache.json"
    )


def _temporary_path(path: Path, *, suffix: str) -> Path:
    token = f"{os.getpid()}_{uuid.uuid4().hex}"
    return path.with_name(f".{path.name}.{token}{suffix}")


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value)) or "cache"


def _relative_or_absolute(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_payload_path(manifest_path: Path, raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("Flat ROI cache manifest is missing array.bin_path.")
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Flat ROI cache manifest missing mapping '{key}'.")
    return value


def _validate_manifest_against_expected(
    manifest: Mapping[str, Any],
    *,
    expected_archive_path: str | Path | None,
    expected_crop_run: str | None,
    expected_shape: Sequence[int] | None,
) -> None:
    source = _require_mapping(manifest, "source")
    if expected_archive_path is not None:
        expected = str(Path(expected_archive_path).expanduser().resolve())
        actual = str(source.get("archive_path") or "")
        if actual and str(Path(actual).expanduser().resolve()) != expected:
            raise ValueError(
                f"Flat ROI cache archive mismatch: manifest has {actual!r}, expected {expected!r}."
            )
    if expected_crop_run is not None and source.get("crop_run_name") != expected_crop_run:
        raise ValueError(
            f"Flat ROI cache crop run mismatch: manifest has {source.get('crop_run_name')!r}, "
            f"expected {expected_crop_run!r}."
        )
    if expected_shape is not None:
        array = _require_mapping(manifest, "array")
        actual_shape = tuple(int(v) for v in array.get("shape") or [])
        expected_shape_tuple = tuple(int(v) for v in expected_shape)
        if actual_shape != expected_shape_tuple:
            raise ValueError(
                f"Flat ROI cache shape mismatch: manifest has {actual_shape}, expected {expected_shape_tuple}."
            )
