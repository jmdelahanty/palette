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
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

FLAT_ROI_CACHE_SCHEMA = "palette_roi_cache_flat_bin_v1"
FLAT_ROI_CACHE_LAYOUT = "flat_bin_v1"


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
    console: Any | None = None,
) -> dict[str, Any]:
    """Materialize the selected crop run into a flat binary ROI cache."""

    archive_path = Path(zarr_path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {archive_path}")
    if manifest_path is None and output_dir is None:
        raise ValueError("Provide either output_dir or manifest_path.")

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
            return manifest

        manifest_tmp = _temporary_path(resolved_manifest_path, suffix=".tmp.json")
        bin_path = resolved_manifest_path.with_suffix(".bin")
        bin_tmp = _temporary_path(bin_path, suffix=".tmp.bin")
        total_bytes = int(np.prod(source.shape, dtype=np.int64))
        started = time.perf_counter()
        sha = hashlib.sha256() if compute_sha256 else None

        with bin_tmp.open("wb") as handle:
            for start in range(0, source.total_rois, max(1, int(batch_size))):
                end = min(start + max(1, int(batch_size)), source.total_rois)
                batch = np.ascontiguousarray(source.read_slice(start, end), dtype=np.uint8)
                if tuple(batch.shape[1:]) != tuple(source.roi_shape):
                    raise ValueError(
                        f"ROI batch shape mismatch: expected (*, {source.roi_shape[0]}, {source.roi_shape[1]}), "
                        f"got {tuple(batch.shape)}."
                    )
                payload = batch.tobytes(order="C")
                handle.write(payload)
                if sha is not None:
                    sha.update(payload)

        os.replace(bin_tmp, bin_path)
        manifest = _build_manifest(
            source=source,
            archive_path=archive_path,
            manifest_path=resolved_manifest_path,
            bin_path=bin_path,
            cache_key=cache_key,
            batch_size=int(batch_size),
            duration_seconds=float(time.perf_counter() - started),
            payload_sha256=sha.hexdigest() if sha is not None else None,
            total_bytes=total_bytes,
        )
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, resolved_manifest_path)
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
            "format_note": (
                "flat_bin_v1 stores all ROI rows contiguously as raw uint8 bytes. "
                "It is optimized for simple sequential reads and cheap transfer, "
                "not chunked concurrent writes."
            ),
        },
    }


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
